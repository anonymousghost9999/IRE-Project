"""Leave-Future-Out (LFO) Track 1 Evaluation Engine (Phase 6.4 / plan.md §6.4 / design.md §13.1).

Executes the offline benchmark over test authors from the cohort:
1. Full tuned Paper Feed system.
2. Component ablations:
   - No-MMR (lambda_div = 0)
   - No-Graph (w_cite = 0)
   - No-Facets (w_facet = w_nov = 0)
3. Diagnostic Re-Citation run (Citations(u, <= T) dropped from E_u).
4. Five competitive baselines (plan §6.2):
   - Popularity, BM25 Lexical, Dense Centroid, Citation-only, Hybrid-Persona (with its own tuned gamma_rec, n_terms).
5. Anchors:
   - Random Anchor (100 uniform draws), Oracle@10 ceiling.
6. Slices:
   - History volume (Warm, Moderate, Sparse, Zero-Pub).
   - Item cold-start (targets with citation_count_T <= 1 vs >= 20).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set

import numpy as np

from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import clean_id
from paperfeed.eval.baselines import (
    BASELINE_NAMES,
    random_anchor_ids,
    run_baseline,
)
from paperfeed.eval.metrics import (
    aggregate_metrics,
    ndcg_at_k,
    per_user_metrics,
    recall_at_k,
)
from paperfeed.rec.recommender import (
    RecommendConfig,
    RecommenderResources,
    UserProfile,
    build_user_profile,
    recommend,
)
from paperfeed.rec.scoring import ScoringWeights

logger = get_logger("eval.lfo")


@dataclass
class UserEvaluationRow:
    """All metrics and metadata for one evaluation author across systems."""

    author_id: str
    split: str
    history_slice: str
    n_authored: int
    n_targets: int
    system_metrics: Dict[str, Dict[str, float]]  # system_name -> metric_name -> value
    # system_name -> its top-k paper IDs (not kept for the random anchor, which is a mean over many draws).
    # Saved so slices over items (item cold start) and any later re-analysis work from the same run.
    system_recommendations: Dict[str, List[str]] = field(default_factory=dict)
    # Phase 11: seconds per feed for each system that was timed, and the full system's Explanation Coverage
    latency_seconds: Dict[str, float] = field(default_factory=dict)
    explanation_coverage: Optional[float] = None


@dataclass
class LFOEvaluationResult:
    """Aggregated evaluation results across the entire cohort."""

    system_aggregates: Dict[str, Dict[str, float]]  # system_name -> macro-averaged metrics
    per_user_results: List[UserEvaluationRow]
    config: RecommendConfig
    n_authors: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_authors": self.n_authors,
            "system_aggregates": self.system_aggregates,
            "per_user_results": [
                {
                    "author_id": r.author_id,
                    "split": r.split,
                    "history_slice": r.history_slice,
                    "n_authored": r.n_authored,
                    "n_targets": r.n_targets,
                    "system_metrics": r.system_metrics,
                    "system_recommendations": r.system_recommendations,
                    "latency_seconds": r.latency_seconds,
                    "explanation_coverage": r.explanation_coverage,
                }
                for r in self.per_user_results
            ],
        }

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def evaluate_author(
    resources: RecommenderResources,
    author_info: Mapping[str, Any],
    ground_truth_targets: Sequence[str],
    recitation_targets: Sequence[str],
    topic_map: Dict[str, str],
    paper_id_index: Dict[str, int],
    embedding_matrix: np.ndarray,
    paper_facet_map: Dict[str, Set[str]],
    base_config: Optional[RecommendConfig] = None,
    k: int = 10,
    include_anchors: bool = True,
    n_random_draws: int = 100,
    baseline_kwargs: Optional[Mapping[str, Mapping[str, Any]]] = None,
    extra_systems: Optional[Mapping[str, Callable[..., List[str]]]] = None,
) -> UserEvaluationRow:
    """Evaluate one author across full system, ablations, baselines, and diagnostic runs.

    ``extra_systems`` (Phase 11 comparison systems): name -> callable(profile, author_info, full_result) returning a
    ranked list of paper IDs; each is scored like a baseline as "SOTA: <name>" and timed per feed."""
    cfg = base_config or RecommendConfig()
    aid = clean_id(author_info["author_id"])
    authored_ids = [clean_id(p) for p in author_info.get("authored_paper_ids_T", [])]
    seed_topics = author_info.get("seed_topics", [])
    history_slice = str(author_info.get("history_size_slice", "warm"))
    split = str(author_info.get("split", "test"))

    profile = build_user_profile(resources, aid, authored_ids, seed_topics, cfg)
    user_facets_set = set(profile.user_facets.ids)
    targets = [clean_id(t) for t in ground_truth_targets]

    system_metrics: Dict[str, Dict[str, float]] = {}
    system_recommendations: Dict[str, List[str]] = {}

    def _eval_list(rec_pids: List[str], cand_pids: List[str], tgt: List[str]) -> Dict[str, float]:
        return per_user_metrics(
            recommended_ids=rec_pids,
            ground_truth_ids=tgt,
            candidate_ids=cand_pids,
            topic_map=topic_map,
            embedding_matrix=embedding_matrix,
            paper_id_index=paper_id_index,
            user_canonical_facet_ids=user_facets_set,
            paper_facet_map=paper_facet_map,
            k=k,
        )

    def _record(name: str, rec_pids: List[str], cand_pids: List[str], tgt: List[str]) -> None:
        system_metrics[name] = _eval_list(rec_pids, cand_pids, tgt)
        system_recommendations[name] = list(rec_pids[:k])

    # 1. Full tuned system
    latency: Dict[str, float] = {}
    t0 = time.perf_counter()
    res_full = recommend(resources, profile, session=None, mode="feed", config=cfg)
    latency["Paper Feed (Full)"] = time.perf_counter() - t0
    _record("Paper Feed (Full)", res_full.paper_ids, res_full.candidate_ids, targets)
    items = getattr(res_full, "items", None) or []
    coverage = (sum(1 for it in items if it.explanation is not None and it.explanation.covered) / len(items)) if items else None

    # 2. Ablations (one component changed at a time, same candidate pool). With the V4 graph channel, No-Graph also
    #    drops the channel (design §13.8 point 5), so it removes every use of the citation graph.
    #    With the V3 learned ranker the same components are removed from the model's input instead (zeroing a linear
    #    weight would not change a tree model), and No-MMR also sets V3's own lambda to 0.
    v3 = bool(cfg.learned_ranker)
    ablations = {
        "Ablation: No-MMR": cfg.with_overrides(lambda_div=0.0, v3_lambda_div=0.0 if v3 else cfg.v3_lambda_div),
        "Ablation: No-Graph": cfg.with_overrides(
            weights=cfg.weights.with_overrides(w_cite=0.0), graph_channel=False,
            v3_off_features=("citation",) if v3 else cfg.v3_off_features,
        ),
        "Ablation: No-Facets": cfg.with_overrides(
            weights=cfg.weights.with_overrides(w_facet=0.0, w_nov=0.0),
            v3_off_features=("facet", "new_method") if v3 else cfg.v3_off_features,
        ),
    }
    for name, ablation_cfg in ablations.items():
        res = recommend(resources, profile, session=None, mode="feed", config=ablation_cfg)
        _record(name, res.paper_ids, res.candidate_ids, targets)

    # 3. Diagnostic Re-Citation run (Citations(u, <= T) dropped from E_u)
    res_diag = recommend(resources, profile, session=None, mode="feed", config=cfg.with_overrides(diagnostic=True))
    _record("Diagnostic: Re-Citation", res_diag.paper_ids, res_diag.candidate_ids, [clean_id(t) for t in recitation_targets])

    # 4. Baselines
    for b_name in BASELINE_NAMES:
        b_pids = run_baseline(b_name, resources, profile, k=k, baseline_kwargs=(baseline_kwargs or {}).get(b_name))
        _record(f"Baseline: {b_name}", b_pids, b_pids, targets)

    # 4b. Phase 11 comparison systems (same eligible pool and E_u, applied inside each callable)
    for name, fn in (extra_systems or {}).items():
        t0 = time.perf_counter()
        ids = list(fn(profile, author_info, res_full))
        latency[f"SOTA: {name}"] = time.perf_counter() - t0
        _record(f"SOTA: {name}", ids, ids, targets)

    # 5. Anchors
    if include_anchors:
        draws = random_anchor_ids(resources, profile, k=k, n_draws=n_random_draws)
        draw_metrics = [_eval_list(d, d, targets) for d in draws]
        system_metrics[f"Anchor: Random ({n_random_draws}-draw)"] = aggregate_metrics(draw_metrics)

    return UserEvaluationRow(
        author_id=aid,
        split=split,
        history_slice=history_slice,
        n_authored=len(authored_ids),
        n_targets=len(targets),
        system_metrics=system_metrics,
        system_recommendations=system_recommendations,
        latency_seconds=latency,
        explanation_coverage=coverage,
    )


# Baselines that apply when the profile is seed topics only (design §13.1): Citation-only has no authored papers.
ZERO_PUB_BASELINES = ("popularity", "bm25_author", "dense_centroid", "hybrid_persona")


def evaluate_zero_pub_author(
    resources: RecommenderResources,
    author_info: Mapping[str, Any],
    ground_truth_targets: Sequence[str],
    topic_map: Dict[str, str],
    paper_id_index: Dict[str, int],
    embedding_matrix: np.ndarray,
    paper_facet_map: Dict[str, Set[str]],
    base_config: Optional[RecommendConfig] = None,
    k: int = 10,
    include_anchors: bool = True,
    n_random_draws: int = 100,
    seed_subfield: Optional[str] = None,
    baseline_kwargs: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> UserEvaluationRow:
    """Zero-Publication Seed-Only slice for one author (plan §6.4 / design §13.1).

    The profile is replaced by the author's 3 seed topics: no authored papers, E_u = empty, and the ground truth is
    Ref(B) with year <= T minus Authored(u) (``zero_pub_in_corpus``). Lexical and Dense baselines query with the seed
    topics, Popularity uses the seed topics' subfield (``seed_subfield``), Citation-only is not applicable.
    """
    cfg = base_config or RecommendConfig()
    aid = clean_id(author_info["author_id"])
    seed_topics = list(author_info.get("seed_topics") or [])
    if not seed_topics:
        raise ValueError(f"Author {aid} has no seed topics for the zero-publication slice.")
    profile = build_user_profile(resources, aid, (), seed_topics, cfg)
    if not profile.zero_pub:
        raise RuntimeError(f"Zero-publication profile for {aid} was not built from seed topics.")
    user_facets_set = set(profile.user_facets.ids)
    targets = [clean_id(t) for t in ground_truth_targets]

    system_metrics: Dict[str, Dict[str, float]] = {}
    system_recommendations: Dict[str, List[str]] = {}

    def _eval_list(rec_pids: List[str], cand_pids: List[str]) -> Dict[str, float]:
        return per_user_metrics(
            recommended_ids=rec_pids, ground_truth_ids=targets, candidate_ids=cand_pids, topic_map=topic_map,
            embedding_matrix=embedding_matrix, paper_id_index=paper_id_index,
            user_canonical_facet_ids=user_facets_set, paper_facet_map=paper_facet_map, k=k,
        )

    res = recommend(resources, profile, session=None, mode="feed", config=cfg)
    system_metrics["Paper Feed (Full)"] = _eval_list(res.paper_ids, res.candidate_ids)
    system_recommendations["Paper Feed (Full)"] = list(res.paper_ids[:k])

    kwargs = {name: dict((baseline_kwargs or {}).get(name) or {}) for name in ZERO_PUB_BASELINES}
    kwargs["popularity"]["subfield"] = seed_subfield
    for b_name in ZERO_PUB_BASELINES:
        b_pids = run_baseline(b_name, resources, profile, k=k, baseline_kwargs=kwargs[b_name])
        system_metrics[f"Baseline: {b_name}"] = _eval_list(b_pids, b_pids)
        system_recommendations[f"Baseline: {b_name}"] = list(b_pids[:k])

    if include_anchors:
        draws = random_anchor_ids(resources, profile, k=k, n_draws=n_random_draws)
        system_metrics[f"Anchor: Random ({n_random_draws}-draw)"] = aggregate_metrics([_eval_list(d, d) for d in draws])

    return UserEvaluationRow(
        author_id=aid,
        split=str(author_info.get("split", "test")),
        history_slice="zero_pub",
        n_authored=0,
        n_targets=len(targets),
        system_metrics=system_metrics,
        system_recommendations=system_recommendations,
    )


def _aggregate(rows: Sequence[UserEvaluationRow]) -> Dict[str, Dict[str, float]]:
    system_names = list(dict.fromkeys(name for r in rows for name in r.system_metrics))
    return {
        name: aggregate_metrics([r.system_metrics[name] for r in rows if name in r.system_metrics])
        for name in system_names
    }


def run_zero_pub_evaluation(
    resources: RecommenderResources,
    cohort: Sequence[Dict[str, Any]],
    zero_pub_ground_truth: Mapping[str, Sequence[str]],
    topic_map: Dict[str, str],
    paper_id_index: Dict[str, int],
    embedding_matrix: np.ndarray,
    paper_facet_map: Dict[str, Set[str]],
    seed_subfields: Mapping[str, Optional[str]],
    config: Optional[RecommendConfig] = None,
    k: int = 10,
    include_anchors: bool = True,
    baseline_kwargs: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> LFOEvaluationResult:
    """Re-run the cohort with seed-only profiles. Any failing author aborts the run (no silent drops)."""
    cfg = config or RecommendConfig()
    rows: List[UserEvaluationRow] = []
    failed: List[str] = []
    for entry in cohort:
        aid = clean_id(entry["author_id"])
        try:
            rows.append(
                evaluate_zero_pub_author(
                    resources, entry, zero_pub_ground_truth.get(aid, []), topic_map, paper_id_index, embedding_matrix,
                    paper_facet_map, cfg, k, include_anchors, seed_subfield=seed_subfields.get(aid),
                    baseline_kwargs=baseline_kwargs,
                )
            )
        except Exception as exc:
            logger.error("Error evaluating author %s in the zero-publication slice: %s", aid, exc)
            failed.append(aid)
    if failed:
        raise RuntimeError(f"Zero-publication slice failed for {len(failed)} of {len(cohort)} authors: {failed[:5]}")
    return LFOEvaluationResult(system_aggregates=_aggregate(rows), per_user_results=rows, config=cfg, n_authors=len(rows))


def run_lfo_evaluation(
    resources: RecommenderResources,
    cohort: Sequence[Dict[str, Any]],
    ground_truth_map: Mapping[str, Sequence[str]],
    recitation_map: Mapping[str, Sequence[str]],
    topic_map: Dict[str, str],
    paper_id_index: Dict[str, int],
    embedding_matrix: np.ndarray,
    paper_facet_map: Dict[str, Set[str]],
    config: Optional[RecommendConfig] = None,
    k: int = 10,
    include_anchors: bool = True,
    filter_empty_targets: bool = False,
    baseline_kwargs: Optional[Mapping[str, Mapping[str, Any]]] = None,
    extra_systems: Optional[Mapping[str, Callable[..., List[str]]]] = None,
) -> LFOEvaluationResult:
    """Run Track 1 LFO evaluation over all authors in cohort."""
    cfg = config or RecommendConfig()
    rows: List[UserEvaluationRow] = []

    eval_cohort = cohort
    if filter_empty_targets:
        eval_cohort = [
            entry for entry in cohort
            if len(ground_truth_map.get(clean_id(entry["author_id"]), [])) > 0
        ]
        if len(eval_cohort) < len(cohort):
            logger.info(
                "filter_empty_targets: evaluating %d active authors with non-empty ground truth targets (%d excluded).",
                len(eval_cohort),
                len(cohort) - len(eval_cohort),
            )

    failed_authors: List[str] = []
    for entry in eval_cohort:
        aid = clean_id(entry["author_id"])
        targets = ground_truth_map.get(aid, [])
        rec_targets = recitation_map.get(aid, [])
        try:
            row = evaluate_author(
                resources=resources,
                author_info=entry,
                ground_truth_targets=targets,
                recitation_targets=rec_targets,
                topic_map=topic_map,
                paper_id_index=paper_id_index,
                embedding_matrix=embedding_matrix,
                paper_facet_map=paper_facet_map,
                base_config=cfg,
                k=k,
                include_anchors=include_anchors,
                baseline_kwargs=baseline_kwargs,
                extra_systems=extra_systems,
            )
            rows.append(row)
        except Exception as exc:
            logger.error("Error evaluating author %s in LFO: %s", aid, exc)
            failed_authors.append(aid)

    if failed_authors:
        raise RuntimeError(
            f"LFO evaluation failed for {len(failed_authors)} of {len(eval_cohort)} authors: {failed_authors[:5]}"
        )

    system_aggregates = _aggregate(rows)

    return LFOEvaluationResult(
        system_aggregates=system_aggregates,
        per_user_results=rows,
        config=cfg,
        n_authors=len(rows),
    )
