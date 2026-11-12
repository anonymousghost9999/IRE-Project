"""Leave-Future-Out (LFO) Track 1 Evaluation Engine (Phase 6.4 / plan.md §6.4 / design.md §13.1).

Executes the offline benchmark over test authors from the cohort:
1. Full tuned Paper Feed system.
2. Component ablations:
   - No-MMR (lambda_div = 0)
   - No-Graph (w_cite = 0)
   - No-Facets (w_facet = w_nov = 0)
3. Diagnostic Re-Citation run (Citations(u, <= T) dropped from E_u).
4. Five competitive baselines:
   - Random, Popularity, BM25 Lexical, Dense Centroid, Hybrid-Persona.
5. Anchors:
   - Random Anchor (100 uniform draws), Oracle@10 ceiling.
6. Slices:
   - History volume (Warm, Moderate, Sparse, Zero-Pub).
   - Item cold-start (targets with citation_count_T <= 1 vs >= 20).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

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
) -> UserEvaluationRow:
    """Evaluate one author across full system, ablations, baselines, and diagnostic runs."""
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

    # 1. Full tuned system
    res_full = recommend(resources, profile, session=None, mode="feed", config=cfg)
    system_metrics["Paper Feed (Full)"] = _eval_list(res_full.paper_ids, res_full.candidate_ids, targets)

    # 2. Ablations
    # No-MMR: lambda_div = 0
    cfg_no_mmr = cfg.with_overrides(lambda_div=0.0)
    res_no_mmr = recommend(resources, profile, session=None, mode="feed", config=cfg_no_mmr)
    system_metrics["Ablation: No-MMR"] = _eval_list(res_no_mmr.paper_ids, res_no_mmr.candidate_ids, targets)

    # No-Graph: w_cite = 0
    cfg_no_graph = cfg.with_overrides(weights=cfg.weights.with_overrides(w_cite=0.0))
    res_no_graph = recommend(resources, profile, session=None, mode="feed", config=cfg_no_graph)
    system_metrics["Ablation: No-Graph"] = _eval_list(res_no_graph.paper_ids, res_no_graph.candidate_ids, targets)

    # No-Facets: w_facet = 0, w_nov = 0
    cfg_no_facets = cfg.with_overrides(weights=cfg.weights.with_overrides(w_facet=0.0, w_nov=0.0))
    res_no_facets = recommend(resources, profile, session=None, mode="feed", config=cfg_no_facets)
    system_metrics["Ablation: No-Facets"] = _eval_list(res_no_facets.paper_ids, res_no_facets.candidate_ids, targets)

    # 3. Diagnostic Re-Citation run (Citations(u, <= T) dropped from E_u)
    cfg_diag = cfg.with_overrides(diagnostic=True)
    res_diag = recommend(resources, profile, session=None, mode="feed", config=cfg_diag)
    diag_tgt = [clean_id(t) for t in recitation_targets]
    system_metrics["Diagnostic: Re-Citation"] = _eval_list(res_diag.paper_ids, res_diag.candidate_ids, diag_tgt)

    # 4. Baselines
    for b_name in BASELINE_NAMES:
        b_pids = run_baseline(b_name, resources, profile, k=k)
        system_metrics[f"Baseline: {b_name}"] = _eval_list(b_pids, b_pids, targets)

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
    )


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
            )
            rows.append(row)
        except Exception as exc:
            logger.error("Error evaluating author %s in LFO: %s", aid, exc)
            failed_authors.append(aid)

    if failed_authors:
        raise RuntimeError(
            f"LFO evaluation failed for {len(failed_authors)} of {len(eval_cohort)} authors: {failed_authors[:5]}"
        )

    # Aggregate by system
    system_names = list(dict.fromkeys(k for r in rows for k in r.system_metrics)) if rows else []
    system_aggregates: Dict[str, Dict[str, float]] = {}

    for sys_name in system_names:
        user_dicts = [r.system_metrics[sys_name] for r in rows if sys_name in r.system_metrics]
        system_aggregates[sys_name] = aggregate_metrics(user_dicts)

    return LFOEvaluationResult(
        system_aggregates=system_aggregates,
        per_user_results=rows,
        config=cfg,
        n_authors=len(rows),
    )
