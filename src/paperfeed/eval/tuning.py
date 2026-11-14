"""Hyperparameter tuning on the 80 tuning authors (Phase 6.3 / plan.md §6.3 / design.md §13.1).

Caches each tuning author's candidate pool and precomputed features ONCE, then runs a grid search
in-memory over scoring weights (w_cite, w_facet, w_nov) and MMR diversity (lambda_div).
The optimal parameter configuration is frozen and saved to disk before the test split is evaluated.

Tuning objective: Mean Discovery NDCG@10 across the tuning cohort.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import clean_id
from paperfeed.eval.metrics import ndcg_at_k
from paperfeed.rec.candidates import feed_candidates
from paperfeed.rec.features import CandidateFeatures, compute_features
from paperfeed.rec.mmr import select_top_k
from paperfeed.rec.recommender import (
    RecommendConfig,
    RecommenderResources,
    UserProfile,
    build_user_profile,
    effective_weights,
    apply_reranker,
    graph_hits_for,
)
from paperfeed.rec.scoring import ScoringWeights, score_candidates
from paperfeed.user.query import build_feed_query
from paperfeed.user.session import SessionState

logger = get_logger("eval.tuning")

DEFAULT_TRACK1_GRID = {
    "w_cite": [0.0, 0.25, 0.5, 1.0],
    "w_facet": [0.0, 0.25, 0.5, 1.0],
    "w_nov": [0.0, 0.25, 0.5, 1.0],
    "lambda_div": [0.0, 0.1, 0.2, 0.3, 0.5],
}


@dataclass
class CachedUserCandidateData:
    """Precomputed candidates and raw features for one tuning author."""

    author_id: str
    profile: UserProfile
    candidate_ids: List[str]
    features: List[CandidateFeatures]
    targets: List[str]  # in-corpus discovery targets
    pseudo_query: str = ""  # the feed pseudo-query (V2 reranks the pool against it)
    # paper -> (dense rank, lexical rank) in the fused pool, None where the channel missed it (V3 features)
    channel_ranks: Dict[str, Tuple[Optional[int], Optional[int]]] = field(default_factory=dict)


@dataclass
class TuningResult:
    """Best parameters found on the tuning split."""

    best_weights: ScoringWeights
    best_lambda_div: float
    best_mean_ndcg: float
    grid_size: int
    n_users: int
    top_configurations: List[Dict[str, Any]] = field(default_factory=list)
    config_signature: Optional[Dict[str, Any]] = None
    hybrid_persona: Optional[Dict[str, Any]] = None  # the baseline's own tuned gamma_rec / n_terms (plan §6.3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_weights": asdict(self.best_weights),
            "best_lambda_div": self.best_lambda_div,
            "best_mean_ndcg": self.best_mean_ndcg,
            "grid_size": self.grid_size,
            "n_users": self.n_users,
            "top_configurations": self.top_configurations,
            "config_signature": self.config_signature,
            "hybrid_persona": self.hybrid_persona,
        }

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def precompute_tuning_cache(
    resources: RecommenderResources,
    cohort: Sequence[Dict[str, Any]],
    ground_truth_map: Mapping[str, Sequence[str]],
    config: Optional[RecommendConfig] = None,
) -> List[CachedUserCandidateData]:
    """Retrieve candidate pools and compute candidate features once for each tuning author."""
    cfg = config or RecommendConfig()
    r = resources
    cached: List[CachedUserCandidateData] = []
    session = SessionState(dim=768)  # Track 1 has empty session

    for entry in cohort:
        aid = clean_id(entry["author_id"])
        targets = [clean_id(t) for t in ground_truth_map.get(aid, [])]
        authored_ids = [clean_id(p) for p in entry.get("authored_paper_ids_T", [])]
        seed_topics = entry.get("seed_topics", [])

        try:
            profile = build_user_profile(r, aid, authored_ids, seed_topics, cfg)
            excluded = profile.exclusion_sources.build(
                session_consumed=session.consumed, diagnostic=False, zero_pub=profile.zero_pub
            )

            pseudo_query = build_feed_query(
                profile.user_facets,
                session,
                r.db,
                profile.authored_papers,
                cfg.n_terms,
                cfg.n_titles,
                cutoff_year=r.cutoff_year,
            )

            cands = feed_candidates(
                r.retriever,
                profile.persona.vector,
                pseudo_query,
                session_vec=None,
                exclusion_set=excluded,
                candidate_filter=r.eligible,
                rho=cfg.rho,
                channel_depth=cfg.channel_depth,
                top_m=cfg.pool_size,
                graph_hits=graph_hits_for(r, profile, excluded, cfg),
            )
            cand_ids = [c.paper_id for c in cands]

            features = compute_features(
                cand_ids,
                r.store,
                profile.persona.vector,
                profile.user_facets,
                profile.citation_ctx,
                r.db,
                r.cutoff_year,
                session_vec=None,
                negative_ids=(),
                facet_store=r.facet_store,
            )
            features, _ = apply_reranker(r, features, pseudo_query, cfg)  # V2 only; unchanged in Core

            cached.append(
                CachedUserCandidateData(
                    author_id=aid,
                    profile=profile,
                    candidate_ids=cand_ids,
                    features=features,
                    targets=targets,
                    pseudo_query=pseudo_query,
                    channel_ranks={c.paper_id: (c.dense_rank, c.lexical_rank) for c in cands},
                )
            )
        except Exception as exc:
            logger.warning("Failed to precompute tuning cache for author %s: %s", aid, exc)

    logger.info("Successfully precomputed candidate features for %d tuning authors.", len(cached))
    return cached


@dataclass
class GridPoint:
    """One grid configuration and its per-user Discovery NDCG@k on the tuning authors (in ``cached_data`` order)."""

    params: Dict[str, float]  # w_cite, w_facet, w_nov, w_pop, lambda_div
    weights: ScoringWeights
    lambda_div: float
    per_user_ndcg: np.ndarray

    @property
    def mean_ndcg(self) -> float:
        return float(np.mean(self.per_user_ndcg)) if self.per_user_ndcg.size else 0.0


def _ranked_ids(user_data: CachedUserCandidateData, resources: RecommenderResources, weights: ScoringWeights,
                lambda_div: float, k: int, adaptive_graph: bool):
    scored = score_candidates(user_data.features, effective_weights(weights, adaptive_graph, user_data.profile))
    return scored, select_top_k(scored, resources.store, k=k, lambda_div=lambda_div)


def grid_per_user_ndcg(
    cached_data: Sequence[CachedUserCandidateData],
    resources: RecommenderResources,
    grid: Optional[Mapping[str, Sequence[float]]] = None,
    k: int = 10,
    adaptive_graph: bool = True,
    base_weights: Optional[ScoringWeights] = None,
) -> List[GridPoint]:
    """Score every grid point on every cached tuning author, in grid order.

    Weights that are not on the grid (w_pers, w_sess, lambda_neg and, unless the grid lists them, w_pop and w_ce) come
    from ``base_weights``: the run's configured weights, so the grid is tuned with the same scoring rule it is
    evaluated with. ``w_ce`` (V2) varies fastest after the other weights, so on a tie the smaller w_ce wins.
    """
    base = base_weights or ScoringWeights()
    g = dict(grid or DEFAULT_TRACK1_GRID)
    combos = itertools.product(
        list(g.get("w_cite", [0.5])), list(g.get("w_facet", [0.5])), list(g.get("w_nov", [0.5])),
        list(g.get("w_pop", [base.w_pop])), list(g.get("w_ce", [base.w_ce])), list(g.get("lambda_div", [0.2])),
    )
    points: List[GridPoint] = []
    for w_cite, w_facet, w_nov, w_pop, w_ce, lam in combos:
        weights = base.with_overrides(w_cite=w_cite, w_facet=w_facet, w_nov=w_nov, w_pop=w_pop, w_ce=w_ce)
        ndcgs = []
        for user_data in cached_data:
            if not user_data.features:
                ndcgs.append(0.0)
                continue
            _, picks = _ranked_ids(user_data, resources, weights, lam, k, adaptive_graph)
            ndcgs.append(ndcg_at_k([p.paper_id for p in picks], user_data.targets, k=k))
        params = {"w_cite": w_cite, "w_facet": w_facet, "w_nov": w_nov, "w_pop": w_pop, "lambda_div": lam}
        if "w_ce" in g:  # only V2 grids record it, so Core records stay as they were
            params["w_ce"] = w_ce
        points.append(GridPoint(params, weights, lam, np.asarray(ndcgs, dtype=np.float64)))
    return points


def best_grid_point(points: Sequence[GridPoint]) -> Optional[GridPoint]:
    """The grid optimum: highest mean NDCG, ties going to the earlier grid point."""
    best: Optional[GridPoint] = None
    for pt in points:
        if best is None or pt.mean_ndcg > best.mean_ndcg:
            best = pt
    return best


def evaluate_parameter_grid(
    cached_data: Sequence[CachedUserCandidateData],
    resources: RecommenderResources,
    grid: Optional[Mapping[str, Sequence[float]]] = None,
    k: int = 10,
    adaptive_graph: bool = True,
    config_signature: Optional[Dict[str, Any]] = None,
    base_weights: Optional[ScoringWeights] = None,
) -> TuningResult:
    """Evaluate grid search in memory across all cached tuning authors (see ``grid_per_user_ndcg``)."""
    points = grid_per_user_ndcg(cached_data, resources, grid, k, adaptive_graph, base_weights)
    return tuning_result_from_points(points, len(cached_data), config_signature, base_weights)


def tuning_result_from_points(
    points: Sequence[GridPoint],
    n_users: int,
    config_signature: Optional[Dict[str, Any]] = None,
    base_weights: Optional[ScoringWeights] = None,
) -> TuningResult:
    """The TuningResult (the saved tuning file) for an already evaluated grid."""
    best = best_grid_point(points)
    records = sorted(({**pt.params, "mean_ndcg": pt.mean_ndcg} for pt in points), key=lambda x: x["mean_ndcg"], reverse=True)
    return TuningResult(
        best_weights=best.weights if best else (base_weights or ScoringWeights()),
        best_lambda_div=best.lambda_div if best else 0.2,
        best_mean_ndcg=best.mean_ndcg if best else -1.0,
        grid_size=len(points),
        n_users=n_users,
        top_configurations=records[:10],
        config_signature=config_signature,
    )


# ---------------------------------------------------------------------------
# Facet-active variant (design §9: selection rule fixed before any test run)
# ---------------------------------------------------------------------------

FACET_TEMPLATES = ("facet", "new_method")


def relative_loss_ci(
    best: np.ndarray,
    candidate: np.ndarray,
    n_resamples: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Relative mean loss of ``candidate`` against ``best`` with a paired percentile-bootstrap CI over users.

    loss = (mean(best) - mean(candidate)) / mean(best); each resample draws the same users for both systems.
    """
    best, candidate = np.asarray(best, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    if best.shape != candidate.shape or best.size == 0:
        raise ValueError("relative_loss_ci needs two non-empty per-user arrays of the same length.")
    if best.mean() <= 0:
        raise ValueError("The best configuration has zero mean NDCG; a relative loss is undefined.")
    point = float((best.mean() - candidate.mean()) / best.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, best.size, size=(n_resamples, best.size))
    b, c = best[idx].mean(axis=1), candidate[idx].mean(axis=1)
    losses = np.where(b > 0, (b - c) / np.where(b > 0, b, 1.0), 0.0)
    return point, float(np.percentile(losses, 100 * alpha / 2)), float(np.percentile(losses, 100 * (1 - alpha / 2)))


def explanation_template_shares(
    cached_data: Sequence[CachedUserCandidateData],
    resources: RecommenderResources,
    weights: ScoringWeights,
    lambda_div: float,
    k: int = 10,
    adaptive_graph: bool = True,
) -> Dict[str, float]:
    """Share of the tuning authors' Top-k items by dominant explainable feature (the template the explainer would
    show; ``none`` when nothing contributes positively). Grounding is not checked here: it needs the abstracts."""
    from paperfeed.rec.explain import dominant_feature

    counts: Dict[str, int] = {}
    total = 0
    for user_data in cached_data:
        if not user_data.features:
            continue
        scored, picks = _ranked_ids(user_data, resources, weights, lambda_div, k, adaptive_graph)
        by_id = {s.paper_id: s for s in scored}
        for p in picks:
            key = dominant_feature(by_id[p.paper_id].contributions) or "none"
            counts[key] = counts.get(key, 0) + 1
            total += 1
    return {key: n / total for key, n in sorted(counts.items())} if total else {}


def select_facet_active_config(
    points: Sequence[GridPoint],
    cached_data: Sequence[CachedUserCandidateData],
    resources: RecommenderResources,
    k: int = 10,
    adaptive_graph: bool = True,
    max_relative_loss: float = 0.05,
    min_facet_share: float = 0.10,
    n_candidates: int = 10,
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Apply the facet-active selection rule of design §9 on the tuning split.

    Candidates are the grid points with w_facet > 0 in order of mean NDCG (at most ``n_candidates`` are examined).
    The first one whose relative-loss CI upper end is <= ``max_relative_loss`` and whose facet/new-method share of
    Top-k items is >= ``min_facet_share`` is selected; ``selected`` is None when no candidate qualifies.
    """
    best = best_grid_point(points)
    if best is None:
        raise ValueError("No grid points to select from.")
    candidates = sorted((pt for pt in points if pt.params["w_facet"] > 0), key=lambda pt: -pt.mean_ndcg)[:n_candidates]
    examined: List[Dict[str, Any]] = []
    selected: Optional[Dict[str, Any]] = None
    for pt in candidates:
        loss, lo, hi = relative_loss_ci(best.per_user_ndcg, pt.per_user_ndcg, n_resamples, seed)
        shares = explanation_template_shares(cached_data, resources, pt.weights, pt.lambda_div, k, adaptive_graph)
        facet_share = sum(shares.get(t, 0.0) for t in FACET_TEMPLATES)
        record = {
            **pt.params,
            "weights": asdict(pt.weights),
            "mean_ndcg": pt.mean_ndcg,
            "relative_loss": loss,
            "relative_loss_ci": [lo, hi],
            "non_inferior": hi <= max_relative_loss,
            "template_shares": shares,
            "facet_share": facet_share,
            "enough_facet_explanations": facet_share >= min_facet_share,
        }
        record["accepted"] = record["non_inferior"] and record["enough_facet_explanations"]
        examined.append(record)
        if record["accepted"] and selected is None:
            selected = record
    return {
        "rule": {
            "max_relative_loss": max_relative_loss,
            "min_facet_share": min_facet_share,
            "facet_templates": list(FACET_TEMPLATES),
            "n_candidates": n_candidates,
            "n_resamples": n_resamples,
            "seed": seed,
            "k": k,
        },
        "best": {**best.params, "weights": asdict(best.weights), "mean_ndcg": best.mean_ndcg,
                 "template_shares": explanation_template_shares(cached_data, resources, best.weights, best.lambda_div, k, adaptive_graph)},
        "n_users": len(cached_data),
        "candidates": examined,
        "selected": selected,
    }


DEFAULT_HYBRID_GRID = {
    "gamma_rec": [0.7, 0.8, 0.9, 1.0],
    "n_terms": [5, 10, 20],
}


def tune_hybrid_persona(
    cached_data: Sequence[CachedUserCandidateData],
    resources: RecommenderResources,
    grid: Optional[Mapping[str, Sequence[float]]] = None,
    n_titles: int = 3,
    k: int = 10,
) -> Dict[str, Any]:
    """Tune the Hybrid-Persona baseline's gamma_rec and pseudo-query length on the tuning authors (plan §6.3).

    Objective: mean Discovery NDCG@10, the same objective and authors as the system's grid. Ties keep the earlier
    grid point, so the choice is reproducible.
    """
    from paperfeed.eval.baselines import hybrid_persona_baseline

    g = dict(grid or DEFAULT_HYBRID_GRID)
    records: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    for gamma_rec, n_terms in itertools.product(g.get("gamma_rec", [0.9]), g.get("n_terms", [10])):
        ndcgs = [
            ndcg_at_k(
                hybrid_persona_baseline(resources, u.profile, k, int(n_terms), n_titles, float(gamma_rec)), u.targets, k=k
            )
            for u in cached_data
        ]
        record = {"gamma_rec": float(gamma_rec), "n_terms": int(n_terms), "mean_ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0}
        records.append(record)
        if best is None or record["mean_ndcg"] > best["mean_ndcg"]:
            best = record
    records.sort(key=lambda r: r["mean_ndcg"], reverse=True)
    return {**(best or {}), "n_titles": n_titles, "grid_size": len(records), "n_users": len(cached_data), "top_configurations": records[:10]}


TUNING_VERSION = 2  # 2: grid uses the configured w_pop; Hybrid-Persona baseline tuned too


def tuning_signature(
    config: RecommendConfig,
    tuning_author_ids: Sequence[str],
    filter_popular: Optional[int] = None,
) -> Dict[str, Any]:
    """What a cached tuning file (``tuning_artifact_path``) is valid for. Any script that reuses tuned weights must match it exactly,
    so weights tuned for one configuration (pool size, w_pop, popularity filter, tuning authors) are never reused for
    another."""
    signature = {
        "pool_size": config.pool_size,
        "channel_depth": config.channel_depth,
        "w_pop": config.weights.w_pop,
        "adaptive_graph": config.adaptive_graph,
        "filter_popular": filter_popular,
        "tuning_version": TUNING_VERSION,
        "tuning_authors": sorted(clean_id(a) for a in tuning_author_ids),
    }
    # Added only when on, so the Core signatures (and their existing tuning files) are unchanged
    if config.graph_channel:
        signature["graph_channel"] = True
    if config.cross_encoder:
        signature["cross_encoder"] = config.cross_encoder
    if config.dense_encoder != "specter2":  # V1: weights tuned with one encoder are never reused with another
        signature["dense_encoder"] = config.dense_encoder
    return signature


# ---------------------------------------------------------------------------
# V4 citation-graph channel: decision on the tuning split (design §13.8, plan 10.4c)
# ---------------------------------------------------------------------------

def candidate_recall(cached_data: Sequence[CachedUserCandidateData]) -> np.ndarray:
    """Per-user share of the in-corpus Discovery targets that reach the candidate pool (the Recall@100 ceiling)."""
    return np.array(
        [len(set(u.targets) & set(u.candidate_ids)) / len(u.targets) if u.targets else 0.0 for u in cached_data],
        dtype=np.float64,
    )


def _point_record(pt: GridPoint) -> Dict[str, Any]:
    return {**pt.params, "mean_ndcg": pt.mean_ndcg}


def _compare_to_core(
    core_points: Sequence[GridPoint],
    variant_points: Sequence[GridPoint],
    core_cached: Sequence[CachedUserCandidateData],
    variant_cached: Sequence[CachedUserCandidateData],
    n_resamples: int,
    seed: int,
    alpha: float,
) -> Dict[str, Any]:
    """Each arm at its own grid optimum; paired-bootstrap NDCG@10 and candidate Recall@100 gains of the variant."""
    from paperfeed.eval.stats import paired_bootstrap_test

    core_ids = [u.author_id for u in core_cached]
    if core_ids != [u.author_id for u in variant_cached]:
        raise ValueError("Core and variant tuning caches must cover the same authors in the same order.")
    core_best, var_best = best_grid_point(core_points), best_grid_point(variant_points)
    if core_best is None or var_best is None:
        raise ValueError("Both arms need at least one grid point.")
    if len(core_best.per_user_ndcg) != len(core_ids) or len(var_best.per_user_ndcg) != len(core_ids):
        raise ValueError("Grid points do not match the cached authors.")

    diff, lo, hi, p_value, d = paired_bootstrap_test(
        var_best.per_user_ndcg, core_best.per_user_ndcg, n_resamples=n_resamples, alpha=alpha, seed=seed
    )
    core_recall, var_recall = candidate_recall(core_cached), candidate_recall(variant_cached)
    r_diff, r_lo, r_hi, _, _ = paired_bootstrap_test(var_recall, core_recall, n_resamples=n_resamples, alpha=alpha, seed=seed)
    core_mean = core_best.mean_ndcg
    return {
        "n_users": len(core_ids),
        "core": {**_point_record(core_best), "candidate_recall_at_100": float(core_recall.mean())},
        "variant": {**_point_record(var_best), "candidate_recall_at_100": float(var_recall.mean())},
        "ndcg_gain": {
            "mean_diff": diff,
            "relative": diff / core_mean if core_mean > 0 else None,
            "ci": [lo, hi],
            "p_value": p_value,
            "cohens_d": d,
        },
        "candidate_recall_gain": {"mean_diff": r_diff, "ci": [r_lo, r_hi]},
        "adopt": bool(lo > 0),
    }


def decide_graph_channel(
    core_points: Sequence[GridPoint],
    v4_points: Sequence[GridPoint],
    core_cached: Sequence[CachedUserCandidateData],
    v4_cached: Sequence[CachedUserCandidateData],
    n_resamples: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Pre-registered V4 rule (D§13.8 point 4): V4 replaces the Core pool only if the paired-bootstrap 95% CI of its
    tuning-split mean NDCG@10 gain over Core, each at its own grid optimum, lies above zero.

    Both arms must cover the same tuning authors in the same order. Candidate Recall@100 is reported alongside as the
    mechanism check; it does not enter the decision.
    """
    cmp = _compare_to_core(core_points, v4_points, core_cached, v4_cached, n_resamples, seed, alpha)
    return {
        "rule": {
            "statement": "V4 replaces the Core pool only if the paired-bootstrap 95% CI of its tuning-split mean "
            "NDCG@10 gain over Core (each at its own grid optimum) lies above zero (design §13.8).",
            "n_resamples": n_resamples,
            "alpha": alpha,
            "seed": seed,
        },
        "n_users": cmp["n_users"],
        "core": cmp["core"],
        "v4": cmp["variant"],
        "ndcg_gain": cmp["ndcg_gain"],
        "candidate_recall_gain": cmp["candidate_recall_gain"],
        "adopt_v4": cmp["adopt"],
    }


# ---------------------------------------------------------------------------
# V2 cross-encoder: decision on the tuning split (design §13.8, plan 10.2b-c)
# ---------------------------------------------------------------------------

V2_GRID = {**DEFAULT_TRACK1_GRID, "w_ce": [0.0, 0.25, 0.5, 1.0]}


def with_reranker_scores(
    cached_data: Sequence[CachedUserCandidateData],
    resources: RecommenderResources,
    reranker: Any,
) -> Tuple[List[CachedUserCandidateData], List[float]]:
    """Copies of the cached tuning authors with the reranker score on every candidate, and the seconds per feed.

    V2 never changes the pool, so the Core pools and features are reused as they are and only ``ce_score`` is added;
    this makes the V2 and Core arms rank exactly the same candidates.
    """
    from dataclasses import replace as dc_replace

    from paperfeed.rec.rerank import add_reranker_scores, candidate_texts

    out: List[CachedUserCandidateData] = []
    seconds: List[float] = []
    for u in cached_data:
        texts = candidate_texts(resources.db, [f.paper_id for f in u.features])
        feats, secs = add_reranker_scores(u.features, u.pseudo_query, texts, reranker)
        out.append(dc_replace(u, features=feats))
        seconds.append(secs)
    return out, seconds


def latency_summary(seconds: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(seconds, dtype=np.float64)
    if arr.size == 0:
        return {"mean_s": 0.0, "p95_s": 0.0, "n_feeds": 0}
    return {"mean_s": float(arr.mean()), "p95_s": float(np.percentile(arr, 95)), "n_feeds": int(arr.size)}


def decide_cross_encoder(
    core_points: Sequence[GridPoint],
    model_points: Mapping[str, Sequence[GridPoint]],
    core_cached: Sequence[CachedUserCandidateData],
    model_cached: Mapping[str, Sequence[CachedUserCandidateData]],
    n_resamples: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Pre-registered V2 rule (D§13.8 V2 point 5): the reranker with the higher tuning-split NDCG@10 at its own grid
    optimum is the V2 candidate (ties: the first listed); it replaces Core only if the paired-bootstrap 95% CI of its
    NDCG@10 gain over Core lies above zero. An optimum with w_ce = 0 is Core itself and is never adopted."""
    if not model_points:
        raise ValueError("At least one reranker is needed.")
    per_model = {}
    for name, points in model_points.items():
        best = best_grid_point(points)
        if best is None:
            raise ValueError(f"No grid points for {name}.")
        per_model[name] = _point_record(best)
    chosen = max(model_points, key=lambda m: (per_model[m]["mean_ndcg"], -list(model_points).index(m)))
    cmp = _compare_to_core(core_points, model_points[chosen], core_cached, model_cached[chosen], n_resamples, seed, alpha)
    uses_reranker = cmp["variant"].get("w_ce", 0.0) > 0
    return {
        "rule": {
            "statement": "The reranker with the higher tuning-split NDCG@10 at its own grid optimum is the V2 "
            "candidate; it replaces Core only if the paired-bootstrap 95% CI of its NDCG@10 gain over Core lies above "
            "zero, and never when its optimum has w_ce = 0 (design §13.8, V2 point 5).",
            "w_ce_grid": V2_GRID["w_ce"],
            "n_resamples": n_resamples,
            "alpha": alpha,
            "seed": seed,
        },
        "n_users": cmp["n_users"],
        "per_model_optimum": per_model,
        "chosen_model": chosen,
        "core": cmp["core"],
        "v2": cmp["variant"],
        "ndcg_gain": cmp["ndcg_gain"],
        "adopt_v2": bool(cmp["adopt"] and uses_reranker),
    }


def load_v2_decision(artifacts_dir: Path | str) -> Dict[str, Any]:
    """The frozen V2 decision (plan 10.2c), required before any --cross-encoder test run or audit."""
    path = Path(artifacts_dir) / "variant_v2_decision.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: freeze the V2 decision with scripts/decide_cross_encoder.py "
                                "before the test run (plan 10.2c).")
    decision = json.loads(path.read_text(encoding="utf-8"))
    if decision.get("v2", {}).get("w_ce", 0.0) <= 0:
        raise ValueError("The frozen V2 optimum has w_ce = 0: V2 is the Core system, so there is no V2 run to make.")
    return decision


def signature_hash(signature: Mapping[str, Any]) -> str:
    """Short stable hash of a tuning signature (names the per-configuration tuning file)."""
    return hashlib.sha256(json.dumps(dict(signature), sort_keys=True).encode("utf-8")).hexdigest()[:12]


def tuning_artifact_path(artifacts_dir: Path | str, signature: Mapping[str, Any]) -> Path:
    """One tuning file per configuration, so tuning the main, filtered and enhanced runs never overwrite each other."""
    return Path(artifacts_dir) / f"tuning_results_{signature_hash(signature)}.json"


def find_tuning_artifact(artifacts_dir: Path | str, signature: Mapping[str, Any]) -> Optional[Path]:
    """The tuning file made for exactly this configuration, or None.

    Looks at the per-configuration file first, then at older ``tuning_results*.json`` files (the single shared
    ``tuning_results.json`` used before per-configuration names, or hand-made copies), accepting one only when its
    stored signature matches. The returned path is the file actually used, for the run manifest."""
    expected = dict(signature)
    canonical = tuning_artifact_path(artifacts_dir, signature)
    others = sorted(p for p in Path(artifacts_dir).glob("tuning_results*.json") if p != canonical)
    for path in [canonical, *others]:
        if not path.exists():
            continue
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable tuning file %s: %s", path, exc)
            continue
        if saved.get("config_signature") == expected:
            return path
    return None


def default_facet_selection_path(artifacts_dir: Path | str, enhanced: bool = False) -> Path:
    """Where the frozen facet-active selection lives (one file per Track 1 configuration)."""
    return Path(artifacts_dir) / ("facet_config_selection_enhanced.json" if enhanced else "facet_config_selection.json")


def load_facet_selection(path: Path | str, expected_signature: Mapping[str, Any]) -> Tuple[ScoringWeights, float]:
    """The facet-active variant's weights and lambda_div, only from a selection made for this exact configuration."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: run scripts/select_facet_config.py with the same options first.")
    saved = json.loads(p.read_text(encoding="utf-8"))
    if saved.get("config_signature") != dict(expected_signature):
        raise ValueError(
            f"{p} was selected for a different configuration ({saved.get('config_signature')} != {dict(expected_signature)})."
        )
    chosen = saved.get("selected")
    if not chosen:
        raise ValueError(f"{p}: no grid point met the facet-active rule, so there is no facet-active variant to run.")
    weights = ScoringWeights(**{k: v for k, v in chosen["weights"].items() if k in ScoringWeights.__dataclass_fields__})
    return weights, float(chosen["lambda_div"])
