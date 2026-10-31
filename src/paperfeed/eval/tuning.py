"""Hyperparameter tuning on the 80 tuning authors (Phase 6.3 / plan.md §6.3 / design.md §13.1).

Caches each tuning author's candidate pool and precomputed features ONCE, then runs a grid search
in-memory over scoring weights (w_cite, w_facet, w_nov) and MMR diversity (lambda_div).
The optimal parameter configuration is frozen and saved to disk before the test split is evaluated.

Tuning objective: Mean Discovery NDCG@10 across the tuning cohort.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
from paperfeed.rec.recommender import RecommendConfig, RecommenderResources, UserProfile, build_user_profile
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


@dataclass
class TuningResult:
    """Best parameters found on the tuning split."""

    best_weights: ScoringWeights
    best_lambda_div: float
    best_mean_ndcg: float
    grid_size: int
    n_users: int
    top_configurations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_weights": asdict(self.best_weights),
            "best_lambda_div": self.best_lambda_div,
            "best_mean_ndcg": self.best_mean_ndcg,
            "grid_size": self.grid_size,
            "n_users": self.n_users,
            "top_configurations": self.top_configurations,
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

            cached.append(
                CachedUserCandidateData(
                    author_id=aid,
                    profile=profile,
                    candidate_ids=cand_ids,
                    features=features,
                    targets=targets,
                )
            )
        except Exception as exc:
            logger.warning("Failed to precompute tuning cache for author %s: %s", aid, exc)

    logger.info("Successfully precomputed candidate features for %d tuning authors.", len(cached))
    return cached


def evaluate_parameter_grid(
    cached_data: Sequence[CachedUserCandidateData],
    resources: RecommenderResources,
    grid: Optional[Mapping[str, Sequence[float]]] = None,
    k: int = 10,
) -> TuningResult:
    """Evaluate grid search in memory across all cached tuning authors."""
    g = dict(grid or DEFAULT_TRACK1_GRID)
    w_cites = list(g.get("w_cite", [0.5]))
    w_facets = list(g.get("w_facet", [0.5]))
    w_novs = list(g.get("w_nov", [0.5]))
    w_pops = list(g.get("w_pop", [0.0]))
    lambdas = list(g.get("lambda_div", [0.2]))

    combinations = list(itertools.product(w_cites, w_facets, w_novs, w_pops, lambdas))
    results: List[Dict[str, Any]] = []

    best_mean_ndcg = -1.0
    best_combo = None

    for w_cite, w_facet, w_nov, w_pop, lam in combinations:
        weights = ScoringWeights(w_pers=1.0, w_sess=1.0, w_cite=w_cite, w_facet=w_facet, w_nov=w_nov, w_pop=w_pop)
        ndcgs: List[float] = []

        for user_data in cached_data:
            if not user_data.features:
                ndcgs.append(0.0)
                continue

            scored = score_candidates(user_data.features, weights)
            picks = select_top_k(scored, resources.store, k=k, lambda_div=lam)
            recommended_ids = [p.paper_id for p in picks]
            score = ndcg_at_k(recommended_ids, user_data.targets, k=k)
            ndcgs.append(score)

        mean_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
        record = {
            "w_cite": w_cite,
            "w_facet": w_facet,
            "w_nov": w_nov,
            "w_pop": w_pop,
            "lambda_div": lam,
            "mean_ndcg": mean_ndcg,
        }
        results.append(record)

        if mean_ndcg > best_mean_ndcg:
            best_mean_ndcg = mean_ndcg
            best_combo = (weights, lam)

    results.sort(key=lambda x: x["mean_ndcg"], reverse=True)

    if best_combo is None:
        best_weights = ScoringWeights()
        best_lam = 0.2
    else:
        best_weights, best_lam = best_combo

    return TuningResult(
        best_weights=best_weights,
        best_lambda_div=best_lam,
        best_mean_ndcg=best_mean_ndcg,
        grid_size=len(combinations),
        n_users=len(cached_data),
        top_configurations=results[:10],
    )
