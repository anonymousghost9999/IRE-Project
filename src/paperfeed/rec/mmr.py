"""Maximal Marginal Relevance selection (Phase 5.4 / docs/plan.md §5.4 / docs/design.md §10).

Greedy: starting from an empty result set R, repeatedly add

    p* = argmax_{p in pool \\ R} [ rel(p) - lambda_div * max_{q in R} cos(e_p, e_q) ]

until K papers are chosen. rel(p) is the linear score of 5.3, which already contains the new-method term, so novelty is
folded into relevance before diversification. The max over an empty R is 0, so the first pick is the top-scored paper.
Ties are broken by paper_id so the result is deterministic.
"""

from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence

import numpy as np

from paperfeed.data.cohort import clean_id
from paperfeed.index.store import EmbeddingStore
from paperfeed.rec.scoring import ScoredCandidate

DEFAULT_K = 10
DEFAULT_LAMBDA_DIV = 0.2


@dataclass
class Selection:
    """One pick of the greedy loop."""

    scored: ScoredCandidate
    rank: int  # 1-based position in the final list
    mmr_value: float  # rel(p) - lambda_div * max_sim at the moment it was picked
    max_similarity: float  # max cosine to the papers already picked (0 for the first pick)

    @property
    def paper_id(self) -> str:
        return self.scored.paper_id


def mmr_select(
    scored: Sequence[ScoredCandidate],
    vectors: np.ndarray,
    k: int = DEFAULT_K,
    lambda_div: float = DEFAULT_LAMBDA_DIV,
) -> List[Selection]:
    """Pick up to ``k`` papers from the pool; ``vectors`` holds one embedding row per entry of ``scored`` (same order)."""
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}.")
    if lambda_div < 0:
        raise ValueError(f"lambda_div must be >= 0, got {lambda_div}.")
    n = len(scored)
    vectors = np.asarray(vectors, dtype=np.float32)
    if len(vectors) != n:
        raise ValueError(f"Length mismatch: {n} candidates vs {len(vectors)} vectors.")
    ids = [s.paper_id for s in scored]
    if len(set(ids)) != n:
        raise ValueError("MMR needs unique paper_ids in the pool.")
    if n == 0 or k == 0:
        return []

    unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    sims = unit @ unit.T
    rel = np.array([s.score for s in scored], dtype=np.float64)

    remaining = set(range(n))
    max_sim = np.full(n, -np.inf)  # max cosine to the picked set; -inf while nothing is picked

    def penalty_sim(i: int) -> float:
        return float(max_sim[i]) if np.isfinite(max_sim[i]) else 0.0  # empty set: no diversity penalty

    picks: List[Selection] = []
    while remaining and len(picks) < k:
        # rounding keeps the choice identical across platforms when two values differ only by float noise
        best = min(remaining, key=lambda i: (-round(rel[i] - lambda_div * penalty_sim(i), 12), ids[i]))
        picks.append(
            Selection(scored=scored[best], rank=len(picks) + 1, mmr_value=float(rel[best] - lambda_div * penalty_sim(best)),
                      max_similarity=penalty_sim(best))
        )
        remaining.discard(best)
        for i in remaining:
            max_sim[i] = max(max_sim[i], float(sims[i, best]))
    return picks


def select_top_k(
    scored: Sequence[ScoredCandidate],
    store: EmbeddingStore,
    k: int = DEFAULT_K,
    lambda_div: float = DEFAULT_LAMBDA_DIV,
) -> List[Selection]:
    """``mmr_select`` with the candidates' embeddings read from the embedding store."""
    if not scored:
        return []
    return mmr_select(scored, store.select([clean_id(s.paper_id) for s in scored]), k, lambda_div)


def mmr_settings(cfg: Mapping[str, Any]) -> dict:
    """K (``retrieval.feed_size``) and lambda_div (``scoring_weights.lambda_div``) from configs/default.yaml."""
    return {
        "k": int(cfg.get("retrieval", {}).get("feed_size", DEFAULT_K)),
        "lambda_div": float(cfg.get("scoring_weights", {}).get("lambda_div", DEFAULT_LAMBDA_DIV)),
    }
