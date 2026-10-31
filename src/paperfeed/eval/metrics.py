"""IR evaluation metrics for the Track 1 Leave-Future-Out benchmark (Phase 6.1 / plan.md §6.1 / design.md §13.1).

All metrics follow binary relevance: a recommended paper is relevant iff its paper_id appears in ``ground_truth_ids``.
Users with zero hits score 0 on all rank-based metrics — they are NEVER dropped from the average.

Metric catalogue
----------------
ndcg_at_k          -- DCG@k / ideal-DCG (ideal uses min(k, |targets|) relevant docs)
recall_at_k        -- |hits| / |targets|   (0 when no targets)
hit_rate_at_k      -- 1 if any hit else 0
mrr                -- 1 / rank_of_first_hit  (0 if no hit)
oracle_at_k        -- DCG achievable if targets are always placed first (the ceiling)
topic_entropy      -- Shannon entropy over primary OpenAlex topic_id of the top-k list
intra_list_distance -- mean pairwise cosine distance among the top-k embeddings
facet_novelty      -- fraction of unique canonical facet ids in top-k not seen in user profile facets
recall_at_100      -- Recall@100 for the first-stage candidate ceiling
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set

import numpy as np


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _hits(recommended: Sequence[str], relevant: Set[str], k: int) -> List[str]:
    """Return the hit paper_ids in the top-k recommended list."""
    return [p for p in recommended[:k] if p in relevant]


# ---------------------------------------------------------------------------
# Rank-quality metrics
# ---------------------------------------------------------------------------

def ndcg_at_k(
    recommended_ids: Sequence[str],
    ground_truth_ids: Sequence[str],
    k: int = 10,
) -> float:
    """Normalised Discounted Cumulative Gain at rank k.

    Binary relevance: rel_i = 1 if recommended_ids[i] in ground_truth_ids else 0.
    Ideal DCG uses min(k, |targets|) relevant documents placed at ranks 1..min(k,n).
    Returns 0.0 when ground_truth_ids is empty or the list has no hits.
    """
    relevant = set(ground_truth_ids)
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, pid in enumerate(recommended_ids[:k])
        if pid in relevant
    )
    n_ideal = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(n_ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def recall_at_k(
    recommended_ids: Sequence[str],
    ground_truth_ids: Sequence[str],
    k: int = 10,
) -> float:
    """Recall@k = |hits in top-k| / |targets|. Returns 0.0 when targets is empty."""
    relevant = set(ground_truth_ids)
    if not relevant:
        return 0.0
    return float(len(_hits(recommended_ids, relevant, k)) / len(relevant))


def hit_rate_at_k(
    recommended_ids: Sequence[str],
    ground_truth_ids: Sequence[str],
    k: int = 10,
) -> float:
    """1.0 if at least one recommended paper is relevant, else 0.0."""
    relevant = set(ground_truth_ids)
    return 1.0 if _hits(recommended_ids, relevant, k) else 0.0


def mrr(
    recommended_ids: Sequence[str],
    ground_truth_ids: Sequence[str],
) -> float:
    """Mean Reciprocal Rank — reciprocal of the first hit position (1-indexed). 0 if no hit."""
    relevant = set(ground_truth_ids)
    for rank, pid in enumerate(recommended_ids, start=1):
        if pid in relevant:
            return float(1.0 / rank)
    return 0.0


# ---------------------------------------------------------------------------
# Oracle (ceiling) and candidate ceiling
# ---------------------------------------------------------------------------

def oracle_at_k(
    ground_truth_ids: Sequence[str],
    k: int = 10,
) -> float:
    """The maximum possible NDCG@k — all relevant targets that can fit in top-k placed first.

    Returns 0.0 when ground_truth_ids is empty, else 1.0.
    """
    n = len(set(ground_truth_ids))
    if n == 0:
        return 0.0
    return 1.0


def recall_at_100(
    candidate_ids: Sequence[str],
    ground_truth_ids: Sequence[str],
) -> float:
    """Candidate-stage ceiling: Recall@100 of the first-stage retrieved pool."""
    return recall_at_k(candidate_ids, ground_truth_ids, k=100)


# ---------------------------------------------------------------------------
# Diversity / novelty metrics
# ---------------------------------------------------------------------------

def topic_entropy(
    paper_ids: Sequence[str],
    topic_map: Dict[str, str],
    k: int = 10,
    unknown_label: str = "UNKNOWN",
) -> float:
    """Shannon entropy of primary OpenAlex topic_id distribution over top-k papers.

    ``topic_map`` maps paper_id -> primary_topic_id (or "" / None when missing).
    Missing topics are assigned ``unknown_label`` (treated as one category, per plan §6.1).
    Returns 0.0 for an empty list.
    """
    topics = [topic_map.get(pid) or unknown_label for pid in paper_ids[:k]]
    if not topics:
        return 0.0
    counts: Dict[str, int] = {}
    for t in topics:
        counts[t] = counts.get(t, 0) + 1
    n = len(topics)
    entropy = -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)
    return float(entropy)


def topic_entropy_normalised(
    paper_ids: Sequence[str],
    topic_map: Dict[str, str],
    k: int = 10,
    unknown_label: str = "UNKNOWN",
) -> float:
    """topic_entropy / log2(k) — maps to [0, 1] for k items."""
    raw = topic_entropy(paper_ids, topic_map, k, unknown_label)
    denom = math.log2(k) if k > 1 else 1.0
    return float(raw / denom)


def intra_list_distance(
    paper_ids: Sequence[str],
    embedding_matrix: np.ndarray,
    paper_id_index: Dict[str, int],
    k: int = 10,
) -> float:
    """Mean pairwise cosine distance among the top-k paper embeddings.

    ``embedding_matrix`` is shape (N, D) with L2-normalised rows.
    ``paper_id_index`` maps paper_id -> row index.
    Returns 0.0 for fewer than 2 papers with known embeddings.
    """
    ids = [p for p in paper_ids[:k] if p in paper_id_index]
    if len(ids) < 2:
        return 0.0
    vecs = np.stack([embedding_matrix[paper_id_index[p]] for p in ids])  # (m, D)
    # Cosine similarity matrix (vecs already L2-normalised -> dot product = cosine)
    sim = vecs @ vecs.T  # (m, m)
    m = len(ids)
    # Upper triangle, excluding diagonal
    mask = np.triu(np.ones((m, m), dtype=bool), k=1)
    similarities = sim[mask]
    # Distance = 1 - cosine_similarity
    return float(1.0 - similarities.mean()) if similarities.size > 0 else 0.0


def facet_novelty(
    paper_ids: Sequence[str],
    user_canonical_facet_ids: Set[str],
    paper_facet_map: Dict[str, Set[str]],
    k: int = 10,
) -> float:
    """Fraction of unique canonical facet IDs in top-k that are NOT in the user's profile facets.

    ``paper_facet_map`` maps paper_id -> set of canonical_id strings.
    ``user_canonical_facet_ids`` is the user's known facet set F_u.
    Returns 0.0 when the top-k papers have no facets at all.
    """
    all_facets: Set[str] = set()
    for pid in paper_ids[:k]:
        all_facets.update(paper_facet_map.get(pid, set()))
    if not all_facets:
        return 0.0
    novel = all_facets - user_canonical_facet_ids
    return float(len(novel) / len(all_facets))


# ---------------------------------------------------------------------------
# Aggregation utilities
# ---------------------------------------------------------------------------

def aggregate_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute macro-averaged metrics from a list of per-user metric dicts."""
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: float(np.mean([r.get(k, 0.0) for r in rows])) for k in keys}


def per_user_metrics(
    recommended_ids: Sequence[str],
    ground_truth_ids: Sequence[str],
    candidate_ids: Sequence[str],
    topic_map: Dict[str, str],
    embedding_matrix: Optional[np.ndarray] = None,
    paper_id_index: Optional[Dict[str, int]] = None,
    user_canonical_facet_ids: Optional[Set[str]] = None,
    paper_facet_map: Optional[Dict[str, Set[str]]] = None,
    k: int = 10,
) -> Dict[str, float]:
    """Compute all per-user metrics for one recommendation result."""
    gt = list(ground_truth_ids)
    rec = list(recommended_ids)
    cands = list(candidate_ids)

    result: Dict[str, float] = {
        "ndcg_at_10": ndcg_at_k(rec, gt, k),
        "recall_at_10": recall_at_k(rec, gt, k),
        "hit_rate_at_10": hit_rate_at_k(rec, gt, k),
        "mrr": mrr(rec, gt),
        "oracle_at_10": oracle_at_k(gt, k),
        "recall_at_100": recall_at_100(cands, gt),
        "topic_entropy": topic_entropy(rec, topic_map, k),
        "topic_entropy_norm": topic_entropy_normalised(rec, topic_map, k),
        "n_targets": float(len(set(gt))),
    }

    if embedding_matrix is not None and paper_id_index is not None:
        result["ild"] = intra_list_distance(rec, embedding_matrix, paper_id_index, k)
    else:
        result["ild"] = 0.0

    if user_canonical_facet_ids is not None and paper_facet_map is not None:
        result["facet_novelty"] = facet_novelty(rec, user_canonical_facet_ids, paper_facet_map, k)
    else:
        result["facet_novelty"] = 0.0

    return result

