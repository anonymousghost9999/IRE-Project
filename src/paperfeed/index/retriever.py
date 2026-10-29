"""Hybrid Candidate Generation & Reciprocal Rank Fusion (Phase 2, Step 2.5).

Governed by docs/prob.md §6D and docs/design.md §5, §8:
- Channels: Dense (FAISS IndexFlatIP) + Lexical (BM25s)
- Channel depth: M_channel = 200 candidates per channel
- Fusion: Reciprocal Rank Fusion (RRF) with k_rrf = 60
- Output pool: M = 100 candidates
- Hard exclusion: Filter out user's past authored papers and citations (exclusion_set)
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np

from paperfeed.common.logger import get_logger
from paperfeed.index.dense import DenseIndex, clean_id
from paperfeed.index.lexical import LexicalIndex

logger = get_logger("index.retriever")


def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k_rrf: int = 60,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF) across multiple ranked lists of document identifiers.
    
    Formula:
        RRF_score(d) = sum_{c} 1 / (k_rrf + rank_c(d))
    where rank_c(d) is 1-based (rank 1 is top).
    
    Deterministic tie-breaking by descending score, then ascending document ID.
    """
    scores: Dict[str, float] = defaultdict(float)

    for r_list in ranked_lists:
        for rank_0, doc_id in enumerate(r_list):
            doc_id = clean_id(doc_id)
            rank_1 = rank_0 + 1
            scores[doc_id] += 1.0 / (k_rrf + rank_1)

    # Sort descending by score, ascending by doc_id for determinism
    sorted_items = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return sorted_items


@dataclass
class RetrievedCandidate:
    """A fused candidate with its position in each channel (ranks are 1-based; None when the channel missed it)."""

    paper_id: str
    rrf_score: float
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    lexical_rank: Optional[int] = None
    lexical_score: Optional[float] = None


class HybridRetriever:
    """
    First-stage two-channel candidate generator (Dense + BM25s) with RRF fusion.
    """

    def __init__(
        self,
        dense_index: DenseIndex,
        lexical_index: LexicalIndex,
        channel_depth: int = 200,
        rrf_k: int = 60,
        candidate_pool_size: int = 100,
    ) -> None:
        self.dense_index = dense_index
        self.lexical_index = lexical_index
        self.channel_depth = channel_depth
        self.rrf_k = rrf_k
        self.candidate_pool_size = candidate_pool_size

    def retrieve(
        self,
        dense_query_vec: Optional[np.ndarray] = None,
        lexical_query: Optional[str] = None,
        channel_depth: Optional[int] = None,
        top_m: Optional[int] = None,
        exclusion_set: Optional[Union[Set[str], Iterable[str]]] = None,
        candidate_filter: Optional[Union[Set[str], Iterable[str]]] = None,
    ) -> List[Tuple[str, float]]:
        """Unified multi-channel retrieval with RRF fusion, as ``(paper_id, rrf_score)`` pairs.

        See ``retrieve_candidates`` for the arguments; this drops the per-channel ranks.
        """
        return [
            (c.paper_id, c.rrf_score)
            for c in self.retrieve_candidates(
                dense_query_vec, lexical_query, channel_depth, top_m, exclusion_set, candidate_filter
            )
        ]

    def retrieve_candidates(
        self,
        dense_query_vec: Optional[np.ndarray] = None,
        lexical_query: Optional[str] = None,
        channel_depth: Optional[int] = None,
        top_m: Optional[int] = None,
        exclusion_set: Optional[Union[Set[str], Iterable[str]]] = None,
        candidate_filter: Optional[Union[Set[str], Iterable[str]]] = None,
    ) -> List[RetrievedCandidate]:
        """
        Unified multi-channel retrieval with RRF fusion, keeping each channel's rank and score per candidate.

        Args:
            dense_query_vec: Dense query vector (e.g. q_t or adhoc_query vector).
            lexical_query: Text query string (e.g. pseudo-query or search terms).
            channel_depth: Depth per channel before fusion (default 200).
            top_m: Number of candidates to return after RRF fusion (default 100).
            exclusion_set: User exclusion set E_u (past authored/citations/session-consumed).
            candidate_filter: Whitelist of allowed candidate paper IDs (year <= T, in_candidates = 1).
        """
        depth = channel_depth if channel_depth is not None else self.channel_depth
        m_out = top_m if top_m is not None else self.candidate_pool_size

        ranked_lists: List[List[str]] = []
        dense_info: Dict[str, Tuple[int, float]] = {}
        lexical_info: Dict[str, Tuple[int, float]] = {}

        # 1. Dense retrieval channel
        if dense_query_vec is not None and len(self.dense_index) > 0:
            dense_hits = self.dense_index.search(
                query_vector=dense_query_vec,
                k=depth,
                exclusion_set=exclusion_set,
                candidate_filter=candidate_filter,
            )
            ranked_lists.append([pid for pid, _ in dense_hits])
            dense_info = {pid: (rank, score) for rank, (pid, score) in enumerate(dense_hits, start=1)}

        # 2. Lexical retrieval channel
        if lexical_query and len(self.lexical_index) > 0:
            lexical_hits = self.lexical_index.search(
                query=lexical_query,
                k=depth,
                exclusion_set=exclusion_set,
                candidate_filter=candidate_filter,
            )
            ranked_lists.append([pid for pid, _ in lexical_hits])
            lexical_info = {pid: (rank, score) for rank, (pid, score) in enumerate(lexical_hits, start=1)}

        if not ranked_lists:
            return []

        # 3. Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(ranked_lists, k_rrf=self.rrf_k)[:m_out]
        return [
            RetrievedCandidate(
                paper_id=pid,
                rrf_score=score,
                dense_rank=dense_info[pid][0] if pid in dense_info else None,
                dense_score=dense_info[pid][1] if pid in dense_info else None,
                lexical_rank=lexical_info[pid][0] if pid in lexical_info else None,
                lexical_score=lexical_info[pid][1] if pid in lexical_info else None,
            )
            for pid, score in fused
        ]
