"""Candidate generation (Phase 5.1 / docs/plan.md §5.1 / docs/design.md §8).

Both modes filter to year <= T, ``in_candidates`` and remove E_u *before* anything is scored, take the top
M_channel (200) per channel after filtering, fuse by RRF (k = 60) and keep M (100). Every returned candidate
keeps its rank in each channel (needed by the V3 learned ranker and for the Recall@100 candidate ceiling).

- Feed mode: dense kNN on q_t = normalize((1 - rho) v_sess + rho v_pers) (the persona alone when the session is
  empty) + BM25s on the pseudo-query of 4.5.
- Search mode: ``adhoc_query`` embedding of the query text + BM25s on the same text. The optional cross-encoder
  rerank is an Extended experiment and is not part of this module.
"""

from typing import Any, Iterable, List, Mapping, Optional, Protocol, Set

import numpy as np

from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import clean_id
from paperfeed.index.retriever import HybridRetriever, RetrievedCandidate

logger = get_logger("rec.candidates")

DEFAULT_RHO = 0.3


class QueryEncoder(Protocol):
    """The slice of ``SPECTER2Encoder`` used by search mode."""

    def embed_query(self, query: str) -> np.ndarray: ...


def query_vector(
    persona_vec: np.ndarray,
    session_vec: Optional[np.ndarray] = None,
    rho: float = DEFAULT_RHO,
    normalize_session: bool = False,
) -> np.ndarray:
    """Dense query vector q_t for feed mode.

    With an empty session (``None`` or an all-zero vector) this is the persona itself. Otherwise
    normalize((1 - rho) v_sess + rho v_pers). ``normalize_session`` rescales v_sess to unit length first: the EMA
    of 4.4 starts from 0, so after one click ||v_sess|| = 1 - alpha and the session would be under-weighted
    against rho. It is off by default to follow the design formula literally and is an ablation switch.
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho}.")
    persona = np.asarray(persona_vec, dtype=np.float32)
    if session_vec is None:
        return persona
    session = np.asarray(session_vec, dtype=np.float32)
    session_norm = float(np.linalg.norm(session))
    if session_norm <= 1e-12:
        return persona
    if normalize_session:
        session = session / session_norm
    blended = (1.0 - rho) * session + rho * persona
    norm = float(np.linalg.norm(blended))
    if norm <= 1e-12:  # session exactly opposes the persona at rho = 0.5: nothing meaningful to blend
        logger.warning("Session and persona cancel out; falling back to the persona for q_t.")
        return persona
    return (blended / norm).astype(np.float32)


def eligible_paper_ids(db, cutoff_year: int) -> Set[str]:
    """Bare IDs of every paper that may be recommended: ``in_candidates = 1`` and year <= T."""
    with db.get_connection() as conn:
        rows = conn.execute(
            "SELECT paper_id FROM papers WHERE in_candidates = 1 AND year <= ?", (cutoff_year,)
        ).fetchall()
    return {clean_id(r[0]) for r in rows}


def feed_candidates(
    retriever: HybridRetriever,
    persona_vec: np.ndarray,
    pseudo_query: str,
    session_vec: Optional[np.ndarray] = None,
    exclusion_set: Optional[Iterable[str]] = None,
    candidate_filter: Optional[Iterable[str]] = None,
    rho: float = DEFAULT_RHO,
    normalize_session: bool = False,
    channel_depth: Optional[int] = None,
    top_m: Optional[int] = None,
) -> List[RetrievedCandidate]:
    """Feed-mode candidates: dense kNN on q_t + BM25s on the pseudo-query, fused by RRF.

    ``channel_depth`` / ``top_m`` override the retriever's M_channel / M for this call (ablations).
    """
    q_t = query_vector(persona_vec, session_vec, rho, normalize_session)
    return retriever.retrieve_candidates(
        dense_query_vec=q_t,
        lexical_query=pseudo_query,
        channel_depth=channel_depth,
        top_m=top_m,
        exclusion_set=exclusion_set,
        candidate_filter=candidate_filter,
    )


def search_candidates(
    retriever: HybridRetriever,
    encoder: QueryEncoder,
    query: str,
    exclusion_set: Optional[Iterable[str]] = None,
    candidate_filter: Optional[Iterable[str]] = None,
    channel_depth: Optional[int] = None,
    top_m: Optional[int] = None,
) -> List[RetrievedCandidate]:
    """Search-mode candidates: ``adhoc_query`` embedding + BM25s on the query text, fused by RRF."""
    query = (query or "").strip()
    if not query:
        return []
    return retriever.retrieve_candidates(
        dense_query_vec=encoder.embed_query(query),
        lexical_query=query,
        channel_depth=channel_depth,
        top_m=top_m,
        exclusion_set=exclusion_set,
        candidate_filter=candidate_filter,
    )


def retrieval_settings(cfg: Mapping[str, Any]) -> dict:
    """Retriever and q_t parameters from the ``retrieval`` / ``user_model`` sections of configs/default.yaml."""
    r, um = cfg.get("retrieval", {}), cfg.get("user_model", {})
    return {
        "channel_depth": int(r.get("channel_depth", 200)),
        "rrf_k": int(r.get("rrf_k", 60)),
        "candidate_pool_size": int(r.get("candidate_pool_size", 100)),
        "rho": float(um.get("rho_persona_weight", DEFAULT_RHO)),
    }
