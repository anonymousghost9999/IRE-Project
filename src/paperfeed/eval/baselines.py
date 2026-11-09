"""Baseline recommendation systems for Track 1 evaluation (Phase 6.2 / plan.md §6.2 / design.md §13.1).

Five baselines — all enforcing the same temporal filter (year <= T) and exclusion set E_u
as the full Paper Feed system:

1. random_baseline         -- random sample from eligible papers in user's primary domain
2. popularity_baseline     -- highest citation_count_T in user's primary subfield
3. bm25_author_baseline    -- BM25s retrieval using concatenated titles/abstracts of past papers
4. dense_centroid_baseline -- dense kNN on unweighted mean SPECTER2 embedding (no session)
5. hybrid_persona_baseline -- persona vector + pseudo-query RRF, no citation/facets/MMR

Two anchors (no significance tests):
- random_anchor_ids        -- n_draws independent random samples of k eligible papers
- oracle_at_k              -- from metrics.py (upper bound; ground-truth targets ranked first)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

import numpy as np

from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import clean_id
from paperfeed.rec.candidates import feed_candidates
from paperfeed.rec.recommender import RecommenderResources, UserProfile

logger = get_logger("eval.baselines")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eligible_not_excluded(eligible: Set[str], excluded: Set[str]) -> List[str]:
    """Sorted list of eligible papers that are not in the exclusion set."""
    return sorted(eligible - excluded)


def _primary_subfield(profile: UserProfile, db: Any) -> Optional[str]:
    """Modal primary OpenAlex subfield among the user's authored papers (<= T).

    Returns None when no authored papers have a known subfield.
    """
    if not profile.authored_papers:
        return None
    paper_ids = [p["paper_id"] for p in profile.authored_papers]
    subfields: Dict[str, int] = {}
    try:
        with db.get_connection() as conn:
            placeholders = ",".join("?" * len(paper_ids))
            rows = conn.execute(
                f"SELECT primary_topic_id FROM papers WHERE paper_id IN ({placeholders})",
                paper_ids,
            ).fetchall()
        for row in rows:
            sf = row[0]
            if sf:
                subfields[sf] = subfields.get(sf, 0) + 1
    except Exception:
        return None
    return max(subfields, key=subfields.__getitem__) if subfields else None


def _get_eligible_in_subfield(
    db: Any,
    eligible: Set[str],
    excluded: Set[str],
    subfield: Optional[str],
) -> List[str]:
    """Papers that are eligible, not excluded, and share the given primary_topic_id.

    Falls back to all eligible papers when subfield is None or returns nothing.
    """
    pool = _eligible_not_excluded(eligible, excluded)
    if not subfield:
        return pool
    try:
        with db.get_connection() as conn:
            rows = conn.execute(
                "SELECT paper_id FROM papers WHERE primary_topic_id = ?", (subfield,)
            ).fetchall()
        in_subfield = {clean_id(r[0]) for r in rows}
        filtered = [p for p in pool if p in in_subfield]
        return filtered if filtered else pool
    except Exception:
        return pool


# ---------------------------------------------------------------------------
# Baseline 1: Random
# ---------------------------------------------------------------------------

def random_baseline(
    resources: RecommenderResources,
    profile: UserProfile,
    k: int = 10,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """Return k random papers from the eligible pool, respecting E_u."""
    excluded = profile.exclusion_sources.build(diagnostic=False, zero_pub=profile.zero_pub)
    pool = _eligible_not_excluded(resources.eligible, excluded)
    r = rng or random.Random(42)
    return r.sample(pool, min(k, len(pool)))


# ---------------------------------------------------------------------------
# Baseline 2: Popularity
# ---------------------------------------------------------------------------

def popularity_baseline(
    resources: RecommenderResources,
    profile: UserProfile,
    k: int = 10,
) -> List[str]:
    """Top-k papers by citation_count_T in user's primary subfield (year >= 2012)."""
    excluded = profile.exclusion_sources.build(diagnostic=False, zero_pub=profile.zero_pub)
    subfield = _primary_subfield(profile, resources.db)
    query = """
        SELECT paper_id
        FROM papers
        WHERE in_candidates = 1
          AND (year IS NULL OR year <= ?)
          AND (year IS NULL OR year >= 2012)
    """
    params = [resources.cutoff_year]
    if subfield:
        query += " AND primary_topic_id = ?"
        params.append(subfield)
    query += " ORDER BY citation_count_T DESC LIMIT 300"

    try:
        with resources.db.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        out = []
        for r in rows:
            pid = clean_id(r[0])
            if pid not in excluded and pid in resources.eligible:
                out.append(pid)
            if len(out) >= k:
                break
        return out if out else random_baseline(resources, profile, k)
    except Exception as exc:
        logger.warning("popularity_baseline DB error: %s", exc)
        return random_baseline(resources, profile, k)


# ---------------------------------------------------------------------------
# Baseline 3: BM25 lexical
# ---------------------------------------------------------------------------

def bm25_author_baseline(
    resources: RecommenderResources,
    profile: UserProfile,
    k: int = 10,
    n_terms: int = 50,
) -> List[str]:
    """BM25s retrieval using the concatenation of past paper titles and abstracts."""
    excluded = profile.exclusion_sources.build(diagnostic=False, zero_pub=profile.zero_pub)
    parts: List[str] = []
    for p in profile.authored_papers:
        if p.get("title"):
            parts.append(str(p["title"]))
        if p.get("abstract"):
            parts.append(str(p["abstract"]))
    query_text = " ".join(parts)
    tokens = query_text.split()
    query_text = " ".join(tokens[:n_terms]) if len(tokens) > n_terms else query_text

    if not query_text.strip():
        return random_baseline(resources, profile, k)

    try:
        results = resources.retriever.lexical_index.search(query_text, k=k + len(excluded) + 10)
        out = []
        for item in results:
            pid = clean_id(item[0] if isinstance(item, (tuple, list)) else getattr(item, "paper_id", item))
            if pid not in excluded and pid in resources.eligible:
                out.append(pid)
            if len(out) >= k:
                break
        return out
    except Exception as exc:
        logger.warning("bm25_author_baseline retrieval error: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Baseline 4: Dense centroid (unweighted mean)
# ---------------------------------------------------------------------------

def dense_centroid_baseline(
    resources: RecommenderResources,
    profile: UserProfile,
    k: int = 10,
) -> List[str]:
    """Dense kNN on unweighted mean SPECTER2 embedding of past authored papers."""
    excluded = profile.exclusion_sources.build(diagnostic=False, zero_pub=profile.zero_pub)
    store = resources.store
    paper_ids = [p["paper_id"] for p in profile.authored_papers]
    present = [pid for pid in paper_ids if not store.missing([pid])]
    if not present:
        if profile.persona is not None and getattr(profile.persona, "vector", None) is not None:
            centroid = profile.persona.vector
        else:
            return random_baseline(resources, profile, k)
    else:
        vecs = store.select(present)  # (n, D)
        centroid = vecs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

    try:
        depth = k + len(excluded) + 50
        hits = resources.retriever.dense_index.search(centroid, k=depth)
        out = []
        for item in hits:
            pid = clean_id(item[0] if isinstance(item, (tuple, list)) else getattr(item, "paper_id", item))
            if pid not in excluded and pid in resources.eligible:
                out.append(pid)
            if len(out) >= k:
                break
        return out
    except Exception as exc:
        logger.warning("dense_centroid_baseline search error: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Baseline 5: Hybrid-Persona (persona + pseudo-query RRF, no citation/facets/MMR)
# ---------------------------------------------------------------------------

def hybrid_persona_baseline(
    resources: RecommenderResources,
    profile: UserProfile,
    k: int = 10,
    n_terms: int = 10,
    n_titles: int = 3,
) -> List[str]:
    """Hybrid retrieval with persona vector + pseudo-query BM25s fused by RRF.

    No citation topology, no facet scoring, and no MMR reranking.
    """
    from paperfeed.user.query import build_feed_query
    from paperfeed.user.session import SessionState

    excluded = profile.exclusion_sources.build(diagnostic=False, zero_pub=profile.zero_pub)
    session = SessionState(dim=len(profile.persona.vector))

    pseudo_query = build_feed_query(
        profile.user_facets,
        session,
        resources.db,
        profile.authored_papers,
        n_terms=n_terms,
        n_titles=n_titles,
        cutoff_year=resources.cutoff_year,
    )

    try:
        cands = feed_candidates(
            resources.retriever,
            profile.persona.vector,
            pseudo_query,
            session_vec=None,
            exclusion_set=excluded,
            candidate_filter=resources.eligible,
            rho=0.0,
            normalize_session=False,
        )
        return [c.paper_id for c in cands[:k]]
    except Exception as exc:
        logger.warning("hybrid_persona_baseline error: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Random anchor
# ---------------------------------------------------------------------------

def random_anchor_ids(
    resources: RecommenderResources,
    profile: UserProfile,
    k: int = 10,
    n_draws: int = 100,
    seed: int = 42,
) -> List[List[str]]:
    """Return n_draws independent random samples of k eligible papers."""
    excluded = profile.exclusion_sources.build(diagnostic=False, zero_pub=profile.zero_pub)
    pool = _eligible_not_excluded(resources.eligible, excluded)
    rng = random.Random(seed)
    k_eff = min(k, len(pool))
    return [rng.sample(pool, k_eff) for _ in range(n_draws)]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BASELINE_NAMES = [
    "random",
    "popularity",
    "bm25_author",
    "dense_centroid",
    "hybrid_persona",
]


def run_baseline(
    name: str,
    resources: RecommenderResources,
    profile: UserProfile,
    k: int = 10,
    rng: Optional[random.Random] = None,
    baseline_kwargs: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Dispatch to named baseline."""
    kwargs = dict(baseline_kwargs or {})
    if name == "random":
        return random_baseline(resources, profile, k, rng)
    if name == "popularity":
        return popularity_baseline(resources, profile, k)
    if name == "bm25_author":
        return bm25_author_baseline(resources, profile, k)
    if name == "dense_centroid":
        return dense_centroid_baseline(resources, profile, k)
    if name == "hybrid_persona":
        n_terms = int(kwargs.get("n_terms", 10))
        n_titles = int(kwargs.get("n_titles", 3))
        return hybrid_persona_baseline(resources, profile, k, n_terms, n_titles)
    raise ValueError(f"Unknown baseline name '{name}'. Expected one of {BASELINE_NAMES}.")

