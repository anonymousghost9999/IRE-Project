"""Baseline recommendation systems for Track 1 evaluation (Phase 6.2 / plan.md §6.2 / design.md §13.1).

Five baselines — all enforcing the same temporal filter (year <= T) and exclusion set E_u
as the full Paper Feed system:

1. popularity_baseline     -- highest citation_count_T in user's primary subfield (papers since 2012)
2. bm25_author_baseline    -- BM25s retrieval using the full concatenated titles/abstracts of past papers
3. dense_centroid_baseline -- dense kNN on unweighted mean SPECTER2 embedding (no session)
4. citation_only_baseline  -- log(1 + BC + CC) over the eligible corpus, no text
5. hybrid_persona_baseline -- persona vector + pseudo-query RRF, no citation/facets/MMR (own tuned gamma_rec, n_terms)

Two anchors (no significance tests):
- random_anchor_ids        -- n_draws independent random samples of k eligible papers
- oracle_at_k              -- from metrics.py (upper bound; ground-truth targets ranked first)

``random_baseline`` (one draw) is kept for ad-hoc use and is not part of the H1 family. No baseline falls back to
another one: a failure raises, so it cannot show up in the tables disguised as a real baseline score.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

import numpy as np

from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import clean_id
from paperfeed.data.ground_truth import to_uri
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
    query_ids = list(dict.fromkeys([to_uri(pid) for pid in paper_ids] + paper_ids))
    subfields: Dict[str, int] = {}
    try:
        with db.get_connection() as conn:
            placeholders = ",".join("?" * len(query_ids))
            rows = conn.execute(
                f"SELECT subfield_id FROM papers WHERE paper_id IN ({placeholders})",
                query_ids,
            ).fetchall()
        for row in rows:
            sf = row[0]
            if sf:
                subfields[sf] = subfields.get(sf, 0) + 1
    except Exception as exc:
        logger.warning("Failed to query primary subfield: %s", exc)
        return None
    return max(subfields, key=subfields.__getitem__) if subfields else None


def _get_eligible_in_subfield(
    db: Any,
    eligible: Set[str],
    excluded: Set[str],
    subfield: Optional[str],
) -> List[str]:
    """Papers that are eligible, not excluded, and share the given subfield_id.

    Falls back to all eligible papers when subfield is None or returns nothing.
    """
    pool = _eligible_not_excluded(eligible, excluded)
    if not subfield:
        return pool
    try:
        with db.get_connection() as conn:
            rows = conn.execute(
                "SELECT paper_id FROM papers WHERE subfield_id = ?", (subfield,)
            ).fetchall()
        in_subfield = {clean_id(r[0]) for r in rows}
        filtered = [p for p in pool if p in in_subfield]
        return filtered if filtered else pool
    except Exception as exc:
        logger.warning("Failed to filter eligible papers by subfield %s: %s", subfield, exc)
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
    subfield: Optional[str] = None,
) -> List[str]:
    """Top-k papers by citation_count_T in the user's primary subfield (papers since 2012).

    The primary subfield is the modal subfield of the authored papers; ``subfield`` overrides it (the seed topics'
    subfield in the zero-publication slice, design §13.1).
    """
    excluded = profile.exclusion_sources.build(diagnostic=False, zero_pub=profile.zero_pub)
    if subfield is None:
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
        query += " AND subfield_id = ?"
        params.append(subfield)
    # Deep enough that excluded papers (the user's own and cited ones) cannot exhaust the list.
    query += " ORDER BY citation_count_T DESC, paper_id ASC LIMIT ?"
    params.append(k + len(excluded) + 300)

    # No fallback to random: a failure here must be visible, not reported as a "popularity" score.
    with resources.db.get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    out = []
    for r in rows:
        pid = clean_id(r[0])
        if pid not in excluded and pid in resources.eligible:
            out.append(pid)
        if len(out) >= k:
            break
    if len(out) < k:
        logger.warning("popularity_baseline: only %d papers for subfield %s.", len(out), subfield)
    return out


# ---------------------------------------------------------------------------
# Baseline 3: BM25 lexical
# ---------------------------------------------------------------------------

def bm25_author_baseline(
    resources: RecommenderResources,
    profile: UserProfile,
    k: int = 10,
    n_terms: Optional[int] = None,
) -> List[str]:
    """BM25s retrieval using the concatenation of past paper titles and abstracts (plan §6.2: the full text).

    ``n_terms`` truncates the query only when given explicitly; the evaluation uses the full text.
    """
    excluded = profile.exclusion_sources.build(diagnostic=False, zero_pub=profile.zero_pub)
    parts: List[str] = []
    if profile.zero_pub:
        # Zero-publication slice: the seed-topic text is the user's only text (design §13.1).
        parts = [str(t) for t in getattr(profile, "seed_topics", []) if t]
    for p in profile.authored_papers:
        if p.get("title"):
            parts.append(str(p["title"]))
        if p.get("abstract"):
            parts.append(str(p["abstract"]))
    query_text = " ".join(parts)
    if n_terms is not None:
        query_text = " ".join(query_text.split()[:n_terms])

    if not query_text.strip():
        logger.warning("bm25_author_baseline: user has no title or abstract text to query with.")
        return []

    results = resources.retriever.lexical_index.search(
        query_text, k=k, exclusion_set=excluded, candidate_filter=resources.eligible
    )
    return [clean_id(pid) for pid, _ in results][:k]


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
        # No stored vector for any authored paper: the persona (title-only embeddings) is the closest stand-in.
        centroid = profile.persona.vector
    else:
        vecs = store.select(present)  # (n, D)
        centroid = vecs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

    hits = resources.retriever.dense_index.search(
        centroid, k=k, exclusion_set=excluded, candidate_filter=resources.eligible
    )
    return [clean_id(pid) for pid, _ in hits][:k]


# ---------------------------------------------------------------------------
# Baseline 4b: Citation-only (log(1 + BC + CC), no text)
# ---------------------------------------------------------------------------

def citation_only_baseline(
    resources: RecommenderResources,
    profile: UserProfile,
    k: int = 10,
) -> List[str]:
    """Rank eligible papers by log(1 + BC + CC) with the user's authored papers (plan §6.2, baseline 4).

    BC(p) counts p's references shared with the authored papers; CC(p) counts third-party papers (year <= T) that
    cite both p and an authored paper: the same definitions as the system's citation feature (D§4.4, §9), but over
    the whole eligible corpus instead of the 100 candidates. Not applicable to zero-publication users (returns []).
    """
    from collections import Counter

    from paperfeed.rec.features import edges

    ctx = profile.citation_ctx
    if not ctx.authored:
        return []
    excluded = profile.exclusion_sources.build(diagnostic=False, zero_pub=profile.zero_pub)

    bc: Counter = Counter()  # paper -> references it shares with the authored papers
    for src, _tgt in edges(resources.db, "target_paper_id", sorted(ctx.ref_to_authored), resources.cutoff_year):
        bc[src] += 1
    cc: Counter = Counter()  # paper -> third-party papers co-citing it with an authored paper
    for _src, tgt in edges(resources.db, "source_paper_id", sorted(ctx.citer_to_authored), resources.cutoff_year):
        cc[tgt] += 1

    scores = {
        p: math.log1p(bc[p] + cc[p])
        for p in set(bc) | set(cc)
        if p in resources.eligible and p not in excluded
    }
    return sorted(scores, key=lambda p: (-scores[p], p))[:k]


# ---------------------------------------------------------------------------
# Baseline 5: Hybrid-Persona (persona + pseudo-query RRF, no citation/facets/MMR)
# ---------------------------------------------------------------------------

def hybrid_persona_baseline(
    resources: RecommenderResources,
    profile: UserProfile,
    k: int = 10,
    n_terms: int = 10,
    n_titles: int = 3,
    gamma_rec: Optional[float] = None,
) -> List[str]:
    """Hybrid retrieval with persona vector + pseudo-query BM25s fused by RRF.

    No citation topology, no facet scoring, and no MMR reranking. ``gamma_rec`` and ``n_terms`` are this baseline's
    own tuned values (plan §6.3); with ``gamma_rec=None`` the profile's persona is used as is.
    """
    from paperfeed.user.query import build_feed_query
    from paperfeed.user.session import SessionState

    excluded = profile.exclusion_sources.build(diagnostic=False, zero_pub=profile.zero_pub)
    persona_vec = _persona_vector_for_gamma(resources, profile, gamma_rec)
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

    cands = feed_candidates(
        resources.retriever,
        persona_vec,
        pseudo_query,
        session_vec=None,
        exclusion_set=excluded,
        candidate_filter=resources.eligible,
        rho=0.0,
        normalize_session=False,
    )
    return [c.paper_id for c in cands[:k]]


def _persona_vector_for_gamma(
    resources: RecommenderResources,
    profile: UserProfile,
    gamma_rec: Optional[float],
) -> np.ndarray:
    """The persona rebuilt with another recency decay; the profile's own persona when that is not possible."""
    if gamma_rec is None or profile.zero_pub or profile.persona.source != "authored":
        return profile.persona.vector
    from paperfeed.user.persona import build_persona

    return build_persona(
        profile.authored_papers, resources.store, resources.encoder, resources.cutoff_year, gamma_rec, allow_partial=True
    ).vector


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

# The five baselines of plan §6.2 / design §13.1 (the H1 family). Random is an anchor, not a baseline: it is reported
# through ``random_anchor_ids`` (mean over 100 draws) and never enters the significance tests.
BASELINE_NAMES = [
    "popularity",
    "bm25_author",
    "dense_centroid",
    "citation_only",
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
        return popularity_baseline(resources, profile, k, subfield=kwargs.get("subfield"))
    if name == "bm25_author":
        return bm25_author_baseline(resources, profile, k)
    if name == "dense_centroid":
        return dense_centroid_baseline(resources, profile, k)
    if name == "citation_only":
        return citation_only_baseline(resources, profile, k)
    if name == "hybrid_persona":
        n_terms = int(kwargs.get("n_terms", 10))
        n_titles = int(kwargs.get("n_titles", 3))
        gamma_rec = kwargs.get("gamma_rec")
        return hybrid_persona_baseline(
            resources, profile, k, n_terms, n_titles, float(gamma_rec) if gamma_rec is not None else None
        )
    raise ValueError(f"Unknown baseline name '{name}'. Expected one of {BASELINE_NAMES}.")

