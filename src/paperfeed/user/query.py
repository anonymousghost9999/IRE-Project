"""Pseudo-query builder for the lexical channel of feed mode (Phase 4.5 / docs/plan.md §4.5 / docs/design.md §8.1).

Feed mode has no user-written query, so the BM25s channel gets a short facet-focused one:
(a) the most frequent canonical task / method / dataset terms of F_u, then
(b) the titles of the last 3 session papers, or, when the session is empty (always in Track 1), the titles of the
    user's 3 most recent authored papers.
The number of facet terms (``n_terms``) is the query-length parameter tuned for the Hybrid-Persona baseline.
"""

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import clean_id
from paperfeed.data.ground_truth import to_uri
from paperfeed.user.facets import UserFacets
from paperfeed.user.session import SessionState

logger = get_logger("user.query")

DEFAULT_N_TERMS = 10
DEFAULT_N_TITLES = 3


def _clean_titles(titles: Iterable[str]) -> List[str]:
    """Non-empty titles in order, without placeholders or case-insensitive duplicates."""
    seen, out = set(), []
    for t in titles:
        t = (t or "").strip()
        if t and t.lower() != "untitled" and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def most_recent_titles(authored_papers: Sequence[Dict[str, Any]], n: int = DEFAULT_N_TITLES) -> List[str]:
    """Titles of the ``n`` most recent authored papers (year descending, ties by paper ID for reproducibility)."""
    ranked = sorted(authored_papers, key=lambda p: (-int(p.get("year") or 0), p["paper_id"]))
    return _clean_titles(p.get("title", "") for p in ranked)[:n]


def titles_for(db, paper_ids: Sequence[str]) -> List[str]:
    """Titles of the given papers (bare or URI IDs) in the given order; papers missing from the database are skipped."""
    if not paper_ids:
        return []
    uris = [to_uri(p) for p in paper_ids]
    by_id = {clean_id(r["paper_id"]): r["title"] for r in db.get_papers_batch(uris)}
    return _clean_titles(by_id.get(clean_id(p), "") for p in paper_ids)


def build_pseudo_query(
    user_facets: UserFacets,
    title_texts: Sequence[str] = (),
    n_terms: int = DEFAULT_N_TERMS,
) -> str:
    """Facet terms (most frequent first) followed by the given titles, as one whitespace-joined string.

    Returns an empty string when there is nothing to query with; the retriever then skips the lexical channel.
    """
    terms = _clean_titles(user_facets.top_names(max(n_terms, 0)))
    return " ".join(terms + _clean_titles(title_texts))


def build_feed_query(
    user_facets: UserFacets,
    session: Optional[SessionState],
    db,
    authored_papers: Sequence[Dict[str, Any]],
    n_terms: int = DEFAULT_N_TERMS,
    n_titles: int = DEFAULT_N_TITLES,
) -> str:
    """The feed-mode pseudo-query: titles come from the session when it has positive actions, else from authored papers."""
    if session is not None and not session.is_empty:
        titles = titles_for(db, session.recent_positives(n_titles))
    else:
        titles = most_recent_titles(authored_papers, n_titles)
    return build_pseudo_query(user_facets, titles, n_terms)


def query_settings(cfg: Mapping[str, Any]) -> Dict[str, int]:
    """``n_terms`` / ``n_titles`` from the ``user_model`` section of configs/default.yaml."""
    um = cfg.get("user_model", {})
    return {
        "n_terms": int(um.get("pseudo_query_terms", DEFAULT_N_TERMS)),
        "n_titles": int(um.get("pseudo_query_titles", DEFAULT_N_TITLES)),
    }
