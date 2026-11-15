"""Bridge module between Streamlit UI and the core PaperFeed recommender engine."""

import json
from typing import Any, Dict, List, Optional, Sequence, Set
import numpy as np
import streamlit as st

from demo.components.facet_pills import is_informative_facet
from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import clean_id
from paperfeed.data.ground_truth import to_uri
from paperfeed.rec.recommender import (
    RecommendConfig,
    RecommenderResources,
    UserProfile,
    build_user_profile,
    recommend,
)
from paperfeed.user.session import SessionState

logger = get_logger("demo.bridge")


def get_or_build_profile(resources: RecommenderResources) -> UserProfile:
    """Retrieve existing profile from session_state or build a new one."""
    cached = st.session_state.get("profile")
    current_mode = st.session_state.get("user_mode", "author")
    current_user_id = st.session_state.get("user_id", "A5060364305")
    
    if cached is not None and cached.user_id == current_user_id:
        if (current_mode == "author" and not cached.zero_pub) or (current_mode == "seed" and cached.zero_pub):
            return cached

    config: RecommendConfig = st.session_state.get("config")
    if current_mode == "author":
        authored_ids = st.session_state.get("authored_ids", [])
        profile = build_user_profile(
            resources,
            user_id=current_user_id,
            authored_ids=authored_ids,
            config=config,
        )
    else:
        seed_topics = st.session_state.get("seed_topics", ["Large Language Models"])
        profile = build_user_profile(
            resources,
            user_id=f"seed_{abs(hash(tuple(seed_topics))) % 100000}",
            seed_topics=seed_topics,
            config=config,
        )
    
    st.session_state["profile"] = profile
    return profile


def _format_authors(authors_val: Any) -> str:
    """Format authors list from DB string, JSON list, or list of strings/dicts."""
    if not authors_val:
        return "Unknown Authors"
    if isinstance(authors_val, str):
        try:
            parsed = json.loads(authors_val)
            if isinstance(parsed, list):
                authors_val = parsed
            else:
                return authors_val
        except Exception:
            return authors_val

    if isinstance(authors_val, list):
        names = []
        for a in authors_val:
            if not a:
                continue
            if isinstance(a, dict):
                names.append(a.get("display_name") or a.get("name") or str(a))
            else:
                names.append(str(a))
        if len(names) > 3:
            return f"{names[0]}, {names[1]}, et al."
        return ", ".join(names)
    return str(authors_val)


def _enrich_paper_metadata(resources: RecommenderResources, paper_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch raw paper metadata and facets for a list of paper IDs."""
    if not paper_ids:
        return {}
    
    uris = [to_uri(p) for p in paper_ids]
    clean_pids = [clean_id(p) for p in paper_ids]
    
    meta_by_id = {}
    with resources.db.get_connection() as conn:
        placeholders = ",".join("?" for _ in (uris + clean_pids))
        cursor = conn.execute(
            f"""
            SELECT paper_id, title, abstract, authors, year, venue, doi, url, citation_count_T, primary_topic_id
            FROM papers
            WHERE paper_id IN ({placeholders})
            """,
            (uris + clean_pids),
        )
        for row in cursor.fetchall():
            pid = clean_id(row[0])
            meta_by_id[pid] = {
                "paper_id": pid,
                "title": row[1] or "Untitled",
                "abstract": row[2] or "",
                "authors_str": _format_authors(row[3]),
                "year": row[4] or 2024,
                "venue": row[5] or "OpenAlex",
                "doi": row[6] or "",
                "url": row[7] or f"https://openalex.org/{pid}",
                "citations": row[8] or 0,
                "primary_topic": row[9] or "",
            }

    # Fetch canonical facets
    facets_by_id = {}
    if resources.facet_store is not None:
        try:
            facets_by_id = resources.facet_store.get_facets_by_paper_ids(clean_pids)
        except Exception as exc:
            logger.warning("Error fetching facets: %s", exc)

    # Attach facets to meta
    for pid, meta in meta_by_id.items():
        raw_facets = facets_by_id.get(pid, [])
        # Deduplicate and format top facets
        formatted_facets = []
        seen = set()
        for f in raw_facets:
            val = f.get("facet_value", "").strip()
            ftype = f.get("facet_type", "").lower()
            if val and is_informative_facet(val) and (ftype, val.lower()) not in seen:
                seen.add((ftype, val.lower()))
                formatted_facets.append({
                    "type": ftype,
                    "value": val,
                    "canonical_id": f.get("canonical_id", ""),
                    "confidence": f.get("confidence", 0.0),
                    "evidence_sentence": f.get("evidence_sentence", ""),
                })
        meta["facets"] = formatted_facets[:6]

    return meta_by_id


def generate_feed(resources: RecommenderResources, force: bool = False) -> List[Dict[str, Any]]:
    """Execute recommendation engine in feed mode and return enriched cards."""
    if not force and not st.session_state.get("feed_dirty", True):
        cached = st.session_state.get("feed_items")
        if cached:
            return cached

    profile = get_or_build_profile(resources)
    session: SessionState = st.session_state.get("session")
    config: RecommendConfig = st.session_state.get("config")

    result = recommend(
        resources=resources,
        profile=profile,
        session=session,
        mode="feed",
        config=config,
    )

    paper_ids = [item.paper_id for item in result.items]
    metadata = _enrich_paper_metadata(resources, paper_ids)

    enriched_items = []
    for item in result.items:
        pid = clean_id(item.paper_id)
        paper_info = metadata.get(pid, {
            "paper_id": pid,
            "title": item.title,
            "abstract": "",
            "authors_str": "Research Community",
            "year": item.year or 2024,
            "venue": "OpenAlex",
            "doi": "",
            "url": f"https://openalex.org/{pid}",
            "citations": 0,
            "facets": [],
        })

        expl_res = item.explanation
        expl_text = expl_res.explanation.text if (expl_res and expl_res.explanation) else None
        expl_evidence = expl_res.explanation.evidence if (expl_res and expl_res.explanation) else None
        failure = expl_res.failure if expl_res else None

        enriched_items.append({
            "rank": item.rank,
            "paper_id": pid,
            "title": paper_info.get("title", item.title),
            "abstract": paper_info.get("abstract", ""),
            "authors": paper_info.get("authors_str", ""),
            "year": paper_info.get("year", item.year),
            "venue": paper_info.get("venue", "OpenAlex"),
            "url": paper_info.get("url", f"https://openalex.org/{pid}"),
            "doi": paper_info.get("doi", ""),
            "citations": paper_info.get("citations", 0),
            "facets": paper_info.get("facets", []),
            "score": round(item.score, 4) if item.score is not None else 0.0,
            "rrf_score": round(item.rrf_score, 4),
            "dense_rank": item.dense_rank,
            "lexical_rank": item.lexical_rank,
            "contributions": item.contributions or {},
            "dominant_feature": expl_res.dominant_feature if expl_res else None,
            "explanation_text": expl_text,
            "explanation_evidence": expl_evidence,
            "explanation_failure": failure,
        })

    st.session_state["feed_items"] = enriched_items
    st.session_state["feed_meta"] = {
        "pseudo_query": result.pseudo_query,
        "n_candidates": result.n_candidates,
        "n_excluded": result.n_excluded,
        "session_active": result.session_active,
    }
    st.session_state["feed_dirty"] = False
    return enriched_items


def search_papers(resources: RecommenderResources, query_text: str, limit: int = 15) -> List[Dict[str, Any]]:
    """Run search mode for a query text."""
    if not query_text or not query_text.strip():
        return []

    profile = get_or_build_profile(resources)
    session: SessionState = st.session_state.get("session")
    config: RecommendConfig = st.session_state.get("config").with_overrides(k=limit)

    result = recommend(
        resources=resources,
        profile=profile,
        session=session,
        mode="search",
        query=query_text.strip(),
        config=config,
    )

    paper_ids = [item.paper_id for item in result.items]
    metadata = _enrich_paper_metadata(resources, paper_ids)

    results = []
    for item in result.items:
        pid = clean_id(item.paper_id)
        info = metadata.get(pid, {
            "paper_id": pid,
            "title": item.title,
            "abstract": "",
            "authors_str": "",
            "year": item.year or 2024,
            "venue": "OpenAlex",
            "url": f"https://openalex.org/{pid}",
            "citations": 0,
            "facets": [],
        })
        results.append({
            "rank": item.rank,
            "paper_id": pid,
            "title": info.get("title", item.title),
            "abstract": info.get("abstract", ""),
            "authors": info.get("authors_str", ""),
            "year": info.get("year", item.year),
            "venue": info.get("venue", "OpenAlex"),
            "url": info.get("url", f"https://openalex.org/{pid}"),
            "citations": info.get("citations", 0),
            "facets": info.get("facets", []),
            "dense_rank": item.dense_rank,
            "lexical_rank": item.lexical_rank,
        })
    return results


def apply_user_action(resources: RecommenderResources, paper_id: str, action: str, paper_meta: Optional[Dict[str, Any]] = None) -> None:
    """Process a user action (click, save, skip) and update session state."""
    session: SessionState = st.session_state.get("session")
    pid = clean_id(paper_id)

    if action == "skip":
        session.skip(pid)
        st.session_state["skipped_ids"].add(pid)
        st.session_state["feed_dirty"] = True
        return

    # For positive actions (click / save), we need the paper embedding
    embedding = None
    if pid in resources.store.paper_id_index:
        embedding = resources.store.select([pid])[0]
    elif paper_meta and resources.encoder is not None:
        title = paper_meta.get("title", "")
        abstract = paper_meta.get("abstract", "")
        paper_dict = {"paper_id": pid, "title": title, "abstract": abstract}
        embedding = resources.encoder.embed_papers([paper_dict])[0]
    else:
        # Fallback to zero vector if embedding unavailable
        embedding = np.zeros(768, dtype=np.float32)

    if action == "save":
        session.save(pid, embedding)
        # Add to library if not already present
        lib: List[Dict[str, Any]] = st.session_state.get("library", [])
        if not any(clean_id(p.get("paper_id", "")) == pid for p in lib):
            item = paper_meta.copy() if paper_meta else {"paper_id": pid}
            lib.append(item)
            st.session_state["library"] = lib
    elif action == "click":
        session.click(pid, embedding)

    st.session_state["feed_dirty"] = True
