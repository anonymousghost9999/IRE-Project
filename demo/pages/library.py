"""Page 4: My Library — Saved Papers and Export."""

import json
from typing import Any, Dict, List
import streamlit as st

from demo.components.facet_pills import render_pills_container
from paperfeed.data.cohort import clean_id


def generate_bibtex(papers: List[Dict[str, Any]]) -> str:
    """Generate BibTeX entries for saved papers."""
    entries = []
    for idx, p in enumerate(papers, 1):
        pid = clean_id(p.get("paper_id", f"paper_{idx}"))
        title = p.get("title", "Untitled")
        authors = p.get("authors", "Anonymous")
        year = p.get("year", 2024)
        venue = p.get("venue", "OpenAlex")
        
        entry = (
            f"@article{{{pid},\n"
            f"  title = {{{title}}},\n"
            f"  author = {{{authors}}},\n"
            f"  year = {{{year}}},\n"
            f"  journal = {{{venue}}},\n"
            f"  url = {{{p.get('url', '')}}}\n"
            f"}}"
        )
        entries.append(entry)
    return "\n\n".join(entries)


def render_library_page() -> None:
    library: List[Dict[str, Any]] = st.session_state.get("library", [])

    st.markdown(
        f"""
        <div style="margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <h2 style="margin: 0; font-size: 1.6rem; font-weight: 700; color: #F8FAFC;">🔖 My Library</h2>
                <span style="background: rgba(251, 191, 36, 0.15); color: #FBBF24; border: 1px solid rgba(251, 191, 36, 0.3); padding: 3px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600;">
                    {len(library)} Saved
                </span>
            </div>
            <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 4px;">
                Papers you've bookmarked during research sessions. Automatically excluded from future discovery candidate sets.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not library:
        st.markdown(
            """
            <div style="text-align: center; padding: 40px 20px; background-color: #1C1F2A; border: 1px dashed #2E3348; border-radius: 8px; margin-top: 20px;">
                <div style="font-size: 2.2rem; margin-bottom: 8px;">📚</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #E2E8F0; margin-bottom: 6px;">Your library is currently empty</div>
                <div style="font-size: 0.85rem; color: #94A3B8; max-width: 420px; margin: 0 auto 16px auto;">
                    Click the <strong>⭐ Save</strong> button on any paper card in your feed to curate your reading list here.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Export toolbar
    col_exp1, col_exp2, col_sp = st.columns([1.5, 1.5, 4.0])
    with col_exp1:
        bibtex_str = generate_bibtex(library)
        st.download_button(
            "📥 Export BibTeX",
            data=bibtex_str,
            file_name="paperfeed_library.bib",
            mime="text/plain",
            use_container_width=True,
        )
    with col_exp2:
        json_str = json.dumps(library, indent=2)
        st.download_button(
            "📥 Export JSON",
            data=json_str,
            file_name="paperfeed_library.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("<hr style='border: 1px solid #2E3348; margin: 16px 0;'>", unsafe_allow_html=True)

    # Render saved paper cards
    for idx, item in enumerate(library, 1):
        pid = clean_id(item.get("paper_id", ""))
        title = item.get("title", "Untitled")
        authors = item.get("authors", "Research Community")
        venue = item.get("venue", "OpenAlex")
        year = item.get("year", 2024)
        url = item.get("url", f"https://openalex.org/{pid}")
        citations = item.get("citations", 0)
        facets = item.get("facets", [])

        pills_html = render_pills_container(facets)

        st.markdown(
            f"""
            <div style="background-color: #1C1F2A; border: 1px solid #2E3348; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <a href="{url}" target="_blank" style="font-size: 1.05rem; font-weight: 600; color: #60A5FA; text-decoration: none;">
                        {idx}. {title} ↗
                    </a>
                </div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px; margin-bottom: 6px;">
                    {authors} &bull; <span style="color: #CBD5E1;">{venue}</span> ({year}) &bull; {citations:,} citations
                </div>
                {pills_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_rm, col_gap = st.columns([1.2, 6.0])
        with col_rm:
            if st.button("🗑 Remove", key=f"rm_lib_{pid}"):
                library = [p for p in library if clean_id(p.get("paper_id", "")) != pid]
                st.session_state["library"] = library
                st.toast("Removed from library")
                st.rerun()
