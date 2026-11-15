"""Page 4: My Library — Saved Papers and Export."""

import json
from typing import Any, Dict, List
import streamlit as st

from demo.components.facet_pills import render_pills_container
from demo.theme import clean_html, get_theme
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
    t = get_theme()
    library: List[Dict[str, Any]] = st.session_state.get("library", [])

    header_html = clean_html(f"""
<div style="margin-bottom: 16px;">
<div style="display: flex; align-items: center; gap: 12px;">
<h2 style="margin: 0; font-size: 1.6rem; font-weight: 700; color: {t['text_primary']};">🔖 My Library</h2>
<span style="background: rgba(251, 191, 36, 0.15); color: #FBBF24; border: 1px solid rgba(251, 191, 36, 0.3); padding: 3px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600;">
{len(library)} Saved
</span>
</div>
<div style="font-size: 0.85rem; color: {t['text_secondary']}; margin-top: 4px;">
Papers you've bookmarked during research sessions. Automatically excluded from future discovery candidate sets.
</div>
</div>
""")
    st.markdown(header_html, unsafe_allow_html=True)

    if not library:
        empty_html = clean_html(f"""
<div style="text-align: center; padding: 40px 20px; background-color: {t['card_bg']}; border: 1px dashed {t['card_border']}; border-radius: 8px; margin-top: 20px;">
<div style="font-size: 2.2rem; margin-bottom: 8px;">📚</div>
<div style="font-size: 1.1rem; font-weight: 600; color: {t['text_primary']}; margin-bottom: 6px;">Your library is currently empty</div>
<div style="font-size: 0.85rem; color: {t['text_secondary']}; max-width: 420px; margin: 0 auto 16px auto;">
Click the <strong>⭐ Save</strong> button on any paper card in your feed to curate your reading list here.
</div>
</div>
""")
        st.markdown(empty_html, unsafe_allow_html=True)
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

    st.markdown(f"<hr style='border: 1px solid {t['card_border']}; margin: 16px 0;'>", unsafe_allow_html=True)

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

        card_html = clean_html(f"""
<div style="background-color: {t['card_bg']}; border: 1px solid {t['card_border']}; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
<div style="display: flex; justify-content: space-between; align-items: flex-start;">
<a href="{url}" target="_blank" style="font-size: 1.05rem; font-weight: 600; color: {t['link']}; text-decoration: none;">
{idx}. {title} ↗
</a>
</div>
<div style="font-size: 0.8rem; color: {t['text_secondary']}; margin-top: 4px; margin-bottom: 6px;">
{authors} &bull; <span style="color: {t['text_sub']};">{venue}</span> ({year}) &bull; {citations:,} citations
</div>
{pills_html}
</div>
""")
        st.markdown(card_html, unsafe_allow_html=True)

        col_rm, col_gap = st.columns([1.2, 6.0])
        with col_rm:
            if st.button("🗑 Remove", key=f"rm_lib_{pid}"):
                library = [p for p in library if clean_id(p.get("paper_id", "")) != pid]
                st.session_state["library"] = library
                st.toast("Removed from library")
                st.rerun()
