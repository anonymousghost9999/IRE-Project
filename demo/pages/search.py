"""Page 3: Hybrid Corpus Search and Topic Steering."""

import streamlit as st

from demo.components.facet_pills import render_pills_container
from demo.recommend_bridge import apply_user_action, search_papers
from demo.theme import clean_html, get_theme
from paperfeed.rec.recommender import RecommenderResources


def render_search_page(resources: RecommenderResources) -> None:
    t = get_theme()

    header_html = clean_html(f"""
<div style="margin-bottom: 16px;">
<h2 style="margin: 0; font-size: 1.6rem; font-weight: 700; color: {t['text_primary']};">🔍 Search Papers</h2>
<div style="font-size: 0.85rem; color: {t['text_secondary']}; margin-top: 4px;">
Direct hybrid dense (SPECTER2 <code>adhoc_query</code>) + BM25s lexical search across the &le; 2024 corpus.
</div>
</div>
""")
    st.markdown(header_html, unsafe_allow_html=True)

    # Search Bar
    default_q = st.session_state.get("search_query", "graph neural networks for molecular property prediction")
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query_text = st.text_input("Enter search query or keywords", value=default_q, label_visibility="collapsed")
    with col_btn:
        search_clicked = st.button("Search", type="primary", use_container_width=True)

    # Quick Suggestion Chips
    suggestions = [
        "dense passage retrieval open domain qa",
        "diffusion models for image synthesis",
        "reinforcement learning from human feedback",
        "code large language models",
        "graph neural networks link prediction",
    ]
    st.markdown(
        f"<div style='font-size: 0.75rem; color: {t['text_muted']}; margin-top: -6px; margin-bottom: 12px;'>Quick queries:</div>",
        unsafe_allow_html=True,
    )
    cols_sug = st.columns(len(suggestions))
    for idx, sug in enumerate(suggestions):
        with cols_sug[idx]:
            if st.button(sug.split()[0].title() + " " + sug.split()[1], key=f"sug_{idx}", use_container_width=True):
                query_text = sug
                st.session_state["search_query"] = sug
                st.rerun()

    # Perform Search
    if query_text.strip():
        with st.spinner(f"Searching corpus for '{query_text.strip()}'..."):
            results = search_papers(resources, query_text.strip(), limit=15)

        st.markdown(
            f"<div style='font-size: 0.85rem; color: {t['text_secondary']}; margin-bottom: 14px;'>Found <strong>{len(results)}</strong> top hybrid matches:</div>",
            unsafe_allow_html=True,
        )

        for item in results:
            pid = item["paper_id"]
            rank = item["rank"]
            title = item["title"]
            authors = item["authors"]
            venue = item["venue"]
            year = item["year"]
            url = item["url"]
            citations = item["citations"]
            facets = item.get("facets", [])
            dense_r = item.get("dense_rank")
            lex_r = item.get("lexical_rank")

            pills_html = render_pills_container(facets)

            item_html = clean_html(f"""
<div style="background-color: {t['card_bg']}; border: 1px solid {t['card_border']}; border-radius: 8px; padding: 14px 16px; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 4px;">
<a href="{url}" target="_blank" style="font-size: 1.0rem; font-weight: 600; color: {t['link']}; text-decoration: none;">
{rank}. {title} ↗
</a>
<div style="display: flex; gap: 4px; font-size: 0.7rem; font-family: monospace;">
<span style="background: rgba(59, 130, 246, 0.15); color: {t['link']}; padding: 2px 6px; border-radius: 4px;">Dense #{dense_r or '-'}</span>
<span style="background: rgba(16, 185, 129, 0.15); color: #10B981; padding: 2px 6px; border-radius: 4px;">BM25 #{lex_r or '-'}</span>
</div>
</div>
<div style="font-size: 0.78rem; color: {t['text_secondary']}; margin-bottom: 6px;">
{authors} &bull; <span style="color: {t['text_sub']};">{venue}</span> ({year}) &bull; {citations:,} citations
</div>
{pills_html}
</div>
""")
            st.markdown(item_html, unsafe_allow_html=True)

            # Action button to add into session
            col_act1, col_act2, col_sp = st.columns([1.8, 1.2, 5.0])
            with col_act1:
                if st.button("➕ Add to Feed Context", key=f"search_steer_{pid}", help="Add this paper to your session vector to pivot your feed toward this topic"):
                    apply_user_action(resources, pid, "click", item)
                    st.toast(f"Added '{title[:35]}...' to session! Head to My Feed to see updated recommendations.")
            with col_act2:
                if st.button("⭐ Save", key=f"search_save_{pid}"):
                    apply_user_action(resources, pid, "save", item)
                    st.toast("Saved to My Library!")
