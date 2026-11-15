"""Page 2: Personalized Feed View with Live Session Steering."""

import streamlit as st

from demo.components.paper_card import render_paper_card
from demo.components.session_panel import render_session_snapshot
from demo.recommend_bridge import apply_user_action, generate_feed
from demo.state import reset_user_session
from demo.theme import clean_html, get_theme
from paperfeed.rec.recommender import RecommenderResources


def render_feed_page(resources: RecommenderResources) -> None:
    t = get_theme()
    author_name = st.session_state.get("author_name", "Researcher")
    user_mode = st.session_state.get("user_mode", "author")
    mode_badge = "OpenAlex Profile" if user_mode == "author" else "Zero-Pub Seed Mode"

    header_html = clean_html(f"""
<div style="margin-bottom: 14px;">
<div style="display: flex; align-items: center; gap: 10px;">
<h2 style="margin: 0; font-size: 1.6rem; font-weight: 700; color: {t['text_primary']};">My Feed</h2>
<span style="background: rgba(59, 130, 246, 0.15); color: {t['link']}; border: 1px solid rgba(59, 130, 246, 0.3); padding: 3px 8px; border-radius: 9999px; font-size: 0.72rem; font-weight: 600;">{mode_badge}</span>
</div>
<div style="font-size: 0.85rem; color: {t['text_secondary']}; margin-top: 4px;">
Personalized discovery for <strong style="color: {t['text_primary']};">{author_name}</strong>. Ranked with MMR diversity and verified grounded explanations.
</div>
</div>
""")
    st.markdown(header_html, unsafe_allow_html=True)

    # 68% feed column | 32% session snapshot column
    col_feed, col_side = st.columns([2.1, 1.0], gap="large")

    def handle_card_action(paper_id: str, action: str, paper_meta: dict) -> None:
        apply_user_action(resources, paper_id, action, paper_meta)
        if action == "save":
            st.toast("Saved paper to My Library!")
        elif action == "click":
            st.toast("Interest updated! Session adapted.")
        elif action == "skip":
            st.toast("Paper dismissed.")

    with col_feed:
        with st.spinner("Retrieving and ranking personalized feed..."):
            items = generate_feed(resources)

        if not items:
            st.info("No candidate recommendations found for this profile. Try resetting your session or choosing another persona.")
        else:
            for item in items:
                render_paper_card(
                    item=item,
                    on_action=handle_card_action,
                    show_actions=True,
                    show_pilot_ratings=True,
                    card_key_prefix="feed",
                )

    with col_side:
        session = st.session_state.get("session")
        render_session_snapshot(
            session=session,
            on_refresh=lambda: generate_feed(resources, force=True),
            on_reset=reset_user_session,
        )

        # Feed Telemetry Card
        meta = st.session_state.get("feed_meta", {})
        if meta:
            telemetry_html = clean_html(f"""
<div style="margin-top: 14px; background-color: {t['card_bg']}; border: 1px solid {t['card_border']}; border-radius: 8px; padding: 12px 14px; font-size: 0.78rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
<div style="font-weight: 600; color: {t['text_primary']}; margin-bottom: 6px;">🔍 Retrieval Context</div>
<div style="margin-bottom: 4px; color: {t['text_secondary']};">
Candidate Pool: <strong style="color: {t['text_primary']};">{meta.get('n_candidates', 100)}</strong> papers
</div>
<div style="margin-bottom: 6px; color: {t['text_secondary']};">
Excluded Papers (|E<sub>u</sub>|): <strong style="color: {t['text_primary']};">{meta.get('n_excluded', 0)}</strong>
</div>
<div style="font-size: 0.72rem; color: {t['text_muted']}; word-break: break-all;">
<strong>Pseudo-Query:</strong> {meta.get('pseudo_query', '')[:120]}...
</div>
</div>
""")
            st.markdown(telemetry_html, unsafe_allow_html=True)
