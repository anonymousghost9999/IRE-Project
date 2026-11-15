"""Page 2: Personalized Feed View with Live Session Steering."""

import streamlit as st

from demo.components.paper_card import render_paper_card
from demo.components.session_panel import render_session_snapshot
from demo.recommend_bridge import apply_user_action, generate_feed
from demo.state import reset_user_session
from paperfeed.rec.recommender import RecommenderResources


def render_feed_page(resources: RecommenderResources) -> None:
    # Header area
    author_name = st.session_state.get("author_name", "Researcher")
    user_mode = st.session_state.get("user_mode", "author")
    mode_badge = "OpenAlex Profile" if user_mode == "author" else "Zero-Pub Seed Mode"

    col_title, col_stat = st.columns([4, 2])
    with col_title:
        st.markdown(
            f"""
            <div style="margin-bottom: 12px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <h2 style="margin: 0; font-size: 1.6rem; font-weight: 700; color: #F8FAFC;">My Feed</h2>
                    <span style="background: rgba(59, 130, 246, 0.15); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.3); padding: 3px 8px; border-radius: 9999px; font-size: 0.72rem; font-weight: 600;">{mode_badge}</span>
                </div>
                <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 4px;">
                    Personalized discovery for <strong style="color: #E2E8F0;">{author_name}</strong>. Ranked with MMR diversity and verified grounded explanations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
            st.markdown(
                f"""
                <div style="margin-top: 14px; background-color: #1C1F2A; border: 1px solid #2E3348; border-radius: 8px; padding: 12px 14px; font-size: 0.78rem;">
                    <div style="font-weight: 600; color: #CBD5E1; margin-bottom: 6px;">🔍 Retrieval Context</div>
                    <div style="margin-bottom: 4px; color: #94A3B8;">
                        Candidate Pool: <strong style="color: #F1F5F9;">{meta.get('n_candidates', 100)}</strong> papers
                    </div>
                    <div style="margin-bottom: 6px; color: #94A3B8;">
                        Excluded Papers (|E<sub>u</sub>|): <strong style="color: #F1F5F9;">{meta.get('n_excluded', 0)}</strong>
                    </div>
                    <div style="font-size: 0.72rem; color: #64748B; word-break: break-all;">
                        <strong>Pseudo-Query:</strong> {meta.get('pseudo_query', '')[:120]}...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
