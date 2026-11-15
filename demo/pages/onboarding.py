"""Page 1: Onboarding and Persona Initialization."""

from typing import List
import streamlit as st

from demo.state import get_featured_authors, get_popular_seed_topics, reset_user_session
from paperfeed.data.cohort import clean_id


def render_onboarding_page() -> None:
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-size: 2.2rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;">
                🗞 PaperFeed
            </h1>
            <p style="font-size: 1.05rem; color: #94A3B8; max-width: 620px; margin: 0 auto;">
                A personalized research-paper discovery engine powered by hybrid dense-lexical retrieval, 
                dynamic session adaptation, citation-graph signals, and faithful grounded explanations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_center = st.columns([1, 4, 1])[1]

    with col_center:
        st.markdown(
            """
            <div style="background-color: #1C1F2A; border: 1px solid #2E3348; border-radius: 10px; padding: 24px; margin-bottom: 20px;">
                <h3 style="margin-top: 0; color: #F1F5F9; font-size: 1.2rem; font-weight: 600;">
                    Choose Your Researcher Profile
                </h3>
                <p style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 16px;">
                    Select an author identity from the OpenAlex corpus or enter custom seed topics for cold-start exploration.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        mode = st.radio(
            "Profile Construction Mode",
            options=["OpenAlex Author Profile (Track 1 / Full Personalization)", "Seed Topics (Cold-Start / Zero-Publication Mode)"],
            index=0 if st.session_state.get("user_mode") == "author" else 1,
            label_visibility="collapsed",
        )

        featured_authors = get_featured_authors()

        if "OpenAlex Author" in mode:
            st.session_state["user_mode"] = "author"

            st.markdown("##### 👤 Featured Indexed Researchers")
            author_labels = [f"{a['name']} ({a['n_papers']} papers in corpus) — {', '.join(a['topics'])}" for a in featured_authors]
            selected_idx = st.selectbox(
                "Choose a researcher profile to test",
                options=range(len(featured_authors)),
                format_func=lambda i: author_labels[i],
                index=0,
            )

            chosen_author = featured_authors[selected_idx]

            # Optional manual input
            with st.expander("Or enter custom OpenAlex Author ID / URL", expanded=False):
                manual_id = st.text_input("OpenAlex Author ID (e.g. A5060364305 or full URL)", value="")
                if manual_id.strip():
                    clean_aid = clean_id(manual_id.strip().split("/")[-1])
                    if clean_aid:
                        chosen_author = {
                            "author_id": clean_aid,
                            "name": f"Author {clean_aid}",
                            "authored_ids": [],
                            "n_papers": 0,
                        }

            # Preview card
            st.markdown(
                f"""
                <div style="background-color: #161822; border: 1px solid #262B3A; border-radius: 8px; padding: 14px; margin: 16px 0;">
                    <div style="font-size: 0.8rem; color: #64748B; text-transform: uppercase; font-weight: 600;">Selected Profile</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #60A5FA; margin: 4px 0;">{chosen_author['name']}</div>
                    <div style="font-size: 0.82rem; color: #94A3B8;">
                        OpenAlex ID: <code style="color: #F1F5F9;">{chosen_author['author_id']}</code> &bull; 
                        <strong>{chosen_author['n_papers']}</strong> authored papers indexed
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("🚀 Build My Feed", type="primary", use_container_width=True):
                st.session_state["user_id"] = chosen_author["author_id"]
                st.session_state["author_name"] = chosen_author["name"]
                st.session_state["authored_ids"] = chosen_author["authored_ids"]
                st.session_state["profile"] = None  # Force rebuild
                reset_user_session()
                st.session_state["active_page"] = "Feed"
                st.toast("Profile loaded! Generating recommendations...")
                st.rerun()

        else:
            st.session_state["user_mode"] = "seed"
            st.markdown("##### 🏷️ Select Research Focus Areas")
            popular_topics = get_popular_seed_topics()

            selected_seeds = st.multiselect(
                "Select 1 to 5 topics to anchor your persona",
                options=popular_topics,
                default=st.session_state.get("seed_topics", ["Natural Language Processing", "Large Language Models"]),
                max_selections=5,
            )

            custom_topic = st.text_input("Or add a custom research topic (press Enter to include)")
            if custom_topic and custom_topic.strip() and custom_topic.strip() not in selected_seeds:
                selected_seeds.append(custom_topic.strip())

            st.markdown(
                f"""
                <div style="background-color: #161822; border: 1px solid #262B3A; border-radius: 8px; padding: 14px; margin: 16px 0;">
                    <div style="font-size: 0.8rem; color: #64748B; text-transform: uppercase; font-weight: 600;">Seed Anchors ({len(selected_seeds)})</div>
                    <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 6px;">
                        {", ".join(selected_seeds) if selected_seeds else "None selected yet"}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            btn_disabled = len(selected_seeds) == 0
            if st.button("🚀 Build Zero-Pub Feed", type="primary", disabled=btn_disabled, use_container_width=True):
                st.session_state["seed_topics"] = selected_seeds
                st.session_state["user_id"] = f"seed_{abs(hash(tuple(selected_seeds))) % 100000}"
                st.session_state["author_name"] = f"Researcher ({selected_seeds[0]})"
                st.session_state["profile"] = None  # Force rebuild
                reset_user_session()
                st.session_state["active_page"] = "Feed"
                st.toast("Seed persona synthesized! Generating recommendations...")
                st.rerun()
