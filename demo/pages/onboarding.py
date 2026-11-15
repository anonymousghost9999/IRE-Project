"""Page 1: Onboarding and Persona Initialization."""

from typing import List
import streamlit as st

from demo.state import ROOT_DIR, get_featured_authors, get_popular_seed_topics, reset_user_session
from demo.theme import clean_html, get_theme
from paperfeed.data.cohort import clean_id
import json


def render_onboarding_page() -> None:
    t = get_theme()

    banner_html = clean_html(f"""
<div style="text-align: center; margin-bottom: 2rem;">
<h1 style="font-size: 2.2rem; font-weight: 700; color: {t['text_primary']}; margin-bottom: 0.5rem;">
🗞 PaperFeed
</h1>
<p style="font-size: 1.05rem; color: {t['text_secondary']}; max-width: 620px; margin: 0 auto;">
A personalized research-paper discovery engine powered by hybrid dense-lexical retrieval, 
dynamic session adaptation, citation-graph signals, and faithful grounded explanations.
</p>
</div>
""")
    st.markdown(banner_html, unsafe_allow_html=True)

    col_center = st.columns([1, 4, 1])[1]

    with col_center:
        card_intro = clean_html(f"""
<div style="background-color: {t['card_bg']}; border: 1px solid {t['card_border']}; border-radius: 10px; padding: 24px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
<h3 style="margin-top: 0; color: {t['text_primary']}; font-size: 1.2rem; font-weight: 600;">
Choose Your Researcher Profile
</h3>
<p style="font-size: 0.85rem; color: {t['text_secondary']}; margin-bottom: 0;">
Select an author identity from the OpenAlex corpus or enter custom seed topics for cold-start exploration.
</p>
</div>
""")
        st.markdown(card_intro, unsafe_allow_html=True)

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
                        found_author = None
                        candidates_file = ROOT_DIR / "data" / "candidate_authors.jsonl"
                        if candidates_file.exists():
                            with open(candidates_file, "r", encoding="utf-8") as f:
                                for line in f:
                                    entry = json.loads(line)
                                    if clean_id(entry.get("author_id")) == clean_aid:
                                        found_author = {
                                            "author_id": clean_aid,
                                            "name": entry.get("author_name", f"Author {clean_aid}"),
                                            "authored_ids": entry.get("authored_paper_ids_T", []),
                                            "n_papers": len(entry.get("authored_paper_ids_T", [])),
                                        }
                                        break
                        if not found_author:
                            found_author = {
                                "author_id": clean_aid,
                                "name": f"Author {clean_aid}",
                                "authored_ids": [],
                                "n_papers": 0,
                            }
                        chosen_author = found_author

            # Preview card
            preview_html = clean_html(f"""
<div style="background-color: {t['sub_box']}; border: 1px solid {t['sub_border']}; border-radius: 8px; padding: 14px; margin: 16px 0;">
<div style="font-size: 0.8rem; color: {t['text_muted']}; text-transform: uppercase; font-weight: 600;">Selected Profile</div>
<div style="font-size: 1.1rem; font-weight: 600; color: {t['link']}; margin: 4px 0;">{chosen_author['name']}</div>
<div style="font-size: 0.82rem; color: {t['text_secondary']};">
OpenAlex ID: <code style="color: {t['text_primary']}; background: {t['code_bg']}; padding: 1px 4px; border-radius: 3px;">{chosen_author['author_id']}</code> &bull; 
<strong>{chosen_author['n_papers']}</strong> authored papers indexed
</div>
</div>
""")
            st.markdown(preview_html, unsafe_allow_html=True)

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

            seeds_html = clean_html(f"""
<div style="background-color: {t['sub_box']}; border: 1px solid {t['sub_border']}; border-radius: 8px; padding: 14px; margin: 16px 0;">
<div style="font-size: 0.8rem; color: {t['text_muted']}; text-transform: uppercase; font-weight: 600;">Seed Anchors ({len(selected_seeds)})</div>
<div style="font-size: 0.85rem; color: {t['text_sub']}; margin-top: 6px;">
{", ".join(selected_seeds) if selected_seeds else "None selected yet"}
</div>
</div>
""")
            st.markdown(seeds_html, unsafe_allow_html=True)

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
