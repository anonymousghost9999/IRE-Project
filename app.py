"""PaperFeed Streamlit Application Entrypoint (Phase 12)."""

import sys
from pathlib import Path

# Ensure repo root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import streamlit as st

from demo.pages.about import render_about_page
from demo.pages.feed import render_feed_page
from demo.pages.library import render_library_page
from demo.pages.onboarding import render_onboarding_page
from demo.pages.profile import render_profile_page
from demo.pages.search import render_search_page
from demo.state import init_session_state, load_app_resources, reset_user_session


def set_custom_css() -> None:
    st.markdown(
        """
        <style>
            /* Main background and fonts */
            .main {
                background-color: #0F1117;
            }
            body {
                color: #E2E8F0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }
            /* Sidebar styling */
            section[data-testid="stSidebar"] {
                background-color: #161822;
                border-right: 1px solid #262B3A;
            }
            /* Button styling */
            .stButton > button {
                border-radius: 6px;
                font-weight: 500;
                transition: all 0.15s ease-in-out;
            }
            /* Metric text colors */
            div[data-testid="stMetricValue"] {
                color: #60A5FA;
                font-size: 1.4rem;
            }
            /* Expander headers */
            .streamlit-expanderHeader {
                font-size: 0.85rem;
                color: #CBD5E1;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="PaperFeed — Personalized Research Discovery",
        page_icon="🗞",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    set_custom_css()
    init_session_state()

    # Load engine resources
    resources = load_app_resources()

    # Sidebar Navigation
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 18px;">
                <span style="font-size: 1.8rem;">🗞</span>
                <div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #F8FAFC; line-height: 1.1;">PaperFeed</div>
                    <div style="font-size: 0.72rem; color: #64748B; text-transform: uppercase; font-weight: 600;">Research Recommender</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        author_name = st.session_state.get("author_name", "Researcher")
        user_mode = st.session_state.get("user_mode", "author")
        icon = "👤" if user_mode == "author" else "🏷️"

        st.markdown(
            f"""
            <div style="background-color: #1C1F2A; border: 1px solid #2E3348; border-radius: 6px; padding: 10px 12px; margin-bottom: 16px;">
                <div style="font-size: 0.7rem; color: #64748B; text-transform: uppercase; font-weight: 600;">Active Identity</div>
                <div style="font-size: 0.92rem; font-weight: 600; color: #60A5FA; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                    {icon} {author_name}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        library_count = len(st.session_state.get("library", []))
        pages = {
            "Feed": "📄  My Feed",
            "Search": "🔍  Search Papers",
            "Library": f"🔖  My Library ({library_count})",
            "Profile": "🧠  Interest Profile",
            "Onboarding": "🏠  Change Identity",
            "About": "ℹ️  About & Pilot Study",
        }

        current_active = st.session_state.get("active_page", "Feed")
        selected_key = st.radio(
            "Navigation",
            options=list(pages.keys()),
            format_func=lambda k: pages[k],
            index=list(pages.keys()).index(current_active) if current_active in pages else 0,
            label_visibility="collapsed",
        )
        st.session_state["active_page"] = selected_key

        st.markdown("<hr style='border: 1px solid #262B3A; margin: 16px 0;'>", unsafe_allow_html=True)

        if st.button("🗑 Reset Session", use_container_width=True, help="Clear in-session interaction history"):
            reset_user_session()
            st.toast("Session vector reset to 0.")
            st.rerun()

        st.markdown(
            """
            <div style="margin-top: 24px; font-size: 0.72rem; color: #64748B; line-height: 1.4;">
                <div><strong>Corpus:</strong> 110,229 CS/AI papers</div>
                <div><strong>Index:</strong> OpenAlex &le; 2024</div>
                <div><strong>Architecture:</strong> SPECTER2 + BM25s + MMR</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Route Page
    page = st.session_state.get("active_page", "Feed")
    if page == "Feed":
        render_feed_page(resources)
    elif page == "Search":
        render_search_page(resources)
    elif page == "Library":
        render_library_page()
    elif page == "Profile":
        render_profile_page(resources)
    elif page == "Onboarding":
        render_onboarding_page()
    elif page == "About":
        render_about_page()


if __name__ == "__main__":
    main()
