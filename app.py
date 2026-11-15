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
from demo.theme import apply_theme_css, clean_html, get_theme


def main() -> None:
    st.set_page_config(
        page_title="PaperFeed — Personalized Research Discovery",
        page_icon="🗞",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    apply_theme_css()

    # Load engine resources
    resources = load_app_resources()
    t = get_theme()

    # Sidebar Navigation
    with st.sidebar:
        sidebar_logo_html = clean_html(f"""
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 18px;">
<span style="font-size: 1.8rem;">🗞</span>
<div>
<div style="font-size: 1.25rem; font-weight: 700; color: {t['text_primary']}; line-height: 1.1;">PaperFeed</div>
<div style="font-size: 0.72rem; color: {t['text_muted']}; text-transform: uppercase; font-weight: 600;">Research Recommender</div>
</div>
</div>
""")
        st.markdown(sidebar_logo_html, unsafe_allow_html=True)

        author_name = st.session_state.get("author_name", "Researcher")
        user_mode = st.session_state.get("user_mode", "author")
        icon = "👤" if user_mode == "author" else "🏷️"

        identity_html = clean_html(f"""
<div style="background-color: {t['card_bg']}; border: 1px solid {t['card_border']}; border-radius: 6px; padding: 10px 12px; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
<div style="font-size: 0.7rem; color: {t['text_muted']}; text-transform: uppercase; font-weight: 600;">Active Identity</div>
<div style="font-size: 0.92rem; font-weight: 600; color: {t['link']}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
{icon} {author_name}
</div>
</div>
""")
        st.markdown(identity_html, unsafe_allow_html=True)

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

        st.markdown(f"<hr style='border: 1px solid {t['sidebar_border']}; margin: 16px 0;'>", unsafe_allow_html=True)

        # Theme toggle button
        cur_theme = st.session_state.get("theme", "dark")
        new_theme = "light" if cur_theme == "dark" else "dark"
        theme_btn_label = "☀️ Switch to Light Mode" if cur_theme == "dark" else "🌙 Switch to Dark Mode"
        if st.button(theme_btn_label, use_container_width=True, help="Toggle light / dark UI theme"):
            st.session_state["theme"] = new_theme
            st.rerun()

        if st.button("🗑 Reset Session", use_container_width=True, help="Clear in-session interaction history"):
            reset_user_session()
            st.toast("Session vector reset to 0.")
            st.rerun()

        footer_html = clean_html(f"""
<div style="margin-top: 24px; font-size: 0.72rem; color: {t['text_muted']}; line-height: 1.4;">
<div><strong>Corpus:</strong> 110,229 CS/AI papers</div>
<div><strong>Index:</strong> OpenAlex &le; 2024</div>
<div><strong>Architecture:</strong> SPECTER2 + BM25s + MMR</div>
</div>
""")
        st.markdown(footer_html, unsafe_allow_html=True)

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
