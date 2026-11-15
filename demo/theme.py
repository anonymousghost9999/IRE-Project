"""Theme configuration and styling for PaperFeed (Dark / Light mode)."""

import textwrap
from typing import Any, Dict
import streamlit as st

THEMES = {
    "dark": {
        "name": "dark",
        "bg": "#0F1117",
        "card_bg": "#1C1F2A",
        "card_border": "#2E3348",
        "sidebar_bg": "#161822",
        "sidebar_border": "#262B3A",
        "text_primary": "#F8FAFC",
        "text_secondary": "#94A3B8",
        "text_sub": "#CBD5E1",
        "text_muted": "#64748B",
        "link": "#60A5FA",
        "badge_bg": "#262B3A",
        "accent": "#4A9EFF",
        "sub_box": "#161822",
        "sub_border": "#262B3A",
        "code_bg": "#1E293B",
        "expl_bg": "rgba(30, 41, 59, 0.7)",
        "expl_border": "rgba(148, 163, 184, 0.2)",
    },
    "light": {
        "name": "light",
        "bg": "#F8FAFC",
        "card_bg": "#FFFFFF",
        "card_border": "#E2E8F0",
        "sidebar_bg": "#F1F5F9",
        "sidebar_border": "#CBD5E1",
        "text_primary": "#0F172A",
        "text_secondary": "#475569",
        "text_sub": "#334155",
        "text_muted": "#64748B",
        "link": "#2563EB",
        "badge_bg": "#E2E8F0",
        "accent": "#2563EB",
        "sub_box": "#F1F5F9",
        "sub_border": "#E2E8F0",
        "code_bg": "#F1F5F9",
        "expl_bg": "#F8FAFC",
        "expl_border": "#CBD5E1",
    },
}


def get_theme() -> Dict[str, str]:
    """Retrieve active theme dictionary."""
    theme_name = st.session_state.get("theme", "dark")
    return THEMES.get(theme_name, THEMES["dark"])


def clean_html(html_str: str) -> str:
    """Safely unindent HTML string so Markdown never converts it to a <pre><code> block."""
    return textwrap.dedent(html_str).strip()


def apply_theme_css() -> None:
    """Inject dynamic global CSS for the active theme."""
    t = get_theme()
    css = f"""
    <style>
        .stApp, .main {{
            background-color: {t["bg"]} !important;
            color: {t["text_primary"]} !important;
        }}
        body, p, span, label, div {{
            color: {t["text_primary"]};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {t["sidebar_bg"]} !important;
            border-right: 1px solid {t["sidebar_border"]} !important;
        }}
        section[data-testid="stSidebar"] * {{
            color: {t["text_primary"]} !important;
        }}
        .stButton > button {{
            border-radius: 6px;
            font-weight: 500;
            background-color: {t["card_bg"]} !important;
            color: {t["text_primary"]} !important;
            border: 1px solid {t["card_border"]} !important;
            transition: all 0.15s ease-in-out;
        }}
        .stButton > button:hover {{
            border-color: {t["accent"]} !important;
            color: {t["accent"]} !important;
        }}
        div[data-testid="stMetricValue"] {{
            color: {t["link"]} !important;
            font-size: 1.4rem;
        }}
        div[data-testid="stMetricLabel"] {{
            color: {t["text_secondary"]} !important;
        }}
        .streamlit-expanderHeader {{
            background-color: {t["card_bg"]} !important;
            color: {t["text_sub"]} !important;
            border-radius: 6px;
            border: 1px solid {t["card_border"]};
        }}
        div[data-testid="stExpander"] {{
            background-color: transparent !important;
            border: none !important;
        }}
        input, select, textarea {{
            background-color: {t["card_bg"]} !important;
            color: {t["text_primary"]} !important;
            border: 1px solid {t["card_border"]} !important;
        }}
    </style>
    """
    st.markdown(clean_html(css), unsafe_allow_html=True)
