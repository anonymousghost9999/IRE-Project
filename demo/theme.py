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
        "code_text": "#60A5FA",
        "code_border": "#334155",
        "expl_bg": "rgba(30, 41, 59, 0.7)",
        "expl_border": "rgba(148, 163, 184, 0.2)",
        "button_bg": "#1C1F2A",
        "button_text": "#F8FAFC",
        "button_border": "#2E3348",
        "button_hover_bg": "#262B3A",
        "chart_grid": "#2E3348",
        "chart_label": "#94A3B8",
        "chart_title": "#CBD5E1",
    },
    "light": {
        "name": "light",
        "bg": "#F8FAFC",
        "card_bg": "#FFFFFF",
        "card_border": "#CBD5E1",
        "sidebar_bg": "#F1F5F9",
        "sidebar_border": "#CBD5E1",
        "text_primary": "#0F172A",
        "text_secondary": "#475569",
        "text_sub": "#1E293B",
        "text_muted": "#64748B",
        "link": "#2563EB",
        "badge_bg": "#E2E8F0",
        "accent": "#2563EB",
        "sub_box": "#FFFFFF",
        "sub_border": "#CBD5E1",
        "code_bg": "#E2E8F0",
        "code_text": "#0F172A",
        "code_border": "#94A3B8",
        "expl_bg": "#F8FAFC",
        "expl_border": "#CBD5E1",
        "button_bg": "#FFFFFF",
        "button_text": "#0F172A",
        "button_border": "#CBD5E1",
        "button_hover_bg": "#F1F5F9",
        "chart_grid": "#E2E8F0",
        "chart_label": "#475569",
        "chart_title": "#0F172A",
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
    is_light = t["name"] == "light"

    css = f"""
    <style>
        /* Base page background and default font color */
        .stApp, .main {{
            background-color: {t["bg"]} !important;
            color: {t["text_primary"]} !important;
        }}
        header[data-testid="stHeader"] {{
            background-color: {t["bg"]} !important;
        }}
        body, p, span, label, div {{
            color: {t["text_primary"]};
        }}

        /* Sidebar styling */
        section[data-testid="stSidebar"] {{
            background-color: {t["sidebar_bg"]} !important;
            border-right: 1px solid {t["sidebar_border"]} !important;
        }}
        section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label {{
            color: {t["text_primary"]} !important;
        }}

        /* Secondary and standard buttons */
        button[data-testid="stBaseButton-secondary"],
        .stButton > button {{
            background-color: {t["button_bg"]} !important;
            color: {t["button_text"]} !important;
            border: 1px solid {t["button_border"]} !important;
            border-radius: 6px !important;
            font-weight: 500 !important;
            transition: all 0.15s ease-in-out !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
        }}
        button[data-testid="stBaseButton-secondary"] p,
        button[data-testid="stBaseButton-secondary"] span,
        .stButton > button p,
        .stButton > button span,
        .stButton > button div {{
            color: {t["button_text"]} !important;
            font-weight: 500 !important;
        }}
        button[data-testid="stBaseButton-secondary"]:hover,
        .stButton > button:hover {{
            background-color: {t["button_hover_bg"]} !important;
            border-color: {t["accent"]} !important;
            color: {t["accent"]} !important;
        }}
        button[data-testid="stBaseButton-secondary"]:hover p,
        button[data-testid="stBaseButton-secondary"]:hover span,
        .stButton > button:hover p,
        .stButton > button:hover span {{
            color: {t["accent"]} !important;
        }}

        /* Primary action buttons */
        button[data-testid="stBaseButton-primary"],
        button[kind="primary"] {{
            background-color: {t["accent"]} !important;
            color: #FFFFFF !important;
            border: 1px solid {t["accent"]} !important;
        }}
        button[data-testid="stBaseButton-primary"] p,
        button[data-testid="stBaseButton-primary"] span,
        button[kind="primary"] p,
        button[kind="primary"] span {{
            color: #FFFFFF !important;
        }}

        /* Inline code chips */
        code {{
            background-color: {t["code_bg"]} !important;
            color: {t["code_text"]} !important;
            border: 1px solid {t["code_border"]} !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
            font-size: 0.85em !important;
            font-weight: 600 !important;
        }}

        /* Radio options in sidebar */
        div[data-testid="stRadio"] label p,
        div[data-testid="stRadio"] label span {{
            color: {t["text_primary"]} !important;
            font-size: 0.9rem !important;
        }}

        /* Tabs styling */
        button[data-baseweb="tab"] {{
            color: {t["text_secondary"]} !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {t["accent"]} !important;
            border-bottom-color: {t["accent"]} !important;
        }}
        button[data-baseweb="tab"] p,
        button[data-baseweb="tab"] span {{
            color: inherit !important;
        }}

        /* Metrics */
        div[data-testid="stMetricValue"] {{
            color: {t["link"]} !important;
            font-size: 1.4rem !important;
            font-weight: 700 !important;
        }}
        div[data-testid="stMetricLabel"] p,
        div[data-testid="stMetricLabel"] span {{
            color: {t["text_secondary"]} !important;
        }}

        /* Expanders */
        .streamlit-expanderHeader {{
            background-color: {t["card_bg"]} !important;
            color: {t["text_sub"]} !important;
            border-radius: 6px !important;
            border: 1px solid {t["card_border"]} !important;
        }}
        div[data-testid="stExpander"] {{
            background-color: transparent !important;
            border: none !important;
        }}

        /* Inputs & Textareas */
        input, select, textarea {{
            background-color: {t["card_bg"]} !important;
            color: {t["text_primary"]} !important;
            border: 1px solid {t["card_border"]} !important;
        }}
    </style>
    """
    st.markdown(clean_html(css), unsafe_allow_html=True)
