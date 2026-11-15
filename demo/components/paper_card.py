"""Paper card UI component with interactive action buttons and pilot ratings."""

from typing import Any, Callable, Dict, Optional
import streamlit as st

from demo.components.explanation import render_explanation_block
from demo.components.facet_pills import render_pills_container
from demo.theme import clean_html, get_theme


def render_paper_card(
    item: Dict[str, Any],
    on_action: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    show_actions: bool = True,
    show_pilot_ratings: bool = True,
    card_key_prefix: str = "card",
) -> None:
    """Render a clean, research-grade card for a paper."""
    t = get_theme()
    pid = item.get("paper_id", "")
    rank = item.get("rank", 1)
    title = item.get("title", "Untitled Paper")
    abstract = item.get("abstract", "")
    authors = item.get("authors", "Research Community")
    year = item.get("year", 2024)
    venue = item.get("venue", "OpenAlex")
    url = item.get("url", f"https://openalex.org/{pid}")
    citations = item.get("citations", 0)
    score = item.get("score", 0.0)
    facets = item.get("facets", [])

    is_skipped = pid in st.session_state.get("skipped_ids", set())
    is_saved = any(p.get("paper_id") == pid for p in st.session_state.get("library", []))
    session_obj = st.session_state.get("session")
    is_clicked = pid in session_obj.positives if (session_obj and hasattr(session_obj, "positives")) else False

    opacity = "0.55" if is_skipped else "1.0"
    border_color = t["accent"] if is_saved else ("#F87171" if is_skipped else t["card_border"])

    # Card header status badges
    status_badges = []
    if is_saved:
        status_badges.append(
            '<span style="background-color: rgba(251, 191, 36, 0.2); color: #FBBF24; padding: 2px 7px; border-radius: 4px; font-size: 0.72rem; font-weight: 600;">⭐ Saved</span>'
        )
    if is_skipped:
        status_badges.append(
            '<span style="background-color: rgba(248, 113, 113, 0.2); color: #F87171; padding: 2px 7px; border-radius: 4px; font-size: 0.72rem; font-weight: 600;">✕ Skipped</span>'
        )
    if is_clicked and not is_saved:
        status_badges.append(
            f'<span style="background-color: rgba(59, 130, 246, 0.2); color: {t["link"]}; padding: 2px 7px; border-radius: 4px; font-size: 0.72rem; font-weight: 600;">✓ Clicked</span>'
        )

    status_html = " ".join(status_badges)
    pills_html = render_pills_container(facets)

    # Explanation block
    expl_text = item.get("explanation_text")
    dominant_feat = item.get("dominant_feature")
    expl_evidence = item.get("explanation_evidence")
    failure_reason = item.get("explanation_failure")
    expl_html = render_explanation_block(expl_text, dominant_feat, expl_evidence, failure_reason)

    # Unindented card HTML to ensure Markdown treats it strictly as rendered markup, never as code
    card_html = clean_html(f"""
<div style="background-color: {t['card_bg']}; border: 1px solid {border_color}; border-radius: 8px; padding: 16px 18px; margin-bottom: 14px; opacity: {opacity}; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
<div style="display: flex; align-items: center; gap: 8px;">
<span style="background-color: {t['badge_bg']}; color: {t['accent']}; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; font-family: monospace;">#{rank}</span>
<span style="font-size: 0.75rem; color: {t['text_secondary']};">Relevance Score: <strong style="color: {t['text_primary']};">{score}</strong></span>
</div>
<div>{status_html}</div>
</div>
<div style="margin-bottom: 6px;">
<a href="{url}" target="_blank" style="font-size: 1.05rem; font-weight: 600; color: {t['link']}; text-decoration: none; line-height: 1.35;">
{title} ↗
</a>
</div>
<div style="font-size: 0.8rem; color: {t['text_secondary']}; margin-bottom: 8px; line-height: 1.3;">
<span>{authors}</span> &bull; 
<span style="color: {t['text_sub']};">{venue}</span> ({year}) &bull; 
<span>{citations:,} citations (&le; 2024)</span>
</div>
{pills_html}
{expl_html}
</div>
""")

    st.markdown(card_html, unsafe_allow_html=True)

    # Collapsible abstract snippet
    if abstract:
        with st.expander("📄 View Abstract", expanded=False):
            st.markdown(
                f'<div style="font-size: 0.85rem; color: {t["text_sub"]}; line-height: 1.5;">{abstract}</div>',
                unsafe_allow_html=True,
            )

    # Actions row
    if show_actions and not is_skipped:
        col_act1, col_act2, col_act3, col_space = st.columns([1.2, 1.1, 1.0, 3.5])
        with col_act1:
            if st.button("✓ Relevant", key=f"{card_key_prefix}_click_{pid}", help="Register click interest and update session EMA", use_container_width=True):
                if on_action:
                    on_action(pid, "click", item)
                st.rerun()

        with col_act2:
            save_label = "★ Saved" if is_saved else "⭐ Save"
            if st.button(save_label, key=f"{card_key_prefix}_save_{pid}", help="Save to My Library and update session", use_container_width=True):
                if on_action:
                    on_action(pid, "save", item)
                st.rerun()

        with col_act3:
            if st.button("✕ Skip", key=f"{card_key_prefix}_skip_{pid}", help="Dismiss paper (adds negative feedback penalty)", use_container_width=True):
                if on_action:
                    on_action(pid, "skip", item)
                st.rerun()

    # Optional pilot study rating box (§12.2)
    if show_pilot_ratings and not is_skipped:
        with st.expander("📝 Pilot Evaluation (Rate Relevance & Clarity)", expanded=False):
            col_r1, col_r2 = st.columns(2)
            cur_ratings = st.session_state.get("pilot_ratings", {}).get(pid, {})
            with col_r1:
                rel = st.slider(
                    "Recommendation Relevance (1=Irrelevant, 5=Highly Relevant)",
                    min_value=1,
                    max_value=5,
                    value=cur_ratings.get("relevance", 4),
                    key=f"pilot_rel_{pid}",
                )
            with col_r2:
                clarity = st.slider(
                    "Explanation Clarity & Grounding (1=Vague, 5=Crystal Clear)",
                    min_value=1,
                    max_value=5,
                    value=cur_ratings.get("clarity", 4),
                    key=f"pilot_clar_{pid}",
                )
            if st.button("Save Rating", key=f"save_rate_{pid}"):
                if "pilot_ratings" not in st.session_state:
                    st.session_state["pilot_ratings"] = {}
                st.session_state["pilot_ratings"][pid] = {
                    "paper_id": pid,
                    "title": title,
                    "relevance": rel,
                    "clarity": clarity,
                }
                st.toast(f"Rating recorded for #{rank}!")
