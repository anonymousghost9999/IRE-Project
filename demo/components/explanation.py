"""Explanation rendering component with grounding transparency."""

from typing import Any, Dict, Optional
from demo.theme import clean_html, get_theme


FEATURE_ICONS = {
    "facet": "💡",
    "new_method": "🔬",
    "citation": "🔗",
    "session": "⚡",
    "persona": "👤",
}

FEATURE_LABELS = {
    "facet": "Facet Match",
    "new_method": "New Method",
    "citation": "Citation Graph",
    "session": "Session Focus",
    "persona": "Profile Match",
}


def render_explanation_badge(dominant_feature: Optional[str]) -> str:
    """Return small chip showing which feature drove the ranking."""
    feat = dominant_feature or "persona"
    icon = FEATURE_ICONS.get(feat, "📌")
    label = FEATURE_LABELS.get(feat, "Ranking Feature")
    return (
        f'<span style="'
        f'display: inline-flex; align-items: center; gap: 4px;'
        f'padding: 2px 7px; border-radius: 4px; font-size: 0.72rem;'
        f'color: #0284C7; background-color: rgba(56, 189, 248, 0.12);'
        f'border: 1px solid rgba(56, 189, 248, 0.3); font-weight: 500;">'
        f'{icon} {label}'
        f'</span>'
    )


def render_explanation_block(
    text: Optional[str],
    dominant_feature: Optional[str],
    evidence: Optional[Dict[str, Any]] = None,
    failure_reason: Optional[str] = None,
) -> str:
    """Render full explanation callout box."""
    t = get_theme()
    if not text:
        if failure_reason:
            return (
                f'<div style="margin: 6px 0; padding: 6px 10px; border-radius: 6px;'
                f'background-color: {t["card_bg"]}; border-left: 3px solid #64748B;'
                f'border: 1px solid {t["card_border"]};'
                f'font-size: 0.8rem; color: {t["text_muted"]};">'
                f'<span style="font-style: italic;">No verified grounded explanation available ({failure_reason}).</span>'
                f'</div>'
            )
        return ""

    feat = dominant_feature or "persona"
    icon = FEATURE_ICONS.get(feat, "💡")
    badge = render_explanation_badge(dominant_feature)

    # Clean markdown formatting in text: replace **bold** with <strong>
    formatted_text = text.replace("**", "<strong>", 1)
    while "**" in formatted_text:
        formatted_text = formatted_text.replace("**", "</strong>", 1)

    evidence_html = ""
    if evidence and evidence.get("evidence_sentence"):
        sent = evidence["evidence_sentence"].replace("<", "&lt;").replace(">", "&gt;")
        span = evidence.get("span", "")
        if span and span in sent:
            sent = sent.replace(
                span,
                f'<span style="background-color: rgba(245, 158, 11, 0.25); color: #D97706; padding: 0 3px; border-radius: 3px; font-weight: 600;">{span}</span>',
            )
        evidence_html = (
            f'<div style="margin-top: 6px; padding: 6px 8px; border-radius: 4px;'
            f'background-color: {t["sub_box"]}; border-left: 2px solid #F59E0B;'
            f'font-size: 0.75rem; color: {t["text_sub"]}; font-style: italic;">'
            f'"{sent}"'
            f'</div>'
        )

    return (
        f'<div style="'
        f'margin: 8px 0 10px 0; padding: 8px 12px; border-radius: 6px;'
        f'background-color: {t["expl_bg"]};'
        f'border: 1px solid {t["expl_border"]};'
        f'border-left: 3px solid #38BDF8;">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">'
        f'<div style="font-size: 0.82rem; font-weight: 600; color: {t["text_primary"]};">{icon} Why this paper?</div>'
        f'{badge}'
        f'</div>'
        f'<div style="font-size: 0.82rem; color: {t["text_sub"]}; line-height: 1.4;">{formatted_text}</div>'
        f'{evidence_html}'
        f'</div>'
    )
