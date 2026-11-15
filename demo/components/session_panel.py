"""Right-hand session monitoring panel for the Feed view."""

from typing import Any, Callable, Dict, List, Optional
import streamlit as st

from demo.theme import clean_html, get_theme
from paperfeed.user.session import SessionState


def render_session_snapshot(
    session: SessionState,
    on_refresh: Optional[Callable[[], None]] = None,
    on_reset: Optional[Callable[[], None]] = None,
) -> None:
    """Render real-time session feedback and telemetry panel."""
    t = get_theme()

    header_html = clean_html(f"""
<div style="background-color: {t['card_bg']}; border: 1px solid {t['card_border']}; border-radius: 8px; padding: 14px 16px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
<span style="font-size: 0.95rem; font-weight: 600; color: {t['text_primary']};">⚡ Session Monitor</span>
<span style="font-size: 0.72rem; color: #10B981; background: rgba(16, 185, 129, 0.15); padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(16, 185, 129, 0.3);">Live State</span>
</div>
<div style="font-size: 0.78rem; color: {t['text_secondary']}; line-height: 1.35;">
Tracks in-session clicks & saves to adapt the retrieval query vector <code style="color: {t['link']}; background: {t['sub_box']}; padding: 1px 4px; border-radius: 3px;">q_t</code> in real time.
</div>
</div>
""")
    st.markdown(header_html, unsafe_allow_html=True)

    # Metric chips in 2x2 grid
    n_positives = len(session.positives)
    n_saved = len(session.consumed)
    n_skipped = len(session.negatives)
    n_gates = getattr(session, "n_gate_resets", 0)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Clicks / Saves", f"{n_positives}")
        st.metric("Saved in Library", f"{n_saved}")
    with col2:
        st.metric("Skipped Papers", f"{n_skipped}")
        st.metric("Topic Pivots", f"{n_gates}", help="Number of times session vector reset due to topic shift (γ_pivot > 0.35)")

    # Controls
    col_ref, col_res = st.columns(2)
    with col_ref:
        if st.button("🔄 Refresh Feed", use_container_width=True, help="Re-compute feed with updated session vector"):
            if on_refresh:
                on_refresh()
            st.rerun()

    with col_res:
        if st.button("🗑 Reset Session", use_container_width=True, help="Clear session vector and negative penalties"):
            if on_reset:
                on_reset()
            st.rerun()

    # Active Hyperparameters Card
    params_html = clean_html(f"""
<div style="margin-top: 14px; background-color: {t['sub_box']}; border: 1px solid {t['sub_border']}; border-radius: 6px; padding: 10px 12px; font-size: 0.75rem; color: {t['text_secondary']};">
<div style="font-weight: 600; color: {t['text_primary']}; margin-bottom: 6px;">Frozen System Parameters (Phase 9/11)</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
<span>Session EMA &alpha;<sub>sess</sub>:</span>
<strong style="color: {t['link']};">0.30</strong>
</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
<span>Persona Weight &rho;:</span>
<strong style="color: {t['link']};">0.30</strong>
</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
<span>Topic Pivot Gate:</span>
<strong style="color: #10B981;">Active (&gamma;=0.35)</strong>
</div>
<div style="display: flex; justify-content: space-between;">
<span>Negative Penalty &lambda;<sub>neg</sub>:</span>
<strong style="color: #F87171;">0.50</strong>
</div>
</div>
""")
    st.markdown(params_html, unsafe_allow_html=True)

    # Recent Positive Interactions
    if session.positives:
        st.markdown(
            f"<div style='font-size: 0.82rem; font-weight: 600; color: {t['text_primary']}; margin-top: 16px; margin-bottom: 6px;'>Active Session Clicks (Recent)</div>",
            unsafe_allow_html=True,
        )
        recent_pids = session.recent_positives(n=4)
        for pid in recent_pids:
            item_html = clean_html(f"""
<div style="background-color: {t['card_bg']}; border-left: 3px solid {t['accent']}; border: 1px solid {t['card_border']}; padding: 6px 10px; margin-bottom: 4px; border-radius: 0 4px 4px 0; font-size: 0.75rem; color: {t['text_sub']};">
<span style="font-family: monospace; color: {t['link']};">{pid}</span>
</div>
""")
            st.markdown(item_html, unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div style='font-size: 0.78rem; color: {t['text_muted']}; font-style: italic; margin-top: 14px;'>Click '✓ Relevant' on any paper card to begin steering your session.</div>",
            unsafe_allow_html=True,
        )
