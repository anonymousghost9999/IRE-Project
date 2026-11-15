"""Right-hand session monitoring panel for the Feed view."""

from typing import Any, Callable, Dict, List, Optional
import streamlit as st

from demo.components.charts import create_score_breakdown_chart
from demo.components.facet_pills import render_pills_container
from paperfeed.user.session import SessionState


def render_session_snapshot(
    session: SessionState,
    on_refresh: Optional[Callable[[], None]] = None,
    on_reset: Optional[Callable[[], None]] = None,
) -> None:
    """Render real-time session feedback and telemetry panel."""
    st.markdown(
        """
        <div style="background-color: #1C1F2A; border: 1px solid #2E3348; border-radius: 8px; padding: 14px 16px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.95rem; font-weight: 600; color: #F1F5F9;">⚡ Session Monitor</span>
                <span style="font-size: 0.72rem; color: #34D399; background: rgba(52, 211, 153, 0.15); padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(52, 211, 153, 0.3);">Live State</span>
            </div>
            <div style="font-size: 0.78rem; color: #94A3B8; line-height: 1.35;">
                Tracks in-session clicks & saves to adapt the retrieval query vector <code style="color: #60A5FA;">q_t</code> in real time.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
    st.markdown(
        """
        <div style="margin-top: 14px; background-color: #161822; border: 1px solid #262B3A; border-radius: 6px; padding: 10px 12px; font-size: 0.75rem; color: #94A3B8;">
            <div style="font-weight: 600; color: #CBD5E1; margin-bottom: 6px;">Frozen System Parameters (Phase 9/11)</div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                <span>Session EMA &alpha;<sub>sess</sub>:</span>
                <strong style="color: #60A5FA;">0.30</strong>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                <span>Persona Weight &rho;:</span>
                <strong style="color: #60A5FA;">0.30</strong>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                <span>Topic Pivot Gate:</span>
                <strong style="color: #34D399;">Active (&gamma;=0.35)</strong>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>Negative Penalty &lambda;<sub>neg</sub>:</span>
                <strong style="color: #F87171;">0.50</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Recent Positive Interactions
    if session.positives:
        st.markdown("<div style='font-size: 0.82rem; font-weight: 600; color: #CBD5E1; margin-top: 16px; margin-bottom: 6px;'>Active Session Clicks (Recent)</div>", unsafe_allow_html=True)
        recent_pids = session.recent_positives(n=4)
        for pid in recent_pids:
            st.markdown(
                f"""
                <div style="background-color: #1C1F2A; border-left: 3px solid #3B82F6; padding: 4px 8px; margin-bottom: 4px; border-radius: 0 4px 4px 0; font-size: 0.75rem; color: #CBD5E1;">
                    <span style="font-family: monospace; color: #60A5FA;">{pid}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<div style='font-size: 0.78rem; color: #64748B; font-style: italic; margin-top: 14px;'>Click '✓ Relevant' on any paper card to begin steering your session.</div>",
            unsafe_allow_html=True,
        )
