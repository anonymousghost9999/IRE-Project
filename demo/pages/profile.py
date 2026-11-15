"""Page 5: Interest Profile & Model Transparency."""

import pandas as pd
import streamlit as st

from demo.components.charts import create_facet_bar_chart, create_timeline_chart
from demo.recommend_bridge import get_or_build_profile
from paperfeed.rec.recommender import RecommenderResources
from paperfeed.user.session import SessionState


def render_profile_page(resources: RecommenderResources) -> None:
    st.markdown(
        """
        <div style="margin-bottom: 16px;">
            <h2 style="margin: 0; font-size: 1.6rem; font-weight: 700; color: #F8FAFC;">🧠 Interest Profile</h2>
            <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 4px;">
                Inspect the underlying user model representations: long-term persona, extracted canonical facets, and short-term session dynamics.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    profile = get_or_build_profile(resources)
    session: SessionState = st.session_state.get("session")

    tab_persona, tab_session = st.tabs(["👤 Long-Term Persona", "⚡ Short-Term Session Dynamics"])

    with tab_persona:
        col_info1, col_info2 = st.columns([1.5, 1.0])
        with col_info1:
            st.markdown(
                f"""
                <div style="background-color: #1C1F2A; border: 1px solid #2E3348; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
                    <div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase; font-weight: 600;">Active Identity</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #60A5FA; margin: 4px 0;">{st.session_state.get('author_name', 'Researcher')}</div>
                    <div style="font-size: 0.82rem; color: #94A3B8; line-height: 1.4;">
                        ID: <code style="color: #F1F5F9;">{profile.user_id}</code> &bull; 
                        Representation: <strong>SPECTER2 (768-d, L2-normalized)</strong> &bull;
                        Authored Papers: <strong>{len(profile.authored_papers)}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_info2:
            st.markdown(
                """
                <div style="background-color: #1C1F2A; border: 1px solid #2E3348; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
                    <div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase; font-weight: 600;">Mathematical Formulation</div>
                    <div style="font-size: 0.82rem; color: #CBD5E1; margin-top: 4px; font-family: monospace;">
                        v_pers = norm( &Sigma; &gamma;<sub>rec</sub><sup>(T - year)</sup> &bull; e<sub>p</sub> )
                    </div>
                    <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 6px;">
                        Annual recency decay &gamma;<sub>rec</sub> = 0.90
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Timeline of authored works
        if profile.authored_papers:
            st.markdown("##### 📅 Authored Publications Across Time (&le; 2024)")
            chart_timeline = create_timeline_chart(profile.authored_papers)
            st.altair_chart(chart_timeline, use_container_width=True)

        # User Facets
        st.markdown("##### 🏷️ Top Canonical Facet Interests")
        user_facets = profile.user_facets
        facets_list = []
        if user_facets:
            for c_id, count in user_facets.counts.items():
                ftype = "topic"
                name = c_id
                if resources.facet_store:
                    canon_map = resources.facet_store.get_canonical_names([c_id])
                    if c_id in canon_map:
                        ftype = canon_map[c_id].get("facet_type", "topic")
                        name = canon_map[c_id].get("canonical_name", c_id)
                facets_list.append({"facet": name, "type": ftype, "count": count})

        facets_list.sort(key=lambda x: x["count"], reverse=True)

        if facets_list:
            col_chart, col_table = st.columns([1.4, 1.0])
            with col_chart:
                chart_facets = create_facet_bar_chart(facets_list, top_n=12)
                st.altair_chart(chart_facets, use_container_width=True)
            with col_table:
                df_facets = pd.DataFrame(facets_list[:15])
                st.dataframe(df_facets, use_container_width=True, hide_index=True)
        else:
            st.info("No canonical facets registered yet for this profile.")

    with tab_session:
        st.markdown(
            f"""
            <div style="background-color: #1C1F2A; border: 1px solid #2E3348; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
                <div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase; font-weight: 600;">Dynamic Query Blending</div>
                <div style="font-size: 0.88rem; color: #E2E8F0; margin-top: 6px; font-family: monospace;">
                    q_t = norm( (1 - &rho;) &bull; v_sess<sub>t</sub> + &rho; &bull; v_pers )
                </div>
                <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 6px;">
                    When session is empty, retrieval uses the persona alone (Track 1). Once papers are clicked/saved, session EMA steers candidate selection.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Total Interacted Clicks", len(session.positives))
        with col_s2:
            st.metric("Total Skipped Papers", len(session.negatives))
        with col_s3:
            st.metric("Pivot Gate Resets", session.n_gate_resets)

        st.markdown("##### 📜 Chronological Action History")
        if session.positives:
            for idx, pid in enumerate(session.positives, 1):
                st.markdown(
                    f"""
                    <div style="background-color: #161822; border-left: 3px solid #3B82F6; padding: 8px 12px; margin-bottom: 6px; border-radius: 0 6px 6px 0; font-size: 0.8rem; color: #CBD5E1;">
                        <span style="font-weight: 600; color: #60A5FA;">Step {idx}:</span> Clicked / Saved <code style="color: #F8FAFC;">{pid}</code>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No actions recorded in this session yet.")
