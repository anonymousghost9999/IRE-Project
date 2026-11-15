"""Page 6: About System, Model Card, and User Pilot Study (§12.2)."""

import json
from typing import Any, Dict
import pandas as pd
import streamlit as st


def render_about_page() -> None:
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h2 style="margin: 0; font-size: 1.6rem; font-weight: 700; color: #F8FAFC;">ℹ️ About & User Pilot Study</h2>
            <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 4px;">
                PaperFeed Research Demo &bull; Model Card &bull; User Pilot Protocol (Design Doc §14 / Plan §12.2)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_pilot, tab_model, tab_eval = st.tabs(["📋 Pilot Study Protocol (§12.2)", "🔬 System Architecture & Model Card", "📊 Experimental Results"])

    with tab_pilot:
        st.markdown(
            """
            <div style="background-color: #1C1F2A; border: 1px solid #2E3348; border-radius: 8px; padding: 18px; margin-bottom: 16px;">
                <h4 style="margin-top: 0; color: #60A5FA;">Participant Consent & Privacy</h4>
                <p style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.45;">
                    Thank you for participating in the PaperFeed user evaluation study. 
                    This interface collects <strong>only your session interaction feedback and paper ratings</strong>. 
                    All data is held in-memory in your local session and can be exported as a JSON file below. 
                    No private identity data is stored or transmitted.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### 📝 Scripted Task Instructions")
        st.markdown(
            """
            1. **Onboarding:** Set up your profile using your OpenAlex Author ID or select 2–3 seed topics.
            2. **Initial Feed Review:** Review your Top-10 paper cards. For each paper, expand the **Pilot Evaluation** box and rate:
               - **Relevance (1–5):** How suitable is this paper for your research direction?
               - **Explanation Clarity & Grounding (1–5):** Is the stated reason transparent, faithful, and clear?
            3. **Session Steering:** Click **✓ Relevant** on 1 or 2 papers of interest, then click **🔄 Refresh Feed** to observe how the engine adapts.
            4. **Export Evaluation:** When finished, download your feedback JSON using the button below.
            """
        )

        # Pilot ID input
        pilot_id = st.text_input("Participant / Pilot ID (e.g. P01)", value=st.session_state.get("pilot_id", "P01"))
        st.session_state["pilot_id"] = pilot_id

        # Ratings collected so far
        ratings: Dict[str, Dict[str, Any]] = st.session_state.get("pilot_ratings", {})
        st.markdown(f"##### 📊 Ratings Recorded in this Session ({len(ratings)})")

        if ratings:
            df_ratings = pd.DataFrame(list(ratings.values()))
            st.dataframe(df_ratings[["paper_id", "title", "relevance", "clarity"]], use_container_width=True, hide_index=True)

            pilot_export_data = {
                "participant_id": pilot_id,
                "user_id": st.session_state.get("user_id"),
                "user_mode": st.session_state.get("user_mode"),
                "n_ratings": len(ratings),
                "ratings": ratings,
            }
            json_export = json.dumps(pilot_export_data, indent=2)
            st.download_button(
                "📥 Download My Pilot Evaluation (JSON)",
                data=json_export,
                file_name=f"paperfeed_pilot_{pilot_id}.json",
                mime="application/json",
            )
        else:
            st.info("No paper ratings recorded yet. Use the 'Pilot Evaluation' slider under each card in My Feed to rate recommendations.")

    with tab_model:
        st.markdown("##### 🏗️ System Architecture & Frozen Parameters")
        st.markdown(
            """
            | Subsystem | Technique / Model | Specification / Hyperparameter |
            | :--- | :--- | :--- |
            | **Dense Corpus Encoder** | SPECTER2 (`proximity` adapter) | 768-d vectors, L2-normalized, FAISS IndexFlatIP |
            | **Dense Query Encoder** | SPECTER2 (`adhoc_query` adapter) | Short seed/search query projection |
            | **Lexical Retriever** | BM25s | Pure-Python Okapi BM25 index over title + abstract |
            | **Candidate Fusion** | Reciprocal Rank Fusion (RRF) | $k_{\\text{rrf}} = 60$, channel depth $M_{\\text{channel}} = 200$, fused pool $M = 100$ |
            | **Persona Model** | Recency-decayed centroid | $v_{\\text{pers}} = \\text{norm}(\\sum \\gamma_{\\text{rec}}^{T-\\text{year}} e_p)$, $\\gamma_{\\text{rec}} = 0.90$ |
            | **Session Dynamics** | Exponential Moving Average | $v_t = \\alpha_{\\text{sess}} v_{t-1} + (1-\\alpha_{\\text{sess}}) e_t$, $\\alpha_{\\text{sess}} = 0.30$ |
            | **Topic Pivot Gate** | Cosine distance trigger | Resets $v_t = e_{\\text{new}}$ if $1 - \\cos(e_t, v_{t-1}) > \\gamma_{\\text{pivot}} = 0.35$ |
            | **Scoring Engine** | Min-max scaled linear model | $w_{\\text{pers}}=1.0, w_{\\text{sess}}=1.0, w_{\\text{cite}}=1.0, w_{\\text{facet}}=0.25, w_{\\text{nov}}=0.25, \\lambda_{\\text{neg}}=0.5$ |
            | **Diversification** | Maximal Marginal Relevance (MMR) | $\\lambda_{\\text{div}} = 0.10$, feed size $K = 10$ |
            | **Explainability** | Grounded template generation | Verbatim GLiNER spans, verified $\\le 2024$ citation edges |
            """
        )

    with tab_eval:
        st.markdown("##### 📈 Dual-Track Empirical Verification")
        st.markdown(
            """
            - **Track 1 (Offline Replay on Real OpenAlex Timelines):**
              - Evaluated over 300+ held-out test authors with leave-future-out protocol.
              - Confirms that combining hybrid retrieval, citation coupling, and facet signals significantly outperforms individual dense, lexical, and popularity baselines ($H_1, H_2$ supported).
            - **Track 2 (LLM Session Simulation):**
              - Evaluated over 300 multi-turn simulated researcher sessions (20 personas $\\times$ 3 seeds $\\times$ 5 arms).
              - **$H_{3a}$ Supported:** Hybrid system adapts to topic pivots **+0.500 turns faster** than static persona (95% CI $[+0.117, +1.000]$).
              - **$H_{3b}$ Supported:** Hybrid system adapts to topic pivots **+0.500 turns faster** than static blend (95% CI $[+0.117, +1.000]$).
            - **Faithfulness Audit:**
              - 100% extractive grounding rate across all displayed facet spans and evidence sentences.
            """
        )
