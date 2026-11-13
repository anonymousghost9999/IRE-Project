# Paper Feed — Phase 8: Explanation Faithfulness Audit Report

> [!NOTE]
> **Audit Configuration:**
> - Sample Size: 100 explanations across 95 distinct researchers
> - Total Evaluated Top-10 Recommendations: 3,200
> - Extractive Grounding Target: 100.0% | Explanation Coverage Target: ≥ 90.0%

## 1. Core Verification Metrics

| Metric | Target | Realized | Status |
| :--- | :---: | :---: | :---: |
| **Extractive Grounding Rate** | 100.0% | **100.00%** | ✅ PASS |
| **Counterfactual Necessity (Drop-Out Rate)** | — | **100.00%** | ✅ HIGH |
| **Explanation Coverage (Top-10)** | ≥ 90.0% | **100.00%** | ✅ PASS |
| **Mean Rank Drop (ΔRank)** | — | **+5.83** | INFO |
| **Median Rank Drop** | — | **+6.0** | INFO |

## 2. Counterfactual Impact Breakdown

When the dominant explained feature is zeroed out (ΔScore = 0) and MMR is re-diversified:

| Counterfactual Outcome | Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
| **Dropped Out of Top-10** | 100 | 100.0% | Item completely exited the recommendation feed |
| **Fell ≥ 3 Ranks** | 0 | 0.0% | Item stayed in Top-10 but suffered major rank demotion |
| **Retained Rank (Drop < 3)** | 0 | 0.0% | Item was strongly supported by secondary score components |
| **Total Counterfactually Necessary** | **100** | **100.0%** | **(Fell out of Top-10 OR dropped ≥ 3 ranks)** |

## 3. Explanation Template Distribution

| Template / Dominant Feature | Sample Count | Sample Share (%) |
| :--- | :---: | :---: |
| `persona` | 66 | 66.0% |
| `citation` | 34 | 34.0% |

Per template in the audited sample (grounding / counterfactual):

| Template | Audited | Grounded | Counterfactually necessary | Left Top-10 | Fell >= 3 ranks | In all covered Top-10 items |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `citation` | 34 | 34 | 34 | 34 | 0 | 1024 |
| `persona` | 66 | 66 | 66 | 66 | 0 | 2176 |

**Reading these numbers:**
- The plan's grounding check is about quoted spans (facet and new-method templates); 0 of the 100 audited explanations quote spans.
- **No span-quoting explanation was audited**, so the Extractive Grounding Rate above covers only citation edges (re-verified in the database) and persona/session cosines, not verbatim spans. With the tuned weights (e.g. w_facet = 0) the system produced no facet explanations to audit.
- Zeroing the dominant term removes the largest part of a paper's score, so a high drop-out rate is expected by construction; it shows the explained signal is necessary, not that the other signals are irrelevant.

## 4. Sample Audited Explanations

| User | Paper | Rank | Template | Drop (New Rank) | Grounded? | Explanation Text |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| A5107508490 | W4306179905 | 1 | `citation` | +10 (Fell Out) | ✅ Yes | Shares **2** references with your paper *Poverty Analysis and Prediction in South Africa Using Remotely Sensed Data* |
| A5101811549 | W2896503173 | 1 | `citation` | +10 (Fell Out) | ✅ Yes | Shares **8** references with your paper *Rational design of hyper‐crosslinked polymers for biomedical applications* |
| A5044442196 | W4403342810 | 1 | `citation` | +10 (Fell Out) | ✅ Yes | Shares **5** references with your paper *Barriers and Drivers Regarding the Use of Mobile Health Apps Among Patients With Type 2 Diabetes Mellitus in the Netherlands: Explanatory Sequential Design Study* |
| A5100404991 | W3169902780 | 2 | `citation` | +9 (Fell Out) | ✅ Yes | Shares **19** references with your paper *Artificial hummingbird algorithm: A new bio-inspired optimizer with its engineering applications* |
| A5100404991 | W4205175607 | 3 | `persona` | +8 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5100628775 | W4382138863 | 1 | `persona` | +10 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5053822883 | W4226002507 | 1 | `citation` | +10 (Fell Out) | ✅ Yes | Frequently cited together with your paper *Training data-efficient image transformers & distillation through attention* |
| A5001978884 | W3206087679 | 1 | `persona` | +10 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5024751564 | W4396944181 | 1 | `citation` | +10 (Fell Out) | ✅ Yes | Shares **6** references with your paper *Factors Influencing Consumer Acceptance of Mobile Payment during the COVID-19 Pandemic & Usage Continuance Intent: A Quantitative Study* |
| A5116548271 | W2897450989 | 6 | `persona` | +5 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5059013679 | W3133769291 | 4 | `citation` | +7 (Fell Out) | ✅ Yes | Shares **14** references with your paper *Visibility Restoration: A Systematic Review and Meta-Analysis* |
| A5055166499 | W4402569329 | 5 | `persona` | +6 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5001991004 | W4210257598 | 3 | `persona` | +8 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5005943811 | W4393126053 | 9 | `citation` | +2 (Fell Out) | ✅ Yes | Frequently cited together with your paper *Unpacking Chinese EFL Students’ Academic Engagement and Psychological Well-Being: The Roles of Language Teachers’ Affective Scaffolding* |
| A5007880405 | W3173159770 | 3 | `persona` | +8 (Fell Out) | ✅ Yes | Close to your overall research profile |
