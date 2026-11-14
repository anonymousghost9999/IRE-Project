# Paper Feed — Phase 8: Explanation Faithfulness Audit Report

> [!NOTE]
> **Audit Configuration:**
> - Sample Size: 100 explanations across 97 distinct researchers
> - Total Evaluated Top-10 Recommendations: 3,200
> - Extractive Grounding Target: 100.0% | Explanation Coverage Target: ≥ 90.0%

## 1. Core Verification Metrics

| Metric | Target | Realized | Status |
| :--- | :---: | :---: | :---: |
| **Extractive Grounding Rate** | 100.0% | **100.00%** | ✅ PASS |
| **Counterfactual Necessity (Drop-Out Rate)** | — | **100.00%** | ✅ HIGH |
| **Explanation Coverage (Top-10)** | ≥ 90.0% | **100.00%** | ✅ PASS |
| **Mean Rank Drop (ΔRank)** | — | **+5.88** | INFO |
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
| `citation` | 31 | 31.0% |
| `facet` | 3 | 3.0% |

Per template in the audited sample (grounding / counterfactual):

| Template | Audited | Grounded | Counterfactually necessary | Left Top-10 | Fell >= 3 ranks | In all covered Top-10 items |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `citation` | 31 | 31 | 31 | 31 | 0 | 959 |
| `facet` | 3 | 3 | 3 | 3 | 0 | 3 |
| `persona` | 66 | 66 | 66 | 66 | 0 | 2238 |

**Reading these numbers:**
- The plan's grounding check is about quoted spans (facet and new-method templates); 3 of the 100 audited explanations quote spans.
- Zeroing the dominant term removes the largest part of a paper's score, so a high drop-out rate is expected by construction; it shows the explained signal is necessary, not that the other signals are irrelevant.

## 4. Sample Audited Explanations

| User | Paper | Rank | Template | Drop (New Rank) | Grounded? | Explanation Text |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| A5002622547 | W2134295053 | 6 | `citation` | +5 (Fell Out) | ✅ Yes | Shares **4** references with your paper *A realistic named data networking architecture for the Internet of things* |
| A5053635944 | W4403143620 | 10 | `persona` | +1 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5095690828 | W4401867536 | 5 | `persona` | +6 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5112867666 | W4387424085 | 3 | `persona` | +8 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5101151571 | W4361213044 | 9 | `persona` | +2 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5022849698 | W4398151550 | 1 | `persona` | +10 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5102531513 | W4405180660 | 5 | `persona` | +6 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5030028173 | W2186589370 | 6 | `persona` | +5 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5095084159 | W3159952799 | 4 | `persona` | +7 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5037568194 | W4289792856 | 8 | `persona` | +3 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5045291334 | W4395695324 | 1 | `persona` | +10 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5051895612 | W2056663787 | 4 | `citation` | +7 (Fell Out) | ✅ Yes | Shares **3** references with your paper *Effect of management commitment to internal marketing on employee work attitude* |
| A5093315589 | W4375846667 | 7 | `persona` | +4 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5055166499 | W4400406773 | 9 | `persona` | +2 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5065793456 | W4387078269 | 6 | `persona` | +5 (Fell Out) | ✅ Yes | Close to your overall research profile |
