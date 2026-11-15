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
| **Mean Rank Drop (ΔRank)** | — | **+5.37** | INFO |
| **Median Rank Drop** | — | **+5.0** | INFO |

## 2. Counterfactual Impact Breakdown

When the dominant explained feature is zeroed out (ΔScore = 0) and MMR is re-diversified:

| Counterfactual Outcome | Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
| **Dropped Out of Top-10** | 97 | 97.0% | Item completely exited the recommendation feed |
| **Fell ≥ 3 Ranks** | 3 | 3.0% | Item stayed in Top-10 but suffered major rank demotion |
| **Retained Rank (Drop < 3)** | 0 | 0.0% | Item was strongly supported by secondary score components |
| **Total Counterfactually Necessary** | **100** | **100.0%** | **(Fell out of Top-10 OR dropped ≥ 3 ranks)** |

## 3. Explanation Template Distribution

| Template / Dominant Feature | Sample Count | Sample Share (%) |
| :--- | :---: | :---: |
| `persona` | 82 | 82.0% |
| `new_method` | 7 | 7.0% |
| `facet` | 6 | 6.0% |
| `citation` | 5 | 5.0% |

Per template in the audited sample (grounding / counterfactual):

| Template | Audited | Grounded | Counterfactually necessary | Left Top-10 | Fell >= 3 ranks | In all covered Top-10 items |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `citation` | 5 | 5 | 5 | 5 | 0 | 6 |
| `facet` | 6 | 6 | 6 | 5 | 1 | 55 |
| `new_method` | 7 | 7 | 7 | 6 | 1 | 64 |
| `persona` | 82 | 82 | 82 | 81 | 1 | 3075 |

**Reading these numbers:**
- The plan's grounding check is about quoted spans (facet and new-method templates); 13 of the 100 audited explanations quote spans.
- Zeroing the dominant term removes the largest part of a paper's score, so a high drop-out rate is expected by construction; it shows the explained signal is necessary, not that the other signals are irrelevant.

## 4. Sample Audited Explanations

| User | Paper | Rank | Template | Drop (New Rank) | Grounded? | Explanation Text |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| A5017262847 | W3217592719 | 10 | `persona` | +1 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5047299059 | W4402733576 | 1 | `persona` | +10 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5002858009 | W2753899727 | 5 | `persona` | +6 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5046104514 | W2944170415 | 6 | `persona` | +5 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5070869667 | W4404021044 | 8 | `persona` | +3 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5012572669 | W1526441817 | 8 | `persona` | +3 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5100399848 | W4403046562 | 7 | `persona` | +4 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5099032966 | W4313425238 | 4 | `persona` | +7 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5031916377 | W4323338839 | 4 | `persona` | +7 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5001473724 | W4395115917 | 6 | `persona` | +5 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5077173629 | W255556494 | 4 | `persona` | +7 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5038233087 | W255556494 | 2 | `persona` | +9 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5018676156 | W3184210465 | 5 | `persona` | +6 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5013076694 | W4391249990 | 9 | `persona` | +2 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5047691220 | W4377041914 | 5 | `new_method` | +6 (Fell Out) | ✅ Yes | Proposes **Natural language processing**, which is new to your work, for **peer‐feedback process** that you work on |
