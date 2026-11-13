# Paper Feed — Phase 8: Explanation Faithfulness Audit Report

> [!NOTE]
> **Audit Configuration:**
> - Sample Size: 100 explanations across 93 distinct researchers
> - Total Evaluated Top-10 Recommendations: 3,200
> - Extractive Grounding Target: 100.0% | Explanation Coverage Target: ≥ 90.0%

## 1. Core Verification Metrics

| Metric | Target | Realized | Status |
| :--- | :---: | :---: | :---: |
| **Extractive Grounding Rate** | 100.0% | **100.00%** | ✅ PASS |
| **Counterfactual Necessity (Drop-Out Rate)** | — | **98.00%** | ✅ HIGH |
| **Explanation Coverage (Top-10)** | ≥ 90.0% | **100.00%** | ✅ PASS |
| **Mean Rank Drop (ΔRank)** | — | **+4.49** | INFO |
| **Median Rank Drop** | — | **+4.0** | INFO |

## 2. Counterfactual Impact Breakdown

When the dominant explained feature is zeroed out (ΔScore = 0) and MMR is re-diversified:

| Counterfactual Outcome | Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
| **Dropped Out of Top-10** | 92 | 92.0% | Item completely exited the recommendation feed |
| **Fell ≥ 3 Ranks** | 6 | 6.0% | Item stayed in Top-10 but suffered major rank demotion |
| **Retained Rank (Drop < 3)** | 2 | 2.0% | Item was strongly supported by secondary score components |
| **Total Counterfactually Necessary** | **98** | **98.0%** | **(Fell out of Top-10 OR dropped ≥ 3 ranks)** |

## 3. Explanation Template Distribution

| Template / Dominant Feature | Sample Count | Sample Share (%) |
| :--- | :---: | :---: |
| `persona` | 34 | 34.0% |
| `new_method` | 29 | 29.0% |
| `facet` | 19 | 19.0% |
| `citation` | 18 | 18.0% |

Per template in the audited sample (grounding / counterfactual):

| Template | Audited | Grounded | Counterfactually necessary | Left Top-10 | Fell >= 3 ranks | In all covered Top-10 items |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `citation` | 18 | 18 | 18 | 18 | 0 | 527 |
| `facet` | 19 | 19 | 17 | 12 | 5 | 565 |
| `new_method` | 29 | 29 | 29 | 28 | 1 | 961 |
| `persona` | 34 | 34 | 34 | 34 | 0 | 1147 |

**Reading these numbers:**
- The plan's grounding check is about quoted spans (facet and new-method templates); 48 of the 100 audited explanations quote spans.
- Zeroing the dominant term removes the largest part of a paper's score, so a high drop-out rate is expected by construction; it shows the explained signal is necessary, not that the other signals are irrelevant.

## 4. Sample Audited Explanations

| User | Paper | Rank | Template | Drop (New Rank) | Grounded? | Explanation Text |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| A5111756734 | W4403676123 | 8 | `persona` | +3 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5037493401 | W3048886990 | 10 | `new_method` | +1 (Fell Out) | ✅ Yes | Proposes **deep transfer learning techniques**, which is new to your work, for **testing** that you work on |
| A5083549838 | W4285259820 | 8 | `persona` | +3 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5113880986 | W4380995531 | 1 | `facet` | +3 (Rank 4) | ✅ Yes | Addresses **classification problem**, studied in 1 of your papers |
| A5007729521 | W4319300975 | 7 | `facet` | +4 (Fell Out) | ✅ Yes | Evaluates on **CNN**, used in 2 of your papers |
| A5101831987 | W2610352123 | 10 | `persona` | +1 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5100307907 | W4296624876 | 5 | `persona` | +6 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5065193592 | W4400111632 | 2 | `facet` | +9 (Fell Out) | ✅ Yes | Addresses **challenge**, studied in 9 of your papers |
| A5090650251 | W2996075766 | 9 | `persona` | +2 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5100432232 | W2499240546 | 1 | `citation` | +10 (Fell Out) | ✅ Yes | Shares **10** references with your paper *Common Variants at 9p21 and 8q22 Are Associated with Increased Susceptibility to Optic Nerve Degeneration in Glaucoma* |
| A5016669171 | W2754960012 | 6 | `citation` | +5 (Fell Out) | ✅ Yes | Shares **4** references with your paper *Cognitive Enhancement – A Critical Look at the Recent Debate* |
| A5037974949 | W4390278281 | 8 | `persona` | +3 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5037551885 | W4386838102 | 6 | `persona` | +5 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5116548271 | W3083572825 | 10 | `persona` | +1 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5068400798 | W4291891037 | 5 | `citation` | +6 (Fell Out) | ✅ Yes | Shares **8** references with your paper *Automatic Recognition of Mexican Sign Language Using a Depth Camera and Recurrent Neural Networks* |
