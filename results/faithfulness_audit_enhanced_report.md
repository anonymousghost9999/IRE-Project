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
| **Mean Rank Drop (ΔRank)** | — | **+5.74** | INFO |
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
| `persona` | 65 | 65.0% |
| `citation` | 35 | 35.0% |

Per template in the audited sample (grounding / counterfactual):

| Template | Audited | Grounded | Counterfactually necessary | Left Top-10 | Fell >= 3 ranks | In all covered Top-10 items |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `citation` | 35 | 35 | 35 | 35 | 0 | 1068 |
| `persona` | 65 | 65 | 65 | 65 | 0 | 2132 |

**Reading these numbers:**
- The plan's grounding check is about quoted spans (facet and new-method templates); 0 of the 100 audited explanations quote spans.
- **No span-quoting explanation was audited**, so the Extractive Grounding Rate above covers only citation edges (re-verified in the database) and persona/session cosines, not verbatim spans. With the audited weights the facet and new-method terms never dominated a Top-10 score, so no span explanation was shown.
- Zeroing the dominant term removes the largest part of a paper's score, so a high drop-out rate is expected by construction; it shows the explained signal is necessary, not that the other signals are irrelevant.

## 4. Sample Audited Explanations

| User | Paper | Rank | Template | Drop (New Rank) | Grounded? | Explanation Text |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| A5017117569 | W3120802658 | 4 | `citation` | +7 (Fell Out) | ✅ Yes | Shares **10** references with your paper *Fast and efficient recursive algorithm of Meixner polynomials* |
| A5102531513 | W1988134541 | 8 | `persona` | +3 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5022888377 | W4385423342 | 1 | `persona` | +10 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5113968850 | W4285178897 | 9 | `persona` | +2 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5020264645 | W4385461227 | 5 | `persona` | +6 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5051930823 | W3138224139 | 5 | `citation` | +6 (Fell Out) | ✅ Yes | Shares **3** references with your paper *A Reliable Data Hiding Scheme Using Jigsaw Sudoku* |
| A5045879529 | W2762911482 | 1 | `citation` | +10 (Fell Out) | ✅ Yes | Shares **6** references with your paper *Dual iterative adaptive dynamic programming for a class of discrete-time nonlinear systems with time-delays* |
| A5101859619 | W4391607014 | 3 | `citation` | +8 (Fell Out) | ✅ Yes | Shares **5** references with your paper *CardioGuard: AI-driven ECG authentication hybrid neural network for predictive health monitoring in telehealth systems* |
| A5072512285 | W4304694200 | 9 | `persona` | +2 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5092611926 | W4389719314 | 1 | `persona` | +10 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5053822883 | W4226512186 | 8 | `citation` | +3 (Fell Out) | ✅ Yes | Shares **21** references with your paper *LeViT: a Vision Transformer in ConvNet’s Clothing for Faster Inference* |
| A5012572669 | W2979417040 | 6 | `citation` | +5 (Fell Out) | ✅ Yes | Shares **2** references with your paper *Sosyal medyada nefret söylemi çerçevesi: Geçmişten günümüze kapsamlı bir derleme* |
| A5047345391 | W4313458650 | 2 | `persona` | +9 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5117691222 | W4400286290 | 1 | `persona` | +10 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5100404991 | W4385259378 | 5 | `persona` | +6 (Fell Out) | ✅ Yes | Close to your overall research profile |
