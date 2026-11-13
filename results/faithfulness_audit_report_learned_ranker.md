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
| **Counterfactual Necessity (Drop-Out Rate)** | — | **85.00%** | ✅ HIGH |
| **Explanation Coverage (Top-10)** | ≥ 90.0% | **91.50%** | ✅ PASS |
| **Mean Rank Drop (ΔRank)** | — | **+4.58** | INFO |
| **Median Rank Drop** | — | **+4.0** | INFO |

## 2. Counterfactual Impact Breakdown

When the dominant explained feature is zeroed out (ΔScore = 0) and MMR is re-diversified:

| Counterfactual Outcome | Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
| **Dropped Out of Top-10** | 75 | 75.0% | Item completely exited the recommendation feed |
| **Fell ≥ 3 Ranks** | 10 | 10.0% | Item stayed in Top-10 but suffered major rank demotion |
| **Retained Rank (Drop < 3)** | 15 | 15.0% | Item was strongly supported by secondary score components |
| **Total Counterfactually Necessary** | **85** | **85.0%** | **(Fell out of Top-10 OR dropped ≥ 3 ranks)** |

## 3. Explanation Template Distribution

| Template / Dominant Feature | Sample Count | Sample Share (%) |
| :--- | :---: | :---: |
| `citation` | 56 | 56.0% |
| `persona` | 25 | 25.0% |
| `facet` | 13 | 13.0% |
| `new_method` | 6 | 6.0% |

Per template in the audited sample (grounding / counterfactual):

| Template | Audited | Grounded | Counterfactually necessary | Left Top-10 | Fell >= 3 ranks | In all covered Top-10 items |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `citation` | 56 | 56 | 50 | 47 | 3 | 1880 |
| `facet` | 13 | 13 | 11 | 9 | 2 | 286 |
| `new_method` | 6 | 6 | 2 | 1 | 1 | 21 |
| `persona` | 25 | 25 | 22 | 18 | 4 | 741 |

**Reading these numbers:**
- The plan's grounding check is about quoted spans (facet and new-method templates); 19 of the 100 audited explanations quote spans.
- **Learned ranker (V3):** explanations come from SHAP contributions and the counterfactual switches the explained feature off and re-predicts. In 1042 of 3200 Top-10 items (32.6%) the channel ranks outweigh every explainable feature. Reliability rule (design §13.8, V3 point 7: drop-out >= 50% and that share <= 50%): reliable.
- Zeroing the dominant term removes the largest part of a paper's score, so a high drop-out rate is expected by construction; it shows the explained signal is necessary, not that the other signals are irrelevant.

## 4. Sample Audited Explanations

| User | Paper | Rank | Template | Drop (New Rank) | Grounded? | Explanation Text |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| A5024751564 | W4396855402 | 3 | `facet` | +3 (Rank 6) | ✅ Yes | Evaluates on **data**, used in 2 of your papers |
| A5027988796 | W2404402506 | 8 | `citation` | +1 (Rank 9) | ✅ Yes | Shares **1** references with your paper *Improvement of the DAA protocol based on TPM* |
| A5107077385 | W4404514854 | 7 | `persona` | +4 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5044037112 | W4401539871 | 2 | `new_method` | +0 (Rank 2) | ✅ Yes | Proposes **sliding-mode mechanism**, which is new to your work, for **simulation example** that you work on |
| A5028604274 | W4403980797 | 9 | `new_method` | +2 (Fell Out) | ✅ Yes | Proposes **neural networks**, which is new to your work, for **classification** that you work on |
| A5025254339 | W4401879778 | 7 | `persona` | +4 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5007729521 | W4401210436 | 3 | `citation` | +8 (Fell Out) | ✅ Yes | Shares **3** references with your paper *How Shift Equivariance Impacts Metric Learning for Instance Segmentation* |
| A5100342308 | W4225847472 | 7 | `citation` | +4 (Fell Out) | ✅ Yes | Shares **1** references with your paper *Large-Scale Transcriptional Profiling of Molecular Perturbations Reveals Cell Type Specific Responses and Implications for Environmental Screening* |
| A5102974206 | W4400579489 | 3 | `citation` | +8 (Fell Out) | ✅ Yes | Shares **1** references with your paper *Rapid A*: a robust path planning scheme for UAVs* |
| A5102634892 | W4409685329 | 7 | `persona` | +4 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5019483041 | W4321249043 | 4 | `citation` | +7 (Fell Out) | ✅ Yes | Shares **1** references with your paper *Deep Learning in Image Classification Using Modified ResNet50 and Texture Features for Lung Cancer Detection* |
| A5081660906 | W1494484168 | 8 | `facet` | +3 (Fell Out) | ✅ Yes | Evaluates on **biological sequences**, used in 1 of your papers |
| A5003613031 | W111380116 | 3 | `citation` | +8 (Fell Out) | ✅ Yes | Shares **1** references with your paper *The Relationship Between Self-Esteem and Self-Presentation Strategies and the Construction of a Network Self-Image by Users of Social Networks* |
| A5088581659 | W2976187852 | 7 | `citation` | +4 (Fell Out) | ✅ Yes | Shares **15** references with your paper *QSFVQA: A Time Efficient, Scalable and Optimized VQA Framework* |
| A5013754013 | W255556494 | 2 | `citation` | +9 (Fell Out) | ✅ Yes | Shares **9** references with your paper *Comprehensive analysis of clustering algorithms: exploring limitations and innovative solutions* |
