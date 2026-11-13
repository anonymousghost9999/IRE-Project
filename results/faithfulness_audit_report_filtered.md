# Paper Feed — Phase 8: Explanation Faithfulness Audit Report

> [!NOTE]
> **Audit Configuration:**
> - Sample Size: 100 explanations across 94 distinct researchers
> - Total Evaluated Top-10 Recommendations: 3,200
> - Extractive Grounding Target: 100.0% | Explanation Coverage Target: ≥ 90.0%

## 1. Core Verification Metrics

| Metric | Target | Realized | Status |
| :--- | :---: | :---: | :---: |
| **Extractive Grounding Rate** | 100.0% | **100.00%** | ✅ PASS |
| **Counterfactual Necessity (Drop-Out Rate)** | — | **100.00%** | ✅ HIGH |
| **Explanation Coverage (Top-10)** | ≥ 90.0% | **100.00%** | ✅ PASS |
| **Mean Rank Drop (ΔRank)** | — | **+5.11** | INFO |
| **Median Rank Drop** | — | **+5.0** | INFO |

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
| `persona` | 67 | 67.0% |
| `citation` | 33 | 33.0% |

Per template in the audited sample (grounding / counterfactual):

| Template | Audited | Grounded | Counterfactually necessary | Left Top-10 | Fell >= 3 ranks | In all covered Top-10 items |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `citation` | 33 | 33 | 33 | 33 | 0 | 1000 |
| `persona` | 67 | 67 | 67 | 67 | 0 | 2200 |

**Reading these numbers:**
- The plan's grounding check is about quoted spans (facet and new-method templates); 0 of the 100 audited explanations quote spans.
- **No span-quoting explanation was audited**, so the Extractive Grounding Rate above covers only citation edges (re-verified in the database) and persona/session cosines, not verbatim spans. With the tuned weights (e.g. w_facet = 0) the system produced no facet explanations to audit.
- Zeroing the dominant term removes the largest part of a paper's score, so a high drop-out rate is expected by construction; it shows the explained signal is necessary, not that the other signals are irrelevant.

## 4. Sample Audited Explanations

| User | Paper | Rank | Template | Drop (New Rank) | Grounded? | Explanation Text |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| A5082602929 | W4220691145 | 3 | `citation` | +8 (Fell Out) | ✅ Yes | Shares **10** references with your paper *A quantitative framework reveals ecological drivers of grassland microbial community assembly in response to warming* |
| A5022888377 | W4385423342 | 1 | `citation` | +10 (Fell Out) | ✅ Yes | Shares **5** references with your paper *ALOC: Attack-Aware by Utilizing the Adversarially Learned One-Class Classifier for SCADA System* |
| A5101966856 | W4405893789 | 1 | `citation` | +10 (Fell Out) | ✅ Yes | Shares **9** references with your paper *Pedagogical agents in K-12 education: a scoping review* |
| A5058765046 | W4401980346 | 9 | `citation` | +2 (Fell Out) | ✅ Yes | Shares **26** references with your paper *S-Adapter: Generalizing Vision Transformer for Face Anti-Spoofing With Statistical Tokens* |
| A5089729827 | W3107763902 | 4 | `persona` | +7 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5010113677 | W4366783381 | 4 | `persona` | +7 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5090650251 | W4322766882 | 2 | `citation` | +9 (Fell Out) | ✅ Yes | Frequently cited together with your paper *A Survey on Mixture of Experts* |
| A5001016117 | W2898288652 | 9 | `persona` | +2 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5045016749 | W4385766743 | 4 | `citation` | +7 (Fell Out) | ✅ Yes | Shares **10** references with your paper *Explainable Artificial Intelligence (XAI) 2.0: A manifesto of open challenges and interdisciplinary research directions* |
| A5092611926 | W4376139682 | 5 | `persona` | +6 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5040252691 | W4387653441 | 9 | `citation` | +2 (Fell Out) | ✅ Yes | Shares **6** references with your paper *Semi-Supervised Person Detection in Aerial Images with Instance Segmentation and Maximum Mean Discrepancy Distance* |
| A5064993007 | W4309802094 | 5 | `persona` | +6 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5100438513 | W2765794674 | 10 | `persona` | +1 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5010566508 | W4389748543 | 1 | `citation` | +10 (Fell Out) | ✅ Yes | Shares **2** references with your paper *Exploring the Impacts of ChatGPT on Future Scientific Work* |
| A5058906826 | W4401777108 | 2 | `persona` | +9 (Fell Out) | ✅ Yes | Close to your overall research profile |
