# Paper Feed — Phase 8: Explanation Faithfulness Audit Report

> [!NOTE]
> **Audit Configuration:**
> - Sample Size: 100 explanations across 96 distinct researchers
> - Total Evaluated Top-10 Recommendations: 3,200
> - Extractive Grounding Target: 100.0% | Explanation Coverage Target: ≥ 90.0%

## 1. Core Verification Metrics

| Metric | Target | Realized | Status |
| :--- | :---: | :---: | :---: |
| **Extractive Grounding Rate** | 100.0% | **100.00%** | ✅ PASS |
| **Counterfactual Necessity (Drop-Out Rate)** | — | **100.00%** | ✅ HIGH |
| **Explanation Coverage (Top-10)** | ≥ 90.0% | **100.00%** | ✅ PASS |
| **Mean Rank Drop (ΔRank)** | — | **+5.30** | INFO |
| **Median Rank Drop** | — | **+5.0** | INFO |

## 2. Counterfactual Impact Breakdown

When the dominant explained feature is zeroed out (ΔScore = 0) and MMR is re-diversified:

| Counterfactual Outcome | Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
| **Dropped Out of Top-10** | 99 | 99.0% | Item completely exited the recommendation feed |
| **Fell ≥ 3 Ranks** | 1 | 1.0% | Item stayed in Top-10 but suffered major rank demotion |
| **Retained Rank (Drop < 3)** | 0 | 0.0% | Item was strongly supported by secondary score components |
| **Total Counterfactually Necessary** | **100** | **100.0%** | **(Fell out of Top-10 OR dropped ≥ 3 ranks)** |

## 3. Explanation Template Distribution

| Template / Dominant Feature | Sample Count | Sample Share (%) |
| :--- | :---: | :---: |
| `persona` | 62 | 62.0% |
| `citation` | 32 | 32.0% |
| `new_method` | 6 | 6.0% |

Per template in the audited sample (grounding / counterfactual):

| Template | Audited | Grounded | Counterfactually necessary | Left Top-10 | Fell >= 3 ranks | In all covered Top-10 items |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `citation` | 32 | 32 | 32 | 31 | 1 | 1024 |
| `new_method` | 6 | 6 | 6 | 6 | 0 | 26 |
| `persona` | 62 | 62 | 62 | 62 | 0 | 2150 |

**Reading these numbers:**
- The plan's grounding check is about quoted spans (facet and new-method templates); 6 of the 100 audited explanations quote spans.
- **Reranker (V2):** in 0 of 3200 Top-10 items (0.0%) the cross-encoder term outweighs every explainable term, so the shown explanation is not the main reason for the rank (design §13.8, V2 point 6).
- Zeroing the dominant term removes the largest part of a paper's score, so a high drop-out rate is expected by construction; it shows the explained signal is necessary, not that the other signals are irrelevant.

## 4. Sample Audited Explanations

| User | Paper | Rank | Template | Drop (New Rank) | Grounded? | Explanation Text |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| A5079637636 | W3092424940 | 9 | `new_method` | +2 (Fell Out) | ✅ Yes | Proposes **partial least squares structural equation modeling**, which is new to your work, for **assessment tasks** that you work on |
| A5077564485 | W4391855109 | 7 | `persona` | +4 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5050708271 | W4390577207 | 6 | `persona` | +5 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5020264645 | W4312290969 | 5 | `citation` | +6 (Fell Out) | ✅ Yes | Shares **15** references with your paper *Targeted Sentiment Classification Based on Attentional Encoding and Graph Convolutional Networks* |
| A5082602929 | W4384692334 | 3 | `persona` | +8 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5064920618 | W3096845075 | 4 | `persona` | +7 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5052866036 | W3203932036 | 1 | `citation` | +10 (Fell Out) | ✅ Yes | Shares **25** references with your paper *Evaluating Federated Learning for intrusion detection in Internet of Things: Review and challenges* |
| A5037551885 | W4386838102 | 5 | `persona` | +6 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5100442534 | W2586688824 | 2 | `persona` | +9 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5016669171 | W4304943299 | 7 | `citation` | +4 (Fell Out) | ✅ Yes | Shares **3** references with your paper *Ethics and Transparency Issues in Digital Platforms: An Overview* |
| A5086382599 | W3208529845 | 1 | `citation` | +10 (Fell Out) | ✅ Yes | Shares **1** references with your paper *Perancangan Sistem Otomatisasi Pada Aquascape Berbasis Mikrokontroller Arduino Nano* |
| A5101966856 | W4403458300 | 7 | `citation` | +4 (Fell Out) | ✅ Yes | Shares **6** references with your paper *A framework for inclusive AI learning design for diverse learners* |
| A5107266965 | W4353090637 | 3 | `citation` | +8 (Fell Out) | ✅ Yes | Shares **3** references with your paper *Synthetic Minority Over-sampling Technique for Student Performance Prediction: A Comparative Analysis of Ensemble and Linear Models* |
| A5053635944 | W4387071876 | 10 | `persona` | +1 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5108647667 | W4401295314 | 9 | `persona` | +2 (Fell Out) | ✅ Yes | Close to your overall research profile |
