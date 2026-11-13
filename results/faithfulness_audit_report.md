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
| **Mean Rank Drop (ΔRank)** | — | **+5.87** | INFO |
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
| `persona` | 67 | 67.0% |
| `citation` | 33 | 33.0% |

Per template in the audited sample (grounding / counterfactual):

| Template | Audited | Grounded | Counterfactually necessary | Left Top-10 | Fell >= 3 ranks | In all covered Top-10 items |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `citation` | 33 | 33 | 33 | 33 | 0 | 1012 |
| `persona` | 67 | 67 | 67 | 67 | 0 | 2188 |

**Reading these numbers:**
- The plan's grounding check is about quoted spans (facet and new-method templates); 0 of the 100 audited explanations quote spans.
- **No span-quoting explanation was audited**, so the Extractive Grounding Rate above covers only citation edges (re-verified in the database) and persona/session cosines, not verbatim spans. With the audited weights the facet and new-method terms never dominated a Top-10 score, so no span explanation was shown.
- Zeroing the dominant term removes the largest part of a paper's score, so a high drop-out rate is expected by construction; it shows the explained signal is necessary, not that the other signals are irrelevant.

## 4. Sample Audited Explanations

| User | Paper | Rank | Template | Drop (New Rank) | Grounded? | Explanation Text |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| A5061391366 | W4392739263 | 6 | `citation` | +5 (Fell Out) | ✅ Yes | Shares **3** references with your paper *Breast Cancer Immunohistochemical Image Generation: a Benchmark Dataset and Challenge Review* |
| A5101323013 | W4360884927 | 4 | `citation` | +7 (Fell Out) | ✅ Yes | Frequently cited together with your paper *CamoDiffusion: Camouflaged Object Detection via Conditional Diffusion Models* |
| A5036932232 | W4387824313 | 10 | `citation` | +1 (Fell Out) | ✅ Yes | Shares **5** references with your paper *PKSmart: An Open-Source Computational Model to Predict in vivo Pharmacokinetics of Small Molecules* |
| A5059013679 | W4391115847 | 2 | `citation` | +9 (Fell Out) | ✅ Yes | Shares **13** references with your paper *Visibility Restoration: A Systematic Review and Meta-Analysis* |
| A5091224174 | W4394930977 | 9 | `persona` | +2 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5010341431 | W4400111632 | 5 | `persona` | +6 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5009129791 | W4404079283 | 1 | `citation` | +10 (Fell Out) | ✅ Yes | Shares **8** references with your paper *Effects of Well-being, Grit, Emotion Regulation, and Resilience Interventions on L2 Learners’ Writing Skills* |
| A5001016117 | W2145747124 | 7 | `persona` | +4 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5020661868 | W4403318601 | 9 | `citation` | +2 (Fell Out) | ✅ Yes | Shares **6** references with your paper *Swift: Expedited Failure Recovery for Large-Scale DNN Training* |
| A5034682935 | W4402508171 | 10 | `persona` | +1 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5113880986 | W4292506131 | 3 | `citation` | +8 (Fell Out) | ✅ Yes | Shares **2** references with your paper *Efficient, Lightweight Cyber Intrusion Detection System for IoT Ecosystems Using MI2G Algorithm* |
| A5065193592 | W4385420376 | 8 | `persona` | +3 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5100655268 | W2965481926 | 6 | `persona` | +5 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5074421743 | W4368367371 | 10 | `citation` | +1 (Fell Out) | ✅ Yes | Shares **6** references with your paper *Automated multi-modal Transformer network (AMTNet) for 3D medical images segmentation* |
| A5101831987 | W2003678792 | 5 | `persona` | +6 (Fell Out) | ✅ Yes | Close to your overall research profile |
