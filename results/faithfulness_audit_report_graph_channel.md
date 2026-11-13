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
| **Counterfactual Necessity (Drop-Out Rate)** | — | **100.00%** | ✅ HIGH |
| **Explanation Coverage (Top-10)** | ≥ 90.0% | **100.00%** | ✅ PASS |
| **Mean Rank Drop (ΔRank)** | — | **+5.50** | INFO |
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
| `persona` | 55 | 55.0% |
| `citation` | 43 | 43.0% |
| `facet` | 2 | 2.0% |

Per template in the audited sample (grounding / counterfactual):

| Template | Audited | Grounded | Counterfactually necessary | Left Top-10 | Fell >= 3 ranks | In all covered Top-10 items |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `citation` | 43 | 43 | 43 | 43 | 0 | 1393 |
| `facet` | 2 | 2 | 2 | 2 | 0 | 2 |
| `persona` | 55 | 55 | 55 | 55 | 0 | 1805 |

**Reading these numbers:**
- The plan's grounding check is about quoted spans (facet and new-method templates); 2 of the 100 audited explanations quote spans.
- Zeroing the dominant term removes the largest part of a paper's score, so a high drop-out rate is expected by construction; it shows the explained signal is necessary, not that the other signals are irrelevant.

## 4. Sample Audited Explanations

| User | Paper | Rank | Template | Drop (New Rank) | Grounded? | Explanation Text |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| A5004246659 | W4401281117 | 5 | `citation` | +6 (Fell Out) | ✅ Yes | Shares **16** references with your paper *GLDM: hit molecule generation with constrained graph latent diffusion model* |
| A5101859619 | W4313458650 | 9 | `persona` | +2 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5072448304 | W3016640217 | 6 | `facet` | +5 (Fell Out) | ✅ Yes | Evaluates on **data**, used in 1 of your papers |
| A5101794413 | W2884159461 | 6 | `citation` | +5 (Fell Out) | ✅ Yes | Shares **8** references with your paper *Language Style Transfer from Sentences with Arbitrary Unknown Styles* |
| A5011782760 | W4393356472 | 7 | `citation` | +4 (Fell Out) | ✅ Yes | Shares **3** references with your paper *Al(OH) 3 facilitated synthesis of water-soluble, magnetic, radiolabelled and fluorescent hydroxyapatite nanoparticles* |
| A5045291334 | W3049223401 | 8 | `citation` | +3 (Fell Out) | ✅ Yes | Frequently cited together with your paper *Opportunities and risks of artificial intelligence in recruitment and selection* |
| A5073274738 | W4364323147 | 9 | `citation` | +2 (Fell Out) | ✅ Yes | Shares **21** references with your paper *Backbones-review: Feature extractor networks for deep learning and deep reinforcement learning approaches in computer vision* |
| A5100360494 | W4307321660 | 7 | `citation` | +4 (Fell Out) | ✅ Yes | Shares **6** references with your paper *A scoping review of COVID-19 research adopting quantitative geographical methods in geography, urban studies, and planning: a text mining approach* |
| A5010566508 | W4390493134 | 1 | `citation` | +10 (Fell Out) | ✅ Yes | Shares **7** references with your paper *Exploring the Impacts of ChatGPT on Future Scientific Work* |
| A5100628775 | W4396855832 | 1 | `citation` | +10 (Fell Out) | ✅ Yes | Shares **25** references with your paper *Performance and Challenges of 3D Object Detection Methods in Complex Scenes for Autonomous Driving* |
| A5024412258 | W3107561632 | 8 | `citation` | +3 (Fell Out) | ✅ Yes | Shares **19** references with your paper *Local Motion and Contrast Priors Driven Deep Network for Infrared Small Target Superresolution* |
| A5116288016 | W4390230213 | 7 | `persona` | +4 (Fell Out) | ✅ Yes | Close to your overall research profile |
| A5072558771 | W3018757597 | 4 | `citation` | +7 (Fell Out) | ✅ Yes | Shares **31** references with your paper *A Comprehensive Review of YOLO Architectures in Computer Vision: From YOLOv1 to YOLOv8 and YOLO-NAS* |
| A5067216432 | W4387323762 | 3 | `citation` | +8 (Fell Out) | ✅ Yes | Shares **13** references with your paper *Insights from Social Shaping Theory: The Appropriation of Large Language Models in an Undergraduate Programming Course* |
| A5013853709 | W4210846584 | 4 | `citation` | +7 (Fell Out) | ✅ Yes | Shares **6** references with your paper *DelPhi Suite: New Developments and Review of Functionalities* |
