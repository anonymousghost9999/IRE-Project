# Leak-Free Sensitivity Analysis of the Published Results

> [!NOTE]
> - A post-hoc audit (2026-11-15, design §18a) found that 23 of the 400 authors of the original cohort had a version of their own target paper B (same title, another OpenAlex ID, e.g. a preprint) in their history <= T: 19 test and 4 tuning authors. Check: `paperfeed.data.leakage.titles_match` (exact after normalisation, containment, or token Jaccard >= 0.75).
> - Every comparison below is recomputed from the saved per-author results on the 301 test authors without the leak. Every system was frozen before its test run, so this is a valid paired comparison of the same systems; nothing was re-tuned or re-run. Families, bootstrap and BH correction are the published ones (10,000 resamples).
> - The 4 leaked tuning authors could only have influenced which weights were chosen; that influence is not removed here and is reported as a limitation.
> - **2 conclusion(s) change** when the leaked authors are removed.

## 1. H1: Paper Feed vs the five baselines (H1 supported: all False, leak-free False)

| Comparison | All test authors (n = 320) | Leak-free test authors (n = 301) | Conclusion |
| :--- | :--- | :--- | :---: |
| Baseline: popularity | ndcg_at_10 | 0.0216 vs 0.0129, diff +0.0087 [-0.0011, +0.0185], p_BH 0.092 (n.s.) | 0.0185 vs 0.0138, diff +0.0047 [-0.0050, +0.0142], p_BH 0.337 (n.s.) | same |
| Baseline: bm25_author | ndcg_at_10 | 0.0216 vs 0.0142, diff +0.0074 [-0.0015, +0.0163], p_BH 0.102 (n.s.) | 0.0185 vs 0.0091, diff +0.0094 [+0.0023, +0.0170], p_BH 0.018 (**sig.**) | **CHANGED** |
| Baseline: dense_centroid | ndcg_at_10 | 0.0216 vs 0.0111, diff +0.0105 [+0.0038, +0.0175], p_BH 0.009 (**sig.**) | 0.0185 vs 0.0080, diff +0.0104 [+0.0051, +0.0164], p_BH 0.003 (**sig.**) | same |
| Baseline: citation_only | ndcg_at_10 | 0.0216 vs 0.0348, diff -0.0132 [-0.0244, -0.0024], p_BH 0.025 (**sig.**) | 0.0185 vs 0.0334, diff -0.0150 [-0.0266, -0.0042], p_BH 0.015 (**sig.**) | same |
| Baseline: hybrid_persona | ndcg_at_10 | 0.0216 vs 0.0128, diff +0.0088 [+0.0027, +0.0153], p_BH 0.010 (**sig.**) | 0.0185 vs 0.0107, diff +0.0078 [+0.0023, +0.0136], p_BH 0.015 (**sig.**) | same |
| Baseline: popularity | recall_at_10 | 0.0156 vs 0.0062, diff +0.0094 [+0.0029, +0.0161], p_BH 0.010 (**sig.**) | 0.0143 vs 0.0066, diff +0.0076 [+0.0010, +0.0144], p_BH 0.029 (**sig.**) | same |
| Baseline: bm25_author | recall_at_10 | 0.0156 vs 0.0094, diff +0.0062 [+0.0004, +0.0121], p_BH 0.046 (**sig.**) | 0.0143 vs 0.0076, diff +0.0067 [+0.0011, +0.0126], p_BH 0.029 (**sig.**) | same |
| Baseline: dense_centroid | recall_at_10 | 0.0156 vs 0.0085, diff +0.0071 [+0.0027, +0.0120], p_BH 0.009 (**sig.**) | 0.0143 vs 0.0070, diff +0.0073 [+0.0030, +0.0122], p_BH 0.008 (**sig.**) | same |
| Baseline: citation_only | recall_at_10 | 0.0156 vs 0.0273, diff -0.0117 [-0.0196, -0.0042], p_BH 0.009 (**sig.**) | 0.0143 vs 0.0267, diff -0.0124 [-0.0205, -0.0047], p_BH 0.008 (**sig.**) | same |
| Baseline: hybrid_persona | recall_at_10 | 0.0156 vs 0.0093, diff +0.0063 [+0.0024, +0.0107], p_BH 0.009 (**sig.**) | 0.0143 vs 0.0085, diff +0.0057 [+0.0019, +0.0101], p_BH 0.015 (**sig.**) | same |

## 2. H2: MMR diversity with non-inferiority (H2 supported: all True, leak-free True)

| Comparison | All test authors (n = 320) | Leak-free test authors (n = 301) | Conclusion |
| :--- | :--- | :--- | :---: |
| entropy | 1.5405 vs 1.5127, diff +0.0278 [+0.0168, +0.0399], p_BH 1.000 (**sig.**) | 1.5395 vs 1.5109, diff +0.0286 [+0.0170, +0.0412], p_BH 1.000 (**sig.**) | same |
| ndcg retention | 0.0216 vs 0.0212, diff +0.0004 [-0.0002, +0.0013], p_BH 1.000 (**sig.**) | 0.0185 vs 0.0183, diff +0.0002 [-0.0003, +0.0009], p_BH 1.000 (**sig.**) | same |

## 3. Phase 10: each variant vs Core (BH over the variant family)

| Comparison | All test authors (n = 320) | Leak-free test authors (n = 301) | Conclusion |
| :--- | :--- | :--- | :---: |
| v1 | ndcg_at_10 | 0.0262 vs 0.0216, diff +0.0046 [-0.0015, +0.0109], p_BH 0.283 (n.s.) | 0.0230 vs 0.0185, diff +0.0046 [-0.0015, +0.0108], p_BH 0.292 (n.s.) | same |
| v1 | recall_at_10 | 0.0200 vs 0.0156, diff +0.0044 [+0.0004, +0.0086], p_BH 0.154 (n.s.) | 0.0187 vs 0.0143, diff +0.0044 [+0.0005, +0.0087], p_BH 0.101 (n.s.) | same |
| v2 | ndcg_at_10 | 0.0205 vs 0.0216, diff -0.0011 [-0.0034, +0.0011], p_BH 0.442 (n.s.) | 0.0176 vs 0.0185, diff -0.0009 [-0.0033, +0.0014], p_BH 0.605 (n.s.) | same |
| v2 | recall_at_10 | 0.0154 vs 0.0156, diff -0.0002 [-0.0023, +0.0018], p_BH 0.828 (n.s.) | 0.0140 vs 0.0143, diff -0.0002 [-0.0025, +0.0019], p_BH 0.952 (n.s.) | same |
| v3 | ndcg_at_10 | 0.0202 vs 0.0216, diff -0.0014 [-0.0064, +0.0037], p_BH 0.657 (n.s.) | 0.0183 vs 0.0185, diff -0.0001 [-0.0050, +0.0048], p_BH 0.955 (n.s.) | same |
| v3 | recall_at_10 | 0.0136 vs 0.0156, diff -0.0021 [-0.0057, +0.0017], p_BH 0.421 (n.s.) | 0.0129 vs 0.0143, diff -0.0014 [-0.0052, +0.0021], p_BH 0.605 (n.s.) | same |
| v4 | ndcg_at_10 | 0.0264 vs 0.0216, diff +0.0047 [-0.0002, +0.0105], p_BH 0.212 (n.s.) | 0.0245 vs 0.0185, diff +0.0061 [+0.0010, +0.0120], p_BH 0.101 (n.s.) | same |
| v4 | recall_at_10 | 0.0197 vs 0.0156, diff +0.0040 [+0.0003, +0.0080], p_BH 0.154 (n.s.) | 0.0183 vs 0.0143, diff +0.0041 [+0.0004, +0.0081], p_BH 0.101 (n.s.) | same |

## 4. Phase 11: Paper Feed vs each comparison system (BH over the comparison family)

| Comparison | All test authors (n = 320) | Leak-free test authors (n = 301) | Conclusion |
| :--- | :--- | :--- | :---: |
| SOTA: RP3beta | ndcg_at_10 | 0.0216 vs 0.0607, diff -0.0391 [-0.0553, -0.0238], p_BH 0.000 (**sig.**) | 0.0185 vs 0.0625, diff -0.0441 [-0.0605, -0.0284], p_BH 0.000 (**sig.**) | same |
| SOTA: RankGPT (Qwen2.5-3B) | ndcg_at_10 | 0.0216 vs 0.0054, diff +0.0162 [+0.0093, +0.0241], p_BH 0.000 (**sig.**) | 0.0185 vs 0.0050, diff +0.0135 [+0.0069, +0.0209], p_BH 0.001 (**sig.**) | same |
| SOTA: Scholar Inbox (GTE-large + LR) | ndcg_at_10 | 0.0216 vs 0.0174, diff +0.0042 [-0.0046, +0.0126], p_BH 0.341 (n.s.) | 0.0185 vs 0.0136, diff +0.0049 [-0.0023, +0.0122], p_BH 0.185 (n.s.) | same |
| SOTA: SciNCL centroid kNN | ndcg_at_10 | 0.0216 vs 0.0123, diff +0.0094 [+0.0024, +0.0167], p_BH 0.013 (**sig.**) | 0.0185 vs 0.0116, diff +0.0069 [+0.0002, +0.0139], p_BH 0.068 (n.s.) | **CHANGED** |
| SOTA: RP3beta | recall_at_10 | 0.0156 vs 0.0386, diff -0.0229 [-0.0341, -0.0127], p_BH 0.000 (**sig.**) | 0.0143 vs 0.0394, diff -0.0252 [-0.0365, -0.0144], p_BH 0.000 (**sig.**) | same |
| SOTA: RankGPT (Qwen2.5-3B) | recall_at_10 | 0.0156 vs 0.0042, diff +0.0115 [+0.0065, +0.0170], p_BH 0.000 (**sig.**) | 0.0143 vs 0.0039, diff +0.0103 [+0.0054, +0.0158], p_BH 0.000 (**sig.**) | same |
| SOTA: Scholar Inbox (GTE-large + LR) | recall_at_10 | 0.0156 vs 0.0119, diff +0.0037 [-0.0012, +0.0089], p_BH 0.163 (n.s.) | 0.0143 vs 0.0110, diff +0.0033 [-0.0015, +0.0085], p_BH 0.185 (n.s.) | same |
| SOTA: SciNCL centroid kNN | recall_at_10 | 0.0156 vs 0.0091, diff +0.0065 [+0.0019, +0.0114], p_BH 0.012 (**sig.**) | 0.0143 vs 0.0087, diff +0.0055 [+0.0009, +0.0106], p_BH 0.038 (**sig.**) | same |

*RankGPT ran with Qwen2.5-1.5B in fp16 on the run machine (design §18a); its saved label says 3B.*

## 5. Leaked authors

| Author | Split | History paper | Its title | Title of B |
| :--- | :---: | :--- | :--- | :--- |
| A5001991004 | test | W4403679101 | Deep Graph Anomaly Detection: A Survey and New Perspectives | Deep Graph Anomaly Detection: A Survey and New Perspectives |
| A5004310412 | test | W4403883700 | Emotion-Aware Embedding Fusion in LLMs (Flan-T5, LLAMA 2, DeepSeek-R1, and ChatG | Emotion-Aware Embedding Fusion in Large Language Models (Flan-T5, Llama 2, DeepS |
| A5006995271 | test | W4394716879 | Development and Validation of Generative Artificial Intelligence Attitude Scale  | Development and validation of generative artificial intelligence attitude scale  |
| A5011404590 | test | W4388092696 | Generative Artificial Intelligence for Software Engineering -- A Research Agenda | Generative Artificial Intelligence for Software Engineering—A Research Agenda |
| A5017859484 | test | W4387221424 | The Promises and Pitfalls of LLMs as Feedback Providers: A Study of Prompt Engin | The Promises and Pitfalls of Large Language Models as Feedback Providers: A Stud |
| A5021494379 | test | W4200187172 | Explainable AI and Random Forest Based Reliable Intrusion Detection system | Explainable AI and Random Forest based reliable intrusion detection system |
| A5052866036 | test | W4386069007 | Intrusion Detection based on Federated Learning: a systematic review | Intrusion Detection Based on Federated Learning: A Systematic Review |
| A5053822883 | test | W4390962868 | The Faiss library | The Faiss Library |
| A5069589370 | test | W4388184510 | A Systematic Review for Transformer-based Long-term Series Forecasting | A systematic review for transformer-based long-term series forecasting |
| A5086254582 | test | W4225011431 | A review of Federated Learning in Intrusion Detection Systems for IoT | A Review of Federated Learning Applications in Intrusion Detection Systems |
| A5090650251 | test | W4400445270 | A Survey on Mixture of Experts | A Survey on Mixture of Experts in Large Language Models |
| A5091464412 | test | W4401307342 | A Systematic Review on Long-Tailed Learning | A Systematic Review on Long-Tailed Learning |
| A5099611070 | test | W4402740813 | Knowledge Graph Enhanced Retrieval-Augmented Generation for Failure Mode and Eff | Knowledge graph enhanced retrieval-augmented generation for failure mode and eff |
| A5100343402 | test | W4415392087 | Author response for "Enhanced Deep Image Prior for Unsupervised Hyperspectral Im | Enhanced Deep Image Prior for Unsupervised Hyperspectral Image Super-Resolution |
| A5101966856 | test | W4404415725 | Learning about AI: A Systematic Review of Reviews on AI Literacy | Learning About AI: A Systematic Review of Reviews on AI Literacy |
| A5102531513 | test | W4383180643 | From Portfolio Optimization to Quantum Blockchain and Security: A Systematic Rev | From portfolio optimization to quantum blockchain and security: a systematic rev |
| A5107077385 | test | W4402706047 | Benchmarking foundation models as feature extractors for weakly-supervised compu | Benchmarking foundation models as feature extractors for weakly supervised compu |
| A5111756734 | test | W4405095387 | SIDA: Social Media Image Deepfake Detection, Localization and Explanation with L | SIDA: Social Media Image Deepfake Detection, Localization and Explanation with L |
| A5115740218 | test | W4405980344 | Security in the Internet of Things: Identifying and Analysing Critical Categorie | A Literature Review on Security in the Internet of Things: Identifying and Analy |
| A5015776510 | tuning | W4400488358 | RNNs, CNNs and Transformers in Human Action Recognition: A Survey and a Hybrid M | CNNs, RNNs and Transformers in human action recognition: a survey and a hybrid m |
| A5078765527 | tuning | W4402398497 | A Systematic Review of Intermediate Fusion in Multimodal Deep Learning for Biome | A systematic review of intermediate fusion in multimodal deep learning for biome |
| A5079533447 | tuning | W4313679638 | Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers | Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers |
| A5080582343 | tuning | W4405428531 | Gradient-Weighted Class Activation Mapping: A Systematic Literature Review | Grad-CAM (Gradient-weighted Class Activation Mapping): A systematic literature r |
