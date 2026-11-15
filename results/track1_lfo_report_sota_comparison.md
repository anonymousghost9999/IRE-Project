# Paper Feed — Track 1: Leave-Future-Out Empirical Evaluation Report

> [!NOTE]
> **Evaluation Run Configuration:**
> - Mode: Baseline
> - Candidate Pool Size (M): 100
> - Candidate Channels: SPECTER2 + BM25s
> - Channel Depth (M_channel): 200
> - Popularity Prior Weight (w_pop): 0.00
> - Scoring Weights: w_pers=1.00, w_cite=1.00, w_facet=0.25, w_nov=0.25, lambda_div=0.50
> - System variant: frozen best configuration (configs/best.yaml: Core) with the Phase 11 comparison systems
> - Hybrid-Persona baseline: gamma_rec=1.00, n_terms=5
> - Report: `track1_lfo_report_sota_comparison.md` | Raw JSON: `track1_lfo_results_sota_comparison.json` | Run manifest: `results/runs/track1_track1_lfo_report_sota_comparison_20261114T201106Z/manifest.json`


## 1. Main Benchmark Results

| System Architecture | NDCG@10 | % of Oracle | Recall@10 | HitRate@10 | MRR | Topic Entropy | ILD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Feed (Full)** | 0.0216 | 2.2% | 0.0156 | 0.1281 | 0.0557 | 1.5405 | 0.0808 |
| Ablation: No-MMR | 0.0212 | 2.1% | 0.0150 | 0.1250 | 0.0553 | 1.5127 | 0.0797 |
| Ablation: No-Graph | 0.0109 | 1.1% | 0.0082 | 0.0719 | 0.0240 | 1.8414 | 0.0766 |
| Ablation: No-Facets | 0.0234 | 2.3% | 0.0182 | 0.1437 | 0.0584 | 1.5600 | 0.0800 |
| Baseline: popularity | 0.0129 | 1.3% | 0.0062 | 0.0656 | 0.0286 | 1.6884 | 0.1260 |
| Baseline: bm25_author | 0.0142 | 1.4% | 0.0094 | 0.0813 | 0.0279 | 1.6350 | 0.0993 |
| Baseline: dense_centroid | 0.0111 | 1.1% | 0.0085 | 0.0688 | 0.0224 | 1.7728 | 0.0691 |
| Baseline: citation_only | 0.0348 | 3.5% | 0.0273 | 0.1875 | 0.0711 | 1.6173 | 0.1092 |
| Baseline: hybrid_persona | 0.0128 | 1.3% | 0.0093 | 0.0813 | 0.0321 | 1.7977 | 0.0895 |
| SOTA: RP3beta | 0.0607 | 6.1% | 0.0386 | 0.2844 | 0.1262 | 1.6363 | 0.1148 |
| SOTA: Scholar Inbox (GTE-large + LR) | 0.0174 | 1.7% | 0.0119 | 0.1125 | 0.0392 | 1.1438 | 0.0816 |
| SOTA: SciNCL centroid kNN | 0.0123 | 1.2% | 0.0091 | 0.0875 | 0.0319 | 1.5944 | 0.0850 |
| SOTA: RankGPT (Qwen2.5-3B) | 0.0054 | 0.5% | 0.0042 | 0.0500 | 0.0095 | 2.3458 | 0.1169 |
| Anchor: Random (100-draw) | 0.0001 | 0.0% | 0.0001 | 0.0013 | 0.0004 | 3.2613 | 0.2011 |

### Candidate Coverage Ceiling

| System | Candidate pool Recall@100 (ceiling) | Recall@10 | Share of ceiling reached |
| :--- | :---: | :---: | :---: |
| Paper Feed (Full) | 0.0510 | 0.0156 | 30.6% |
| Ablation: No-MMR | 0.0510 | 0.0150 | 29.4% |
| Ablation: No-Graph | 0.0510 | 0.0082 | 16.1% |
| Ablation: No-Facets | 0.0510 | 0.0182 | 35.7% |

*The pool comes from the text channels (BM25 + SPECTER2, RRF); targets outside it cannot be recommended whatever the scoring weights. For reference, Citation-only ranks the whole eligible corpus by the graph signal and reaches Recall@10 = 0.0273.*

### Diagnostic: Re-Citation

| Run | Ground truth | Recall@10 | NDCG@10 | HitRate@10 | Candidate Recall@100 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Diagnostic: Re-Citation | ReCitation (own past references cited again after T) | 0.0415 | 0.0363 | 0.1281 | 0.0781 |

*Citations(u, <= T) are removed from the exclusion set, so the feed may re-recommend the author's own past references. Different ground truth from every other table: not comparable with the Discovery numbers.*

## 2. Component Ablations

| Configuration | NDCG@10 | Delta NDCG | Recall@10 | Topic Entropy | Delta Entropy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Paper Feed (Full) | 0.0216 | +0.0000 | 0.0156 | 1.5405 | +0.0000 |
| Ablation: No-MMR | 0.0212 | -0.0004 | 0.0150 | 1.5127 | -0.0278 |
| Ablation: No-Graph | 0.0109 | -0.0108 | 0.0082 | 1.8414 | +0.3009 |
| Ablation: No-Facets | 0.0234 | +0.0018 | 0.0182 | 1.5600 | +0.0195 |

## 3. Statistical Hypothesis Testing

### Hypothesis Testing Summary

**Sample Size:** N = 320 test researchers | **Bootstrap Resamples:** B = 10,000

#### Hypothesis H1: Paper Feed Outperforms the Five Baselines (NDCG@10 and Recall@10)
**Verdict:** REJECTED / INCONCLUSIVE

| Comparison Baseline | Metric | System | Baseline | Rel. Gain | 95% CI of Difference | p (raw) | p (BH) | Cohen's d | MDE | Significant? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline: popularity | ndcg_at_10 | 0.0216 | 0.0129 | +66.94% | [-0.0011, +0.0185] | 8.2600e-02 | 9.1778e-02 | 0.097 | 0.0140 | NO |
| Baseline: bm25_author | ndcg_at_10 | 0.0216 | 0.0142 | +52.38% | [-0.0015, +0.0163] | 1.0230e-01 | 1.0230e-01 | 0.092 | 0.0127 | NO |
| Baseline: dense_centroid | ndcg_at_10 | 0.0216 | 0.0111 | +95.18% | [+0.0038, +0.0175] | 2.6000e-03 | 9.0000e-03 | 0.171 | 0.0097 | YES (p < 0.05) |
| Baseline: citation_only | ndcg_at_10 | 0.0216 | 0.0348 | -37.83% | [-0.0244, -0.0024] | 1.7200e-02 | 2.4571e-02 | -0.131 | 0.0155 | YES (p < 0.05) |
| Baseline: hybrid_persona | ndcg_at_10 | 0.0216 | 0.0128 | +68.66% | [+0.0027, +0.0153] | 5.8000e-03 | 9.6667e-03 | 0.153 | 0.0090 | YES (p < 0.05) |
| Baseline: popularity | recall_at_10 | 0.0156 | 0.0062 | +150.63% | [+0.0029, +0.0161] | 5.8000e-03 | 9.6667e-03 | 0.156 | 0.0094 | YES (p < 0.05) |
| Baseline: bm25_author | recall_at_10 | 0.0156 | 0.0094 | +65.87% | [+0.0004, +0.0121] | 3.6900e-02 | 4.6125e-02 | 0.116 | 0.0084 | YES (p < 0.05) |
| Baseline: dense_centroid | recall_at_10 | 0.0156 | 0.0085 | +83.50% | [+0.0027, +0.0120] | 3.8000e-03 | 9.5000e-03 | 0.170 | 0.0066 | YES (p < 0.05) |
| Baseline: citation_only | recall_at_10 | 0.0156 | 0.0273 | -42.78% | [-0.0196, -0.0042] | 2.2000e-03 | 9.0000e-03 | -0.165 | 0.0110 | YES (p < 0.05) |
| Baseline: hybrid_persona | recall_at_10 | 0.0156 | 0.0093 | +67.25% | [+0.0024, +0.0107] | 2.7000e-03 | 9.0000e-03 | 0.167 | 0.0059 | YES (p < 0.05) |

*BH correction runs over all baseline x metric comparisons. MDE: smallest true difference detectable with 80% power at alpha = 0.05 and this N; a non-significant row whose observed difference is below its MDE is inconclusive, not evidence of no effect.*

#### Hypothesis H2: MMR Diversity with Non-Inferiority
**Verdict:** ACCEPTED (Supported by empirical data)

- **Entropy Increase (one-sided):** Delta = +0.0278 (+1.84%), 95% CI = [+0.0168, +0.0399], p = 1.0000e-04, MDE = 0.0164 (Significant: True)
- **NDCG Retention (non-inferiority: relative loss <= 3%):** NDCG@10 change with MMR = +2.10% (Delta = +0.0004, 95% CI = [-0.0002, +0.0013]); worst-case relative loss from the CI = 0.91%, MDE = 0.0011 (Non-inferior: True)

## 4. History Slice Stratification

| Stratum | Authors (n) | Paper Feed NDCG@10 | Best Baseline NDCG@10 | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| Warm | 248 | 0.0171 | 0.0309 | -44.56% |
| Moderate | 37 | 0.0382 | 0.0401 | -4.87% |
| Sparse | 35 | 0.0360 | 0.0567 | -36.53% |

## 5. Item Cold-Start Slice

| System | Recall@10 fresh (n = 101 authors) | Recall@10 established (n = 318 authors) |
| :--- | :---: | :---: |
| Paper Feed (Full) | 0.0223 | 0.0152 |
| Ablation: No-MMR | 0.0223 | 0.0147 |
| Ablation: No-Graph | 0.0000 | 0.0065 |
| Ablation: No-Facets | 0.0223 | 0.0174 |
| Baseline: popularity | 0.0000 | 0.0077 |
| Baseline: bm25_author | 0.0437 | 0.0039 |
| Baseline: dense_centroid | 0.0173 | 0.0061 |
| Baseline: citation_only | 0.0404 | 0.0260 |
| Baseline: hybrid_persona | 0.0152 | 0.0062 |
| SOTA: RP3beta | 0.0000 | 0.0457 |
| SOTA: Scholar Inbox (GTE-large + LR) | 0.0193 | 0.0106 |
| SOTA: SciNCL centroid kNN | 0.0000 | 0.0079 |
| SOTA: RankGPT (Qwen2.5-3B) | 0.0000 | 0.0080 |

*fresh: citation_count_T <= 1 and published in [T-1, T]; established: citation_count_T >= 20 and published since 2012 (design §12). Each class averages over the authors with at least one target in it.*

## 6. Phase 11: Comparison with SOTA Paper Recommenders

Plan 11.3-11.5 / design §13.9. The best configuration (`configs/best.yaml`) is Core, so this run compares Core with the comparison systems fixed by the literature review of 2026-11-14 (design §13.9 Phase 11 specification). Every system uses the same eligible papers, exclusion set E_u and test authors; each was tuned on the tuning split with a grid fixed in advance. The test split was already used once for the Phase 10 variant table, so this run is confirmatory. Published numbers from other datasets are context only.

| System | NDCG@10 | % of Oracle | Recall@10 | HitRate@10 | MRR | Topic Entropy | Explanation Coverage | Latency per feed (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Feed (Full)** | 0.0216 | 2.2% | 0.0156 | 0.1281 | 0.0557 | 1.5405 | 100.0% | 0.12 |
| SOTA: RP3beta | 0.0607 | 6.1% | 0.0386 | 0.2844 | 0.1262 | 1.6363 | none | 0.87 |
| SOTA: Scholar Inbox (GTE-large + LR) | 0.0174 | 1.7% | 0.0119 | 0.1125 | 0.0392 | 1.1438 | none | 0.21 |
| SOTA: SciNCL centroid kNN | 0.0123 | 1.2% | 0.0091 | 0.0875 | 0.0319 | 1.5944 | none | 0.08 |
| SOTA: RankGPT (Qwen2.5-3B) | 0.0054 | 0.5% | 0.0042 | 0.0500 | 0.0095 | 2.3458 | none | 45.94 |
| Baseline: popularity | 0.0129 | 1.3% | 0.0062 | 0.0656 | 0.0286 | 1.6884 | none | - |
| Baseline: bm25_author | 0.0142 | 1.4% | 0.0094 | 0.0813 | 0.0279 | 1.6350 | none | - |
| Baseline: dense_centroid | 0.0111 | 1.1% | 0.0085 | 0.0688 | 0.0224 | 1.7728 | none | - |
| Baseline: citation_only | 0.0348 | 3.5% | 0.0273 | 0.1875 | 0.0711 | 1.6173 | none | - |
| Baseline: hybrid_persona | 0.0128 | 1.3% | 0.0093 | 0.0813 | 0.0321 | 1.7977 | none | - |
| Anchor: Random (100-draw) | 0.0001 | 0.0% | 0.0001 | 0.0013 | 0.0004 | 3.2613 | none | - |

*Explanation Coverage is a capability difference, not ranked against accuracy: only Paper Feed explains its recommendations. Latency is the mean time to produce one feed on the run machine; baselines and the anchor were not timed ("-"). The comparison systems' one-off setup (graph, PCA, model loading) is excluded.*

Paper Feed against each comparison system (positive = Paper Feed better; BH over this family of 8 comparisons):

| Comparison Baseline | Metric | System | Baseline | Rel. Gain | 95% CI of Difference | p (raw) | p (BH) | Cohen's d | MDE | Significant? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SOTA: RP3beta | ndcg_at_10 | 0.0216 | 0.0607 | -64.38% | [-0.0553, -0.0238] | 1.0000e-04 | 2.0000e-04 | -0.272 | 0.0223 | YES (p < 0.05) |
| SOTA: Scholar Inbox (GTE-large + LR) | ndcg_at_10 | 0.0216 | 0.0174 | +24.04% | [-0.0046, +0.0126] | 3.4130e-01 | 3.4130e-01 | 0.053 | 0.0123 | NO |
| SOTA: SciNCL centroid kNN | ndcg_at_10 | 0.0216 | 0.0123 | +76.38% | [+0.0024, +0.0167] | 9.6000e-03 | 1.2800e-02 | 0.144 | 0.0102 | YES (p < 0.05) |
| SOTA: RankGPT (Qwen2.5-3B) | ndcg_at_10 | 0.0216 | 0.0054 | +300.46% | [+0.0093, +0.0241] | 1.0000e-04 | 2.0000e-04 | 0.241 | 0.0106 | YES (p < 0.05) |
| SOTA: RP3beta | recall_at_10 | 0.0156 | 0.0386 | -59.46% | [-0.0341, -0.0127] | 1.0000e-04 | 2.0000e-04 | -0.234 | 0.0151 | YES (p < 0.05) |
| SOTA: Scholar Inbox (GTE-large + LR) | recall_at_10 | 0.0156 | 0.0119 | +31.43% | [-0.0012, +0.0089] | 1.4250e-01 | 1.6286e-01 | 0.082 | 0.0072 | NO |
| SOTA: SciNCL centroid kNN | recall_at_10 | 0.0156 | 0.0091 | +71.45% | [+0.0019, +0.0114] | 7.3000e-03 | 1.1680e-02 | 0.151 | 0.0068 | YES (p < 0.05) |
| SOTA: RankGPT (Qwen2.5-3B) | recall_at_10 | 0.0156 | 0.0042 | +274.40% | [+0.0065, +0.0170] | 1.0000e-04 | 2.0000e-04 | 0.240 | 0.0075 | YES (p < 0.05) |

Comparison systems and their frozen settings:

- **RP3beta:** {"alpha": 0.5, "beta": 0.25, "tuning_ndcg": 0.06614082494485748, "graph_papers": 3415898, "graph_edges": 7622860}
- **Scholar Inbox (GTE-large + LR):** {"C": 0.1, "tuning_ndcg": 0.023224770699195835, "pca_explained_variance": 0.8019320964813232, "model": "thenlper/gte-large"}
- **SciNCL centroid kNN:** {"model": "malteos/scincl", "n_papers": 111149}
- **RankGPT (Qwen2.5-3B):** {"model": "Qwen/Qwen2.5-1.5B-Instruct", "precision": "torch.float16", "window": 20, "step": 10}

RankGPT: 2880 windows, 0 unparsable (kept their input order).

Excluded (design §13.9 Phase 11 spec point 5): LightGCN-style graph neural recommenders (compute and a collaborative user base; RP3beta represents graph recommenders); hosted feeds such as Semantic Scholar, Google Scholar and Scholar Inbox itself (cannot be restricted to <= T); SPECTER2 user-centroid kNN is the Dense Centroid baseline. The zero-publication slice below is Core only: the comparison systems need authored papers.

## 7. Phase 11: Best Configuration

Best configuration: **core**. An option replaces the Core component only if its tuning-split NDCG@10 gain has a 95% bootstrap CI above zero (design §13.9). Test-split results are never used. Not needed: no component was replaced, so the best configuration is Core.

| Part | Chosen option | Replaces Core | Tuning evidence | n | Rationale |
| :--- | :--- | :---: | :--- | :---: | :--- |
| dense_encoder | `specter2` | no | gain +0.0105 [-0.0019, +0.0237] | 80 | V1 candidate's gain CI includes zero; SPECTER2 is kept. |
| candidate_channels | `['dense', 'bm25s']` | no | gain +0.0061 [-0.0033, +0.0168] | 80 | V4 citation channel's gain CI includes zero; text channels only. |
| pool_size | `{'pool_size': 100, 'channel_depth': 200}` | no | gain +0.0004 [-0.0050, +0.0060] | 80 | Larger pool's gain CI includes zero; M = 100, M_channel = 200 kept. |
| feed_reranking | `None` | no | gain +0.0021 [-0.0006, +0.0056] | 80 | V2 cross-encoder's gain CI includes zero; no reranker. |
| scorer | `linear` | no | gain +0.0022 [-0.0135, +0.0150] | 80 | V3 out-of-fold gain CI includes zero; the linear scorer is kept. |
| diversification | `{'lambda_div': 0.5}` | no | NDCG@10 0.0241 | 80 | lambda_div is part of the §9 grid tuned on the tuning split. |
| session_model | `{'alpha_sess': 0.5, 'rho': 0.3, 'use_pivot_gate': False, 'gamma_pivot': 0.35}` | no | none | - | No evidence: Track 2 (Phase 9) is deferred, so the design defaults are kept. Track 1 feeds use an empty session, so these values do not affect any Track 1 or Phase 11 number. |

## 8. Zero-Publication Seed-Only Slice

Re-run of the test cohort with each profile replaced by the author's 3 seed topics (n = 320): E_u is empty and the ground truth is Ref(B) <= T minus Authored(u). Lexical and Dense baselines query with the seed topics, Popularity uses the seed topics' subfield, Citation-only is not applicable. Compare only within this table.

| System Architecture | NDCG@10 | % of Oracle | Recall@10 | HitRate@10 | MRR | Topic Entropy | ILD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Feed (Full)** | 0.0048 | 0.5% | 0.0020 | 0.0312 | 0.0094 | 1.9509 | 0.0853 |
| Baseline: popularity | 0.0223 | 2.2% | 0.0087 | 0.1094 | 0.0533 | 1.5729 | 0.1223 |
| Baseline: bm25_author | 0.0072 | 0.7% | 0.0046 | 0.0469 | 0.0180 | 1.7395 | 0.1086 |
| Baseline: dense_centroid | 0.0022 | 0.2% | 0.0012 | 0.0187 | 0.0044 | 1.9960 | 0.0817 |
| Baseline: hybrid_persona | 0.0043 | 0.4% | 0.0016 | 0.0250 | 0.0112 | 1.8310 | 0.0901 |
| Anchor: Random (100-draw) | 0.0001 | 0.0% | 0.0001 | 0.0014 | 0.0004 | 3.2584 | 0.2009 |

| Comparison Baseline | Metric | System | Baseline | Rel. Gain | 95% CI of Difference | p (raw) | p (BH) | Cohen's d | MDE | Significant? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline: popularity | ndcg_at_10 | 0.0048 | 0.0223 | -78.33% | [-0.0281, -0.0083] | 7.0000e-04 | 4.8000e-03 | -0.190 | 0.0142 | YES (p < 0.05) |
| Baseline: bm25_author | ndcg_at_10 | 0.0048 | 0.0072 | -32.46% | [-0.0070, +0.0021] | 3.0490e-01 | 4.0653e-01 | -0.056 | 0.0065 | NO |
| Baseline: dense_centroid | ndcg_at_10 | 0.0048 | 0.0022 | +122.38% | [+0.0008, +0.0049] | 1.2300e-02 | 3.2800e-02 | 0.141 | 0.0030 | YES (p < 0.05) |
| Baseline: hybrid_persona | ndcg_at_10 | 0.0048 | 0.0043 | +11.21% | [-0.0018, +0.0032] | 7.0070e-01 | 7.0070e-01 | 0.022 | 0.0035 | NO |
| Baseline: popularity | recall_at_10 | 0.0020 | 0.0087 | -77.24% | [-0.0106, -0.0032] | 1.2000e-03 | 4.8000e-03 | -0.194 | 0.0053 | YES (p < 0.05) |
| Baseline: bm25_author | recall_at_10 | 0.0020 | 0.0046 | -57.07% | [-0.0066, +0.0004] | 1.2610e-01 | 2.5220e-01 | -0.083 | 0.0049 | NO |
| Baseline: dense_centroid | recall_at_10 | 0.0020 | 0.0012 | +59.31% | [-0.0004, +0.0018] | 1.7500e-01 | 2.8000e-01 | 0.074 | 0.0016 | NO |
| Baseline: hybrid_persona | recall_at_10 | 0.0020 | 0.0016 | +25.17% | [-0.0010, +0.0018] | 5.6470e-01 | 6.4537e-01 | 0.031 | 0.0020 | NO |