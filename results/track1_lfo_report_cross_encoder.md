# Paper Feed — Track 1: Leave-Future-Out Empirical Evaluation Report

> [!NOTE]
> **Evaluation Run Configuration:**
> - Mode: Baseline
> - Candidate Pool Size (M): 100
> - Candidate Channels: SPECTER2 + BM25s
> - Reranker Feature (V2): cross-encoder/ms-marco-MiniLM-L-6-v2, w_ce=0.25
> - Channel Depth (M_channel): 200
> - Popularity Prior Weight (w_pop): 0.00
> - Scoring Weights: w_pers=1.00, w_cite=1.00, w_facet=0.25, w_nov=0.50, lambda_div=0.00
> - System variant: V2 cross-encoder feature, cross-encoder/ms-marco-MiniLM-L-6-v2 (design §13.8; exploratory, plan 10.2d); tuning-split decision: keep Core
> - Hybrid-Persona baseline: gamma_rec=1.00, n_terms=5
> - Report: `track1_lfo_report_cross_encoder.md` | Raw JSON: `track1_lfo_results_cross_encoder.json` | Run manifest: `results/runs/track1_track1_lfo_report_cross_encoder_20261113T205605Z/manifest.json`


## 1. Main Benchmark Results

| System Architecture | NDCG@10 | % of Oracle | Recall@10 | HitRate@10 | MRR | Topic Entropy | ILD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Feed (Full)** | 0.0205 | 2.1% | 0.0154 | 0.1219 | 0.0482 | 1.5072 | 0.0822 |
| Ablation: No-MMR | 0.0205 | 2.1% | 0.0154 | 0.1219 | 0.0482 | 1.5072 | 0.0822 |
| Ablation: No-Graph | 0.0105 | 1.1% | 0.0085 | 0.0719 | 0.0226 | 1.7214 | 0.0807 |
| Ablation: No-Facets | 0.0231 | 2.3% | 0.0174 | 0.1313 | 0.0553 | 1.4943 | 0.0801 |
| Baseline: popularity | 0.0129 | 1.3% | 0.0062 | 0.0656 | 0.0286 | 1.6884 | 0.1260 |
| Baseline: bm25_author | 0.0142 | 1.4% | 0.0094 | 0.0813 | 0.0279 | 1.6350 | 0.0993 |
| Baseline: dense_centroid | 0.0111 | 1.1% | 0.0085 | 0.0688 | 0.0224 | 1.7728 | 0.0691 |
| Baseline: citation_only | 0.0348 | 3.5% | 0.0273 | 0.1875 | 0.0711 | 1.6173 | 0.1092 |
| Baseline: hybrid_persona | 0.0128 | 1.3% | 0.0093 | 0.0813 | 0.0321 | 1.7977 | 0.0895 |
| Anchor: Random (100-draw) | 0.0001 | 0.0% | 0.0001 | 0.0013 | 0.0004 | 3.2613 | 0.2011 |

### Candidate Coverage Ceiling

| System | Candidate pool Recall@100 (ceiling) | Recall@10 | Share of ceiling reached |
| :--- | :---: | :---: | :---: |
| Paper Feed (Full) | 0.0510 | 0.0154 | 30.2% |
| Ablation: No-MMR | 0.0510 | 0.0154 | 30.2% |
| Ablation: No-Graph | 0.0510 | 0.0085 | 16.7% |
| Ablation: No-Facets | 0.0510 | 0.0174 | 34.0% |

*The pool comes from the text channels (BM25 + SPECTER2, RRF); targets outside it cannot be recommended whatever the scoring weights. For reference, Citation-only ranks the whole eligible corpus by the graph signal and reaches Recall@10 = 0.0273.*

### Diagnostic: Re-Citation

| Run | Ground truth | Recall@10 | NDCG@10 | HitRate@10 | Candidate Recall@100 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Diagnostic: Re-Citation | ReCitation (own past references cited again after T) | 0.0403 | 0.0347 | 0.1250 | 0.0781 |

*Citations(u, <= T) are removed from the exclusion set, so the feed may re-recommend the author's own past references. Different ground truth from every other table: not comparable with the Discovery numbers.*

## 2. Component Ablations

| Configuration | NDCG@10 | Delta NDCG | Recall@10 | Topic Entropy | Delta Entropy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Paper Feed (Full) | 0.0205 | +0.0000 | 0.0154 | 1.5072 | +0.0000 |
| Ablation: No-MMR | 0.0205 | +0.0000 | 0.0154 | 1.5072 | +0.0000 |
| Ablation: No-Graph | 0.0105 | -0.0100 | 0.0085 | 1.7214 | +0.2142 |
| Ablation: No-Facets | 0.0231 | +0.0026 | 0.0174 | 1.4943 | -0.0129 |

## 3. Statistical Hypothesis Testing

### Hypothesis Testing Summary

**Sample Size:** N = 320 test researchers | **Bootstrap Resamples:** B = 10,000

#### Hypothesis H1: Paper Feed Outperforms the Five Baselines (NDCG@10 and Recall@10)
**Verdict:** REJECTED / INCONCLUSIVE

| Comparison Baseline | Metric | System | Baseline | Rel. Gain | 95% CI of Difference | p (raw) | p (BH) | Cohen's d | MDE | Significant? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline: popularity | ndcg_at_10 | 0.0205 | 0.0129 | +58.46% | [-0.0020, +0.0172] | 1.2170e-01 | 1.3522e-01 | 0.086 | 0.0137 | NO |
| Baseline: bm25_author | ndcg_at_10 | 0.0205 | 0.0142 | +44.63% | [-0.0024, +0.0147] | 1.5080e-01 | 1.5080e-01 | 0.081 | 0.0123 | NO |
| Baseline: dense_centroid | ndcg_at_10 | 0.0205 | 0.0111 | +85.26% | [+0.0028, +0.0160] | 5.0000e-03 | 1.5571e-02 | 0.155 | 0.0095 | YES (p < 0.05) |
| Baseline: citation_only | ndcg_at_10 | 0.0205 | 0.0348 | -40.99% | [-0.0253, -0.0038] | 1.0900e-02 | 1.5571e-02 | -0.143 | 0.0154 | YES (p < 0.05) |
| Baseline: hybrid_persona | ndcg_at_10 | 0.0205 | 0.0128 | +60.08% | [+0.0018, +0.0137] | 9.7000e-03 | 1.5571e-02 | 0.143 | 0.0084 | YES (p < 0.05) |
| Baseline: popularity | recall_at_10 | 0.0154 | 0.0062 | +147.14% | [+0.0025, +0.0162] | 8.3000e-03 | 1.5571e-02 | 0.148 | 0.0097 | YES (p < 0.05) |
| Baseline: bm25_author | recall_at_10 | 0.0154 | 0.0094 | +63.57% | [+0.0002, +0.0120] | 4.6000e-02 | 5.7500e-02 | 0.112 | 0.0085 | NO |
| Baseline: dense_centroid | recall_at_10 | 0.0154 | 0.0085 | +80.95% | [+0.0020, +0.0122] | 9.1000e-03 | 1.5571e-02 | 0.149 | 0.0073 | YES (p < 0.05) |
| Baseline: citation_only | recall_at_10 | 0.0154 | 0.0273 | -43.58% | [-0.0199, -0.0041] | 2.2000e-03 | 1.5571e-02 | -0.166 | 0.0111 | YES (p < 0.05) |
| Baseline: hybrid_persona | recall_at_10 | 0.0154 | 0.0093 | +64.92% | [+0.0022, +0.0104] | 4.2000e-03 | 1.5571e-02 | 0.163 | 0.0059 | YES (p < 0.05) |

*BH correction runs over all baseline x metric comparisons. MDE: smallest true difference detectable with 80% power at alpha = 0.05 and this N; a non-significant row whose observed difference is below its MDE is inconclusive, not evidence of no effect.*

#### Hypothesis H2: MMR Diversity with Non-Inferiority
**Verdict:** NOT TESTABLE: the tuned lambda_div is 0, so the full system and the No-MMR ablation are identical and there is no MMR effect to measure.


## 4. History Slice Stratification

| Stratum | Authors (n) | Paper Feed NDCG@10 | Best Baseline NDCG@10 | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| Warm | 248 | 0.0151 | 0.0309 | -51.17% |
| Moderate | 37 | 0.0394 | 0.0401 | -1.80% |
| Sparse | 35 | 0.0391 | 0.0567 | -31.08% |

## 5. Item Cold-Start Slice

| System | Recall@10 fresh (n = 101 authors) | Recall@10 established (n = 318 authors) |
| :--- | :---: | :---: |
| Paper Feed (Full) | 0.0124 | 0.0165 |
| Ablation: No-MMR | 0.0124 | 0.0165 |
| Ablation: No-Graph | 0.0025 | 0.0059 |
| Ablation: No-Facets | 0.0223 | 0.0163 |
| Baseline: popularity | 0.0000 | 0.0077 |
| Baseline: bm25_author | 0.0437 | 0.0039 |
| Baseline: dense_centroid | 0.0173 | 0.0061 |
| Baseline: citation_only | 0.0404 | 0.0260 |
| Baseline: hybrid_persona | 0.0152 | 0.0062 |

*fresh: citation_count_T <= 1 and published in [T-1, T]; established: citation_count_T >= 20 and published since 2012 (design §12). Each class averages over the authors with at least one target in it.*

## 6. Zero-Publication Seed-Only Slice

Re-run of the test cohort with each profile replaced by the author's 3 seed topics (n = 320): E_u is empty and the ground truth is Ref(B) <= T minus Authored(u). Lexical and Dense baselines query with the seed topics, Popularity uses the seed topics' subfield, Citation-only is not applicable. Compare only within this table.

| System Architecture | NDCG@10 | % of Oracle | Recall@10 | HitRate@10 | MRR | Topic Entropy | ILD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Feed (Full)** | 0.0058 | 0.6% | 0.0021 | 0.0312 | 0.0139 | 1.8404 | 0.0865 |
| Baseline: popularity | 0.0223 | 2.2% | 0.0087 | 0.1094 | 0.0533 | 1.5729 | 0.1223 |
| Baseline: bm25_author | 0.0072 | 0.7% | 0.0046 | 0.0469 | 0.0180 | 1.7395 | 0.1086 |
| Baseline: dense_centroid | 0.0022 | 0.2% | 0.0012 | 0.0187 | 0.0044 | 1.9960 | 0.0817 |
| Baseline: hybrid_persona | 0.0044 | 0.4% | 0.0016 | 0.0250 | 0.0112 | 1.8310 | 0.0901 |
| Anchor: Random (100-draw) | 0.0001 | 0.0% | 0.0001 | 0.0014 | 0.0004 | 3.2584 | 0.2009 |

| Comparison Baseline | Metric | System | Baseline | Rel. Gain | 95% CI of Difference | p (raw) | p (BH) | Cohen's d | MDE | Significant? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline: popularity | ndcg_at_10 | 0.0058 | 0.0223 | -74.16% | [-0.0273, -0.0070] | 2.0000e-03 | 8.0000e-03 | -0.175 | 0.0146 | YES (p < 0.05) |
| Baseline: bm25_author | ndcg_at_10 | 0.0058 | 0.0072 | -19.46% | [-0.0066, +0.0036] | 5.8670e-01 | 5.8670e-01 | -0.030 | 0.0073 | NO |
| Baseline: dense_centroid | ndcg_at_10 | 0.0058 | 0.0022 | +165.19% | [+0.0012, +0.0065] | 9.7000e-03 | 2.5867e-02 | 0.146 | 0.0038 | YES (p < 0.05) |
| Baseline: hybrid_persona | ndcg_at_10 | 0.0058 | 0.0044 | +31.71% | [-0.0014, +0.0044] | 3.3580e-01 | 4.4773e-01 | 0.053 | 0.0041 | NO |
| Baseline: popularity | recall_at_10 | 0.0021 | 0.0087 | -76.04% | [-0.0106, -0.0030] | 1.3000e-03 | 8.0000e-03 | -0.190 | 0.0054 | YES (p < 0.05) |
| Baseline: bm25_author | recall_at_10 | 0.0021 | 0.0046 | -54.81% | [-0.0065, +0.0005] | 1.4480e-01 | 2.3168e-01 | -0.079 | 0.0050 | NO |
| Baseline: dense_centroid | recall_at_10 | 0.0021 | 0.0012 | +67.68% | [-0.0003, +0.0019] | 1.3630e-01 | 2.3168e-01 | 0.082 | 0.0016 | NO |
| Baseline: hybrid_persona | recall_at_10 | 0.0021 | 0.0016 | +31.75% | [-0.0009, +0.0020] | 4.8970e-01 | 5.5966e-01 | 0.038 | 0.0021 | NO |