# Paper Feed — Track 1: Leave-Future-Out Empirical Evaluation Report

> [!NOTE]
> **Evaluation Run Configuration:**
> - Mode: Baseline
> - Candidate Pool Size (M): 100
> - Candidate Channels: SPECTER2 + BM25s + citation graph (V4)
> - Channel Depth (M_channel): 200
> - Popularity Prior Weight (w_pop): 0.00
> - Scoring Weights: w_pers=1.00, w_cite=1.00, w_facet=0.50, w_nov=0.00, lambda_div=0.00
> - System variant: V4 citation-graph channel (design §13.8; exploratory, plan 10.4d); tuning-split decision: keep Core
> - Hybrid-Persona baseline: gamma_rec=1.00, n_terms=5
> - Report: `track1_lfo_report_graph_channel.md` | Raw JSON: `track1_lfo_results_graph_channel.json` | Run manifest: `results/runs/track1_track1_lfo_report_graph_channel_20261113T192404Z/manifest.json`


## 1. Main Benchmark Results

| System Architecture | NDCG@10 | % of Oracle | Recall@10 | HitRate@10 | MRR | Topic Entropy | ILD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Feed (Full)** | 0.0264 | 2.6% | 0.0197 | 0.1750 | 0.0592 | 1.5141 | 0.0893 |
| Ablation: No-MMR | 0.0264 | 2.6% | 0.0197 | 0.1750 | 0.0592 | 1.5141 | 0.0893 |
| Ablation: No-Graph | 0.0111 | 1.1% | 0.0091 | 0.0875 | 0.0233 | 1.8302 | 0.0770 |
| Ablation: No-Facets | 0.0276 | 2.8% | 0.0214 | 0.1750 | 0.0661 | 1.4638 | 0.0867 |
| Baseline: popularity | 0.0129 | 1.3% | 0.0062 | 0.0656 | 0.0286 | 1.6884 | 0.1260 |
| Baseline: bm25_author | 0.0142 | 1.4% | 0.0094 | 0.0813 | 0.0279 | 1.6350 | 0.0993 |
| Baseline: dense_centroid | 0.0111 | 1.1% | 0.0085 | 0.0688 | 0.0224 | 1.7728 | 0.0691 |
| Baseline: citation_only | 0.0348 | 3.5% | 0.0273 | 0.1875 | 0.0711 | 1.6173 | 0.1092 |
| Baseline: hybrid_persona | 0.0128 | 1.3% | 0.0093 | 0.0813 | 0.0321 | 1.7977 | 0.0895 |
| Anchor: Random (100-draw) | 0.0001 | 0.0% | 0.0001 | 0.0013 | 0.0004 | 3.2613 | 0.2011 |

### Candidate Coverage Ceiling

| System | Candidate pool Recall@100 (ceiling) | Recall@10 | Share of ceiling reached |
| :--- | :---: | :---: | :---: |
| Paper Feed (Full) | 0.0901 | 0.0197 | 21.8% |
| Ablation: No-MMR | 0.0901 | 0.0197 | 21.8% |
| Ablation: No-Graph | 0.0510 | 0.0091 | 17.8% |
| Ablation: No-Facets | 0.0901 | 0.0214 | 23.7% |

*The pool fuses the text channels (BM25 + SPECTER2) and the V4 citation-graph channel by RRF; the No-Graph ablation drops the graph channel too, so its ceiling is the text-only pool. Targets outside the pool cannot be recommended whatever the scoring weights. For reference, Citation-only ranks the whole eligible corpus by the graph signal and reaches Recall@10 = 0.0273.*

### Diagnostic: Re-Citation

| Run | Ground truth | Recall@10 | NDCG@10 | HitRate@10 | Candidate Recall@100 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Diagnostic: Re-Citation | ReCitation (own past references cited again after T) | 0.0475 | 0.0392 | 0.1375 | 0.1475 |

*Citations(u, <= T) are removed from the exclusion set, so the feed may re-recommend the author's own past references. Different ground truth from every other table: not comparable with the Discovery numbers.*

## 2. Component Ablations

| Configuration | NDCG@10 | Delta NDCG | Recall@10 | Topic Entropy | Delta Entropy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Paper Feed (Full) | 0.0264 | +0.0000 | 0.0197 | 1.5141 | +0.0000 |
| Ablation: No-MMR | 0.0264 | +0.0000 | 0.0197 | 1.5141 | +0.0000 |
| Ablation: No-Graph | 0.0111 | -0.0152 | 0.0091 | 1.8302 | +0.3161 |
| Ablation: No-Facets | 0.0276 | +0.0012 | 0.0214 | 1.4638 | -0.0503 |

## 3. Statistical Hypothesis Testing

### Hypothesis Testing Summary

**Sample Size:** N = 320 test researchers | **Bootstrap Resamples:** B = 10,000

#### Hypothesis H1: Paper Feed Outperforms the Five Baselines (NDCG@10 and Recall@10)
**Verdict:** REJECTED / INCONCLUSIVE

| Comparison Baseline | Metric | System | Baseline | Rel. Gain | 95% CI of Difference | p (raw) | p (BH) | Cohen's d | MDE | Significant? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline: popularity | ndcg_at_10 | 0.0264 | 0.0129 | +103.57% | [+0.0037, +0.0235] | 8.4000e-03 | 1.2000e-02 | 0.148 | 0.0141 | YES (p < 0.05) |
| Baseline: bm25_author | ndcg_at_10 | 0.0264 | 0.0142 | +85.81% | [+0.0027, +0.0216] | 1.2500e-02 | 1.5625e-02 | 0.142 | 0.0134 | YES (p < 0.05) |
| Baseline: dense_centroid | ndcg_at_10 | 0.0264 | 0.0111 | +138.00% | [+0.0078, +0.0231] | 2.0000e-04 | 5.0000e-04 | 0.220 | 0.0108 | YES (p < 0.05) |
| Baseline: citation_only | ndcg_at_10 | 0.0264 | 0.0348 | -24.19% | [-0.0180, +0.0005] | 7.5000e-02 | 7.5000e-02 | -0.097 | 0.0133 | NO |
| Baseline: hybrid_persona | ndcg_at_10 | 0.0264 | 0.0128 | +105.66% | [+0.0065, +0.0212] | 5.0000e-04 | 1.0000e-03 | 0.201 | 0.0105 | YES (p < 0.05) |
| Baseline: popularity | recall_at_10 | 0.0197 | 0.0062 | +215.42% | [+0.0066, +0.0205] | 1.0000e-04 | 5.0000e-04 | 0.212 | 0.0099 | YES (p < 0.05) |
| Baseline: bm25_author | recall_at_10 | 0.0197 | 0.0094 | +108.75% | [+0.0040, +0.0164] | 8.0000e-04 | 1.3333e-03 | 0.185 | 0.0088 | YES (p < 0.05) |
| Baseline: dense_centroid | recall_at_10 | 0.0197 | 0.0085 | +130.94% | [+0.0064, +0.0164] | 1.0000e-04 | 5.0000e-04 | 0.247 | 0.0071 | YES (p < 0.05) |
| Baseline: citation_only | recall_at_10 | 0.0197 | 0.0273 | -27.99% | [-0.0148, -0.0007] | 3.5200e-02 | 3.9111e-02 | -0.117 | 0.0101 | YES (p < 0.05) |
| Baseline: hybrid_persona | recall_at_10 | 0.0197 | 0.0093 | +110.48% | [+0.0055, +0.0155] | 2.0000e-04 | 5.0000e-04 | 0.224 | 0.0072 | YES (p < 0.05) |

*BH correction runs over all baseline x metric comparisons. MDE: smallest true difference detectable with 80% power at alpha = 0.05 and this N; a non-significant row whose observed difference is below its MDE is inconclusive, not evidence of no effect.*

#### Hypothesis H2: MMR Diversity with Non-Inferiority
**Verdict:** NOT TESTABLE: the tuned lambda_div is 0, so the full system and the No-MMR ablation are identical and there is no MMR effect to measure.


## 4. History Slice Stratification

| Stratum | Authors (n) | Paper Feed NDCG@10 | Best Baseline NDCG@10 | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| Warm | 248 | 0.0234 | 0.0309 | -24.33% |
| Moderate | 37 | 0.0432 | 0.0401 | +7.65% |
| Sparse | 35 | 0.0298 | 0.0567 | -47.48% |

## 5. Item Cold-Start Slice

| System | Recall@10 fresh (n = 101 authors) | Recall@10 established (n = 318 authors) |
| :--- | :---: | :---: |
| Paper Feed (Full) | 0.0363 | 0.0153 |
| Ablation: No-MMR | 0.0363 | 0.0153 |
| Ablation: No-Graph | 0.0157 | 0.0073 |
| Ablation: No-Facets | 0.0272 | 0.0196 |
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
| **Paper Feed (Full)** | 0.0047 | 0.5% | 0.0018 | 0.0281 | 0.0137 | 1.9369 | 0.0833 |
| Baseline: popularity | 0.0223 | 2.2% | 0.0087 | 0.1094 | 0.0533 | 1.5729 | 0.1223 |
| Baseline: bm25_author | 0.0072 | 0.7% | 0.0046 | 0.0469 | 0.0180 | 1.7395 | 0.1086 |
| Baseline: dense_centroid | 0.0022 | 0.2% | 0.0012 | 0.0187 | 0.0044 | 1.9960 | 0.0817 |
| Baseline: hybrid_persona | 0.0044 | 0.4% | 0.0016 | 0.0250 | 0.0112 | 1.8310 | 0.0901 |
| Anchor: Random (100-draw) | 0.0001 | 0.0% | 0.0001 | 0.0014 | 0.0004 | 3.2584 | 0.2009 |

| Comparison Baseline | Metric | System | Baseline | Rel. Gain | 95% CI of Difference | p (raw) | p (BH) | Cohen's d | MDE | Significant? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline: popularity | ndcg_at_10 | 0.0047 | 0.0223 | -78.86% | [-0.0282, -0.0084] | 4.0000e-04 | 3.2000e-03 | -0.193 | 0.0141 | YES (p < 0.05) |
| Baseline: bm25_author | ndcg_at_10 | 0.0047 | 0.0072 | -34.12% | [-0.0070, +0.0019] | 2.6770e-01 | 3.5693e-01 | -0.061 | 0.0063 | NO |
| Baseline: dense_centroid | ndcg_at_10 | 0.0047 | 0.0022 | +116.94% | [+0.0006, +0.0051] | 2.6800e-02 | 7.1467e-02 | 0.123 | 0.0033 | NO |
| Baseline: hybrid_persona | ndcg_at_10 | 0.0047 | 0.0044 | +7.75% | [-0.0023, +0.0033] | 8.0470e-01 | 8.0470e-01 | 0.013 | 0.0040 | NO |
| Baseline: popularity | recall_at_10 | 0.0018 | 0.0087 | -79.64% | [-0.0108, -0.0034] | 9.0000e-04 | 3.6000e-03 | -0.203 | 0.0053 | YES (p < 0.05) |
| Baseline: bm25_author | recall_at_10 | 0.0018 | 0.0046 | -61.60% | [-0.0068, +0.0001] | 9.6100e-02 | 1.5376e-01 | -0.090 | 0.0049 | NO |
| Baseline: dense_centroid | recall_at_10 | 0.0018 | 0.0012 | +42.51% | [+0.0001, +0.0011] | 6.3400e-02 | 1.2680e-01 | 0.106 | 0.0008 | NO |
| Baseline: hybrid_persona | recall_at_10 | 0.0018 | 0.0016 | +11.97% | [-0.0012, +0.0016] | 7.7850e-01 | 8.0470e-01 | 0.015 | 0.0020 | NO |