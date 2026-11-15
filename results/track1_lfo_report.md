# Paper Feed — Track 1: Leave-Future-Out Empirical Evaluation Report

> [!NOTE]
> **Evaluation Run Configuration:**
> - Mode: Baseline
> - Candidate Pool Size (M): 100
> - Candidate Channels: SPECTER2 + BM25s
> - Channel Depth (M_channel): 200
> - Popularity Prior Weight (w_pop): 0.00
> - Scoring Weights: w_pers=1.00, w_cite=0.50, w_facet=0.50, w_nov=0.50, lambda_div=0.20
> - System variant: grid optimum (headline)
> - Hybrid-Persona baseline: gamma_rec=0.90, n_terms=10
> - Report: `track1_lfo_report.md` | Raw JSON: `track1_lfo_results.json` | Run manifest: `results/runs/track1_track1_lfo_report_20261115T103423Z/manifest.json`


## 1. Main Benchmark Results

| System Architecture | NDCG@10 | % of Oracle | Recall@10 | HitRate@10 | MRR | Topic Entropy | ILD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Feed (Full)** | 0.0150 | 1.5% | 0.0112 | 0.0969 | 0.0387 | 1.7057 | 0.0822 |
| Ablation: No-MMR | 0.0151 | 1.5% | 0.0112 | 0.1031 | 0.0391 | 1.6951 | 0.0816 |
| Ablation: No-Graph | 0.0116 | 1.2% | 0.0084 | 0.0813 | 0.0264 | 1.8940 | 0.0820 |
| Ablation: No-Facets | 0.0168 | 1.7% | 0.0151 | 0.1250 | 0.0365 | 1.6391 | 0.0755 |
| Baseline: popularity | 0.0107 | 1.1% | 0.0048 | 0.0625 | 0.0227 | 1.7269 | 0.1266 |
| Baseline: bm25_author | 0.0107 | 1.1% | 0.0087 | 0.0719 | 0.0225 | 1.6858 | 0.1003 |
| Baseline: dense_centroid | 0.0094 | 0.9% | 0.0073 | 0.0625 | 0.0206 | 1.8372 | 0.0695 |
| Baseline: citation_only | 0.0346 | 3.5% | 0.0265 | 0.1844 | 0.0723 | 1.6466 | 0.1087 |
| Baseline: hybrid_persona | 0.0103 | 1.0% | 0.0076 | 0.0688 | 0.0253 | 1.7367 | 0.0863 |
| Anchor: Random (100-draw) | 0.0002 | 0.0% | 0.0001 | 0.0015 | 0.0005 | 3.2612 | 0.2010 |

### Candidate Coverage Ceiling

| System | Candidate pool Recall@100 (ceiling) | Recall@10 | Share of ceiling reached |
| :--- | :---: | :---: | :---: |
| Paper Feed (Full) | 0.0471 | 0.0112 | 23.7% |
| Ablation: No-MMR | 0.0471 | 0.0112 | 23.7% |
| Ablation: No-Graph | 0.0471 | 0.0084 | 17.9% |
| Ablation: No-Facets | 0.0471 | 0.0151 | 32.1% |

*The pool comes from the text channels (BM25 + SPECTER2, RRF); targets outside it cannot be recommended whatever the scoring weights. For reference, Citation-only ranks the whole eligible corpus by the graph signal and reaches Recall@10 = 0.0265.*

### Diagnostic: Re-Citation

| Run | Ground truth | Recall@10 | NDCG@10 | HitRate@10 | Candidate Recall@100 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Diagnostic: Re-Citation | ReCitation (own past references cited again after T) | 0.0256 | 0.0204 | 0.0813 | 0.0733 |

*Citations(u, <= T) are removed from the exclusion set, so the feed may re-recommend the author's own past references. Different ground truth from every other table: not comparable with the Discovery numbers.*

## 2. Component Ablations

| Configuration | NDCG@10 | Delta NDCG | Recall@10 | Topic Entropy | Delta Entropy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Paper Feed (Full) | 0.0150 | +0.0000 | 0.0112 | 1.7057 | +0.0000 |
| Ablation: No-MMR | 0.0151 | +0.0001 | 0.0112 | 1.6951 | -0.0105 |
| Ablation: No-Graph | 0.0116 | -0.0034 | 0.0084 | 1.8940 | +0.1884 |
| Ablation: No-Facets | 0.0168 | +0.0018 | 0.0151 | 1.6391 | -0.0665 |

## 3. Statistical Hypothesis Testing

### Hypothesis Testing Summary

**Sample Size:** N = 320 test researchers | **Bootstrap Resamples:** B = 10,000

#### Hypothesis H1: Paper Feed Outperforms the Five Baselines (NDCG@10 and Recall@10)
**Verdict:** REJECTED / INCONCLUSIVE

| Comparison Baseline | Metric | System | Baseline | Rel. Gain | 95% CI of Difference | p (raw) | p (BH) | Cohen's d | MDE | Significant? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline: popularity | ndcg_at_10 | 0.0150 | 0.0107 | +39.52% | [-0.0035, +0.0119] | 2.6930e-01 | 2.8600e-01 | 0.061 | 0.0109 | NO |
| Baseline: bm25_author | ndcg_at_10 | 0.0150 | 0.0107 | +39.96% | [-0.0011, +0.0098] | 1.2410e-01 | 1.5513e-01 | 0.086 | 0.0078 | NO |
| Baseline: dense_centroid | ndcg_at_10 | 0.0150 | 0.0094 | +58.72% | [+0.0011, +0.0102] | 1.7500e-02 | 5.1250e-02 | 0.133 | 0.0065 | NO |
| Baseline: citation_only | ndcg_at_10 | 0.0150 | 0.0346 | -56.64% | [-0.0306, -0.0091] | 8.0000e-04 | 4.0000e-03 | -0.199 | 0.0154 | YES (p < 0.05) |
| Baseline: hybrid_persona | ndcg_at_10 | 0.0150 | 0.0103 | +45.46% | [+0.0003, +0.0091] | 3.9000e-02 | 6.5000e-02 | 0.115 | 0.0064 | NO |
| Baseline: popularity | recall_at_10 | 0.0112 | 0.0048 | +133.82% | [+0.0013, +0.0122] | 2.0500e-02 | 5.1250e-02 | 0.130 | 0.0077 | NO |
| Baseline: bm25_author | recall_at_10 | 0.0112 | 0.0087 | +28.01% | [-0.0020, +0.0071] | 2.8600e-01 | 2.8600e-01 | 0.059 | 0.0065 | NO |
| Baseline: dense_centroid | recall_at_10 | 0.0112 | 0.0073 | +53.67% | [-0.0001, +0.0085] | 7.5000e-02 | 1.0714e-01 | 0.099 | 0.0062 | NO |
| Baseline: citation_only | recall_at_10 | 0.0112 | 0.0265 | -57.79% | [-0.0228, -0.0079] | 1.0000e-04 | 1.0000e-03 | -0.220 | 0.0107 | YES (p < 0.05) |
| Baseline: hybrid_persona | recall_at_10 | 0.0112 | 0.0076 | +46.91% | [+0.0005, +0.0071] | 3.4600e-02 | 6.5000e-02 | 0.116 | 0.0048 | NO |

*BH correction runs over all baseline x metric comparisons. MDE: smallest true difference detectable with 80% power at alpha = 0.05 and this N; a non-significant row whose observed difference is below its MDE is inconclusive, not evidence of no effect.*

#### Hypothesis H2: MMR Diversity with Non-Inferiority
**Verdict:** REJECTED / INCONCLUSIVE

- **Entropy Increase (one-sided):** Delta = +0.0105 (+0.62%), 95% CI = [+0.0011, +0.0199], p = 1.3900e-02, MDE = 0.0135 (Significant: True)
- **NDCG Retention (non-inferiority: relative loss <= 3%):** NDCG@10 change with MMR = -0.82% (Delta = -0.0001, 95% CI = [-0.0008, +0.0005]); worst-case relative loss from the CI = 5.26%, MDE = 0.0010 (Non-inferior: False)

## 4. History Slice Stratification

| Stratum | Authors (n) | Paper Feed NDCG@10 | Best Baseline NDCG@10 | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| Warm | 254 | 0.0119 | 0.0285 | -58.27% |
| Moderate | 33 | 0.0228 | 0.0461 | -50.60% |
| Sparse | 33 | 0.0311 | 0.0699 | -55.51% |

## 5. Item Cold-Start Slice

| System | Recall@10 fresh (n = 101 authors) | Recall@10 established (n = 317 authors) |
| :--- | :---: | :---: |
| Paper Feed (Full) | 0.0198 | 0.0078 |
| Ablation: No-MMR | 0.0198 | 0.0077 |
| Ablation: No-Graph | 0.0231 | 0.0053 |
| Ablation: No-Facets | 0.0231 | 0.0125 |
| Baseline: popularity | 0.0000 | 0.0058 |
| Baseline: bm25_author | 0.0347 | 0.0035 |
| Baseline: dense_centroid | 0.0116 | 0.0049 |
| Baseline: citation_only | 0.0413 | 0.0270 |
| Baseline: hybrid_persona | 0.0135 | 0.0063 |

*fresh: citation_count_T <= 1 and published in [T-1, T]; established: citation_count_T >= 20 and published since 2012 (design §12). Each class averages over the authors with at least one target in it.*

## 6. Zero-Publication Seed-Only Slice

Re-run of the test cohort with each profile replaced by the author's 3 seed topics (n = 320): E_u is empty and the ground truth is Ref(B) <= T minus Authored(u). Lexical and Dense baselines query with the seed topics, Popularity uses the seed topics' subfield, Citation-only is not applicable. Compare only within this table.

| System Architecture | NDCG@10 | % of Oracle | Recall@10 | HitRate@10 | MRR | Topic Entropy | ILD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Feed (Full)** | 0.0052 | 0.5% | 0.0019 | 0.0344 | 0.0144 | 1.9302 | 0.0880 |
| Baseline: popularity | 0.0158 | 1.6% | 0.0064 | 0.0875 | 0.0385 | 1.6360 | 0.1238 |
| Baseline: bm25_author | 0.0053 | 0.5% | 0.0039 | 0.0344 | 0.0130 | 1.7064 | 0.1077 |
| Baseline: dense_centroid | 0.0017 | 0.2% | 0.0010 | 0.0156 | 0.0049 | 1.9878 | 0.0815 |
| Baseline: hybrid_persona | 0.0036 | 0.4% | 0.0018 | 0.0219 | 0.0073 | 1.8212 | 0.0879 |
| Anchor: Random (100-draw) | 0.0001 | 0.0% | 0.0001 | 0.0015 | 0.0004 | 3.2584 | 0.2009 |

| Comparison Baseline | Metric | System | Baseline | Rel. Gain | 95% CI of Difference | p (raw) | p (BH) | Cohen's d | MDE | Significant? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline: popularity | ndcg_at_10 | 0.0052 | 0.0158 | -67.10% | [-0.0186, -0.0033] | 7.0000e-03 | 3.0000e-02 | -0.153 | 0.0108 | YES (p < 0.05) |
| Baseline: bm25_author | ndcg_at_10 | 0.0052 | 0.0053 | -2.44% | [-0.0052, +0.0050] | 9.5820e-01 | 9.5820e-01 | -0.003 | 0.0072 | NO |
| Baseline: dense_centroid | ndcg_at_10 | 0.0052 | 0.0017 | +211.24% | [+0.0007, +0.0070] | 2.6800e-02 | 7.1467e-02 | 0.121 | 0.0045 | NO |
| Baseline: hybrid_persona | ndcg_at_10 | 0.0052 | 0.0036 | +42.97% | [-0.0011, +0.0047] | 2.8340e-01 | 3.7787e-01 | 0.059 | 0.0041 | NO |
| Baseline: popularity | recall_at_10 | 0.0019 | 0.0064 | -69.91% | [-0.0079, -0.0015] | 7.5000e-03 | 3.0000e-02 | -0.153 | 0.0046 | YES (p < 0.05) |
| Baseline: bm25_author | recall_at_10 | 0.0019 | 0.0039 | -50.25% | [-0.0056, +0.0011] | 2.6030e-01 | 3.7787e-01 | -0.061 | 0.0049 | NO |
| Baseline: dense_centroid | recall_at_10 | 0.0019 | 0.0010 | +95.58% | [-0.0005, +0.0025] | 2.1440e-01 | 3.7787e-01 | 0.069 | 0.0021 | NO |
| Baseline: hybrid_persona | recall_at_10 | 0.0019 | 0.0018 | +5.46% | [-0.0017, +0.0019] | 9.0770e-01 | 9.5820e-01 | 0.006 | 0.0025 | NO |