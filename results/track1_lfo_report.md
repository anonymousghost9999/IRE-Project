# Paper Feed — Track 1: Leave-Future-Out Empirical Evaluation Report

> [!NOTE]
> **Evaluation Run Configuration:**
> - Mode: Baseline
> - Candidate Pool Size (M): 100
> - Channel Depth (M_channel): 200
> - Popularity Prior Weight (w_pop): 0.00
> - Scoring Weights: w_pers=1.00, w_cite=1.00, w_facet=0.25, w_nov=0.25, lambda_div=0.50
> - System variant: grid optimum (headline)
> - Hybrid-Persona baseline: gamma_rec=1.00, n_terms=5
> - Report: `track1_lfo_report.md` | Raw JSON: `track1_lfo_results.json` | Run manifest: `results/runs/track1_track1_lfo_report_20261113T135204Z/manifest.json`


## 1. Main Benchmark Results

| System Architecture | NDCG@10 | % of Oracle | Recall@10 | HitRate@10 | MRR | Topic Entropy | ILD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Feed (Full)** | 0.0216 | 2.2% | 0.0156 | 0.1281 | 0.0557 | 1.5405 | 0.0808 |
| Ablation: No-MMR | 0.0212 | 2.1% | 0.0150 | 0.1250 | 0.0553 | 1.5127 | 0.0797 |
| Ablation: No-Graph | 0.0109 | 1.1% | 0.0082 | 0.0719 | 0.0240 | 1.8414 | 0.0766 |
| Ablation: No-Facets | 0.0234 | 2.3% | 0.0182 | 0.1437 | 0.0584 | 1.5600 | 0.0800 |
| Diagnostic: Re-Citation | 0.0363 | 3.6% | 0.0415 | 0.1281 | 0.0577 | 1.5154 | 0.0801 |
| Baseline: popularity | 0.0129 | 1.3% | 0.0062 | 0.0656 | 0.0286 | 1.6884 | 0.1260 |
| Baseline: bm25_author | 0.0142 | 1.4% | 0.0094 | 0.0813 | 0.0279 | 1.6350 | 0.0993 |
| Baseline: dense_centroid | 0.0111 | 1.1% | 0.0085 | 0.0688 | 0.0224 | 1.7728 | 0.0691 |
| Baseline: citation_only | 0.0348 | 3.5% | 0.0273 | 0.1875 | 0.0711 | 1.6173 | 0.1092 |
| Baseline: hybrid_persona | 0.0128 | 1.3% | 0.0093 | 0.0813 | 0.0321 | 1.7977 | 0.0895 |
| Anchor: Random (100-draw) | 0.0001 | 0.0% | 0.0001 | 0.0013 | 0.0004 | 3.2613 | 0.2011 |

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
- **NDCG Retention (non-inferiority <= 3% loss):** Rel. Loss = 2.10%, 95% CI = [-0.0002, +0.0013], p = 1.2890e-01, MDE = 0.0011 (Non-inferior: True)

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

*fresh: citation_count_T <= 1 and published in [T-1, T]; established: citation_count_T >= 20 and published since 2012 (design §12). Each class averages over the authors with at least one target in it.*

## 6. Zero-Publication Seed-Only Slice

Re-run of the test cohort with each profile replaced by the author's 3 seed topics (n = 320): E_u is empty and the ground truth is Ref(B) <= T minus Authored(u). Lexical and Dense baselines query with the seed topics, Popularity uses the seed topics' subfield, Citation-only is not applicable. Compare only within this table.

| System Architecture | NDCG@10 | % of Oracle | Recall@10 | HitRate@10 | MRR | Topic Entropy | ILD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Feed (Full)** | 0.0048 | 0.5% | 0.0020 | 0.0312 | 0.0094 | 1.9509 | 0.0853 |
| Baseline: popularity | 0.0223 | 2.2% | 0.0087 | 0.1094 | 0.0533 | 1.5729 | 0.1223 |
| Baseline: bm25_author | 0.0072 | 0.7% | 0.0046 | 0.0469 | 0.0180 | 1.7395 | 0.1086 |
| Baseline: dense_centroid | 0.0022 | 0.2% | 0.0012 | 0.0187 | 0.0044 | 1.9960 | 0.0817 |
| Baseline: hybrid_persona | 0.0044 | 0.4% | 0.0016 | 0.0250 | 0.0112 | 1.8310 | 0.0901 |
| Anchor: Random (100-draw) | 0.0001 | 0.0% | 0.0001 | 0.0014 | 0.0004 | 3.2584 | 0.2009 |

| Comparison Baseline | Metric | System | Baseline | Rel. Gain | 95% CI of Difference | p (raw) | p (BH) | Cohen's d | MDE | Significant? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline: popularity | ndcg_at_10 | 0.0048 | 0.0223 | -78.33% | [-0.0281, -0.0083] | 7.0000e-04 | 4.8000e-03 | -0.190 | 0.0142 | YES (p < 0.05) |
| Baseline: bm25_author | ndcg_at_10 | 0.0048 | 0.0072 | -32.46% | [-0.0070, +0.0021] | 3.0490e-01 | 4.0653e-01 | -0.056 | 0.0065 | NO |
| Baseline: dense_centroid | ndcg_at_10 | 0.0048 | 0.0022 | +122.38% | [+0.0008, +0.0049] | 1.2300e-02 | 3.2800e-02 | 0.141 | 0.0030 | YES (p < 0.05) |
| Baseline: hybrid_persona | ndcg_at_10 | 0.0048 | 0.0044 | +10.45% | [-0.0018, +0.0031] | 7.1960e-01 | 7.1960e-01 | 0.020 | 0.0035 | NO |
| Baseline: popularity | recall_at_10 | 0.0020 | 0.0087 | -77.24% | [-0.0106, -0.0032] | 1.2000e-03 | 4.8000e-03 | -0.194 | 0.0053 | YES (p < 0.05) |
| Baseline: bm25_author | recall_at_10 | 0.0020 | 0.0046 | -57.07% | [-0.0066, +0.0004] | 1.2610e-01 | 2.5220e-01 | -0.083 | 0.0049 | NO |
| Baseline: dense_centroid | recall_at_10 | 0.0020 | 0.0012 | +59.31% | [-0.0004, +0.0018] | 1.7500e-01 | 2.8000e-01 | 0.074 | 0.0016 | NO |
| Baseline: hybrid_persona | recall_at_10 | 0.0020 | 0.0016 | +25.17% | [-0.0010, +0.0018] | 5.6470e-01 | 6.4537e-01 | 0.031 | 0.0020 | NO |