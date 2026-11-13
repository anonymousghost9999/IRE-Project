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
> - Report: `track1_lfo_filtered_report.md` | Raw JSON: `track1_lfo_filtered_results.json` | Run manifest: `results/runs/track1_track1_lfo_report_20261113T141415Z/manifest.json`
> 
> **Popularity Filter Active:**
> - Threshold: citation_count_T >= 200
> - Targets removed: 3237/6199 (52.22%)
> - Authors with remaining targets: 378
> - Evaluated cohort: Macro averages and hypothesis testing over 304 active authors with targets


## 1. Main Benchmark Results

| System Architecture | NDCG@10 | % of Oracle | Recall@10 | HitRate@10 | MRR | Topic Entropy | ILD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Feed (Full)** | 0.0231 | 2.3% | 0.0255 | 0.1086 | 0.0443 | 1.5306 | 0.0808 |
| Ablation: No-MMR | 0.0231 | 2.3% | 0.0253 | 0.1086 | 0.0444 | 1.5028 | 0.0797 |
| Ablation: No-Graph | 0.0095 | 0.9% | 0.0124 | 0.0559 | 0.0165 | 1.8303 | 0.0768 |
| Ablation: No-Facets | 0.0268 | 2.7% | 0.0296 | 0.1250 | 0.0516 | 1.5512 | 0.0800 |
| Diagnostic: Re-Citation | 0.0371 | 3.7% | 0.0426 | 0.1316 | 0.0591 | 1.5078 | 0.0801 |
| Baseline: popularity | 0.0002 | 0.0% | 0.0002 | 0.0033 | 0.0004 | 1.7046 | 0.1265 |
| Baseline: bm25_author | 0.0158 | 1.6% | 0.0169 | 0.0789 | 0.0280 | 1.6256 | 0.0995 |
| Baseline: dense_centroid | 0.0115 | 1.2% | 0.0142 | 0.0658 | 0.0183 | 1.7564 | 0.0691 |
| Baseline: citation_only | 0.0304 | 3.0% | 0.0387 | 0.1513 | 0.0479 | 1.6039 | 0.1089 |
| Baseline: hybrid_persona | 0.0138 | 1.4% | 0.0141 | 0.0691 | 0.0294 | 1.8036 | 0.0895 |
| Anchor: Random (100-draw) | 0.0001 | 0.0% | 0.0001 | 0.0008 | 0.0003 | 3.2612 | 0.2011 |

## 2. Component Ablations

| Configuration | NDCG@10 | Delta NDCG | Recall@10 | Topic Entropy | Delta Entropy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Paper Feed (Full) | 0.0231 | +0.0000 | 0.0255 | 1.5306 | +0.0000 |
| Ablation: No-MMR | 0.0231 | -0.0001 | 0.0253 | 1.5028 | -0.0278 |
| Ablation: No-Graph | 0.0095 | -0.0137 | 0.0124 | 1.8303 | +0.2997 |
| Ablation: No-Facets | 0.0268 | +0.0037 | 0.0296 | 1.5512 | +0.0206 |

## 3. Statistical Hypothesis Testing

### Hypothesis Testing Summary

**Sample Size:** N = 304 test researchers | **Bootstrap Resamples:** B = 10,000

#### Hypothesis H1: Paper Feed Outperforms the Five Baselines (NDCG@10 and Recall@10)
**Verdict:** REJECTED / INCONCLUSIVE

| Comparison Baseline | Metric | System | Baseline | Rel. Gain | 95% CI of Difference | p (raw) | p (BH) | Cohen's d | MDE | Significant? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline: popularity | ndcg_at_10 | 0.0231 | 0.0002 | +10509.15% | [+0.0145, +0.0328] | 1.0000e-04 | 5.0000e-04 | 0.280 | 0.0131 | YES (p < 0.05) |
| Baseline: bm25_author | ndcg_at_10 | 0.0231 | 0.0158 | +45.94% | [-0.0031, +0.0180] | 1.7730e-01 | 1.9320e-01 | 0.076 | 0.0151 | NO |
| Baseline: dense_centroid | ndcg_at_10 | 0.0231 | 0.0115 | +100.28% | [+0.0038, +0.0197] | 5.0000e-03 | 1.6667e-02 | 0.161 | 0.0114 | YES (p < 0.05) |
| Baseline: citation_only | ndcg_at_10 | 0.0231 | 0.0304 | -23.97% | [-0.0185, +0.0037] | 1.9320e-01 | 1.9320e-01 | -0.074 | 0.0157 | NO |
| Baseline: hybrid_persona | ndcg_at_10 | 0.0231 | 0.0138 | +67.14% | [+0.0013, +0.0183] | 3.3200e-02 | 4.7429e-02 | 0.122 | 0.0121 | YES (p < 0.05) |
| Baseline: popularity | recall_at_10 | 0.0255 | 0.0002 | +10746.70% | [+0.0158, +0.0362] | 1.0000e-04 | 5.0000e-04 | 0.278 | 0.0146 | YES (p < 0.05) |
| Baseline: bm25_author | recall_at_10 | 0.0255 | 0.0169 | +50.50% | [-0.0019, +0.0197] | 1.1570e-01 | 1.4463e-01 | 0.089 | 0.0153 | NO |
| Baseline: dense_centroid | recall_at_10 | 0.0255 | 0.0142 | +79.48% | [+0.0031, +0.0202] | 9.3000e-03 | 2.0600e-02 | 0.147 | 0.0122 | YES (p < 0.05) |
| Baseline: citation_only | recall_at_10 | 0.0255 | 0.0387 | -34.18% | [-0.0256, -0.0014] | 3.1600e-02 | 4.7429e-02 | -0.122 | 0.0174 | YES (p < 0.05) |
| Baseline: hybrid_persona | recall_at_10 | 0.0255 | 0.0141 | +80.78% | [+0.0034, +0.0205] | 1.0300e-02 | 2.0600e-02 | 0.150 | 0.0122 | YES (p < 0.05) |

*BH correction runs over all baseline x metric comparisons. MDE: smallest true difference detectable with 80% power at alpha = 0.05 and this N; a non-significant row whose observed difference is below its MDE is inconclusive, not evidence of no effect.*

#### Hypothesis H2: MMR Diversity with Non-Inferiority
**Verdict:** REJECTED / INCONCLUSIVE

- **Entropy Increase (one-sided):** Delta = +0.0278 (+1.85%), 95% CI = [+0.0165, +0.0399], p = 1.0000e-04, MDE = 0.0167 (Significant: True)
- **NDCG Retention (non-inferiority <= 3% loss):** Rel. Loss = 0.22%, 95% CI = [-0.0007, +0.0008], p = 4.4140e-01, MDE = 0.0011 (Non-inferior: False)

## 4. History Slice Stratification

| Stratum | Authors (n) | Paper Feed NDCG@10 | Best Baseline NDCG@10 | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| Warm | 237 | 0.0207 | 0.0287 | -27.81% |
| Moderate | 35 | 0.0369 | 0.0396 | -6.63% |
| Sparse | 32 | 0.0256 | 0.0328 | -21.94% |

## 5. Item Cold-Start Slice

| System | Recall@10 fresh (n = 101 authors) | Recall@10 established (n = 283 authors) |
| :--- | :---: | :---: |
| Paper Feed (Full) | 0.0223 | 0.0213 |
| Ablation: No-MMR | 0.0223 | 0.0213 |
| Ablation: No-Graph | 0.0000 | 0.0096 |
| Ablation: No-Facets | 0.0223 | 0.0239 |
| Baseline: popularity | 0.0000 | 0.0003 |
| Baseline: bm25_author | 0.0437 | 0.0054 |
| Baseline: dense_centroid | 0.0173 | 0.0094 |
| Baseline: citation_only | 0.0404 | 0.0335 |
| Baseline: hybrid_persona | 0.0152 | 0.0088 |

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