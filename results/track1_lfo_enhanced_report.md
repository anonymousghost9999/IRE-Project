# Paper Feed — Track 1: Leave-Future-Out Empirical Evaluation Report

> [!NOTE]
> **Evaluation Run Configuration:**
> - Mode: Enhanced (Option 2 & Option 3)
> - Candidate Pool Size (M): 250
> - Channel Depth (M_channel): 500
> - Popularity Prior Weight (w_pop): 0.20
> - Scoring Weights: w_pers=1.00, w_cite=1.00, w_facet=0.00, w_nov=0.25, lambda_div=0.00
> - Report: `track1_lfo_enhanced_report.md` | Raw JSON: `track1_lfo_enhanced_results.json`


## 1. Main Benchmark Results

| System Architecture | NDCG@10 | % of Oracle | Recall@10 | HitRate@10 | MRR | Topic Entropy | ILD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Feed (Full)** | 0.0290 | 2.9% | 0.0224 | 0.1938 | 0.0693 | 1.5139 | 0.0813 |
| Ablation: No-MMR | 0.0290 | 2.9% | 0.0224 | 0.1938 | 0.0693 | 1.5139 | 0.0813 |
| Ablation: No-Graph | 0.0160 | 1.6% | 0.0111 | 0.1156 | 0.0413 | 1.8612 | 0.0766 |
| Ablation: No-Facets | 0.0300 | 3.0% | 0.0224 | 0.1875 | 0.0734 | 1.5055 | 0.0805 |
| Diagnostic: Re-Citation | 0.0466 | 4.7% | 0.0481 | 0.1500 | 0.0822 | 1.4830 | 0.0805 |
| Baseline: random | 0.0000 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 3.2651 | 0.2075 |
| Baseline: popularity | 0.0000 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 3.2651 | 0.2075 |
| Baseline: bm25_author | 0.0086 | 0.9% | 0.0059 | 0.0437 | 0.0186 | 1.8144 | 0.1090 |
| Baseline: dense_centroid | 0.0111 | 1.1% | 0.0085 | 0.0688 | 0.0224 | 1.7728 | 0.0691 |
| Baseline: hybrid_persona | 0.0149 | 1.5% | 0.0100 | 0.0906 | 0.0354 | 1.7077 | 0.0862 |
| Anchor: Random (100-draw) | 0.0001 | 0.0% | 0.0001 | 0.0013 | 0.0004 | 3.2613 | 0.2011 |

## 2. Component Ablations

| Configuration | NDCG@10 | Delta NDCG | Recall@10 | Topic Entropy | Delta Entropy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Paper Feed (Full) | 0.0290 | +0.0000 | 0.0224 | 1.5139 | +0.0000 |
| Ablation: No-MMR | 0.0290 | +0.0000 | 0.0224 | 1.5139 | +0.0000 |
| Ablation: No-Graph | 0.0160 | -0.0130 | 0.0111 | 1.8612 | +0.3473 |
| Ablation: No-Facets | 0.0300 | +0.0009 | 0.0224 | 1.5055 | -0.0085 |

## 3. Statistical Hypothesis Testing

### Hypothesis Testing Summary

**Sample Size:** N = 320 test researchers | **Bootstrap Resamples:** B = 10,000

#### Hypothesis H1: Paper Feed Outperforms Baselines (NDCG@10)
**Verdict:** ACCEPTED (Supported by empirical data)

| Comparison Baseline | System NDCG | Baseline NDCG | Rel. Gain | 95% CI of Difference | p-value (raw) | p-value (BH FDR) | Cohen's d | Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline: random | 0.0290 | 0.0000 | +0.00% | [+0.0214, +0.0374] | 1.0000e-04 | 1.0000e-04 | 0.395 | YES (p < 0.05) |
| Baseline: popularity | 0.0290 | 0.0000 | +0.00% | [+0.0214, +0.0374] | 1.0000e-04 | 1.0000e-04 | 0.395 | YES (p < 0.05) |
| Baseline: bm25_author | 0.0290 | 0.0086 | +236.75% | [+0.0119, +0.0290] | 1.0000e-04 | 1.0000e-04 | 0.264 | YES (p < 0.05) |
| Baseline: dense_centroid | 0.0290 | 0.0111 | +162.00% | [+0.0103, +0.0257] | 1.0000e-04 | 1.0000e-04 | 0.258 | YES (p < 0.05) |
| Baseline: hybrid_persona | 0.0290 | 0.0149 | +94.54% | [+0.0065, +0.0218] | 1.0000e-04 | 1.0000e-04 | 0.203 | YES (p < 0.05) |

#### Hypothesis H2: MMR Diversity with Non-Inferiority
**Verdict:** REJECTED / INCONCLUSIVE

- **Entropy Increase (one-sided):** Delta = +0.0000 (+0.00%), 95% CI = [+0.0000, +0.0000], p = 1.0000e+00 (Significant: False)
- **NDCG Retention (non-inferiority <= 3% loss):** Rel. Loss = 0.00%, 95% CI = [+0.0000, +0.0000], p = 1.0000e+00 (Non-inferior: True)

## 4. History Slice Stratification

| Stratum | Authors (n) | Paper Feed NDCG@10 | Best Baseline NDCG@10 | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| Warm | 248 | 0.0204 | 0.0104 | +96.31% |
| Moderate | 37 | 0.0591 | 0.0320 | +84.75% |
| Sparse | 35 | 0.0579 | 0.0323 | +79.16% |