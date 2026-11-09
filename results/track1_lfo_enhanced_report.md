# Paper Feed — Track 1: Leave-Future-Out Empirical Evaluation Report

> [!NOTE]
> **Evaluation Run Configuration:**
> - Mode: Enhanced (Option 2 & Option 3)
> - Candidate Pool Size (M): 250
> - Channel Depth (M_channel): 500
> - Popularity Prior Weight (w_pop): 0.20
> - Scoring Weights: w_pers=1.00, w_cite=1.00, w_facet=0.50, w_nov=0.25, lambda_div=0.20
> - Report: `track1_lfo_enhanced_report.md` | Raw JSON: `track1_lfo_enhanced_results.json`
> 
> **Popularity Filter Active:**
> - Threshold: citation_count_T >= 200
> - Targets removed: 3237/6199 (52.22%)
> - Authors with remaining targets: 378
> - Evaluated cohort: Macro averages and hypothesis testing over 304 active authors with targets


## 1. Main Benchmark Results

| System Architecture | NDCG@10 | % of Oracle | Recall@10 | HitRate@10 | MRR | Topic Entropy | ILD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Feed (Full)** | 0.0250 | 2.5% | 0.0282 | 0.1250 | 0.0453 | 1.4996 | 0.0842 |
| Ablation: No-MMR | 0.0254 | 2.5% | 0.0287 | 0.1283 | 0.0452 | 1.4889 | 0.0837 |
| Ablation: No-Graph | 0.0081 | 0.8% | 0.0101 | 0.0625 | 0.0144 | 1.8003 | 0.0866 |
| Ablation: No-Facets | 0.0290 | 2.9% | 0.0338 | 0.1480 | 0.0532 | 1.4748 | 0.0812 |
| Diagnostic: Re-Citation | 0.0380 | 3.8% | 0.0454 | 0.1414 | 0.0616 | 1.4807 | 0.0835 |
| Baseline: random | 0.0000 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 3.2640 | 0.2075 |
| Baseline: popularity | 0.0000 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 1.9388 | 0.1177 |
| Baseline: bm25_author | 0.0066 | 0.7% | 0.0084 | 0.0329 | 0.0111 | 1.9952 | 0.1146 |
| Baseline: dense_centroid | 0.0115 | 1.2% | 0.0142 | 0.0658 | 0.0183 | 1.7458 | 0.0690 |
| Baseline: hybrid_persona | 0.0149 | 1.5% | 0.0140 | 0.0658 | 0.0284 | 1.6280 | 0.0857 |
| Anchor: Random (100-draw) | 0.0001 | 0.0% | 0.0001 | 0.0008 | 0.0003 | 3.2563 | 0.2020 |

## 2. Component Ablations

| Configuration | NDCG@10 | Delta NDCG | Recall@10 | Topic Entropy | Delta Entropy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Paper Feed (Full) | 0.0250 | +0.0000 | 0.0282 | 1.4996 | +0.0000 |
| Ablation: No-MMR | 0.0254 | +0.0004 | 0.0287 | 1.4889 | -0.0107 |
| Ablation: No-Graph | 0.0081 | -0.0169 | 0.0101 | 1.8003 | +0.3007 |
| Ablation: No-Facets | 0.0290 | +0.0040 | 0.0338 | 1.4748 | -0.0248 |

## 3. Statistical Hypothesis Testing

### Hypothesis Testing Summary

**Sample Size:** N = 304 test researchers | **Bootstrap Resamples:** B = 10,000

#### Hypothesis H1: Paper Feed Outperforms Baselines (NDCG@10)
**Verdict:** REJECTED / INCONCLUSIVE

| Comparison Baseline | System NDCG | Baseline NDCG | Rel. Gain | 95% CI of Difference | p-value (raw) | p-value (BH FDR) | Cohen's d | Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline: random | 0.0250 | 0.0000 | +0.00% | [+0.0160, +0.0353] | 1.0000e-04 | 2.5000e-04 | 0.289 | YES (p < 0.05) |
| Baseline: popularity | 0.0250 | 0.0000 | +0.00% | [+0.0160, +0.0353] | 1.0000e-04 | 2.5000e-04 | 0.289 | YES (p < 0.05) |
| Baseline: bm25_author | 0.0250 | 0.0066 | +280.97% | [+0.0093, +0.0284] | 5.0000e-04 | 8.3333e-04 | 0.215 | YES (p < 0.05) |
| Baseline: dense_centroid | 0.0250 | 0.0115 | +116.24% | [+0.0047, +0.0227] | 4.2000e-03 | 5.2500e-03 | 0.167 | YES (p < 0.05) |
| Baseline: hybrid_persona | 0.0250 | 0.0149 | +67.89% | [-0.0002, +0.0204] | 5.4900e-02 | 5.4900e-02 | 0.111 | NO |

#### Hypothesis H2: MMR Diversity with Non-Inferiority
**Verdict:** REJECTED / INCONCLUSIVE

- **Entropy Increase (one-sided):** Delta = +0.0107 (+0.72%), 95% CI = [+0.0032, +0.0189], p = 6.7000e-03 (Significant: True)
- **NDCG Retention (non-inferiority <= 3% loss):** Rel. Loss = 1.53%, 95% CI = [-0.0012, +0.0002], p = 8.6350e-01 (Non-inferior: False)

## 4. History Slice Stratification

| Stratum | Authors (n) | Paper Feed NDCG@10 | Best Baseline NDCG@10 | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| Warm | 237 | 0.0240 | 0.0134 | +79.88% |
| Moderate | 35 | 0.0415 | 0.0206 | +101.39% |
| Sparse | 32 | 0.0138 | 0.0284 | -51.39% |