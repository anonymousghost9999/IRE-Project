# Paper Feed — Track 1: Leave-Future-Out Empirical Evaluation Report

## 1. Main Benchmark Results

| System Architecture | NDCG@10 | % of Oracle | Recall@10 | HitRate@10 | MRR | Topic Entropy | ILD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Feed (Full)** | 0.0237 | 2.4% | 0.0170 | 0.1401 | 0.0555 | 1.4942 | 0.0821 |
| Ablation: No-MMR | 0.0235 | 2.3% | 0.0168 | 0.1401 | 0.0555 | 1.4840 | 0.0817 |
| Ablation: No-Graph | 0.0124 | 1.2% | 0.0089 | 0.0828 | 0.0238 | 1.7343 | 0.0806 |
| Ablation: No-Facets | 0.0261 | 2.6% | 0.0196 | 0.1497 | 0.0610 | 1.4730 | 0.0795 |
| Diagnostic: Re-Citation | 0.0358 | 3.6% | 0.0397 | 0.1178 | 0.0600 | 1.4868 | 0.0817 |
| Baseline: random | 0.0000 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 3.2646 | 0.2074 |
| Baseline: popularity | 0.0240 | 2.4% | 0.0105 | 0.1338 | 0.0649 | 1.9417 | 0.1178 |
| Baseline: bm25_author | 0.0059 | 0.6% | 0.0058 | 0.0382 | 0.0114 | 1.9895 | 0.1141 |
| Baseline: dense_centroid | 0.0113 | 1.1% | 0.0087 | 0.0701 | 0.0228 | 1.7792 | 0.0694 |
| Baseline: hybrid_persona | 0.0150 | 1.5% | 0.0093 | 0.0764 | 0.0320 | 1.6346 | 0.0858 |
| Anchor: Random (100-draw) | 0.0001 | 0.0% | 0.0001 | 0.0010 | 0.0003 | 3.2567 | 0.2019 |

## 2. Component Ablations

| Configuration | NDCG@10 | Delta NDCG | Recall@10 | Topic Entropy | Delta Entropy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Paper Feed (Full) | 0.0237 | +0.0000 | 0.0170 | 1.4942 | +0.0000 |
| Ablation: No-MMR | 0.0235 | -0.0002 | 0.0168 | 1.4840 | -0.0102 |
| Ablation: No-Graph | 0.0124 | -0.0113 | 0.0089 | 1.7343 | +0.2401 |
| Ablation: No-Facets | 0.0261 | +0.0024 | 0.0196 | 1.4730 | -0.0212 |

## 3. Statistical Hypothesis Testing

### Hypothesis Testing Summary

**Sample Size:** N = 314 test researchers | **Bootstrap Resamples:** B = 10,000

#### Hypothesis H1: Paper Feed Outperforms Baselines (NDCG@10)
**Verdict:** REJECTED / INCONCLUSIVE

| Comparison Baseline | System NDCG | Baseline NDCG | Rel. Gain | 95% CI of Difference | p-value (raw) | p-value (BH FDR) | Cohen's d | Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline: random | 0.0237 | 0.0000 | +0.00% | [+0.0162, +0.0324] | 1.0000e-04 | 2.5000e-04 | 0.326 | YES (p < 0.05) |
| Baseline: popularity | 0.0237 | 0.0240 | -1.21% | [-0.0124, +0.0118] | 9.6340e-01 | 9.6340e-01 | -0.003 | NO |
| Baseline: bm25_author | 0.0237 | 0.0059 | +300.89% | [+0.0105, +0.0259] | 1.0000e-04 | 2.5000e-04 | 0.257 | YES (p < 0.05) |
| Baseline: dense_centroid | 0.0237 | 0.0113 | +109.69% | [+0.0068, +0.0183] | 2.0000e-04 | 3.3333e-04 | 0.241 | YES (p < 0.05) |
| Baseline: hybrid_persona | 0.0237 | 0.0150 | +57.38% | [+0.0017, +0.0152] | 1.3400e-02 | 1.6750e-02 | 0.141 | YES (p < 0.05) |

#### Hypothesis H2: MMR Diversity with Non-Inferiority
**Verdict:** ACCEPTED (Supported by empirical data)

- **Entropy Increase (one-sided):** Delta = +0.0102 (+0.69%), 95% CI = [+0.0029, +0.0180], p = 5.9000e-03 (Significant: True)
- **NDCG Retention (non-inferiority <= 3% loss):** Rel. Loss = 0.74%, 95% CI = [-0.0001, +0.0006], p = 6.2200e-01 (Non-inferior: True)

## 4. History Slice Stratification

| Stratum | Authors (n) | Paper Feed NDCG@10 | Best Baseline NDCG@10 | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| Warm | 247 | 0.0204 | 0.0257 | -20.88% |
| Moderate | 36 | 0.0338 | 0.0306 | +10.46% |
| Sparse | 31 | 0.0382 | 0.0365 | +4.64% |