# Paper Feed — Track 1: Leave-Future-Out Empirical Evaluation Report

> [!NOTE]
> **Evaluation Run Configuration:**
> - Mode: Enhanced (Option 2 & Option 3)
> - Candidate Pool Size (M): 250
> - Channel Depth (M_channel): 500
> - Popularity Prior Weight (w_pop): 0.20
> - Scoring Weights: w_pers=1.00, w_cite=1.00, w_facet=0.50, w_nov=0.25, lambda_div=0.20
> - Report: `track1_lfo_enhanced_report.md` | Raw JSON: `track1_lfo_enhanced_results.json`


## 1. Main Benchmark Results

| System Architecture | NDCG@10 | % of Oracle | Recall@10 | HitRate@10 | MRR | Topic Entropy | ILD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Feed (Full)** | 0.0267 | 2.7% | 0.0186 | 0.1656 | 0.0693 | 1.5189 | 0.0848 |
| Ablation: No-MMR | 0.0269 | 2.7% | 0.0189 | 0.1688 | 0.0687 | 1.5101 | 0.0842 |
| Ablation: No-Graph | 0.0128 | 1.3% | 0.0080 | 0.0924 | 0.0278 | 1.8230 | 0.0868 |
| Ablation: No-Facets | 0.0302 | 3.0% | 0.0223 | 0.1847 | 0.0734 | 1.5005 | 0.0818 |
| Diagnostic: Re-Citation | 0.0404 | 4.0% | 0.0456 | 0.1401 | 0.0670 | 1.5079 | 0.0841 |
| Baseline: random | 0.0000 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 3.2646 | 0.2074 |
| Baseline: popularity | 0.0240 | 2.4% | 0.0105 | 0.1338 | 0.0649 | 1.9417 | 0.1178 |
| Baseline: bm25_author | 0.0059 | 0.6% | 0.0058 | 0.0382 | 0.0114 | 1.9895 | 0.1141 |
| Baseline: dense_centroid | 0.0113 | 1.1% | 0.0087 | 0.0701 | 0.0228 | 1.7792 | 0.0694 |
| Baseline: hybrid_persona | 0.0150 | 1.5% | 0.0093 | 0.0764 | 0.0320 | 1.6346 | 0.0858 |
| Anchor: Random (100-draw) | 0.0001 | 0.0% | 0.0001 | 0.0010 | 0.0003 | 3.2567 | 0.2019 |

## 2. Component Ablations

| Configuration | NDCG@10 | Delta NDCG | Recall@10 | Topic Entropy | Delta Entropy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Paper Feed (Full) | 0.0267 | +0.0000 | 0.0186 | 1.5189 | +0.0000 |
| Ablation: No-MMR | 0.0269 | +0.0002 | 0.0189 | 1.5101 | -0.0088 |
| Ablation: No-Graph | 0.0128 | -0.0140 | 0.0080 | 1.8230 | +0.3041 |
| Ablation: No-Facets | 0.0302 | +0.0034 | 0.0223 | 1.5005 | -0.0184 |

## 3. Statistical Hypothesis Testing

### Hypothesis Testing Summary

**Sample Size:** N = 314 test researchers | **Bootstrap Resamples:** B = 10,000

#### Hypothesis H1: Paper Feed Outperforms Baselines (NDCG@10)
**Verdict:** REJECTED / INCONCLUSIVE

| Comparison Baseline | System NDCG | Baseline NDCG | Rel. Gain | 95% CI of Difference | p-value (raw) | p-value (BH FDR) | Cohen's d | Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline: random | 0.0267 | 0.0000 | +0.00% | [+0.0191, +0.0353] | 1.0000e-04 | 1.6667e-04 | 0.368 | YES (p < 0.05) |
| Baseline: popularity | 0.0267 | 0.0240 | +11.63% | [-0.0094, +0.0146] | 6.5630e-01 | 6.5630e-01 | 0.025 | NO |
| Baseline: bm25_author | 0.0267 | 0.0059 | +352.98% | [+0.0130, +0.0290] | 1.0000e-04 | 1.6667e-04 | 0.285 | YES (p < 0.05) |
| Baseline: dense_centroid | 0.0267 | 0.0113 | +136.93% | [+0.0086, +0.0225] | 1.0000e-04 | 1.6667e-04 | 0.245 | YES (p < 0.05) |
| Baseline: hybrid_persona | 0.0267 | 0.0150 | +77.83% | [+0.0035, +0.0199] | 4.5000e-03 | 5.6250e-03 | 0.159 | YES (p < 0.05) |

#### Hypothesis H2: MMR Diversity with Non-Inferiority
**Verdict:** ACCEPTED (Supported by empirical data)

- **Entropy Increase (one-sided):** Delta = +0.0088 (+0.58%), 95% CI = [+0.0017, +0.0167], p = 1.6500e-02 (Significant: True)
- **NDCG Retention (non-inferiority <= 3% loss):** Rel. Loss = 0.67%, 95% CI = [-0.0008, +0.0003], p = 5.2130e-01 (Non-inferior: True)

## 4. History Slice Stratification

| Stratum | Authors (n) | Paper Feed NDCG@10 | Best Baseline NDCG@10 | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| Warm | 247 | 0.0221 | 0.0257 | -14.23% |
| Moderate | 36 | 0.0413 | 0.0306 | +34.93% |
| Sparse | 31 | 0.0470 | 0.0365 | +28.74% |