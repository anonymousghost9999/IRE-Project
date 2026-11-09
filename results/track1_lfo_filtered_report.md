# Paper Feed — Track 1: Leave-Future-Out Empirical Evaluation Report

> [!NOTE]
> **Evaluation Run Configuration:**
> - Mode: Baseline
> - Candidate Pool Size (M): 100
> - Channel Depth (M_channel): 200
> - Popularity Prior Weight (w_pop): 0.00
> - Scoring Weights: w_pers=1.00, w_cite=1.00, w_facet=0.50, w_nov=0.25, lambda_div=0.20
> - Report: `track1_lfo_filtered_report.md` | Raw JSON: `track1_lfo_filtered_results.json`
> 
> **Popularity Filter Active:**
> - Threshold: citation_count_T >= 200
> - Targets removed: 3237/6199 (52.22%)
> - Authors with remaining targets: 378


## 1. Main Benchmark Results

| System Architecture | NDCG@10 | % of Oracle | Recall@10 | HitRate@10 | MRR | Topic Entropy | ILD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Feed (Full)** | 0.0241 | 2.4% | 0.0275 | 0.1178 | 0.0424 | 1.4942 | 0.0821 |
| Ablation: No-MMR | 0.0239 | 2.4% | 0.0273 | 0.1178 | 0.0425 | 1.4840 | 0.0817 |
| Ablation: No-Graph | 0.0111 | 1.1% | 0.0141 | 0.0669 | 0.0163 | 1.7343 | 0.0806 |
| Ablation: No-Facets | 0.0285 | 2.9% | 0.0316 | 0.1274 | 0.0520 | 1.4730 | 0.0795 |
| Diagnostic: Re-Citation | 0.0358 | 3.6% | 0.0397 | 0.1178 | 0.0600 | 1.4868 | 0.0817 |
| Baseline: random | 0.0000 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 3.2646 | 0.2074 |
| Baseline: popularity | 0.0000 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 1.9417 | 0.1178 |
| Baseline: bm25_author | 0.0061 | 0.6% | 0.0079 | 0.0287 | 0.0102 | 1.9895 | 0.1141 |
| Baseline: dense_centroid | 0.0112 | 1.1% | 0.0137 | 0.0637 | 0.0177 | 1.7792 | 0.0694 |
| Baseline: hybrid_persona | 0.0144 | 1.4% | 0.0136 | 0.0637 | 0.0275 | 1.6346 | 0.0858 |
| Anchor: Random (100-draw) | 0.0001 | 0.0% | 0.0001 | 0.0008 | 0.0003 | 3.2567 | 0.2019 |

## 2. Component Ablations

| Configuration | NDCG@10 | Delta NDCG | Recall@10 | Topic Entropy | Delta Entropy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Paper Feed (Full) | 0.0241 | +0.0000 | 0.0275 | 1.4942 | +0.0000 |
| Ablation: No-MMR | 0.0239 | -0.0002 | 0.0273 | 1.4840 | -0.0102 |
| Ablation: No-Graph | 0.0111 | -0.0130 | 0.0141 | 1.7343 | +0.2401 |
| Ablation: No-Facets | 0.0285 | +0.0044 | 0.0316 | 1.4730 | -0.0212 |

## 3. Statistical Hypothesis Testing

### Hypothesis Testing Summary

**Sample Size:** N = 314 test researchers | **Bootstrap Resamples:** B = 10,000

#### Hypothesis H1: Paper Feed Outperforms Baselines (NDCG@10)
**Verdict:** ACCEPTED (Supported by empirical data)

| Comparison Baseline | System NDCG | Baseline NDCG | Rel. Gain | 95% CI of Difference | p-value (raw) | p-value (BH FDR) | Cohen's d | Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline: random | 0.0241 | 0.0000 | +0.00% | [+0.0156, +0.0340] | 1.0000e-04 | 2.5000e-04 | 0.293 | YES (p < 0.05) |
| Baseline: popularity | 0.0241 | 0.0000 | +0.00% | [+0.0156, +0.0340] | 1.0000e-04 | 2.5000e-04 | 0.293 | YES (p < 0.05) |
| Baseline: bm25_author | 0.0241 | 0.0061 | +295.38% | [+0.0101, +0.0269] | 2.0000e-04 | 3.3333e-04 | 0.234 | YES (p < 0.05) |
| Baseline: dense_centroid | 0.0241 | 0.0112 | +115.58% | [+0.0061, +0.0205] | 1.1000e-03 | 1.3750e-03 | 0.201 | YES (p < 0.05) |
| Baseline: hybrid_persona | 0.0241 | 0.0144 | +67.38% | [+0.0013, +0.0184] | 2.5300e-02 | 2.5300e-02 | 0.127 | YES (p < 0.05) |

#### Hypothesis H2: MMR Diversity with Non-Inferiority
**Verdict:** ACCEPTED (Supported by empirical data)

- **Entropy Increase (one-sided):** Delta = +0.0102 (+0.69%), 95% CI = [+0.0029, +0.0180], p = 5.9000e-03 (Significant: True)
- **NDCG Retention (non-inferiority <= 3% loss):** Rel. Loss = 0.68%, 95% CI = [-0.0001, +0.0006], p = 2.4800e-01 (Non-inferior: True)

## 4. History Slice Stratification

| Stratum | Authors (n) | Paper Feed NDCG@10 | Best Baseline NDCG@10 | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| Warm | 247 | 0.0222 | 0.0128 | +73.54% |
| Moderate | 36 | 0.0327 | 0.0200 | +63.29% |
| Sparse | 31 | 0.0288 | 0.0293 | -1.63% |