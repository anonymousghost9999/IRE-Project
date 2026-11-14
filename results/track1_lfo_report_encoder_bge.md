# Paper Feed — Track 1: Leave-Future-Out Empirical Evaluation Report

> [!NOTE]
> **Evaluation Run Configuration:**
> - Mode: Baseline
> - Candidate Pool Size (M): 100
> - Candidate Channels: SPECTER2 + BM25s (dense channel: bge, V1)
> - Channel Depth (M_channel): 200
> - Popularity Prior Weight (w_pop): 0.00
> - Scoring Weights: w_pers=1.00, w_cite=1.00, w_facet=0.50, w_nov=0.00, lambda_div=0.50
> - System variant: V1 dense encoder bge (design §13.8; exploratory, plan 10.1d); tuning-split decision: keep SPECTER2; weights re-tuned on the full tuning split with this encoder; the Dense Centroid and Hybrid-Persona baselines also use it
> - Hybrid-Persona baseline: gamma_rec=0.70, n_terms=20
> - Report: `track1_lfo_report_encoder_bge.md` | Raw JSON: `track1_lfo_results_encoder_bge.json` | Run manifest: `results/runs/track1_track1_lfo_report_encoder_bge_20261114T004801Z/manifest.json`


## 1. Main Benchmark Results

| System Architecture | NDCG@10 | % of Oracle | Recall@10 | HitRate@10 | MRR | Topic Entropy | ILD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Feed (Full)** | 0.0262 | 2.6% | 0.0200 | 0.1594 | 0.0658 | 1.5151 | 0.2177 |
| Ablation: No-MMR | 0.0252 | 2.5% | 0.0185 | 0.1469 | 0.0631 | 1.4660 | 0.2129 |
| Ablation: No-Graph | 0.0131 | 1.3% | 0.0104 | 0.0969 | 0.0273 | 1.7938 | 0.2255 |
| Ablation: No-Facets | 0.0273 | 2.7% | 0.0213 | 0.1688 | 0.0670 | 1.4787 | 0.2140 |
| Baseline: popularity | 0.0129 | 1.3% | 0.0062 | 0.0656 | 0.0286 | 1.6884 | 0.3187 |
| Baseline: bm25_author | 0.0142 | 1.4% | 0.0094 | 0.0813 | 0.0279 | 1.6350 | 0.2504 |
| Baseline: dense_centroid | 0.0164 | 1.6% | 0.0118 | 0.1000 | 0.0354 | 1.5811 | 0.2019 |
| Baseline: citation_only | 0.0348 | 3.5% | 0.0273 | 0.1875 | 0.0711 | 1.6173 | 0.2695 |
| Baseline: hybrid_persona | 0.0214 | 2.1% | 0.0131 | 0.1125 | 0.0516 | 1.5429 | 0.2220 |
| Anchor: Random (100-draw) | 0.0001 | 0.0% | 0.0001 | 0.0013 | 0.0004 | 3.2613 | 0.4607 |

### Candidate Coverage Ceiling

| System | Candidate pool Recall@100 (ceiling) | Recall@10 | Share of ceiling reached |
| :--- | :---: | :---: | :---: |
| Paper Feed (Full) | 0.0620 | 0.0200 | 32.2% |
| Ablation: No-MMR | 0.0620 | 0.0185 | 29.8% |
| Ablation: No-Graph | 0.0620 | 0.0104 | 16.8% |
| Ablation: No-Facets | 0.0620 | 0.0213 | 34.4% |

*The pool comes from the text channels (BM25 + SPECTER2, RRF); targets outside it cannot be recommended whatever the scoring weights. For reference, Citation-only ranks the whole eligible corpus by the graph signal and reaches Recall@10 = 0.0273.*

### Diagnostic: Re-Citation

| Run | Ground truth | Recall@10 | NDCG@10 | HitRate@10 | Candidate Recall@100 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Diagnostic: Re-Citation | ReCitation (own past references cited again after T) | 0.0417 | 0.0351 | 0.1375 | 0.0898 |

*Citations(u, <= T) are removed from the exclusion set, so the feed may re-recommend the author's own past references. Different ground truth from every other table: not comparable with the Discovery numbers.*

## 2. Component Ablations

| Configuration | NDCG@10 | Delta NDCG | Recall@10 | Topic Entropy | Delta Entropy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Paper Feed (Full) | 0.0262 | +0.0000 | 0.0200 | 1.5151 | +0.0000 |
| Ablation: No-MMR | 0.0252 | -0.0010 | 0.0185 | 1.4660 | -0.0490 |
| Ablation: No-Graph | 0.0131 | -0.0131 | 0.0104 | 1.7938 | +0.2787 |
| Ablation: No-Facets | 0.0273 | +0.0010 | 0.0213 | 1.4787 | -0.0364 |

## 3. Statistical Hypothesis Testing

### Hypothesis Testing Summary

**Sample Size:** N = 320 test researchers | **Bootstrap Resamples:** B = 10,000

#### Hypothesis H1: Paper Feed Outperforms the Five Baselines (NDCG@10 and Recall@10)
**Verdict:** REJECTED / INCONCLUSIVE

| Comparison Baseline | Metric | System | Baseline | Rel. Gain | 95% CI of Difference | p (raw) | p (BH) | Cohen's d | MDE | Significant? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline: popularity | ndcg_at_10 | 0.0262 | 0.0129 | +102.69% | [+0.0036, +0.0233] | 8.7000e-03 | 1.4500e-02 | 0.145 | 0.0142 | YES (p < 0.05) |
| Baseline: bm25_author | ndcg_at_10 | 0.0262 | 0.0142 | +85.01% | [+0.0035, +0.0204] | 5.9000e-03 | 1.4500e-02 | 0.156 | 0.0121 | YES (p < 0.05) |
| Baseline: dense_centroid | ndcg_at_10 | 0.0262 | 0.0164 | +60.08% | [+0.0027, +0.0171] | 7.5000e-03 | 1.4500e-02 | 0.151 | 0.0102 | YES (p < 0.05) |
| Baseline: citation_only | ndcg_at_10 | 0.0262 | 0.0348 | -24.52% | [-0.0191, +0.0016] | 1.0780e-01 | 1.1978e-01 | -0.089 | 0.0148 | NO |
| Baseline: hybrid_persona | ndcg_at_10 | 0.0262 | 0.0214 | +22.89% | [-0.0034, +0.0126] | 2.2880e-01 | 2.2880e-01 | 0.067 | 0.0114 | NO |
| Baseline: popularity | recall_at_10 | 0.0200 | 0.0062 | +220.53% | [+0.0070, +0.0208] | 4.0000e-04 | 2.0000e-03 | 0.214 | 0.0100 | YES (p < 0.05) |
| Baseline: bm25_author | recall_at_10 | 0.0200 | 0.0094 | +112.13% | [+0.0047, +0.0165] | 3.0000e-04 | 2.0000e-03 | 0.196 | 0.0085 | YES (p < 0.05) |
| Baseline: dense_centroid | recall_at_10 | 0.0200 | 0.0118 | +69.80% | [+0.0028, +0.0141] | 4.2000e-03 | 1.4000e-02 | 0.159 | 0.0081 | YES (p < 0.05) |
| Baseline: citation_only | recall_at_10 | 0.0200 | 0.0273 | -26.82% | [-0.0148, +0.0000] | 5.2000e-02 | 6.5000e-02 | -0.108 | 0.0106 | NO |
| Baseline: hybrid_persona | recall_at_10 | 0.0200 | 0.0131 | +52.44% | [+0.0014, +0.0125] | 1.5300e-02 | 2.1857e-02 | 0.135 | 0.0079 | YES (p < 0.05) |

*BH correction runs over all baseline x metric comparisons. MDE: smallest true difference detectable with 80% power at alpha = 0.05 and this N; a non-significant row whose observed difference is below its MDE is inconclusive, not evidence of no effect.*

#### Hypothesis H2: MMR Diversity with Non-Inferiority
**Verdict:** ACCEPTED (Supported by empirical data)

- **Entropy Increase (one-sided):** Delta = +0.0490 (+3.35%), 95% CI = [+0.0337, +0.0653], p = 1.0000e-04, MDE = 0.0227 (Significant: True)
- **NDCG Retention (non-inferiority: relative loss <= 3%):** NDCG@10 change with MMR = +4.13% (Delta = +0.0010, 95% CI = [-0.0002, +0.0025]); worst-case relative loss from the CI = 0.91%, MDE = 0.0019 (Non-inferior: True)

## 4. History Slice Stratification

| Stratum | Authors (n) | Paper Feed NDCG@10 | Best Baseline NDCG@10 | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| Warm | 248 | 0.0234 | 0.0309 | -24.32% |
| Moderate | 37 | 0.0429 | 0.0401 | +6.97% |
| Sparse | 35 | 0.0290 | 0.0567 | -48.82% |

## 5. Item Cold-Start Slice

| System | Recall@10 fresh (n = 101 authors) | Recall@10 established (n = 318 authors) |
| :--- | :---: | :---: |
| Paper Feed (Full) | 0.0355 | 0.0211 |
| Ablation: No-MMR | 0.0388 | 0.0197 |
| Ablation: No-Graph | 0.0289 | 0.0080 |
| Ablation: No-Facets | 0.0355 | 0.0196 |
| Baseline: popularity | 0.0000 | 0.0077 |
| Baseline: bm25_author | 0.0437 | 0.0039 |
| Baseline: dense_centroid | 0.0272 | 0.0107 |
| Baseline: citation_only | 0.0404 | 0.0260 |
| Baseline: hybrid_persona | 0.0454 | 0.0120 |

*fresh: citation_count_T <= 1 and published in [T-1, T]; established: citation_count_T >= 20 and published since 2012 (design §12). Each class averages over the authors with at least one target in it.*

## 6. Zero-Publication Seed-Only Slice

Re-run of the test cohort with each profile replaced by the author's 3 seed topics (n = 320): E_u is empty and the ground truth is Ref(B) <= T minus Authored(u). Lexical and Dense baselines query with the seed topics, Popularity uses the seed topics' subfield, Citation-only is not applicable. Compare only within this table.

| System Architecture | NDCG@10 | % of Oracle | Recall@10 | HitRate@10 | MRR | Topic Entropy | ILD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Feed (Full)** | 0.0053 | 0.5% | 0.0024 | 0.0312 | 0.0141 | 1.6585 | 0.2256 |
| Baseline: popularity | 0.0223 | 2.2% | 0.0087 | 0.1094 | 0.0533 | 1.5729 | 0.3092 |
| Baseline: bm25_author | 0.0072 | 0.7% | 0.0046 | 0.0469 | 0.0180 | 1.7395 | 0.2789 |
| Baseline: dense_centroid | 0.0036 | 0.4% | 0.0019 | 0.0312 | 0.0107 | 1.5391 | 0.2168 |
| Baseline: hybrid_persona | 0.0043 | 0.4% | 0.0033 | 0.0406 | 0.0085 | 1.3816 | 0.2106 |
| Anchor: Random (100-draw) | 0.0001 | 0.0% | 0.0001 | 0.0014 | 0.0004 | 3.2584 | 0.4619 |

| Comparison Baseline | Metric | System | Baseline | Rel. Gain | 95% CI of Difference | p (raw) | p (BH) | Cohen's d | MDE | Significant? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline: popularity | ndcg_at_10 | 0.0053 | 0.0223 | -76.43% | [-0.0276, -0.0077] | 1.1000e-03 | 8.8000e-03 | -0.187 | 0.0142 | YES (p < 0.05) |
| Baseline: bm25_author | ndcg_at_10 | 0.0053 | 0.0072 | -26.52% | [-0.0064, +0.0022] | 3.7710e-01 | 5.9074e-01 | -0.048 | 0.0061 | NO |
| Baseline: dense_centroid | ndcg_at_10 | 0.0053 | 0.0036 | +48.18% | [-0.0009, +0.0047] | 2.2620e-01 | 4.5240e-01 | 0.067 | 0.0040 | NO |
| Baseline: hybrid_persona | ndcg_at_10 | 0.0053 | 0.0043 | +22.80% | [-0.0018, +0.0041] | 5.1690e-01 | 5.9074e-01 | 0.035 | 0.0043 | NO |
| Baseline: popularity | recall_at_10 | 0.0024 | 0.0087 | -72.77% | [-0.0103, -0.0026] | 2.4000e-03 | 9.6000e-03 | -0.180 | 0.0055 | YES (p < 0.05) |
| Baseline: bm25_author | recall_at_10 | 0.0024 | 0.0046 | -48.65% | [-0.0063, +0.0009] | 2.0890e-01 | 4.5240e-01 | -0.069 | 0.0051 | NO |
| Baseline: dense_centroid | recall_at_10 | 0.0024 | 0.0019 | +23.10% | [-0.0015, +0.0022] | 6.2320e-01 | 6.2320e-01 | 0.026 | 0.0026 | NO |
| Baseline: hybrid_persona | recall_at_10 | 0.0024 | 0.0033 | -27.36% | [-0.0034, +0.0013] | 4.4830e-01 | 5.9074e-01 | -0.042 | 0.0033 | NO |