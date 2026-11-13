# Paper Feed — Track 1: Leave-Future-Out Empirical Evaluation Report

> [!NOTE]
> **Evaluation Run Configuration:**
> - Mode: Enhanced (Option 2 & Option 3)
> - Candidate Pool Size (M): 250
> - Channel Depth (M_channel): 500
> - Popularity Prior Weight (w_pop): 0.20
> - Scoring Weights: w_pers=1.00, w_cite=1.00, w_facet=0.00, w_nov=0.25, lambda_div=0.00
> - System variant: grid optimum (headline)
> - Hybrid-Persona baseline: gamma_rec=1.00, n_terms=5
> - Report: `track1_lfo_enhanced_report.md` | Raw JSON: `track1_lfo_enhanced_results.json` | Run manifest: `results/runs/track1_track1_lfo_report_enhanced_20261113T173902Z/manifest.json`


## 1. Main Benchmark Results

| System Architecture | NDCG@10 | % of Oracle | Recall@10 | HitRate@10 | MRR | Topic Entropy | ILD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Feed (Full)** | 0.0290 | 2.9% | 0.0224 | 0.1938 | 0.0693 | 1.5139 | 0.0813 |
| Ablation: No-MMR | 0.0290 | 2.9% | 0.0224 | 0.1938 | 0.0693 | 1.5139 | 0.0813 |
| Ablation: No-Graph | 0.0160 | 1.6% | 0.0111 | 0.1156 | 0.0413 | 1.8612 | 0.0766 |
| Ablation: No-Facets | 0.0300 | 3.0% | 0.0224 | 0.1875 | 0.0734 | 1.5055 | 0.0805 |
| Baseline: popularity | 0.0129 | 1.3% | 0.0062 | 0.0656 | 0.0286 | 1.6884 | 0.1260 |
| Baseline: bm25_author | 0.0142 | 1.4% | 0.0094 | 0.0813 | 0.0279 | 1.6350 | 0.0993 |
| Baseline: dense_centroid | 0.0111 | 1.1% | 0.0085 | 0.0688 | 0.0224 | 1.7728 | 0.0691 |
| Baseline: citation_only | 0.0348 | 3.5% | 0.0273 | 0.1875 | 0.0711 | 1.6173 | 0.1092 |
| Baseline: hybrid_persona | 0.0128 | 1.3% | 0.0093 | 0.0813 | 0.0321 | 1.7977 | 0.0895 |
| Anchor: Random (100-draw) | 0.0001 | 0.0% | 0.0001 | 0.0013 | 0.0004 | 3.2613 | 0.2011 |

### Candidate Coverage Ceiling

| System | Candidate pool Recall@100 (ceiling) | Recall@10 | Share of ceiling reached |
| :--- | :---: | :---: | :---: |
| Paper Feed (Full) | 0.0531 | 0.0224 | 42.2% |
| Ablation: No-MMR | 0.0531 | 0.0224 | 42.2% |
| Ablation: No-Graph | 0.0531 | 0.0111 | 20.9% |
| Ablation: No-Facets | 0.0531 | 0.0224 | 42.2% |

*The pool comes from the text channels (BM25 + SPECTER2, RRF); targets outside it cannot be recommended whatever the scoring weights. For reference, Citation-only ranks the whole eligible corpus by the graph signal and reaches Recall@10 = 0.0273.*

### Diagnostic: Re-Citation

| Run | Ground truth | Recall@10 | NDCG@10 | HitRate@10 | Candidate Recall@100 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Diagnostic: Re-Citation | ReCitation (own past references cited again after T) | 0.0481 | 0.0466 | 0.1500 | 0.0796 |

*Citations(u, <= T) are removed from the exclusion set, so the feed may re-recommend the author's own past references. Different ground truth from every other table: not comparable with the Discovery numbers.*

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

#### Hypothesis H1: Paper Feed Outperforms the Five Baselines (NDCG@10 and Recall@10)
**Verdict:** REJECTED / INCONCLUSIVE

| Comparison Baseline | Metric | System | Baseline | Rel. Gain | 95% CI of Difference | p (raw) | p (BH) | Cohen's d | MDE | Significant? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline: popularity | ndcg_at_10 | 0.0290 | 0.0129 | +124.10% | [+0.0058, +0.0261] | 2.6000e-03 | 3.7143e-03 | 0.173 | 0.0145 | YES (p < 0.05) |
| Baseline: bm25_author | ndcg_at_10 | 0.0290 | 0.0142 | +104.55% | [+0.0052, +0.0243] | 3.1000e-03 | 3.8750e-03 | 0.172 | 0.0136 | YES (p < 0.05) |
| Baseline: dense_centroid | ndcg_at_10 | 0.0290 | 0.0111 | +162.00% | [+0.0103, +0.0257] | 1.0000e-04 | 2.0000e-04 | 0.258 | 0.0109 | YES (p < 0.05) |
| Baseline: citation_only | ndcg_at_10 | 0.0290 | 0.0348 | -16.55% | [-0.0152, +0.0033] | 2.1650e-01 | 2.1650e-01 | -0.068 | 0.0131 | NO |
| Baseline: hybrid_persona | ndcg_at_10 | 0.0290 | 0.0128 | +126.40% | [+0.0093, +0.0235] | 1.0000e-04 | 2.0000e-04 | 0.248 | 0.0102 | YES (p < 0.05) |
| Baseline: popularity | recall_at_10 | 0.0224 | 0.0062 | +259.13% | [+0.0089, +0.0234] | 1.0000e-04 | 2.0000e-04 | 0.247 | 0.0103 | YES (p < 0.05) |
| Baseline: bm25_author | recall_at_10 | 0.0224 | 0.0094 | +137.68% | [+0.0062, +0.0199] | 2.0000e-04 | 3.3333e-04 | 0.214 | 0.0096 | YES (p < 0.05) |
| Baseline: dense_centroid | recall_at_10 | 0.0224 | 0.0085 | +162.95% | [+0.0083, +0.0198] | 1.0000e-04 | 2.0000e-04 | 0.265 | 0.0083 | YES (p < 0.05) |
| Baseline: citation_only | recall_at_10 | 0.0224 | 0.0273 | -18.01% | [-0.0122, +0.0022] | 1.8180e-01 | 2.0200e-01 | -0.075 | 0.0102 | NO |
| Baseline: hybrid_persona | recall_at_10 | 0.0224 | 0.0093 | +139.65% | [+0.0081, +0.0185] | 1.0000e-04 | 2.0000e-04 | 0.278 | 0.0073 | YES (p < 0.05) |

*BH correction runs over all baseline x metric comparisons. MDE: smallest true difference detectable with 80% power at alpha = 0.05 and this N; a non-significant row whose observed difference is below its MDE is inconclusive, not evidence of no effect.*

#### Hypothesis H2: MMR Diversity with Non-Inferiority
**Verdict:** NOT TESTABLE: the tuned lambda_div is 0, so the full system and the No-MMR ablation are identical and there is no MMR effect to measure.


## 4. History Slice Stratification

| Stratum | Authors (n) | Paper Feed NDCG@10 | Best Baseline NDCG@10 | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| Warm | 248 | 0.0204 | 0.0309 | -33.75% |
| Moderate | 37 | 0.0591 | 0.0401 | +47.17% |
| Sparse | 35 | 0.0579 | 0.0567 | +2.12% |

## 5. Item Cold-Start Slice

| System | Recall@10 fresh (n = 101 authors) | Recall@10 established (n = 318 authors) |
| :--- | :---: | :---: |
| Paper Feed (Full) | 0.0223 | 0.0227 |
| Ablation: No-MMR | 0.0223 | 0.0227 |
| Ablation: No-Graph | 0.0000 | 0.0126 |
| Ablation: No-Facets | 0.0223 | 0.0228 |
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
| **Paper Feed (Full)** | 0.0071 | 0.7% | 0.0027 | 0.0344 | 0.0206 | 2.0098 | 0.0877 |
| Baseline: popularity | 0.0223 | 2.2% | 0.0087 | 0.1094 | 0.0533 | 1.5729 | 0.1223 |
| Baseline: bm25_author | 0.0072 | 0.7% | 0.0046 | 0.0469 | 0.0180 | 1.7395 | 0.1086 |
| Baseline: dense_centroid | 0.0022 | 0.2% | 0.0012 | 0.0187 | 0.0044 | 1.9960 | 0.0817 |
| Baseline: hybrid_persona | 0.0044 | 0.4% | 0.0016 | 0.0250 | 0.0112 | 1.8310 | 0.0901 |
| Anchor: Random (100-draw) | 0.0001 | 0.0% | 0.0001 | 0.0014 | 0.0004 | 3.2584 | 0.2009 |

| Comparison Baseline | Metric | System | Baseline | Rel. Gain | 95% CI of Difference | p (raw) | p (BH) | Cohen's d | MDE | Significant? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline: popularity | ndcg_at_10 | 0.0071 | 0.0223 | -67.98% | [-0.0263, -0.0050] | 5.8000e-03 | 1.6800e-02 | -0.155 | 0.0151 | YES (p < 0.05) |
| Baseline: bm25_author | ndcg_at_10 | 0.0071 | 0.0072 | -0.18% | [-0.0056, +0.0056] | 9.9650e-01 | 9.9650e-01 | -0.000 | 0.0080 | NO |
| Baseline: dense_centroid | ndcg_at_10 | 0.0071 | 0.0022 | +228.67% | [+0.0017, +0.0089] | 6.3000e-03 | 1.6800e-02 | 0.152 | 0.0051 | YES (p < 0.05) |
| Baseline: hybrid_persona | ndcg_at_10 | 0.0071 | 0.0044 | +63.24% | [-0.0008, +0.0069] | 1.5330e-01 | 2.4528e-01 | 0.080 | 0.0054 | NO |
| Baseline: popularity | recall_at_10 | 0.0027 | 0.0087 | -68.49% | [-0.0101, -0.0022] | 3.6000e-03 | 1.6800e-02 | -0.164 | 0.0056 | YES (p < 0.05) |
| Baseline: bm25_author | recall_at_10 | 0.0027 | 0.0046 | -40.57% | [-0.0058, +0.0013] | 3.0240e-01 | 3.4560e-01 | -0.057 | 0.0051 | NO |
| Baseline: dense_centroid | recall_at_10 | 0.0027 | 0.0012 | +120.52% | [-0.0000, +0.0030] | 5.6900e-02 | 1.1380e-01 | 0.108 | 0.0022 | NO |
| Baseline: hybrid_persona | recall_at_10 | 0.0027 | 0.0016 | +73.27% | [-0.0005, +0.0030] | 1.8510e-01 | 2.4680e-01 | 0.072 | 0.0025 | NO |