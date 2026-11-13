# Paper Feed — Track 1: Leave-Future-Out Empirical Evaluation Report

> [!NOTE]
> **Evaluation Run Configuration:**
> - Mode: Baseline
> - Candidate Pool Size (M): 100
> - Candidate Channels: SPECTER2 + BM25s
> - Scorer (V3): learned ranker variant_v3_model.txt, its lambda_div=0.00 (the linear weights below are used only for the zero-publication slice)
> - Channel Depth (M_channel): 200
> - Popularity Prior Weight (w_pop): 0.00
> - Scoring Weights: w_pers=1.00, w_cite=1.00, w_facet=0.25, w_nov=0.25, lambda_div=0.50
> - System variant: V3 learned ranker, LightGBM LambdaMART (design §13.8; exploratory, plan 10.3d); tuning-split decision: keep the linear scorer; the zero-publication slice keeps the linear scorer
> - Hybrid-Persona baseline: gamma_rec=1.00, n_terms=5
> - Report: `track1_lfo_report_learned_ranker.md` | Raw JSON: `track1_lfo_results_learned_ranker.json` | Run manifest: `results/runs/track1_track1_lfo_report_learned_ranker_20261113T220205Z/manifest.json`


## 1. Main Benchmark Results

| System Architecture | NDCG@10 | % of Oracle | Recall@10 | HitRate@10 | MRR | Topic Entropy | ILD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Feed (Full)** | 0.0202 | 2.0% | 0.0136 | 0.1125 | 0.0520 | 1.6645 | 0.0927 |
| Ablation: No-MMR | 0.0202 | 2.0% | 0.0136 | 0.1125 | 0.0520 | 1.6645 | 0.0927 |
| Ablation: No-Graph | 0.0112 | 1.1% | 0.0081 | 0.0781 | 0.0238 | 1.8602 | 0.0955 |
| Ablation: No-Facets | 0.0207 | 2.1% | 0.0141 | 0.1219 | 0.0536 | 1.6573 | 0.0896 |
| Baseline: popularity | 0.0129 | 1.3% | 0.0062 | 0.0656 | 0.0286 | 1.6884 | 0.1260 |
| Baseline: bm25_author | 0.0142 | 1.4% | 0.0094 | 0.0813 | 0.0279 | 1.6350 | 0.0993 |
| Baseline: dense_centroid | 0.0111 | 1.1% | 0.0085 | 0.0688 | 0.0224 | 1.7728 | 0.0691 |
| Baseline: citation_only | 0.0348 | 3.5% | 0.0273 | 0.1875 | 0.0711 | 1.6173 | 0.1092 |
| Baseline: hybrid_persona | 0.0128 | 1.3% | 0.0093 | 0.0813 | 0.0321 | 1.7977 | 0.0895 |
| Anchor: Random (100-draw) | 0.0001 | 0.0% | 0.0001 | 0.0013 | 0.0004 | 3.2613 | 0.2011 |

### Candidate Coverage Ceiling

| System | Candidate pool Recall@100 (ceiling) | Recall@10 | Share of ceiling reached |
| :--- | :---: | :---: | :---: |
| Paper Feed (Full) | 0.0510 | 0.0136 | 26.6% |
| Ablation: No-MMR | 0.0510 | 0.0136 | 26.6% |
| Ablation: No-Graph | 0.0510 | 0.0081 | 15.8% |
| Ablation: No-Facets | 0.0510 | 0.0141 | 27.6% |

*The pool comes from the text channels (BM25 + SPECTER2, RRF); targets outside it cannot be recommended whatever the scoring weights. For reference, Citation-only ranks the whole eligible corpus by the graph signal and reaches Recall@10 = 0.0273.*

### Diagnostic: Re-Citation

| Run | Ground truth | Recall@10 | NDCG@10 | HitRate@10 | Candidate Recall@100 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Diagnostic: Re-Citation | ReCitation (own past references cited again after T) | 0.0338 | 0.0325 | 0.1125 | 0.0781 |

*Citations(u, <= T) are removed from the exclusion set, so the feed may re-recommend the author's own past references. Different ground truth from every other table: not comparable with the Discovery numbers.*

## 2. Component Ablations

| Configuration | NDCG@10 | Delta NDCG | Recall@10 | Topic Entropy | Delta Entropy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Paper Feed (Full) | 0.0202 | +0.0000 | 0.0136 | 1.6645 | +0.0000 |
| Ablation: No-MMR | 0.0202 | +0.0000 | 0.0136 | 1.6645 | +0.0000 |
| Ablation: No-Graph | 0.0112 | -0.0090 | 0.0081 | 1.8602 | +0.1957 |
| Ablation: No-Facets | 0.0207 | +0.0005 | 0.0141 | 1.6573 | -0.0072 |

## 3. Statistical Hypothesis Testing

### Hypothesis Testing Summary

**Sample Size:** N = 320 test researchers | **Bootstrap Resamples:** B = 10,000

#### Hypothesis H1: Paper Feed Outperforms the Five Baselines (NDCG@10 and Recall@10)
**Verdict:** REJECTED / INCONCLUSIVE

| Comparison Baseline | Metric | System | Baseline | Rel. Gain | 95% CI of Difference | p (raw) | p (BH) | Cohen's d | MDE | Significant? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline: popularity | ndcg_at_10 | 0.0202 | 0.0129 | +55.96% | [-0.0031, +0.0180] | 1.6880e-01 | 1.7060e-01 | 0.077 | 0.0148 | NO |
| Baseline: bm25_author | ndcg_at_10 | 0.0202 | 0.0142 | +42.35% | [-0.0023, +0.0148] | 1.7060e-01 | 1.7060e-01 | 0.076 | 0.0123 | NO |
| Baseline: dense_centroid | ndcg_at_10 | 0.0202 | 0.0111 | +82.34% | [+0.0020, +0.0168] | 1.5500e-02 | 5.1667e-02 | 0.134 | 0.0106 | NO |
| Baseline: citation_only | ndcg_at_10 | 0.0202 | 0.0348 | -41.92% | [-0.0255, -0.0035] | 1.0800e-02 | 5.1667e-02 | -0.142 | 0.0159 | NO |
| Baseline: hybrid_persona | ndcg_at_10 | 0.0202 | 0.0128 | +57.56% | [+0.0013, +0.0141] | 2.3500e-02 | 5.2000e-02 | 0.126 | 0.0091 | NO |
| Baseline: popularity | recall_at_10 | 0.0136 | 0.0062 | +117.26% | [+0.0011, +0.0143] | 3.0700e-02 | 5.2000e-02 | 0.120 | 0.0095 | NO |
| Baseline: bm25_author | recall_at_10 | 0.0136 | 0.0094 | +43.79% | [-0.0011, +0.0102] | 1.4680e-01 | 1.7060e-01 | 0.080 | 0.0081 | NO |
| Baseline: dense_centroid | recall_at_10 | 0.0136 | 0.0085 | +59.07% | [+0.0006, +0.0097] | 3.1200e-02 | 5.2000e-02 | 0.121 | 0.0065 | NO |
| Baseline: citation_only | recall_at_10 | 0.0136 | 0.0273 | -50.40% | [-0.0218, -0.0059] | 6.0000e-04 | 6.0000e-03 | -0.188 | 0.0113 | YES (p < 0.05) |
| Baseline: hybrid_persona | recall_at_10 | 0.0136 | 0.0093 | +44.98% | [+0.0002, +0.0087] | 5.1100e-02 | 7.3000e-02 | 0.109 | 0.0060 | NO |

*BH correction runs over all baseline x metric comparisons. MDE: smallest true difference detectable with 80% power at alpha = 0.05 and this N; a non-significant row whose observed difference is below its MDE is inconclusive, not evidence of no effect.*

#### Hypothesis H2: MMR Diversity with Non-Inferiority
**Verdict:** NOT TESTABLE: the tuned lambda_div is 0, so the full system and the No-MMR ablation are identical and there is no MMR effect to measure.


## 4. History Slice Stratification

| Stratum | Authors (n) | Paper Feed NDCG@10 | Best Baseline NDCG@10 | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| Warm | 248 | 0.0166 | 0.0309 | -46.07% |
| Moderate | 37 | 0.0315 | 0.0401 | -21.58% |
| Sparse | 35 | 0.0334 | 0.0567 | -41.14% |

## 5. Item Cold-Start Slice

| System | Recall@10 fresh (n = 101 authors) | Recall@10 established (n = 318 authors) |
| :--- | :---: | :---: |
| Paper Feed (Full) | 0.0157 | 0.0158 |
| Ablation: No-MMR | 0.0157 | 0.0158 |
| Ablation: No-Graph | 0.0045 | 0.0049 |
| Ablation: No-Facets | 0.0157 | 0.0162 |
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