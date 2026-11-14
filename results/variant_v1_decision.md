# V1 Encoder Study: Tuning-Split Decision

Rule (design §13.8, fixed before implementation): The alternative encoder with the higher full-system tuning NDCG@10 at its own grid optimum is the V1 candidate; it replaces SPECTER2 only if the paired-bootstrap 95% CI of its gain over SPECTER2 (both on the 20k subset, each at its own optimum) lies above zero (design §13.8, V1 point 6). 80 tuning authors, 10,000 paired-bootstrap resamples. The test split is not touched.

- Subset: 20,000 papers (1,929 tuning-author papers, 6,021 one-hop and 12,050 two-hop citation neighbours, chosen from the <= T graph only); 67.2% of the tuning authors' Discovery targets fall inside it. Candidates are restricted to the subset, so absolute numbers are not comparable with Track 1.
- BM25s and facets are shared by all encoders; Qwen3-Embedding (post-T) is excluded from this decision.

| Encoder | Grid optimum | Full system NDCG@10 | Dense Centroid NDCG@10 | Candidate Recall@100 |
| :--- | :--- | :---: | :---: | :---: |
| SPECTER2 | w_cite=1.0, w_facet=0.25, w_nov=0.25, lambda_div=0.0 | 0.0215 | 0.0067 | 0.0634 |
| SciNCL | w_cite=1.0, w_facet=0.25, w_nov=0.25, lambda_div=0.0 | 0.0278 | 0.0174 | 0.0663 |
| BGE (V1 candidate) | w_cite=1.0, w_facet=0.5, w_nov=0.0, lambda_div=0.1 | 0.0320 | 0.0122 | 0.0786 |

- Full-system NDCG@10 gain of BGE over SPECTER2: +0.0105 (+48.8%), 95% CI [-0.0019, +0.0237], p = 0.1099, Cohen's d = 0.179
- Dense Centroid gain of SciNCL over SPECTER2: +0.0107, 95% CI [-0.0008, +0.0233] (reported, not decisive)
- Dense Centroid gain of BGE over SPECTER2: +0.0055, 95% CI [-0.0021, +0.0132] (reported, not decisive)

**Decision: keep SPECTER2** (the CI of the full-system NDCG@10 gain does not lie above zero).

*Either way the V1 candidate (BGE) gets the full corpus embedded, its grid re-tuned on the full tuning split and one labelled Track 1 test run (plan 10.1d).*
