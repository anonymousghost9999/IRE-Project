# V4 Citation-Graph Channel: Tuning-Split Decision

Rule (design §13.8, fixed before implementation): V4 replaces the Core pool only if the paired-bootstrap 95% CI of its tuning-split mean NDCG@10 gain over Core (each at its own grid optimum) lies above zero (design §13.8). 80 tuning authors, 10,000 paired-bootstrap resamples. The test split is not touched.

| Arm | Grid optimum | Tuning NDCG@10 | Candidate Recall@100 |
| :--- | :--- | :---: | :---: |
| Core (text channels) | w_cite=1.0, w_facet=0.25, w_nov=0.25, lambda_div=0.5 | 0.0241 | 0.0426 |
| V4 (+ citation-graph channel) | w_cite=1.0, w_facet=0.5, w_nov=0.0, lambda_div=0.0 | 0.0302 | 0.0781 |

- NDCG@10 gain (V4 - Core): +0.0061 (+25.5%), 95% CI [-0.0033, +0.0168], p = 0.2403, Cohen's d = 0.130
- Candidate Recall@100 gain (mechanism check): +0.0355, 95% CI [+0.0167, +0.0576]

**Decision: keep the Core candidate pool** (the CI of the NDCG@10 gain does not lie above zero).

*Either way V4 gets one Track 1 test run (`run_track1_eval.py --graph-channel`) as a labelled exploratory variant (plan 10.4d); the Core test run remains the answer to H1.*
