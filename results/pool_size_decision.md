# Phase 11a: Candidate-Pool Size, Tuning-Split Decision

Rule (design §13.9, fixed before implementation): The larger candidate pool (M = 250, M_channel = 500, no popularity prior) replaces the Core pool (M = 100, M_channel = 200) only if the paired-bootstrap 95% CI of its tuning-split NDCG@10 gain over Core (each at its own grid optimum) lies above zero (design §13.9, Phase 11 specification point 2). 80 tuning authors, 10,000 paired-bootstrap resamples. The test split is not touched.

| Arm | Pool | Grid optimum | Tuning NDCG@10 | Candidate Recall@k |
| :--- | :--- | :--- | :---: | :---: |
| Core | M = 100, M_channel = 200 | w_cite=1.0, w_facet=0.25, w_nov=0.25, lambda_div=0.5 | 0.0241 | 0.0426 |
| Larger pool | M = 250, M_channel = 500, w_pop = 0 | w_cite=1.0, w_facet=0.0, w_nov=0.25, lambda_div=0.0 | 0.0245 | 0.0743 |

- NDCG@10 gain (larger - Core): +0.0004 (+1.7%), 95% CI [-0.0050, +0.0060], p = 0.8827, Cohen's d = 0.016
- Candidate recall gain (mechanism check; the pool sizes differ): +0.0317, 95% CI [+0.0159, +0.0507]

**Decision: keep the Core pool** (the CI of the NDCG@10 gain does not lie above zero).

*This is a single change, so it is also the §13.9 assembled-system check. The decision goes into `configs/best.yaml` (scripts/freeze_best_config.py) before any comparison system runs on the test split.*
