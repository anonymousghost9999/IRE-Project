# V3 Learned Ranker: Tuning-Split Decision

Rule (design §13.8, fixed before implementation): V3 replaces the linear scorer only if the paired-bootstrap 95% CI of its out-of-fold NDCG@10 gain (5 author folds inside the tuning split, both scorers fitted on the same training folds) lies above zero (design §13.8, V3 point 5). 80 tuning authors, 10,000 paired-bootstrap resamples. The test split is not touched.

- Model: LightGBM lambdarank with fixed settings {"objective": "lambdarank", "n_estimators": 100, "learning_rate": 0.05, "num_leaves": 7, "max_depth": 3, "min_child_samples": 20, "reg_lambda": 1.0, "colsample_bytree": 0.8, "subsample": 1.0, "deterministic": true, "random_state": 42}
- Features: persona, citation, facet, new_method, dense_rank, lexical_rank

| Fold | Held-out authors | Linear optimum (fitted on the other folds) | V3 lambda_div | Linear NDCG@10 | V3 NDCG@10 |
| :---: | :---: | :--- | :---: | :---: | :---: |
| 0 | 16 | w_cite=0.5, w_facet=0.0, w_nov=0.25, lambda_div=0.0 | 0.3 | 0.0185 | 0.0191 |
| 1 | 16 | w_cite=1.0, w_facet=0.5, w_nov=0.25, lambda_div=0.2 | 0.5 | 0.0201 | 0.0215 |
| 2 | 16 | w_cite=1.0, w_facet=0.25, w_nov=0.5, lambda_div=0.5 | 0.5 | 0.0000 | 0.0168 |
| 3 | 16 | w_cite=0.5, w_facet=0.25, w_nov=0.0, lambda_div=0.5 | 0.5 | 0.0087 | 0.0250 |
| 4 | 16 | w_cite=1.0, w_facet=0.25, w_nov=0.25, lambda_div=0.5 | 0.0 | 0.0465 | 0.0226 |

- Out-of-fold NDCG@10: linear 0.0188, V3 0.0210
- Gain (V3 - linear): +0.0022 (+12.0%), 95% CI [-0.0135, +0.0150], p = 0.7527, Cohen's d = 0.035
- Final model (all 80 tuning authors): lambda_div = 0.0, feature importance (splits) {'persona': 122, 'citation': 159, 'facet': 38, 'new_method': 8, 'dense_rank': 101, 'lexical_rank': 153}

**Decision: keep the linear scorer** (the CI of the out-of-fold NDCG@10 gain does not lie above zero).

*Either way V3 gets one Track 1 test run (`run_track1_eval.py --learned-ranker`) as a labelled exploratory variant (plan 10.3d), with the final model above.*
