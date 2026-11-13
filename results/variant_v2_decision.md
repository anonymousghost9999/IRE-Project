# V2 Cross-Encoder Feature: Tuning-Split Decision

Rule (design §13.8, fixed before implementation): The reranker with the higher tuning-split NDCG@10 at its own grid optimum is the V2 candidate; it replaces Core only if the paired-bootstrap 95% CI of its NDCG@10 gain over Core lies above zero, and never when its optimum has w_ce = 0 (design §13.8, V2 point 5). 80 tuning authors, 10,000 paired-bootstrap resamples. The test split is not touched. Picking the better of two rerankers before the test slightly favours V2.

| Arm | Grid optimum | Tuning NDCG@10 | Reranker latency per feed (mean / p95) |
| :--- | :--- | :---: | :---: |
| Core | w_cite=1.0, w_facet=0.25, w_nov=0.25, w_ce=0.0, lambda_div=0.5 | 0.0241 | - |
| V2: BAAI/bge-reranker-base | w_cite=1.0, w_facet=0.25, w_nov=0.25, w_ce=0.0, lambda_div=0.5 | 0.0241 | 3.95 s / 4.36 s |
| V2: cross-encoder/ms-marco-MiniLM-L-6-v2 (chosen) | w_cite=1.0, w_facet=0.25, w_nov=0.5, w_ce=0.25, lambda_div=0.0 | 0.0262 | 0.44 s / 0.50 s |

- NDCG@10 gain of the chosen reranker over Core: +0.0021 (+8.6%), 95% CI [-0.0006, +0.0056], p = 0.1888, Cohen's d = 0.145
- Latency: GPU NVIDIA GeForce RTX 3050 6GB Laptop GPU, 100 candidates per feed

**Decision: keep Core** (the CI of the NDCG@10 gain does not lie above zero).

*Either way V2 gets one Track 1 test run (`run_track1_eval.py --cross-encoder`) as a labelled exploratory variant (plan 10.2d), unless the chosen optimum has w_ce = 0, in which case it is Core and there is nothing to run.*
