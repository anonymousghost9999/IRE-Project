# Phase 11c: Scholar Inbox-style Classifier Tuning (tuning split)

Grid fixed in advance (design §13.9): C [0.01, 0.1, 1.0, 10.0]. GTE-large vectors of 111,149 papers, PCA-256 (explained variance 80.2%); per author a logistic regression on E_u positives vs 10 seeded eligible negatives each. 80 tuning authors; the test split is not touched.

| C | Tuning NDCG@10 |
| :---: | :---: |
| 0.01 | 0.0218 |
| 0.1 | 0.0232 **(chosen)** |
| 1.0 | 0.0228 |
| 10.0 | 0.0197 |
