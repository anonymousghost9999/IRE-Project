# Phase 11b: RP3β Tuning (tuning split)

Grid fixed in advance (design §13.9): alpha [0.5, 1.0, 1.5], beta [0.0, 0.25, 0.5, 0.75] (beta = 0 is P3α). 80 tuning authors; citation graph of 3,415,898 papers and 7,622,860 edges (citing year <= 2024). The test split is not touched.

| alpha | beta | Tuning NDCG@10 |
| :---: | :---: | :---: |
| 0.5 | 0.0 | 0.0652 |
| 0.5 | 0.25 | 0.0661 **(chosen)** |
| 0.5 | 0.5 | 0.0497 |
| 0.5 | 0.75 | 0.0297 |
| 1.0 | 0.0 | 0.0654 |
| 1.0 | 0.25 | 0.0560 |
| 1.0 | 0.5 | 0.0342 |
| 1.0 | 0.75 | 0.0217 |
| 1.5 | 0.0 | 0.0466 |
| 1.5 | 0.25 | 0.0345 |
| 1.5 | 0.5 | 0.0188 |
| 1.5 | 0.75 | 0.0110 |
