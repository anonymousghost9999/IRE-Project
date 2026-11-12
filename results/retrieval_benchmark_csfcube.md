# Phase 7 — First-Stage Retrieval Benchmark: csfcube

> [!NOTE]
> - Corpus: 4,207 papers (0 without abstract, indexed by title) | Queries: 50 | Relevant: grade >= 2
> - Source: https://github.com/iesl/CSFCube (license: CC BY-NC 4.0)
> - Dense: allenai/specter2_base + `proximity` documents, `proximity` queries (query papers use their own proximity vector) | Lexical: BM25s
> - Every system is scored on its top 100 (the recommender's candidate pool M); RRF fuses each channel's top 200 with k = 60.
> - Recall@k and MRR: binary relevance; nDCG@k: graded gains, 1/log2(rank+1) discount. CI: percentile bootstrap over queries (10,000 resamples).
> - Run manifest: `results/runs/retrieval_benchmark_20261112T174057Z/manifest.json`

## 1. Candidate generation over the whole corpus

Unjudged papers count as non-relevant. The query paper itself is excluded from its results.

| System | Recall@100 [95% CI] | nDCG@10 | nDCG@100 | MRR | Recall@5 | Recall@20 | Queries |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BM25 | 0.8106 [0.7518, 0.8641] | 0.4892 | 0.6133 | 0.5962 | 0.1832 | 0.4504 | 50 |
| SPECTER2 | 0.7651 [0.6964, 0.8274] | 0.5261 | 0.6194 | 0.6282 | 0.1962 | 0.4874 | 50 |
| RRF | 0.8733 [0.8175, 0.9196] | 0.5713 | 0.6885 | 0.6773 | 0.2184 | 0.5304 | 50 |
| RRF + CE (BAAI/bge-reranker-base) | 0.8733 [0.8175, 0.9196] | 0.3464 | 0.5767 | 0.4162 | 0.1266 | 0.3224 | 50 |
| RRF + CE (cross-encoder/ms-marco-MiniLM-L-6-v2) | 0.8733 [0.8175, 0.9196] | 0.4278 | 0.6268 | 0.5207 | 0.1694 | 0.4184 | 50 |

## 2. Official CSFCube protocol (re-rank each query's annotated pool)

Every pool document is ranked; documents a channel did not return come last. Plan metrics:

| System | Recall@100 [95% CI] | nDCG@10 | nDCG@100 | MRR | Recall@5 | Recall@20 | Queries |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BM25 | 0.9478 [0.9160, 0.9761] | 0.5088 | 0.7268 | 0.6193 | 0.1937 | 0.4795 | 50 |
| SPECTER2 | 0.9760 [0.9510, 0.9948] | 0.5715 | 0.7693 | 0.6320 | 0.2067 | 0.5631 | 50 |
| RRF | 0.9674 [0.9410, 0.9888] | 0.5842 | 0.7781 | 0.6765 | 0.2300 | 0.5663 | 50 |
| RRF + CE (BAAI/bge-reranker-base) | 0.9348 [0.8955, 0.9698] | 0.3576 | 0.6351 | 0.3837 | 0.1581 | 0.3552 | 50 |
| RRF + CE (cross-encoder/ms-marco-MiniLM-L-6-v2) | 0.9478 [0.9138, 0.9777] | 0.4483 | 0.7037 | 0.5160 | 0.1812 | 0.4629 | 50 |

CSFCube's official metrics (re-implemented from its `eval_scripts`: MAP and R-Precision at grade >= 2, nDCG over the top 20% of the pool with CSFCube's discount; R-Precision follows CSFCube's definition, precision at the rank of the last relevant paper). Use these for comparison with published CSFCube results; the system is not facet-aware, so each query paper gets the same ranking for every facet.

| System | background: MAP / R-Prec / nDCG%20 | method: MAP / R-Prec / nDCG%20 | result: MAP / R-Prec / nDCG%20 | all: MAP / R-Prec / nDCG%20 |
| :--- | :---: | :---: | :---: | :---: |
| BM25 | 0.4583 / 0.1913 / 0.6641 | 0.2529 / 0.0955 / 0.4109 | 0.3343 / 0.1613 / 0.5318 | 0.3463 / 0.1485 / 0.5330 |
| SPECTER2 | 0.5197 / 0.2707 / 0.7106 | 0.2637 / 0.1041 / 0.4506 | 0.4186 / 0.2242 / 0.6470 | 0.3983 / 0.1982 / 0.6006 |
| RRF | 0.5297 / 0.2640 / 0.7343 | 0.2701 / 0.1085 / 0.4898 | 0.4004 / 0.1882 / 0.6285 | 0.3975 / 0.1854 / 0.6152 |
| RRF + CE (BAAI/bge-reranker-base) | 0.2593 / 0.1596 / 0.4744 | 0.1617 / 0.1015 / 0.3354 | 0.2200 / 0.1368 / 0.3807 | 0.2127 / 0.1321 / 0.3953 |
| RRF + CE (cross-encoder/ms-marco-MiniLM-L-6-v2) | 0.4236 / 0.2419 / 0.6163 | 0.2207 / 0.1168 / 0.3775 | 0.2984 / 0.1464 / 0.4862 | 0.3120 / 0.1669 / 0.4909 |
