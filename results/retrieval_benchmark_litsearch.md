# Phase 7 — First-Stage Retrieval Benchmark: litsearch

> [!NOTE]
> - Corpus: 64,183 papers (7,989 without abstract, indexed by title) | Queries: 597 | Relevant: grade >= 1
> - Source: https://huggingface.co/datasets/princeton-nlp/LitSearch (license: not stated on the dataset card)
> - Dense: allenai/specter2_base + `proximity` documents, `adhoc_query` queries | Lexical: BM25s
> - Every system is scored on its top 100 (the recommender's candidate pool M); RRF fuses each channel's top 200 with k = 60.
> - Recall@k and MRR: binary relevance; nDCG@k: graded gains, 1/log2(rank+1) discount. CI: percentile bootstrap over queries (10,000 resamples).
> - Run manifest: `results/runs/retrieval_benchmark_20261112T174057Z/manifest.json`

## 1. Candidate generation over the whole corpus

Unjudged papers count as non-relevant.

| System | Recall@100 [95% CI] | nDCG@10 | nDCG@100 | MRR | Recall@5 | Recall@20 | Queries |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BM25 | 0.7190 [0.6830, 0.7541] | 0.3936 | 0.4350 | 0.3630 | 0.4460 | 0.6018 | 597 |
| SPECTER2 | 0.6604 [0.6231, 0.6979] | 0.2601 | 0.3155 | 0.2353 | 0.3074 | 0.4524 | 597 |
| RRF | 0.7780 [0.7451, 0.8098] | 0.4181 | 0.4609 | 0.3820 | 0.4855 | 0.6581 | 597 |
| RRF + CE (BAAI/bge-reranker-base) | 0.7780 [0.7451, 0.8098] | 0.4849 | 0.5177 | 0.4510 | 0.5537 | 0.6796 | 597 |
| RRF + CE (cross-encoder/ms-marco-MiniLM-L-6-v2) | 0.7780 [0.7451, 0.8098] | 0.4656 | 0.5023 | 0.4302 | 0.5319 | 0.6830 | 597 |
