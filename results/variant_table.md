# Phase 10 — Final Variant Table (plan 10.5)

> [!NOTE]
> - Every variant was decided on the 80 tuning authors before its single test run on the same 320 test authors (design §13.8). This table reports; it is never used to choose a variant (Phase 11 uses tuning evidence only).
> - Variant vs Core: paired bootstrap over authors (10,000 resamples), two-sided, Benjamini-Hochberg across all variant x metric comparisons. All variants are exploratory; the Core run answers H1.
> - ILD is computed with each run's own embeddings, so V1 (BGE) ILD is not comparable with the SPECTER2 rows. V1's Dense Centroid and Hybrid-Persona baselines also use BGE. V3 ranks authors with papers only; its zero-publication slice is the linear scorer.

## 1. Variants

| Variant | What changes | Tuning-split NDCG@10 gain [95% CI] | Decision | NDCG@10 | Recall@10 | HitRate@10 | MRR | Pool Recall@100 | Topic Entropy | ILD |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Core** | linear score over the SPECTER2 + BM25s pool | - | reference | 0.0216 | 0.0156 | 0.1281 | 0.0557 | 0.0510 | 1.5405 | 0.0808 |
| V1 encoder (BGE) | dense encoder bge-base-en-v1.5; grid re-tuned with it | +0.0105 (+49%) [-0.0019, +0.0237] | not adopted | 0.0262 | 0.0200 | 0.1594 | 0.0658 | 0.0620 | 1.5151 | 0.2177* |
| V2 cross-encoder | + w_ce * reranker score (ms-marco-MiniLM-L-6-v2) | +0.0021 (+9%) [-0.0006, +0.0056] | not adopted | 0.0205 | 0.0154 | 0.1219 | 0.0482 | 0.0510 | 1.5072 | 0.0822 |
| V3 learned ranker | LightGBM LambdaMART instead of the linear score | +0.0022 (+12%) [-0.0135, +0.0150] | not adopted | 0.0202 | 0.0136 | 0.1125 | 0.0520 | 0.0510 | 1.6645 | 0.0927 |
| V4 citation channel | + citation-graph candidate channel in RRF | +0.0061 (+25%) [-0.0033, +0.0168] | not adopted | 0.0264 | 0.0197 | 0.1750 | 0.0592 | 0.0901 | 1.5141 | 0.0893 |
| *Baseline: Citation-only* | graph signal over the whole corpus | - | - | 0.0348 | 0.0273 | 0.1875 | 0.0711 | - | 1.6173 | - |

*\* not comparable: computed with BGE embeddings. Pool Recall@100 for Citation-only is not defined (it has no pool).*

## 2. Each variant against Core on the test split

| Variant | Metric | Variant | Core | Difference [95% CI] | Rel. | p (raw) | p (BH) | Cohen's d | Significant? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| V1 encoder (BGE) | ndcg_at_10 | 0.0262 | 0.0216 | +0.0046 [-0.0015, +0.0109] | +21.4% | 0.1416 | 0.2832 | +0.081 | no |
| V1 encoder (BGE) | recall_at_10 | 0.0200 | 0.0156 | +0.0044 [+0.0004, +0.0086] | +27.9% | 0.0374 | 0.1536 | +0.116 | no |
| V2 cross-encoder | ndcg_at_10 | 0.0205 | 0.0216 | -0.0011 [-0.0034, +0.0011] | -5.1% | 0.3316 | 0.4421 | -0.054 | no |
| V2 cross-encoder | recall_at_10 | 0.0154 | 0.0156 | -0.0002 [-0.0023, +0.0018] | -1.4% | 0.8279 | 0.8279 | -0.012 | no |
| V3 learned ranker | ndcg_at_10 | 0.0202 | 0.0216 | -0.0014 [-0.0064, +0.0037] | -6.6% | 0.5747 | 0.6568 | -0.031 | no |
| V3 learned ranker | recall_at_10 | 0.0136 | 0.0156 | -0.0021 [-0.0057, +0.0017] | -13.3% | 0.2633 | 0.4213 | -0.063 | no |
| V4 citation channel | ndcg_at_10 | 0.0264 | 0.0216 | +0.0047 [-0.0002, +0.0105] | +21.9% | 0.0795 | 0.2120 | +0.097 | no |
| V4 citation channel | recall_at_10 | 0.0197 | 0.0156 | +0.0040 [+0.0003, +0.0080] | +25.9% | 0.0384 | 0.1536 | +0.116 | no |
