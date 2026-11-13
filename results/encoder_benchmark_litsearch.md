# V1 Encoder Study: litsearch (retrieval benchmark, reported, not decisive)

> [!NOTE]
> - design §13.8 V1: SciNCL and BGE are the Track 1 candidates; Qwen3-Embedding-0.6B was released after T and is shown here only. Each encoder uses its documented text format (V1 point 2); SPECTER2 keeps its adapters. BM25s is the same for every encoder.
> - Top 100 scored; RRF fuses each channel's top 200 with k = 60. CIs: bootstrap over queries (10,000 resamples).
> - Run manifest: `results/runs/encoder_benchmark_20261113T225842Z/manifest.json`

| Encoder | Model | Dim | Embedding s / 1k docs |
| :--- | :--- | :---: | :---: |
| SPECTER2 | allenai/specter2_base | 768 | 7.44 |
| SciNCL | malteos/scincl | 768 | 3.74 |
| BGE | BAAI/bge-base-en-v1.5 | 768 | 3.94 |
| Qwen3-Emb-0.6B (post-T) | Qwen/Qwen3-Embedding-0.6B | 1024 | 29.92 |

## corpus: whole corpus

| System | Recall@100 [95% CI] | nDCG@10 | nDCG@100 | MRR | Recall@5 | Recall@20 | Queries |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BM25 | 0.7190 [0.6830, 0.7541] | 0.3936 | 0.4350 | 0.3630 | 0.4460 | 0.6018 | 597 |
| Dense: SPECTER2 | 0.6604 [0.6231, 0.6979] | 0.2601 | 0.3155 | 0.2353 | 0.3074 | 0.4524 | 597 |
| RRF: SPECTER2 | 0.7780 [0.7451, 0.8098] | 0.4181 | 0.4609 | 0.3820 | 0.4855 | 0.6581 | 597 |
| Dense: SciNCL | 0.5844 [0.5448, 0.6238] | 0.2413 | 0.2899 | 0.2207 | 0.2739 | 0.4303 | 597 |
| RRF: SciNCL | 0.7612 [0.7270, 0.7939] | 0.3910 | 0.4408 | 0.3624 | 0.4504 | 0.6271 | 597 |
| Dense: BGE | 0.7925 [0.7606, 0.8229] | 0.4488 | 0.4899 | 0.4142 | 0.5040 | 0.6685 | 597 |
| RRF: BGE | 0.8279 [0.7983, 0.8561] | 0.4885 | 0.5325 | 0.4608 | 0.5444 | 0.6942 | 597 |
| Dense: Qwen3-Emb-0.6B (post-T) | 0.8540 [0.8244, 0.8811] | 0.5308 | 0.5664 | 0.4930 | 0.6128 | 0.7464 | 597 |
| RRF: Qwen3-Emb-0.6B (post-T) | 0.8749 [0.8477, 0.9003] | 0.5135 | 0.5604 | 0.4833 | 0.5760 | 0.7210 | 597 |

| Compared with SPECTER2 | Recall@100 diff [95% CI] | nDCG@10 diff [95% CI] |
| :--- | :---: | :---: |
| Dense: SciNCL | -0.0760 [-0.1160, -0.0358] | -0.0189 [-0.0485, +0.0112] |
| Dense: BGE | +0.1321 [+0.0966, +0.1670] | +0.1886 [+0.1578, +0.2191] |
| Dense: Qwen3-Emb-0.6B (post-T) | +0.1936 [+0.1556, +0.2310] | +0.2707 [+0.2383, +0.3026] |
| RRF: SciNCL | -0.0168 [-0.0391, +0.0057] | -0.0271 [-0.0513, -0.0026] |
| RRF: BGE | +0.0499 [+0.0273, +0.0734] | +0.0704 [+0.0481, +0.0931] |
| RRF: Qwen3-Emb-0.6B (post-T) | +0.0969 [+0.0712, +0.1231] | +0.0954 [+0.0728, +0.1186] |
