# V1 Encoder Study: csfcube (retrieval benchmark, reported, not decisive)

> [!NOTE]
> - design §13.8 V1: SciNCL and BGE are the Track 1 candidates; Qwen3-Embedding-0.6B was released after T and is shown here only. Each encoder uses its documented text format (V1 point 2); SPECTER2 keeps its adapters. BM25s is the same for every encoder.
> - Top 100 scored; RRF fuses each channel's top 200 with k = 60. CIs: bootstrap over queries (10,000 resamples).
> - Run manifest: `results/runs/encoder_benchmark_20261113T225451Z/manifest.json`

| Encoder | Model | Dim | Embedding s / 1k docs |
| :--- | :--- | :---: | :---: |
| SPECTER2 | allenai/specter2_base | 768 | 9.88 |
| SciNCL | malteos/scincl | 768 | 8.15 |
| BGE | BAAI/bge-base-en-v1.5 | 768 | 7.39 |
| Qwen3-Emb-0.6B (post-T) | Qwen/Qwen3-Embedding-0.6B | 1024 | 50.57 |

## corpus: whole corpus

| System | Recall@100 [95% CI] | nDCG@10 | nDCG@100 | MRR | Recall@5 | Recall@20 | Queries |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BM25 | 0.8106 [0.7518, 0.8641] | 0.4892 | 0.6133 | 0.5962 | 0.1832 | 0.4504 | 50 |
| Dense: SPECTER2 | 0.7651 [0.6964, 0.8274] | 0.5261 | 0.6194 | 0.6282 | 0.1962 | 0.4874 | 50 |
| RRF: SPECTER2 | 0.8733 [0.8175, 0.9196] | 0.5713 | 0.6885 | 0.6773 | 0.2184 | 0.5304 | 50 |
| Dense: SciNCL | 0.8015 [0.7380, 0.8590] | 0.5174 | 0.6191 | 0.5860 | 0.2080 | 0.4622 | 50 |
| RRF: SciNCL | 0.8732 [0.8191, 0.9185] | 0.5543 | 0.6892 | 0.6402 | 0.2271 | 0.5086 | 50 |
| Dense: BGE | 0.8107 [0.7460, 0.8681] | 0.5213 | 0.6268 | 0.6021 | 0.2070 | 0.4870 | 50 |
| RRF: BGE | 0.8714 [0.8155, 0.9186] | 0.5569 | 0.6821 | 0.6203 | 0.2287 | 0.5131 | 50 |
| Dense: Qwen3-Emb-0.6B (post-T) | 0.8347 [0.7696, 0.8925] | 0.5570 | 0.6652 | 0.6932 | 0.2396 | 0.5644 | 50 |
| RRF: Qwen3-Emb-0.6B (post-T) | 0.8739 [0.8196, 0.9195] | 0.5659 | 0.6872 | 0.6613 | 0.2424 | 0.5103 | 50 |

| Compared with SPECTER2 | Recall@100 diff [95% CI] | nDCG@10 diff [95% CI] |
| :--- | :---: | :---: |
| Dense: SciNCL | +0.0364 [+0.0055, +0.0698] | -0.0087 [-0.0341, +0.0173] |
| Dense: BGE | +0.0456 [+0.0048, +0.0884] | -0.0048 [-0.0353, +0.0254] |
| Dense: Qwen3-Emb-0.6B (post-T) | +0.0696 [+0.0233, +0.1168] | +0.0309 [+0.0044, +0.0599] |
| RRF: SciNCL | -0.0001 [-0.0173, +0.0176] | -0.0169 [-0.0393, +0.0054] |
| RRF: BGE | -0.0018 [-0.0281, +0.0241] | -0.0144 [-0.0390, +0.0086] |
| RRF: Qwen3-Emb-0.6B (post-T) | +0.0006 [-0.0227, +0.0252] | -0.0054 [-0.0258, +0.0141] |

## pool: official CSFCube pool (re-rank each query's annotated pool)

| System | Recall@100 [95% CI] | nDCG@10 | nDCG@100 | MRR | Recall@5 | Recall@20 | Queries |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BM25 | 0.9478 [0.9160, 0.9761] | 0.5088 | 0.7268 | 0.6193 | 0.1937 | 0.4795 | 50 |
| Dense: SPECTER2 | 0.9760 [0.9510, 0.9948] | 0.5715 | 0.7693 | 0.6320 | 0.2067 | 0.5631 | 50 |
| RRF: SPECTER2 | 0.9674 [0.9410, 0.9888] | 0.5842 | 0.7781 | 0.6765 | 0.2300 | 0.5663 | 50 |
| Dense: SciNCL | 0.9740 [0.9512, 0.9921] | 0.5681 | 0.7687 | 0.5906 | 0.2234 | 0.5361 | 50 |
| RRF: SciNCL | 0.9710 [0.9487, 0.9893] | 0.5848 | 0.7812 | 0.6644 | 0.2468 | 0.5465 | 50 |
| Dense: BGE | 0.9779 [0.9587, 0.9931] | 0.5638 | 0.7659 | 0.6127 | 0.2270 | 0.5477 | 50 |
| RRF: BGE | 0.9697 [0.9473, 0.9887] | 0.5728 | 0.7704 | 0.6353 | 0.2440 | 0.5321 | 50 |
| Dense: Qwen3-Emb-0.6B (post-T) | 0.9741 [0.9520, 0.9922] | 0.5969 | 0.7898 | 0.6992 | 0.2485 | 0.6176 | 50 |
| RRF: Qwen3-Emb-0.6B (post-T) | 0.9765 [0.9583, 0.9917] | 0.5838 | 0.7775 | 0.6658 | 0.2457 | 0.5692 | 50 |

| Compared with SPECTER2 | Recall@100 diff [95% CI] | nDCG@10 diff [95% CI] |
| :--- | :---: | :---: |
| Dense: SciNCL | -0.0020 [-0.0119, +0.0072] | -0.0034 [-0.0308, +0.0245] |
| Dense: BGE | +0.0019 [-0.0109, +0.0192] | -0.0077 [-0.0367, +0.0213] |
| Dense: Qwen3-Emb-0.6B (post-T) | -0.0019 [-0.0224, +0.0198] | +0.0254 [-0.0019, +0.0550] |
| RRF: SciNCL | +0.0036 [-0.0024, +0.0122] | +0.0006 [-0.0141, +0.0155] |
| RRF: BGE | +0.0023 [-0.0048, +0.0114] | -0.0114 [-0.0306, +0.0072] |
| RRF: Qwen3-Emb-0.6B (post-T) | +0.0090 [+0.0000, +0.0238] | -0.0004 [-0.0184, +0.0174] |
