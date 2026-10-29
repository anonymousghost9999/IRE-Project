# Phase 2 Benchmark Report: Representations & Retrieval Indices

**Date Generated:** 2026-10-29 08:27:25 UTC  
**Candidate Pool:** 110,229 papers (evaluated $\le 2024$)  
**Dense Architecture:** `allenai/specter2_base` (adapters: `proximity` + `adhoc_query`, fp16)  
**Lexical Engine:** `bm25s` with English stop-word tokenization  
**Fusion Method:** Reciprocal Rank Fusion (RRF, $k=60$, $M_{channel}=200$, $M=100$)  

---

## 1. Storage & Memory Footprint

| Artifact | Type | Storage Size | Notes |
|---|---|---|---|
| `embeddings.npy` | Dense Tensor | **325.63 MB** | 768-d float32 L2-normalized embeddings |
| `faiss.index` | FAISS Index | **322.94 MB** | Exact inner-product (`IndexFlatIP`) |
| `paper_ids.json` | ID Mapping | **1.57 MB** | Internal ID $\leftrightarrow$ OpenAlex ID mapping |
| `bm25s/` | Inverted Index | **87.67 MB** | Serialized sparse lexical index |
| **Total Footprint** | Combined | **737.81 MB** | Fits easily in server/client memory |

---

## 2. Offline Build Throughput

| Build Metric | Measurement | Description |
|---|---|---|
| Total Candidate Papers | **110,229** | Papers $\le 2024$ with non-empty abstracts |
| Embedding Device | **`cuda`** | Hardware acceleration |
| Total Embedding Time | **968.28 s** | SPECTER2 forward pass in fp16 |
| Embedding Rate | **8.71 s / 1k papers** | Throughput per 1,000 papers |
| FAISS Build Time | **0.82 s** | Time to populate IndexFlatIP |
| BM25s Build Time | **9.48 s** | Text tokenization and inverted index build |

---

## 3. Online Retrieval Latency Benchmark

Measured across 4 representative CS/AI/IR domain queries (100k+ candidates). Each stage is timed once at the production channel depth ($M_{channel}=200$); the RRF row is the fusion step alone, and the end-to-end row is the sum of the four stages.

| Retrieval Pipeline Stage | Mean Latency | Min Latency | Max Latency | Target / Requirement |
|---|---|---|---|---|
| **Query Encoding (SPECTER2)** | **29.6 ms** | 18.8 ms | 33.6 ms | $< 100$ ms |
| **Dense Search (FAISS FlatIP)** | **14.28 ms** | 13.89 ms | 14.62 ms | $< 25$ ms |
| **Lexical Search (BM25s)** | **14.88 ms** | 13.90 ms | 16.38 ms | $< 20$ ms |
| **RRF Rank Fusion ($k=60$)** | **0.17 ms** | 0.16 ms | 0.17 ms | $< 5$ ms |
| **End-to-End First Stage** | **58.9 ms** | 47.7 ms | 63.3 ms | **Interactive (< 150 ms)** |

---

## 4. Qualitative Sanity Check: Top-5 Hybrid Recommendations

### Query: *"Dense passage retrieval and bi-encoder ranking for open-domain question answering"*
- **Retrieval Latency:** 47.7 ms (Dense: 14.2 ms, BM25s: 14.5 ms, RRF: 0.2 ms)

| Rank | Year | OpenAlex ID | Title | Citations ($\le 2024$) | RRF Score |
|---|---|---|---|---|---|
| 1 | 2020 | `W3015883388` | Dense Passage Retrieval for Open-Domain Question Answering | 1849 | 0.03202 |
| 2 | 2021 | `W3168875417` | RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering | 286 | 0.03128 |
| 3 | 2021 | `W3138967041` | Open Domain Question Answering over Tables via Dense Retrieval | 53 | 0.02939 |
| 4 | 2003 | `W2077428231` | Quantitative evaluation of passage retrieval algorithms for question answering | 137 | 0.02901 |
| 5 | 2018 | `W4289865932` | R3: Reinforced Ranker-Reader for Open-Domain Question Answering | 281 | 0.02837 |

### Query: *"Transformer architecture for language modeling and self-attention mechanisms"*
- **Retrieval Latency:** 62.3 ms (Dense: 14.6 ms, BM25s: 13.9 ms, RRF: 0.2 ms)

| Rank | Year | OpenAlex ID | Title | Citations ($\le 2024$) | RRF Score |
|---|---|---|---|---|---|
| 1 | 2023 | `W4313558932` | End-to-End Transformer-Based Models in Textual-Based NLP | 57 | 0.03078 |
| 2 | 2020 | `W2988945824` | Improving Transformer Models by Reordering their Sublayers | 57 | 0.03012 |
| 3 | 2020 | `W4288795042` | SG-Net: Syntax Guided Transformer for Language Representation | 4 | 0.02973 |
| 4 | 2019 | `W2946567085` | Adaptive Attention Span in Transformers | 250 | 0.02895 |
| 5 | 2023 | `W4386921320` | Self-Attention and Transformers: Driving the Evolution of Large Language Models | 5 | 0.02889 |

### Query: *"Graph neural networks and node representation learning on citation graphs"*
- **Retrieval Latency:** 63.3 ms (Dense: 13.9 ms, BM25s: 16.4 ms, RRF: 0.2 ms)

| Rank | Year | OpenAlex ID | Title | Citations ($\le 2024$) | RRF Score |
|---|---|---|---|---|---|
| 1 | 2018 | `W2804057010` | Representation Learning on Graphs with Jumping Knowledge Networks | 1188 | 0.03252 |
| 2 | 2023 | `W4323338660` | Fuzzy Representation Learning on Graph | 12 | 0.03102 |
| 3 | 2018 | `W2894175828` | How Powerful are Graph Neural Networks? | 2422 | 0.03012 |
| 4 | 2016 | `W2406128552` | Learning Convolutional Neural Networks for Graphs | 1484 | 0.02722 |
| 5 | 2021 | `W3169741886` | Subset Node Representation Learning over Large Dynamic Graphs | 11 | 0.02703 |

### Query: *"Contrastive learning for self-supervised visual representation"*
- **Retrieval Latency:** 62.2 ms (Dense: 14.3 ms, BM25s: 14.8 ms, RRF: 0.2 ms)

| Rank | Year | OpenAlex ID | Title | Citations ($\le 2024$) | RRF Score |
|---|---|---|---|---|---|
| 1 | 2020 | `W3005680577` | A Simple Framework for Contrastive Learning of Visual Representations | 9301 | 0.03227 |
| 2 | 2021 | `W3204265989` | Weakly Supervised Contrastive Learning | 89 | 0.03128 |
| 3 | 2021 | `W3035725276` | Self-supervised Learning: Generative or Contrastive | 1302 | 0.03054 |
| 4 | 2021 | `W3196792593` | Preservational Learning Improves Self-supervised Medical Image Models by Reconstructing Diverse Contexts | 71 | 0.02920 |
| 5 | 2020 | `W3018378048` | Supervised Contrastive Learning | 2926 | 0.02861 |
