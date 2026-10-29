# Phase 2 Benchmark Report: Representations & Retrieval Indices

**Date Generated:** 2026-10-29 07:14:42 UTC  
**Candidate Pool:** 111,149 papers (evaluated $\le 2024$)  
**Dense Architecture:** `allenai/specter2_base` (adapters: `proximity` + `adhoc_query`, fp16)  
**Lexical Engine:** `bm25s` with English stop-word tokenization  
**Fusion Method:** Reciprocal Rank Fusion (RRF, $k=60$, $M_{channel}=200$, $M=100$)  

---

## 1. Storage & Memory Footprint

| Artifact | Type | Storage Size | Notes |
|---|---|---|---|
| `embeddings.npy` | Dense Tensor | **325.63 MB** | 768-d float32 L2-normalized embeddings |
| `faiss.index` | FAISS Index | **325.63 MB** | Exact inner-product (`IndexFlatIP`) |
| `paper_ids.json` | ID Mapping | **1.59 MB** | Internal ID $\leftrightarrow$ OpenAlex ID mapping |
| `bm25s/` | Inverted Index | **87.80 MB** | Serialized sparse lexical index |
| **Total Footprint** | Combined | **740.65 MB** | Fits easily in server/client memory |

---

## 2. Offline Build Throughput

| Build Metric | Measurement | Description |
|---|---|---|
| Total Candidate Papers | **111,149** | Papers $\le 2024$ with non-empty abstracts |
| Embedding Device | **`cuda`** | Hardware acceleration |
| Total Embedding Time | **954.55 s** | SPECTER2 forward pass in fp16 |
| Embedding Rate | **8.59 s / 1k papers** | Throughput per 1,000 papers |
| FAISS Build Time | **0.38 s** | Time to populate IndexFlatIP |
| BM25s Build Time | **13.59 s** | Text tokenization and inverted index build |

---

## 3. Online Retrieval Latency Benchmark

Measured across 4 representative CS/AI/IR domain queries (100k+ candidates):

| Retrieval Pipeline Stage | Mean Latency | Min Latency | Max Latency | Target / Requirement |
|---|---|---|---|---|
| **Query Encoding (SPECTER2)** | **17.5 ms** | 16.2 ms | 20.0 ms | $< 100$ ms |
| **Dense Search (FAISS FlatIP)** | **9.88 ms** | 8.76 ms | 10.78 ms | $< 25$ ms |
| **Lexical Search (BM25s)** | **1.53 ms** | 1.16 ms | 2.58 ms | $< 20$ ms |
| **RRF Rank Fusion ($k=60$)** | **11.43 ms** | 8.82 ms | 17.23 ms | $< 5$ ms |
| **End-to-End First Stage** | **40.4 ms** | 36.7 ms | 49.2 ms | **Interactive (< 150 ms)** |

---

## 4. Qualitative Sanity Check: Top-5 Hybrid Recommendations

### Query: *"Dense passage retrieval and bi-encoder ranking for open-domain question answering"*
- **Retrieval Latency:** 49.2 ms (Dense: 10.8 ms, BM25s: 1.2 ms, RRF: 17.2 ms)

| Rank | Year | OpenAlex ID | Title | Citations ($\le 2024$) | RRF Score |
|---|---|---|---|---|---|
| 1 | 2020 | `W3015883388` | Dense Passage Retrieval for Open-Domain Question Answering | 1849 | 0.03202 |
| 2 | 2021 | `W3168875417` | RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering | 286 | 0.03128 |
| 3 | 2021 | `W3138967041` | Open Domain Question Answering over Tables via Dense Retrieval | 53 | 0.02939 |
| 4 | 2003 | `W2077428231` | Quantitative evaluation of passage retrieval algorithms for question answering | 137 | 0.02901 |
| 5 | 2022 | `W4385573057` | Improving Passage Retrieval with Zero-Shot Question Generation | 38 | 0.02830 |

### Query: *"Transformer architecture for language modeling and self-attention mechanisms"*
- **Retrieval Latency:** 36.7 ms (Dense: 9.7 ms, BM25s: 1.2 ms, RRF: 9.7 ms)

| Rank | Year | OpenAlex ID | Title | Citations ($\le 2024$) | RRF Score |
|---|---|---|---|---|---|
| 1 | 2023 | `W4313558932` | End-to-End Transformer-Based Models in Textual-Based NLP | 57 | 0.03078 |
| 2 | 2020 | `W2988945824` | Improving Transformer Models by Reordering their Sublayers | 57 | 0.03012 |
| 3 | 2020 | `W4288795042` | SG-Net: Syntax Guided Transformer for Language Representation | 4 | 0.02973 |
| 4 | 2019 | `W2946567085` | Adaptive Attention Span in Transformers | 250 | 0.02895 |
| 5 | 2023 | `W4386921320` | Self-Attention and Transformers: Driving the Evolution of Large Language Models | 5 | 0.02889 |

### Query: *"Graph neural networks and node representation learning on citation graphs"*
- **Retrieval Latency:** 38.0 ms (Dense: 8.8 ms, BM25s: 2.6 ms, RRF: 10.0 ms)

| Rank | Year | OpenAlex ID | Title | Citations ($\le 2024$) | RRF Score |
|---|---|---|---|---|---|
| 1 | 2018 | `W2804057010` | Representation Learning on Graphs with Jumping Knowledge Networks | 1188 | 0.03252 |
| 2 | 2023 | `W4323338660` | Fuzzy Representation Learning on Graph | 12 | 0.03102 |
| 3 | 2018 | `W2894175828` | How Powerful are Graph Neural Networks? | 2422 | 0.03012 |
| 4 | 2016 | `W2406128552` | Learning Convolutional Neural Networks for Graphs | 1484 | 0.02722 |
| 5 | 2021 | `W3169741886` | Subset Node Representation Learning over Large Dynamic Graphs | 11 | 0.02692 |

### Query: *"Contrastive learning for self-supervised visual representation"*
- **Retrieval Latency:** 37.6 ms (Dense: 10.3 ms, BM25s: 1.2 ms, RRF: 8.8 ms)

| Rank | Year | OpenAlex ID | Title | Citations ($\le 2024$) | RRF Score |
|---|---|---|---|---|---|
| 1 | 2020 | `W3005680577` | A Simple Framework for Contrastive Learning of Visual Representations | 9301 | 0.03227 |
| 2 | 2021 | `W3204265989` | Weakly Supervised Contrastive Learning | 89 | 0.03128 |
| 3 | 2021 | `W3035725276` | Self-supervised Learning: Generative or Contrastive | 1302 | 0.03054 |
| 4 | 2021 | `W3196792593` | Preservational Learning Improves Self-supervised Medical Image Models by Reconstructing Diverse Contexts | 71 | 0.02920 |
| 5 | 2020 | `W3018378048` | Supervised Contrastive Learning | 2926 | 0.02861 |
