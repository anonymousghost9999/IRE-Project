# PaperFeed: Grounded, Explainable Scientific Paper Recommendation Engine

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](#environment-setup)
[![Tests](https://img.shields.io/badge/pytest-333%20passed%2C%201%20skipped-brightgreen.svg)](#running-tests)
[![Temporal Leakage](https://img.shields.io/badge/leakage%20check-clean%20(0%20violations)-success.svg)](#2-cohort-selection-ground-truth--leakage-check)
[![SciERC Core F1](https://img.shields.io/badge/SciERC%20Core%20Relaxed%20F1-41.15%25-orange.svg)](#7-scierc-extraction-benchmark-phase-34)

**PaperFeed** is an AI-powered personalized scientific literature feed and paper recommender. Built on an evaluation-first paradigm, it models researcher interests using temporal history, extracts verbatim scientific facets (*Task, Method, Metric, Material/Dataset, Topic*) via open-vocabulary entity extraction (GLiNER), and provides fully grounded, counterfactually verifiable explanations for every recommended paper under strict time-bounded constraints.

---

## Table of Contents
1. [Core Architecture & Evaluation Formulation](#core-architecture--evaluation-formulation)
2. [Project Implementation Status](#project-implementation-status)
3. [Environment Setup](#environment-setup)
   - [1. Requirements](#1-requirements)
   - [2. Virtual Environment & Dependencies](#2-virtual-environment--dependencies)
   - [3. Environment Configuration (`.env`)](#3-environment-configuration-env)
4. [Step-by-Step Pipeline Execution](#step-by-step-pipeline-execution)
   - [1. Data Crawling & Graph Expansion](#1-data-crawling--graph-expansion)
   - [2. Cohort Selection, Ground Truth & Leakage Check](#2-cohort-selection-ground-truth--leakage-check)
   - [3. Index Building (Dense & Lexical)](#3-index-building-dense--lexical)
   - [4. First-Stage Retrieval Verification](#4-first-stage-retrieval-verification)
   - [5. Facet Extraction & Canonicalization](#5-facet-extraction--canonicalization)
   - [6. Facet Profile & Grounding Audit](#6-facet-profile--grounding-audit)
   - [7. SciERC Extraction Benchmark](#7-scierc-extraction-benchmark-phase-34)
   - [8. Recommender Engine Verification](#8-recommender-engine-verification)
   - [9. Running Tests](#9-running-tests)
   - [10. Track 1 Leave-Future-Out Evaluation](#10-track-1-leave-future-out-evaluation-phase-6)
   - [11. First-Stage Retrieval Benchmark](#11-first-stage-retrieval-benchmark-phase-7)
5. [Benchmark Results Summary](#benchmark-results-summary)
6. [Repository Structure](#repository-structure)

---

## Core Architecture & Evaluation Formulation

The system addresses the **Leave-Future-Out (LFO)** formulation:
- **Temporal Cutoff $T$ ($2024$):** All candidate papers, persona embeddings, citation graph traversals, and user facets are strictly time-bounded to publication years $\le T$.
- **Future Target Paper $B$ ($T+1 = 2025$):** A future paper authored by evaluation researchers serves as natural ground truth.
- **Discovery Ground Truth:** $\text{Discovery}(B \mid u) = \text{Ref}_{\le T}(B) \setminus (\text{Citations}(u, \le T) \cup \text{Authored}(u, \le T))$. The goal is recommending papers the author had *not* previously cited or authored, but subsequently cited in their future work.
- **Explainability & Grounding:** Recommends Top-$K$ items diversified via Maximal Marginal Relevance (MMR) where every recommendation features a grounded template explanation (verbatim facet match, new-method indicator, bibliographic coupling, co-citation, or session/persona similarity).

```
OpenAlex (<= 2024) ──► SQLite Corpus (157k papers, 7.6M edges)
                             │
                             ├─► Dense Index (SPECTER2 + FAISS IndexFlatIP)
                             ├─► Lexical Index (BM25s)
                             └─► Facet Extraction (GLiNER + Canonicalization)
                                       │
User Profile (<= 2024) ────────────────┼─► Hybrid Retrieval (RRF, M=100)
                                       │         │
Dynamic Session State ─────────────────┼─► 6-Feature Linear Scoring
(EMA + Cosine Pivot Gate)              │         │
                                       ├─► MMR Diversification (K=10)
                                       │         │
                                       └─► Grounded Explanations (100% Verifiable)
```

---

## Project Implementation Status

| Phase | Description | Status | Verification & Artifacts |
|---|---|---|---|
| **Phase 0** | **Project Setup & Architecture** | Complete | Clean skeleton, configuration (`configs/default.yaml`), logging, SHA256 disk cache, manifests. |
| **Phase 1** | **Data Acquisition & Leakage-Free Corpus** | Complete | 157,117 papers, 7,622,860 citation edges in SQLite. 400 cohort authors (310 warm, 46 moderate, 44 sparse) split 80 tuning / 320 test. Leakage check clean (`data/processed/data_report.md`). |
| **Phase 2** | **Representations & Hybrid Indexes** | Complete | SPECTER2 base + proximity & query adapters, FAISS `IndexFlatIP`, BM25s lexical index, RRF ($k=60$). Verified in `results/phase2_benchmark_report.md`. |
| **Phase 3** | **Facet Extraction & Canonicalization** | Complete | GLiNER entity extraction (`task`, `method`, `dataset`, `metric`, `topic`), exact character-offset verbatim grounding guard, greedy leader clustering into `facet_canon`. Audit passed (`results/facet_audit_report.md`). |
| **Phase 3.4** | **SciERC Gold Extraction Benchmark** | Complete | Full benchmark suite evaluated on official SciERC test split (Luan et al., 2018). Strict & relaxed multiset matching, confidence threshold sweep, 3-label core averages recorded (`results/scierc_benchmark_report.md`). |
| **Phase 4** | **User Modeling & Dynamic Session State** | Complete | Recency-weighted persona vectors ($\gamma=0.9$), user facet profiles $\mathcal{F}_u$, exclusion sets $\mathcal{E}_u$, dynamic session state $\mathbf{v}_t^{\text{sess}}$ (EMA $\alpha=0.5$), negative skip penalty set $\mathcal{S}_u^-$, cosine pivot gate ($\gamma_{\text{pivot}}=0.35$). |
| **Phase 5** | **Recommendation Engine** | Complete | `recommend(resources, profile, session, mode)`: Hybrid candidate generation ($M=100$), multi-feature linear scoring, greedy MMR diversity ($\lambda=0.2$), deterministic grounded explanations. End-to-end audit passed (`results/phase5_verification_report.md`: 222ms latency, 100% explanation coverage, 0 leaks). |
| **Phase 6** | **Track 1: Full LFO Benchmark Runs** | Complete | 320 test authors, run manifests in `results/runs/track1_*`. Headline (`results/track1_lfo_report.md`): $H_1$ **rejected**: significantly better than Dense, Hybrid-Persona and (Recall@10) Popularity/BM25, but significantly worse than Citation-only (NDCG@10 0.0216 vs 0.0348), whose graph search covers the whole corpus while the system re-ranks a text-retrieved pool (Recall@100 ceiling 0.051). $H_2$ supported. Enhanced run (M=250, w_pop=0.2): Citation-only gap not significant, $H_2$ not testable (tuned $\lambda_{div}=0$). Popularity-filtered run: $H_2$ fails narrowly (worst-case loss 3.14%). Facet-active variant: none qualified (`results/facet_config_selection*.md`). |
| **Phase 7** | **Retrieval Benchmark (CSFCube, LitSearch)** | Complete | BM25s, SPECTER2 and RRF (+ cross-encoder rows) on each benchmark's corpus; CSFCube also under official pool protocol. Tables in `results/retrieval_benchmark_{csfcube,litsearch}.md`. |
| **Phase 8** | **Explanation Faithfulness Audit** | Complete | `results/faithfulness_audit_report.md` (+ `_enhanced`, `_filtered`): 100 stratified explanations, grounding 100%, counterfactual drop-out 100%, coverage 100%; templates persona/citation only. The filtered audit equals the main one (the popularity filter changes targets, not recommendations). No span-quoting explanation occurs at the tuned weights; `--span-check` (w_facet = w_nov = 1, labelled supplementary) exercises span grounding. |
| **Phases 9–12** | **Extended Track 2 Simulation, SOTA Variants, Best Configuration & SOTA Comparison, UI** | Planned | Empty module skeletons in `src/paperfeed/sim/`, `src/paperfeed/variants/`, and `src/paperfeed/ui/`. Phase 11 assembles the best option for each part from the Phase 10 tuning evidence and compares it with current SOTA paper recommenders (`docs/plan.md` Phase 11). |

---

## Environment Setup

### 1. Requirements
- Linux / macOS (tested on Linux x86_64, CUDA available)
- Python $\ge$ 3.10 (tested through Python 3.14)
- Recommended: GPU with $\ge$ 4GB VRAM for batched SPECTER2 and GLiNER inference. CPU execution is supported out-of-the-box.

### 2. Virtual Environment & Dependencies
```bash
# Clone the repository
git clone https://github.com/anonymousghost9999/IRE-Project.git
cd IRE-Project

# Create and activate virtual environment
python -m venv .venv
# On Linux / macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install editable package (or pip install -e ".[ml]" for full ML stack)
pip install -e .
```

### 3. Environment Configuration (`.env`)
Copy `.env.example` to `.env` and add your OpenAlex details/credentials:
```bash
cp .env.example .env
```
Open `.env` and configure:
- `OPENALEX_EMAIL`: Your academic or personal email. OpenAlex requires this to grant access to the polite pool (up to 10 requests/sec).
- `OPENALEX_API_KEY`: *(Optional)* Your free API key from [OpenAlex](https://openalex.org) for elevated crawling limits (up to 100 requests/sec) to avoid 429 budget errors.

---

## Step-by-Step Pipeline Execution

All pipeline steps read centralized parameters from [`configs/default.yaml`](configs/default.yaml) and can be executed via standalone scripts in `scripts/`.

### 1. Data Crawling & Graph Expansion
> [!NOTE]
> Ensure you have created `.env` from `.env.example` with your `OPENALEX_EMAIL` before executing data acquisition.

Mines candidate 2025 authors from OpenAlex and crawls the multi-hop time-bounded citation graph up to cutoff year $T=2024$:
```bash
python scripts/build_corpus.py --max-authors 1000 --target-corpus-size 120000
```
- **Stages executed:** `authored` $\to$ `hop1` $\to$ `hop2` $\to$ `anchors` $\to$ `fill`.
- **Output:** Raw responses saved to `data/raw/`, populated SQLite tables in `data/processed/paperfeed.sqlite3`.

To inspect candidate authors:
```bash
python scripts/analyze_authors.py
```

### 2. Cohort Selection, Ground Truth & Leakage Check
Filters target papers $B$, validates that no citations to $B$ exist prior to $T+1$, stratifies the cohort by publication history (Warm, Moderate, Sparse), constructs ground truth, and executes strict temporal leakage assertions:
```bash
python scripts/build_cohort.py --authors-file data/candidate_authors.jsonl --out-dir data/processed
```
- **Outputs generated:**
  - `data/processed/cohort.jsonl` (400 frozen evaluation authors)
  - `data/processed/ground_truth.jsonl` (Discovery & Re-citation targets)
  - `data/processed/split.json` (80 tuning / 320 test authors)
  - `data/processed/data_report.json` & `data/processed/data_report.md`
- **Leakage check:** Asserts `Clean: True` (0 papers or citation edges with year $> 2024$).

### 3. Index Building (Dense & Lexical)
Computes L2-normalized SPECTER2 embeddings for all corpus abstracts and builds both the FAISS `IndexFlatIP` vector index and `bm25s` lexical index:
```bash
python scripts/build_index.py --db-path data/processed/paperfeed.sqlite3 --out-dir data/processed
```
- **Outputs generated:**
  - `data/processed/embeddings.npy` & `data/processed/embedding_ids.json` (Checkpointable embedding store)
  - `data/processed/faiss.index` (Dense inner-product index)
  - `data/processed/bm25s/` (Serialized BM25 index)
  - `data/processed/index_manifest.json` (Latency & indexing metadata)

### 4. First-Stage Retrieval Verification
Benchmarks latency, memory usage, and retrieval recall for query encoding, dense retrieval, BM25s lexical search, and hybrid RRF fusion:
```bash
python scripts/verify_retrieval.py --k 10 --out-report results/phase2_benchmark_report.md
```
- **Output:** Formatted verification report in `results/phase2_benchmark_report.md`.

### 5. Facet Extraction & Canonicalization
Extracts typed scientific entities from paper abstracts with GLiNER, verifies exact character-level offsets against abstract text, and clusters them per type into canonical facets:
```bash
# Full candidate corpus extraction
python scripts/extract_facets.py --threshold 0.5 --batch-size 32

# Or limit to sample (for quick dry runs)
python scripts/extract_facets.py --limit 500
```
- **Outputs generated:** `paper_facets` and `facet_canon` tables in SQLite, plus `data/processed/facet_manifest.json`.

### 6. Facet Profile & Grounding Audit
Executes Phase 3.3 automated audit over a stratified sample of 100 candidate papers to verify verbatim grounding rates, core facet density, and canonical cluster connectivity:
```bash
python scripts/audit_facets.py --sample-size 100
```
- **Outputs generated:**
  - `results/facet_audit_report.md` & `results/facet_audit_report.json`
  - `results/facet_annotation_sample.json` (Precision@facet human annotation sample)

### 7. SciERC Extraction Benchmark (Phase 3.4)
Evaluates the extraction pipeline on the official SciERC Gold Standard test split (Luan et al., 2018):
```bash
# 1. Download SciERC official test split (if not already downloaded)
mkdir -p data/raw/scierc
curl -L -o data/raw/scierc/test.json https://huggingface.co/datasets/sthoran/scierc_processed_data/resolve/main/test.json

# 2. Run standard production evaluation (with threshold sweep 0.3-0.7)
python scripts/evaluate_scierc.py --sweep 0.3,0.4,0.5,0.6,0.7

# 3. Run native-labels evaluation
python scripts/evaluate_scierc.py --native-labels --sweep 0.3,0.4,0.5,0.6,0.7
```
- **Outputs generated:**
  - `results/scierc_benchmark_report.json` & `results/scierc_benchmark_report.md`
  - `results/scierc_benchmark_report_native.json` & `results/scierc_benchmark_report_native.md`

### 8. Recommender Engine Verification
Executes an end-to-end audit of `recommend()` across cohort strata (Warm, Moderate, Sparse, and Zero-Publication) in both Feed mode and Track 2 dynamic session mode:
```bash
python scripts/verify_recommender.py --n-sample 2 --report results/phase5_verification_report.md
```
- **Verification checks:**
  - Feed generation latency (Average: **222 ms**, well below the 500 ms SLA).
  - Explanation coverage (**100%** grounded explanations generated).
  - Temporal leakage (**0** violations).
  - Interactive session progression (positive click updates persona EMA, negative skip applies penalty).
- **Output:** `results/phase5_verification_report.md`.

### 9. Running Tests
Run the entire offline test suite covering all phases (fixtures require zero network or GPU):
```bash
pytest
```
*Current test suite: **333 passed, 1 skipped** (total 334 tests across all modules).*

To run specific test modules:
```bash
pytest tests/test_rec.py       # Recommendation engine & features
pytest tests/test_user.py      # Persona, exclusion & session state
pytest tests/test_scierc.py    # SciERC benchmark evaluation module
pytest tests/test_facets.py    # GLiNER extraction & canonicalization
pytest tests/test_index.py     # Embeddings, FAISS & BM25s
pytest tests/test_cohort.py    # Cohort stratification & leakage tests
```

---

### 10. Track 1 Leave-Future-Out Evaluation (Phase 6)
Tunes on the 80 tuning authors, then runs the full system, ablations, the five baselines, the random anchor and all slices once on the 320 test authors. Needs the full canonicalized facet database and loads SPECTER2 (title-only authored papers and seed topics are embedded on the fly):
```bash
# Main run (plan configuration)
python scripts/run_track1_eval.py

# Secondary analyses, each labelled in its report
python scripts/run_track1_eval.py --enhanced                 # M=250, M_channel=500, w_pop=0.20
python scripts/run_track1_eval.py --filter-popular 200       # targets with citation_count_T >= 200 removed

# Quick check on a few authors (not a result)
python scripts/run_track1_eval.py --test-sample 10 --tuning-sample 10 --n-bootstrap 500 --skip-zero-pub
```
- **Outputs:** `results/<report-name>.md`, `results/<results-name>.json` (per-user metrics and top-10 lists), `results/<results-name>_zero_pub.json`, and `results/runs/<run_id>/manifest.json` with copies of the outputs, the git commit, config, arguments, tuned values, input-file hashes and timings.
- **Tuning cache:** one file per configuration, `data/artifacts/tuning_results_<signature hash>.json`, so the main, `--filter-popular` and `--enhanced` runs never overwrite each other. A file is reused only when its stored configuration signature matches the run (an older shared `tuning_results.json` is still picked up on a match); otherwise the run re-tunes. The manifest records which file was used.
- The headline is the main run; the popularity filter removes the targets the Popularity baseline can hit, so it is a secondary analysis only.

### 11. First-Stage Retrieval Benchmark (Phase 7)
Compares BM25s, SPECTER2 and RRF (k = 60, top 200 per channel, top 100 scored) as candidate generators on two public benchmarks, each with its own corpus and indexes built with the Phase 2 code:
```bash
python scripts/prepare_retrieval_benchmarks.py --download   # CSFCube (CC BY-NC 4.0) and LitSearch (~1.3 GB raw parquet)
python scripts/build_benchmark_indexes.py                   # ~1 min CSFCube, ~8 min LitSearch on GPU; resumable
python scripts/run_retrieval_benchmark.py [--cross-encoders]
```
- **CSFCube** (50 query papers x facet, graded 0-3, relevant >= 2): *corpus mode* retrieves from all 4,207 released papers; *pool mode* is the official protocol (re-rank each query's annotated pool) and adds CSFCube's own MAP / R-Precision / nDCG%20 per facet for comparison with published results.
- **LitSearch** (597 questions, 64,183 papers, `adhoc_query` adapter): corpus mode; Recall@5 / @20 are included for comparison with published numbers.
- **Metrics:** Recall@100 (primary, with 95% bootstrap CI), nDCG@10 / @100 (graded), MRR. Outputs: `results/retrieval_benchmark_<name>.{md,json}` and a run manifest in `results/runs/`.

## Benchmark Results Summary

### SciERC Extraction Benchmark (Phase 3.4 / Test Split, 100 Documents)
Evaluated on official SciERC gold annotations:

| Metric Track | Strict Precision | Strict Recall | Strict F1 | Relaxed Precision | Relaxed Recall | Relaxed F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Core 3 Labels (`Task`, `Method`, `Metric`)** | **36.59%** | **26.96%** | **31.05%** | **48.49%** | **35.73%** | **41.15%** |
| **All 5 Labels (Micro-Average)** | 34.18% | 20.50% | 25.63% | 44.92% | 26.94% | 33.68% |
| **All 5 Labels (Macro-Average)** | 35.70% | 27.97% | 25.49% | 46.51% | 36.44% | 33.44% |

#### Confidence Threshold Sweep
| Confidence Threshold | Strict Micro-F1 | Relaxed Micro-F1 | Relaxed Precision | Relaxed Recall |
|:---:|:---:|:---:|:---:|:---:|
| `0.30` | 27.62% | **37.18%** | 37.46% | 36.91% |
| `0.40` | 26.85% | 36.26% | 41.39% | 32.27% |
| `0.50` (default) | 25.63% | 33.68% | 44.92% | 26.94% |
| `0.60` | 23.02% | 29.70% | 47.85% | 21.54% |
| `0.70` | 20.53% | 26.40% | 54.19% | 17.45% |

### Recommendation Engine KPIs (Phase 5 Verification)
Tested across all cohort strata (Warm, Moderate, Sparse, Zero-Publication):
- **Average Feed Latency:** `222.2 ms` (Design Budget: `< 500 ms`)
- **Explanation Grounding Coverage:** `100.0%` (Design Target: `≥ 90.0%`)
- **Temporal Leakage Violations:** `0 papers` (Design Constraint: strictly 0)

---

## Repository Structure

```text
├── configs/
│   └── default.yaml                # Master configuration file (T, K, M, weights, thresholds)
├── data/                           # (Git-ignored data directory)
│   ├── raw/                        # Compressed raw OpenAlex responses & scierc/ test data
│   └── processed/
│       ├── paperfeed.sqlite3       # SQLite corpus (157k papers, 7.6M edges, facets)
│       ├── cohort.jsonl            # 400 frozen evaluation authors
│       ├── ground_truth.jsonl      # Leave-Future-Out discovery targets
│       ├── split.json              # 80 tuning / 320 test stratification
│       └── data_report.md          # Corpus distribution & leakage verification report
├── results/                        # Benchmark and audit reports
│   ├── scierc_benchmark_report.md  # SciERC 3.4 extraction report
│   ├── facet_audit_report.md       # Phase 3.3 profile audit report
│   ├── phase2_benchmark_report.md  # Phase 2 retrieval benchmark report
│   └── phase5_verification_report.md # Phase 5 end-to-end recommender audit report
├── scripts/                        # Numbered and standalone execution entry points
│   ├── build_corpus.py             # Phase 1: Mine authors & crawl corpus
│   ├── analyze_authors.py          # Phase 1: Author candidate inspection
│   ├── build_cohort.py             # Phase 1: Cohort selection & leakage check
│   ├── build_index.py              # Phase 2: Compute embeddings & build FAISS/BM25s
│   ├── verify_retrieval.py         # Phase 2: Retrieval benchmark & latency test
│   ├── extract_facets.py           # Phase 3: GLiNER extraction & canonicalization
│   ├── audit_facets.py             # Phase 3.3: In-corpus profile audit
│   ├── evaluate_scierc.py          # Phase 3.4: SciERC benchmark evaluation
│   ├── verify_recommender.py       # Phase 5: End-to-end recommender verification
│   ├── run_track1_eval.py          # Phase 6: Track 1 LFO evaluation (tuning, test run, slices, statistics)
│   ├── prepare_retrieval_benchmarks.py # Phase 7: download + normalize CSFCube and LitSearch
│   ├── build_benchmark_indexes.py  # Phase 7: SPECTER2 + BM25s indexes per benchmark corpus
│   └── run_retrieval_benchmark.py  # Phase 7: BM25 / SPECTER2 / RRF (+ cross-encoder) tables
├── src/paperfeed/                  # Core package source
│   ├── openalex/                   # REST client & abstract reconstruction
│   ├── data/                       # Database manager, schema, graph, cohort, leakage
│   ├── index/                      # SPECTER2 encoder, FAISS dense, BM25s, hybrid RRF
│   ├── facets/                     # GLiNER extractor, sentence splitter, canonicalizer
│   ├── user/                       # Persona vectors, user facets, exclusion sets, session state
│   ├── rec/                        # Candidate retrieval, feature scoring, MMR, explainability
│   ├── bench/                      # SciERC benchmark evaluation module
│   ├── common/                     # Config loader, manifest generator, disk cache, logger
│   ├── eval/                       # Metrics, baselines, statistical significance (Phase 6)
│   ├── sim/                        # Track 2 LLM session simulation (Phase 9)
│   ├── variants/                   # SOTA variants (Phase 10), best configuration & SOTA comparison (Phase 11)
│   └── ui/                         # Streamlit interactive interface (Phase 12)
├── tests/                          # Automated test suite (334 unit & integration tests)
├── pyproject.toml                  # Packaging & dependencies
└── requirements.txt                # Pinned dependency manifest
```
