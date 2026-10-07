# Paper Feed: Empirical Recommender Systems Benchmark
## Non-Profile SOTA (V-SKNN) vs. Profile SOTA (PaperFlow 2026)

This repository contains an end-to-end, reproducible empirical benchmark evaluating whether maintaining a **structured user persona profile** (as proposed in recent architectures like *PaperFlow*, arXiv:2606.07454) outperforms a **State-of-the-Art Non-Profile Session Recommender** (*Vector Session-based k-NN with position decay*, Jannach & Ludewig) across **100 diverse researcher trajectories** on **3,901 real scientific papers** fetched from arXiv.

---

## 1. Quick Start: 1-Command Reproduction

If you already have Python and dependencies installed, simply run:

```bash
# Run the complete reproduction pipeline in ~8 seconds
./run_reproduction.sh
```

To launch the interactive Streamlit demonstration dashboard:
```bash
streamlit run ui/app.py --server.port 8501
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 2. Environment Setup (From Scratch)

### Prerequisites
* Python 3.10, 3.11, or 3.12 (Recommended)
* Git, bash, and a standard CPU (GPU optional, embeddings are precomputed in `data/embeddings.npy`)

### Step-by-Step Installation
```bash
# 1. Clone the repository and enter directory
cd /path/to/project

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 3. The Core Research Question

> **"Does maintaining an explicit, structured user persona profile actually outperform a State-of-the-Art Non-Profile Session Recommender that learns purely from sequential reading dynamics?"**

To avoid unfair "strawman" comparisons (e.g., comparing a profile model against a random feed), both systems are evaluated under identical conditions:
* **Non-Profile SOTA: Vector Session-based k-NN (V-SKNN)**
  * *Origin:* Jannach & Ludewig (2017), Ludewig & Jannach (2018).
  * *Mechanism:* Uses exponential decay discount over session items ($\mathbf{s}_t = \sum w_i \mathbf{d}_{(i)}$ where $w_i = e^{-\delta(t-i)}$). It operates with **zero persona text or LLM bio**.
* **Profile SOTA: PaperFlow (2026) Architecture**
  * *Origin:* arXiv:2606.07454.
  * *Mechanism:* Maintains an explicit typed persona vector ($\mathbf{p}_u$) coupled with an adaptive gated streaming memory that detects topic divergence ($\Delta_t$) and resets upon interest pivots to break out of filter bubbles.

---

## 4. Empirical Benchmark Results (100 Researchers)

Evaluated across **100 diverse researcher trajectories** (50 stationary reading trajectories, 50 cross-domain interest pivot trajectories) on the **3,901 real arXiv papers**:

| Model Architecture | Overall NDCG@10 | Overall Recall@10 | MRR | Steady-State (Stationary) | Topic Drift (Pivot Phase) | Cold-Start ($t=1, 2$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 0 (Seed-Only)** | 0.2038 | 0.1262 | 0.3882 | 0.2013 | 0.1997 | 0.1822 |
| **Baseline 1 (Static Centroid Profile)** | 0.2981 | 0.1787 | 0.5265 | 0.3848 | **0.0082** *(Crashes)* | 0.2990 |
| **Non-Profile SOTA (V-SKNN)** | **0.3179** | **0.1901** | **0.5805** | **0.3641** | **0.0904** | **0.2910** |
| **Profile SOTA (PaperFlow 2026)** | **0.4228** | **0.2538** | **0.6887** | **0.5081** | **0.0963** | **0.4907** |

### Statistical Significance:
* **Relative Gain (PaperFlow vs. V-SKNN):** **+33.00% NDCG@10**
* **Cold-Start Relative Gain ($t=1$):** **+68.62% NDCG@10**
* **Wilcoxon Signed-Rank Test:** $W = 18699.0$, **$p < 10^{-10}$** (Highly statistically significant).

---

## 5. Key Findings for the Professor & TA

1. **V-SKNN is a formidable non-profile baseline:** It achieves **0.3179 NDCG@10** (+56.0% over naive seed-only) and avoids the static centroid collapse during topic drift (0.0904 vs. 0.0082).
2. **Where Profile SOTA (PaperFlow) Conclusively Dominates:**
   * **Cold-Start ($t=1, 2$ papers):** Achieves **0.4907 vs. 0.2910** (+68.6% gain). In early sessions, interaction history is sparse, but a declared persona profile immediately constrains the search space to relevant methods and datasets.
   * **Deep Steady-State Reading:** Achieves **0.5081 vs. 0.3641** (+39.5% gain). Session decay inevitably forgets foundational papers read early in the session, whereas a typed profile retains them.
3. **Drift Resilience:** PaperFlow's gated divergence reset successfully matches/edges V-SKNN during abrupt pivots (**0.0963 vs. 0.0904**), while naive static profiles completely fail (**0.0082**).

---

## 6. Codebase Architecture

```text
├── README.md                          # This documentation & reproduction guide
├── requirements.txt                   # Pinned Python dependencies
├── run_reproduction.sh                # 1-command executable benchmark script
├── plan.md                            # Detailed scientific protocol and design doc
├── idea.md                            # Original project concept documentation
│
├── scripts/
│   ├── fetch_corpus.py                # Ingestion: Downloads 3,901 papers from arXiv API
│   ├── embed_corpus.py                # Encoding: Encodes paper abstracts on GPU (BGE model)
│   └── benchmark_sota_vs_sota.py      # Benchmark: Evaluates V-SKNN vs. PaperFlow on 100 users
│
├── data/
│   ├── corpus.jsonl                   # 3,901 real papers across cs.IR, cs.CL, cs.CV, q-bio.QM
│   ├── embeddings.npy                 # Dense 384-d normalized vector matrix (3901 x 384)
│   ├── paper_ids.json                 # Ordered paper ID mapping
│   └── sota_benchmark_results.json    # JSON dump of official benchmark metrics
│
└── ui/
    └── app.py                         # Streamlit dashboard (Benchmark table + Live feeds)
```

---

## 7. Re-running the Full Pipeline from Scratch

If you want to re-download the corpus and re-compute embeddings from the arXiv API:

```bash
# 1. Fetch real papers from arXiv API (takes ~40s)
python scripts/fetch_corpus.py

# 2. Compute dense embeddings on GPU/CPU (takes ~30s on RTX 3050 GPU)
python scripts/embed_corpus.py

# 3. Run the 100-user SOTA benchmark (takes ~8s)
python scripts/benchmark_sota_vs_sota.py

# 4. View in web UI
streamlit run ui/app.py --server.port 8501
```
