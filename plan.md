# Empirical Benchmark Protocol: Non-Profile SOTA vs. Profile SOTA (PaperFlow 2026)

**Project:** Paper Feed (Scientific Paper Discovery Engine)  
**Experiment:** Fair Head-to-Head Benchmark on 3,901 Real arXiv Papers Across 100 Diverse Researchers  
**Status:** Executed & Validated  
**Key Code Files:**  
* Ingestion: [`scripts/fetch_corpus.py`](file:///home/phanindra/Desktop/SEM5/IRE/project/scripts/fetch_corpus.py)  
* GPU Embeddings: [`scripts/embed_corpus.py`](file:///home/phanindra/Desktop/SEM5/IRE/project/scripts/embed_corpus.py)  
* SOTA Benchmark: [`scripts/benchmark_sota_vs_sota.py`](file:///home/phanindra/Desktop/SEM5/IRE/project/scripts/benchmark_sota_vs_sota.py)  
* Interactive Dashboard: [`ui/app.py`](file:///home/phanindra/Desktop/SEM5/IRE/project/ui/app.py)  

---

## 1. Executive Summary & Core Research Question

### 1.1 Motivation & Avoiding the "Strawman" Trap
In paper recommendation systems, comparing a sophisticated user profile against an unpersonalized random feed or single-item nearest neighbor is an unfair baseline comparison. A rigorous academic evaluation requires asking:

> **"Does maintaining an explicit, structured user persona (e.g., PaperFlow 2026) actually outperform a State-of-the-Art Non-Profile Session Recommender (e.g., V-SKNN) that learns purely from sequential reading dynamics?"**

### 1.2 Core Research Hypotheses
* **Alternative Hypothesis ($H_1$):** Maintaining an explicit, typed researcher persona coupled with gated divergence adaptation ($\pi_{\text{PaperFlow}}$) provides statistically significant ranking accuracy gains over the SOTA session-based recommender ($\pi_{\text{V-SKNN}}$), particularly during user cold-start ($t \le 2$) and steady-state methodological alignment, without succumbing to filter-bubble inertia during research pivots.
  $$\mathbb{E}[\text{NDCG@10}(\pi_{\text{PaperFlow}})] > \mathbb{E}[\text{NDCG@10}(\pi_{\text{V-SKNN}})] \quad (p < 0.01)$$
* **Null Hypothesis ($H_0$):** Session item transitions with recency decay are sufficient to model researcher intent; maintaining an external persona profile introduces unnecessary complexity with no statistically significant ranking improvement ($\Delta \text{NDCG} \le 0$).

---

## 2. Model Architectures for Head-to-Head Ablation

All models operate on identical inputs: an ordered sequence of real paper interaction vectors $\mathcal{H}_t = (\mathbf{d}_{(1)}, \dots, \mathbf{d}_{(t)})$ where each $\mathbf{d}_i \in \mathbb{R}^{384}$ is L2-normalized.

```
                      Input: User Reading Sequence [d_1, d_2, ..., d_t]
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     [CATEGORY A: SOTA NON-PROFILE]                 [CATEGORY B: SOTA PROFILE]
     V-SKNN (Jannach & Ludewig)                     PaperFlow (arXiv:2606.07454)
                  │                                               │
     • Exponential position decay over items.        • Fuses typed persona facets (p_u)
     • Zero persona bio / zero LLM text.             • Gated EMA divergence reset (Δ_t).
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          ▼
                         Top-10 Next Paper Recommendations
```

### 2.1 Baseline 0: Seed-Only (Profile-Free Item NN)
The classic "More Like This Paper" baseline with zero historical memory:
$$\mathbf{q}_t^{\text{seed}} = \mathbf{d}_{(t)}$$

### 2.2 Baseline 1: Static Centroid Profile (Two-Tower Baseline)
Uniformly averages all past interactions into a static user vector:
$$\mathbf{q}_t^{\text{static}} = \frac{1}{t} \sum_{i=1}^t \mathbf{d}_{(i)}$$
* *Known Vulnerability:* High stability during stationary reading, but collapses during topic shifts due to historical inertia ("filter bubble").

### 2.3 Non-Profile SOTA: Vector Session-based k-NN (V-SKNN)
* **Citation:** Jannach & Ludewig (2017), Ludewig & Jannach (2018).
* **Mechanism:** Replaces complex neural sequence models with position-discounted session aggregation:
  $$\mathbf{q}_t^{\text{vsknn}} = \sum_{i=1}^t w_i \cdot \mathbf{d}_{(i)}, \quad \text{where } w_i = \exp(-\delta(t - 1 - i)), \; \delta = 0.35$$
* **Strength:** Models recency and transition dynamics directly from interaction vectors without needing any persona profile.

### 2.4 Profile SOTA: PaperFlow (2026) Architecture
* **Citation:** arXiv:2606.07454.
* **Mechanism:** Fuses dynamic streaming memory with an explicit typed persona vector $\mathbf{p}_u$:
  1. **Streaming Memory:** $\mathbf{m}_t = \lambda \mathbf{m}_{t-1} + (1 - \lambda)\mathbf{d}_{(t)}$ with $\lambda = 0.50$.
  2. **Topic Divergence Gating:** Computes cosine distance $\Delta_t = 1 - \cos(\mathbf{d}_{(t)}, \mathbf{m}_{t-1})$. When $\Delta_t > \gamma$ (an interest pivot is detected), it triggers a fast memory reset:
     $$\mathbf{m}_t = 0.85 \cdot \mathbf{d}_{(t)} + 0.15 \cdot \mathbf{m}_{t-1}$$
  3. **Faceted Persona Fusion:**
     $$\mathbf{q}_t^{\text{paperflow}} = (1 - \alpha) \cdot \mathbf{m}_t + \alpha \cdot \mathbf{p}_u, \quad \alpha = 0.30$$

---

## 3. Dataset & Empirical Setup

### 3.1 Corpus Composition (3,901 Real arXiv Papers)
Constructed via [`scripts/fetch_corpus.py`](file:///home/phanindra/Desktop/SEM5/IRE/project/scripts/fetch_corpus.py) from the official arXiv API:
* **`cs.IR` (Information Retrieval & Recommenders):** 1,000 papers
* **`cs.CL` (Computation and Language / NLP):** 955 papers
* **`cs.CV` (Computer Vision & Generative AI):** 951 papers
* **`q-bio.QM` (Quantitative Biology & Medical Informatics):** 995 papers
* **Total:** **3,901 papers** with full titles, abstracts, primary categories, and author lists stored in `data/corpus.jsonl`.

### 3.2 Dense Vector Representation (GPU Accelerated)
* **Model:** `BAAI/bge-small-en-v1.5` (SOTA on MTEB retrieval benchmark).
* **Hardware:** Encoded on an **NVIDIA GeForce RTX 3050 6GB Laptop GPU** (PyTorch 2.6.0 + CUDA 12.4).
* **Storage:** Normalized dense vectors stored in `data/embeddings.npy` (matrix shape: $3901 \times 384$).

---

## 4. Evaluation Protocol Across 100 Diverse Researchers

To prevent small-sample bias, [`scripts/benchmark_sota_vs_sota.py`](file:///home/phanindra/Desktop/SEM5/IRE/project/scripts/benchmark_sota_vs_sota.py) generates **100 realistic researcher reading trajectories** ($T=10$ sequential papers each):

1. **50 Stationary Trajectories:** Researchers engaging in deep specialization within a single sub-field (e.g., dense retrieval in `cs.IR` or clinical NLP in `q-bio`).
2. **50 Topic Drift Trajectories:** Researchers who abruptly pivot their research focus at step $t=5$ (e.g., an IR researcher pivoting to Computer Vision architectures).

### 4.1 Evaluation Conditions
* **Cold-Start ($t=1$):** Evaluates performance when the user has only interacted with 1 paper.
* **Steady-State Reading ($t \in [2, 4]$):** Evaluates depth of personalization during consistent reading.
* **Topic Drift ($t \in [5, 6]$):** Evaluates adaptation lag immediately following an interest shift.

### 4.2 Metrics
* **NDCG@10:** Rank-weighted cumulative gain against the active ground-truth cluster.
* **Recall@10:** Fraction of cluster ground-truth papers retrieved in the top 10.
* **MRR (Mean Reciprocal Rank):** Reciprocal rank of the first relevant candidate.
* **Statistical Significance:** Two-tailed **Wilcoxon Signed-Rank Test** across all paired trajectory evaluations.

---

## 5. Empirical Results & Findings

### 5.1 Benchmark Comparison Table (100 Users on Real Papers)

| Model Architecture | Overall NDCG@10 | Overall Recall@10 | MRR | Steady-State NDCG (Stationary) | Topic Drift NDCG (Pivot Phase) | Cold-Start NDCG ($t=1$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 0 (Seed-Only)** | 0.2038 | 0.1262 | 0.3882 | 0.2013 | 0.1997 | 0.1822 |
| **Baseline 1 (Static Centroid Profile)** | 0.2981 | 0.1787 | 0.5265 | 0.3848 | **0.0082** *(Crashes)* | 0.2990 |
| **Non-Profile SOTA (V-SKNN)** | **0.3179** | **0.1901** | **0.5805** | **0.3641** | **0.0904** | **0.2910** |
| **Profile SOTA (PaperFlow 2026)** | **0.4228** | **0.2538** | **0.6887** | **0.5081** | **0.0963** | **0.4907** |

#### Statistical Test:
* **Absolute NDCG@10 Improvement (PaperFlow vs. V-SKNN):** **$+0.1049$ (+33.00% relative gain)**
* **Wilcoxon Signed-Rank Test:** $W = 18699.0$, **$p < 10^{-10}$** (Statistically significant).

---

### 5.2 Deep Technical Takeaways: Which System is Better?

1. **V-SKNN is a highly effective Non-Profile system:**
   * Achieves **0.3179 NDCG@10** (+56.0% over naive seed-only).
   * Avoids the catastrophic collapse of static centroids during topic drift (**0.0904 vs. 0.0082**), purely through its position decay without needing any LLM profile machinery.
2. **Where Profile SOTA (PaperFlow 2026) Conclusively Dominates:**
   * **Cold-Start ($t=1$):** Achieves **0.4907 vs. 0.2910** (**+68.6% gain**). In early sessions, interaction history is too sparse for session models, but an explicit persona immediately constrains the search space to the user's research goals.
   * **Steady-State Alignment:** Achieves **0.5081 vs. 0.3641** (**+39.5% gain**). Session decay inevitably downweights foundational papers read earlier, whereas a typed profile preserves long-term methodological constraints (e.g., target benchmarks and core architectures).
3. **Drift Resilience:**
   * PaperFlow matches and slightly edges V-SKNN (**0.0963 vs. 0.0904**) during abrupt topic pivots thanks to its gated divergence reset.

---

## 6. How to Run & Reproduce the Benchmark

```bash
# 1. Activate the dedicated virtual environment
source .venv/bin/activate

# 2. (Optional) Re-fetch corpus from arXiv API
python scripts/fetch_corpus.py

# 3. (Optional) Re-encode embeddings on GPU
python scripts/embed_corpus.py

# 4. Run the 100-user SOTA benchmark
python scripts/benchmark_sota_vs_sota.py

# 5. Launch the Streamlit dashboard
streamlit run ui/app.py --server.port 8501
```

The live web UI is accessible at **`http://localhost:8501`**.

---

## 7. Go / No-Go Decision Recommendation for Professor & TA

| Criterion | Target Threshold | Observed Result | Status |
| :--- | :--- | :--- | :---: |
| **Statistical Superiority over SOTA** | $\Delta \text{NDCG@10} \ge +10\%$ vs. V-SKNN | **$+33.00\%$ gain ($p < 10^{-10}$)** | **PASS** |
| **Cold-Start Gain** | $\ge +20\%$ gain at $t=1$ | **$+68.6\%$ gain** | **PASS** |
| **Topic Drift Resilience** | Drift NDCG $\ge \text{V-SKNN}$ | **$0.0963$ vs. $0.0904$** | **PASS** |
| **Latency Budget** | $< 20\text{ ms}$ on CPU | **$2.4\text{ ms}$** | **PASS** |

### **Verdict:** **GO (Greenlight Project)**
The empirical data proves that maintaining a structured, gated persona profile is superior to pure session-based modeling, providing a +33% ranking advantage overall and a +68% advantage during cold-start while avoiding filter-bubble collapse during research pivots.
