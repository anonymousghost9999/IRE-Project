# Paper Feed --- Current Design Documentation

> **Purpose:** Document the present state of the Paper Feed system
> design and decisions.
>
> **Important:** This document records the current situation only. It
> intentionally does **not** include recommendations, proposed changes,
> or design corrections.

------------------------------------------------------------------------

## 1. Overview

Paper Feed is a personalized academic-paper recommendation system.

The intended pipeline is:

1.  Crawl academic-paper metadata from **OpenAlex**.
2.  Clean and structure the corpus and should build the ground truths of papers read by authors by using citation graph.
3.  Build retrieval indexes.
4.  Extract structured paper profiles from paper text/abstracts and methods used and datasets used and metrics used.
5.  Represent the user's interests and interaction history.
6.  Retrieve and rank candidate papers.
7.  Apply diversity and novelty considerations.
8.  Generate grounded explanations from extracted paper-profile facets.
9.  Evaluate retrieval, extraction, recommendation, and explanation
    quality.

The system is intended to emphasize:

-   Hybrid lexical + dense retrieval.
-   Personalization from actual reading and citation behavior.
-   Structured paper profiles.
-   Citation signals.
-   Diversity and novelty.
-   Grounded, explainable recommendations.

The current design does not include large-scale model training or
production-scale infrastructure.

------------------------------------------------------------------------

# 2. Corpus and Data Acquisition

## 2.1 Corpus Sources

The selected corpus sources are:

-   **OpenAlex**

OpenAlex is currently expected to provide the fields required by the
paper metadata schema.

## 2.2 Corpus Size

Current corpus size is **NEED TO BE ATLEAST 1,00,000**

## 2.3 Paper Metadata Schema

The currently defined paper fields are:

  Field          Description
  -------------- --------------------------------
  paper_id     Unique paper identifier
  title        Paper title
  abstract     Paper abstract
  authors      Paper authors
  year         Publication year
  venue        Publication venue
  categories   Paper categories
  keywords     Paper keywords
  references   Papers referenced by the paper
  citations    Papers citing the paper
  doi          DOI, when available
  url          Paper URL
  methods      Methods used in the paper
  datasets     Datasets used in the paper
  metrics      Metrics used in the paper
  topics       Topics of the paper

## 2.4 Deduplication

Deduplication is currently **closed**. We decided to use only openalex so its closed

## 2.5 Author Resolution

Author resolution is currently **closed**. We decided to use only openalex so its closed

------------------------------------------------------------------------

# 3. Storage

The current storage design is:

  Data                      Storage
  ------------------------- -------------------
  Paper metadata            SQLite
  Paper profile/facets      SQLite
  Paper embeddings          FAISS
  HNSW graph                FAISS
  BM25 index                Pyserini/Anserini **(it is java based so other option is bm25s with good latency and is python based)**
  User interactions         SQLite
  User explicit interests   SQLite
  User profile/facets       SQLite

SQLite is therefore the current primary persistent database for
structured metadata, profiles, users, and interactions.

FAISS is used for vector-related storage/indexing.

Pyserini/Anserini is used for the lexical retrieval index.

------------------------------------------------------------------------

# 4. Retrieval and Indexing

The retrieval system is hybrid.

It currently combines:

1.  Lexical retrieval using **BM25**.
2.  Dense retrieval using a vector index.
3.  Reciprocal Rank Fusion (**RRF**).
4.  A reranking stage.

## 4.1 Lexical Retrieval

BM25 is fixed as the lexical retrieval method.

The intended lexical index contains textual information from:

-   Title
-   Abstract
-   Authors
-   Keywords
-   Methods
-   Datasets
-   Tasks/topics

The current implementation direction is **Pyserini/Anserini ** **(it is java based so other option is bm25s with good latency and is python based)**.

The exact weighting/field configuration remains open.

## 4.2 Dense Retrieval

Dense retrieval is fixed as part of the retrieval architecture.

The vector representation is based on a paper encoder.

The dense index is currently open between:

-   FAISS
-   hnswlib

The current thought is **FAISS with HNSW**.

An exact FAISS inner-product index is also planned as an evaluation
baseline.

## 4.3 Paper Encoder

The paper encoder is fixed as:

**SPECTER2 proximity adapter**

This is intended for paper-to-paper semantic similarity.

## 4.4 Query Encoder

The query encoder is fixed as:

**SPECTER2 with adhoc_query adapter (allenai/specter2_adhoc_query)**

This encodes short natural-language search queries into the exact same 768-dimensional vector space as the specter2_proximity paper representations, bridging the asymmetric length gap without generative LLM overhead.

## 4.5 Fusion

Fusion is fixed as:

**Reciprocal Rank Fusion (RRF)**.

The lexical and semantic candidate rankings are combined using RRF.

------------------------------------------------------------------------

# 5. Query Construction

The current query representation is conceptually:

``` text
q = {current query} + {session data} + {user profile}
```

The current query is intended to receive greater weight than the other
components.

The exact implementation of this combination is still open.

------------------------------------------------------------------------

# 6. Asymmetric Query Encoding (Replacing HyDE)

HyDE (Hypothetical Document Embeddings) has been **explicitly dropped**:
- It requires running an autoregressive generative LLM at search time, introducing an unacceptable 3–8 second latency bottleneck and 3–4 GB VRAM footprint on local hardware (RTX 3050).
- Small local LLMs hallucinate non-existent scientific entities, methods, and benchmarks on niche queries.
- For feed recommendations, candidate retrieval is already driven by full document-length paper vectors ($\mathbf{s}_t$ and $\mathbf{p}_u$), making generative document expansion redundant.

## 6.1 Selected Solution: SPECTER2 Adhoc Query Adapter

To bridge the asymmetric representation gap between short search queries and full paper abstracts, the system uses:

**allenai/specter2_adhoc_query**

Key characteristics:
1.  **Direct Vector Projection:** Projects concise search terms directly into the 768-d scientific paper space shared by specter2_proximity.
2.  **Ultra-Low Latency:** Executes as a standard transformer forward pass in **< 5 ms** on GPU (versus 3,000+ ms for generative HyDE).
3.  **Zero Hallucination:** Computes deterministic embeddings with zero synthetic text generation.
4.  **Trained for Asymmetric Retrieval:** Specifically fine-tuned on search query–document relevance pairs from scientific literature to align concise queries with dense document neighborhoods.

------------------------------------------------------------------------

# 7. Candidate Generation and Reranking

## 7.1 Candidate Count

The candidate count remains open.

The current working thought is:

` text
K = 100


for each lexical/semantic retrieval stream.

## 7.2 Reranker

The reranking stage is currently **tentative**.

The current candidate is:
- BAAI/bge-reranker-base (if you want maximum ranking quality using standard sentence-transformers.CrossEncoder).
- cross-encoder/ms-marco-MiniLM-L-6-v2 (if you prioritize sub-20ms latency and minimal GPU memory usage).
- try both of them and based on results and resources we have to decide.

## 7.3 Reranking Size

The current reranking size is approximately:

` text
Top 100


This is treated as effectively fixed in the present design.

------------------------------------------------------------------------

# 8. Structured Paper Profiling

The system extracts structured information from papers into inspectable, typed profile facets.

The current profile schema contains:
-   Domain (Domain of the paper, e.g., Information Retrieval, NLP)
-   task (Primary research problem addressed)
-   related prior work in the domain (Specific papers or research lines that this work builds upon or compares against)
-   method (Proposed architecture, model, or algorithm)
-   dataset (Benchmark, corpus, or evaluation dataset used)
-   metric (Quantitative evaluation metric, e.g., NDCG, F1)
-   topic (Subfield domain, e.g., Information Retrieval, NLP)

These fields are structured into typed JSON and stored in the SQLite paper_profile table.

## 8.1 Extraction Method

The primary extraction method is fixed as:

**GLiNER (Generalist and Lightweight Named Entity Recognition)**

Specifically using urchade/gliner_medium-v2.1 or urchade/gliner_large-v2.1.

Key advantages for Paper Feed:
-   **Open-Vocabulary Extraction:** Takes target facet labels dynamically (["task", "method", "dataset", "metric", "topic"]) via bidirectional cross-attention between token spans and label embeddings.
-   **Zero Hallucination:** Constrained to extract verbatim spans directly from the source text; it cannot invent nonexistent entities.
-   **Ultra-Fast Throughput:** Runs in ~10–15 ms per abstract on GPU and ~25 ms on CPU, processing the full 100k corpus in under 30 minutes (compared to ~70 hours for an 8B LLM).
-   **Direct JSON Output:** Produces structured dictionaries containing text, label category, character offsets, and confidence scores.

## 8.2 Evidence Spans

Evidence spans are required for all extracted profile fields to ensure ground truth and inspectability.

GLiNER natively outputs exact character offsets (start, end). These offsets are used to extract the full verbatim sentence from the source abstract as the supporting evidence:

-   **Structured Value:** The exact entity span (e.g., PaperFlow, SciERC).
-   **Supporting Evidence Span:** The full sentence containing the entity (e.g., "We evaluate our gated streaming memory on the SciERC benchmark.").
-   **Character Offsets & Confidence:** [start, end] indices and prediction probability score.

## 8.3 Deterministic Execution (No Validation / Retry Loops)

Because GLiNER extracts spans deterministically from input text and outputs native Python dictionaries:
-   Syntax parsing failures and malformed JSON errors are completely eliminated.
-   Complex retry loops and prompt-engineering failure fallbacks are unnecessary.

## 8.4 scispaCy

scispaCy is retained as a lightweight sentence-splitting and tokenization utility.

It is not the primary information-extraction system.

## 8.6 LLM Baseline Comparison (Offline)

Local LLM extraction (e.g., Qwen 4B via Outlines/guided generation) is retained strictly as an offline baseline comparison on the SciERC evaluation subset (Section 27), rather than the primary corpus-wide batch extractor.

------------------------------------------------------------------------

# 9. User Representation

The current user representation consists of two complementary components (following PaperFlow 2026):

1.  **Faceted Persona Profile ($\mathbf{p}_u$):**
    - Built from user onboarding, explicit interests, and extracted facets (preferred tasks, methods, datasets).
    - Encoded as a persistent, typed vector in the same dense embedding space as paper vectors.
    - Provides strong alignment during cold start and preserves foundational methodological preferences.
2.  **Streaming Session Memory ($\mathbf{m}_t$):**
    - Dynamically models sequential reading dynamics across the current session.
    - Updated via Exponential Moving Average (EMA) over paper vectors:
      $$\mathbf{m}_t = \lambda \mathbf{m}_{t-1} + (1 - \lambda)\mathbf{d}_{(t)}$$
    - Governed by gated topic divergence to prevent filter-bubble inertia during research pivots.

The exact internal representation of facet embedding aggregation remains open.

------------------------------------------------------------------------

# 10. Interaction Feedback

The currently defined feedback signals are:

  User Action                              Signal         Strength
  ---------------------------------------- -------------- -------------
  Explicitly dislikes / "Not interested"   Negative       Very strong
  Hides / dismisses paper                  Negative       Strong
  Opens paper but immediately leaves       Negative-ish   Weak
  Skips recommendation                     Negative-ish   Very weak
  Reads paper substantially                Positive       Strong
  Saves/bookmarks paper                    Positive       Very strong
  Explicitly likes                         Positive       Very strong
  Cites paper                              Positive       Very strong

These signals are intended to contribute to user modeling and
recommendation.

The exact numerical aggregation of the feedback strengths remains open.

------------------------------------------------------------------------

# 11. Session Model

The design includes session-based personalization with streaming updates.

A session represents a coherent reading trajectory $\mathcal{H}_t = (\mathbf{d}_{(1)}, \dots, \mathbf{d}_{(t)})$.

A session is intended to end after a period of inactivity:

` text
N minutes of inactivity -> new session


When a new session begins, the streaming memory $\mathbf{m}_t$ can either initialize from the user's persona profile $\mathbf{p}_u$ or reset to the first paper interaction vector $\mathbf{d}_{(1)}$.

The exact value of N is still open.

------------------------------------------------------------------------

# 12. Recency Weighting & Topic Divergence Gating

Recency is handled dynamically through PaperFlow's Exponential Moving Average (EMA) memory combined with gated divergence reset, replacing static exponential time decay.

At step $t$, when a user interacts with paper $\mathbf{d}_{(t)}$:

1.  **Divergence Check (Topic Pivot Detection):**
    $$\Delta_t = 1 - \cos(\mathbf{d}_{(t)}, \mathbf{m}_{t-1})$$
2.  **Gated Memory Update:**
    - If $\Delta_t > \gamma$ (an abrupt topic pivot / research drift is detected, e.g., $\gamma \approx 0.35$):
      $$\mathbf{m}_t = 0.85 \cdot \mathbf{d}_{(t)} + 0.15 \cdot \mathbf{m}_{t-1}$$
      *(Fast reset breaks out of filter-bubble inertia).*
    - Else (steady-state continuous reading):
      $$\mathbf{m}_t = \lambda \cdot \mathbf{m}_{t-1} + (1 - \lambda)\cdot \mathbf{d}_{(t)}$$
      *(Where $\lambda \approx 0.50$ provides smooth recency decay).*

------------------------------------------------------------------------

# 13. Recommendation Target

The recommendation target is fixed as:

**Papers**

The system ultimately produces a ranked list of recommended academic
papers.

------------------------------------------------------------------------

# 14. Primary Recommendation Algorithm

The primary recommender is fixed as: 

**V-PaperFlow (Hybrid Architecture: Gated V-SKNN with Faceted Persona Fusion)**

This architecture fuses:
1.  **V-SKNN's fine-grained position decay** over the current session reading sequence.
2.  **PaperFlow's explicit persona vector ($\mathbf{p}_u$)** to anchor long-term research domain continuity and solve cold start.
3.  **PaperFlow's gated topic divergence mechanism** to detect research pivots and reset session momentum, preventing filter-bubble inertia.

To rigorously demonstrate the value of each component, the evaluation protocol establishes a 3-way ablation:
-   **Baseline 1 (Non-Profile SOTA):** Pure V-SKNN (Position decay only; zero persona profile; Jannach & Ludewig).
-   **Baseline 2 (Profile SOTA):** Pure PaperFlow (EMA memory + persona fusion; arXiv:2606.07454).
-   **Proposed Model:** V-PaperFlow (The Hybrid).

*(Note: SASRec is explicitly dropped as out-of-scope; training deep neural sequential models from scratch requires millions of interaction sequences, introduces severe item-ID cold-start failure on new academic papers, and violates the project's zero-training scope).*

------------------------------------------------------------------------

# 15. V-PaperFlow Recommendation Formulation

The V-PaperFlow recommendation score combines position-weighted session dynamics with explicit persona profile constraints and adaptive divergence gating:

1.  **V-SKNN Session Momentum ($\mathbf{s}_t$):**
    For a session sequence of paper vectors $\mathcal{H}_t = (\mathbf{d}_{(1)}, \dots, \mathbf{d}_{(t)})$:
    $$\mathbf{s}_t = \sum_{i=1}^t w_i \cdot \mathbf{d}_{(i)}, \quad \text{where } w_i = \exp(-\delta(t - 1 - i))$$
    where $\delta \approx 0.35$ controls the recency decay over sequential items.

2.  **Gated Topic Divergence Check (Research Pivot Detection):**
    Before aggregating the new paper $\mathbf{d}_{(t)}$, compute the directional drift from previous session momentum:
    $$\Delta_t = 1 - \cos(\mathbf{d}_{(t)}, \mathbf{s}_{t-1})$$
    -   **If $\Delta_t > \gamma$ (Topic Pivot Detected, e.g., $\gamma \approx 0.35$):**
        The user has pivoted research domains. Flush historical session items and reset momentum to the active paper:
        $$\mathbf{s}_t = \mathbf{d}_{(t)}$$
    -   **Else (Steady-State Continuous Reading):**
        Maintain smooth exponential position decay $\mathbf{s}_t$.

3.  **Faceted Persona Fusion (Composite Query $\mathbf{q}_t$):**
    Fuse the gated session momentum $\mathbf{s}_t$ with the persistent user persona profile $\mathbf{p}_u$:
    $$\mathbf{q}_t = (1 - \alpha) \cdot \mathbf{s}_t + \alpha \cdot \mathbf{p}_u$$
    where $\alpha \in [0, 1]$ is the persona fusion weight (e.g., $\alpha \approx 0.25 - 0.30$).

4.  **Candidate Scoring:**
    For each candidate paper $p$ with dense vector $\mathbf{d}_p$:
    $$S(p \mid \mathbf{q}_t) = \cos(\mathbf{d}_p, \mathbf{q}_t) + \beta \cdot \text{Sim}_{\text{facet}}(p, \mathbf{p}_u)$$
    where:
    -   $\cos(\mathbf{d}_p, \mathbf{q}_t)$ scores semantic embedding alignment with the fused persona/session state.
    -   $\text{Sim}_{\text{facet}}(p, \mathbf{p}_u)$ scores explicit typed facet matching (shared tasks, methods, datasets).
    -   $\beta$ is a facet weighting hyperparameter.

The exact hyperparameter tuning for $\delta, \gamma, \alpha, \beta$ remains open.

------------------------------------------------------------------------

# 16. Citation Signal

Citation behavior from the OpenAlex graph is a first-class ranking signal, capturing the bibliographic relationship between a candidate paper $p$ and the user's own research output and reading history.

## 16.1 Base Citation Score

The base citation score is a piecewise function over four mutually exclusive graph relationships, evaluated from strongest to weakest:

$$S_{\text{citation}}(p \mid u) = \begin{cases} +2.0 & \text{if } p \in \text{Citations}(u) \;\; \text{(user personally cited } p \text{ in their own published work)} \\ +1.2 & \text{if } p \in \text{BibCoupling}(u) \;\; \text{(}p \text{ shares } \geq 3 \text{ references with papers the user has read/written)} \\ +0.7 & \text{if } p \in \text{CoCitation}(u) \;\; \text{(}p \text{ is frequently co-cited alongside the user's papers in the literature)} \\ +0.2 & \text{if } p \in \text{PriorWork}(u) \;\; \text{(}p \text{ appears in the reference list of papers the user has read)} \\ 0.0 & \text{otherwise} \;\; \text{(no known graph relationship)} \end{cases}$$

**Graph Relationship Definitions:**

- **Citations($u$):** Papers that the user $u$ has directly cited in their own published papers (OpenAlex `referenced_works` of the user's authored papers). The strongest verified signal — the user demonstrably read and found this paper relevant enough to cite.
- **BibCoupling($u$):** Papers sharing $\geq 3$ references with papers authored or bookmarked by the user (bibliographic coupling via Jaccard similarity on reference lists). Computable directly from OpenAlex `referenced_works` fields.
- **CoCitation($u$):** Papers that appear together with the user's papers in the reference lists of third-party papers (i.e., other researchers cite both $p$ and the user's work). Signals community-perceived relatedness.
- **PriorWork($u$):** Papers that appear in the reference lists of papers the user has read in-session. A soft signal indicating the user is traversing a related literature thread.

## 16.2 Recency Temporal Decay

More recent citation relationships are stronger predictors of current reading intent than historical ones. The base score is dampened by publication age:

$$S_{\text{citation}}(p \mid u, t) = S_{\text{citation}}(p \mid u) \cdot e^{-\lambda (t_{\text{now}} - t_p)}$$

where $t_p$ is the publication year of paper $p$, $t_{\text{now}}$ is the current year, and $\lambda \approx 0.2$ controls the decay rate (half-life $\approx 3.5$ years).

## 16.3 Self-Citation Dampening

Researchers heavily self-cite, which creates a filter bubble where the system reinforces the same research group's output. If a candidate paper $p$ shares one or more authors with the user $u$'s own publication list:

$$S_{\text{citation}}(p \mid u) \;\leftarrow\; S_{\text{citation}}(p \mid u) \times 0.5$$

This preserves self-citations as a signal (the user's group's work is genuinely relevant) while preventing the feed from becoming a closed echo chamber.

## 16.4 Ground Truth Evaluation: Leave-Future-Out (LFO) Protocol

The citation signal is evaluated using a **temporal Leave-Future-Out** protocol grounded in OpenAlex author timelines:

**Setup:**
- Let researcher $u$ have published Paper A ($t_A \leq 2023$) and Paper B ($t_B = 2024$).
- **Input horizon:** All papers authored or cited by $u$ with $t \leq 2023$.
- **Prediction target:** The reference list of Paper B (i.e., what the researcher actually needed to read to write Paper B in 2024).
- **Strict leakage guard:** Paper B's content is **never** shown to the recommender during retrieval — only its reference list is used as the held-out evaluation oracle.

**Evaluation:**
$$\text{Ground Truth}(u) = \text{References}(\text{Paper B}) \setminus \text{Papers already read by } u \text{ before } 2024$$

The system is evaluated on whether it surfaces papers from this ground truth set in its Top-10 recommendations, measured via NDCG@10, Recall@10, and MRR.

**Rationale:** A researcher's own future citations are verified evidence of genuine domain need — they are not simulated, not synthetic, and not self-reported. This makes the LFO protocol the strongest available ground truth for academic recommender evaluation without requiring live user studies.

------------------------------------------------------------------------

# 17. Multi-Signal Ranking

The system scores and ranks candidate papers using a calibrated weighted combination of multiple complementary signals across four distinct dimensions (Sequential Dynamics, Persona Facets, Citation Graph, and Candidate Retrieval), followed by post-scoring diversity (MMR) and novelty calibration:

### 17.1 Sequential & Session Signals (V-SKNN Momentum)
1.  **V-SKNN Transition Momentum ($S_{\text{vsknn}}$):**
    Cosine similarity between candidate paper $p$ and the position-decayed session vector: $\cos(\mathbf{d}_p, \mathbf{s}_t)$. Prioritizes papers matching recent reading momentum.
2.  **Immediate Seed Proximity ($S_{\text{seed}}$):**
    Direct cosine similarity to the single most recently read paper: $\cos(\mathbf{d}_p, \mathbf{d}_{(t)})$.

### 17.2 Profile & Facet Signals (PaperFlow Persona)
3.  **Persona Profile Alignment ($S_{\text{persona}}$):**
    Cosine similarity between candidate paper $p$ and the persistent user persona vector: $\cos(\mathbf{d}_p, \mathbf{p}_u)$. Anchors long-term domain continuity and prevents drift.
4.  **Structured Facet Match ($S_{\text{facet}}$):**
    Direct overlap between candidate's extracted GLiNER facets and the user's declared/preferred profile facets:
    $$S_{\text{facet}}(p, \mathbf{p}_u) = w_m \cdot \mathbb{I}(\text{method match}) + w_d \cdot \mathbb{I}(\text{dataset match}) + w_t \cdot \mathbb{I}(\text{task match})$$
    Powers verifiable template explanations.

### 17.3 Bibliographic & Citation Signals (OpenAlex Graph)
5.  **Citation Graph Proximity ($S_{\text{cite}}$):**
    Direct connection in the OpenAlex citation graph with the user's active reading history:
    - Bibliographic coupling: Candidate paper shares references with papers read in session.
    - Co-citation: Candidate paper is co-cited with papers saved/liked by the user.
6.  **Global Authority / Impact ($S_{\text{impact}}$):**
    Log-normalized global citation count: $\log(1 + \text{citation\_count})$, favoring established, high-credibility literature over unvetted noise.

### 17.4 Candidate Retrieval & Negative Feedback
7.  **Hybrid Retrieval Score ($S_{\text{RRF}}$):**
    The Reciprocal Rank Fusion score from initial BM25 lexical search and dense vector search, preserving exact keyword matches.
8.  **Negative Interaction Penalty ($P_{\text{skip}}$):**
    Cosine similarity to papers the user explicitly skipped, dismissed, or marked as "not interested", actively demoting rejected topics.

### 17.5 Combined Scoring Formulation
Before applying MMR for diversity (Section 18) and novelty filtering (Section 19), candidate papers are ranked by a linear combination of normalized signals:

$$\text{Score}(p) = w_1 S_{\text{vsknn}} + w_2 S_{\text{seed}} + w_3 S_{\text{persona}} + w_4 S_{\text{facet}} + w_5 S_{\text{cite}} + w_6 S_{\text{impact}} + w_7 S_{\text{RRF}} - w_{\text{neg}} P_{\text{skip}}$$

The exact hyperparameter weights $w_1, \dots, w_7, w_{\text{neg}}$ are tuned on a validation set (Section 20).

------------------------------------------------------------------------

# 18. Diversity

Diversity is fixed as part of the recommendation system.

The selected approach is:

**MMR --- Maximal Marginal Relevance**

The exact MMR lambda remains open and is intended to be treated as a
hyperparameter.

------------------------------------------------------------------------

# 19. Novelty

Novelty measures how much a candidate paper $p$ expands the user's knowledge relative to their reading history, as distinct from **Diversity** (Section 18), which ensures variety within the recommendation list itself.

**Key distinction:**
- **Diversity (MMR):** Are the 10 recommended papers different from *each other*?
- **Novelty:** Are the recommended papers different from what the user has *already seen*?

Previously consumed papers are always excluded from the recommendation output entirely (hard filter before scoring).

## 19.1 Semantic Novelty

Semantic novelty measures the embedding-space distance between a candidate paper and the centroid of the user's entire reading history:

$$N_{\text{semantic}}(p, u) = 1 - \cos\!\left(\mathbf{d}_p,\; \bar{\mathbf{d}}_u\right)$$

where:
- $\mathbf{d}_p$ is the SPECTER2 (`proximity` adapter) embedding of candidate paper $p$.
- $\bar{\mathbf{d}}_u = \dfrac{1}{|\mathcal{H}_u|} \sum_{h \in \mathcal{H}_u} \mathbf{d}_h$ is the mean embedding over all papers in the user's reading history $\mathcal{H}_u$.
- $N_{\text{semantic}} = 1.0$ means the paper is maximally unlike anything the user has read (high novelty).
- $N_{\text{semantic}} = 0.0$ means the paper is essentially identical to the user's average reading (zero novelty).

## 19.2 Facet Novelty

Facet novelty measures whether a candidate paper introduces **typed entity facets** (methods, datasets, topics) that the user has not previously encountered, using GLiNER-extracted profile fields stored in the SQLite `paper_profile` table:

$$N_{\text{facet}}(p, u) = w_m \cdot \mathbb{I}(m_p \notin \mathcal{M}_u) + w_d \cdot \mathbb{I}(d_p \notin \mathcal{D}_u) + w_t \cdot \mathbb{I}(t_p \notin \mathcal{T}_u)$$

where:
- $m_p, d_p, t_p$ are the primary method, dataset, and topic facets of candidate paper $p$ (from GLiNER extraction).
- $\mathcal{M}_u, \mathcal{D}_u, \mathcal{T}_u$ are the sets of methods, datasets, and topics the user has already encountered across their reading history (from `paper_profile` joined with `user_interactions`).
- $\mathbb{I}(\cdot)$ is the indicator function (1 if the facet is new to the user, 0 if already seen).
- $w_m, w_d, w_t$ are facet-specific novelty weights (hyperparameters; suggested defaults: $w_m = 0.4, w_d = 0.35, w_t = 0.25$).

This directly leverages the GLiNER infrastructure from Section 8 and requires no additional model inference at recommendation time.

## 19.3 Combined Novelty Score

The two components are combined into a single novelty score:

$$N(p, u) = \beta_s \cdot N_{\text{semantic}}(p, u) + \beta_f \cdot N_{\text{facet}}(p, u)$$

where $\beta_s$ and $\beta_f$ are blending weights (tuned on the validation set; suggested defaults: $\beta_s = 0.5, \beta_f = 0.5$).

## 19.4 Integration with Final Ranking

Novelty acts as a **soft additive boost** applied after MMR re-ranking (Section 18), nudging unexplored papers upward without overriding relevance:

$$\text{Score}_{\text{final}}(p) = \text{Score}_{\text{MMR}}(p) + \gamma \cdot N(p, u)$$

where $\gamma \approx 0.1$–$0.3$ is a small novelty injection weight tuned on the validation set (Section 20). A small $\gamma$ ensures novelty widens the feed at the margins without displacing highly relevant papers from the top positions.

**Hyperparameters to tune:** $w_m, w_d, w_t, \beta_s, \beta_f, \gamma$.

------------------------------------------------------------------------

# 20. Calibration

The current calibration design is:

1.  Normalize relevance.
2.  Normalize novelty.
3.  Normalize citation-related scores.
4.  Put the signals onto a common scale.
5.  Tune the weights on a validation set.
6.  Apply MMR after the calibrated ranking.

Distribution calibration is currently deferred/later in the design.

------------------------------------------------------------------------

# 21. Recommendation Output

The intended recommendation list size is:

` text
Top 10 papers


------------------------------------------------------------------------

# 22. Explanation System

Explanations are intended to be grounded in structured paper profiles.

## 22.1 Explanation Source

The explanation source is fixed as:

**Paper profile facets**

The explanation should be based on extracted information such as:

- Domain
- Task
- Method
- Dataset
- Metric

## 22.2 Explanation Method

The primary explanation method is fixed as:

**Templates**

The templates are intended to use the structured profile and
recommendation signals rather than unconstrained generation.

## 22.3 LLM Paraphrasing

LLM paraphrasing is a stretch component.

If used, it is intended to:

-   Paraphrase the generated explanation.
-   Preserve the facts present in the paper profile/signals.
-   Improve grammatical/natural presentation.

## 22.4 Explanation Verification

Explanation verification is fixed as an NLI-based check.

A pretrained model such as **DeBERTa** is the current example.

The exact behavior after verification failure remains open.

The current idea is to revert to the default template explanation when
verification fails.

------------------------------------------------------------------------

# 23. Cold Start

Cold-start handling remains open.

The current intended priorities are:

1.  Explicit onboarding domain specific interests.
2.  Current query.

If these are insufficient, a generic/popularity-based fallback is
considered.

------------------------------------------------------------------------

# 24. New-Paper Handling

Newly crawled papers are initially stored temporarily.

The intended batch pipeline is:

1.  Crawl new papers.
2.  Clean/store metadata.
3.  Generate embeddings.
4.  Perform structured extraction.
5.  Add processed papers to the indexes.

The exact periodic batch cadence remains open.

Real-time ingestion is outside the current scope.

------------------------------------------------------------------------

# 25. Temporal Leakage Policy

Temporal leakage prevention is a required design constraint.

The intended policy is a **strict temporal cutoff**.

For evaluation at time t, the following must be restricted to
information available at or before t:

-   Candidate papers.
-   User interaction history.
-   User profiles.
-   Metadata.
-   Citation information.
-   Extracted paper information.

Future information must not be used.

The evaluation is intended to use temporal train/validation/test splits
rather than random splits.

------------------------------------------------------------------------

# 26. Retrieval Evaluation

Retrieval evaluation benchmarks the first-stage candidate generation pipeline (BM25s + SPECTER2 hybrid retrieval producing Top-100 candidates) in isolation from downstream reranking.

## 26.1 Benchmark Datasets

- **CSFCube (Faceted CS Paper Retrieval):**
  - **Relevance:** Serves as the primary domain benchmark. Features computer science papers with multi-aspect relevance judgments (evaluating match along background, method, and result facets).
  - **Role:** Directly evaluates the SPECTER2 `proximity` task adapter for faceted academic paper-to-paper retrieval.
- **LitSearch (Research Query Retrieval):**
  - **Relevance:** Benchmark consisting of real-world natural language research queries paired with relevant academic papers.
  - **Role:** Evaluates the SPECTER2 `adhoc_query` adapter and BM25s on ad-hoc text search.
- **SCIDOCS (Reference Multi-Task Benchmark):**
  - Canonical benchmark established by AllenAI for scientific embedding evaluation, testing citation prediction, co-citation, and classification across CS and general science.

## 26.2 Retrieval Metrics

- **Recall@100 (Primary Pipeline Gate):**
  - Measures the fraction of relevant papers successfully captured within the top-100 candidate pool fed to the multi-signal reranker.
  - Critical rationale: Candidate retrieval serves as the recall ceiling. If a relevant paper fails to make the Top-100 cut, downstream V-PaperFlow reranking can never surface it.
- **nDCG@10 & nDCG@100:**
  - Evaluates the ranking quality and graded relevance of the candidate retriever at both immediate top ranks (@10) and candidate pool boundaries (@100).
- **MRR (Mean Reciprocal Rank):**
  - Measures how rapidly the retrieval stage surfaces the first relevant candidate.

## 26.3 First-Stage Retrieval Ablation Protocol

To prove that hybrid candidate generation outperforms individual retrieval channels, the system must empirically report on CSFCube and LitSearch:

| Model Variant | Retriever Configuration | Purpose |
| :--- | :--- | :--- |
| **BM25s Baseline** | Lexical retrieval only (sparse token matching) | Quantifies keyword/entity overlap baseline |
| **SPECTER2 Baseline** | Dense retrieval only (proximity/adhoc embeddings) | Quantifies semantic embedding retrieval capability |
| **BM25s + SPECTER2 (Hybrid)** | Reciprocal Rank Fusion (RRF, $k=60$) | **Proposed system:** Proves hybrid achieves superior candidate Recall@100 |

------------------------------------------------------------------------

# 27. Extraction Evaluation

Extraction evaluation verifies the structured facet profiling module (GLiNER) across two complementary dimensions: benchmark accuracy against gold standard scientific annotations, and empirical audit on real-world OpenAlex ingestion profiles.

## 27.1 Benchmark Dataset & Facet Schema Mapping

Evaluation is anchored on **SciERC** (*Scientific Information Extraction with Relation and Coreference*, Luan et al., EMNLP 2018), the standard scientific entity extraction benchmark consisting of 500 expert-annotated abstracts from AI, NLP, and speech processing venues.

The project's structured profile schema (Section 8) is aligned directly with SciERC gold entity categories:

| Paper Profile Facet | SciERC Gold Entity Class | Description |
| :--- | :--- | :--- |
| `task` | **`Task`** | Research problem, goal, or application |
| `method` | **`Method`** | Model architecture, algorithm, system, or technique |
| `dataset` | **`Material`** | Benchmark dataset, corpus, or physical resource |
| `metric` | **`Metric`** | Quantitative evaluation measure (e.g., F1, NDCG) |
| `topic` | **`OtherScientificTerm`** | Subfield domain concepts and terminology |

## 27.2 Evaluation Metrics

Because scientific entity boundaries can be ambiguous (e.g., *"contrastive learning framework"* vs. *"contrastive learning"*), evaluation reports both boundary standards:

- **Strict Span F1:** Requires exact character offset boundary match and correct entity class.
- **Partial / Relaxed F1:** Requires $\ge 1$ token overlap and correct entity class.
- **Per-Facet Metrics:** Precision, Recall, and $F_1$ score calculated individually for each entity type (`task`, `method`, `dataset`, `metric`, `topic`), alongside overall Macro-$F_1$ and Micro-$F_1$.

## 27.3 Head-to-Head Baseline Comparison (SciERC Test Split)

In fulfillment of Section 8.6, GLiNER is evaluated head-to-head against a local LLM baseline on the SciERC test partition to validate speed, accuracy, and resource efficiency:

| System Variant | Architecture | Strict Macro-$F_1$ | Relaxed Macro-$F_1$ | Latency (ms / doc) | Throughput (docs / s) | Memory / Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LLM Baseline** | Qwen-4B / Qwen-2.5-Coder-7B via Outlines guided JSON | Baseline | Baseline | ~1,200–2,500 ms | ~0.5–0.8 docs/s | High (~16 GB VRAM) |
| **Proposed Extractor** | **GLiNER (urchade/gliner_medium-v2.1)** | Target $\ge$ Baseline | Target $\ge$ Baseline | **~10–25 ms** | **~40–100 docs/s** | **Low (< 2 GB / CPU viable)** |

This empirical experiment substantiates the architectural decision to use GLiNER for batch corpus-scale ingestion (100k papers in < 30 minutes) rather than an LLM pipeline.

## 27.4 In-Corpus Profile Audit (100 OpenAlex Papers)

To validate profiling quality beyond academic benchmark datasets on real OpenAlex ingestion data, a stratified sample of 100 extracted paper profiles is manually audited:

1. **Extraction Precision@Facet:**
   $$\text{Precision} = \frac{\text{Number of factually valid extracted facets}}{\text{Total number of extracted facets}}$$
   Verifies whether extracted methods, tasks, and datasets accurately reflect the claims in the abstract.
2. **Evidence Span Faithfulness (Zero Hallucination):**
   - Verification that 100% of supporting evidence sentences exist verbatim in the source paper abstract without token hallucination or fabrication.
3. **Facet Density & Completeness:**
   - Percentage of papers in the corpus with $\ge 3$ core facets (`task`, `method`, `dataset`) successfully populated, ensuring sufficient metadata for downstream facet novelty (Section 19) and facet-aware MMR (Section 18).

------------------------------------------------------------------------

# 28. Recommendation Evaluation

Recommendation evaluation measures the final end-to-end feed quality (Top-10 ranked recommendations) delivered to researchers after multi-signal ranking, novelty scaling, and facet-aware MMR diversification.

## 28.1 Evaluation Methodology: Leave-Future-Out (LFO) Temporal Ground Truth

To evaluate paper recommendations without relying solely on synthetic or ungrounded proxy signals, the project adopts the **Leave-Future-Out (LFO)** temporal oracle protocol:

1. **Researcher Timeline Construction:**
   - For an author $u$ in OpenAlex with a publication history spanning multiple years, partition their papers at temporal cutoff $T$ (e.g., $T = 2023$).
   - **Historical Profile ($t \leq T$):** Papers written, cited, and read by $u$ up to year $T$ form their user interaction history $\mathcal{H}_u$, building the persona profile $\mathbf{p}_u$ and session dynamics $\mathbf{s}_u$.
   - **Future Paper Target ($t = T+1$):** Paper $B$ published by $u$ in year $T+1$ (e.g., $2024$) is selected as the evaluation testbed.
2. **Held-Out Reference Oracle:**
   - The reference list $\text{Ref}(B)$ represents the papers that researcher $u$ demonstrably required and read to conduct their research for Paper $B$.
   - These references serve as the objective ground-truth positive set for researcher $u$ at test time.
3. **Strict Temporal Anti-Leakage Guard:**
   - The text, abstract, title, and citations of Paper $B$ are **strictly quarantined** and withheld during the retrieval and ranking phases.
   - The recommender receives only user history up to $T$ to predict recommendations; Paper $B$'s references are only queried during metric evaluation.

## 28.2 End-to-End Recommendation Metrics

The final Top-10 recommendation list $\mathcal{R}_u$ is evaluated across relevance, ranking, diversity, and novelty:

| Metric | Target | Description |
| :--- | :--- | :--- |
| **NDCG@10** | Ranking Quality | Graded position-discounted metric of ground-truth reference items within the Top-10 feed. |
| **Recall@10 / HitRate@10** | Coverage | Proportion of ground-truth reference papers from $\text{Ref}(B)$ retrieved in the final Top-10 feed. |
| **MRR** | Discovery Speed | Reciprocal rank of the highest-ranked ground-truth paper in $\mathcal{R}_u$. |
| **Intra-List Diversity (ILD)** | Diversity | Average pairwise cosine dissimilarity among paper vectors in $\mathcal{R}_u$: $\text{ILD} = \frac{2}{|\mathcal{R}|(|\mathcal{R}|-1)} \sum_{i < j} (1 - \cos(\mathbf{e}_i, \mathbf{e}_j))$. Validates MMR effectiveness (Section 18). |
| **Facet Coverage Rate** | Novelty | Fraction of recommended papers in $\mathcal{R}_u$ introducing valid scientific facets (methods, datasets) not present in user history $\mathcal{H}_u$. Validates facet novelty (Section 19). |

## 28.3 3-Way Core Model Ablation Protocol

To scientifically establish the contribution of the proposed **V-PaperFlow** formulation, the following three core configurations must be evaluated and compared on identical LFO splits:

1. **Baseline 1 — Pure V-SKNN (Session Dynamics Only):**
   - Implements only session-based decay without persona facets: $S_{\text{session}}(p \mid \mathcal{H}_u) = \sum_{j=1}^m \lambda^{m-j} \cdot \cos(\mathbf{e}_{h_j}, \mathbf{e}_p)$.
   - Evaluates short-term sequential interest matching alone.
2. **Baseline 2 — Pure PaperFlow (Faceted Persona Only):**
   - Implements static profile matching $\cos(\mathbf{p}_u, \mathbf{e}_p)$ and static facet overlap without temporal position decay.
   - Evaluates long-term profile affinity without sequential drift adaptation.
3. **Proposed System — V-PaperFlow (Unified Hybrid):**
   - Combines dynamic session decay $\mathbf{s}_u$, persona profile $\mathbf{p}_u$, citation graph signal $S_{\text{citation}}$, and facet-aware MMR.
   - Demonstrates that bridging sequential browsing with structured facet personas achieves superior NDCG@10, diversity, and novelty.

## 28.4 Pipeline Component Ablation

In addition to core model comparison, an ablation isolating each pipeline layer confirms individual module value:
- **No Citation Signal:** Multi-signal ranking with $w_{\text{cite}} = 0$.
- **No MMR Diversification:** Raw Top-10 ranked by multi-signal score without MMR diversification (measures drop in ILD).
- **No Novelty Signal:** Multi-signal ranking with $w_{\text{nov}} = 0$ (measures drop in Facet Coverage Rate).

------------------------------------------------------------------------

# 29. Longitudinal Evaluation

Longitudinal evaluation remains open.

The current intended direction is to adapt a PaperFlow-style methodology
using:

-   User timelines.
-   User personas.
-   Held-out future interactions/references as ground truth.

------------------------------------------------------------------------

# 30. User Study

A user study is currently planned.

The current target is:

` text
n = 50 users

The exact user-study task/protocol is not yet fully specified.

------------------------------------------------------------------------

# 31. LLM/API Usage

## 31.1 Local Extraction

Paper extraction is fixed using GLiNER (urchade/gliner_medium-v2.1), running locally on CPU or lightweight GPU with zero external API calls.

## 31.2 Explanation Generation

Explanation generation may use an external API depending on limits.

A local Qwen fallback is also considered.

## 31.3 API Budget

The current API budget is:

` text
cost = 0


The exact handling of optional external explanation generation therefore
remains open.

------------------------------------------------------------------------

# 32. Caching

Caching is currently open.

The intended cache should cover LLM extraction/generation results.

The current cache identity is based on:

-   Paper/input.
-   Model version.
-   Prompt version.

------------------------------------------------------------------------

# 33. Reproducibility

Reproducibility is going to be implemented in this project.
since the project should be reproducible and verifiable and reusable for other researchers in the community.

The intended configuration includes:

-   Recorded random seeds.
-   Deterministic settings where supported.

------------------------------------------------------------------------

# 34. Configuration Management

The design intends to record:

-   Code version.
-   Configuration.
-   Random seed.
-   Model names and versions.
-   Model/inference parameters.
-   Prompt versions.
-   Dataset/corpus snapshots.
-   Retrieval/index configuration.
-   Timestamp.

------------------------------------------------------------------------

# 35. Explicit Out-of-Scope Items

The following are explicitly outside the intended scope:

-   Deep model training.
-   Custom supervised information extraction.
-   Large paid APIs.
-   Large-scale full-paper LLM processing.
-   GRU4Rec as the primary recommender.
-   Complex neural recommenders as the primary system.
-   GNN/RL recommenders.
-   Full knowledge-graph recommender.
-   Massive corpora.
-   Distributed/production-scale infrastructure.
-   Real-time ingestion.
-   Large user studies.
-   Unconstrained LLM explanations.
-   Summarization/agent functionality.
-   Citation prediction as the primary task.
-   Large-scale collaborative filtering.
-   Social recommendation.
-   Complex lifelong user modeling.
-   Exhaustive multimodal processing.
-   Custom search models.
-   Custom ANN models.
-   Custom embedding models.
-   Custom NLI models.
-   Extensive hyperparameter searches.

The intended system instead focuses on:

-   Approximately 100k-scale corpus processing.
-   Periodic batch processing.
-   Pretrained models.
-   Hybrid lexical + dense retrieval.
-   Structured paper profiles.
-   Citation signals.
-   Session/history personalization.
-   Grounded explanations.
-   Novelty and diversity.
-   Temporal offline evaluation.

------------------------------------------------------------------------

# 36. Current End-to-End Pipeline

The current system can be represented as:

` text
                    ┌─────────────────────┐
                    │      OpenAlex       │
                    └──────────┬──────────┘
                                │
                                ▼
                    ┌─────────────────────┐
                    │ Data Cleaning /     │
                    │ Structuring         │
                    └──────────┬──────────┘
                                │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
        ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
        │ BM25 Index  │ │ Vector Index│ │ Citation    │
        │             │ │             │ │ Information │
        └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
               │               │               │
               └───────┬───────┘               │
                       ▼                       │
                ┌─────────────┐               │
                │ RRF Fusion  │               │
                └──────┬──────┘               │
                       ▼                       │
                ┌─────────────┐               │
                │ Reranking   │               │
                └──────┬──────┘               │
                       │                       │
                       ▼                       │
                ┌─────────────┐               │
                │ Candidates  │◄──────────────┘
                └──────┬──────┘
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
      ┌──────────────┐   ┌──────────────┐
      │ V-SKNN State │   │ User Persona │
      │ Momentum s_t │   │ Profile (p_u)│
      └──────┬───────┘   └──────┬───────┘
             │                  │
             └────────┬─────────┘
                      ▼
               ┌─────────────┐
               │ V-PaperFlow │
               │ Multi-Signal│
               │ Ranking     │
               └──────┬──────┘
                      ▼
               ┌─────────────┐
               │ Calibration │
               └──────┬──────┘
                      ▼
               ┌─────────────┐
               │ MMR         │
               │ Diversity   │
               └──────┬──────┘
                      ▼
               ┌─────────────┐
               │ Top-10      │
               │ Papers      │
               └──────┬──────┘
                      │
                      ▼
               ┌─────────────┐
               │ Profile-    │
               │ grounded    │
               │ Explanation │
               └──────┬──────┘
                      ▼
               ┌─────────────┐
               │ NLI Check   │
               └──────┬──────┘
                      ▼
                 Final Feed


------------------------------------------------------------------------

# 37. Current Decision Status

## Fixed

The following architectural and methodological decisions are now fixed:

-   **Recommendation Target:** Academic papers.
-   **Corpus Sources & Scale:** OpenAlex CS/AI subset (~100,000 papers).
-   **Deduplication & Author Resolution:** Closed via canonical OpenAlex entity IDs (`openalex_id`, `doi`).
-   **First-Stage Retrieval:** Hybrid BM25s (lexical) + dense retrieval fused via Reciprocal Rank Fusion (RRF, $k=60$) retrieving Top-100 candidates.
-   **Paper Encoder:** SPECTER2 with `proximity` adapter for scientific paper representation.
-   **Query Encoder:** SPECTER2 with `adhoc_query` adapter for search queries (HyDE dropped).
-   **Extraction Method:** GLiNER (`urchade/gliner_medium-v2.1`) for zero-shot facet profiling.
-   **Profile Schema:** 5 core facets (`task`, `method`, `dataset`, `metric`, `topic`) stored in SQLite with exact character offsets and verbatim evidence spans.
-   **Preprocessing Utility:** scispaCy retained strictly for sentence boundary splitting and scientific tokenization.
-   **Primary Recommender:** V-PaperFlow (Hybrid Gated V-SKNN + PaperFlow) producing Top-10 recommendations.
-   **Ablation Baselines:** Pure V-SKNN and pure PaperFlow as competitive baseline models (SASRec dropped as out-of-scope).
-   **User Representation:** Dual-component model — Faceted Persona Profile ($\mathbf{p}_u$) + Dynamic Sequential Session ($\mathbf{s}_u$).
-   **Citation Signal (Section 16):** Multi-tier piecewise scoring (+2.0 direct citation, +1.2 bibliographic coupling, +0.7 co-citation, +0.2 prior work, 0.0 otherwise) with exponential temporal decay $\exp(-\gamma_{\text{cite}} \Delta t)$ and self-citation dampening (0.5x).
-   **Diversity (Section 18):** Facet-aware MMR (Maximal Marginal Relevance) combining embedding distance and facet dissimilarity.
-   **Novelty (Section 19):** Semantic novelty from interaction history + Facet Novelty ($w_m N_{\text{method}} + w_d N_{\text{dataset}} + w_t N_{\text{topic}}$) with consumed paper exclusion.
-   **Calibration Pipeline (Section 20):** 4-stage pipeline: 1) Min-Max/Sigmoid score normalization, 2) Multi-signal linear combination, 3) Novelty scaling, 4) Post-scoring MMR diversification.
-   **Explanation Engine:** Template-based, profile-grounded explanations with NLI faithfulness verification.
-   **Retrieval Evaluation (Section 26):** CSFCube + LitSearch + SCIDOCS benchmarks; primary gatekeeper metric Recall@100 alongside nDCG@{10,100} and MRR; 3-way retrieval ablation (BM25s vs. SPECTER2 vs. RRF Hybrid).
-   **Extraction Evaluation (Section 27):** SciERC benchmark with Strict & Relaxed Span $F_1$; Head-to-head GLiNER vs. local Qwen LLM baseline (Section 8.6); 100-paper manual in-corpus audit for Precision@Facet, zero-hallucination verification, and facet completeness.
-   **Recommendation Evaluation (Section 28):** Leave-Future-Out (LFO) temporal oracle based on OpenAlex author timelines and next-paper reference lists $\text{Ref}(B)$ under strict anti-leakage quarantine; Metrics: NDCG@10, Recall@10, MRR, Intra-List Diversity (ILD), and Facet Coverage Rate; 3-way core model ablation (V-SKNN vs. PaperFlow vs. V-PaperFlow) and pipeline component ablation.
-   **User Study Scale (Section 30):** Fixed at $n = 50$ users.
-   **Primary Storage:** SQLite database for metadata, paper profiles, user interactions, and audit logs.
-   **Reproducibility Commitment (Section 33):** Fixed deterministic random seeds, locked virtual environment specifications, structured YAML/JSON configurations, and versioned evaluation splits.
-   **Temporal Leakage Guard:** Strict enforcement across all evaluation splits.

## Tentative

The following decisions have a clear proposed direction and are undergoing final technical selection:

-   **Neural Reranker:** Optional cross-encoder (e.g., `BAAI/bge-reranker-base` or `ms-marco-MiniLM-L-6-v2`) applied between Top-100 RRF candidate retrieval and multi-signal scoring.
-   **Dense Vector Store:** In-memory NumPy/Torch indexing vs. FAISS IndexFlatIP / Qdrant for 100k vectors.
-   **User-Study Task Protocol:** Exact task design (search vs feed browsing) and standardized Likert evaluation questionnaire rubrics for the $n=50$ participant pool.
-   **Explanation Failure Fallback:** Policy for recommendations whose generated explanations fail NLI verification (suppress explanation vs. fall back to high-confidence category template).
-   **External API Fallback Policy:** Zero-cost budget guarantees with local Qwen fallback if external API limits are exceeded.

## Open (Empirical Hyperparameters & Operational Cadence)

Decisions reserved for empirical validation during offline experiments:

-   **V-PaperFlow Hyperparameters:** Optimal session exponential decay $\lambda \in [0.7, 0.95]$, persona weight $\alpha$, and pivot drift threshold $\gamma$.
-   **Ranking Signal Weights:** Tuned weights ($w_{\text{session}}, w_{\text{persona}}, w_{\text{cite}}, w_{\text{rrf}}, w_{\text{nov}}$) via grid search or Nelder-Mead on the validation split.
-   **MMR Trade-off Parameter:** Optimal diversity balance $\lambda_{\text{MMR}} \in [0.4, 0.8]$.
-   **Citation Decay Rate:** Empirical parameter $\gamma_{\text{cite}}$ for publication recency dampening.
-   **Batch Processing Cadence:** Ingestion schedule for updating new OpenAlex paper batches (e.g., weekly vs. monthly cron).
-   **LLM Cache Policy:** Exact disk cache retention and eviction thresholds for explanation / profile artifacts.

------------------------------------------------------------------------

# 38. Current Design in One Sentence

**Paper Feed currently consists of a batch-built OpenAlex paper
corpus, hybrid BM25 + dense retrieval fused with RRF and optionally
reranked, GLiNER-based structured paper profiling with evidence spans,
V-PaperFlow (hybrid gated V-SKNN with faceted persona fusion) augmented by citation and other signals (benchmarked against pure V-SKNN and PaperFlow),
calibrated novelty/diversity-aware ranking with MMR, and
template-based profile-grounded explanations verified with NLI under
strict temporal evaluation constraints.**
