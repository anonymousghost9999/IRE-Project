# Problem Statement: Personalized Academic Paper Recommender

## 1. The Core Problem in Simple Words
Thousands of research papers are published every month across Computer Science and AI. Researchers cannot possibly read or search through all of them. 

Search engines (like Google Scholar or arXiv search) only help when a researcher already knows exactly what keywords to search for. But researchers also need an **intelligent, personalized recommendation feed** that automatically surfaces relevant, high-quality, and useful papers directly to them based on their daily interactions, reading behavior, click logs, likes, bookmarks, and past publications.

---

## 2. What Does a Researcher Actually Need?
A good academic paper recommender must deliver on four primary objectives:

1. **Relevance to Long-Term Research:**
   - Papers that align with their core domain, primary research tasks (e.g., *Information Extraction*), architectures/methods (e.g., *Transformers, Diffusion*), and evaluation benchmarks (e.g., *SciERC, MS MARCO*).
2. **Relevance to Immediate / Short-Term Interest (Session Drift):**
   - If a researcher temporarily explores a new sub-topic today (e.g., an NLP researcher investigating *"LoRA quantization"* this morning), the feed must adapt rapidly to their active focus within the session without wiping out their broader background.
3. **Genuine Discovery (Not Just What They Already Know):**
   - Simply recommending papers they have already cited or read is useless. The system must surface **new discoveries**—papers introducing fresh methods or alternative perspectives to their research area.
4. **Transparency & Trust ("Why am I seeing this?"):**
   - Researchers are skeptical users. Every recommendation must feature an inspectable, grounded reason (e.g., *"Recommended because this paper tests a new Method on the SciERC dataset you frequently read about"*).

---

## 3. What Information (Signals) Can We Use?

To understand and predict what a researcher wants, the system operates on two complementary signal types:

### A. Explicit Signals (Direct User Inputs)
* **Onboarding Selections:** Research domains and seed topics chosen during sign-up (e.g., *Information Retrieval, Dense Passage Retrieval*).
* **Direct Feedback Actions:** Bookmarking / saving to library, "thumbs up" (interested), or "skip / dismiss" (pushing similar items down).
* **Authored Publications:** Past papers written by the user (providing initial author identity and reference history).

### B. Implicit Signals (Behavioral & Network Telemetry)
* **Real-Time Session Browsing:** Which paper abstracts the user clicked, expanded, or lingered on (dwell time) during their active session.
* **Citation Graph Structure:**
  * Papers the user has cited in past publications.
  * Papers sharing references with the user's reading history (**bibliographic coupling**).
  * Papers frequently co-cited alongside the user's core reading list (**co-citation**).
* **Extracted Scientific Facets:**
  * Fine-grained, typed scientific entities extracted from paper abstracts: **Tasks**, **Methods**, **Datasets**, **Metrics**, and **Topics**.

---

## 4. Key Challenges & The Telemetry Solution

1. **Dual-Horizon Interests (Long-Term Persona vs. Short-Term Drift):**
   - Balancing a permanent profile $\mathbf{p}_u$ with transient session exploration $\mathbf{s}_u$ so that temporary curiosity doesn't permanently ruin future recommendations.
2. **The Cold-Start Problem:**
   - How to serve meaningful recommendations to a brand-new user with zero clicks and no publications (resolved via onboarding seed topics + recent high-impact benchmark papers).
3. **Avoiding Echo Chambers (Relevance vs. Diversity):**
   - Ensuring the Top-10 feed doesn't collapse into 10 minor variations of the exact same model (resolved via Maximal Marginal Relevance diversification).
4. **The Evaluation Dilemma & The LLM Simulation Solution:**
   - **The Problem:** Commercial paper platforms (Google Scholar, Semantic Scholar) keep researcher clickstreams strictly private. OpenAlex provides publication and citation history, but **zero real-time click logs, skips, or dwell times**.
   - **The Solution (Two-Track Evaluation):**
     * **Track 1 (Offline Macro-Benchmark on OpenAlex):** Evaluates long-term discovery using the **Leave-Future-Out (LFO)** protocol. Historical papers up to year $T$ form the user profile; the reference list of their next paper published at $T+1$ serves as the ground-truth oracle for papers they needed to read.
     * **Track 2 (Session Dynamics via LLM User Simulation):** Following the established **Agent4Rec / RecAgent / PaperFlow** methodology, an automated cohort of diverse LLM-simulated researcher personas browses the system, emitting realistic click, skip, and dwell-time events across multi-step episodes to benchmark session adaptation and drift.

---

## 5. Formal Problem Definition

> **Given:**
> 1. A user $u$ defined by:
>    - A long-term profile $\mathbf{p}_u$ (past publications, explicit onboarding topics, and preferred facets).
>    - A dynamic session state $\mathbf{s}_u$ (chronological sequence of papers clicked, read, or skipped in the current session).
> 2. A corpus $\mathcal{C}$ of academic papers (e.g., 100,000 CS/AI papers with titles, abstracts, citation edges, and extracted facets).
> 3. An exclusion set $\mathcal{E}_u$ containing papers authored by $u$ and papers already consumed/saved in the current session.
>
> **Goal:**
> For any interaction step $t$, execute a two-stage recommendation funnel:
> 1. **Candidate Retrieval:** Rapidly filter $\mathcal{C} \setminus \mathcal{E}_u$ down to a Top-100 candidate pool using dense semantic similarity and citation graph expansion.
> 2. **Multi-Signal Reranking & Diversification:** Rank the candidate pool by balancing session momentum, persona alignment, citation strength, and novelty, followed by MMR diversification to output a final list $\mathcal{R}_u$ of size $K=10$ that:
>    - Maximizes relevance to both $\mathbf{p}_u$ and $\mathbf{s}_u$.
>    - Maximizes diversity and discovery (covering distinct methods and unread facets).
>    - Provides an inspectable, factual template explanation for every surfaced paper.

---

## 6. Everything We Need to Build (Complete Component Breakdown)

To turn this problem statement into a working system and an evaluation-ready project, here is the complete checklist of every minute piece that must be built:

### A. The User-Facing Web Application (Frontend)
A clean, modern web interface (HTML/CSS/JS or lightweight frontend framework) designed for researchers to browse and interact:

1. **User Onboarding Screen:**
   - Topic/Domain selector (e.g., *NLP, Computer Vision, IR, Reinforcement Learning*).
   - Seed keyword / task inputs (e.g., *"Large Language Models", "Dense Retrieval"*).
   - Optional: Enter their own past papers or publications.
2. **The Main Recommendation Feed Page (Top-10 Feed):**
   - **Paper Cards:** Clean layout displaying title, author list, publication year, venue/journal badge, and a collapsible abstract snippet.
   - **Direct Paper Links:** Clickable DOI, arXiv link, or OpenAlex link to access the full paper.
   - **Extracted Facet Pills:** Visual badges on each card showing what was extracted from the paper:
     - 🎯 **Task:** (e.g., *Text Summarization*)
     - ⚙️ **Method:** (e.g., *Sparse Attention Transformer*)
     - 📊 **Dataset:** (e.g., *CNN/DailyMail*)
     - 📈 **Metric:** (e.g., *ROUGE-L*)
   - **Explanation Banner:** A dedicated highlight pill explaining why it was shown:
     - e.g., *"Recommended because this paper tests a new Method on the Dataset you frequently read about."*
   - **Interactive User Action Buttons:**
     - ❤️ **Like / Upvote:** Registers strong positive interest.
     - 🔖 **Bookmark / Save to Library:** Saves the paper for later reading.
     - ❌ **Skip / Dismiss:** Dislikes or skips the paper (registers negative feedback and pushes similar papers down).
   - **Passive Telemetry (Click & Dwell-Time Tracking):**
     - JavaScript timer that logs how many seconds a user kept a paper card expanded or hovered over it before moving on (implicit reading behavior).
3. **Ad-Hoc Search Bar (Switching to Search Mode):**
   - Allows the researcher to type a natural-language search query (e.g., *"low rank adaptation for vision"*) when they have a specific question, switching seamlessly from Feed Mode to Search Mode.
4. **User Profile & Library Page:**
   - **My Library:** List of all bookmarked and read papers.
   - **Interest Inspector:** Visual display showing what the system currently thinks the user's persona is (their top tasks, methods, and topics).
   - **Reset Session Drift Button:** One-click button allowing the user to say *"I'm done exploring this side-topic, take me back to my main research field."*

---

### B. The Backend Application Server (FastAPI / Python)
The server orchestrating real-time recommendations and user session management:

1. **API Endpoints:**
   - `POST /api/onboarding`: Saves a new user's initial chosen topics and seed profile.
   - `GET /api/feed?user_id=123`: Generates and returns the personalized Top-10 papers with facets and explanations.
   - `POST /api/interaction`: Receives and records user events (`click`, `dwell_time`, `like`, `bookmark`, `skip`) and immediately updates the user's session vector.
   - `GET /api/search?q=...`: Runs hybrid lexical + dense retrieval for ad-hoc search queries.
   - `GET /api/user/library`: Fetches all saved papers and interaction history.
   - `POST /api/user/reset-session`: Flushes short-term session drift back to the core persona.
2. **Recommendation Logic Engine:**
   - Fast session-decay calculator (updating session vector on each click).
   - Candidate generator (dense vector kNN + citation-graph neighbors).
   - Multi-signal scoring combiner (combining session, persona, citations, and novelty).
   - MMR (Maximal Marginal Relevance) post-ranker to enforce Top-10 diversity.
   - Deterministic template engine that generates the explanation string from extracted facets.

---

### C. Offline Data Ingestion & Preprocessing Pipeline
Background Python scripts to collect, clean, and enrich the paper corpus before the web app runs:

1. **OpenAlex Data Fetcher & Reconstructor:**
   - API crawler pulling ~100,000 CS/AI papers.
   - **Inverted Index Decoder:** OpenAlex stores abstracts as word-position maps (`abstract_inverted_index`); this script reconstructs them into readable plain text.
   - **Missing Abstract Handler:** Identifies papers without abstracts and applies fallback policies (title-only or clean exclusion).
2. **Time-Bounded Citation Graph Builder:**
   - Extracts citation links between papers.
   - Computes **co-citation counts** and **bibliographic coupling** strictly using papers published $\leq$ cutoff year $T$ (preventing temporal leakage).
3. **Batch Dense Embedding Generator:**
   - Runs pretrained **SPECTER2** (with `proximity` adapter) over all 100k paper titles and abstracts.
   - Saves 768-dimensional normalized dense vectors.
4. **Batch Information Extractor (GLiNER):**
   - Runs **GLiNER** (`urchade/gliner_medium-v2.1`) locally on the 100k abstracts.
   - Extracts typed entities: `task`, `method`, `dataset`, `metric`, `topic`.
   - Records verbatim character offsets and supporting evidence sentences.
5. **Search Indexes:**
   - **BM25s Index:** Tokenizes and builds a pure-Python lexical inverted index for keyword search.
   - **Dense Vector Index:** Builds a FAISS (or NumPy cosine) index for fast candidate retrieval.

---

### D. Structured Database (SQLite)
A clean, local relational database storing all structured records:

* `papers`: `paper_id`, `title`, `abstract_text`, `authors`, `year`, `venue`, `citation_count`, `doi`.
* `paper_facets`: `paper_id`, `facet_type`, `facet_value`, `evidence_sentence`, `start_char`, `end_char`.
* `paper_citations`: `source_paper_id`, `target_paper_id`, `publication_year`.
* `users`: `user_id`, `name`, `created_at`.
* `user_profiles`: `user_id`, `persona_vector` (BLOB), `preferred_facets` (JSON).
* `user_sessions`: `session_id`, `user_id`, `session_vector` (BLOB), `last_active_timestamp`.
* `user_interactions`: `interaction_id`, `user_id`, `session_id`, `paper_id`, `action_type` (`view`, `click`, `like`, `bookmark`, `skip`), `dwell_seconds`, `timestamp`.

---

### E. Evaluation & Benchmarking Suite (For Final Report & Paper)
The scientific harness to measure, ablate, and prove the system's performance:

1. **Leave-Future-Out (LFO) Recommendation Evaluation Harness (OpenAlex):**
   - Automatically builds author histories up to year $T=2023$ and tests predictions against references in their next paper $B$ published in $T+1=2024$.
   - Calculates **NDCG@10**, **Recall@10 / HitRate@10**, and **MRR**.
   - Evaluates non-accuracy metrics: **Intra-List Diversity (ILD)** and **Facet Coverage Rate (Novelty)**.
   - Evaluates **Discoveries** (new references) and **Re-citations** separately while filtering out self-citations.
   - Runs the **3-way ablation** (`Session-Only` vs. `Persona-Only` vs. `Hybrid V-PaperFlow`).
2. **LLM-Based User Simulation Harness (Agent4Rec / RecAgent Paradigm):**
   - Instantiates a cohort of 20–50 LLM researcher personas with defined research goals.
   - Simulates multi-turn feed browsing, generating realistic click, dwell-time, and skip events stored directly in SQLite.
   - Evaluates **Session Drift Adaptation Speed**: Measures how many clicks it takes for the recommender to pivot when a user abruptly changes research topics.
3. **First-Stage Retrieval Benchmark Script:**
   - Tests candidate retrieval on **CSFCube** and **LitSearch** evaluating **Recall@100** (BM25s vs. SPECTER2 vs. Hybrid RRF).
4. **Extraction Evaluation Script:**
   - Runs GLiNER against the **SciERC** gold-standard dataset to report Strict and Relaxed Span $F_1$.
5. **Manual Profile Audit Tool:**
   - CLI script to inspect a random sample of 100 extracted profiles to manually check precision and verify 100% zero-hallucination evidence.
