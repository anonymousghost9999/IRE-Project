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
4. **Transparency & Inspectability ("Why am I seeing this?"):**
   - Researchers are skeptical users. Explanations must satisfy two criteria:
     * **Factual Grounding:** The claimed facet or citation relation must be grounded in verbatim abstract text or verified graph edges (never hallucinated).
     * **Ranking Faithfulness:** The stated explanation must be *mechanistically faithful*—meaning the highlighted feature was actually the dominant mathematical signal that elevated the paper into the Top-10 feed (e.g., introducing a *"new Method"* formally defined as $\text{method}_p \notin \mathcal{F}_u$ for a task the user actively studies).

---

## 3. What Information (Signals) Can We Use?

To understand and predict what a researcher wants, the system operates across three distinct information categories:

### A. User Explicit & Behavioral Signals
* **Explicit Onboarding Inputs:** Research domains and seed topics chosen during sign-up (e.g., *Information Retrieval, Dense Passage Retrieval*).
* **Explicit Feedback Actions:** Bookmarking / saving to library, "thumbs up" (positive interest), or "skip / dismiss" (negative penalty).
* **Authored Publications:** Past papers written by the user (providing initial author identity and baseline research profile).
* **Real-Time Session Browsing:** Which paper abstracts the user clicked, expanded, or read during their active browsing session.

### B. Graph Topological Signals (Citation Structure)
* **Direct Historical Citations:** Papers the user personally cited in their past publications ($t \leq T$).
* **Bibliographic Coupling:** Papers sharing references with the user's reading history ($|\text{Refs}(p) \cap \text{Refs}(\mathcal{H}_u)| \ge 1$).
* **Co-Citation Proximity:** Papers frequently cited alongside the user's core publications by third-party papers.

### C. Document Content & Extracted Facets
* **Dense Semantic Embeddings:** 768-dimensional document vectors encoded via SPECTER2 (`proximity` adapter).
* **Extracted Scientific Facets:** Fine-grained, typed scientific entities extracted from paper abstracts via GLiNER: **Tasks**, **Methods**, **Datasets**, **Metrics**, and **Topics**.

---

## 4. Key Challenges & The Telemetry Solution

1. **Dual-Horizon Interests (Long-Term Persona vs. Short-Term Drift):**
   - Balancing a permanent profile $\mathbf{p}_u$ with transient session exploration $\mathbf{s}_u$ so that temporary curiosity doesn't permanently ruin future recommendations.
2. **The User & Item Cold-Start Problems:**
   - **User Cold-Start:** Serving high-quality recommendations to a brand-new user with zero clicks and no publications (resolved via onboarding seed topics + recent high-impact benchmark papers forming initial $\mathbf{p}_u$).
   - **Item Cold-Start & Overcoming Citation Bias:** Newly published papers have zero historical citations. Relying solely on citation strength biases recommendations toward older papers. To surface fresh papers, the ranking engine pairs citation signals with citation-independent **content signals** (SPECTER2 semantic proximity and exact facet alignment), allowing zero-citation preprints to rank high if their methodology matches the user's focus.
3. **Avoiding Echo Chambers (Relevance vs. Diversity):**
   - Ensuring the Top-10 feed doesn't collapse into 10 minor variations of the exact same model (resolved via Maximal Marginal Relevance diversification).
4. **The Evaluation Dilemma & The Ground-Truth Solution:**
   - **The Challenge:** Commercial academic search engines (Google Scholar, Semantic Scholar) keep researcher clickstreams strictly private. OpenAlex provides publication and citation metadata, but zero real-time click logs, skips, or dwell times.
   - **The Two-Track Evaluation Solution:**
     * **Track 1 (Macro-Evaluation on Real Human Behavior via LFO):**
       - Evaluates whether the system can discover and recommend the papers a real researcher ended up citing in their next publication.
       - **Temporal Guard ($\leq T$):** Historical papers up to year $T=2023$ build the user profile $\mathbf{p}_u$. For target Paper $B$ published in $T+1=2024$, its ground truth references are strictly restricted to papers published $\leq T$ (excluding references published in $T+1$ which could not have been known at time $T$).
       - **Formal Definition of Discovery vs. Re-Citation:**
         $$\text{Discovery}(B \mid u) = \{ p \in \text{Ref}(B) \mid \text{year}(p) \leq T \text{ and } p \notin \text{Citations}(u, \leq T) \cup \text{Authored}(u, \leq T) \}$$
         $$\text{ReCitation}(B \mid u) = \{ p \in \text{Ref}(B) \mid \text{year}(p) \leq T \text{ and } p \in \text{Citations}(u, \leq T) \}$$
         *(Self-citations authored by $u$ are strictly filtered out to prevent trivial memorization).*
       - **Corpus Coverage Guard:** The 100k corpus is constructed outward from the evaluation authors' citation neighborhoods to ensure their references are well-represented. Both unconstrained Recall and in-corpus Recall are reported alongside the Corpus Coverage Ceiling.
       - **Author Disambiguation Guard:** Evaluated on single or first authors with clean publication histories to ensure $\text{Ref}(B)$ genuinely reflects user $u$'s research focus rather than co-author noise.
     * **Track 2 (Interactive Session Dynamics via LLM Simulation):**
       - Evaluates real-time session adaptation and click response (detailed below).

---

## 5. Formal Problem Definition

> **Given:**
> 1. A user $u$ modeled by two components:
>    - **Long-Term Persona Profile:** $\mathbf{p}_u = (\mathbf{v}_u^{\text{persona}}, \mathcal{F}_u)$, where $\mathbf{v}_u^{\text{persona}} \in \mathbb{R}^d$ is the normalized dense persona embedding (derived from past publications and onboarding seed topics), and $\mathcal{F}_u$ is the set of historical preferred facets (tasks, methods, datasets).
>    - **Dynamic Session State:** $\mathbf{s}_u = (\mathbf{v}_t^{\text{session}}, \mathcal{S}_u^{-})$, where $\mathbf{v}_t^{\text{session}} \in \mathbb{R}^d$ is the position-decayed representation of positive interactions (clicked/liked papers), and $\mathcal{S}_u^{-} = \{d_1^{-}, d_2^{-}, \dots\}$ is the set of skipped/dismissed papers acting as negative feedback.
> 2. A corpus $\mathcal{C}$ of academic papers, where each paper $p \in \mathcal{C}$ has title/abstract embedding $\mathbf{e}_p \in \mathbb{R}^d$, citation edges $\mathcal{G}$, and extracted scientific facets $\mathcal{F}_p$.
> 3. An explicit exclusion set $\mathcal{E}_u = \text{Authored}(u) \cup \text{Citations}(u, \leq T) \cup \text{SessionConsumed}(u)$, guaranteeing that already written, cited, or read papers are never re-recommended.
>
> **Optimization Objective:**
> For any user state $(\mathbf{p}_u, \mathbf{s}_u)$, select a ranked list $\mathcal{R}_u \subset \mathcal{C} \setminus \mathcal{E}_u$ of size $K$ (e.g., $K=10$) that maximizes the multi-objective utility:
> $$\max_{\mathcal{R}_u, |\mathcal{R}_u|=K} \left[ \sum_{p \in \mathcal{R}_u} \text{Score}(p \mid \mathbf{p}_u, \mathbf{s}_u, \mathcal{G}) + \lambda_{\text{div}} \cdot \text{Diversity}(\mathcal{R}_u) + \lambda_{\text{nov}} \cdot \text{Novelty}(\mathcal{R}_u \mid \mathcal{F}_u) - \lambda_{\text{neg}} \cdot \sum_{p \in \mathcal{R}_u} \max_{n \in \mathcal{S}_u^{-}} \cos(\mathbf{e}_p, \mathbf{e}_n) \right]$$
> accompanied by an inspectable, faithful justification for every paper $p \in \mathcal{R}_u$.
> *(In our two-stage architecture, this optimization is solved efficiently by first retrieving $M=100$ candidate papers via hybrid dense+sparse retrieval, then executing multi-signal scoring and MMR greedy diversification).*

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
   - Automatically builds author histories up to year $T=2023$ and tests predictions against references in their next paper $B$ published in $T+1=2024$ (filtered strictly to references published $\leq T$).
   - Calculates **NDCG@10**, **Recall@10 / HitRate@10**, and **MRR** against both the full reference set and the in-corpus subset (reporting the Corpus Coverage Ceiling).
   - Evaluates non-accuracy metrics: **Intra-List Diversity (ILD)** and **Facet Coverage Rate (Novelty)**.
   - Evaluates **Discoveries** (first-time cited papers) and **Re-citations** separately while strictly filtering out self-citations.
   - **Core Baseline Comparisons to Beat:**
     1. *Popularity Baseline:* Most-cited recent papers in the user's primary domain (counts $\leq T$).
     2. *Lexical Baseline:* BM25s retrieval using the user's combined past paper titles/abstracts as a query.
     3. *Dense Centroid Baseline:* Unweighted mean of SPECTER2 embeddings across user's past papers (no persona weights, no decay).
     4. *Citation-Only Baseline:* Pure bibliographic coupling and co-citation graph score without text features.
   - **Component Ablations (Isolating Model Features):**
     * *No-MMR:* Disables diversity re-ranking (measures drop in Intra-List Diversity).
     * *No-Graph ($w_{\text{cite}} = 0$):* Content-only ranking (measures drop in discovery of authoritative papers).
     * *No-Facets ($w_{\text{facet}} = 0$):* Dense-only ranking without GLiNER task/method alignment.
   - **Formal Success Hypotheses (Project Acceptance Criteria):**
     * **$H_1$ (Accuracy):** The proposed hybrid model achieves statistically significant improvements in NDCG@10 and Discovery Recall@10 over all four baselines (paired bootstrap test, $p < 0.05$).
     * **$H_2$ (Diversity):** MMR diversification significantly increases Intra-List Diversity (ILD) with $\le 3\%$ degradation in NDCG@10.
     * **$H_3$ (Session Adaptation in Track 2):** Position decay achieves lower Pivot Latency ($k_{\text{pivot}}$) than static persona matching during topic shifts.
2. **LLM-Based User Simulation Harness (Agent4Rec / RecAgent Paradigm):**
   - **Role:** Serves strictly as a behavioral *sanity check on interactive session dynamics* (not proof of human relevance), testing how the algorithm adapts to real-time feedback.
   - Instantiates a cohort of 20–50 LLM researcher personas with defined research goals, seeded from actual OpenAlex author profiles.
   - Evaluates discrete interaction choices (`click`, `skip`, `save`) across multi-turn sessions (discarding arbitrary dwell-time integers).
   - **Core Metric — Pivot Latency ($k_{\text{pivot}}$):** When a user abruptly switches research intent from Topic A to Topic B, counts the exact number of clicks until Topic B papers represent $\ge 50\%$ of the Top-10 feed.
   - **Track 2 Ablation Suite (Session Dynamics):** Compares *Pure Session-Only Decay* vs. *Pure Persona-Only Static* vs. *Hybrid V-PaperFlow* to prove the hybrid adapts to session drift faster without permanently losing long-term focus.
3. **First-Stage Retrieval Benchmark Script:**
   - Tests candidate retrieval on **CSFCube** and **LitSearch** evaluating **Recall@100** (BM25s vs. SPECTER2 vs. Hybrid RRF).
4. **Extraction Evaluation Script (SciERC Benchmark):**
   - Runs GLiNER against the **SciERC** gold-standard dataset across both **Strict Span $F_1$** (exact offsets + type) and **Relaxed Span $F_1$** (token overlap $\ge 1$ + type).
   - Formally maps profile schema to SciERC annotations: `task` $\leftrightarrow$ `Task`, `method` $\leftrightarrow$ `Method`, `metric` $\leftrightarrow$ `Metric`, `dataset` $\leftrightarrow$ `Material`, `topic` $\leftrightarrow$ `OtherScientificTerm` (acknowledging `Material` and `OtherScientificTerm` as broader approximations).
5. **Manual Profile & Explanation Faithfulness Audit Tool:**
   - CLI script to inspect a random sample of 100 extracted profiles and recommendations:
     - **Extractive Grounding Verification:** Verifies that all extracted spans and evidence sentences exist verbatim in the abstract text (preventing generative token fabrication, though label classification accuracy is measured separately via F1).
     - **Explanation Faithfulness:** Verifies that the displayed reason corresponds to the highest positive feature contribution in the ranking score.
     - **Explanation Coverage:** Measures percentage of Top-10 recommendations successfully annotated with a grounded explanation (Target: $\ge 90\%$).
     - Optional qualitative pilot ($n=8\text{--}10$ peers) scoring explanation clarity and helpfulness on a 1–5 Likert scale.
