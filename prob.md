# Problem Statement: Personalized Academic Paper Recommender

## 1. The Core Problem in Simple Words
Thousands of research papers are published every month across Computer Science and AI. Researchers cannot possibly read or search through all of them.

Search engines (like Google Scholar or arXiv search) only help when a researcher already knows exactly what keywords to search for. But researchers also need an **intelligent, personalized recommendation feed** that automatically surfaces relevant, high-quality, and useful papers directly to them based on their research profile, past publications, and active session browsing behavior.

The primary academic focus of this project is **Retrieval and Recommendation Quality**—specifically balancing long-term research profiles with real-time session exploration, promoting scientific discovery rather than echo chambers, and providing mathematically faithful explanations.

---

## 2. What Does a Researcher Actually Need?
A good academic paper recommender must deliver on four primary objectives:

1. **Relevance to Long-Term Research:**
   - Papers that align with their core domain, primary research tasks (e.g., *Information Extraction*), architectures/methods (e.g., *Transformers, Diffusion*), and evaluation benchmarks (e.g., *SciERC, MS MARCO*).
2. **Relevance to Immediate / Short-Term Interest (Session Drift):**
   - If a researcher temporarily explores a new sub-topic today (e.g., an NLP researcher investigating *"LoRA quantization"* this morning), the feed must adapt rapidly to their active focus within the session without wiping out their broader research background.
3. **Genuine Discovery (Not Just What They Already Know):**
   - Simply recommending papers the researcher has already cited or authored is an echo-chamber failure. The system must surface **new discoveries**—relevant papers introducing fresh methods or alternative perspectives that the user has not cited in their past publications.
4. **Transparency & Inspectability ("Why am I seeing this?"):**
   - Researchers are skeptical users. Explanations must satisfy two criteria:
     * **Factual Grounding:** The claimed facet or citation relation must be grounded in verbatim abstract text or verified citation graph edges (never hallucinated).
     * **Ranking Faithfulness:** The stated explanation must be *mechanistically faithful*—meaning the highlighted feature was actually the dominant mathematical signal that elevated the paper into the Top-10 feed (e.g., introducing a *"new Method"* formally defined as $\text{method}_p \notin \mathcal{F}_u$ whose positive score component $w_{\text{nov}} \cdot \mathbf{1}[\text{method}_p \notin \mathcal{F}_u]$ provided the critical score contribution).

---

## 3. What Information (Signals) Can We Use?

To understand and predict what a researcher wants, the system operates across three distinct information categories:

### A. User Explicit & Behavioral Signals
* **Explicit Onboarding Inputs:** Research domains and seed topics chosen during sign-up (e.g., *Information Retrieval, Dense Passage Retrieval*).
* **Explicit Feedback Actions:** Bookmarking / saving to library, "thumbs up" (positive interest), or "skip / dismiss" (negative penalty).
* **Authored Publications:** Past papers published by the user up to time $T$ (providing author identity and baseline research profile).
* **Real-Time Session Browsing:** Which paper abstracts the user clicked, expanded, or bookmarked during their active browsing session.

### B. Graph Topological Signals (Citation Structure)
* **Direct Historical Citations:** Papers the user personally cited in their past publications ($t \leq T$).
* **Bibliographic Coupling:** Papers sharing references with the user's historical publications:
  $$\text{BC}(p \mid u) = |\text{Refs}(p) \cap \text{Refs}(\text{Authored}(u, \le T))|$$
* **Co-Citation Proximity:** Papers frequently cited alongside the user's core publications by third-party papers published $\leq T$.

### C. Document Content & Extracted Facets
* **Dense Semantic Embeddings:** 768-dimensional document vectors encoded via SPECTER2 (`proximity` adapter).
* **Extracted Scientific Facets:** Fine-grained, typed scientific entities extracted from paper abstracts via GLiNER: **Tasks**, **Methods**, **Datasets**, **Metrics**, and **Topics**.

---

## 4. Key Challenges & Scientific Evaluation Strategy

1. **Dual-Horizon Interests (Long-Term Persona vs. Short-Term Drift):**
   - Balancing a permanent profile $\mathbf{p}_u$ with transient session exploration $\mathbf{s}_u$ so that temporary curiosity doesn't permanently ruin future recommendations.
2. **The User & Item Cold-Start Problems:**
   - **User Cold-Start:** Serving high-quality recommendations to a brand-new user with zero clicks and no publications (resolved via onboarding seed topics + recent high-impact benchmark papers forming initial $\mathbf{p}_u$).
   - **Item Cold-Start & Overcoming Citation Bias:** Newly published papers have zero historical citations. Relying solely on citation strength biases recommendations toward older papers. To surface fresh papers, the ranking engine pairs citation signals with citation-independent **content signals** (SPECTER2 semantic proximity and exact facet alignment), allowing zero-citation preprints to rank high if their methodology matches the user's focus.
3. **Avoiding Echo Chambers (Relevance vs. Diversity):**
   - Ensuring the Top-10 feed doesn't collapse into 10 minor variations of the exact same model (resolved via Maximal Marginal Relevance diversification).
4. **The Evaluation Dilemma & The Ground-Truth Solution:**
   - **The Challenge:** Commercial academic search engines (Google Scholar, Semantic Scholar) keep researcher clickstreams strictly private. OpenAlex provides publication and citation metadata, but zero real-time click logs.
   - **The Two-Track Evaluation Solution:**
     * **Track 1 (Macro-Evaluation on Real Human Behavior via LFO):**
       - Evaluates whether the system can discover and recommend the papers a real researcher ended up citing in their next publication $B$.
       - **Temporal Guard ($\leq T$):** Historical papers published up to year $T=2023$ build the user profile $\mathbf{p}_u$. For target Paper $B$ published in $T+1=2024$, its ground truth references are strictly restricted to papers published $\leq T$ (excluding references published in $T+1$ which could not have been known at time $T$).
       - **Resolution of Discovery vs. Re-Citation:**
         - In a personalized discovery feed, re-recommending papers the author has already cited repeatedly in past publications is an echo-chamber failure.
         - The **headline evaluation objective** is strictly **Discovery Recall@10** and **Discovery NDCG@10**, evaluated on the set of previously uncited references:
           $$\text{Discovery}(B \mid u) = \{ p \in \text{Ref}(B) \mid \text{year}(p) \leq T \text{ and } p \notin \text{Citations}(u, \leq T) \cup \text{Authored}(u, \leq T) \}$$
         - Re-citations ($\text{ReCitation}(B \mid u) = \{ p \in \text{Ref}(B) \mid \text{year}(p) \leq T \text{ and } p \in \text{Citations}(u, \leq T) \}$) are evaluated strictly in a separate **diagnostic ablation** where the past citation filter is temporarily lifted, measuring the citation graph's capacity to retrieve familiar foundation papers.
       - **Leakage-Free Corpus Construction:** 
         - The candidate corpus $\mathcal{C}$ (~100,000 papers) is constructed strictly outward from the evaluation authors' publication and citation neighborhoods **as of $\leq T$**, supplemented with random CS/AI papers published $\leq T$.
         - The crawl **never** traverses the reference list of target paper $B$ ($T+1$), preventing ground-truth stuffing.
         - Candidate retrieval is strictly restricted to papers with $\text{year} \leq T$.
         - Both unconstrained Recall and in-corpus Recall are reported alongside the empirical Corpus Coverage Ceiling.
       - **Author Disambiguation & Cohort Selection:** 
         - Evaluated on $N = 300\text{--}500$ single or first authors with clean publication histories ($\ge 3$ papers prior to $T$; Paper $B$ possessing $\ge 5$ in-corpus references published $\leq T$). This guarantees that $\text{Ref}(B)$ genuinely reflects user $u$'s research trajectory.
         - *Selection Bias Note:* Single/first authorship controls for co-author citation noise at the cost of slight skew toward early-career researchers; this empirical trade-off will be explicitly noted in the experimental results.
     * **Track 2 (Interactive Session Dynamics via Multi-Turn LLM Simulation):**
       - Evaluates real-time session adaptation and session drift response (detailed below).

---

## 5. Formal Problem Definition

> **Given:**
> 1. A user $u$ modeled by two components:
>    - **Long-Term Persona Profile:** $\mathbf{p}_u = (\mathbf{v}_u^{\text{persona}}, \mathcal{F}_u)$, where $\mathbf{v}_u^{\text{persona}} \in \mathbb{R}^d$ is the normalized dense persona embedding (derived from past publications and onboarding seed topics), and $\mathcal{F}_u$ is the set of historical preferred facets (tasks, methods, datasets).
>    - **Dynamic Session State:** $\mathbf{s}_u = (\mathbf{v}_t^{\text{session}}, \mathcal{S}_u^{-})$, where $\mathbf{v}_t^{\text{session}} \in \mathbb{R}^d$ is the position-decayed representation of positive interactions (clicked/liked/saved papers), and $\mathcal{S}_u^{-} = \{d_1^{-}, d_2^{-}, \dots\}$ is the set of skipped/dismissed papers acting as negative feedback.
> 2. A corpus $\mathcal{C}$ of academic papers published $\leq T$, where each paper $p \in \mathcal{C}$ has title/abstract embedding $\mathbf{e}_p \in \mathbb{R}^d$, citation graph edges $\mathcal{G}$, and extracted scientific facets $\mathcal{F}_p$.
> 3. An explicit exclusion set $\mathcal{E}_u = \text{Authored}(u, \le T) \cup \text{Citations}(u, \leq T) \cup \text{SessionConsumed}(u)$, guaranteeing that already written, cited, or read papers are never re-recommended in discovery mode.
>
> **Optimization Objective:**
> For any user state $(\mathbf{p}_u, \mathbf{s}_u)$, select a ranked list $\mathcal{R}_u \subset \mathcal{C} \setminus \mathcal{E}_u$ of size $K$ (e.g., $K=10$) that maximizes the multi-objective utility:
> $$\max_{\mathcal{R}_u, |\mathcal{R}_u|=K} \left[ \sum_{p \in \mathcal{R}_u} \text{Score}(p \mid \mathbf{p}_u, \mathbf{s}_u, \mathcal{G}) + \lambda_{\text{div}} \cdot \text{Diversity}(\mathcal{R}_u) + \lambda_{\text{nov}} \cdot \text{Novelty}(\mathcal{R}_u \mid \mathcal{F}_u) - \lambda_{\text{neg}} \cdot \sum_{p \in \mathcal{R}_u} \text{Penalty}(p, \mathcal{S}_u^{-}) \right]$$
> where:
> - **Intra-List Diversity:**
>   $$\text{Diversity}(\mathcal{R}_u) = \frac{2}{K(K-1)} \sum_{i=1}^K \sum_{j=i+1}^K (1 - \cos(\mathbf{e}_{p_i}, \mathbf{e}_{p_j}))$$
> - **Facet Novelty:**
>   $$\text{Novelty}(\mathcal{R}_u \mid \mathcal{F}_u) = \frac{1}{K} \sum_{p \in \mathcal{R}_u} \frac{|\mathcal{F}_p \setminus \mathcal{F}_u|}{|\mathcal{F}_p| + \epsilon}$$
> - **Negative Feedback Penalty:**
>   $$\text{Penalty}(p, \mathcal{S}_u^{-}) = \begin{cases} \max_{n \in \mathcal{S}_u^{-}} \cos(\mathbf{e}_p, \mathbf{e}_n) & \text{if } \mathcal{S}_u^{-} \neq \emptyset \\ 0 & \text{if } \mathcal{S}_u^{-} = \emptyset \end{cases}$$
> accompanied by an inspectable, faithful justification for every paper $p \in \mathcal{R}_u$.
> *(In our two-stage architecture, this optimization is solved efficiently by first retrieving $M=100$ candidate papers via hybrid dense+sparse retrieval, then executing multi-signal scoring and MMR greedy diversification).*

---

## 6. Implementation & Scientific Evaluation Suite

The core focus of implementation is establishing a clean, reproducible scientific testbed for retrieval, re-ranking, and recommendation quality.

### A. Lightweight Interactive Interface & Telemetry (For Pilot & Auditing)
A streamlined interface (Streamlit or minimal Python web frontend) allowing qualitative inspection, profile verification, and session simulation:
* **Interactive Top-10 Feed:** Displays paper cards with title, year, venue, abstract snippet, and extracted facet pills (🎯 **Task**, ⚙️ **Method**, 📊 **Dataset**, 📈 **Metric**).
* **Explanation Pill:** Displays grounded justification (e.g., *"Recommended because this paper tests a new Method on the Dataset you frequently read about"*).
* **Explicit Feedback Controls:** Like/Save (positive interaction), Skip/Dismiss (negative feedback), and Reset Session Drift button.
* **Discrete Behavioral Logging:** Logs discrete user actions (`click`, `like`, `save`, `skip`) to update $\mathbf{s}_u$ in real time (avoiding noisy dwell-time heuristics).
* **Ad-Hoc Seed Search:** Enables keyword queries (e.g., *"LoRA quantization"*) to simulate a researcher initiating a topic shift.

---

### B. Core Recommendation Engine (Python)
* **Session Vector Dynamic Updater:** Updates $\mathbf{v}_t^{\text{session}} = \alpha \mathbf{v}_{t-1}^{\text{session}} + (1 - \alpha) \mathbf{e}_{\text{clicked}}$ on positive interaction; adds dismissed papers to $\mathcal{S}_u^{-}$.
* **Candidate Generator (First Stage):** Hybrid lexical (BM25s) and dense kNN (SPECTER2) candidate retrieval producing $M=100$ candidate items.
* **Multi-Signal Scorer (Second Stage):** Computes linear combination of session similarity, persona alignment, time-bounded citation graph strength ($\text{BC}$ and co-citation), and facet novelty.
* **MMR Diversification:** Maximal Marginal Relevance greedy selection to optimize intra-list diversity.
* **Deterministic Faithful Explainer:** Identifies the feature with the highest positive score contribution $\Delta \text{Score}$ and maps it to a factual template grounded in extracted abstract spans.

---

### C. Offline Data & Graph Pipeline
1. **OpenAlex Data Fetcher & Reconstructor:** Pulls ~100k CS/AI papers published $\leq T$, decodes inverted abstracts to clean plain text, and filters malformed records.
2. **Time-Bounded Citation Graph ($\le T$):** Computes bibliographic coupling and co-citation weights strictly using citations established $\le T$.
3. **Dense Index:** Pretrained SPECTER2 (`proximity` adapter) document embeddings stored in a FAISS index.
4. **Information Extractor:** GLiNER (`urchade/gliner_medium-v2.1`) extracting typed entities (`task`, `method`, `dataset`, `metric`, `topic`) with verbatim character spans.
5. **Lexical Index:** BM25s inverted index over titles and abstracts.

---

### D. Structured Metadata Storage (SQLite)
Local SQLite database storing structured records:
* `papers`: `paper_id`, `title`, `abstract_text`, `authors`, `year`, `venue`, `citation_count`, `doi`.
* `paper_facets`: `paper_id`, `facet_type`, `facet_value`, `evidence_sentence`, `start_char`, `end_char`.
* `paper_citations`: `source_paper_id`, `target_paper_id`, `publication_year`.
* `users`: `user_id`, `name`, `created_at`.
* `user_profiles`: `user_id`, `persona_vector` (BLOB), `preferred_facets` (JSON).
* `user_sessions`: `session_id`, `user_id`, `session_vector` (BLOB), `last_active_timestamp`.
* `user_interactions`: `interaction_id`, `user_id`, `session_id`, `paper_id`, `action_type` (`click`, `like`, `bookmark`, `skip`), `timestamp`.

---

### E. Scientific Evaluation Suite (Core Contribution)

#### 1. Leave-Future-Out (LFO) Recommendation Evaluation Harness (OpenAlex)
* **Cohort Scale:** $N = 300\text{--}500$ verified single or first authors with $\ge 3$ papers prior to $T=2023$, and target paper $B$ in $T+1=2024$ having $\ge 5$ in-corpus references published $\leq T$.
* **Primary Metrics (Headline):** 
  - **Discovery NDCG@10**, **Discovery Recall@10 / HitRate@10**, and **Discovery MRR**, evaluated strictly against $\text{Discovery}(B \mid u)$ with $\mathcal{E}_u$ active. Both unconstrained Recall and in-corpus Recall are reported alongside the Corpus Coverage Ceiling.
  - **Intra-List Diversity (ILD):** Mean pairwise cosine distance between Top-10 papers.
  - **Facet Novelty:** Fraction of extracted facets in recommended papers not present in author's historical facet set $\mathcal{F}_u$.
* **Diagnostic Ablation (Re-citation Recovery):** Evaluates Re-citation Recall@10 in a separate run where $\text{Citations}(u, \le T)$ is omitted from $\mathcal{E}_u$, testing graph retrieval capacity for known foundation papers.
* **Cold-Start Stratification (Mandatory Scientific Slices):**
  - **User Cold-Start:** Performance stratified by author history size: Cold-start ($1\text{--}2$ historical papers) vs. Warm ($5+$ historical papers).
  - **Item Cold-Start:** Recall stratified by candidate citation count at time $T$: Fresh / zero-citation preprints ($\le 1$ citation) vs. Established papers ($\ge 20$ citations).
* **Core Baselines to Beat:**
  1. *Popularity Baseline:* Most-cited recent papers in the user's primary domain (counts $\leq T$).
  2. *Lexical Baseline:* BM25s retrieval using the user's combined past paper titles/abstracts as a query.
  3. *Dense Centroid Baseline:* Unweighted mean of SPECTER2 embeddings across user's past papers (no persona weights, no decay).
  4. *Citation-Only Baseline:* Pure bibliographic coupling and co-citation graph score without text features.
* **Component Ablations:**
  - *No-MMR:* Disables diversity re-ranking ($\lambda_{\text{div}} = 0$).
  - *No-Graph:* Content-only ranking ($w_{\text{cite}} = 0$).
  - *No-Facets:* Dense-only ranking without GLiNER task/method alignment ($w_{\text{facet}} = 0$).
* **Formal Success Hypotheses:**
  - **$H_1$ (Accuracy):** The proposed hybrid model achieves statistically significant improvements in Discovery NDCG@10 and Discovery Recall@10 over all four baselines using paired bootstrap testing with Benjamini-Hochberg (FDR) multiple-testing correction at $\alpha = 0.05$. Effect sizes (Cohen's $d$) are reported.
  - **$H_2$ (Diversity):** MMR diversification significantly increases ILD with $\le 3\%$ relative degradation in Discovery NDCG@10.

#### 2. LLM-Based User Simulation Harness (Track 2: Interactive Session Dynamics)
* **Role:** Behavioral sanity check testing algorithm adaptation under sudden intent shifts.
* **Simulated Cohort:** 30 LLM researcher personas instantiated from real OpenAlex author profiles, executed using an independent LLM (e.g., Claude 3.5 Haiku / GPT-4o-mini) completely distinct from any model in the recommendation pipeline. Evaluated across 5 random seeds with 95% bootstrap confidence intervals.
* **Independent Topic Labeling:** Topic membership (Topic A vs. Topic B) is defined using external **OpenAlex primary Concept/Topic IDs**, avoiding circular evaluation with internal GLiNER or SPECTER2 features.
* **Reachability Protocol:** The shift to Topic B is initiated by the simulated user executing an ad-hoc search query for Topic B, exposing initial Topic B candidates in the feed.
* **Metrics:**
  - **Pivot Latency ($k_{\text{pivot}}$):** Number of interaction steps following the query until Topic B papers represent $\ge 50\%$ of the Top-10 feed.
  - **Long-Term Focus Retention Rate ($\text{Retention}@k$):** Fraction of Top-10 papers belonging to core Topic A after a session reset, or $k$ steps after the user re-engages with a Topic A paper.
* **Track 2 Ablation Arms:**
  1. *Pure Session-Only Decay* (fastest drift, zero retention).
  2. *Pure Persona-Only Static* (zero drift adaptation, perfect retention).
  3. *Proposed Hybrid Recommender* (fast drift with robust retention).
* **Success Hypothesis $H_3$:** The Hybrid Recommender achieves significantly lower Pivot Latency ($k_{\text{pivot}}$) than Persona-Only, while maintaining $\ge 80\%$ Long-Term Focus Retention ($\text{Retention}@1$) compared to Session-Only (which drops to $\approx 0\%$).

#### 3. First-Stage Retrieval Benchmark (CSFCube & LitSearch)
* Evaluates candidate generation Recall@100 across BM25s, SPECTER2, and Hybrid RRF.

#### 4. Extraction Benchmark (SciERC Gold Standard)
* Evaluates GLiNER on SciERC across Strict and Relaxed Span $F_1$, mapping `task` $\leftrightarrow$ `Task`, `method` $\leftrightarrow$ `Method`, `metric` $\leftrightarrow$ `Metric`, `dataset` $\leftrightarrow$ `Material`, `topic` $\leftrightarrow$ `OtherScientificTerm`.

#### 5. Profile & Explanation Faithfulness Audit
* CLI script sampling 100 recommendation explanations:
  - **Extractive Grounding Rate:** Percentage of displayed facet spans and evidence sentences existing verbatim in the abstract (Target: 100%).
  - **Ranking Faithfulness:** Verifies that the displayed feature was the maximum positive contributor $\Delta \text{Score}$ in the ranking formula (Target: 100%).
  - **Explanation Coverage:** Percentage of Top-10 recommendations successfully annotated with grounded explanations (Target: $\ge 90\%$).
