# Paper Feed — Execution Plan

> **Purpose:** The step-by-step plan for building and evaluating Paper Feed.
> **Implements:** `prob.md` (what must be achieved and how it is evaluated) and `design.md` (how the system is built). Section references such as "D§9" mean `design.md` §9, and "P§6E" means `prob.md` §6E. If this plan and those documents disagree, the documents win and this plan is corrected.
> **Not covered here:** schedule and team assignment (deliberately left out).

---

## How to read this plan

- The work is split into **phases**. Each phase lists its **tasks** with an ID (for example `1.4`), what to build, the **outputs** it produces, and a **done when** check.
- Phases 0–8 are **Core** (P§6A). Phases 9–11 are **Extended**. Phase 12 closes the project.
- Order matters mainly for data dependencies: data (1) → indexes (2) → facets (3) → user model (4) → recommender (5) → evaluation (6–8). Phases 7 and 3.4 can run in parallel with others once Phase 2 exists.
- Every number that has a default in D§2 (for example $T=2023$, $K=10$, $M=100$, $k_{\text{rrf}}=60$) lives in **one config file**, never hard-coded.

### Phase dependency overview

```text
0 Setup
└─► 1 Data & ground truth ──► 2 Indexes ──► 3 Facets ──► 4 User model ──► 5 Recommender
                                   │                                          │
                                   ├─► 7 Retrieval benchmark                   ├─► 6 Track 1 evaluation ──► 8 Faithfulness audit
                                   └─► (3.4 SciERC, any time after 3.1)        ├─► 9 Track 2 simulation (Extended)
                                                                              ├─► 10 SOTA variants V1–V3 (Extended)
                                                                              └─► 11 UI + user pilot (Extended)
                                                                                         └─► 12 Reproducibility & report
```

---

## Phase 0 — Project Setup

**Goal:** a reproducible repository skeleton that every later phase plugs into (D§15.3).

### 0.1 Repository layout
Create this structure inside the repository root:
```text
configs/                 default.yaml (+ one file per experiment that overrides it)
src/paperfeed/
  openalex/              client.py, crawl.py, abstracts.py
  data/                  schema.sql, db.py, graph.py, cohort.py, ground_truth.py, leakage.py
  index/                 embed.py, faiss_index.py, bm25_index.py, retrieval.py
  facets/                extract.py, canonicalize.py, audit.py
  user/                  persona.py, session.py, exclusion.py, pseudo_query.py
  rec/                   candidates.py, features.py, scorer.py, mmr.py, explain.py, recommender.py
  eval/                  metrics.py, baselines.py, lfo.py, tuning.py, stats.py, slices.py, faithfulness.py, report.py
  bench/                 retrieval_bench.py, scierc.py
  sim/                   personas.py, llm_client.py, simulator.py, episodes.py, sim_metrics.py
  variants/              encoders.py, cross_encoder.py, ltr.py
  ui/                    app.py
scripts/                 numbered entry points: 01_crawl.py, 02_build_db.py, ... (one per pipeline step)
tests/                   unit and integration tests (pytest)
data/                    raw/, interim/, processed/, artifacts/   (git-ignored)
results/                 runs/<run_id>/ (metrics, tables, figures, manifests)
```

### 0.2 Environment
- Python virtual environment with **pinned** versions in a lock file (`requirements.lock` or equivalent).
- Main dependencies: `httpx`/`requests` (OpenAlex), `numpy`, `pandas`, `pyarrow`, `scipy`, `torch`, `transformers`, `adapters` (SPECTER2 adapters), `faiss` (CPU build is enough for exact search), `bm25s`, `gliner`, `sentence-transformers` (facet canonicalization), `scikit-learn` (clustering), `pyyaml`, `tqdm`, `pytest`.
- Extended only: `lightgbm`, `shap`, `streamlit`, the hosted-LLM client library.
- Record the GPU model, CUDA version and driver in the run manifest.

### 0.3 Configuration
`configs/default.yaml` holds every parameter from D§2: $T$, $K$, $M$, $k_{\text{rrf}}$, $\gamma_{\text{rec}}$, $\alpha_{\text{sess}}$, $\rho$, $\gamma_{\text{pivot}}$, `use_gate`, weight defaults, tuning grids, model names and versions, paths, random seeds.

### 0.4 Run manifest and logging
Every script writes `results/runs/<run_id>/manifest.json` containing: git commit, config hash and full config, seeds, model names and versions, corpus snapshot ID, index configuration, start/end timestamps, hardware. Use one logger with timestamps.

### 0.5 Caching utility
A disk cache keyed by `(item id, model name/version, config hash)` (D§15.3), used for OpenAlex responses, embeddings, GLiNER outputs and simulator LLM responses.

### 0.6 Test scaffold
`pytest` set up with small fixture data (a handful of fake works) so that later unit tests run without network access.

**Done when:** a fresh clone can create the environment from the lock file, run `pytest` (passing on the fixtures), and run an empty script that writes a valid manifest.

---

## Phase 1 — Data Acquisition, Corpus and Ground Truth

**Goal:** a frozen, leakage-free corpus of about 120,000 CS/AI/IR papers published up to $T$ (2024), a time-bounded citation graph, and the evaluation cohort with ground truth (D§4, P§4).

### 1.1 OpenAlex client
- Base: `https://api.openalex.org`, endpoint `/works` (and `/authors` where useful).
- Use the polite pool (`mailto=` parameter). Check the current OpenAlex documentation for rate limits and authentication (an API key may be required) and respect them.
- Cursor pagination (`cursor=*`, `per-page=200`), `select=` to fetch only needed fields, retries with exponential backoff, and resumable crawling (store the last cursor).
- Save every raw response to `data/raw/` as compressed JSON lines, so the crawl never has to be repeated.
- *Alternative:* if network or rate limits are a problem, the OpenAlex data snapshot can replace the API. The rest of the plan is unchanged.

**Fields to request per work:** `id`, `doi`, `title`, `publication_year`, `publication_date`, `type`, `language`, `authorships` (author ID, `author_position`), `primary_location` (venue), `referenced_works`, `counts_by_year`, `abstract_inverted_index`, `primary_topic` (with its subfield, field and domain), `topics`.

### 1.2 Candidate evaluation authors (D§4.1 step 1, D§13.1)
1. Query works with `publication_year = T+1` (2025) whose primary topic is in Computer Science.
2. For each such work, take the **first author** (`author_position = "first"`), or the only author of a single-author paper. One seed author per paper $B$.
3. If an author has several qualifying 2025 papers, choose one $B$ by a fixed rule (for example the earliest `publication_date`, tie-broken by lowest work ID) and record the rule.
4. Keep authors with at least one paper published $\le T$ (2024) (cold-start slices need 1+ papers; the zero-publication slice is a re-run, Phase 6).
5. Filter implausible author records (for example, too many papers per year, or papers spread over unrelated fields). Record the thresholds.
6. Over-sample to about **1,000** candidate authors.

**Output:** `candidate_authors.jsonl` (author ID, chosen paper $B$, rule used).

### 1.3 Author histories
For each candidate author, fetch all works with `publication_year` $\le T$ (2024) (`Authored(u, ≤T)`) and their `referenced_works` (gives $\text{Citations}(u,\le T)$).

### 1.4 Corpus expansion (D§4.1 steps 2–3)
- **Hop 1:** the references of every authored paper, plus papers published $\le T$ that cite an authored paper (the inbound citations that co-citation CC needs). Budget: up to 5 seed papers per author (most-cited first, with 1–300 citations; mega-papers time out), up to 40 citing papers per seed, processed rank by rank so every author gets their best seed before anyone gets a second, with a global cap of 40,000 newly added papers.
- **Hop 2:** references of hop-1 papers, with a per-author cap (150) so the corpus stays near 120k.
- **Foundational CS/AI/IR Anchors:** classic papers published $\le T$ in CS, AI, ML, and IR (e.g. ~15k papers) ensuring high graph connectivity, fetched sorted by citations. Selecting them by present-day citation counts nudges corpus membership towards papers that became classics; this is a corpus-construction choice, shared by every system and baseline, and is stated as a limitation.
- **Fill:** random and topic-stratified CS/AI/IR papers published $\le T$ (OpenAlex `sample` with a fixed `seed`), up to about 120,000 papers in total.
- **Rules:** never add a paper with year $> T$ (2024); never use paper $B$ (2025) (or any work with year $> T$) as a traversal source. $\text{Ref}(B)$ is fetched only for ground truth and coverage (1.9).
- Keep `type` in {article, preprint, proceedings paper, …} and English language for ordinary papers; record what was dropped. **Exception:** the evaluation authors' own papers (everything in `Authored(u, ≤T)`) are always ingested, whatever their type or language, because their reference lists feed $\text{Citations}(u,\le T)$ and BC/CC. Authored papers that fail the filter are stored as graph nodes only (`in_candidates = false`); they are never recommendable anyway because they are in the exclusion set.
- Papers outside CS/AI (the crawl pulls in many, for example medicine and engineering) are **kept**. They are legitimate references and act as easy negatives; nothing filters or re-weights the corpus by field.

### 1.5 Abstract reconstruction (D§4.2)
- Rebuild text from `abstract_inverted_index` by placing each word at each of its positions and joining in order; normalize whitespace.
- Papers with no abstract: keep them as **graph nodes** (their references count for BC/CC), but mark them `in_candidates = false` (no embedding, no facets).
- An author's own paper without an abstract contributes a **title-only** embedding to the persona.
- Report the fraction of papers without abstracts.

### 1.6 SQLite database (D§5)
Create `schema.sql` with these tables and load the data:
- `papers` (`paper_id`, `title`, `abstract`, `authors`, `year`, `publication_date`, `venue`, `doi`, `url`, `citation_count_T`, `primary_topic_id`, `subfield_id`, `field_id`, `in_candidates`). If `primary_topic` is missing, use the highest-scoring entry in `topics`, else `UNKNOWN`; report the `UNKNOWN` share in the data report (1.11).
- `paper_citations` (`source_paper_id`, `target_paper_id`, `source_year`), including references to papers outside the corpus (needed for BC)
- `paper_facets`, `facet_canon` (filled in Phase 3)
- `users`, `user_profiles`, `user_sessions`, `user_interactions` (used by Track 2 and the UI)

`citation_count_T` = sum of `counts_by_year` entries with year $\le T$. OpenAlex only provides `counts_by_year` from 2012, so this is the citation count in 2012–$T$ and is complete only for papers published since 2012; the item cold-start slice and the Popularity baseline use that window (D§12). **Never** store the present-day `cited_by_count` or the cited-by list as a feature (D§4.3).

### 1.7 Time-bounded citation graph (D§4.4)
- Keep only edges whose **citing** paper has year $\le T$.
- **Bibliographic coupling:** $\text{BC}(p\mid u) = |\text{Refs}(p) \cap \text{Refs}(\text{Authored}(u,\le T))|$, computed from reference lists (references outside the corpus count too).
- **Co-citation:** $\text{CC}(p\mid u)$ = number of third-party papers (year $\le T$) that cite both $p$ and a paper authored by $u$. Fetch the works citing each authored paper with a year $\le T$ filter (`cites:` filter), then count their other references.
- Store the per-user BC/CC as a sparse table or compute on demand with caching.

### 1.8 Ground truth (P§4)
For each candidate author:
- $\text{Ref}_{\le T}(B)$ = references of $B$ with year $\le T$.
- $\text{Discovery}(B\mid u) = \text{Ref}_{\le T}(B) \setminus (\text{Citations}(u,\le T) \cup \text{Authored}(u,\le T))$.
- $\text{ReCitation}(B\mid u) = \text{Ref}_{\le T}(B) \cap \text{Citations}(u,\le T) \setminus \text{Authored}(u,\le T)$ (self-citations removed).
- **Zero-publication slice ground truth:** $\text{Ref}_{\le T}(B) \setminus \text{Authored}(u,\le T)$ (P§6E.1).
- Mark which targets are in the corpus **and** have an abstract (in-corpus targets).

### 1.9 Cohort selection and split (D§13.1)
- **Validate target papers first.** Drop an author if their $B$ is not a genuine $T+1$ paper: OpenAlex sometimes dates an old work (a reissue, a merged preprint/journal record) to $T+1$. Two tests, either one drops the author: (a) a paper published $\le T$ in the corpus cites $B$; (b) OpenAlex `counts_by_year` shows citations to $B$ in a year $\le T$. In the first build this removed 294 of 1,000 candidate authors. The API result is cached, so reruns are offline (`--offline`).
- Keep authors with $|\text{Discovery}(B\mid u) \cap \mathcal{C}| \ge 5$.
- Select up to $N=400$ (record the realized $N$). Moderate and sparse authors are rare, so every qualifying one is kept first and the rest of the places are filled with a seeded random draw of warm authors; this tilts the cohort towards less-published authors, which the report states.
- Split into **tuning** (20%, about 80) and **test** (80%, about 320) with a fixed seed, stratified by history-size slice (Warm ≥ 5 papers, Moderate 3–4, Sparse 1–2).
- Seed topics for the zero-publication slice: the 3 most frequent primary OpenAlex Topics of the author's papers $\le T$ (their display names).

**Output:** `cohort.jsonl`, `ground_truth.jsonl`, `split.json` (written to `data/processed/` by `scripts/build_cohort.py`) — frozen and versioned. $B$ is never counted among its own references (OpenAlex occasionally lists a work among its own references).

### 1.10 Leakage checker (D§4.4)
An automated check, run in tests and before every evaluation, asserting that:
- no candidate paper has year $> T$;
- no citation edge has `source_year` $> T$;
- no paper $B$ is in the corpus or used as a crawl source, and no validated cohort $B$ is cited by any paper in the corpus (a $\le T$ paper citing $B$ means $B$ is not a future paper);
- `citation_count_T` uses only years $\le T$;
- for every evaluation user, the persona, facets, BC/CC and pseudo-query use only papers with year $\le T$.

### 1.11 Data report
Corpus size; abstract coverage; type and topic distribution; the **coverage ceiling** (fraction of Discovery references present in the corpus with an abstract) before and after cohort filtering; realized $N$; slice sizes.

**Done when:** the database and cohort files are frozen, the leakage checker passes, and the data report exists.

---

## Phase 2 — Representations and Indexes

**Goal:** dense and lexical indexes plus shared retrieval utilities (D§5, D§6.1, D§8).

### 2.1 SPECTER2 paper embeddings
- Load `allenai/specter2_base` with the **proximity** adapter (`allenai/specter2`).
- Input: `title + [SEP] + abstract`, max 512 tokens, take the [CLS] embedding, L2-normalize, store as float32 with an ID map.
- Batch with fp16 on the GPU; cache results.
- **Record the embedding time** per 1,000 papers in the manifest (D§15.1).

### 2.2 Query encoder
Load the `allenai/specter2_adhoc_query` adapter for short text (search queries, onboarding seed topics). Same output space as 2.1.

### 2.3 Dense index
FAISS `IndexFlatIP` over the normalized vectors. Filtering (year $\le T$, exclusion set, `in_candidates`) is done by retrieving a larger list (for example 1,000) and filtering, or with an ID selector.

### 2.4 Lexical index
bm25s over title + abstract: lowercase, stop-word removal, and the same tokenizer for indexing and querying. Save the index.

### 2.5 Retrieval utilities
`dense_search(vector, k, filters)`, `bm25_search(text, k, filters)`, `rrf(ranked_lists, k=60)`. Unit-test RRF on a small hand-computed example.

**Done when:** a query returns sensible neighbours for a few known papers, and the timing figures are recorded.

---

## Phase 3 — Facet Extraction and Canonicalization

**Goal:** typed, grounded, matchable facets for every candidate paper (D§6.2–6.4).

### 3.1 GLiNER extraction
- Model `urchade/gliner_medium-v2.1`, labels `task`, `method`, `dataset`, `metric`, `topic`.
- Split abstracts into sentences with the built-in rule-based splitter (`split_sentences`: no optional dependency, so every machine produces identical sentences and offsets) and feed sentence chunks that fit GLiNER's input length.
- For each entity store: `paper_id`, `facet_type`, `facet_value` (verbatim span), `start_char`, `end_char` (offsets in the full abstract), `confidence`, and `evidence_sentence`.
- Confidence threshold: start at 0.5; adjust using the audit in 3.3 (and SciERC in 3.4 if available).
- Cache outputs; **record the extraction time** per 1,000 abstracts.

### 3.2 Canonicalization (D§6.3)
1. Lowercase and lemmatize each span.
2. Embed spans with a small sentence-embedding model.
3. Cluster spans **per facet type** with greedy leader clustering over exact nearest-neighbour lists (most frequent span first; a span joins a leader within the cosine-distance threshold; no full distance matrix, so it scales to the whole corpus) and assign a `canonical_id`; the canonical name is the most frequent surface form in the cluster.
4. Choose the threshold by inspecting samples of clusters (too loose merges unrelated methods; too tight never matches).
5. Store `canonical_id` in `paper_facets` and the cluster table in `facet_canon`.

### 3.3 In-corpus profile audit (Core, D§13.5)
- Stratified sample of 100 papers; export an annotation sheet (paper, abstract, extracted facets).
- Manually mark each facet as correct or not → **Precision@facet** per type.
- Automatic checks: **grounding rate** (span and evidence sentence exist verbatim at the stored offsets; target 100%), **facet density** (share of papers with ≥ 3 core facets), **canonical match rate** (share of papers sharing at least one canonical task/method/dataset with another paper).

### 3.4 SciERC benchmark (Extended, D§13.4)
- Load the official SciERC release (Luan et al., 2018) test split.
- Map labels: `task→Task`, `method→Method`, `metric→Metric`, `dataset→Material`, `topic→OtherScientificTerm`.
- Report strict and relaxed span F1 per label, macro and micro.
- Optional LLM baseline: at most 100 abstracts, a model of at most 4B parameters in 4-bit on the 4 GB GPU, reported as indicative only.

**Done when:** facets exist for all candidate papers, canonical IDs are assigned, and the audit numbers are recorded.

---

## Phase 4 — User Model

**Goal:** persona, facet set, exclusion set and session state for any user (D§7).

### 4.1 Persona vector
$\mathbf{v}^{\text{pers}}_u$ = L2-normalized, recency-weighted sum of the user's authored-paper embeddings, with weights $\gamma_{\text{rec}}^{\,T-\text{year}(p)}$ (default 0.9). Title-only embeddings for authored papers without abstracts.
**Zero-publication users:** unweighted mean of the `adhoc_query` embeddings of the 3 seed topics.

### 4.2 User facet set $\mathcal{F}_u$
Canonical task, method and dataset facets of the authored papers. Zero-publication users: facets of the 20 corpus papers nearest to the persona vector.

### 4.3 Exclusion set $\mathcal{E}_u$
$\text{Authored}(u,\le T) \cup \text{Citations}(u,\le T) \cup \text{SessionConsumed}(u)$. Two switches: **diagnostic mode** (drop $\text{Citations}(u,\le T)$, for the re-citation run) and **zero-publication mode** ($\mathcal{E}_u = \emptyset$).

### 4.4 Session state
A class holding $\mathbf{v}^{\text{sess}}_t$, $\mathcal{S}^-_u$ and `SessionConsumed`:
- `click`/`save`: $\mathbf{v}^{\text{sess}}_t = \alpha_{\text{sess}}\mathbf{v}^{\text{sess}}_{t-1} + (1-\alpha_{\text{sess}})\mathbf{e}_{\text{clicked}}$; `save` also adds to `SessionConsumed`.
- `skip`: add to $\mathcal{S}^-_u$.
- Optional gate (`use_gate`): if $1-\cos(\mathbf{e}_{\text{clicked}}, \mathbf{v}^{\text{sess}}_{t-1}) > \gamma_{\text{pivot}}$, reset $\mathbf{v}^{\text{sess}}_t = \mathbf{e}_{\text{clicked}}$.
- `reset()` and inactivity timeout (UI only): clear the session.
In Track 1 the session stays empty.

### 4.5 Pseudo-query builder (D§8.1)
Most frequent canonical task/method/dataset terms of $\mathcal{F}_u$ plus titles of the last 3 session papers (or, with no session, the user's 3 most recent authored papers). The query length is a parameter (tuned for the Hybrid-Persona baseline).

**Done when:** unit tests cover the EMA update, the gate, the exclusion modes and the zero-publication path.

---

## Phase 5 — Recommendation Engine

**Goal:** `recommend(user, mode) → Top-10 with scores, feature contributions and explanations` (D§8–§11).

### 5.1 Candidate generation
- **Feed mode:** dense kNN on $\mathbf{q}_t = \text{normalize}((1-\rho)\mathbf{v}^{\text{sess}}_t + \rho\,\mathbf{v}^{\text{pers}}_u)$ (just the persona when the session is empty) + BM25s on the pseudo-query; apply year $\le T$, `in_candidates` and $\mathcal{E}_u$ filters; take the top $M_{\text{channel}}=200$ per channel after filtering; fuse by RRF ($k=60$); keep $M=100$.
- **Search mode:** `adhoc_query` embedding + BM25s on the query text, RRF with the same depths; optional cross-encoder rerank (Extended).
- Return the candidate list with each channel's rank (needed by V3 and for Recall@100).

### 5.2 Features (D§9)
For each candidate: $\cos(\mathbf{e}_p,\mathbf{v}^{\text{pers}}_u)$; $\cos(\mathbf{e}_p,\mathbf{v}^{\text{sess}}_t)$ (dropped when the session is empty); $\log(1+\text{BC}+\text{CC})$; $S_{\text{facet}}$ (frequency-weighted coverage of the candidate's canonical task and dataset facets, formula in D§9); new-method indicator ($\exists$ shared task and $\exists$ method of $p$ not in $\mathcal{F}_u$); Penalty ($\max$ cosine to $\mathcal{S}^-_u$, 0 if empty).

### 5.3 Scaling and score
Min-max scale continuous features to $[0,1]$ within the pool (a constant feature → 0); binary indicators stay 0/1; penalty unscaled. Linear score with $w_{\text{pers}}=1$ and the configured weights. Keep the per-feature contributions $w_k\,\text{mm}_k$.

### 5.4 MMR
Greedy: pick $\arg\max\,[\text{rel}(p) - \lambda_{\text{div}}\max_{q\in\mathcal{R}}\cos(\mathbf{e}_p,\mathbf{e}_q)]$ until $K=10$. Deterministic tie-break by `paper_id`.

### 5.5 Explainer (D§11)
- Pick the feature with the largest positive contribution among {facet match, new method, citation, session, persona}.
- Fill its template with grounded content: verbatim facet spans and evidence sentences; for citation, the shared-reference count (BC) or the co-cited authored paper (CC), whichever contributes more; for session/persona, the most similar session paper or the profile.
- Verify grounding before display (span exists at its offsets; citation edge exists with year $\le T$). If it cannot be grounded, show no explanation and count it as not covered.

### 5.6 Recommender interface
One deterministic function used by every evaluation, the simulator and the UI. Switches for every ablation (each weight, $\lambda_{\text{div}}$, `use_gate`, exclusion modes).

### 5.7 Tests
Small fixture corpus: filters remove excluded and future papers; scaling handles constant features; MMR returns $K$ unique papers; the explainer's choice equals the largest contribution; grounding failures are reported.

**Done when:** the recommender runs end to end for a sample of cohort users and every output paper passes the leakage checker.

---

## Phase 6 — Track 1: Leave-Future-Out Evaluation (Core)

**Goal:** all Track 1 numbers, tests and tables (D§13.1, P§6E.1).

### 6.1 Metrics (`eval/metrics.py`)
- **Headline:** Discovery NDCG@10 (binary relevance; ideal DCG uses $\min(10, |\text{targets}|)$), Recall@10, HitRate@10, MRR — computed against full references and against in-corpus targets.
- **Anchored versions:** each metric as a percentage of **Oracle@10**.
- **Candidate ceiling:** Discovery Recall@100 of the first stage.
- **Diversity/novelty:** Topic Entropy over primary OpenAlex Topics (raw and divided by $\log_2 10$); Topic Discovery Rate; embedding ILD; Facet Novelty ratio. Missing topics use the 1.6 fallback; `UNKNOWN` is one category in entropy and never "new".
- **Edge cases:** users with no hits score 0 (not dropped); deterministic tie-breaking; unit tests on hand-computed lists.

### 6.2 Baselines and anchors (all with the same year filter and $\mathcal{E}_u$)
1. Popularity: highest `citation_count_T` among papers published since 2012 (the `counts_by_year` window) in the user's primary domain (modal OpenAlex subfield of their papers $\le T$).
2. Lexical: BM25s with the concatenated titles and abstracts of the user's past papers as the query.
3. Dense centroid: unweighted mean SPECTER2 embedding of past papers.
4. Citation-only: rank by $\log(1+\text{BC}+\text{CC})$.
5. Hybrid-Persona: persona vector + pseudo-query, RRF, no citation/facets/MMR (tuned in 6.3).
6. Optional: recent papers by co-authors.
- **Anchors:** Random (mean over 100 draws of 10 eligible papers) and Oracle@10 (in-corpus targets ranked first). No significance tests against anchors.

### 6.3 Tuning (tuning split only)
- Cache each user's candidate pool and raw features once; tuning then only rescores.
- Grid: $w_{\text{cite}}, w_{\text{facet}}, w_{\text{nov}} \in \{0, 0.25, 0.5, 1\}$, $\lambda_{\text{div}} \in \{0, 0.1, 0.2, 0.3, 0.5\}$. Objective: mean Discovery NDCG@10.
- Hybrid-Persona baseline: tune $\gamma_{\text{rec}}$ and pseudo-query length with the same budget.
- Freeze the chosen values in a config file before touching the test split.

### 6.4 Test runs (test split, once)
- Full system and baselines 1–6, anchors.
- Ablations: No-MMR ($\lambda_{\text{div}}=0$), No-Graph ($w_{\text{cite}}=0$), No-Facets ($w_{\text{facet}}=w_{\text{nov}}=0$).
- Diagnostic re-citation run ($\text{Citations}(u,\le T)$ removed from $\mathcal{E}_u$; ReCitation Recall@10; "re-recommend own past references" reported here only).
- Slices: Warm / Moderate / Sparse; Zero-publication re-run (seed-only profile, $\mathcal{E}_u=\emptyset$, its own ground truth; Lexical and Dense baselines use the seed text, Citation-only not applicable); Item cold start (recall over targets with `citation_count_T` $\le 1$ vs $\ge 20$). Report $n$ for every slice.

### 6.5 Statistics (`eval/stats.py`)
- Paired bootstrap over users (for example 10,000 resamples) for every system-vs-baseline difference; 95% CIs; Benjamini–Hochberg correction across the $H_1$ family; paired Cohen's $d$.
- **$H_1$:** system vs all five baselines on Discovery NDCG@10 and Recall@10.
- **$H_2$:** one-sided test that MMR raises Topic Entropy; non-inferiority if the upper end of the 95% CI of the relative NDCG@10 loss is $\le 3\%$.
- Minimum detectable effect at the realized $N_{\text{test}}$ (from the bootstrap standard error).

### 6.6 Tables and figures
Main results table (with Oracle-normalized columns), ablation table, slice table, diagnostic table, coverage ceiling and Recall@100 figure. Generated by `eval/report.py` from saved run outputs, never by hand.

**Done when:** all tables are generated from one test run with a manifest, and $H_1$/$H_2$ are answered with CIs.

---

## Phase 7 — First-Stage Retrieval Benchmark (Core)

**Goal:** show how BM25s, SPECTER2 and RRF compare as candidate generators (D§13.3).

### 7.1 Datasets
- **CSFCube:** paper-to-paper, facet-conditioned queries (proximity adapter).
- **LitSearch:** natural-language queries to papers (`adhoc_query` adapter).
- Optional: **SCIDOCS**.
Each benchmark has its own corpus, so build separate indexes with the Phase 2 code.

### 7.2 Runs and metrics
BM25s, SPECTER2, RRF hybrid ($k=60$, top 200 per channel); optional cross-encoder rerank rows. Metrics: Recall@100 (primary), nDCG@10, nDCG@100, MRR.

**Done when:** one table per benchmark is generated by script.

---

## Phase 8 — Explanation Faithfulness Audit (Core)

**Goal:** evidence that explanations are grounded and tied to the ranking (D§13.6, P§6E.5).

- Sample 100 explanations from the Track 1 test run (stratified over users and templates).
- **Extractive Grounding Rate:** displayed spans and evidence sentences exist verbatim at their offsets (target 100%).
- **Counterfactual necessity:** set the explained feature to 0, re-score and re-run MMR, and record whether the paper leaves the Top-10 or drops $\ge 3$ ranks (drop-out rate).
- **Explanation Coverage:** share of Top-10 items with a grounded explanation (target $\ge 90\%$).
- Report the distribution of explanation templates.

**Done when:** the four numbers and the template distribution are in the results.

---

## Phase 9 — Track 2: LLM Session Simulation (Extended)

**Goal:** session adaptation results with clearly simulated status (D§13.2, P§6E.2).

### 9.1 Personas
- 30 evaluation personas and about 10 disjoint pilot personas, built from real OpenAlex authors (their papers $\le T$ define the persona and goal text).
- Topic A = the persona's subfield; Topic B = a different CS/AI subfield with at least ~500 corpus papers; a short search query for Topic B.

### 9.2 Simulator
- One free-tier hosted LLM at a fixed checkpoint; record its exact name, version and run dates.
- Prompt: the persona's current goal plus about 10 papers (title + abstract); the response is a 1–5 rating per paper in JSON; parse strictly and retry on malformed output.
- Policy: rating ≥ 4 → `click`, 5 → `save`, ≤ 2 → `skip`.
- Sampling temperature above 0 so the 5 seeds differ (the recommender itself is deterministic).
- Cache every response (persona ID, turn, prompt hash, model); respect rate limits and resume interrupted runs.

### 9.3 Episode protocol
1. Turn 0: feed built from the persona (all arms).
2. Topic A turns (fixed number).
3. Shift: the agent runs the Topic B search query; its clicks on results update the session.
4. Topic B turns until the maximum number of turns; record pivot latency (censored at the maximum).
5. Re-engagement: the goal switches back to Topic A; the agent runs a fixed Topic A search query and clicks results it rates ≥ 4 (if none, the highest-rated Topic A result). Record Retention@$k$ for $k = 1..3$ from that click.
Topic membership comes only from OpenAlex Topic labels (subfield level).

### 9.4 Arms
Persona-only, Session-only, Static blend (50/50, un-decayed session mean), Hybrid, Hybrid + gate, with the per-arm settings of D§13.2. Persona-only has no session input anywhere ($\rho=1$, $w_{\text{sess}}=0$, no session titles in the pseudo-query, no skip penalty). Same citation/facet signals and MMR across arms.

### 9.5 Pilot tuning
On the pilot personas only: $\alpha_{\text{sess}}$, $\rho$, $w_{\text{sess}}$, $\lambda_{\text{neg}}$, $\gamma_{\text{pivot}}$, using a pre-stated criterion that combines pivot latency and retention. Track 1 weights are reused. Freeze before the evaluation run.

### 9.6 Evaluation and $H_3$
Run 30 personas × 5 seeds × 5 arms. Paired bootstrap over persona × seed units; $H_3$ supported if the 95% CI of each difference excludes 0. Report absolute retention as descriptive.

**Done when:** the Track 2 table and the released response cache exist.

---

## Phase 10 — SOTA Variant Experiments (Extended)

**Goal:** test whether stronger components help, without touching the test split for selection (D§13.8, D§9.1).

### 10.1 V1 — Alternative encoders
- Build a fixed 20k-paper subset containing the tuning authors' neighbourhoods.
- Embed it with SPECTER2 and with alternatives (SciNCL; a general embedding model such as BGE, E5 or a small Qwen-embedding model), each with its recommended query format.
- Compare on CSFCube/LitSearch and on Track 1 tuning authors (Dense Centroid and the full system).
- If one wins on tuning data, re-embed the full corpus with it and re-run the pipeline.

### 10.2 V2 — Cross-encoder in feed mode
Profile query text (recent authored titles + top canonical facet terms); rerank the 100 candidates with `bge-reranker-base` and `ms-marco-MiniLM-L-6-v2`; add the reranker score as a feature; tune its weight on the tuning split; record latency per feed.

### 10.3 V3 — Learned ranker
LightGBM LambdaMART over the min-max-scaled features plus channel ranks; labels = in-corpus Discovery references; 5-fold cross-validation inside the tuning split; shallow trees and strong regularization; compare with the linear scorer on the same folds. SHAP contributions for the explainer; if they are unreliable, report V3 as accuracy-only.

### 10.4 Final variant table
Freeze all choices, then run every variant once on the test split and report all of them (including losers) with BH-corrected comparisons.

**Done when:** the variant table exists and the frozen choice is documented.

---

## Phase 11 — Demo UI and User Pilot (Extended)

### 11.1 Streamlit UI
Onboarding (seed topics or an OpenAlex author ID), feed with paper cards (title, year, venue, abstract snippet, facet pills, explanation), `click` / `save` / `skip` buttons, search bar, library page, interest inspector (top canonical facets), reset-session button. It calls the same recommender interface as the evaluation.

### 11.2 User pilot (D§14)
n ≈ 10–15 participants; consent form; scripted task (view a Top-10 feed from their OpenAlex profile or seed topics; rate relevance and explanation clarity 1–5); store only ratings; report as qualitative support.

---

## Phase 12 — Reproducibility Package and Final Report

### 12.1 Reproducibility package
Locked environment, configs, run manifests, frozen cohort/ground-truth/split files, leakage checker, scripts that regenerate every table from saved outputs, Track 2 response cache.

### 12.2 Final report contents
1. Problem and formal definition (from `prob.md`).
2. System design (from `design.md`), with the notation table.
3. Data: corpus construction, coverage ceiling, cohort and slices.
4. Results: retrieval benchmark, Track 1 (main table with anchors, ablations, slices, diagnostic, $H_1$, $H_2$), extraction audit (and SciERC), faithfulness audit, Track 2 and $H_3$, SOTA variants, pilot.
5. Threats to validity (D§13.7) and **known limitations** (P§7, D§13.7a).

**Done when:** every number in the report can be regenerated from the package.

---

## Project-level definition of done

- Core: Phases 0–8 complete, with the leakage checker passing on the final runs.
- $H_1$ and $H_2$ answered with confidence intervals on the held-out test split; all baselines, anchors, ablations and slices reported.
- Retrieval benchmark, in-corpus audit and faithfulness audit reported.
- Extended phases reported for whatever was completed, each with its own table.
- Final report states the known limitations of P§7.
