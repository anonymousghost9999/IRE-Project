# Paper Feed — Design Document

> **Purpose:** Records the current design of Paper Feed and the rationale behind each decision.
> **Governing spec:** `prob.md` (problem statement, formal definition, evaluation protocol). Where this document and `prob.md` differ, `prob.md` wins unless a decision in §18 says otherwise.
> **Revision note:** Rewritten after the LLM Council review (`issues.md`) so the design matches `prob.md`. Issue IDs (E1, L1, C1, …) refer to `issues.md`; Appendix A maps each issue to where it is resolved.

---

## 1. Overview and Central Claim

**Project aim.** Apply state-of-the-art techniques from Information Retrieval and paper recommendation (hybrid dense + lexical retrieval, scientific document embeddings, citation-graph signals, structured facets, diversity-aware reranking, grounded explanations) to build a high-quality research-paper recommender, and verify its performance rigorously with offline metrics. No component is adopted because a particular paper proposed it; each is kept only if the evaluation (§13) shows it helps.

Paper Feed is a personalized academic-paper recommender for researchers. It is built as a batch-processed, pretrained-models-only system (no deep model training; one optional light gradient-boosted ranker appears only as an Extended experiment, §9.1) over an OpenAlex CS/AI corpus of about 100,000 papers.

**Central claim (what the evaluation can actually test).** A user model that combines a recency-weighted persona, a decayed session state, time-bounded citation-graph signals and typed paper facets, followed by diversity-aware reranking, recommends **papers the researcher has not cited before** (discoveries) better than strong lexical, dense, popularity and citation-only baselines, and can explain each recommendation with a grounded, ranking-faithful reason.

Two evaluation tracks support the claim (see §13):
- **Track 1 (real behaviour, no session):** Leave-Future-Out on OpenAlex author timelines. Tests persona + citation + facets + MMR.
- **Track 2 (simulated behaviour, session):** LLM-simulated researchers. Tests session adaptation and long-term retention.

LFO does not test session mechanics (E1). Session results are therefore reported only from Track 2, labelled as simulated, and never mixed with Track 1 numbers.

Pipeline at a glance:
1. Crawl OpenAlex (author-first, ≤ T), reconstruct abstracts, build a time-bounded citation graph.
2. Embed papers (SPECTER2), extract facets (GLiNER), canonicalize facets, build indexes.
3. Build the user model: persona vector + facet set (from authored papers or onboarding seeds) and an optional session state.
4. Generate 100 candidates (dense + BM25 pseudo-query, fused by RRF), score them with a small linear model, diversify with MMR.
5. Produce template explanations from the signal that actually drove the ranking.
6. Evaluate retrieval, extraction, recommendation, session dynamics and explanation faithfulness.

---

## 2. Notation

One symbol, one meaning (C4, D4). Values marked *tuned* are chosen on the tuning split, not the test split.

| Symbol | Meaning | Default / range | Used in |
|---|---|---|---|
| $T$ | Evaluation cutoff year | 2023 | §4, §13 |
| $u$ | User / seed author | — | all |
| $\mathcal{C}$ | Candidate corpus, papers with year $\le T$ | ~100k | §4 |
| $\mathbf{e}_p$ | SPECTER2 (`proximity`) embedding of paper $p$, L2-normalized | 768-d | §6 |
| $\mathcal{F}_p$ | Canonical facet set of paper $p$ | — | §6 |
| $\mathbf{v}^{\text{pers}}_u$ | Persona vector | — | §7.1 |
| $\gamma_{\text{rec}}$ | Persona recency decay per year ($\gamma$ in prob.md §5) | 0.9 | §7.1 |
| $\mathcal{F}_u$ | User facet set (tasks, methods, datasets) | — | §7.1 |
| $\mathbf{v}^{\text{sess}}_t$ | Session vector at step $t$ (EMA) | $\mathbf{0}$ at start | §7.2 |
| $\alpha_{\text{sess}}$ | Session EMA decay (prob.md's $\alpha$) | 0.5; range [0.3, 0.8]; *tuned in Track 2 pilot* | §7.2 |
| $\gamma_{\text{pivot}}$ | Pivot-gate threshold (optional gate) | 0.35 | §7.2 |
| $\mathcal{S}^-_u$ | Skipped / dismissed papers in the session | — | §7.2 |
| $\rho$ | Persona weight in the retrieval query vector $\mathbf{q}_t$ | 0.3; *tuned in Track 2 pilot* | §8 |
| $M$, $K$ | Candidate pool size, feed size | 100, 10 | §8, §10 |
| $k_{\text{rrf}}$ | RRF constant | 60 | §8 |
| $M_{\text{channel}}$ | Papers each channel (dense, BM25) returns before fusion | 200 | §8 |
| $\text{mm}[\cdot]$ | Min-max scaling to $[0,1]$ across the candidate pool (binary indicators stay 0/1) | — | §9 |
| $w_{\text{pers}}, w_{\text{sess}}$ | Persona / session similarity weights | $w_{\text{pers}}=1$ fixed | §9 |
| $w_{\text{cite}}, w_{\text{facet}}, w_{\text{nov}}$ | Citation, facet-match, new-method weights | *tuned (Track 1 tuning split)* | §9 |
| $\lambda_{\text{div}}$ | MMR diversity weight ($\lambda_{\text{div}}$ in prob.md §5) | *tuned (Track 1 tuning split)* | §10 |
| $\lambda_{\text{neg}}$ | Negative-feedback penalty weight | *tuned in Track 2 pilot* | §9 |
| $\text{BC}(p\mid u)$ | Bibliographic coupling count (prob.md §3B) | — | §9 |
| $\text{CC}(p\mid u)$ | Co-citation count with user's papers, citing papers $\le T$ | — | §9 |

---

## 3. Scope

**Core (mandatory)**, as in `prob.md` §6A:
1. Offline data pipeline (corpus, ≤ T citation graph, GLiNER facets, SPECTER2 + BM25s indexes).
2. Two-stage recommendation engine (hybrid candidates, linear scoring, MMR, faithful explanations).
3. Track 1 LFO benchmark (baselines, ablations, cold-start slices, significance tests).
4. First-stage retrieval benchmark (CSFCube, LitSearch).
5. Explanation faithfulness audit.

**Extended (supplementary):**
1. Track 2 LLM session simulation.
2. SciERC extraction evaluation (the in-corpus 100-paper audit is Core, as it validates the facets that drive explanations).
3. Streamlit demo UI.
4. Small user pilot (n ≈ 10–15), §14.
5. SOTA variant experiments (§13.8): alternative dense encoders, cross-encoder reranking in feed mode, and a learned ranker (§9.1). The Core linear scorer remains the system of record; a variant replaces it only if it wins on tuning data.
6. Best configuration and comparison with state-of-the-art paper recommenders (§13.9): the best option for each part of the architecture, chosen from the §13.8 tuning evidence, is assembled into one system, evaluated once on the test split, and compared with current SOTA research-paper recommenders under the same protocol.

If the schedule slips, the minimum deliverable is: corpus + LFO harness + BM25 / SPECTER2 baselines + the recommender with ablations and confidence intervals (R5).

---

## 4. Corpus and Data

### 4.1 Source and construction (E5, L1)
- **Source:** OpenAlex only. Deduplication and author resolution use OpenAlex IDs (`openalex_id`, `doi`). No further resolution is attempted.
- **Author-first construction.**
  1. Select candidate evaluation authors (first or single authors; §13.1). Over-sample (for example ~1000) because the cohort is filtered after coverage is known.
  2. Crawl each author's papers with year $\le T$, their reference lists, and 1–2 hop citation neighbourhoods, **as of $\le T$**.
  3. Anchor with top-cited foundational CS/AI/IR papers plus thematic fill published $\le T$ (2024), scaling to $\ge 120,000$ papers.
  4. The crawl **never** traverses the reference list of target paper $B$ ($T+1 = 2025$). $\text{Ref}(B)$ is read only to compute coverage and ground truth.
- **Coverage report (before any model runs).** Fraction of $\text{Discovery}(B\mid u)$ present in $\mathcal{C}$ with an abstract, before and after cohort filtering. Recall is reported against the full reference set and the in-corpus subset, next to the empirical Corpus Coverage Ceiling.
- **Target validation.** A genuine $T+1$ paper cannot be cited by a paper from $\le T$. Authors whose $B$ is cited in a year $\le T$ (by a corpus paper, or per OpenAlex `counts_by_year`) are dropped: OpenAlex has dated an old work or a merged preprint/journal record to $T+1$, so they would not be a leave-future-out test. In the first build this removed 294 of 1,000 candidate authors.
- **Selection bias to state in the report.** Keeping authors with $\ge 5$ in-corpus discovery references selects authors whose references the crawl happened to reach, and single/first authors skew early-career (prob.md §4).
- **Non-CS papers are kept.** The expansion crawl pulls in papers from other OpenAlex fields (about 40% of the corpus in the first build). They are legitimate references and easy negatives, so the corpus is not filtered by field. The in-corpus Topic Entropy and Topic Discovery Rate are computed over all candidates as-is.
- **Authored papers are always ingested.** The evaluation authors' own papers bypass the type / language filter so their references always feed $\text{Citations}(u,\le T)$, BC and CC; those that would have failed the filter are graph nodes only (`in_candidates = 0`).
- **Hop-1 inbound citations** (needed for CC) use a fair per-author budget: up to 5 seed papers per author, up to 40 citing papers (≤ $T$) per seed, rank-by-rank across authors, with a global cap on newly added papers (`configs/default.yaml`, `corpus.hop1_*`).
- **Candidate restriction.** Only papers with year $\le T$ (2024) are retrievable. Corpus target is $\ge 120,000$ CS/AI/IR papers.

### 4.2 Abstracts (P5)
OpenAlex stores abstracts as `abstract_inverted_index`. The ingestion step reconstructs plain text from word positions.
- Papers **without an abstract** are kept as nodes in the citation graph (they still contribute reference lists and counts) but are **excluded from the candidate corpus** (no embedding or facets). The fraction affected is reported.
- A user's own authored paper without an abstract contributes a **title-only** embedding to the persona.

### 4.3 Paper metadata schema
`paper_id` (OpenAlex ID), `title`, `abstract`, `authors`, `year`, `venue`, `doi`, `url`, `references` (`referenced_works`), `citation_count_T` (from `counts_by_year`, up to $T$ only), `primary_topic` / `subfield` (OpenAlex Topic IDs, used as the **independent** labels in §13; if `primary_topic` is missing, use the highest-scoring entry in `topics`, else `UNKNOWN`), and the five extracted facets (§6).

**Never stored for use in features:** present-day `cited_by_count`, the cited-by (`citations`) list, or any count that includes years $> T$ (L1).

### 4.4 Time-bounded citation graph (L1)
- Edges are kept only if the **citing** paper has `publication_year` $\le T$. The graph is stored as a separate artifact per evaluation cutoff.
- Citation counts for any feature or baseline come from `counts_by_year` summed to $T$.
- $\text{BC}(p\mid u) = |\text{Refs}(p) \cap \text{Refs}(\text{Authored}(u,\le T))|$, and $\text{CC}(p\mid u)$ counts third-party papers (year $\le T$) citing both $p$ and a paper authored by $u$. Both are used as **counts**, not tiers (F3), and min-max scaled (§9).
- **Automated leakage check (runs in CI and before every evaluation):** for every evaluation user, assert that no feature, candidate, index entry or profile references paper $B$ or any paper with year $> T$.

---

## 5. Storage

| Data | Storage |
|---|---|
| Paper metadata, facets, canonical facet IDs, citation edges | SQLite |
| Users, profiles, sessions, interactions | SQLite |
| Paper embeddings | FAISS `IndexFlatIP` (exact inner product on normalized vectors) |
| Lexical index | **bm25s** (pure Python, no JVM) |

Exact FAISS is used because 100k × 768 vectors search in milliseconds; HNSW would add approximation noise to evaluation with no benefit at this scale (C6, C7).

Tables (aligned with `prob.md` §6D): `papers`, `paper_facets` (adds `canonical_id`), `facet_canon`, `paper_citations` (`source_paper_id`, `target_paper_id`, `source_year`), `users`, `user_profiles`, `user_sessions`, `user_interactions` (`action_type` ∈ `click`, `save`, `skip`).

---

## 6. Paper Representation and Facet Extraction

### 6.1 Dense encoder
- **Papers:** SPECTER2 with the `proximity` adapter, 768-d, L2-normalized. Input is title + abstract.
- **Short text (search queries, onboarding seeds):** SPECTER2 `adhoc_query` adapter, projecting into the same space. No generative LLM is involved (HyDE is dropped: latency, VRAM, hallucination).
- **Default and alternatives.** SPECTER2 is the default encoder. Other pretrained scientific / general embedding models are compared against it in the Extended encoder study (§13.8, V1). If one wins on tuning data, it replaces SPECTER2 for the whole pipeline (dimension $d$ may then differ from 768).

### 6.2 Facet extraction (P2)
- **Extractor:** GLiNER (`urchade/gliner_medium-v2.1`), zero-shot, run locally on every candidate abstract.
- **Five facets:** `task`, `method`, `dataset`, `metric`, `topic`. Each is a verbatim span with character offsets, confidence, and the full sentence containing it as evidence.
- **Domain and prior work are not GLiNER facets.** Domain comes from OpenAlex topics; related prior work is `referenced_works`.
- A built-in rule-based splitter (`split_sentences`) is used for sentence splitting; it has no optional dependency so every machine produces identical facets.

### 6.3 Facet canonicalization (F5)
Exact string equality on verbatim spans ("BERT-based dual encoder", "dense passage retriever") almost never matches across papers. Facets are therefore canonicalized before any matching:
1. Lowercase and lemmatize.
2. Embed the span with a small sentence-embedding model, cluster spans per facet type with a similarity threshold, and assign a `canonical_id`.
3. $\mathcal{F}_p$, $\mathcal{F}_u$, facet match and the new-method indicator all operate on `canonical_id`s.
4. The in-corpus audit (§13.5) reports facet match rates so a degenerate canonicalization (everything matches, or nothing does) is visible.

Explanations still display the original verbatim span and evidence sentence.

### 6.4 Limits
GLiNER is zero-shot and may be noisy on scientific text. The verbatim-span guarantee means the displayed text exists in the abstract (grounding); it does **not** mean the label is correct, so label accuracy is measured separately (§13.5).

---

## 7. User Model

### 7.1 Long-term persona $\mathbf{p}_u = (\mathbf{v}^{\text{pers}}_u, \mathcal{F}_u)$
As in `prob.md` §5:
$$\mathbf{v}^{\text{pers}}_u = \frac{\sum_{p\in\text{Authored}(u,\le T)} \gamma_{\text{rec}}^{\,T-\text{year}(p)}\,\mathbf{e}_p}{\left\|\sum_{p\in\text{Authored}(u,\le T)} \gamma_{\text{rec}}^{\,T-\text{year}(p)}\,\mathbf{e}_p\right\|_2}$$
- $\mathcal{F}_u$ is the set of canonical task, method and dataset facets of the user's authored papers.
- **Zero-publication users (user cold start):** $\mathbf{v}^{\text{pers}}_u$ is the unweighted mean of the `adhoc_query` embeddings of the onboarding seed topics. $\mathcal{F}_u$ is initialized from the facets of the 20 corpus papers nearest to that vector.
- In LFO, citations made by $u$ in papers with year $\le T$ feed only the **exclusion set** and the BC / CC signals. Nothing from year $> T$ enters the profile (L3).

### 7.2 Session state $\mathbf{s}_u = (\mathbf{v}^{\text{sess}}_t, \mathcal{S}^-_u)$ (C1, F1)
There is **one** canonical session update, the EMA from `prob.md` §6B:
$$\mathbf{v}^{\text{sess}}_t = \alpha_{\text{sess}}\,\mathbf{v}^{\text{sess}}_{t-1} + (1-\alpha_{\text{sess}})\,\mathbf{e}_{\text{clicked}},\qquad \mathbf{v}^{\text{sess}}_0=\mathbf{0}$$
applied on positive actions (`click`, `save`); skipped or dismissed papers are added to $\mathcal{S}^-_u$.

This is mathematically an exponentially position-decayed, normalized sum of the session's paper vectors with weights $\propto \alpha_{\text{sess}}^{\,t-i}$. The newest item has the largest weight. The earlier "position-decay" and "EMA" formulations in the previous design were the same model; there is no separate "V-SKNN" arm (E6). Real session-kNN over other users' sessions is not implemented because no session data exists.

**Optional pivot gate (decision §18).** A boolean flag `use_gate`, off by default:
$$\Delta_t = 1 - \cos(\mathbf{e}_{\text{clicked}}, \mathbf{v}^{\text{sess}}_{t-1});\quad \text{if } \Delta_t > \gamma_{\text{pivot}}:\ \mathbf{v}^{\text{sess}}_t = \mathbf{e}_{\text{clicked}}\ \text{(hard reset)}.$$
It is evaluated as an additional Track 2 arm (§13.2), not as the main model. It is described in the report as a "divergence-gated session reset" and is kept in the final system only if it helps in Track 2.

**Session boundaries and reset.**
- After $N$ minutes of inactivity (default 30; UI only) a new session starts with $\mathbf{v}^{\text{sess}}=\mathbf{0}$ and $\mathcal{S}^-_u=\emptyset$; the persona carries long-term context.
- The "Reset Session Drift" button does the same.
- **In Track 1, the session is inactive** ($\mathbf{v}^{\text{sess}}=\mathbf{0}$, $\mathcal{S}^-_u=\emptyset$), as stated in `prob.md` §4.

### 7.3 Feedback signals
Discrete actions only; dwell time is not used (noisy and not simulable).

| Action | Effect |
|---|---|
| `click` (expand / open) | Positive: updates $\mathbf{v}^{\text{sess}}$ |
| `save` | Positive: updates $\mathbf{v}^{\text{sess}}$, adds to library, added to $\text{SessionConsumed}$ |
| `skip` / dismiss | Negative: adds to $\mathcal{S}^-_u$ |

Papers the user has cited are used only through $\text{Citations}(u,\le T)$ (exclusion, BC, CC).

---

## 8. Candidate Generation (P1, C3)

Two explicit modes. Both filter to year $\le T$ and remove $\mathcal{E}_u$ **before** scoring.

### 8.1 Feed mode (used by Track 1, Track 2 and the UI feed)
No text query exists, so one is constructed from the user model.
1. **Dense channel:** exact kNN on
   $$\mathbf{q}_t = \frac{(1-\rho)\,\mathbf{v}^{\text{sess}}_t + \rho\,\mathbf{v}^{\text{pers}}_u}{\|\cdot\|_2}$$
   (when $\mathbf{v}^{\text{sess}}_t=\mathbf{0}$, $\mathbf{q}_t=\mathbf{v}^{\text{pers}}_u$).
2. **Lexical channel:** BM25s over title + abstract with a **pseudo-query** built from (a) the user's most frequent canonical task / method / dataset facet terms and (b) the titles of the last 3 session papers, if any. The Lexical Baseline (§13.1) shares the same BM25s code path but uses the full text of the user's past papers as its query; the pseudo-query is deliberately shorter and facet-focused. For Track 1 (no session) part (b) uses the titles of the user's 3 most recent authored papers instead.
3. **Fusion:** each channel returns its top $M_{\text{channel}}=200$ papers after the filters above; Reciprocal Rank Fusion with $k_{\text{rrf}}=60$ keeps the top $M=100$ candidates.

**Core feed mode has no citation channel:** the citation signal enters only as a score feature over the text-retrieved pool. An Extended variant (§13.8, V4) adds the time-bounded citation graph as a third candidate channel; it is adopted only if it beats Core on tuning data.

**Core feed mode has no cross-encoder:** there is no user-written query to rerank against. An Extended variant (§13.8, V2) builds a *profile query text* and reranks the 100 candidates with a cross-encoder; it is adopted only if it beats the linear score on tuning data.

### 8.2 Search mode (UI search bar; retrieval benchmark)
- Query encoded with the `adhoc_query` adapter for dense kNN; BM25s on the query text; RRF ($k_{\text{rrf}}=60$), with the same $M_{\text{channel}}$ and $M$ as feed mode.
- An optional cross-encoder rerank (`BAAI/bge-reranker-base` or `cross-encoder/ms-marco-MiniLM-L-6-v2`, compare both) is applied to the fused list. It is evaluated only in the retrieval benchmark and is **not** part of Core.
- A search query can seed a topic shift: the query's top results are exposed to the session, matching Track 2's shift protocol.

---

## 9. Scoring (C2, S1)

One score, a small linear model over candidates in the pool. Every continuous feature is **min-max scaled to $[0,1]$ across the 100 candidates** (written $\text{mm}[\cdot]$) so weights are comparable and the explainer can compare contributions (prob.md §6E.5). Binary indicators stay raw 0/1, and a feature that is constant in the pool scales to 0. Min-max is used instead of z-scores because a rare binary feature that fires once among 100 candidates would get $z\approx 10$ and always look dominant. The negative-feedback penalty is not scaled.

$$\text{Score}(p) = w_{\text{pers}}\,\text{mm}\!\big[\cos(\mathbf{e}_p,\mathbf{v}^{\text{pers}}_u)\big] + w_{\text{sess}}\,\text{mm}\!\big[\cos(\mathbf{e}_p,\mathbf{v}^{\text{sess}}_t)\big] + w_{\text{cite}}\,\text{mm}\!\big[\log(1+\text{BC}+\text{CC})\big] + w_{\text{facet}}\,\text{mm}\big[S_{\text{facet}}\big] + w_{\text{nov}}\,\mathbb{1}_{\text{new-method}}(p) - \lambda_{\text{neg}}\,\text{Penalty}(p,\mathcal{S}^-_u)$$

- **Facet match (frequency-weighted coverage):** for each type $\tau\in\{\text{task},\text{dataset}\}$,
  $$S_\tau(p,u)=\frac{\sum_{f\in \mathcal{F}^\tau_p\cap\mathcal{F}^\tau_u}\text{freq}_u(f)}{|\mathcal{F}^\tau_p|}\quad(0\text{ if }\mathcal{F}^\tau_p=\emptyset),\qquad S_{\text{facet}}=\tfrac12\big(S_{\text{task}}+S_{\text{dataset}}\big)\in[0,1],$$
  where $\text{freq}_u(f)$ is the share of the user's authored papers (for zero-publication users, of the 20 nearest papers) that contain canonical facet $f$. It rewards the user's core tasks and datasets and is not inflated by abstracts that list many facets.
- **New-method indicator** (the explainer's "new Method" feature): $\mathbb{1}_{\text{new-method}}(p)=\mathbf{1}\big[\exists\,\text{task}\in\mathcal{F}_p\cap\mathcal{F}_u\ \text{ and }\ \exists\,\text{method}\in\mathcal{F}_p:\ \text{method}\notin\mathcal{F}_u\big]$ (a method of $p$ that is new to the user, on a task the user already works on). Matching on task/dataset and rewarding a new method are deliberate, so these two terms do not cancel each other.
- **Penalty:** $\max_{n\in\mathcal{S}^-_u}\cos(\mathbf{e}_p,\mathbf{e}_n)$, and 0 when $\mathcal{S}^-_u=\emptyset$ (`prob.md` §5).
- **When $\mathbf{v}^{\text{sess}}=\mathbf{0}$** the session term is dropped.
- **Zero-publication users** have no authored papers, so $\text{BC}=\text{CC}=0$ and the citation term vanishes; their facet set comes from §7.1.

**Deliberately not in the score (S1, L1, item cold start).**
- Global authority $\log(1+\text{citation\_count})$: biases toward old papers (prob.md §4, item cold start) and is leakage-prone. Counts up to $T$ exist only for the Popularity Baseline.
- Seed-proximity, a separate RRF term, citation recency decay (F4), self-citation dampening (own papers are hard-excluded), and the tiered +2.0 / +1.2 / +0.7 / +0.2 citation scale. In the main LFO run the +2.0 "user cited $p$" tier is a no-op because $\text{Citations}(u,\le T)\subseteq\mathcal{E}_u$ (E2); direct re-citation is evaluated only in the diagnostic run (§13.1).
- Extra signals are added only if they improve **tuning-split** NDCG@10.

**Tuned parameters.** Track 1 tuning split (80 authors): $w_{\text{cite}}, w_{\text{facet}}, w_{\text{nov}}, \lambda_{\text{div}}$ via a coarse grid. Track 2 pilot personas (about 10, disjoint from the 30 evaluation personas): $\alpha_{\text{sess}}, \rho, w_{\text{sess}}, \lambda_{\text{neg}}, \gamma_{\text{pivot}}$; Track 2 reuses the Track 1 values of $w_{\text{cite}}, w_{\text{facet}}, w_{\text{nov}}, \lambda_{\text{div}}$. Everything else is fixed at the defaults in §2.

**Facet-active variant (selection rule fixed before any test run).** The headline system is the unconstrained grid optimum above. Because the facet and new-method terms are what make span-grounded explanations possible (§11), a secondary *facet-active* configuration is also defined, chosen on the tuning split only:
1. Candidates: grid points with $w_{\text{facet}} > 0$ (and $w_{\text{pers}} = 1$, as everywhere), taken in order of tuning-split mean Discovery NDCG@10.
2. *Non-inferiority:* the paired-bootstrap 95% CI (over the 80 tuning authors) of its relative NDCG@10 loss against the grid optimum has an upper end of at most 5%.
3. *Explainability:* at least 10% of the tuning authors' Top-10 items have a facet or new-method dominant contribution (the feature their explanation would show).
4. The first candidate meeting both is the facet-active variant. If none does, there is no facet-active variant and that is reported, with the explainability cost or accuracy cost that ruled each candidate out.

It is reported as a labelled secondary system next to the headline (Track 1 tables, faithfulness audit), never in place of it.

### 9.1 Extended: learned ranker (V3)
A small **gradient-boosted learning-to-rank model** (LightGBM LambdaMART) over the same min-max-scaled features as the linear score (persona, session, citation, facet match, new-method, penalty, plus the RRF rank of each channel). It is light training, not deep learning, and is the only trained component in the project.
- **Training data:** the Track 1 **tuning split only** (80 authors; candidates labelled 1 if in-corpus $\text{Discovery}(B\mid u)$, else 0). Because 80 queries is small, use a shallow model, strong regularization and 5-fold cross-validation **inside the tuning split**; compare with the linear scorer on the same folds.
- **Test rule:** the held-out test split is used once, after the choice between linear and learned scorer is frozen.
- **No leakage:** features are the same ≤ T features as the linear model; the labels use $B$ only on the tuning authors.
- **Explanations:** when the learned ranker is used, the explainer falls back to per-feature contributions from SHAP values (faithfulness audit unchanged). If this proves unreliable, the learned ranker is reported as an accuracy experiment only and the linear scorer remains the shipped, explainable system.
- Because it needs labels, the learned ranker is **not** used in Track 2 or for the zero-publication slice.

---

## 10. Diversity and Novelty (F2)

Novelty is **folded into relevance before MMR**, not added afterwards, so the diversified ordering is not undone:
$$\text{rel}(p) = \text{Score}(p)\ \ (\text{which already contains } w_{\text{nov}}\,\mathbb{1}_{\text{new-method}}).$$

**MMR (greedy, $K=10$):**
$$p^\star = \arg\max_{p\in\text{pool}\setminus\mathcal{R}}\ \big[\text{rel}(p) - \lambda_{\text{div}}\max_{q\in\mathcal{R}}\cos(\mathbf{e}_p,\mathbf{e}_q)\big].$$

The set-level terms in `prob.md` §5 (Intra-List Diversity, Facet Novelty ratio, Penalty) are the *objective* that MMR approximates greedily; they are also reported as metrics (§13.1). The set-level Novelty term of `prob.md` §5 is realized per item by the new-method feature in the score; the Facet Novelty ratio is a secondary evaluation metric, not a score term. Because diversity is optimized with SPECTER2 similarity, the **primary** diversity and novelty metrics are computed from independent OpenAlex topic labels (§13.1).

---

## 11. Explanations (P4, S4)

- **Method:** deterministic templates. The explanation is filled from the extracted facets, their verbatim spans and evidence sentences, and verified citation edges. No LLM paraphrasing, no external API, **no NLI verification** (it would be near-tautological on template output).
- **Faithfulness rule:** the explainer takes the feature with the highest positive **scaled** contribution $w_k\,\text{mm}_k$ (binary indicators contribute $w_k\cdot\{0,1\}$) to the paper's score among {facet match, new method, citation, session, persona} and maps it to a template, for example:
  - new method: *"Proposes **{method}**, which is new to your work, for **{task}** that you work on"*
  - facet match: *"Evaluates on **{dataset}**, used in {n} of your papers"* / *"Addresses **{task}**, studied in {n} of your papers"* (the matched facet with the highest $\text{freq}_u$)
  - citation (bibliographic coupling): *"Shares **{n}** references with your paper *{title}*"*; citation (co-citation): *"Frequently cited together with your paper *{title}*"* (both edge-verified; the template follows whichever of BC / CC contributes more)
  - session / persona: *"Similar to **{session paper most similar to it}** you opened"* / *"Close to your overall research profile"*
- **Two criteria** (prob.md §2.4): factual grounding (every displayed span exists verbatim in the abstract, every citation relation is a real ≤ T edge) and ranking faithfulness (the shown feature is the dominant positive contribution, checked also by the counterfactual test in §13.6).
- A recommendation with no grounded explanation is shown without one and counts against Explanation Coverage (target ≥ 90%).

---

## 12. Cold Start and New Papers

- **User cold start:** onboarding seed topics → persona (§7.1). Evaluated explicitly by the *Zero-Publication Seed-Only* slice (§13.1).
- **Item cold start:** the score has no authority or raw-popularity term. Zero-citation papers can rank through content (SPECTER2, facet match, new method). Evaluated by the *Item Cold-Start* slice (fresh ≤ 1 citation vs established ≥ 20 citations at $T$).
- **Citation-count window.** OpenAlex `counts_by_year` starts in 2012, so `citation_count_T` is the number of citations in 2012–$T$ and is complete only for papers published from 2012 on; an old paper whose citations all predate 2012 would otherwise look uncited. Hence "fresh" means $\le 1$ citation **and** published in $[T-1, T]$, "established" means $\ge 20$ citations and published since 2012, and the Popularity Baseline ranks papers published since 2012 only.
- **Popularity fallback** (Popularity Baseline, and a UI fallback when nothing else is available) uses `counts_by_year` up to $T$ only (L4).
- **New-paper ingestion:** periodic batch (crawl → clean → embed → GLiNER → canonicalize → index). Real-time ingestion is out of scope. The evaluation corpus is a frozen snapshot with year $\le T$.

---

## 13. Evaluation

Everything in this section follows `prob.md` §4 and §6E. Only the implementation details added here are new.

### 13.1 Track 1: Leave-Future-Out on OpenAlex (Core)
**Cohort (R1, R3).**
- $N=400$ verified first or single authors with target paper $B$ in $T+1=2024$ having $\ge 5$ in-corpus Discovery references.
- Seed-author rule: one seed author per paper $B$, and $B$ must be a genuine $T+1$ paper (§4.1). Moderate and sparse authors are rare, so all qualifying ones are kept and the rest of the cohort is filled with seeded warm authors, which tilts the cohort towards less-published authors. Authors with implausible histories (too many papers per year, multiple unrelated fields) are filtered; the rule and final count are reported.
- Split: **tuning** 80 (20%), **held-out test** 320 (80%). All reported results are on the test split.

**Ground truth.** $\text{Discovery}(B\mid u)$ and $\text{ReCitation}(B\mid u)$ exactly as in `prob.md` §4, with references restricted to year $\le T$ and authored papers (self-citations) removed (R2).

**Headline metrics:** Discovery NDCG@10, Recall@10 / HitRate@10, MRR, with both full-reference and in-corpus recall and the Corpus Coverage Ceiling. Each metric is also reported as a percentage of **Oracle@10** (all in-corpus Discovery references ranked first), and the candidate stage's **Discovery Recall@100** is reported as the ceiling for reranking.

**Secondary / independent metrics (circularity guard):**
- *Topic Entropy* of the Top-10 over OpenAlex primary **Topics** (primary diversity metric), also reported normalized by $\log_2 10$ so its range is $[0,1]$.
- *Topic Discovery Rate*: share of Top-10 whose OpenAlex primary Topic never appeared in $u$'s papers $\le T$ (primary novelty metric). OpenAlex Topics (not the deprecated Concepts) are used for every external label; Track 2 uses the Topic hierarchy's subfield level for Topic A / B.
- *Missing topics* fall back as in §4.3. `UNKNOWN` is one category in Topic Entropy, never "new" in the Discovery Rate, neither Topic A nor B in Track 2, and never a Popularity domain; its share is reported.
- Embedding ILD (secondary) and Facet Novelty (secondary).

**Diagnostic run.** Re-citation Recall@10 with $\text{Citations}(u,\le T)$ removed from $\mathcal{E}_u$. A baseline "re-recommend the author's own past references" is reported only here.

**Cold-start slices:** Warm (≥ 5 prior papers), Moderate (3–4) and Sparse (1–2) partition the test cohort. Zero-Publication Seed-Only (3 seed topics from the author's primary OpenAlex Topics) is a re-run of the whole test cohort, and Item Cold-Start is a cut over candidate items, so these two overlap with the others and are reported separately, each with its $n$.
- **Zero-Publication slice, no oracle information:** $\mathcal{E}_u=\emptyset$ (the system knows nothing about the user) and the ground truth is $\text{Ref}(B)\cap\{\text{year}\le T\}\setminus\text{Authored}(u,\le T)$, so past citations are not removed because a new user has none. Results are compared only with baselines run in the same slice, not with the other slices.
- Report $n$ per slice. The Zero-Publication slice is a re-run of the test cohort with the profile replaced by the 3 seed topics. There, the Lexical and Dense baselines use the seed-topic text as the query, Citation-Only is not applicable (no authored papers), and Popularity is unchanged.
- "Primary domain" for the Popularity baseline is the modal OpenAlex subfield of the author's papers $\le T$ (the seed Topic's subfield for zero-publication users).

**Baselines** (same year ≤ T filter and same $\mathcal{E}_u$ as the system, so differences are not caused by leakage or exclusion — E4):
1. Popularity: most-cited papers up to $T$ in the user's primary domain (`counts_by_year`).
2. Lexical: BM25s with the user's past titles/abstracts as the query.
3. Dense centroid: unweighted mean SPECTER2 embedding of past papers.
4. Citation-only: BC + CC score without text.
5. Hybrid-Persona (strongest simple baseline): SPECTER2 persona vector plus the BM25s pseudo-query fused by RRF, i.e. the system with no citation signal, no facet features and no MMR, given the same tuning budget on the tuning split ($\gamma_{\text{rec}}$ and pseudo-query length). It isolates what graph, facets and MMR add beyond a well-tuned hybrid retriever.
6. Optional extra: recent papers by the author's co-authors (R4).

**Reference points** (anchors, not baselines to beat; no significance tests against them): *Random* (10 papers drawn uniformly from the eligible pool, averaged over 100 draws) is the floor; *Oracle@10* (all in-corpus Discovery references ranked first) is the ceiling that coverage and $K=10$ allow.

**Component ablations** (matched candidate pool; one component changed at a time): No-MMR ($\lambda_{\text{div}}=0$), No-Graph ($w_{\text{cite}}=0$), No-Facets ($w_{\text{facet}}=w_{\text{nov}}=0$, the same definition as in `prob.md`).

**Statistics (R3).** Paired bootstrap across authors with Benjamini–Hochberg correction at $\alpha=0.05$, 95% bootstrap CIs, Cohen's $d$.

**Hypotheses** (as in `prob.md` §6E). $H_1$: significant gains in Discovery NDCG@10 and Recall@10 over all five baselines (the four plus Hybrid-Persona), paired bootstrap with BH correction. $H_2$ (non-inferiority): MMR increases Topic Entropy (one-sided paired bootstrap), and the upper end of the 95% CI of the relative NDCG@10 loss is ≤ 3%. Report the minimum detectable effect at $N_{\text{test}}=320$ so a non-significant result is read as "inconclusive", not "no effect".

### 13.2 Track 2: LLM session simulation (Extended) — decision §18
- **Role:** behavioural sanity check of session dynamics, not proof of human relevance. Results are never mixed with Track 1 numbers.
- **Simulator LLM:** a **free-tier hosted model**, fixed checkpoint, used **only** for the simulated researchers. The recommendation pipeline itself contains no LLM. To keep this reproducible and at zero cost:
  - record the exact model name/version and run dates;
  - cache every response (key: persona ID, turn, prompt hash, model) and release the cache with the results;
  - batch ~10 abstracts per rating call (30 personas × 5 seeds × ~15 turns ≈ 2–3k calls) and spread runs over days to respect rate limits;
  - if quota is insufficient, reduce seeds or personas and say so; do not fall back to a different model mid-experiment;
  - only public OpenAlex metadata is sent.
- **Personas:** 30, seeded from real OpenAlex author profiles, run with 5 simulator seeds (the recommender is deterministic, so seeds vary only the LLM's sampling). Cite Agent4Rec and RecAgent as the simulation paradigm.
- **Click policy:** rating ≥ 4/5 → `click`, 5/5 → `save`, ≤ 2/5 → `skip`.
- **Shift protocol:** Topic A (persona's subfield) and Topic B (different OpenAlex subfield) are labelled with **OpenAlex topic IDs**, never GLiNER or SPECTER2. The agent issues a search query for Topic B (§8.2); the agent's clicks on those search results update the session like any other click. Topic B is chosen among subfields with at least ~500 papers in the corpus so that Topic B candidates exist.
- **Metrics:** Pivot Latency $k_{\text{pivot}}$ (steps until Topic B ≥ 50% of the Top-10; censored at the maximum number of turns if never reached, so Persona-Only is reported as censored, not infinite) and unforced Retention@$k$ (share of Topic A in the Top-10 $k$ steps after re-engagement, with no reset button).
- **Re-engagement:** the agent's goal switches back to Topic A and it runs a fixed Topic A search query, clicking the results it rates ≥ 4; if none qualify, it clicks the highest-rated Topic A result. Search mode does not depend on the arm, so every arm can re-engage.
- **Arms** (identical candidate generator, citation / facet signals and MMR; only the user-state terms differ — E4). The first feed (turn 0) of every arm is built from the persona, since no clicks exist yet; arms diverge from the first click:
  1. Persona-Only Static: $\rho=1$, $w_{\text{sess}}=0$, $\lambda_{\text{neg}}=0$, and the pseudo-query uses authored titles only (no session input anywhere).
  2. Session-Only Decay: $\rho=0$, $w_{\text{pers}}=0$, skip penalty on.
  3. Static Uniform Blend: $\mathbf{q}_t$ and score use a fixed 50/50 blend of persona and un-decayed session mean ($w_{\text{pers}}=w_{\text{sess}}$), skip penalty on.
  4. **Hybrid** (proposed: EMA session + recency-weighted persona + negative penalty).
  5. Hybrid + pivot gate (`use_gate=true`) — the optional gated variant.
- **$H_3$** as in `prob.md`: the Hybrid has lower Pivot Latency than Persona-Only and Static Blend and higher unforced retention than Session-Only, each supported if the 95% paired-bootstrap CI of the difference (over persona × seed units) excludes 0. Absolute levels such as Retention@3 around 70% are descriptive targets reported honestly, not pass/fail criteria. The simulation is circular by nature (E7), so it is described as a sanity check and no number from it is a headline claim.

**Track 2 specification (details fixed before implementation, after Phase 10).** Phase 10 adopted no variant, so Track 2 runs the Core system (SPECTER2, linear score, Track 1 weights $w_{\text{cite}}, w_{\text{facet}}, w_{\text{nov}}, \lambda_{\text{div}}$).
1. *Simulator.* Google Gemini through the AI Studio free tier; the exact model ID is pinned in `configs/default.yaml` (`simulation.model`, a Flash-class model) before the pilot and never changed afterwards; the model ID and version returned by the API and the run dates are recorded. Temperature 0.7. The API key is read from the `GEMINI_API_KEY` environment variable and never stored in the repository. Only public OpenAlex metadata (titles, abstracts, subfield names) is sent.
2. *Rating call.* Up to 10 papers per call (title + abstract), with the persona text and the current goal; the reply is JSON `{"ratings": [{"id": ..., "rating": 1-5}]}`. Parsing is strict: every shown paper exactly once, integer ratings 1–5. A malformed reply is retried up to 3 times with backoff; after that the episode is marked failed and reported, never imputed. Policy: rating $\ge 4$ click, $5$ save, $\le 2$ skip.
3. *Rating cache.* A rating is cached per (persona, goal phase, paper, seed, model, prompt version); only papers without a cached rating are sent. Arms therefore see the same rating for the same paper (common random numbers): differences between arms come from what each arm shows, not from rating noise. This replaces the per-prompt cache key above and is what keeps the call budget feasible. The cache is released with the results.
4. *Personas.* Evaluation: 30 authors from the Track 1 **test** split; pilot: 10 from the **tuning** split (disjoint by construction). Eligible: at least 5 authored papers $\le T$, and a primary subfield (modal OpenAlex subfield of their papers) with at least 500 eligible corpus papers; chosen in a seeded order (seed 42). Persona text: the 3 most recent authored titles and the primary subfield's name. Topic A = the primary subfield; Topic B = a different subfield with at least 500 eligible corpus papers, chosen with a per-persona seed. Topic membership of any paper is its OpenAlex subfield; subfield names come from the raw OpenAlex records (`data/raw/`), and 36 subfields currently meet the 500-paper bound.
5. *Episode.* 5 Topic A feed turns (turns 0–4, goal A). Shift: the goal becomes Topic B and the agent runs the search query "<Topic B subfield name>", rates the top 10 and acts on them. Then up to 10 Topic B feed turns; Pivot Latency = the number of these turns until Topic B is at least 50% of the Top-10, censored at 10. Re-engagement: the goal returns to A, the agent runs the search query "<Topic A subfield name>" and clicks results rated $\ge 4$ (if none, the highest-rated Topic A result); Retention@$k$ = share of Topic A in the Top-10 of the $k$-th feed after that, $k = 1..3$. Every feed is rated and acted on; consumed papers are excluded afterwards, as in §7.2.
6. *Pilot tuning (10 pilot personas, 2 seeds).* Criterion, fixed in advance: $J = \overline{\text{Retention@3}} - \overline{k_{\text{pivot}}}/10$, computed on the Hybrid arm (and on Hybrid + gate for $\gamma_{\text{pivot}}$). One-at-a-time search around the defaults over $\alpha_{\text{sess}} \in \{0.3, 0.5, 0.7\}$, $\rho \in \{0.2, 0.3, 0.5\}$, $w_{\text{sess}} \in \{0.5, 1.0\}$, $\lambda_{\text{neg}} \in \{0, 0.5\}$, $\gamma_{\text{pivot}} \in \{0.25, 0.35, 0.45\}$, two passes, ties keeping the default. Session-Only uses the tuned $\alpha_{\text{sess}}$ and $\lambda_{\text{neg}}$. Frozen in `data/artifacts/track2_config.json` before the evaluation run.
7. *Budget and fallback, fixed in advance.* 30 personas × 5 seeds × 5 arms, about 20 feeds or searches per episode; the cache removes the repeated turn-0 feed and every paper seen by more than one arm. If the free quota cannot finish the evaluation within the project time, seeds are reduced to 3 first, then personas to 20; never a different model. Whatever was run is reported.
8. *Analysis.* $H_3$ as above, paired over persona × seed units; failed episodes are excluded pairwise and counted.

### 13.3 First-stage retrieval benchmark (Core)
CSFCube (paper-to-paper, `proximity` adapter) and LitSearch (query-to-paper, `adhoc_query` adapter); optionally SCIDOCS. Compare BM25s, SPECTER2 and RRF hybrid ($k=60$), plus the optional cross-encoder rerank of §8.2 as extra rows, on **Recall@100** (primary gate), nDCG@10 / nDCG@100, MRR.

### 13.4 Extraction benchmark: SciERC (Extended) (P3)
Strict and Relaxed span $F_1$ per label and macro/micro. Label mapping `task→Task`, `method→Method`, `metric→Metric`, `dataset→Material`, `topic→OtherScientificTerm`. `Material` and `OtherScientificTerm` are broader than `dataset` and `topic`, so report per-label $F_1$ and treat `topic` as low-confidence. An optional LLM baseline is limited to ≤ 100 abstracts with a ≤ 4B model in 4-bit on the 4 GB GPU, reported as indicative only (S2).

### 13.5 In-corpus profile audit (Core)
Stratified sample of 100 papers: Precision@facet (label correctness), evidence-span grounding (target 100% verbatim), facet density (share with ≥ 3 core facets), and facet canonicalization match rates (§6.3).

### 13.6 Explanation faithfulness audit (Core)
Sample of 100 explanations, as in `prob.md` §6E.5: Extractive Grounding Rate (100%), counterfactual necessity (drop-out rate when the explained feature is zeroed: paper leaves the Top-10 or falls ≥ 3 ranks, accounting for MMR), Explanation Coverage (≥ 90%).

### 13.7 Threats to validity (report these)
- **SPECTER2 model-level leakage (L2):** it was trained on Semantic Scholar citation data that may include 2023–24 links, possibly Paper B's references. A data cutoff cannot remove this. BM25 is the only fully clean baseline; check the model card's data cutoff and note it.
- LFO is a proxy for reading need: references of $B$ are what the author cited, not everything they read.
- OpenAlex author disambiguation errors; coverage-based cohort selection; early-career skew.
- GLiNER label noise on the facets that drive the explanation.
- Track 2 depends on one LLM family and is circular by construction.

### 13.7a Known limitations (out of scope; stated, not mitigated)
Same list as `prob.md` §7, to be stated in the final report:
1. **OpenAlex data quality:** author merges/splits, incomplete `referenced_works` and date semantics (preprint vs journal vs indexing date) are not audited; OpenAlex is used as-is with the cohort filters of §13.1.
2. **Pretrained-model contamination:** SPECTER2 (citation-trained), GLiNER and any alternative encoder in §13.8 may have seen test-period papers or links; BM25 is the only fully clean reference.
3. **Proxy validity:** references of paper $B$ are a proxy for reading need, not a human-judged relevance set.
4. **Coverage-based cohort selection** and first/single-author skew; absolute metrics describe the constructed corpus, and the realized cohort size is reported as obtained.
5. **Simulator reproducibility:** one hosted LLM whose behaviour may change across versions.
6. **Feasibility not pre-verified:** compute time, corpus size and cohort size, with no formal go/no-go gates.
7. **Ethics and licences:** no formal ethics review beyond consent for the pilot; OpenAlex terms and model licences are used as documented by their providers.

### 13.8 Extended SOTA variant experiments
The aim (§1) is a state-of-the-art system, so four upgrades are tested against the Core system. All use the same cohort, metrics and statistics as §13.1.

**Selection rule (prevents test-set tuning).** Variants are compared with the Core system on the **tuning split** and on the retrieval benchmarks (§13.3). Any replacement decision is frozen before the test split is touched; the final table reports every variant on the test split, including those that lost. Variant comparisons are exploratory: report effect sizes with BH-corrected p-values over the whole family of comparisons.

| ID | Variant | What changes | Where compared |
|---|---|---|---|
| **V1** | Alternative dense encoders | Swap SPECTER2 for other pretrained models of base size or smaller (to fit ~4 GB VRAM). Candidates: SciNCL (citation-neighbourhood contrastive), and strong general embedding models such as BGE, E5 or a small Qwen-embedding model. Each uses its own recommended query format in place of the `adhoc_query` adapter. | CSFCube / LitSearch Recall@100 and nDCG; LFO with the Dense Centroid baseline and with the full system |
| **V2** | Cross-encoder rerank in feed mode | Build a profile query text (titles of the user's most recent authored papers + top canonical task / method / dataset terms), rerank the 100 candidates with `bge-reranker-base` and `ms-marco-MiniLM-L-6-v2` (compare both), and add the reranker score as a feature of the linear score. | LFO Discovery NDCG@10 / Recall@10; also latency per feed |
| **V3** | Learned ranker | LightGBM LambdaMART as in §9.1. | LFO (5-fold CV on tuning split, then one test run) |
| **V4** | Citation-graph candidate channel | A third feed-mode channel (§8.1) ranks eligible papers by $\log(1+\text{BC}+\text{CC})$ with the user's authored papers, over the whole eligible corpus. Specification below. | LFO tuning split: candidate Recall@100 and Discovery NDCG@10 / Recall@10 |

**V1 specification (details fixed before implementation, after the V2, V3 and V4 runs).**
1. *Encoders.* Track 1 candidates: **SciNCL** (`malteos/scincl`, 2022) and **BGE** (`BAAI/bge-base-en-v1.5`, 2023), both base size, both released before the end of the cutoff year, so neither can encode text from after $T$. **Qwen3-Embedding-0.6B** (2025) is evaluated on the retrieval benchmarks only: it was trained after $T$, so on the LFO protocol it could carry future information (§4.4, §13.7) and it never enters a Track 1 decision.
2. *Text formats (each model's documented usage, 512 tokens for all).* Documents: SciNCL `title [SEP] abstract` with [CLS] pooling; BGE `title. abstract` with [CLS] pooling and L2 normalisation; Qwen3 `title. abstract` with last-token pooling. Text queries (LitSearch, search mode): SciNCL the plain query; BGE prefixed with "Represent this sentence for searching relevant passages: "; Qwen3 as "Instruct: Given a research question, retrieve scientific papers that answer it\nQuery: {query}". Paper-to-paper use (CSFCube, persona centroids, feed mode) uses the document format only. SPECTER2 keeps its adapters (proximity for documents, `adhoc_query` for queries).
3. *Retrieval benchmarks (§13.3), reported, not decisive:* dense-only and RRF (with the same BM25s) rows for all four encoders on CSFCube and LitSearch, with the Phase 7 metrics.
4. *The 20k subset (encoder-neutral).* Seeds: the tuning authors' authored papers $\le T$ that are in the corpus. Add every eligible paper one citation hop from a seed (references and citing papers, citing year $\le T$). If that exceeds 20,000, keep the 20,000 with the most links to the seeds (ties by paper ID); if it falls short, add eligible two-hop papers in the same order until 20,000. The rule uses only the $\le T$ graph, never targets or embeddings; the share of tuning-split Discovery targets inside the subset is reported as context.
5. *Tuning-split comparison on the subset.* For each of SPECTER2, SciNCL and BGE: embed the subset, build its dense index, and rebuild BM25s on the subset (shared by all encoders); candidates are restricted to the subset. Facets do not depend on the encoder and are reused. Run the §9 grid on the 80 tuning authors; report the full system at its optimum, the Dense Centroid baseline and candidate Recall@100.
6. *Decision:* the alternative with the higher full-system tuning NDCG@10 at its optimum is the V1 candidate; it replaces SPECTER2 only if the paired-bootstrap 95% CI of its gain over SPECTER2 (both on the subset, each at its own optimum) lies above zero. Frozen in `variant_v1_decision.json` before any test run.
7. *Test run.* The V1 candidate, adopted or not, is used to embed the full corpus and build its index; the §9 grid is re-run on the full-corpus tuning split for it (the pool differs from the subset); then one labelled Track 1 test run and its audit, as for the other variants.
8. *Risks to report:* SciNCL was trained on citation neighbourhoods (S2ORC), so part of its advantage may be the citation signal the system already has (§13.7); BGE is a general web model; the subset is denser around the tuning authors than the full corpus, so absolute numbers there are not comparable with Track 1.

**V2 specification (details fixed before implementation, after the Core and V4 test runs).** V2 was planned from the start; these points fix what the table above leaves open.
1. *Query and documents:* the profile query text is the feed pseudo-query of §8.1 (top canonical task / method / dataset terms + titles of the 3 most recent authored papers, or the seed-topic profile for zero-publication users), so no new text choices are introduced. Each candidate is its title + abstract, truncated to the model's 512 tokens.
2. *Pool:* the Core two-channel pool (V4 was not adopted on tuning data). V2 changes only the score, never the pool.
3. *Feature:* the reranker's raw score, min-max scaled within the pool like every other continuous feature (§9), enters the linear score as $w_{\text{ce}}\,\text{mm}[\text{CE}(q,p)]$.
4. *Tuning (tuning split, 80 authors):* the §9 grid extended by $w_{\text{ce}} \in \{0, 0.25, 0.5, 1\}$, run separately for each reranker. Reranker scores are computed once per tuning author's pool and cached, so the grid only rescores.
5. *Model choice and decision:* the reranker with the higher tuning-split NDCG@10 at its own grid optimum is the V2 candidate; V2 replaces Core only if the paired-bootstrap 95% CI of that optimum's NDCG@10 gain over Core lies above zero (§13.9). Picking the better of two rerankers before the test favours V2 slightly; this is stated in the report. If the optimum has $w_{\text{ce}}=0$, V2 is Core and is not adopted.
6. *Explanations:* the reranker score has no grounded template. The explainer keeps explaining the largest *explainable* term (§11), and the faithfulness audit additionally reports the share of Top-10 items whose reranker contribution exceeds the explained term, i.e. where the explanation is not the main reason for the rank. A V2 system with a high share is reported as weaker on explainability whatever its accuracy.
7. *Latency:* reranker time per feed (mean and 95th percentile, 100 pairs, with the hardware) is recorded on the tuning authors.
8. *Test run:* once, after the decision is frozen, as a labelled row of the variant table, with the same ablations as Core.

**V3 specification (details of §9.1 fixed before implementation).**
1. *Pool and features:* the Core pool and Core features, whatever V2 and V4 decide (combinations are §13.9). Six features per candidate: $\text{mm}[\cos_{\text{pers}}]$, $\text{mm}[\log(1+\text{BC}+\text{CC})]$, $\text{mm}[S_{\text{facet}}]$, $\mathbb{1}_{\text{new-method}}$, and the dense and lexical channel ranks (a channel that missed the paper gives $M_{\text{channel}}+1$). Session and penalty are 0 in Track 1 and are left out.
2. *Labels:* 1 if the candidate is an in-corpus Discovery target of the author, else 0; one query per author.
3. *Model, fixed in advance (no hyperparameter search on 80 authors):* LightGBM `lambdarank`, 100 trees, learning rate 0.05, `num_leaves` 7, `max_depth` 3, `min_child_samples` 20, `reg_lambda` 1.0, `colsample_bytree` 0.8, no bagging, `deterministic=True`, seed 42.
4. *Diversification:* model scores are min-max scaled to $[0,1]$ within each pool, then MMR as in §10 with $\lambda_{\text{div}}$ chosen from the §9 grid on the training authors.
5. *Comparison (tuning split only):* the 80 tuning authors are split into 5 folds by author (fixed seed). In each fold, the linear scorer is tuned on the §9 grid and V3 is trained (and its $\lambda_{\text{div}}$ chosen) on the same 64 training authors; both are scored on the 16 held-out authors. This gives out-of-fold NDCG@10 for all 80 authors under both scorers. V3 replaces the linear scorer only if the paired-bootstrap 95% CI of its out-of-fold NDCG@10 gain lies above zero.
6. *Final model:* retrained on all 80 tuning authors with the same settings and its $\lambda_{\text{div}}$ chosen on them; saved with the decision before any test run.
7. *Explanations:* per-candidate SHAP contributions (LightGBM `pred_contrib`). The explainer shows the template of the largest positive SHAP contribution among the explainable features (persona, citation, facet, new-method); the channel ranks have no template. The counterfactual sets the explained feature to its "off" value (scaled value 0, indicator 0), re-predicts, rescales and re-runs MMR. *Reliability rule:* if the counterfactual drop-out rate is below 50%, or the channel ranks outweigh every explainable feature in more than 50% of Top-10 items, the SHAP explanations are declared unreliable and V3 is reported as an accuracy experiment only.
8. *Scope:* the zero-publication slice keeps the linear scorer (V3 has no labels for it), and Track 2 never uses V3. One labelled test run after the decision is frozen.

**V4 specification (fixed before implementation, after the Core test run).** The Core test run (§13.1) showed the system losing to the Citation-only baseline while its text-retrieved pool contains only about 5% of the Discovery targets. V4 tests the obvious remedy. Because it was motivated by a test-split result, it is a labelled exploratory variant, decided on the tuning split only, and the Core result stays the pre-registered answer to $H_1$.
1. *Channel:* the eligible papers ($\le T$, $\mathcal{E}_u$ removed) with $\text{BC}+\text{CC}>0$ for the user's authored papers, ranked by $\log(1+\text{BC}+\text{CC})$ with ties broken by paper ID, top $M_{\text{channel}}=200$. Same definitions and code as the Citation-only baseline and the citation feature (§4.4). Zero-publication users have an empty graph channel, so their pool is the Core two-channel pool.
2. *Fusion:* plain three-channel RRF ($k_{\text{rrf}}=60$, equal channel weights), keeping $M=100$. No channel weights are tuned, so the only change is the pool.
3. *Scoring and explanations:* unchanged (§9, §11). The grid of §9 ($w_{\text{cite}}, w_{\text{facet}}, w_{\text{nov}}, \lambda_{\text{div}}$) is re-run on the tuning split with the V4 pool; its tuning file has its own configuration signature.
4. *Decision (tuning split, 80 authors):* V4 replaces the Core pool only if its gain in tuning-split mean Discovery NDCG@10 over Core (each at its own grid optimum) has a paired-bootstrap 95% CI above zero (§13.9). Candidate Recall@100 is reported alongside as the mechanism check.
5. *Test run:* once, after the decision is frozen, as a row of the variant table, compared with Core and with all five baselines (BH-corrected). Ablations: No-Graph removes both the channel and the citation feature; No-MMR and No-Facets as in §13.1.
6. *Expected risk:* the channel and the citation feature use the same counts, so V4 may simply converge to Citation-only. Whether the hybrid beats Citation-only, not Core, is the comparison that matters; a negative result is reported as such.

**Notes and risks to report.**
- *Encoder leakage:* SciNCL and SPECTER-family models were trained on citation data and may encode held-out links (§13.7, L2). General embedding models are trained on web data that may postdate $T$. Treat BM25 as the clean reference and discuss each encoder's training-data cutoff.
- *Reranker domain mismatch:* these cross-encoders are trained on web search relevance, not on paper-to-profile matching; V2 may not help, and a negative result is a valid finding.
- *Compute:* each encoder requires a full re-embedding of the corpus, so run V1 on a fixed 20k-paper subset (same author-first construction, containing the tuning authors' neighbourhoods) first, decide on the tuning authors only, and scale only the best candidate.
- Combined variants (for example V1 + V2 + V3 + V4) are reported only for the best individual choices; assembling them into one system is §13.9.

### 13.9 Best configuration and comparison with SOTA paper recommenders (Extended)
After the variant experiments (§13.8), each part of the architecture has a measured winner. This step assembles those winners into one system and asks the question the project aim (§1) raises: how does the best version of Paper Feed compare with current state-of-the-art research-paper recommenders?

**1. Per-component selection (tuning evidence only).** One decision per part, each from the evidence named here and never from the test-split variant table:

| Part | Options | Evidence |
|---|---|---|
| Dense encoder | SPECTER2 vs V1 alternatives | Retrieval benchmarks (§13.3) + LFO tuning split |
| Candidate generation | dense / BM25s / RRF hybrid; $M_{\text{channel}}$, $M$; with or without the V4 citation channel | Retrieval benchmarks + tuning-split Discovery Recall@100 and NDCG@10 (V4) |
| Feed reranking | none vs V2 cross-encoders | LFO tuning split |
| Scorer | linear vs V3 LambdaMART | 5-fold CV inside the tuning split |
| Diversification | $\lambda_{\text{div}}$ | LFO tuning split |
| Session model | EMA vs EMA + pivot gate; $\alpha_{\text{sess}}, \rho, w_{\text{sess}}, \lambda_{\text{neg}}$ | Track 2 pilot personas (§13.2) |

An option replaces the Core component only if its tuning-split gain has a 95% bootstrap CI above zero; otherwise the simpler Core component stays.

**2. Assembled-system check.** Components chosen one at a time can interact, so the assembled system is compared with Core on the tuning split. Replaced components that do not help are removed one at a time (backward elimination, tuning split only). If nothing beats Core, the best configuration is Core, and that is a valid finding. The result is frozen in `configs/best.yaml` with its evidence before the test split is used.

**3. SOTA comparison systems.**
- A literature review at the start of this step fixes the comparison set; its date and sources are reported, because the field moves faster than this document.
- **Inclusion criteria:** public code or an exact specification; runnable on our corpus with year $\le T$ and no post-$T$ training data or API calls (§4.4); fits the compute budget (§15.1). Each system gets the same tuning budget on the tuning split.
- **Starting shortlist, to be confirmed:** scientific-document embeddings as user-centroid kNN (SPECTER2, SciNCL); random-walk item recommenders on the time-bounded citation graph (P3α, RP3β); a collaborative graph neural recommender such as LightGCN trained on $\le T$ author–paper interactions, if compute allows; zero-shot LLM listwise reranking (RankGPT-style) of the same 100 candidates with a $\le$ 4B model in 4-bit.
- Hosted production recommenders (for example Semantic Scholar or Google Scholar feeds) cannot be restricted to $\le T$, would leak, and are discussed qualitatively only.
- Published numbers from other datasets are context only. Numeric comparison happens only under our LFO protocol and on the public retrieval benchmarks of §13.3, where published results share data and metrics.

**4. Evaluation.** Core, the best configuration and every included comparison system run once on the test split under the §13.1 protocol: same eligible pool, $\mathcal{E}_u$, metrics, anchors and slices. Statistics as in §13.1: paired bootstrap with BH correction over the family (best vs Core, best vs each comparison system), 95% CIs, Cohen's $d$, minimum detectable effect; comparisons are exploratory. The best configuration also gets its retrieval-benchmark rows (§13.3), Explanation Coverage and the faithfulness audit (§13.6, re-run if the scorer or features changed; SHAP-based if V3 is chosen) and latency per feed. Because the test split was already used once for the §13.8 variant table, this run is reported as confirmatory.

**Phase 11 specification (fixed before implementation, 2026-11-14, after Phase 10).**
1. *Per-component decisions from existing tuning evidence.* Dense encoder: SPECTER2 (V1 not adopted). Feed reranking: none (V2 not adopted). Scorer: linear (V3 not adopted). Candidate channels: text only (V4 not adopted). Diversification: the §9 grid optimum of the Track 1 tuning. Session model: the §2 defaults, because Track 2 (§13.2) is deferred; this row is marked "no evidence" in `configs/best.yaml`.
2. *Pool size, the one open part.* The only larger pool run so far (Track 1 "enhanced": $M=250$, $M_{\text{channel}}=500$) also added a popularity prior $w_{\text{pop}}$, which §9 excludes from the score; that run stays a secondary analysis and is never a selection candidate. Pool size alone is decided like a variant: $M=250$, $M_{\text{channel}}=500$, $w_{\text{pop}}=0$, each arm at its own §9 grid optimum on the 80 tuning authors; it replaces $M=100$, $M_{\text{channel}}=200$ only if the paired-bootstrap 95% CI of its NDCG@10 gain lies above zero. Candidate Recall@100 is reported alongside. Because this is a single change, the §13.9 assembled-system check is that comparison itself. The result is frozen in `configs/best.yaml` before any comparison system is run on the test split.
3. *Literature review (2026-11-14).* Sources: the survey of research-paper recommenders published November 2021 to December 2024 (Pinedo, Larrañaga and Arruarte, arXiv 2508.08828), the deployed Scholar Inbox recommender (arXiv 2504.08385, 2025), the strong-baseline reproducibility study of Ferrari Dacrema et al. (RecSys 2019) and work on small open LLM listwise rerankers (RankGPT, RankVicuna; LLM4Ranking, arXiv 2504.07439). The comparison set below follows from it.
4. *Included comparison systems.* Each is scored over the same eligible pool with the same $\mathcal{E}_u$, run on the same test authors, and tuned on the tuning split with a grid fixed here.
   - **Scholar Inbox-style per-user classifier:** a logistic-regression classifier per author on `thenlper/gte-large` embeddings of "title. abstract" (mean pooling), reduced to 256 dimensions by PCA fitted on the corpus, as in Scholar Inbox. Its explicit ratings are replaced by the offline equivalent: positives are the author's authored papers and their references $\le T$; negatives are 10 random eligible papers per positive (seeded). Tuned: $C \in \{0.01, 0.1, 1, 10\}$. All eligible papers are ranked by the classifier's probability.
   - **RP3β** on the $\le T$ citation graph as a bipartite citing-paper × cited-paper graph; the author is a node linked to the references of their authored papers. Score of item $i$: the three-step walk author → reference → citing paper → $i$, each transition probability raised to $\alpha$, divided by $\text{popularity}(i)^{\beta}$. Tuned: $\alpha \in \{0.5, 1.0, 1.5\}$, $\beta \in \{0, 0.25, 0.5, 0.75\}$; $\beta = 0$ is P3α, so both are covered.
   - **SciNCL user-centroid kNN:** the Dense Centroid baseline with `malteos/scincl` full-corpus vectors (V1 point 2 format). No parameters.
   - **RankGPT-style zero-shot listwise reranking** of the Core pool's 100 candidates by `Qwen/Qwen2.5-3B-Instruct` in 4-bit (released September 2024, so it cannot encode citations made after $T$). Query: the feed pseudo-query (§8.1); each candidate as its title and the first 300 characters of its abstract; RankGPT's sliding window of 20 with step 10 from the bottom; deterministic decoding. No tuning (zero-shot); an unparsable window keeps its input order and is counted. If 4-bit loading is unavailable on the run machine, `Qwen/Qwen2.5-1.5B-Instruct` in fp16 is used instead and reported.
5. *Excluded, with reasons.* LightGCN and similar graph neural recommenders: they need a collaborative-filtering user base and training beyond §15.1; RP3β represents graph recommenders, and Ferrari Dacrema et al. report tuned random-walk methods competitive with neural ones. Hosted feeds (Semantic Scholar, Google Scholar, Scholar Inbox itself): cannot be restricted to $\le T$. SPECTER2 user-centroid kNN is already the Dense Centroid baseline.
6. *Final run (§13.9 point 4).* One run with a manifest: the best configuration (Core unless point 2 adopts the larger pool), every included system, the five Track 1 baselines and the anchors on the test authors, with latency per feed. BH family: best vs Core (if different) and best vs each included system, on NDCG@10 and Recall@10.

**Notes and risks to report.**
- *Fairness:* comparison systems get the same corpus, candidate eligibility, exclusion set and tuning budget. A comparator that loses only because it was under-tuned is not evidence for Paper Feed.
- *Leakage:* pretrained comparators carry the same model-level risk as §13.7 (L2); each one's training-data cutoff is stated.
- *Explanations:* most comparison systems produce no explanations, so Explanation Coverage is reported as a capability difference, not ranked against accuracy.
- *Scope:* comparison systems are run as reference points only. They are never adopted into the pipeline, so the out-of-scope list in §16 still describes Paper Feed itself.

**Final report.** The report has a dedicated section on this step: the chosen option for each part and its tuning evidence, the assembled system against Core on the test split, and the SOTA comparison with the review date, inclusion criteria, the excluded systems and why, and the context-only status of published numbers.

---

## 14. Small User Pilot (Extended) — decision §18
- n ≈ 10–15 participants, scripted task: view a Top-10 feed built from their OpenAlex profile or chosen seed topics, rate relevance and explanation clarity on a 1–5 Likert scale.
- Written consent; store only the ratings; check whether the institute requires ethics approval (S3).
- Reported as qualitative support for the transparency objective, never as a headline result.

---

## 15. Compute, Reproducibility and Plan

### 15.1 Compute (R5)
Target hardware: RTX 3050 (≈ 4 GB VRAM), so only pretrained models of modest size run locally. Time for SPECTER2 + GLiNER over ~100k abstracts is **to be measured on day one** and recorded here; a corpus subset with the same author-first construction is the fallback if it does not fit the schedule.

### 15.2 Build order
1. Weeks 1–3: corpus, abstract reconstruction, time-bounded graph, leakage check, LFO harness, BM25 / SPECTER2 baselines.
2. Week 4: GLiNER + canonicalization, in-corpus audit, SciERC.
3. Recommender with its few tuned weights; tuning on the tuning split.
4. Ablations, statistics, faithfulness audit, retrieval benchmark.
5. Extended: Track 2, UI, user pilot, SOTA variant experiments (§13.8: V4 citation channel first, as it needs no new models; then V2, V3, and V1 encoders last, as they need re-embedding).
6. Extended, after §13.8: best-configuration assembly and comparison with SOTA paper recommenders (§13.9); the UI then serves the frozen best configuration.

Owners per component are to be assigned by the team.

### 15.3 Reproducibility and caching (P6)
- Record: code version, config, random seeds, model names/versions, index configuration, corpus snapshot, timestamps.
- Cache: embeddings and GLiNER outputs (key: paper ID + model + config hash) and all Track 2 LLM responses (§13.2). No prompt versioning is needed in the pipeline because it has no prompts.
- Deterministic settings where supported; evaluation splits versioned.

---

## 16. Out of Scope
Deep model training (a small gradient-boosted ranker is allowed as the Extended experiment V3, §9.1), custom supervised IE, paid or external APIs inside the recommendation pipeline (the free-tier simulator in Track 2 is the single exception, §13.2), full-paper LLM processing, neural sequential recommenders (SASRec, GRU4Rec), GNN / RL / knowledge-graph recommenders, real-time ingestion, large user studies, unconstrained LLM explanations, summarization / agents, large-scale collaborative filtering, social recommendation, custom embedding / ANN / NLI models, extensive hyperparameter searches, and mitigating the known limitations of §13.7a. This list describes Paper Feed itself: the SOTA comparison of §13.9 may run some of these methods (for example a graph neural recommender) as external reference systems, without adopting them into the pipeline.

**Citation prediction is not the system's task (E3).** LFO uses the reference list of the author's next paper as a **proxy oracle** for reading need, and the headline is on discoveries with the citation signal ablated (No-Graph) next to the full system.

---

## 17. End-to-End Pipeline

```text
OpenAlex (author-first, <= T)
        │
        ▼
Clean + abstract reconstruction ──► time-bounded citation graph (citing year <= T)
        │
        ├─► SPECTER2 embeddings ──► FAISS (exact IP)
        ├─► BM25s index
        └─► GLiNER facets ──► canonicalization ──► SQLite

 FEED MODE                                   SEARCH MODE
 ─────────                                   ───────────
 persona p_u + session v_t                   text query
        │                                          │
        ├─ q_t ──► dense kNN                       ├─► adhoc_query encoder ──► dense kNN
        └─ profile pseudo-query ──► BM25s          └─► BM25s
                 │                                         │
                 └──────── RRF (k=60) → M = 100 ◄──────────┘
                                   │
            remove E_u, year > T    │          (search mode only: optional cross-encoder)
                                   ▼
           linear score: persona, session, citation, facet match,
           new-method, − negative-feedback penalty  (min-max scaled)
                                   │
                                   ▼
                       MMR (K = 10, lambda_div)
                                   │
                                   ▼
          template explanation from dominant positive contribution
                                   │
                                   ▼
                              Final feed
```

---

## 18. Decision Status

### Decisions made in this revision
| Decision | Choice | Alternatives rejected |
|---|---|---|
| Session model | One EMA session vector (equivalent to exponential position decay) fused with persona; **pivot gate is an optional flag**, evaluated as an extra Track 2 arm | Gate always on (hard flush hurts retention, extra parameter, baselines nearly identical); plain EMA with no gate option |
| Feed candidates | Dense kNN + BM25s **pseudo-query** from profile facets and recent titles, fused by RRF; no cross-encoder in feed mode | Dense + citation neighbours only (would contradict prob.md's "hybrid" wording) |
| Track 2 simulator | **Free-tier hosted LLM**, fixed checkpoint, cached responses, used only for personas | Local 4-bit LLM; rule-based simulator (circular) |
| User study | **Small pilot, n ≈ 10–15**, Extended | Drop; n = 50 |
| SOTA upgrades | Add as **Extended** variants V1 (encoders), V2 (feed cross-encoder), V3 (LightGBM LTR); linear scorer stays the Core system | Leaving the design classic-only; making any upgrade Core |
| Final system and SOTA comparison | **Extended** step §13.9: per-component winners chosen on tuning evidence, assembled, checked against Core on the tuning split, evaluated once on test, compared with literature-reviewed SOTA paper recommenders re-run under the LFO protocol | Choosing components from the test-split variant table; comparing against published numbers from other datasets |

### Fixed
- Recommendation target: papers. Corpus: OpenAlex only, author-first, ≤ T, ~100k.
- First stage: BM25s + SPECTER2 (`proximity`, `adhoc_query`) + RRF ($k=60$), $M_{\text{channel}}=200$ per channel, $M=100$. Exact FAISS. SPECTER2 is the default encoder; the Extended study (§13.8) may replace it, decided on tuning data only.
- Facets: GLiNER (`gliner_medium-v2.1`), 5 facets, canonicalized.
- Scoring: linear min-max-scaled score of §9; MMR after novelty; template explanations; no NLI.
- Evaluation: `prob.md` Track 1 / Track 2 / retrieval / extraction / faithfulness audit, with the leakage rules of §4.
- Reproducibility and temporal-leakage guard.

### Tentative / Open
- Cross-encoder choice for search mode (compare both).
- Facet canonicalization model and threshold.
- Hyperparameter values inside the ranges in §2 (tuned on tuning data only).
- Batch cadence for new-paper ingestion; user-pilot questionnaire wording.

---

## 19. Current Design in One Sentence

**Paper Feed is a batch-built, author-first, time-bounded OpenAlex corpus with BM25s + SPECTER2 hybrid retrieval, GLiNER facets with canonical IDs, a recency-weighted persona plus an EMA session state, a small min-max-scaled linear score (persona, session, citation graph, facet match, new method, negative penalty) followed by MMR, and template explanations tied to the dominant ranking signal, evaluated by Leave-Future-Out discovery recall on real author timelines and by an LLM-simulated session study, under strict temporal-leakage controls.**

---

## Appendix A. Issue → Resolution

| Issue | Resolved in |
|---|---|
| E1 no sessions in LFO | §1, §7.2 (session off in Track 1), §13.2 |
| E2 +2.0 tier vs ground truth | §9 (no-op in main run), §13.1 diagnostic |
| E3 citation prediction | §13.1 No-Graph ablation, §16 |
| E4 confounded ablation | §13.1 (same $\mathcal{E}_u$ / year filter), §13.2 (matched arms) |
| E5 coverage | §4.1 |
| E6 V-SKNN mislabel | §7.2 (renamed, collapsed into EMA) |
| E7 circular simulation | §13.2, §13.7 |
| L1 present-day citation counts | §4.3, §4.4 |
| L2 SPECTER2 leakage | §13.7 |
| L3 "cites paper" label | §7.1, §7.3 |
| L4 popularity fallback | §12 |
| F1 decay exponent | §7.2 |
| F2 novelty after MMR | §10 |
| F3 BC definition | §4.4 |
| F4 citation recency decay | §9 (removed) |
| F5 exact-match facets | §6.3 |
| C1 session update | §7.2 |
| C2 two scoring formulas | §9 |
| C3 query definitions | §8 |
| C4 overloaded symbols | §2 |
| C5 baseline 2 definition | §13.2 arms |
| C6 §37 mismatch | §18 regenerated |
| C7 BM25 | §5 |
| P1 feed without text query | §8.1 |
| P2 schema facets | §6.2 |
| P3 SciERC mapping | §13.4 |
| P4 NLI tautology | §11 |
| P5 abstracts | §4.2 |
| P6 stale LLM sections | §15.3 |
| S1 too many weights | §9 |
| S2 LLM baseline hardware | §13.4 |
| S3 user study | §14 |
| S4 external API | §11, §13.2, §16 |
| R1 author identity | §13.1 |
| R2 self-citations | §13.1 |
| R3 statistics | §13.1 |
| R4 baselines | §13.1 |
| R5 compute / fallback | §3, §15 |
| R6 PaperFlow reference | Dropped: no dependency on PaperFlow (§1 aim, §7.2) |
| D1–D4 documentation | whole document |

## Appendix B. Second council round: fixes applied

| Defect | Resolved in |
|---|---|
| z-normalization blows up rare binary features | §2 (mm), §9, §11, `prob.md` §6B / §6E.5 |
| Zero-publication slice used oracle exclusion; slices called "non-overlapping" | §13.1, `prob.md` §6E.1 |
| Concept vs Topic | §13.1, `prob.md` §6E.1 (OpenAlex Topics everywhere) |
| No-Facets defined differently in the two files | §13.1, `prob.md` §6E.1 (w_facet = w_nov = 0) |
| Missing anchors (tuned hybrid baseline, random, oracle, Recall@100) | §13.1, `prob.md` §6E.1 |
| H2 / H3 | §13.1, §13.2, `prob.md` §6E.1 / §6E.2 (non-inferiority; descriptive targets) |
| Risks to be stated as limitations | §13.7a, `prob.md` §7 |
Scope concerns raised by the council were set aside by the team's decision.
