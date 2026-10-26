# Paper Feed — Design Issues and Fixes

> Source: review of `paper_feed_current_design.md` (and `benchmarking_dataset.md`) by the LLM Council (2026-10-26), cross-checked against the doc.
> Section numbers (§) refer to `paper_feed_current_design.md` unless stated otherwise.
> Severity: **Critical** = invalidates results if unfixed · **High** = will be caught by a grader or breaks a component · **Medium** = weakens rigor · **Low** = cleanup.

---

## Summary

| ID | Severity | Issue | Where |
|----|----------|-------|-------|
| E1 | Critical | LFO evaluation has no sessions, so session mechanics are never tested | §11, §12, §15, §28 |
| E2 | Critical | Citation +2.0 tier can only score papers excluded from ground truth | §16.1, §16.4, §19 |
| E3 | Critical | Evaluation is citation prediction, built on the same graph as the citation signal | §16, §28, §35 |
| E4 | Critical | Ablation is confounded: only the proposed model gets citation, facets and MMR | §14, §28.3 |
| E5 | High | Corpus coverage of ground truth is never measured | §2.2, §28.1 |
| E6 | High | "V-SKNN" baseline is mislabelled | §14, §15, §28.3 |
| E7 | High | +33% simulation result is circular | benchmarking_dataset.md §3 Option 2, §5 |
| L1 | Critical | `cited_by_count`, co-citation and `citations` field leak post-cutoff data | §2.3, §16.1, §17.3 |
| L2 | High | SPECTER2 may already encode held-out citation links | §4.3, §26, §28 |
| L3 | High | "Cites paper" feedback signal is the evaluation label | §10 |
| L4 | Medium | Popularity cold-start fallback leaks future popularity | §23 |
| F1 | High | Position-decay weight gives the newest item weight e^δ > 1 | §15 |
| F2 | High | Novelty is added after MMR, which undoes diversity | §18, §19.4, §20 |
| F3 | Medium | Bibliographic coupling defined two incompatible ways | §16.1 |
| F4 | Medium | Citation "recency" decay actually penalizes paper age | §16.2 |
| F5 | Medium | Exact-match facet indicators on verbatim spans will rarely fire | §17.2, §19.2 |
| C1 | High | Session update defined two different ways | §9, §12, §15 |
| C2 | High | Two different final scoring formulas | §15, §17.5 |
| C3 | High | Three different query definitions | §5, §15, §17 |
| C4 | High | Symbols overloaded (γ, λ, w_m/w_d/w_t, S_cite) | §12, §15–§19, §28.4, §37 |
| C5 | High | Baseline 2 (PaperFlow) defined two ways | §14, §28.3 |
| C6 | High | §37 "Fixed" list disagrees with the body | §37 vs §4, §8, §18, §30 |
| C7 | Medium | BM25 implementation inconsistent | §3, §4.1, §26 |
| P1 | High | Feed mode has no text query for BM25 / RRF / reranker | §4, §6, §7 |
| P2 | Medium | Profile schema facets that GLiNER can't extract | §8, §8.1 |
| P3 | Medium | SciERC label mapping doesn't match the facet semantics | §27.1 |
| P4 | Medium | NLI verification of deterministic templates is near-tautological | §22 |
| P5 | Medium | OpenAlex abstract handling unspecified | §2, §24 |
| P6 | Low | Stale LLM-era sections (caching, prompt versions) | §31, §32, §34 |
| S1 | High | ~15+ tuned weights contradict "no extensive hyperparameter search" | §17.5, §19, §37, §35 |
| S2 | High | LLM baseline needs ~16 GB VRAM on a 4 GB GPU | §27.3 |
| S3 | High | n=50 user study: no protocol, out of scope, no ethics | §30, §35, §37 |
| S4 | Medium | External API listed despite cost = 0 | §22.3, §31 |
| R1 | High | Author identity / whose history is Paper B | §16.4, §28.1 |
| R2 | High | Self-citations in ground truth inflate citation signal | §16.3, §16.4 |
| R3 | High | No statistics: sample size, CIs, significance tests | §28 |
| R4 | High | Missing simple baselines | §28.3 |
| R5 | Medium | No compute budget or fallback deliverable | — |
| R6 | Medium | PaperFlow reference unverified / incomplete | §9, §14 |
| D1 | Low | Header says "no recommendations" but doc is full of them | header |
| D2 | Low | Broken code fences | §7, §11, §21, §30, §31, §36 |
| D3 | Low | Draft text left in | §2.2, §2.4, §2.5, §3, §4.1 |
| D4 | Low | No notation table | whole doc |

---

## A. Evaluation validity

### E1 — LFO evaluation has no sessions · Critical
**Where:** §11, §12, §15, §28.1, §28.3
**Problem:** LFO users are OpenAlex authors whose history is a set of year-stamped papers. There is no reading order within a year, no sessions, no inactivity gap, no skips/likes. Position decay (δ), the pivot gate (γ), the session timeout (N) and `P_skip` therefore have no real input. Any difference between V-SKNN, PaperFlow and V-PaperFlow will come from whatever ordering you impose.
**Fix:** Pick one and write it down:
- **(a) Re-scope the claim.** Make the main hypothesis something LFO can test (e.g. *"faceted persona + time-bounded citation-graph signals beat BM25 / SPECTER2 similarity at predicting next-paper references"*), and demote session gating to a secondary feature.
- **(b) Use a declared proxy order.** Order history items by the publication date of cited works (or by the order of the author's own papers) and state clearly that it is a proxy, not a reading session.
- **(c) Test session mechanics separately** on simulated trajectories (benchmarking_dataset.md Option 2), labelled as simulated and never mixed with LFO numbers.
- Optionally, test the pivot gate on **real pivots**: authors whose Paper B is far (in embedding space) from their history vs. stable authors; report how often the gate fires in each group.

### E2 — Citation +2.0 tier only scores excluded papers · Critical
**Where:** §16.1, §16.4, §19
**Problem:** `Citations(u)` (+2.0) = papers the user already cited. §16.4 removes already-read papers from ground truth, and §19 hard-filters consumed papers before scoring. So the strongest citation tier can only ever point at papers that are filtered out or don't count. If the exclusion is dropped instead, "re-recommend the author's old references" becomes a trivial baseline that beats everything.
**Fix:**
- Split ground truth into **re-citations** (in Ref(B) and previously cited by u) and **discoveries** (in Ref(B), never cited by u before T). Report metrics for each separately.
- The +2.0 tier is only meaningful for the re-citation split; for the discovery split, remove it (or it is a no-op).
- Add "re-recommend own past references" as an explicit baseline (see R4).

### E3 — The evaluation is citation prediction · Critical
**Where:** §16, §28, §35
**Problem:** Predicting Ref(B) is citation recommendation, which §35 lists as out of scope as the primary task. The citation signal (bib-coupling, co-citation) is computed from the same graph the labels come from, so it will win partly because it is close to the label, not because personalization is better.
**Fix:**
- Remove "Citation prediction as the primary task" from §35, or explicitly state that LFO uses citations as a *proxy* for reading need.
- Report results **with and without** the citation signal (already in §28.4 — make it a headline comparison, not an afterthought).
- Build all graph features only from papers published ≤ T (see L1).

### E4 — Confounded ablation · Critical
**Where:** §14, §28.3
**Problem:** The proposed model adds citation signal + facets + MMR on top of the session/persona core; the baselines get none of these. A win can't be attributed to "V-PaperFlow".
**Fix:** Run every arm on the **same candidate pool, same citation/facet signals and same MMR**, changing only the user-state component (session-only vs persona-only vs hybrid). Keep §28.4's component ablation as a separate table.

### E5 — Corpus coverage of ground truth is unknown · High
**Where:** §2.2, §28.1
**Problem:** A generic 100k CS/AI crawl will miss much of each author's Ref(B). Recall@10 is then capped near zero and differences between models are noise.
**Fix:**
- Pick evaluation authors first (e.g. ~500 authors with a paper in T+1).
- Crawl their references + 1–2 hop citation neighbourhoods, then fill up to 100k.
- **Report coverage** (fraction of Ref(B) present in the corpus with an abstract) before running any model. Evaluate only on in-corpus ground truth and state the ceiling.

### E6 — "V-SKNN" baseline is mislabelled · High
**Where:** §14, §15, §28.3
**Problem:** Jannach & Ludewig's V-SKNN is k-nearest-neighbour over *other users' sessions* with position-weighted overlap. The doc's version is a position-decayed centroid of the current session's embeddings — a different method.
**Fix:** Either rename it (e.g. "Decayed Session Centroid") and cite it as your own baseline, or implement actual session-kNN (which needs many sessions from many users — not available from LFO).

### E7 — +33% simulation result is circular · High
**Where:** benchmarking_dataset.md §3 Option 2, §5
**Problem:** In the simulation, a "read" is defined as a candidate matching the active cluster, and the models are scored on retrieving that cluster. The result mostly reflects how the simulator was built. It also shows PaperFlow beating V-SKNN, which undercuts the case for the hybrid.
**Fix:** Don't present +33% as a headline result. If you keep the simulation, describe it as a sanity check of drift adaptation, and state the circularity. Remove it from the "Summary Recommendation for Your Project Report" section.

---

## B. Temporal leakage

### L1 — Present-day citation data leaks the future · Critical
**Where:** §2.3 (`citations` field), §16.1 (CoCitation), §17.3 (`S_impact`, `S_cite`), §25
**Problem:** OpenAlex `cited_by_count`, the `citations` (cited-by) list and any co-citation computed from them are snapshots as of crawl time. They include Paper B itself and everything published after T. §25 forbids this but the design never says how to avoid it.
**Fix:**
- Use `counts_by_year` to compute citation counts up to T only.
- Build the citation graph only from edges whose **citing** paper has `publication_year ≤ T`.
- Store the cutoff-filtered graph as a separate artifact per evaluation split.
- Add an automated leakage check: assert no feature for user u references Paper B or any paper with year > T.

### L2 — SPECTER2 may encode held-out links · High
**Where:** §4.3, §26, §28
**Problem:** SPECTER2 was trained on Semantic Scholar citation data that likely includes 2023–24 citations — possibly Paper B's own references. A data-level cutoff doesn't remove this model-level leakage. "Pure SPECTER2" baselines may be inflated.
**Fix:** Acknowledge it explicitly in the report as a threat to validity. Treat BM25 as the only fully clean baseline. If feasible, evaluate on a T+1 that is after SPECTER2's training data cutoff (check the model card).

### L3 — "Cites paper" feedback signal is the label · High
**Where:** §10
**Problem:** "Cites paper → very strong positive" is exactly what LFO uses as ground truth. If any future citations feed the user model, it is direct label leakage.
**Fix:** In LFO mode, only citations made in papers with year ≤ T may feed the user profile. State this rule in §10 and §25.

### L4 — Popularity fallback leaks future popularity · Medium
**Where:** §23
**Problem:** A popularity cold-start fallback built from current citation counts uses future data.
**Fix:** Compute popularity from `counts_by_year` up to T (same rule as L1).

---

## C. Formula correctness

### F1 — Position-decay weight exceeds 1 for the newest item · High
**Where:** §15 step 1
**Problem:** w_i = exp(−δ(t − 1 − i)). For i = t: w_t = exp(δ) > 1; for i = t−1: w = 1. The exponent is off by one.
**Fix:** Use **w_i = exp(−δ(t − i))**, so the newest item gets weight 1 and older items decay. Normalize s_t (divide by Σw_i or L2-normalize) so it's comparable to p_u when fused.

### F2 — Novelty after MMR undoes diversity · High
**Where:** §18, §19.4, §20
**Problem:** §19.4 adds γ·N(p,u) *after* MMR. MMR produces a greedy ordered list; adding a score afterwards and re-sorting throws that ordering away.
**Fix:** Fold novelty into the relevance score **before** MMR: `rel(p) = Score(p) + w_nov · N(p,u)`, then run MMR on `rel`. Update §19.4, §20 and the §36 pipeline diagram.

### F3 — Bibliographic coupling has two definitions · Medium
**Where:** §16.1
**Problem:** Defined as "shares ≥ 3 references" and, in the same section, "via Jaccard similarity on reference lists". These give different sets.
**Fix:** Pick one. Recommended: `|Refs(p) ∩ Refs(U)| ≥ 3` (simple, explainable), where U = user's papers with year ≤ T. Optionally use Jaccard as a continuous score instead of a tier.

### F4 — Citation "recency" decay penalizes old papers · Medium
**Where:** §16.2
**Problem:** The text says recent *relationships* matter more, but the formula decays by the candidate's *publication age* (t_now − t_p). This systematically buries foundational papers, which researchers frequently cite. Also uses λ, which already means other things (C4).
**Fix:** Either decay by the age of the *interaction* (year the user cited/read the related paper), or rename it "candidate age decay" and justify it. Rename the parameter (e.g. `λ_cite`).

### F5 — Exact-match facets rarely fire · Medium
**Where:** §17.2 (S_facet), §19.2 (N_facet)
**Problem:** GLiNER returns verbatim spans ("BERT-based dual encoder", "dense passage retriever"). Exact string equality across papers will almost never match, so S_facet ≈ 0 and N_facet ≈ 1 for everything.
**Fix:** Normalize facets before matching: lowercase + lemmatize, then cluster/canonicalize spans by embedding similarity (e.g. a small sentence-embedding model + threshold), and match on canonical IDs. Report facet match rates in the §27.4 audit.

---

## D. Internal inconsistencies

### C1 — Session update defined two ways · High
**Where:** §9, §12 vs §15
**Problem:**
- §9/§12: EMA memory `m_t`, λ ≈ 0.5, **soft** reset `m_t = 0.85·d_t + 0.15·m_{t−1}` on pivot.
- §15: position-decayed sum `s_t`, δ ≈ 0.35, **hard** flush `s_t = d_t` on pivot.
- §37: λ ∈ [0.7, 0.95].
**Fix:** Choose one canonical session update for V-PaperFlow (the §15 position-decay form matches the "V-SKNN" framing). Keep the EMA form only as the PaperFlow baseline. Make §37's ranges match.

### C2 — Two different final scoring formulas · High
**Where:** §15 step 4 vs §17.5
**Problem:** §15: `S = cos(d_p, q_t) + β·Sim_facet`. §17.5: an 8-term linear combination with separate session, seed, persona, facet, cite, impact, RRF and negative terms. They can't both be "the" score.
**Fix:** Keep one. Recommended: §15 as the core V-PaperFlow score, plus a *small* set of extra terms (citation, maybe RRF) — see S1. Delete or mark the other as an ablation variant.

### C3 — Three different query definitions · High
**Where:** §5, §15 step 3, §17
**Problem:** §5: q = current query + session + profile. §15: q_t = (1−α)·s_t + α·p_u (no text query). §17 uses s_t, d_t and p_u as separate signals.
**Fix:** Define two explicit modes: **search mode** (text query → adhoc_query encoder + BM25, optionally blended with p_u) and **feed mode** (q_t from session + persona, no text). State which mode each evaluation uses (§26 = search, §28 = feed).

### C4 — Overloaded symbols · High
**Where:** §12, §15, §16.2, §18, §19, §28.4, §37
**Problem:**
- γ = pivot threshold (§12, §15), novelty weight (§19.4), citation decay (§37 `γ_cite`).
- λ = EMA decay (§12), citation decay (§16.2), MMR trade-off (§18, §37).
- w_m, w_d, w_t = facet-match weights (§17.2) **and** novelty weights (§19.2).
- §28.4 uses `w_cite`, `w_nov` which don't appear in §17.5 (`w_5`) or §19.4 (`γ`).
- §16 uses `S_citation`, §17 uses `S_cite` with a different definition.
**Fix:** Add a notation table (D4) with unique names: `γ_pivot`, `λ_ema`, `λ_mmr`, `λ_cite`, `δ_pos`, `α_persona`, `β_facet`, `w_nov`, `w^match_{m,d,t}`, `w^nov_{m,d,t}`. Use one name for the citation signal throughout.

### C5 — Baseline 2 defined two ways · High
**Where:** §14 vs §28.3
**Problem:** §14: "Pure PaperFlow (EMA memory + persona fusion)". §28.3: "static profile matching … without temporal position decay".
**Fix:** Define PaperFlow once: EMA session memory + gated reset + persona fusion, as in the original paper. If you also want a static-persona baseline, add it as a separate 4th arm.

### C6 — §37 "Fixed" list disagrees with the body · High
**Where:** §37 vs body
| Item | Body | §37 |
|------|------|-----|
| Profile schema | 7 fields (§8: domain, task, related prior work, method, dataset, metric, topic) | 5 facets |
| MMR | plain MMR (§18) | "facet-aware MMR" |
| Vector store | FAISS HNSW (§4.2), exact FAISS IP as baseline | NumPy/Torch vs FAISS FlatIP / Qdrant (tentative) |
| Session decay | λ ≈ 0.5 (§12), δ ≈ 0.35 (§15) | λ ∈ [0.7, 0.95] |
| User study | protocol "not yet fully specified" (§30); large studies out of scope (§35) | n = 50 "fixed" |
| Corpus | "at least 100,000" (§2.2) | "~100,000 CS/AI subset" |

**Fix:** Regenerate §37 from the body after resolving C1–C5. Move anything without a spec (user study) out of "Fixed".

### C7 — BM25 implementation inconsistent · Medium
**Where:** §3, §4.1 vs §26
**Problem:** §3/§4.1 name Pyserini/Anserini as current with bm25s as an aside; §26 and §37 assume bm25s.
**Fix:** Pick bm25s (pure Python, no JVM) and update §3 and §4.1.

---

## E. Pipeline gaps

### P1 — Feed mode has no text query · High
**Where:** §4, §6, §7.2, §36
**Problem:** BM25, RRF and the cross-encoder reranker all need a text query. In feed mode (and in all of LFO), there is only a vector q_t. §6 even says candidate retrieval in feed mode is driven by paper vectors. Feed-mode candidate generation is undefined, and the reranker has nothing to rerank against.
**Fix:** In feed mode: candidates = dense kNN on q_t (plus optional citation-neighbourhood candidates); **no BM25, no RRF, no cross-encoder**. Keep BM25 + RRF + reranker for search mode only. Redraw §36 with two branches.

### P2 — Schema facets GLiNER can't extract · Medium
**Where:** §8, §8.1
**Problem:** §8 lists 7 fields, but GLiNER is given only 5 labels. "Domain" and "related prior work" are not entity spans in an abstract (prior work is better taken from the OpenAlex reference list).
**Fix:** Fix the schema at 5 GLiNER facets (task, method, dataset, metric, topic). Take domain from OpenAlex topics/concepts and prior work from `referenced_works`.

### P3 — SciERC label mapping is loose · Medium
**Where:** §27.1
**Problem:** SciERC `Material` covers datasets *and* other resources; `OtherScientificTerm` is a catch-all, not "topic". F1 on these won't mean what the facet names suggest.
**Fix:** Keep the mapping but state the mismatch. Report per-label F1 and treat `topic` / `OtherScientificTerm` as low-confidence. The §27.4 in-corpus audit is the real test for dataset/topic quality.

### P4 — NLI on templates is near-tautological · Medium
**Where:** §22.2, §22.4
**Problem:** Template explanations are filled directly from extracted facets and evidence spans, so an NLI check against the same evidence will almost always pass. It only adds value if an LLM paraphrase is used — which is a stretch goal with cost = 0.
**Fix:** Make NLI verification conditional on paraphrasing. If paraphrasing is dropped, drop NLI (or keep it as an optional check of whether the *facet itself* is entailed by its evidence sentence).

### P5 — OpenAlex abstract handling · Medium
**Where:** §2, §24
**Problem:** OpenAlex stores abstracts as an inverted index (`abstract_inverted_index`), and many works have no abstract at all.
**Fix:** Write the inverted-index → text reconstruction step in the ingestion pipeline. Decide and document a policy for papers without abstracts (e.g. title-only embedding, or exclude from the corpus) and report the fraction affected.

### P6 — Stale LLM-era sections · Low
**Where:** §31.2, §32, §34
**Problem:** Caching keys and config tracking refer to "LLM extraction" and "prompt versions", but extraction is now GLiNER and there are no prompts unless paraphrasing is used.
**Fix:** Update caching to cover embeddings and GLiNER outputs (key: paper ID + model name/version + config hash). Drop prompt versioning unless paraphrasing is kept.

---

## F. Scope and feasibility

### S1 — Too many tuned weights · High
**Where:** §15, §17.5, §19, §37 vs §35
**Problem:** δ, γ, α, β, w_1…w_7, w_neg, w_m/w_d/w_t (×2), β_s, β_f, γ_nov, λ_MMR — 15+ parameters tuned with grid search / Nelder-Mead, while §35 rules out "extensive hyperparameter searches".
**Fix:** Ship a minimal model: **3 signals** (session/persona fused vector, citation, novelty) with **3 weights** tuned on the validation split. Fix all other parameters to stated defaults. Add extra signals only if they improve validation NDCG@10.

### S2 — LLM baseline won't run on the hardware · High
**Where:** §27.3, §8.6
**Problem:** The table lists ~16 GB VRAM for the Qwen baseline; the machine has an RTX 3050 (~4 GB).
**Fix:** Drop the LLM baseline, or use a ≤ 4B model in 4-bit quantization on a small SciERC subset (e.g. 100 abstracts) and report it as indicative only.

### S3 — n = 50 user study · High
**Where:** §30, §35, §37
**Problem:** No protocol, listed as "Fixed" in §37 while §35 puts large user studies out of scope, and no ethics/consent plan.
**Fix:** Reduce to ~10–15 participants with a scripted task (e.g. rate top-10 feed relevance and explanation helpfulness on a Likert scale), get consent forms, and check whether your institute requires ethics approval. Move it to "Tentative" until the protocol exists. Optional: seed each participant's profile from their OpenAlex/ORCID record.

### S4 — External API vs cost = 0 · Medium
**Where:** §22.3, §31.2, §31.3, §37
**Problem:** Explanation generation "may use an external API", but the budget is 0.
**Fix:** Remove the external API option. Paraphrasing (if kept) uses a small local model only.

---

## G. Missing rigor

### R1 — Whose history is Paper B? · High
**Where:** §16.4, §28.1
**Problem:** OpenAlex author disambiguation sometimes merges different people or splits one person into several IDs. Multi-author papers have no single "user" — Ref(B) reflects all co-authors.
**Fix:** One seed author per Paper B (first author, or single-author papers only). Filter authors with implausible histories (e.g. too many papers/year, multiple unrelated fields). Document the selection rule and the final author count.

### R2 — Self-citations inflate the citation signal · High
**Where:** §16.3, §16.4
**Problem:** A large part of Ref(B) is often the author's own earlier papers, which are trivially predictable. §16.3 dampens self-citation in scoring but not in the ground truth.
**Fix:** Remove the author's own papers from Ref(B), or report them as a separate split alongside re-citations/discoveries (E2).

### R3 — No statistics · High
**Where:** §28
**Problem:** No number of LFO users, no confidence intervals, no significance tests. Small differences across ~hundreds of authors may be noise.
**Fix:** Fix the evaluation set size up front (e.g. 300–500 authors). Report mean ± 95% bootstrap CI per metric, and use paired tests (paired bootstrap or Wilcoxon signed-rank) between arms.

### R4 — Missing simple baselines · High
**Where:** §28.3
**Problem:** No trivial baselines, so you can't tell whether the system beats something cheap.
**Fix:** Add:
- **Popular recent:** most-cited papers (counts up to T) in the author's topics.
- **Co-author papers:** recent papers by the author's co-authors.
- **Own past references:** re-recommend the author's previous references (relevant for the re-citation split).
- **BM25-only** and **SPECTER2 centroid** (mean of history embeddings).

### R5 — No compute budget or fallback · Medium
**Problem:** No estimate of embedding/extraction/indexing time on the RTX 3050, no plan for what is delivered if the schedule slips, no task split across team members.
**Fix:** Add a short plan: time estimates for SPECTER2 + GLiNER over 100k abstracts; a "minimum deliverable" (corpus + LFO harness + BM25/SPECTER2/V-PaperFlow results); owners per component.

### R6 — PaperFlow reference unverified · Medium
**Where:** §9, §14, benchmarking_dataset.md
**Problem:** PaperFlow is cited only as "arXiv:2606.07454" / "PaperFlow 2026" with no title or authors. The design depends heavily on it.
**Fix:** Confirm the paper exists and specifies its method (EMA, gating, persona fusion, benchmark) well enough to reimplement. Add the full citation (title, authors, year, venue/arXiv ID). If the details are not public, describe your implementation as "PaperFlow-style" and define it fully yourself.

---

## H. Documentation

### D1 — "No recommendations" header is false · Low
**Where:** header
**Fix:** Change to: "Records the current design and the rationale behind each decision."

### D2 — Broken code fences · Low
**Where:** §7.1, §7.3, §11, §21, §30, §31.3, §36
**Problem:** Blocks opened with a single backtick (`` ` text``) and never closed; the rest of each section renders as code.
**Fix:** Use triple backticks (```` ```text ```` … ```` ``` ````) and close every block.

### D3 — Draft text left in · Low
**Where:** §2.2 ("NEED TO BE ATLEAST 1,00,000"), §2.4/§2.5 ("so its closed"), §3/§4.1 (bold inline asides about Java)
**Fix:** Rewrite as statements: "Target corpus size: ≥ 100,000 papers." Use international digit grouping (100,000). Move implementation asides into the decision text (see C7).

### D4 — No notation table · Low
**Fix:** Add a table near the top listing every symbol, its meaning, its default value and where it's used. Resolves most of C4.

---

## Suggested fix order

1. **Write a one-page LFO evaluation spec** (resolves E1–E5, L1, L3, R1–R3): seed-author rule, self-citation handling, cutoff T, time-bounded graph and counts, corpus construction + coverage target, ground-truth splits (re-citation / discovery / self), number of authors, significance test.
2. **Reconcile the model definition** (C1–C5, F1–F2, P1, D4): one session update, one score, two query modes, notation table, fixed formulas.
3. **Cut scope** (S1–S4, P4): 3 signals / 3 weights, drop LLM baseline and external API, shrink user study.
4. **Fill in rigor** (R4–R6, L2, E6, E7): baselines, PaperFlow citation, V-SKNN rename, SPECTER2 threat-to-validity note, remove +33% claim.
5. **Clean the document** (C6, C7, P2, P3, P5, P6, F3–F5, D1–D3): regenerate §37 from the body, fix markdown and draft text.

Build order after the spec: corpus + LFO harness + BM25/SPECTER2 baselines (weeks 1–3) → GLiNER + SciERC (week 4) → ranker → ablations with CIs.
