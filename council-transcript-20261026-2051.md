# LLM Council Transcript — Paper Feed Design Review

**Date:** 2026-10-26 20:51
**Subject:** `IRE-Project/paper_feed_current_design.md` (+ context from `IRE-Project/benchmarking_dataset.md`)

---

## Original question

> review @IRE-Project/paper_feed_current_design.md using /llm-council

## Framed question

Review the design document for "Paper Feed", a personalized academic-paper recommender built as a semester course project for an IRE (Information Retrieval & Extraction) course by a small undergrad team at IIIT.

Constraints stated in the doc: OpenAlex-only corpus of ≥100k CS/AI papers; no model training; API budget = 0; RTX 3050 (~4 GB VRAM); SQLite + FAISS + BM25 (Pyserini or bm25s); pretrained models only (SPECTER2 proximity + adhoc_query, GLiNER, cross-encoder reranker, DeBERTa NLI); proposed recommender "V-PaperFlow" (V-SKNN session decay + PaperFlow persona/gated divergence); Leave-Future-Out evaluation on OpenAlex author timelines (references of the author's next paper as ground truth); retrieval eval on CSFCube/LitSearch/SCIDOCS; extraction eval on SciERC; n=50 user study.

**Question:** Is this design sound, internally consistent, and achievable within a semester? What are the most significant flaws, risks, or inconsistencies; what is missing; what is undervalued? What should the team change, cut, or prioritize before implementation goes further?

**Stakes:** The document drives implementation and the final report/grade. Evaluation design errors or contradictions discovered late would be expensive.

---

## Advisor responses

### The Contrarian

**Fatal flaw: the primary evaluation (§16.4/§28) does not measure the system you designed.**

1. **LFO has no sessions.** Author histories are yearly citation lists with no within-year order, so V-SKNN position decay (§15), the pivot gate (§12) and session timeout N (§11) are arbitrary on this data. The 3-way ablation (§28.3) tests session mechanics on sessionless data. The +33% PaperFlow-over-V-SKNN claim in benchmarking §5 comes from a simulation where "read" = matching a cluster — circular — and it also undercuts the hybrid.
2. **Confounded ablation.** The proposed model adds citation + facets + MMR (§28.3); the baselines get none, so any win can't be credited to "V-PaperFlow". Also, "V-SKNN" here is a decayed centroid — Jannach's V-SKNN is kNN over neighbouring *sessions*. Mislabelled baseline.
3. **Citation tiers fight the ground truth.** §16.4 removes already-read papers from GT and §19 hard-filters consumed papers, so the +2.0 "user cited p" tier only ever scores excluded items. Drop the exclusion and "re-recommend your old references" becomes a trivial baseline that beats everything.
4. **Leakage that §25 forbids but the data forces.** OpenAlex `cited_by_count` (S_impact) and co-citation are present-day snapshots that include Paper B and later work. Needs year-bounded reconstruction (`counts_by_year`, edges filtered by citing-paper year).
5. **Corpus coverage.** Much of Ref(B) will fall outside a 100k subset, and missing OpenAlex abstracts are common. Report the coverage ceiling first.

**Contradictions:** §12 soft reset (0.85/0.15) vs §15 hard reset; λ≈0.5 vs §37's [0.7,0.95]; γ means three things, w_m/w_d/w_t two; the §15 decay exponent gives the latest item weight e^δ > 1; novelty is added *after* MMR (§19.4), re-sorting and undoing diversity; grid/Nelder-Mead over ~15 weights (§37) violates §35 "no extensive searches"; the 16 GB LLM baseline (§27.3) won't run on a 4 GB GPU; the reranker has no query in feed mode; exact-match facet indicators on verbatim GLiNER spans will almost never fire.

**Cut:** n=50 study (→ ~10), NLI verification, LLM baseline, reranker. Fix the LFO protocol before writing more code.

### The First Principles Thinker

**You're asking whether the design is sound; the real question is what the project's central claim is and whether your evaluation can measure it. Right now it can't.**

1. **Your ground truth can't measure your main idea.** V-PaperFlow (§14–15) is a model of reading sessions: position decay, pivot gating, "N minutes of inactivity" (§11), the §10 feedback signals. LFO (§16.4, §28.1) has none of that — no reading events, no within-year order, no sessions, skips or likes. An author's history is a bag of year-stamped papers. s_t, Δ_t and γ are never really exercised; any ablation gap comes from how you happened to order the papers.
2. **Your evaluation is citation prediction.** Predicting Ref(B) is citation recommendation, which §35 lists as out of scope as the primary task. And the §16.1 citation signal is built from the same graph as the ground truth — it will "win" partly because it is close to the label.
3. **The feed has no query.** §4–5 assume a text query for BM25/RRF. In LFO mode there is only a vector q_t. What does BM25 search with? Feed-mode candidate generation is undefined.
4. **Most answers may not be in your corpus.** Build the corpus outward from the evaluation authors' citation neighbourhoods and report what share of ground truth is reachable.

**What to do:** Write one hypothesis LFO can test, e.g. "faceted persona + citation-graph signals beat pure SPECTER2 similarity for next-paper reference recommendation", and present the method as that. Either cut session gating or test it honestly with simulated sessions (benchmarking_dataset.md) labelled as simulated. Settle §15 vs §17.5 scoring and §12 vs §15 session updates before writing any code.

### The Expansionist

**You're underselling what you built.** V-PaperFlow (§14–15) is framed as the contribution, but it's a weighted cosine blend any grader can reproduce in an afternoon. The valuable assets are buried.

1. **The LFO benchmark is the real contribution (§16.4, §28.1).** Freeze author timelines, the T=2023 cutoff, held-out Ref(B) sets and the corpus snapshot; release as "OpenAlex-CS-LFO" with a dataset card. §33 already promises reuse. A released benchmark outlives the grade.
2. **Split ground truth into re-citations vs discoveries.** Ref(B) mixes papers the author already cited with first-time citations. Report separately — the +2.0 Citations(u) tier dominates re-cites; discoveries are where novelty (§19) and facet coverage (§28.2) show up. Turns one leaderboard number into a finding.
3. **Evaluate pivot detection on real pivots.** Authors whose Paper B moves away from their history are natural labels for the γ gate. Measure how often the gate fires on pivoting vs stable authors.
4. **Onboard via OpenAlex/ORCID ID.** Closes cold start (§23) — Citations(u), BibCoupling and p_u come straight from the record. The n=50 study (§30) can use IIIT faculty/PhDs with feeds built from their real records, so Likert ratings align with offline LFO personas.
5. **Facets as query operators, not just score boosts.** "Same task, different method", "uses dataset X". CSFCube (§26.1) is already facet-conditioned, so it's free evaluation.
6. **Make NLI-verified explanation rate a headline metric (§22.4).**

**Priority:** 1–3 first — near-zero cost beyond existing eval code; turns the project into "a benchmark plus findings".

### The Outsider

**Reading this cold, like a TA would, the document contradicts itself before any technical judgment.**

1. **"V-PaperFlow" is defined three ways.** §12: EMA m_t, λ≈0.5, soft 0.85/0.15 reset. §15: s_t, position decay δ, hard flush. §37: λ∈[0.7,0.95]. Baseline 2 also shifts: §14 "EMA + persona" vs §28.3 "static profile, no decay". Pick one equation and delete the others.
2. **The core ablation has no sessions.** LFO users are OpenAlex authors with year-stamped papers — no clicks, no order, no inactivity (§11). What is V-SKNN's "position" over?
3. **The citation signal works against the evaluation.** §16.1 gives +2.0 to already-cited papers; §16.4 removes already-read papers from GT; §19 hard-filters consumed papers. The strongest signal can only point at excluded items.
4. **One symbol, several meanings.** γ = pivot threshold, novelty weight, citation decay. λ = EMA decay, citation decay, MMR trade-off. w_m/w_d/w_t = facet-match and novelty weights.
5. **§37 "Fixed" list disagrees with the body.** 7 vs 5 profile fields; plain (§18) vs "facet-aware" MMR; FAISS-HNSW (§4.2) vs NumPy/Qdrant; n=50 "fixed" with no protocol while §35 puts large studies out of scope; cost=0 yet external API in §31.2; 16 GB LLM baseline on a 4 GB GPU.
6. **Presentation.** "No recommendations" header contradicted by rationale throughout; single-backtick code blocks never closed (§§7, 11, 21, 30, 31); "1,00,000" and "NEED TO BE ATLEAST" draft text; PaperFlow cited only as arXiv:2606.07454 with no title/authors.

**Fix:** one notation table, one canonical scoring equation, one honest status list — and write down exactly how an LFO author becomes a user with a session.

### The Executor

**What can you actually build on Monday?**

1. **Build the eval harness first — §28 can't run as written.** LFO gives author papers + reference lists, not reading sessions or order. V-SKNN position decay, gated divergence (§12/§15) and P_skip have no input data. Decide Monday: sequence by cited-work publication date and say it's a proxy, or demote V-SKNN.
2. **Pick authors first, then crawl.** A generic 100k crawl (§2.2) misses most of Ref(B) → Recall@10 ≈ 0. Choose ~500 authors who published in 2023–24; pull their references + 1–2 hops; fill to 100k; report Ref(B) coverage before modelling.
3. **Close leakage in day-one code.** S_impact, CoCitation (§16.1) and `cited_by_count` include post-T citations. Use `counts_by_year`; build the graph from ≤T papers only. OpenAlex abstracts are inverted indexes — write reconstruction now.
4. **Cut:** the cross-encoder from the feed path (no query); most tuning (15+ weights vs §35) — ship 3 signals (session, persona, citation), tune 3 weights; the n=50 study → 10–15 people with a scripted task, or drop; LLM paraphrasing and API fallback (cost = 0).
5. **Reconcile before coding.** §12 vs §15 (soft vs hard reset; λ≈0.5 vs [0.7,0.95]). Confirm PaperFlow (arXiv:2606.07454) exists and is specified precisely enough to reimplement.

**Order:** corpus + LFO harness + BM25/SPECTER2 baselines (wk 1–3) → GLiNER + SciERC (wk 4) → ranker → ablations last.

---

## Peer reviews

**Anonymization mapping:** A = Executor, B = Outsider, C = Contrarian, D = Expansionist, E = First Principles Thinker

**Vote tally:** Strongest — C (Contrarian) 5/5, A (Executor) named runner-up 5/5. Biggest blind spot — D (Expansionist) 5/5.

### Reviewer 1
- **Strongest: C** — names the fatal flaw and adds unique catches: confounded ablation, mislabelled V-SKNN, circular +33%, e^δ>1, novelty-after-MMR, exact-match facets, trivial re-recommend baseline. A close behind with best action order.
- **Blind spot: D** — praises LFO as release-worthy without noticing no sessions, post-cutoff leakage, or low coverage. Releasing a leaky benchmark publicizes the problems.
- **All missed:** OpenAlex author disambiguation is error-prone and Ref(B) belongs to all co-authors (one seed author per paper); no significance tests/CIs/variance; no popularity or recent-co-citation baselines; no consent/ethics for the user study; no division of work across team vs grading milestones.

### Reviewer 2
- **Strongest: C**; A close behind (author-first crawl, inverted-index abstracts).
- **Blind spot: D** — adds scope (dataset release, ORCID, facet operators, n=50 with faculty) while ignoring missing sessions, post-T leakage, contradictions, and semester time. Re-cite/discovery split is good but depends on an unfixed harness.
- **All missed:** noisy author timelines (merged/split profiles); Ref(B) full of self-citations inflating citation signals; no LFO sample size, CIs or paired tests; no compute costing of GLiNER + NLI over 100k on 4 GB VRAM; no fallback deliverable; ethics approval.

### Reviewer 3
- **Strongest: C**; A second (most actionable).
- **Blind spot: D**; also B catalogues defects well but never prioritizes or cuts.
- **All missed:** (1) SPECTER2 leakage — trained on citation triplets, embeddings may encode post-T links including Paper B's references; the "pure SPECTER2" baseline is contaminated, only BM25 is clean. (2) Whose persona is Paper B — restrict to first/single authors, drop self-cites. (3) No popularity/co-author/recent-references baselines; no significance testing across ~500 authors.

### Reviewer 4
- **Strongest: C**; A second (easiest fixes to act on).
- **Blind spot: D** — "release the LFO benchmark" isn't credible while it leaks and can't exercise session mechanics. Re-cite vs discovery split is useful.
- **All missed:** model-level leakage — SPECTER2 trained on S2 citation data likely including 2023–24; a data-only cutoff doesn't close it. No sample-size, paired tests, CIs. Missing baselines: "recent popular papers before T", "papers by co-authors". IRB/consent.

### Reviewer 5
- **Strongest: C**; A second (clearest week-by-week order).
- **Blind spot: D** — praises broken parts, adds scope (ORCID, facet operators, NLI headline, keeps n=50), ignores leakage, §12/§15 contradictions, semester capacity.
- **All missed:** author identity unreliable (disambiguation; multi-author papers have no clear user); self-citations in Ref(B) are trivially predictable and must be filtered; paired significance tests with CIs across authors plus a popularity/recency baseline.

---

## Chairman's synthesis

### Where the Council Agrees

- **The main evaluation does not test the system you designed.** Four of five advisors raised this on their own, and every reviewer ranked the Contrarian's version of it the strongest point. LFO author timelines have no sessions, no reading order, no skips and no likes. So V-SKNN position decay, the pivot gate γ and the session timeout never actually run. Any ablation gap between them would come from whatever ordering you happen to impose.
- **The citation signal is mostly testing itself.** The +2.0 "user cited p" tier can only point at papers that §16.4 and §19 already remove from the ground truth. The citation graph is built from the same data as the labels. In practice the evaluation is citation prediction, which §35 says is out of scope as the primary task.
- **Future data leaks into the past.** `cited_by_count` and co-citation are today's snapshots, so they already include Paper B and everything after it. Rebuild the graph from papers up to time T only, and use `counts_by_year`.
- **Corpus coverage is unchecked.** Much of Ref(B) will fall outside an arbitrary 100k subset, and missing abstracts are common. Build the corpus outward from the eval authors' citation neighbourhoods, then report coverage.
- **The doc contradicts itself and must be reconciled before coding.** Examples:
  - The session update differs between §12 and §15.
  - λ is ≈0.5 in one place and [0.7, 0.95] in another.
  - γ and the w_* weights each mean two or three different things.
  - The §15 decay exponent gives the latest item a weight above 1.
  - Novelty is added after MMR, which re-sorts the list and undoes the diversity step.
  - A 16 GB LLM baseline is planned for a 4 GB GPU.
  - Feed mode has no text query, so BM25/RRF candidate generation and the reranker have no input.
- **Cut scope.** Shrink the n=50 user study to about 10–15 people. Take the cross-encoder out of the feed path. Drop the LLM baseline. Tune about 3 weights, not about 15.

### Where the Council Clashes

The Expansionist says your LFO benchmark is the real contribution: release it, add ORCID onboarding, add facet operators, and run the study with faculty. All five reviewers called this the biggest blind spot. Releasing a benchmark that leaks and has low coverage just puts its flaws on public display.

The disagreement is reasonable, though. The Expansionist is right that the *idea* (next-paper references as ground truth, from real author timelines) is the most defensible part of the project. Two of their suggestions survive:
- Report re-citations and discoveries separately.
- Use real topic pivots in Paper B to label the γ gate.

The second one is the only cheap way to test the pivot gate on non-simulated data. The real question is ordering: fix the evaluation harness first, then add these.

### Blind Spots the Council Caught

These came out of peer review; no advisor raised them:
- **SPECTER2 likely leaks too.** It was trained on Semantic Scholar citation data that probably includes 2023–24 citations, possibly including Paper B's references. That would make the "pure SPECTER2" baseline contaminated. BM25 is the only clean baseline. Say this openly in the report.
- **Whose history is Paper B?** OpenAlex author records are sometimes wrongly merged or split, and a multi-author paper has no single "user". Use first or single authors, one seed author per paper.
- **Self-citations in Ref(B)** are easy to predict and inflate the citation signals. Filter them out or report them separately.
- **No statistics.** The doc doesn't say how many LFO users there are, and has no confidence intervals or paired significance tests across authors.
- **Missing obvious baselines.** Add popular recent papers before T, co-author papers, and "re-recommend the author's own past references".
- **Other gaps.** No ethics or consent for the user study, no compute estimate for running GLiNER+NLI over 100k papers on 4 GB VRAM, and no fallback deliverable or division of work among the team.

### The Recommendation

The design isn't sound as written, but it can be saved in a semester if you narrow the claim.

1. Rewrite the main hypothesis into something LFO can test, along the lines of: "faceted persona plus time-bounded citation-graph signals beat SPECTER2 and BM25 similarity for predicting next-paper references, especially for discoveries."
2. Either demote session gating, or test it only with simulated sessions clearly labelled as simulated. You could order items by cited-work publication date, but state that it's a proxy.
3. Rename V-SKNN or implement real neighbour-session kNN; as written it's a decayed centroid, not Jannach's method.
4. Give the baselines the same candidate pool and MMR as your model, so the ablation isn't confounded.
5. Verify that the PaperFlow paper (arXiv:2606.07454) exists and is specified well enough to reimplement, before depending on it.

Build in this order:
- Weeks 1–3: corpus, LFO harness and BM25/SPECTER2 baselines.
- Week 4: GLiNER with SciERC evaluation.
- Then: ranker with 3 signals.
- Last: ablations, plus statistics with confidence intervals.

Treat NLI verification and the user study as optional extras. Add one notation table and one canonical scoring equation, and fix the draft-quality presentation: the "NEED TO BE ATLEAST" text and the unclosed code blocks.

### The One Thing to Do First

Write the LFO evaluation spec as one page and settle it before writing any model code. It should fix:
- how an author becomes a user (seed author rule, self-citation filter);
- the cutoff T and the time-bounded graph and count features;
- the corpus construction and target coverage of Ref(B);
- the ground-truth split into re-citations and discoveries;
- the number of authors and the significance test.

Everything else in the design depends on what this harness can actually measure.
