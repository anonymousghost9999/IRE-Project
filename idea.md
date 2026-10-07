# Paper Feed

## Core idea

Crawl arXiv/OpenAlex → Clean into a structured corpus → Index for retrieval → Extract structured paper profiles → Power a personalized, explainable recommender.

Explanations are generated **from the extracted paper profile**, not freely from the abstract. "Uses SPECTER embeddings, seen in 6 of your saved papers; evaluates on SciDocs, which is new to you" is checkable against the corpus. A free-text LLM explanation is not. This makes information extraction load-bearing, not decorative, and is the sharpest point of difference from the closest prior systems.

Three gaps closed at once:
- Vocabulary mismatch: hybrid lexical + dense retrieval, fused rather than one replacing the other.
- personalization: recommendations driven by what the user actually read and citation based, not author/venue subscriptions.
- Unexplained recommendations: explanations grounded in typed, verifiable profile facets.

## Existing SOTA methodologies:

| System | What it does | Where Paper Feed differs |
| --- | --- | --- |
| [PaperWeaver](https://dl.acm.org/doi/10.1145/3613904.3642196) (CHI 2024) | LLM infers interests from collected papers, writes free-text contextual descriptions on top of an *existing* recommender's output | Sits downstream of retrieval/ranking, reports no retrieval metrics, explanations are narrative not checkable |
| [Scholar Inbox](https://arxiv.org/abs/2504.08385) (ACL 2025 demo, deployed) | Learned rating-based personalization, author-search and a 2D "map of science" for cold start | Personalizes on a trained preference model over ratings, not on typed facets of read papers; no per-recommendation grounded explanation |
| [PaperFlow](https://arxiv.org/abs/2606.07454) (2026) | Three coupled stages - Profiling (structured, inspectable profile), Recommending (multi-signal ranking under a display budget), Adapting (interest drift across days) - with a 1,200-episode longitudinal benchmark | Closest prior art in structure. Its profile is LLM-inferred and inspectable but not evaluated against an annotated extraction ground truth; it reports oracle/behavioral-alignment and human-eval scores, not standard IR retrieval metrics (nDCG, Recall) on a BEIR-style benchmark |

The differentiation to lead with, updated: Paper Feed's profile schema is evaluated against SciERC/SciER human annotations (a number, not a demo), its explanations are template-generated from that typed schema and pass an automatic faithfulness check (see Recommendation section), and it reports offline retrieval-quality metrics that PaperFlow and Scholar Inbox do not.

## Retrieval & indexing

BM25 (Pyserini/Anserini) and dense retrieval fused by Reciprocal Rank Fusion, rather than assuming dense retrieval simply beats lexical. This is the right call - [SPECTER2's own authors report BM25 beating both SPECTER and SciNCL](https://arxiv.org/pdf/2211.13308) on the MDCR citation benchmark, so treat "does semantic search beat BM25 here" as an open empirical question the project answers, not an assumption.

Model choice, refined:
- Use SPECTER2's task-specific adapters as planned: proximity adapter for similar-paper retrieval, search adapter for query-to-paper. Don't force one embedding to do both.
- The project's queries are often short natural-language interest statements, not other papers. Recent evaluations show general-purpose long-context embedding models (e.g. BGE-M3, Jina v3/v4) are competitive with or ahead of SPECTER2's search adapter specifically on that query-to-paper format, while domain-specific citation-trained models (SPECTER2, SciNCL) still lead on paper-to-paper proximity. Test both as the query-side encoder rather than defaulting to SPECTER2 everywhere.
- Keep monoT5 cross-encoder reranking of the top-100 fused candidates, and HyDE for vague queries — both still current best practice and appropriately budget-bounded.
- Keep CSFCube, LitSearch, nDCG@10, Recall@100, MRR as the offline retrieval anchors; these remain the standard scientific-IR evaluation resources.
- Optional stretch, not required: SciMult (mixture-of-experts) is reported strongest on the citation-recommendation-style benchmark and is worth a look if Thread 2/3 are ahead of schedule.

## Information extraction & profiling

Make schema-guided LLM extraction the primary method, not PL-Marker. A single call per abstract against a strict JSON schema (task / method / dataset / metric fields, each with an evidence span), validated and retried on schema failure, is now the dominant, better-supported approach for open-ended scientific IE — recent benchmarks and case studies ([MOLE](https://arxiv.org/pdf/2505.19800), [LLMStructBench](https://arxiv.org/abs/2602.14743)) show this outperforms fixed-category NER pipelines on exactly this kind of flexible, multi-field extraction, without the training-data and fine-tuning overhead PL-Marker would need in one semester.

What stays, because it's what makes the module checkable rather than decorative:
- Evaluate the schema-guided extractor against the SciERC and SciER human-annotated test splits, reporting per-type precision/recall/F1 — same commitment as draft 1, same ground truth, different extractor.
- Keep GLiNER as a cheap, zero-shot, CPU-only pre-filter and fallback: cross-check its spans against the LLM's as an internal consistency signal, and fall back to it when LLM output fails schema validation twice.
- Keep scispaCy only as a lightweight sentence/entity-splitting utility, not as the primary extractor.
- Drop PL-Marker as the primary path; it's the right *comparison* point in a paper, not the right build choice for a 16-week team of three.

Cost note: this pass runs via API per-abstract rather than purely on local CPU. Budget it explicitly in the resources section rather than assuming it's free like the embedding inference calls.

## Recommendation & explanation

Keep draft 1's choice of SKNN as the primary session-aware recommender, with SASRec only as a stretch neural comparison. This is well-supported: Jannach & Ludewig and Ludewig & Jannach show simple session-based k-NN matches or beats GRU4Rec at scale with no training, and draft 2's GRU4Rec-first framing should not survive into the final plan. Keep MMR re-ranking and calibration for diversity/novelty — still the standard fix for the near-duplicate-feed failure mode.

Keep the grounded-explanation design exactly as drafted: template-built from typed profile facets, checkable against the corpus. Add one guardrail the 2025–2026 literature makes clear is still needed even for grounded, non-free-text pipelines: run a cheap automatic faithfulness check (NLI/entailment between the generated explanation sentence and the underlying profile record) before display. Recent audits of LLM-powered recommenders find a real gap between explanations that *look* compelling and explanations that are *actually* correct, even in pipelines that claim to be grounded — this is an inexpensive second-line check, not a new module.

Optional readability layer, only if time allows: an LLM paraphrase of the grounded facets into a more natural sentence, constrained to rephrase only the given facts (no new claims). This gets PaperWeaver-style readability without giving up the checkability that's the project's actual differentiator — don't let it become a substitute for the template.

## Evaluation plan

Keep draft 1's per-thread evaluation resources — this is the single biggest improvement over draft 2, which named no evaluation resources at all.

| Module | Metric | Resource |
| --- | --- | --- |
| Retrieval | nDCG@10, Recall@100, MRR | CSFCube, LitSearch |
| Extraction | Per-type P/R/F1; profile-usefulness rate | SciERC test split, 100 hand-checked profiles |
| Recommendation | Recall@10, nDCG@10, diversity, novelty | Held-out-reference proxy, simulated personas |
| Explanation | Faithfulness (+ automatic entailment score) | User study, n = 8–12 |

One addition: PaperFlow's longitudinal user-day benchmark design (24 simulated users, fixed candidate pools and temporal boundary, a blind human-evaluation protocol validated against automatic metrics) is a directly comparable, very recent methodology for a similar system. Thread 1 should read its evaluation appendix in weeks 1–2 and adapt what fits, rather than designing the simulated-persona protocol from scratch. Keep Beel & Langer's finding that offline and online evaluations often disagree as the reason a small user study stays in scope even though it's the team's weakest skill area.

## Threads, ownership & team-size justification

Draft 1's five-thread structure is sound and unchanged; only the method column moves.

| Thread | Core question | Owner | Evaluation resource |
| --- | --- | --- | --- |
| 1. Survey & scoping | What's been tried, incl. PaperFlow/Scholar Inbox; realistic corpus scope | All three, weeks 1–2 | Produces the eval design itself |
| 2. Data acquisition & cleaning | Clean, large-enough corpus from public APIs | Member A | 200 hand-labelled records; S2AND-style author P/R |
| 3. Retrieval & indexing | Does hybrid fusion beat BM25 and dense alone? | Member B | CSFCube, LitSearch; nDCG@10, Recall@100, MRR |
| 4. Extraction & profiling | Can schema-guided LLM extraction build a usable, checkable profile? | Member C | SciERC/SciER test split P/R/F1; profile-usefulness rate |
| 5. Recommendation & explanation | Does session-aware, facet-grounded recommendation beat static similarity, and is the explanation trustworthy? | Member C, supported by B | Held-out-reference proxy + personas; faithfulness score + user study |

Team-size justification is unchanged and still holds: two frozen interfaces (corpus schema by week 2, index + profile-facet schema by week 3) let threads 2, 3, and 4–5 run in genuine parallel. One person would have to serialize this across all 16 weeks and would likely drop the evaluation entirely; five would leave a thread idle waiting on the others, since the pipeline has three stages of independent complexity. Three remains the right size.

## Skills, hardware & resources

Draft 1's resourcing is realistic and mostly unchanged:
- Python, basic ML/NLP familiarity, no deep-learning training required — SKNN needs no training, and pretrained embedding models are used purely for inference.
- FAISS/hnswlib for the vector index; Pyserini/Anserini for a credible BM25 baseline; `ir_measures`/`pytrec_eval` for conventional metric computation.
- A standard laptop plus free-tier GPU (Colab/Kaggle) is sufficient for embedding inference; a 20k–50k paper corpus plus abstracts is a few hundred MB, comfortably within SQLite/local-storage limits.
- User-study design remains the one acknowledged skill gap — keep the study small (n = 8–12) and within-subjects.

One new budget line the drafts didn't need: the schema-guided LLM extraction pass is API inference per abstract, not local CPU inference like the embedding models. Decide and cost-test the extraction model choice in week 1–2 (a cheap/fast model for the full corpus pass, a stronger model reserved for the sampled SciERC/SciER validation and for cases that fail schema validation twice).

<!-- ## What was cut, and why -->
<!-- 
- **Draft 2's plain spaCy + keyphrase extraction**, because it has no ground truth attached and general NER doesn't distinguish a method from a dataset. Replaced by schema-guided LLM extraction validated against SciERC/SciER, with GLiNER as a cheap fallback.
- **GRU4Rec as the primary recommender** (draft 2's framing), because simple session-based k-NN reproducibly matches or beats it with no training cost. Kept only as historical context; SKNN is primary, SASRec is the stretch neural comparison.
- **Forcing one embedding to do both similarity and search**, because SPECTER2's own task-specific adapters exist precisely to avoid this, and general-purpose long-context embeddings are competitive specifically on the query-to-paper format.
- **The assumption that semantic retrieval beats BM25**, because the SPECTER2 authors themselves report BM25 winning on a citation-recommendation benchmark. Kept as an explicit empirical question the hybrid-fusion design answers rather than assumes.
- **Free-text LLM explanation as the sole design**, already correctly rejected in draft 1. 2025–2026 work on LLM-recommender hallucination reinforces this and adds the reason to keep an automatic faithfulness check even on the grounded, template-based design. -->

<!-- ## Open risks & decisions for weeks 1–2

- [ ] Read PaperFlow and Scholar Inbox in full and rewrite the novelty paragraph of whichever draft becomes the proposal so it differentiates from both, not only from PaperWeaver and arXiv Sanity.
- [ ] Pick and cost-test the schema-guided extraction model before Thread 4 commits to a schedule — this now gates a budget line the drafts didn't have.
- [ ] Decide the fusion vs. learned-reranker trade-off (RRF is simple and robust; monoT5 adds real cost) against the actual time budget once Thread 3 has a working baseline.
- [ ] Confirm the 20k–50k paper corpus target is still adequate if the team adopts a PaperFlow-style simulated longitudinal evaluation rather than a single static offline split. -->

## Sources

- [PaperWeaver, CHI 2024](https://dl.acm.org/doi/10.1145/3613904.3642196)
- [Scholar Inbox, ACL 2025 demo](https://arxiv.org/abs/2504.08385)
- [PaperFlow, 2026](https://arxiv.org/abs/2606.07454)
- [SciRepEval / SPECTER2 (BM25 vs. dense finding)](https://arxiv.org/pdf/2211.13308)
- [SPECTER2 model card](https://huggingface.co/allenai/specter2)
- [Embedding model benchmark, 2026](https://zc277584121.github.io/rag/2026/03/20/embedding-models-benchmark-2026.html)
- [SemCSE (SciRepEval comparison table)](https://arxiv.org/pdf/2507.13105)
- [MOLE: schema-guided scientific metadata extraction](https://arxiv.org/pdf/2505.19800)
- [LLMStructBench: structured-output benchmark](https://arxiv.org/abs/2602.14743)
- [Recommender hallucination / faithfulness auditing](https://arxiv.org/pdf/2608.10008)