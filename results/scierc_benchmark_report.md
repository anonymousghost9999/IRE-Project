# SciERC Extraction Benchmark Report (Phase 3.4)

> **Dataset:** SciERC (Luan et al., 2018) Official Evaluation Split  
> **Total Documents:** 100  
> **Total Gold Entities (Evaluated Labels):** 1444  
> **Total Extracted Entities:** 866  

## 1. Summary Performance Metrics

| Metric Track | Strict Precision | Strict Recall | Strict F1 | Relaxed Precision | Relaxed Recall | Relaxed F1 |
|---|---|---|---|---|---|---|
| **Micro-Average (All 5 Labels)** | 34.18% | 20.50% | **25.63%** | 44.92% | 26.94% | **33.68%** |
| **Macro-Average (All 5 Labels)** | 35.70% | 27.97% | **25.49%** | 46.51% | 36.44% | **33.44%** |
| **Micro-Average (Core 3 Labels: Task, Method, Metric)** | 36.59% | 26.96% | **31.05%** | 48.49% | 35.73% | **41.15%** |
| **Macro-Average (Core 3 Labels: Task, Method, Metric)** | 37.84% | 33.22% | **28.91%** | 50.92% | 42.43% | **37.72%** |

---

## 2. Per-Label Breakdown

### A. Strict Span Matching (Exact Token Boundaries)

| Label | Gold Count | Pred Count | TP | FP | FN | Precision | Recall | Strict F1 |
|---|---|---|---|---|---|---|---|---|
| `Task` | 263 | 50 | 24 | 26 | 239 | 48.00% | 9.13% | **15.34%** |
| `Method` | 430 | 352 | 141 | 211 | 289 | 40.06% | 32.79% | **36.06%** |
| `Metric` | 71 | 161 | 41 | 120 | 30 | 25.47% | 57.75% | **35.34%** |
| `Material` † | 158 | 208 | 52 | 156 | 106 | 25.00% | 32.91% | **28.42%** |
| `OtherScientificTerm` † | 522 | 95 | 38 | 57 | 484 | 40.00% | 7.28% | **12.32%** |

† *Note: `Material` and `OtherScientificTerm` in SciERC are broader than production `dataset` and `topic` facets (docs/design.md §13.4); treat these as low-confidence comparisons.*

### B. Relaxed Span Matching (Overlapping Token Boundaries)

| Label | Gold Count | Pred Count | TP | FP | FN | Precision | Recall | Relaxed F1 |
|---|---|---|---|---|---|---|---|---|
| `Task` | 263 | 50 | 34 | 16 | 229 | 68.00% | 12.93% | **21.73%** |
| `Method` | 430 | 352 | 189 | 163 | 241 | 53.69% | 43.95% | **48.34%** |
| `Metric` | 71 | 161 | 50 | 111 | 21 | 31.06% | 70.42% | **43.10%** |
| `Material` † | 158 | 208 | 74 | 134 | 84 | 35.58% | 46.84% | **40.44%** |
| `OtherScientificTerm` † | 522 | 95 | 42 | 53 | 480 | 44.21% | 8.05% | **13.61%** |

† *Note: `Material` and `OtherScientificTerm` in SciERC are broader than production `dataset` and `topic` facets (docs/design.md §13.4); treat these as low-confidence comparisons.*

---

## 3. Threshold Sweep Analysis

| Threshold | Strict Precision | Strict Recall | Strict F1 | Relaxed Precision | Relaxed Recall | Relaxed F1 |
|---|---|---|---|---|---|---|
| `0.30` | 27.83% | 27.42% | **27.62%** | 37.46% | 36.91% | **37.18%** |
| `0.40` | 30.64% | 23.89% | **26.85%** | 41.39% | 32.27% | **36.26%** |
| `0.50` | 34.18% | 20.50% | **25.63%** | 44.92% | 26.94% | **33.68%** |
| `0.60` | 37.08% | 16.69% | **23.02%** | 47.85% | 21.54% | **29.70%** |
| `0.70` | 42.06% | 13.57% | **20.52%** | 54.08% | 17.45% | **26.39%** |

---

## 4. Notes & Methodology

> [!NOTE]
> - **Matching Criteria:**
>   - **Strict Matching:** Requires exact token boundary match `(start_token, end_token)` and label match, using multiset (1-to-1) counting.
>   - **Relaxed Matching:** Requires any token overlap (> 0 tokens) and label match, resolved via greedy 1-to-1 bipartite matching prioritized by descending IoU.
> - **Label Mapping:** `task -> Task`, `method -> Method`, `metric -> Metric`, `dataset -> Material`, `topic -> OtherScientificTerm`.
> - **Ontology Nuance:** As noted in `docs/design.md §13.4`, `Material` and `OtherScientificTerm` in SciERC are broader than our production `dataset` and `topic` facets.
> - **SciERC `Generic` Category:** SciERC includes anaphoric/pronoun references (`Generic`, e.g. *'this method'*), which are omitted from scientific entity extraction.
