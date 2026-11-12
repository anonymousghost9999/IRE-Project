"""First-stage retrieval benchmark: runs and metrics (Phase 7.2 / docs/plan.md §7.2 / docs/design.md §13.3).

Systems, each evaluated on its top ``EVAL_DEPTH`` (100, the recommender's candidate pool M):
- ``BM25``: BM25s over title + abstract (query: the question, or the query paper's title + abstract);
- ``SPECTER2``: exact dense kNN (query: ``adhoc_query`` vector for questions, ``proximity`` vector for papers);
- ``RRF``: Reciprocal Rank Fusion (k = 60) of both channels' top ``CHANNEL_DEPTH`` (200), as in the recommender;
- optional ``RRF + CE (<model>)``: the RRF list re-ranked by a cross-encoder (design §8.2), an Extended row.

Two modes:
- ``corpus``: retrieve from the benchmark's whole corpus (the query paper itself is excluded). Unjudged documents
  count as non-relevant. This is the first-stage candidate-generation question of Phase 7.
- ``pool`` (CSFCube only): the official protocol -- rank each query's annotated pool, every pool document ranked
  (documents a channel did not return are appended after the ranked ones, by ID). Adds CSFCube's official metrics
  (MAP, R-Precision, nDCG%20 with its discount) so numbers can be compared with published CSFCube results.

Metrics per query: Recall@k and MRR on binary relevance (grade >= the benchmark's threshold), nDCG@k with linear
graded gains and the standard 1/log2(rank + 1) discount, ideal ranking from all judged grades of the query.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from paperfeed.bench.retrieval import BenchmarkQuery, RetrievalBenchmark
from paperfeed.bench.retrieval_index import BenchmarkIndex, lexical_text
from paperfeed.common.logger import get_logger
from paperfeed.index.retriever import reciprocal_rank_fusion

logger = get_logger("bench.retrieval_eval")

EVAL_DEPTH = 100  # recommender candidate pool size M
CHANNEL_DEPTH = 200  # M_channel
RRF_K = 60
BASE_SYSTEMS = ("BM25", "SPECTER2", "RRF")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    rel = set(relevant)
    return len(rel & set(ranked[:k])) / len(rel) if rel else 0.0


def mrr(ranked: Sequence[str], relevant: Sequence[str]) -> float:
    rel = set(relevant)
    for i, d in enumerate(ranked, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    """Graded nDCG@k: gain = grade, discount 1/log2(rank + 1); ideal = all judged grades sorted, cut at k."""
    dcg = sum(qrels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted((g for g in qrels.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def csfcube_official_metrics(ranked_pool: Sequence[str], qrels: Mapping[str, int], threshold: int) -> Dict[str, float]:
    """MAP, R-Precision and nDCG%20 as computed by CSFCube's ``eval_scripts`` (rank_metrics.py, ranking_eval.py).

    ``ranked_pool`` is the whole annotated pool in ranked order. nDCG uses CSFCube's default discount (method 0:
    weights 1, 1, 1/log2(3), ...) and linear gains, with the ideal from the pool's own grades; nDCG%20 is cut at
    int(0.2 * pool size). MAP and R-Precision use binary relevance at ``threshold``. CSFCube's ``r_precision`` is
    not the textbook precision@R: it is the precision at the rank of the *last* relevant document (rank_metrics.py),
    and it is reproduced as such so the numbers stay comparable with published CSFCube results.
    """
    grades = [qrels.get(d, 0) for d in ranked_pool]
    binary = [1 if g >= threshold else 0 for g in grades]

    def dcg_method0(r: Sequence[int], k: int) -> float:
        r = np.asarray(r, dtype=float)[:k]
        if r.size == 0:
            return 0.0
        return float(r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1))))

    def ndcg_method0(r: Sequence[int], k: int) -> float:
        best = dcg_method0(sorted(r, reverse=True), k)
        return dcg_method0(r, k) / best if best else 0.0

    hits, precisions, last = 0, [], 0
    for i, b in enumerate(binary, start=1):
        if b:
            hits += 1
            precisions.append(hits / i)
            last = i
    return {
        "map": float(np.mean(precisions)) if precisions else 0.0,
        "r_precision": (hits / last) if last else 0.0,  # CSFCube definition: precision at the last relevant rank
        "ndcg_pct20": ndcg_method0(grades, int(0.20 * len(grades))),
    }


def query_metrics(ranked: Sequence[str], q: BenchmarkQuery, threshold: int) -> Dict[str, float]:
    relevant = q.relevant(threshold)
    return {
        "recall_at_5": recall_at_k(ranked, relevant, 5),
        "recall_at_10": recall_at_k(ranked, relevant, 10),
        "recall_at_20": recall_at_k(ranked, relevant, 20),
        "recall_at_100": recall_at_k(ranked, relevant, 100),
        "ndcg_at_10": ndcg_at_k(ranked, q.qrels, 10),
        "ndcg_at_100": ndcg_at_k(ranked, q.qrels, 100),
        "mrr": mrr(ranked, relevant),
    }


def bootstrap_ci(values: Sequence[float], n_resamples: int = 10000, seed: int = 42, alpha: float = 0.05):
    """Percentile bootstrap CI of the mean over queries."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_resamples, v.size))].mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def query_text(bench: RetrievalBenchmark, q: BenchmarkQuery) -> str:
    """Lexical / cross-encoder query text: the question, or the query paper's title + abstract."""
    return q.text if q.kind == "text" else lexical_text(bench.corpus[q.doc_id])


def _complete_pool(ranked: List[str], pool: Sequence[str]) -> List[str]:
    """Pool mode ranks every pool document: unreturned ones go last, in ID order (deterministic)."""
    seen = set(ranked)
    return ranked + sorted(d for d in pool if d not in seen)


def rank_query(
    bench: RetrievalBenchmark,
    idx: BenchmarkIndex,
    q: BenchmarkQuery,
    mode: str = "corpus",
    eval_depth: int = EVAL_DEPTH,
    channel_depth: int = CHANNEL_DEPTH,
    rrf_k: int = RRF_K,
) -> Dict[str, List[str]]:
    """Ranked document IDs of BM25, SPECTER2 and RRF for one query."""
    if mode not in ("corpus", "pool"):
        raise ValueError(f"Unknown mode {mode!r}")
    exclude = {q.doc_id} if q.doc_id else set()
    if mode == "pool":
        if not q.pool:
            raise ValueError(f"{bench.name}: query {q.qid} has no annotated pool for pool mode.")
        pool = [d for d in q.pool if d not in exclude]
        allowed, depth, chan = set(pool), len(pool), len(pool)
    else:
        pool, allowed, depth, chan = None, None, eval_depth, channel_depth

    def lexical(k: int) -> List[str]:
        return [d for d, _ in idx.lexical.search(query_text(bench, q), k=k, exclusion_set=exclude, candidate_filter=allowed)]

    def dense(k: int) -> List[str]:
        return [d for d, _ in idx.dense.search(idx.query_vectors[q.qid], k=k, exclusion_set=exclude, candidate_filter=allowed)]

    bm25_chan, dense_chan = lexical(chan), dense(chan)
    runs = {
        "BM25": bm25_chan[:depth],
        "SPECTER2": dense_chan[:depth],
        "RRF": [d for d, _ in reciprocal_rank_fusion([dense_chan, bm25_chan], k_rrf=rrf_k)][:depth],
    }
    if pool is not None:
        runs = {name: _complete_pool(r, pool) for name, r in runs.items()}
    return runs


class CrossEncoderReranker:
    """Re-ranks a candidate list with a sentence-transformers cross-encoder (design §8.2, Extended)."""

    def __init__(self, model_name: str, batch_size: int = 32, max_length: int = 512, device: Optional[str] = None):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)

    def rerank(self, query: str, doc_ids: Sequence[str], docs: Mapping[str, Mapping[str, str]]) -> List[str]:
        if not doc_ids:
            return []
        scores = self.model.predict([(query, lexical_text(docs[d])) for d in doc_ids], batch_size=self.batch_size,
                                    show_progress_bar=False)
        order = sorted(range(len(doc_ids)), key=lambda i: (-float(scores[i]), i))  # ties keep the RRF order
        return [doc_ids[i] for i in order]


@dataclass
class BenchmarkRun:
    benchmark: str
    mode: str
    systems: List[str]
    per_query: Dict[str, Dict[str, Dict[str, float]]]  # system -> qid -> metric -> value
    facets: Dict[str, Optional[str]]  # qid -> facet (CSFCube) or None
    rankings: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)  # system -> qid -> top EVAL_DEPTH ids

    def aggregate(self, facet: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Mean of every metric per system, over all queries or one facet's queries."""
        out: Dict[str, Dict[str, float]] = {}
        for system in self.systems:
            rows = [m for qid, m in self.per_query[system].items() if facet is None or self.facets.get(qid) == facet]
            if rows:
                out[system] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
                out[system]["n_queries"] = float(len(rows))
        return out


def run_benchmark(
    bench: RetrievalBenchmark,
    idx: BenchmarkIndex,
    mode: str = "corpus",
    rerankers: Sequence[CrossEncoderReranker] = (),
    progress: Optional[Callable[[int, int], None]] = None,
) -> BenchmarkRun:
    """Run every system on every query and score it."""
    threshold = bench.relevance_threshold
    systems = list(BASE_SYSTEMS) + [f"RRF + CE ({r.model_name})" for r in rerankers]
    per_query: Dict[str, Dict[str, Dict[str, float]]] = {s: {} for s in systems}
    rankings: Dict[str, Dict[str, List[str]]] = {s: {} for s in systems}
    for n, q in enumerate(bench.queries, start=1):
        runs = rank_query(bench, idx, q, mode)
        for r in rerankers:
            runs[f"RRF + CE ({r.model_name})"] = r.rerank(query_text(bench, q), runs["RRF"], bench.corpus)
        for system, ranked in runs.items():
            m = query_metrics(ranked, q, threshold)
            if mode == "pool":
                m.update(csfcube_official_metrics(ranked, q.qrels, threshold))
            per_query[system][q.qid] = m
            rankings[system][q.qid] = list(ranked[:EVAL_DEPTH])
        if progress:
            progress(n, len(bench.queries))
    return BenchmarkRun(bench.name, mode, systems, per_query, {q.qid: q.facet for q in bench.queries}, rankings)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def format_main_table(run: BenchmarkRun, n_resamples: int = 10000) -> str:
    """One row per system: Recall@100 (95% bootstrap CI over queries), nDCG@10/@100, MRR, Recall@5/@20."""
    agg = run.aggregate()
    lines = [
        "| System | Recall@100 [95% CI] | nDCG@10 | nDCG@100 | MRR | Recall@5 | Recall@20 | Queries |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for system in run.systems:
        m = agg[system]
        lo, hi = bootstrap_ci([v["recall_at_100"] for v in run.per_query[system].values()], n_resamples)
        lines.append(
            f"| {system} | {m['recall_at_100']:.4f} [{lo:.4f}, {hi:.4f}] | {m['ndcg_at_10']:.4f} | {m['ndcg_at_100']:.4f} "
            f"| {m['mrr']:.4f} | {m['recall_at_5']:.4f} | {m['recall_at_20']:.4f} | {int(m['n_queries'])} |"
        )
    return "\n".join(lines)


def format_csfcube_official_table(run: BenchmarkRun, facets: Sequence[str] = ("background", "method", "result")) -> str:
    """Pool mode: CSFCube's official MAP, R-Precision and nDCG%20 per facet and over all queries."""
    if run.mode != "pool":
        raise ValueError("Official CSFCube metrics exist only in pool mode.")
    header = "| System | " + " | ".join(f"{f}: MAP / R-Prec / nDCG%20" for f in [*facets, "all"]) + " |"
    lines = [header, "| :--- |" + " :---: |" * (len(facets) + 1)]
    per_facet = {f: run.aggregate(f) for f in facets}
    per_facet["all"] = run.aggregate()
    for system in run.systems:
        cells = []
        for f in [*facets, "all"]:
            m = per_facet[f].get(system)
            cells.append(f"{m['map']:.4f} / {m['r_precision']:.4f} / {m['ndcg_pct20']:.4f}" if m else "-")
        lines.append(f"| {system} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
