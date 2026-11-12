"""First-stage retrieval benchmarks: datasets (Phase 7.1 / docs/plan.md §7.1 / docs/design.md §13.3).

Two public benchmarks, each with its own corpus, normalized to one format:

- **CSFCube** (Mysore et al., NeurIPS 2021 Datasets & Benchmarks): 50 query papers x facet (background / method /
  result) with graded relevance 0-3 for an annotated pool of 88-253 candidates per query. Queries are papers, so they
  are embedded with the SPECTER2 ``proximity`` adapter. The official protocol re-ranks each query's pool and counts
  grade >= 2 as relevant; the released corpus is the 4,207 papers of all pools plus the queries.
- **LitSearch** (Ajith et al., EMNLP 2024): 597 natural-language literature-search questions, binary relevance, a
  64,183-paper corpus (``corpus_clean``). Queries are text, so they use the ``adhoc_query`` adapter.

On disk, ``<out_dir>/<name>/`` holds ``benchmark.json`` (metadata), ``corpus.jsonl`` (doc_id, title, abstract) and
``queries.jsonl`` (qid, kind, text or doc_id, facet, qrels, optional pool, meta). Every document and query ID is a
string. Raw downloads are parsed as data only (``ast.literal_eval`` for CSFCube's stringified sentence lists).
"""

from __future__ import annotations

import ast
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from paperfeed.common.logger import get_logger

logger = get_logger("bench.retrieval")

CSFCUBE_FACETS = ("background", "method", "result")
CSFCUBE_RELEVANCE_THRESHOLD = 2  # official eval script: grade >= 2 counts as relevant for binary metrics
LITSEARCH_RELEVANCE_THRESHOLD = 1

CSFCUBE_BASE_URL = "https://raw.githubusercontent.com/iesl/CSFCube/master"
CSFCUBE_FILES = (
    "abstracts-csfcube-preds.jsonl",
    "queries-release.csv",
    "evaluation_splits.json",
    *(f"test-pid2anns-csfcube-{f}.json" for f in CSFCUBE_FACETS),
    "LICENSE.md",
)
LITSEARCH_BASE_URL = "https://huggingface.co/datasets/princeton-nlp/LitSearch/resolve/main"
LITSEARCH_FILES = {
    "query.parquet": "query/full-00000-of-00001.parquet",
    **{f"corpus_clean_{i}.parquet": f"corpus_clean/full-0000{i}-of-00006.parquet" for i in range(6)},
}


@dataclass
class BenchmarkQuery:
    qid: str
    kind: str  # "paper" (query-by-example: doc_id is the query paper) or "text" (natural-language query)
    qrels: Dict[str, int]  # doc_id -> relevance grade (unlisted documents count as 0)
    text: str = ""
    doc_id: Optional[str] = None
    facet: Optional[str] = None
    pool: Optional[List[str]] = None  # annotated candidate pool (CSFCube); None when the whole corpus is judged as 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def relevant(self, threshold: int) -> List[str]:
        return [d for d, g in self.qrels.items() if g >= threshold]


@dataclass
class RetrievalBenchmark:
    name: str
    corpus: Dict[str, Dict[str, str]]  # doc_id -> {"title", "abstract"}
    queries: List[BenchmarkQuery]
    relevance_threshold: int
    query_adapter: str  # SPECTER2 adapter for the queries: "proximity" or "adhoc_query"
    source: str = ""
    license: str = ""

    def check(self) -> None:
        """Every judged, pooled or query document must be in the corpus."""
        missing = sorted(
            {d for q in self.queries for d in [*q.qrels, *(q.pool or []), *([q.doc_id] if q.doc_id else [])]}
            - set(self.corpus)
        )
        if missing:
            raise ValueError(f"{self.name}: {len(missing)} judged/pooled/query documents are not in the corpus, e.g. {missing[:3]}")
        for q in self.queries:
            if q.kind == "paper" and not q.doc_id:
                raise ValueError(f"{self.name}: paper query {q.qid} has no doc_id")
            if q.kind == "text" and not q.text.strip():
                raise ValueError(f"{self.name}: text query {q.qid} is empty")
            if not q.relevant(self.relevance_threshold):
                logger.warning("%s: query %s has no relevant document at threshold %d.", self.name, q.qid, self.relevance_threshold)


# ---------------------------------------------------------------------------
# CSFCube
# ---------------------------------------------------------------------------

def _csfcube_text(value: Any) -> str:
    """CSFCube stores abstracts as a stringified Python list of sentences; titles are plain strings."""
    if isinstance(value, list):
        return " ".join(str(s).strip() for s in value)
    text = str(value or "").strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)  # data only: literals, never code
        except (ValueError, SyntaxError):
            return text
        if isinstance(parsed, list):
            return " ".join(str(s).strip() for s in parsed)
    return text


def prepare_csfcube(raw_dir: Union[str, Path]) -> RetrievalBenchmark:
    """Normalize the CSFCube release in ``raw_dir`` (files of ``CSFCUBE_FILES``)."""
    raw = Path(raw_dir)
    corpus: Dict[str, Dict[str, str]] = {}
    with open(raw / "abstracts-csfcube-preds.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                corpus[str(rec["paper_id"])] = {"title": _csfcube_text(rec.get("title")), "abstract": _csfcube_text(rec.get("abstract"))}

    query_meta: Dict[str, Dict[str, Any]] = {}
    with open(raw / "queries-release.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            query_meta[f"{row['pid']}_{row['facet']}"] = {"year": row.get("year"), "paper_type": row.get("paper type")}

    queries: List[BenchmarkQuery] = []
    for facet in CSFCUBE_FACETS:
        with open(raw / f"test-pid2anns-csfcube-{facet}.json", encoding="utf-8") as f:
            anns = json.load(f)
        for pid in sorted(anns):
            cands = [str(c) for c in anns[pid]["cands"]]
            grades = [int(g) for g in anns[pid]["relevance_adju"]]  # adjudicated grades, as in the official script
            if len(cands) != len(grades):
                raise ValueError(f"CSFCube {pid}_{facet}: {len(cands)} candidates vs {len(grades)} grades")
            qid = f"{pid}_{facet}"
            queries.append(BenchmarkQuery(
                qid=qid, kind="paper", doc_id=str(pid), facet=facet, qrels=dict(zip(cands, grades)), pool=cands,
                meta=query_meta.get(qid, {}),
            ))

    bench = RetrievalBenchmark(
        name="csfcube", corpus=corpus, queries=queries, relevance_threshold=CSFCUBE_RELEVANCE_THRESHOLD,
        query_adapter="proximity", source="https://github.com/iesl/CSFCube", license="CC BY-NC 4.0",
    )
    bench.check()
    return bench


# ---------------------------------------------------------------------------
# LitSearch
# ---------------------------------------------------------------------------

def prepare_litsearch(raw_dir: Union[str, Path]) -> RetrievalBenchmark:
    """Normalize LitSearch (``query.parquet`` and the six ``corpus_clean_*.parquet`` shards) in ``raw_dir``."""
    import pyarrow.parquet as pq

    raw = Path(raw_dir)
    corpus: Dict[str, Dict[str, str]] = {}
    shards = sorted(raw.glob("corpus_clean_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No corpus_clean_*.parquet shards in {raw}")
    for shard in shards:
        # full_paper is large and unused: read only the needed columns
        for rec in pq.read_table(shard, columns=["corpusid", "title", "abstract"]).to_pylist():
            corpus[str(rec["corpusid"])] = {"title": (rec["title"] or "").strip(), "abstract": (rec["abstract"] or "").strip()}

    queries: List[BenchmarkQuery] = []
    for i, rec in enumerate(pq.read_table(raw / "query.parquet").to_pylist()):
        queries.append(BenchmarkQuery(
            qid=f"litsearch_{i:04d}", kind="text", text=(rec["query"] or "").strip(),
            qrels={str(c): 1 for c in rec["corpusids"]},
            meta={"query_set": rec.get("query_set"), "specificity": rec.get("specificity"), "quality": rec.get("quality")},
        ))

    bench = RetrievalBenchmark(
        name="litsearch", corpus=corpus, queries=queries, relevance_threshold=LITSEARCH_RELEVANCE_THRESHOLD,
        query_adapter="adhoc_query", source="https://huggingface.co/datasets/princeton-nlp/LitSearch",
        license="not stated on the dataset card",
    )
    bench.check()
    return bench


# ---------------------------------------------------------------------------
# Normalized storage
# ---------------------------------------------------------------------------

def save_benchmark(bench: RetrievalBenchmark, out_dir: Union[str, Path]) -> Path:
    """Write ``<out_dir>/<name>/{benchmark.json, corpus.jsonl, queries.jsonl}``."""
    d = Path(out_dir) / bench.name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "corpus.jsonl", "w", encoding="utf-8") as f:
        for doc_id in sorted(bench.corpus):
            f.write(json.dumps({"doc_id": doc_id, **bench.corpus[doc_id]}, ensure_ascii=False) + "\n")
    with open(d / "queries.jsonl", "w", encoding="utf-8") as f:
        for q in bench.queries:
            f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")
    meta = {
        "name": bench.name, "relevance_threshold": bench.relevance_threshold, "query_adapter": bench.query_adapter,
        "source": bench.source, "license": bench.license, "n_docs": len(bench.corpus), "n_queries": len(bench.queries),
        "n_docs_without_abstract": sum(1 for v in bench.corpus.values() if not v["abstract"]),
    }
    (d / "benchmark.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Saved %s: %d docs, %d queries -> %s", bench.name, len(bench.corpus), len(bench.queries), d)
    return d


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_benchmark(bench_dir: Union[str, Path]) -> RetrievalBenchmark:
    """Load a benchmark written by ``save_benchmark``."""
    d = Path(bench_dir)
    meta = json.loads((d / "benchmark.json").read_text(encoding="utf-8"))
    corpus = {r["doc_id"]: {"title": r["title"], "abstract": r["abstract"]} for r in _read_jsonl(d / "corpus.jsonl")}
    queries = [BenchmarkQuery(**r) for r in _read_jsonl(d / "queries.jsonl")]
    bench = RetrievalBenchmark(
        name=meta["name"], corpus=corpus, queries=queries, relevance_threshold=int(meta["relevance_threshold"]),
        query_adapter=meta["query_adapter"], source=meta.get("source", ""), license=meta.get("license", ""),
    )
    bench.check()
    return bench
