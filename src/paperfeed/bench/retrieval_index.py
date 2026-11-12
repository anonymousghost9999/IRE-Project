"""Per-benchmark indexes for the first-stage retrieval benchmark (Phase 7.1 / docs/plan.md §7.1).

Each benchmark has its own corpus, so it gets its own indexes, built with the Phase 2 code and the same document
text as the main corpus:

- dense: SPECTER2 ``proximity`` embeddings of "title [SEP] abstract" (title only without an abstract) in an exact
  FAISS inner-product index;
- lexical: BM25s over "title abstract";
- query vectors: text queries (LitSearch) with the ``adhoc_query`` adapter; paper queries (CSFCube) reuse the query
  paper's own ``proximity`` vector, since query-by-example compares paper to paper.

Files in ``<benchmark_dir>/index/``: ``embeddings.npy`` + ``embedding_ids.json`` (checkpointed store),
``faiss.index`` + ``paper_ids.json``, ``bm25s/``, ``query_embeddings.npy`` + ``query_ids.json`` and
``index_manifest.json``. The manifest fingerprints the corpus text and the encoder; if either changes the stored
vectors are discarded and rebuilt, so an index can never silently mix two corpora or two models.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Protocol

import numpy as np

from paperfeed.bench.retrieval import RetrievalBenchmark
from paperfeed.common.logger import get_logger
from paperfeed.index.dense import DenseIndex
from paperfeed.index.lexical import LexicalIndex
from paperfeed.index.store import EmbeddingStore

logger = get_logger("bench.retrieval_index")

INDEX_SUBDIR = "index"


class BenchmarkEncoder(Protocol):
    """The slice of ``SPECTER2Encoder`` used here."""

    base_model_name: str
    dense_adapter_name: str
    query_adapter_name: str

    def embed_papers(self, papers: List[Dict[str, Any]], batch_size: int = 32, show_progress: bool = False) -> np.ndarray: ...

    def embed_queries(self, queries: List[str], batch_size: int = 32) -> np.ndarray: ...


@dataclass
class BenchmarkIndex:
    name: str
    dense: DenseIndex
    lexical: LexicalIndex
    store: EmbeddingStore
    query_vectors: Dict[str, np.ndarray]  # qid -> query vector (dense channel)
    manifest: Dict[str, Any]


def lexical_text(doc: Mapping[str, str]) -> str:
    """BM25 document text: the same "title abstract" as the main corpus index (scripts/build_index.py)."""
    return f"{doc.get('title', '')} {doc.get('abstract', '')}".strip()


def corpus_fingerprint(bench: RetrievalBenchmark) -> str:
    """SHA-256 over the sorted doc IDs and their text: changes whenever a document is added, removed or edited."""
    h = hashlib.sha256()
    for doc_id in sorted(bench.corpus):
        doc = bench.corpus[doc_id]
        h.update(f"{doc_id}\t{doc.get('title', '')}\t{doc.get('abstract', '')}\n".encode("utf-8"))
    return h.hexdigest()


def _encoder_info(encoder: Optional[BenchmarkEncoder]) -> Optional[Dict[str, str]]:
    if encoder is None:
        return None
    return {
        "base_model": encoder.base_model_name,
        "dense_adapter": encoder.dense_adapter_name,
        "query_adapter": encoder.query_adapter_name,
    }


def build_benchmark_index(
    bench: RetrievalBenchmark,
    encoder: BenchmarkEncoder,
    bench_dir: Path,
    batch_size: int = 64,
    checkpoint_every: int = 10000,
) -> Dict[str, Any]:
    """Embed (resuming from a matching checkpoint), then build FAISS, BM25s and the query vectors. Returns the manifest."""
    out = Path(bench_dir) / INDEX_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "index_manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    fingerprint = corpus_fingerprint(bench)
    enc_info = _encoder_info(encoder)

    emb_path, ids_path = out / "embeddings.npy", out / "embedding_ids.json"
    reusable = old.get("corpus_sha256") == fingerprint and old.get("encoder") == enc_info
    if not reusable and emb_path.exists():
        logger.warning("%s: corpus or encoder changed since the last build; discarding the stored vectors.", bench.name)
        emb_path.unlink()
        ids_path.unlink(missing_ok=True)
    store = EmbeddingStore.load(emb_path, ids_path)

    doc_ids = sorted(bench.corpus)
    todo = store.missing(doc_ids)
    t_embed = 0.0
    logger.info("%s: %d of %d documents need an embedding.", bench.name, len(todo), len(doc_ids))
    for start in range(0, len(todo), checkpoint_every):
        chunk = todo[start : start + checkpoint_every]
        t0 = time.time()
        vecs = encoder.embed_papers([bench.corpus[d] for d in chunk], batch_size=batch_size, show_progress=True)
        t_embed += time.time() - t0
        store.add(chunk, np.asarray(vecs, dtype=np.float32))
        store.save(emb_path, ids_path)
        logger.info("%s: embedded %d/%d; store checkpointed.", bench.name, min(start + checkpoint_every, len(todo)), len(todo))
    if not ids_path.exists():
        store.save(emb_path, ids_path)
    vectors = store.select(doc_ids)

    t0 = time.time()
    dense = DenseIndex(dimension=vectors.shape[1])
    dense.add_documents(doc_ids, vectors)
    dense.save(out / "faiss.index", out / "paper_ids.json")
    t_faiss = time.time() - t0

    t0 = time.time()
    lexical = LexicalIndex(stopwords="en")
    lexical.index_documents(doc_ids, [lexical_text(bench.corpus[d]) for d in doc_ids], show_progress=False)
    lexical.save(out / "bm25s")
    t_bm25 = time.time() - t0

    t0 = time.time()
    qids, qvecs = _query_vectors(bench, encoder, store, batch_size)
    np.save(out / "query_embeddings.npy", qvecs)
    (out / "query_ids.json").write_text(json.dumps({"ids": qids}), encoding="utf-8")
    t_queries = time.time() - t0

    manifest = {
        "benchmark": bench.name,
        "n_docs": len(doc_ids),
        "n_queries": len(qids),
        "corpus_sha256": fingerprint,
        "encoder": enc_info,
        "query_adapter": bench.query_adapter,
        "doc_text": "title [SEP] abstract (title only without an abstract); BM25: title abstract",
        "embedding_dim": int(vectors.shape[1]),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # embedding time accumulates over resumed runs of the same corpus and encoder
        "embedding_time_seconds": round((old.get("embedding_time_seconds", 0.0) if reusable else 0.0) + t_embed, 2),
        "embedded_docs_timed": (old.get("embedded_docs_timed", 0) if reusable else 0) + len(todo),
        "faiss_build_time_seconds": round(t_faiss, 2),
        "bm25s_build_time_seconds": round(t_bm25, 2),
        "query_embedding_time_seconds": round(t_queries, 2),
    }
    timed = manifest["embedded_docs_timed"]
    manifest["embedding_sec_per_1k"] = round(manifest["embedding_time_seconds"] / timed * 1000, 2) if timed else None
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("%s: index built (%d docs, %d queries) in %s", bench.name, len(doc_ids), len(qids), out)
    return manifest


def _query_vectors(
    bench: RetrievalBenchmark,
    encoder: Optional[BenchmarkEncoder],
    store: EmbeddingStore,
    batch_size: int,
):
    """Dense query vector of every query, in benchmark order."""
    qids = [q.qid for q in bench.queries]
    vecs: Dict[str, np.ndarray] = {}
    paper_q = [q for q in bench.queries if q.kind == "paper"]
    if paper_q:
        for q, v in zip(paper_q, store.select([q.doc_id for q in paper_q])):
            vecs[q.qid] = v
    text_q = [q for q in bench.queries if q.kind == "text"]
    if text_q:
        if encoder is None:
            raise ValueError(f"{bench.name}: text queries need the encoder.")
        if bench.query_adapter != "adhoc_query":
            raise ValueError(f"{bench.name}: text queries must use the adhoc_query adapter, got {bench.query_adapter}.")
        for q, v in zip(text_q, encoder.embed_queries([q.text for q in text_q], batch_size=batch_size)):
            vecs[q.qid] = v
    other = [q.qid for q in bench.queries if q.kind not in ("paper", "text")]
    if other:
        raise ValueError(f"{bench.name}: unknown query kind for {other[:3]}")
    return qids, np.vstack([vecs[q] for q in qids]).astype(np.float32)


def load_benchmark_index(bench: RetrievalBenchmark, bench_dir: Path) -> BenchmarkIndex:
    """Load the indexes and check they were built from exactly this corpus and these queries."""
    out = Path(bench_dir) / INDEX_SUBDIR
    manifest_path = out / "index_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No index for {bench.name} in {out}; run scripts/build_benchmark_indexes.py first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("corpus_sha256") != corpus_fingerprint(bench):
        raise ValueError(f"{bench.name}: the index was built from a different corpus; rebuild it.")
    qids = json.loads((out / "query_ids.json").read_text(encoding="utf-8"))["ids"]
    if qids != [q.qid for q in bench.queries]:
        raise ValueError(f"{bench.name}: the index's query vectors do not match the benchmark queries; rebuild it.")
    qvecs = np.load(out / "query_embeddings.npy")
    return BenchmarkIndex(
        name=bench.name,
        dense=DenseIndex.load(out / "faiss.index", out / "paper_ids.json"),
        lexical=LexicalIndex.load(out / "bm25s"),
        store=EmbeddingStore.load(out / "embeddings.npy", out / "embedding_ids.json"),
        query_vectors=dict(zip(qids, qvecs)),
        manifest=manifest,
    )
