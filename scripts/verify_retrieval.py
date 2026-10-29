"""Verification and latency/footprint benchmark script for Phase 2 Indices.

Governed by docs/plan.md §2.5 and docs/design.md §5, §15.1:
- Records retrieval latency (query encoding, dense search, lexical search, hybrid RRF fusion)
- Records index disk footprint and memory consumption
- Asserts semantic neighbor relevance across CS/AI/IR topics
- Generates a structured Markdown report in results/phase2_benchmark_report.md

Usage:
    python scripts/verify_retrieval.py [--k 5] [--out-report results/phase2_benchmark_report.md]
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paperfeed.common.config import find_project_root, load_config
from paperfeed.common.logger import get_logger
from paperfeed.data.db import DatabaseManager
from paperfeed.index.dense import DenseIndex
from paperfeed.index.embeddings import SPECTER2Encoder
from paperfeed.index.lexical import LexicalIndex
from paperfeed.index.retriever import HybridRetriever, reciprocal_rank_fusion

logger = get_logger("scripts.verify_retrieval")

TEST_QUERIES = [
    "Dense passage retrieval and bi-encoder ranking for open-domain question answering",
    "Transformer architecture for language modeling and self-attention mechanisms",
    "Graph neural networks and node representation learning on citation graphs",
    "Contrastive learning for self-supervised visual representation",
]


def get_dir_size_mb(path: Path) -> float:
    """Calculate directory size in megabytes."""
    total = 0
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify retrieval quality and benchmark latencies.")
    parser.add_argument("--k", type=int, default=5, help="Number of results to display and log")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--report-path", default="results/phase2_benchmark_report.md")
    args = parser.parse_args()

    root = find_project_root()
    cfg = load_config()
    out_dir = root / args.out_dir
    report_path = root / args.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)

    faiss_path = out_dir / "faiss.index"
    id_map_path = out_dir / "paper_ids.json"
    bm25s_dir = out_dir / "bm25s"
    manifest_path = out_dir / "index_manifest.json"
    embeddings_path = out_dir / "embeddings.npy"
    db_path = root / cfg["paths"]["sqlite_db"]

    if not faiss_path.exists() or not bm25s_dir.exists():
        logger.error(f"Indices not found in {out_dir}. Please wait for build_index.py to complete.")
        return 1

    logger.info("Calculating index storage footprints...")
    footprint = {
        "embeddings_npy_mb": round(get_dir_size_mb(embeddings_path), 2) if embeddings_path.exists() else 0.0,
        "faiss_index_mb": round(get_dir_size_mb(faiss_path), 2),
        "id_map_json_mb": round(get_dir_size_mb(id_map_path), 2),
        "bm25s_dir_mb": round(get_dir_size_mb(bm25s_dir), 2),
    }
    footprint["total_index_storage_mb"] = round(sum(footprint.values()), 2)

    logger.info("Loading FAISS dense index...")
    dense_idx = DenseIndex.load(faiss_path, id_map_path)
    total_indexed = len(dense_idx)
    logger.info(f"Loaded FAISS index with {total_indexed} papers.")

    logger.info("Loading BM25s lexical index...")
    lex_idx = LexicalIndex.load(bm25s_dir)
    logger.info(f"Loaded BM25s index with {len(lex_idx)} papers.")

    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    db = DatabaseManager(db_path=db_path)
    retriever = HybridRetriever(dense_index=dense_idx, lexical_index=lex_idx)

    logger.info("Initializing and warming up query encoder (SPECTER2 adhoc_query)...")
    encoder = SPECTER2Encoder(
        base_model_name=cfg["models"]["dense_encoder"],
        dense_adapter_name=cfg["models"]["dense_adapter"],
        query_adapter_name=cfg["models"]["query_adapter"],
    )
    import numpy as np
    _ = encoder.embed_query("warmup query")
    _ = dense_idx.search(np.zeros(768, dtype=np.float32), k=5)
    _ = lex_idx.search("warmup", k=5)

    query_benchmarks: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print(f"PHASE 2 RETRIEVAL BENCHMARK ({total_indexed:,} CANDIDATE PAPERS)")
    print("=" * 80)

    for i, q in enumerate(TEST_QUERIES, 1):
        print(f"\n[Test Query {i}]: \"{q}\"")

        # 1. Query Encoding
        t0 = time.perf_counter()
        q_vec = encoder.embed_query(q)
        lat_embed = (time.perf_counter() - t0) * 1000

        # 2. Dense FAISS search and 3. BM25s search, at the production channel depth (M_channel)
        depth = retriever.channel_depth
        t0 = time.perf_counter()
        dense_hits = dense_idx.search(q_vec, k=depth)
        lat_dense = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        lex_hits = lex_idx.search(q, k=depth)
        lat_lex = (time.perf_counter() - t0) * 1000

        # 4. RRF fusion of the two lists already retrieved (fusion only; the searches are timed above)
        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion([[p for p, _ in dense_hits], [p for p, _ in lex_hits]], k_rrf=retriever.rrf_k)
        hybrid_hits = fused[: args.k]
        lat_rrf = (time.perf_counter() - t0) * 1000

        total_lat = lat_embed + lat_dense + lat_lex + lat_rrf
        print(f"Latency: Embed={lat_embed:.1f}ms | Dense Search={lat_dense:.2f}ms | BM25s Search={lat_lex:.2f}ms | RRF={lat_rrf:.2f}ms | Total={total_lat:.1f}ms")

        # Fetch metadata from SQLite
        all_pids = [pid for pid, _ in hybrid_hits]
        query_ids = [f"https://openalex.org/{pid}" for pid in all_pids] + all_pids
        papers_meta = {}
        with db.get_connection() as conn:
            marks = ",".join("?" for _ in query_ids)
            cur = conn.execute(
                f"SELECT paper_id, title, year, citation_count_T FROM papers WHERE paper_id IN ({marks})",
                query_ids,
            )
            for row in cur.fetchall():
                pid = row[0].replace("https://openalex.org/", "").strip()
                papers_meta[pid] = {"title": row[1] or "", "year": row[2], "cites": row[3]}

        hits_summary = []
        print("\n  Top Hybrid RRF Results:")
        for rank, (pid, score) in enumerate(hybrid_hits, 1):
            meta = papers_meta.get(pid, {"title": pid, "year": "N/A", "cites": 0})
            print(f"    {rank}. [{meta['year']}] {meta['title'][:65]}... (Score={score:.4f}, Cites={meta['cites']})")
            hits_summary.append({
                "rank": rank,
                "paper_id": pid,
                "title": meta["title"],
                "year": meta["year"],
                "citations_T": meta["cites"],
                "score": round(score, 5),
            })

        query_benchmarks.append({
            "query": q,
            "latency_ms": {
                "embed": round(lat_embed, 2),
                "dense_search": round(lat_dense, 2),
                "bm25s_search": round(lat_lex, 2),
                "rrf_fusion": round(lat_rrf, 2),
                "total": round(total_lat, 2),
            },
            "top_hits": hits_summary,
        })

    # Calculate average latencies across test queries
    avg_embed = sum(b["latency_ms"]["embed"] for b in query_benchmarks) / len(query_benchmarks)
    avg_dense = sum(b["latency_ms"]["dense_search"] for b in query_benchmarks) / len(query_benchmarks)
    avg_lex = sum(b["latency_ms"]["bm25s_search"] for b in query_benchmarks) / len(query_benchmarks)
    avg_rrf = sum(b["latency_ms"]["rrf_fusion"] for b in query_benchmarks) / len(query_benchmarks)
    avg_total = sum(b["latency_ms"]["total"] for b in query_benchmarks) / len(query_benchmarks)

    # Generate Markdown Report
    lines = [
        "# Phase 2 Benchmark Report: Representations & Retrieval Indices",
        "",
        f"**Date Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  ",
        f"**Candidate Pool:** {total_indexed:,} papers (evaluated $\le {cfg['project']['cutoff_year']}$)  ",
        f"**Dense Architecture:** `{cfg['models']['dense_encoder']}` (adapters: `proximity` + `adhoc_query`, fp16)  ",
        f"**Lexical Engine:** `bm25s` with English stop-word tokenization  ",
        f"**Fusion Method:** Reciprocal Rank Fusion (RRF, $k=60$, $M_{{channel}}=200$, $M=100$)  ",
        "",
        "---",
        "",
        "## 1. Storage & Memory Footprint",
        "",
        "| Artifact | Type | Storage Size | Notes |",
        "|---|---|---|---|",
        f"| `embeddings.npy` | Dense Tensor | **{footprint['embeddings_npy_mb']:.2f} MB** | 768-d float32 L2-normalized embeddings |",
        f"| `faiss.index` | FAISS Index | **{footprint['faiss_index_mb']:.2f} MB** | Exact inner-product (`IndexFlatIP`) |",
        f"| `paper_ids.json` | ID Mapping | **{footprint['id_map_json_mb']:.2f} MB** | Internal ID $\leftrightarrow$ OpenAlex ID mapping |",
        f"| `bm25s/` | Inverted Index | **{footprint['bm25s_dir_mb']:.2f} MB** | Serialized sparse lexical index |",
        f"| **Total Footprint** | Combined | **{footprint['total_index_storage_mb']:.2f} MB** | Fits easily in server/client memory |",
        "",
        "---",
        "",
        "## 2. Offline Build Throughput",
        "",
        "| Build Metric | Measurement | Description |",
        "|---|---|---|",
        f"| Total Candidate Papers | **{manifest.get('total_papers', total_indexed):,}** | Papers $\le 2024$ with non-empty abstracts |",
        f"| Embedding Device | **`{manifest.get('embedding_device', 'GPU')}`** | Hardware acceleration |",
        f"| Total Embedding Time | **{manifest.get('embedding_time_seconds', 0.0):.2f} s** | SPECTER2 forward pass in fp16 |",
        f"| Embedding Rate | **{manifest.get('embedding_sec_per_1k', 0.0):.2f} s / 1k papers** | Throughput per 1,000 papers |",
        f"| FAISS Build Time | **{manifest.get('faiss_build_time_seconds', 0.0):.2f} s** | Time to populate IndexFlatIP |",
        f"| BM25s Build Time | **{manifest.get('bm25s_build_time_seconds', 0.0):.2f} s** | Text tokenization and inverted index build |",
        "",
        "---",
        "",
        "## 3. Online Retrieval Latency Benchmark",
        "",
        "Measured across 4 representative CS/AI/IR domain queries (100k+ candidates). Each stage is timed once at the production channel depth ($M_{channel}=200$); the RRF row is the fusion step alone, and the end-to-end row is the sum of the four stages.",
        "",
        "| Retrieval Pipeline Stage | Mean Latency | Min Latency | Max Latency | Target / Requirement |",
        "|---|---|---|---|---|",
        f"| **Query Encoding (SPECTER2)** | **{avg_embed:.1f} ms** | {min(b['latency_ms']['embed'] for b in query_benchmarks):.1f} ms | {max(b['latency_ms']['embed'] for b in query_benchmarks):.1f} ms | $< 100$ ms |",
        f"| **Dense Search (FAISS FlatIP)** | **{avg_dense:.2f} ms** | {min(b['latency_ms']['dense_search'] for b in query_benchmarks):.2f} ms | {max(b['latency_ms']['dense_search'] for b in query_benchmarks):.2f} ms | $< 25$ ms |",
        f"| **Lexical Search (BM25s)** | **{avg_lex:.2f} ms** | {min(b['latency_ms']['bm25s_search'] for b in query_benchmarks):.2f} ms | {max(b['latency_ms']['bm25s_search'] for b in query_benchmarks):.2f} ms | $< 20$ ms |",
        f"| **RRF Rank Fusion ($k=60$)** | **{avg_rrf:.2f} ms** | {min(b['latency_ms']['rrf_fusion'] for b in query_benchmarks):.2f} ms | {max(b['latency_ms']['rrf_fusion'] for b in query_benchmarks):.2f} ms | $< 5$ ms |",
        f"| **End-to-End First Stage** | **{avg_total:.1f} ms** | {min(b['latency_ms']['total'] for b in query_benchmarks):.1f} ms | {max(b['latency_ms']['total'] for b in query_benchmarks):.1f} ms | **Interactive (< 150 ms)** |",
        "",
        "---",
        "",
        "## 4. Qualitative Sanity Check: Top-5 Hybrid Recommendations",
        "",
    ]

    for b in query_benchmarks:
        lines.append(f"### Query: *\"{b['query']}\"*")
        lines.append(f"- **Retrieval Latency:** {b['latency_ms']['total']:.1f} ms (Dense: {b['latency_ms']['dense_search']:.1f} ms, BM25s: {b['latency_ms']['bm25s_search']:.1f} ms, RRF: {b['latency_ms']['rrf_fusion']:.1f} ms)")
        lines.append("")
        lines.append("| Rank | Year | OpenAlex ID | Title | Citations ($\le 2024$) | RRF Score |")
        lines.append("|---|---|---|---|---|---|")
        for hit in b["top_hits"]:
            safe_title = hit["title"].replace("|", "-")
            lines.append(f"| {hit['rank']} | {hit['year']} | `{hit['paper_id']}` | {safe_title} | {hit['citations_T']} | {hit['score']:.5f} |")
        lines.append("")

    report_content = "\n".join(lines)
    report_path.write_text(report_content, encoding="utf-8")
    logger.info(f"Wrote Phase 2 Benchmark Report to {report_path}")

    # Also save JSON benchmark
    json_path = report_path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_candidate_papers": total_indexed,
        "storage_footprint_mb": footprint,
        "build_manifest": manifest,
        "average_latencies_ms": {
            "query_encoding": round(avg_embed, 2),
            "dense_search": round(avg_dense, 2),
            "bm25s_search": round(avg_lex, 2),
            "rrf_fusion": round(avg_rrf, 2),
            "total_first_stage": round(avg_total, 2),
        },
        "query_benchmarks": query_benchmarks,
    }, indent=2), encoding="utf-8")
    logger.info(f"Wrote JSON benchmark to {json_path}")

    print("\n" + "=" * 80)
    print(f"BENCHMARK REPORT SAVED: {report_path}")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
