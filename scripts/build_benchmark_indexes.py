#!/usr/bin/env python3
"""Build SPECTER2 + BM25s indexes for the Phase 7 retrieval benchmarks (docs/plan.md §7.1).

Usage:
    python scripts/build_benchmark_indexes.py [--benchmarks csfcube litsearch] [--batch-size 64]

Reads ``data/processed/benchmarks/<name>/`` (from scripts/prepare_retrieval_benchmarks.py) and writes
``data/processed/benchmarks/<name>/index/``. Embedding is checkpointed and resumes on a re-run; a changed corpus or
encoder forces a rebuild.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paperfeed.bench.retrieval import load_benchmark
from paperfeed.bench.retrieval_index import build_benchmark_index
from paperfeed.common.config import load_config
from paperfeed.common.logger import get_logger

logger = get_logger("scripts.build_benchmark_indexes")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build per-benchmark retrieval indexes.")
    parser.add_argument("--benchmarks", nargs="+", default=["csfcube", "litsearch"])
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = load_config()
    bench_root = root / cfg["paths"]["processed_dir"] / "benchmarks"

    from paperfeed.index.embeddings import SPECTER2Encoder

    m = cfg["models"]
    encoder = SPECTER2Encoder(m["dense_encoder"], m["dense_adapter"], m["query_adapter"])
    every = int(cfg.get("index", {}).get("embed_checkpoint_every", 10000))

    for name in args.benchmarks:
        bench_dir = bench_root / name
        if not (bench_dir / "benchmark.json").exists():
            logger.error("%s is not prepared; run scripts/prepare_retrieval_benchmarks.py first.", bench_dir)
            return 1
        manifest = build_benchmark_index(load_benchmark(bench_dir), encoder, bench_dir, args.batch_size, every)
        print(f"{name}: {manifest['n_docs']} docs, {manifest['n_queries']} queries, "
              f"embedding {manifest['embedding_time_seconds']}s ({manifest['embedding_sec_per_1k']} s/1k), "
              f"BM25s {manifest['bm25s_build_time_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
