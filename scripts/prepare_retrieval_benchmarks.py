#!/usr/bin/env python3
"""Download and normalize the Phase 7 retrieval benchmarks (docs/plan.md §7.1 / docs/design.md §13.3).

Usage:
    python scripts/prepare_retrieval_benchmarks.py [--benchmarks csfcube litsearch] [--download]

Raw files go to ``data/raw/<name>/`` (``--download`` fetches any that are missing; otherwise place them there by
hand). The normalized benchmark is written to ``data/processed/benchmarks/<name>/`` (benchmark.json, corpus.jsonl,
queries.jsonl), which the index builder and the benchmark runner read.

Sources:
    CSFCube   https://github.com/iesl/CSFCube              (CC BY-NC 4.0)
    LitSearch https://huggingface.co/datasets/princeton-nlp/LitSearch  (~1.3 GB of parquet; only id/title/abstract are kept)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paperfeed.bench.retrieval import (
    CSFCUBE_BASE_URL,
    CSFCUBE_FILES,
    LITSEARCH_BASE_URL,
    LITSEARCH_FILES,
    prepare_csfcube,
    prepare_litsearch,
    save_benchmark,
)
from paperfeed.common.config import load_config
from paperfeed.common.logger import get_logger

logger = get_logger("scripts.prepare_retrieval_benchmarks")

SOURCES = {
    "csfcube": (prepare_csfcube, {name: f"{CSFCUBE_BASE_URL}/{name}" for name in CSFCUBE_FILES}),
    "litsearch": (prepare_litsearch, {name: f"{LITSEARCH_BASE_URL}/{path}" for name, path in LITSEARCH_FILES.items()}),
}


def download_missing(urls: dict, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, url in urls.items():
        dest = raw_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        logger.info("Downloading %s ...", url)
        tmp = dest.with_name(dest.name + ".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URLs above
        tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and normalize the Phase 7 retrieval benchmarks.")
    parser.add_argument("--benchmarks", nargs="+", choices=sorted(SOURCES), default=sorted(SOURCES))
    parser.add_argument("--download", action="store_true", help="Download missing raw files")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = load_config()
    raw_root = root / cfg["paths"]["raw_dir"]
    out_root = root / cfg["paths"]["processed_dir"] / "benchmarks"

    for name in args.benchmarks:
        prepare, urls = SOURCES[name]
        raw_dir = raw_root / name
        if args.download:
            download_missing(urls, raw_dir)
        missing = [n for n in urls if not (raw_dir / n).exists()]
        if missing:
            logger.error("%s: missing raw files in %s: %s (re-run with --download)", name, raw_dir, missing)
            return 1
        bench = prepare(raw_dir)
        out = save_benchmark(bench, out_root)
        n_rel = sum(len(q.relevant(bench.relevance_threshold)) for q in bench.queries)
        print(f"{name}: {len(bench.corpus)} docs, {len(bench.queries)} queries, {n_rel} relevant (grade >= "
              f"{bench.relevance_threshold}) judgements -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
