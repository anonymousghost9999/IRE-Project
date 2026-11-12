#!/usr/bin/env python3
"""Phase 7 first-stage retrieval benchmark: one table per benchmark (docs/plan.md §7.2 / docs/design.md §13.3).

Usage:
    python scripts/run_retrieval_benchmark.py [--benchmarks csfcube litsearch] [--cross-encoders]

Needs the prepared benchmarks and their indexes (scripts/prepare_retrieval_benchmarks.py, then
scripts/build_benchmark_indexes.py). Writes ``results/retrieval_benchmark_<name>.md`` / ``.json`` (per-query
metrics and top-100 rankings) and a run manifest with copies to ``results/runs/<run_id>/``.

CSFCube is run in two modes: ``corpus`` (retrieve from all 4,207 released papers, the Phase 7 question) and ``pool``
(the official protocol: re-rank each query's annotated pool; adds the official MAP / R-Precision / nDCG%20).
``--cross-encoders`` adds rows that re-rank the RRF list with the configured cross-encoders (design §8.2, Extended).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paperfeed.bench.retrieval import load_benchmark
from paperfeed.bench.retrieval_eval import (
    CHANNEL_DEPTH,
    EVAL_DEPTH,
    RRF_K,
    CrossEncoderReranker,
    format_csfcube_official_table,
    format_main_table,
    run_benchmark,
)
from paperfeed.bench.retrieval_index import load_benchmark_index
from paperfeed.common.config import load_config
from paperfeed.common.logger import get_logger
from paperfeed.common.manifest import RunManifest

logger = get_logger("scripts.run_retrieval_benchmark")

MODES = {"csfcube": ("corpus", "pool"), "litsearch": ("corpus",)}


def build_report(name: str, meta: dict, runs: dict, index_manifest: dict, n_bootstrap: int, run_id: str) -> str:
    parts = [
        f"# Phase 7 — First-Stage Retrieval Benchmark: {name}\n",
        "> [!NOTE]",
        f"> - Corpus: {meta['n_docs']:,} papers ({meta['n_docs_without_abstract']:,} without abstract, indexed by title) | "
        f"Queries: {meta['n_queries']} | Relevant: grade >= {meta['relevance_threshold']}",
        f"> - Source: {meta['source']} (license: {meta['license']})",
        f"> - Dense: {index_manifest['encoder']['base_model']} + `proximity` documents, `{meta['query_adapter']}` queries"
        f"{' (query papers use their own proximity vector)' if meta['query_adapter'] == 'proximity' else ''} | Lexical: BM25s",
        f"> - Every system is scored on its top {EVAL_DEPTH} (the recommender's candidate pool M); RRF fuses each "
        f"channel's top {CHANNEL_DEPTH} with k = {RRF_K}.",
        f"> - Recall@k and MRR: binary relevance; nDCG@k: graded gains, 1/log2(rank+1) discount. CI: percentile "
        f"bootstrap over queries ({n_bootstrap:,} resamples).",
        f"> - Run manifest: `results/runs/{run_id}/manifest.json`\n",
    ]
    if "corpus" in runs:
        parts += [
            "## 1. Candidate generation over the whole corpus\n",
            "Unjudged papers count as non-relevant." + (
                " The query paper itself is excluded from its results." if meta["query_adapter"] == "proximity" else ""),
            "",
            format_main_table(runs["corpus"], n_bootstrap),
        ]
    if "pool" in runs:
        parts += [
            "\n## 2. Official CSFCube protocol (re-rank each query's annotated pool)\n",
            "Every pool document is ranked; documents a channel did not return come last. Plan metrics:\n",
            format_main_table(runs["pool"], n_bootstrap),
            "\nCSFCube's official metrics (re-implemented from its `eval_scripts`: MAP and R-Precision at grade >= 2, "
            "nDCG over the top 20% of the pool with CSFCube's discount; R-Precision follows CSFCube's definition, "
            "precision at the rank of the last relevant paper). Use these for comparison with published CSFCube "
            "results; the system is not facet-aware, so each query paper gets the same ranking for every facet.\n",
            format_csfcube_official_table(runs["pool"]),
        ]
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 7 retrieval benchmark.")
    parser.add_argument("--benchmarks", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--cross-encoders", action="store_true", help="Add RRF + cross-encoder rerank rows (Extended)")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = load_config()
    bench_root = root / cfg["paths"]["processed_dir"] / "benchmarks"
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"retrieval_benchmark_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    manifest = RunManifest(run_id, cfg, runs_dir=root / "results" / "runs")
    manifest.extra_metadata["arguments"] = vars(args)
    try:
        rerankers = []
        if args.cross_encoders:
            m = cfg["models"]
            rerankers = [CrossEncoderReranker(m["cross_encoder_primary"]), CrossEncoderReranker(m["cross_encoder_fallback"])]

        outputs, summary = [], {}
        for name in args.benchmarks:
            bench_dir = bench_root / name
            bench = load_benchmark(bench_dir)
            idx = load_benchmark_index(bench, bench_dir)
            meta = json.loads((bench_dir / "benchmark.json").read_text(encoding="utf-8"))
            runs = {}
            for mode in MODES[name]:
                t0 = time.time()
                runs[mode] = run_benchmark(bench, idx, mode, rerankers)
                logger.info("%s / %s: %d queries in %.1fs", name, mode, len(bench.queries), time.time() - t0)

            report = build_report(name, meta, runs, idx.manifest, args.n_bootstrap, run_id)
            md_path = out_dir / f"retrieval_benchmark_{name}.md"
            json_path = out_dir / f"retrieval_benchmark_{name}.json"
            md_path.write_text(report, encoding="utf-8")
            json_path.write_text(json.dumps({
                "benchmark": name,
                "meta": meta,
                "index_manifest": idx.manifest,
                "runs": {
                    mode: {"systems": r.systems, "aggregate": r.aggregate(), "per_query": r.per_query, "rankings": r.rankings}
                    for mode, r in runs.items()
                },
            }), encoding="utf-8")
            outputs += [md_path, json_path]
            summary[name] = {mode: r.aggregate() for mode, r in runs.items()}
            print(report)

        for f in outputs:
            shutil.copy2(f, manifest.run_dir / f.name)
        manifest.complete(metadata={
            "outputs": [str(f.relative_to(root)) if f.is_relative_to(root) else str(f) for f in outputs],
            "summary": summary,
            "eval_depth": EVAL_DEPTH, "channel_depth": CHANNEL_DEPTH, "rrf_k": RRF_K,
            "cross_encoders": [r.model_name for r in rerankers],
        })
    except BaseException as exc:
        manifest.complete(status="FAILED", metadata={"error": repr(exc)})
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
