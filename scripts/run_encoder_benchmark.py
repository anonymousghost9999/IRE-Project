#!/usr/bin/env python3
"""V1 encoder study on the retrieval benchmarks (plan 10.1b / docs/design.md §13.8 V1 point 3). Reported, not decisive.

Usage:
    python scripts/run_encoder_benchmark.py [--encoders specter2 scincl bge qwen3] [--benchmarks csfcube litsearch]

For every benchmark and encoder: build the encoder's index in ``<benchmark_dir>/index_<key>/`` if it is missing
(SPECTER2 keeps the Phase 7 ``index/``; embedding is checkpointed and resumes on a re-run), then run BM25, the dense
channel and RRF exactly as in Phase 7. Writes one table per benchmark with every encoder's rows and the paired
difference of each encoder against SPECTER2:
    results/encoder_benchmark_<name>.md / .json  and a run manifest in results/runs/.
Qwen3-Embedding-0.6B is released after T: it appears here only, never in a Track 1 decision.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


from paperfeed.bench.retrieval import load_benchmark
from paperfeed.bench.retrieval_eval import CHANNEL_DEPTH, EVAL_DEPTH, RRF_K, BenchmarkRun, format_main_table, run_benchmark
from paperfeed.bench.retrieval_index import build_benchmark_index, index_subdir, load_benchmark_index
from paperfeed.common.config import load_config
from paperfeed.common.logger import get_logger
from paperfeed.common.manifest import RunManifest
from paperfeed.eval.stats import paired_bootstrap_test
from paperfeed.index.encoders import ENCODER_KEYS, ENCODER_SPECS, make_encoder

logger = get_logger("scripts.run_encoder_benchmark")

MODES = {"csfcube": ("corpus", "pool"), "litsearch": ("corpus",)}
LABEL = {"specter2": "SPECTER2", "scincl": "SciNCL", "bge": "BGE", "gte": "GTE-large", "qwen3": "Qwen3-Emb-0.6B (post-T)"}


def expected_model(cfg, key: str) -> str:
    return cfg["models"]["dense_encoder"] if key == "specter2" else ENCODER_SPECS[key].model_name


def ensure_index(bench, bench_dir: Path, key: str, cfg, batch_size: int, every: int):
    """Load the encoder's index, building it first when it is missing or was built with another model."""
    subdir = index_subdir(key)
    try:
        idx = load_benchmark_index(bench, bench_dir, subdir)
        if idx.manifest.get("encoder", {}).get("base_model") == expected_model(cfg, key):
            return idx
        logger.warning("%s/%s was built with another model; rebuilding.", bench.name, subdir)
    except FileNotFoundError:
        pass
    if key == "specter2":
        from paperfeed.index.embeddings import SPECTER2Encoder

        m = cfg["models"]
        encoder = SPECTER2Encoder(m["dense_encoder"], m["dense_adapter"], m["query_adapter"])
    else:
        encoder = make_encoder(key)
    build_benchmark_index(bench, encoder, bench_dir, batch_size, every, subdir=subdir)
    del encoder
    _free_gpu()
    return load_benchmark_index(bench, bench_dir, subdir)


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def merge_runs(runs_by_key: dict) -> BenchmarkRun:
    """One run with BM25 once and each encoder's dense and RRF rows (BM25 is identical across encoders)."""
    first = next(iter(runs_by_key.values()))
    systems, per_query, rankings = ["BM25"], {"BM25": first.per_query["BM25"]}, {"BM25": first.rankings["BM25"]}
    for run in runs_by_key.values():
        for s in run.systems[1:]:
            systems.append(s)
            per_query[s], rankings[s] = run.per_query[s], run.rankings[s]
    return BenchmarkRun(first.benchmark, first.mode, systems, per_query, first.facets, rankings)


def paired_table(run: BenchmarkRun, keys, n_bootstrap: int) -> str:
    """Each encoder's dense and RRF rows against SPECTER2's, paired over queries."""
    lines = [
        "| Compared with SPECTER2 | Recall@100 diff [95% CI] | nDCG@10 diff [95% CI] |",
        "| :--- | :---: | :---: |",
    ]
    qids = sorted(run.per_query["BM25"])
    for kind in ("Dense", "RRF"):
        ref = f"{kind}: {LABEL['specter2']}"
        if ref not in run.per_query:
            continue
        for key in keys:
            if key == "specter2":
                continue
            name = f"{kind}: {LABEL[key]}"
            cells = []
            for metric in ("recall_at_100", "ndcg_at_10"):
                a = [run.per_query[name][q][metric] for q in qids]
                b = [run.per_query[ref][q][metric] for q in qids]
                diff, lo, hi, _, _ = paired_bootstrap_test(a, b, n_resamples=n_bootstrap)
                cells.append(f"{diff:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            lines.append(f"| {name} | {cells[0]} | {cells[1]} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="V1 encoder study on the retrieval benchmarks.")
    # the V1 study's encoders (design §13.8 V1 point 1); GTE belongs to Phase 11 and is only run when asked for
    parser.add_argument("--encoders", nargs="+", default=["specter2", "scincl", "bge", "qwen3"], choices=list(ENCODER_KEYS))
    parser.add_argument("--benchmarks", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    keys = ["specter2"] + [k for k in args.encoders if k != "specter2"]  # SPECTER2 is the reference row

    root = Path(__file__).resolve().parent.parent
    cfg = load_config()
    bench_root = root / cfg["paths"]["processed_dir"] / "benchmarks"
    every = int(cfg.get("index", {}).get("embed_checkpoint_every", 10000))
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"encoder_benchmark_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    manifest = RunManifest(run_id, cfg, runs_dir=root / "results" / "runs")
    manifest.extra_metadata["arguments"] = vars(args)
    try:
        outputs, summary = [], {}
        for name in args.benchmarks:
            bench_dir = bench_root / name
            bench = load_benchmark(bench_dir)
            indexes = {k: ensure_index(bench, bench_dir, k, cfg, args.batch_size, every) for k in keys}
            merged = {}
            for mode in MODES[name]:
                per_key = {}
                for k in keys:
                    t0 = time.time()
                    per_key[k] = run_benchmark(bench, indexes[k], mode, dense_name=f"Dense: {LABEL[k]}",
                                               rrf_name=f"RRF: {LABEL[k]}")
                    logger.info("%s / %s / %s: %.1fs", name, mode, k, time.time() - t0)
                merged[mode] = merge_runs(per_key)

            parts = [
                f"# V1 Encoder Study: {name} (retrieval benchmark, reported, not decisive)\n",
                "> [!NOTE]",
                "> - design §13.8 V1: SciNCL and BGE are the Track 1 candidates; Qwen3-Embedding-0.6B was released "
                "after T and is shown here only. Each encoder uses its documented text format (V1 point 2); "
                "SPECTER2 keeps its adapters. BM25s is the same for every encoder.",
                f"> - Top {EVAL_DEPTH} scored; RRF fuses each channel's top {CHANNEL_DEPTH} with k = {RRF_K}. "
                f"CIs: bootstrap over queries ({args.n_bootstrap:,} resamples).",
                f"> - Run manifest: `results/runs/{run_id}/manifest.json`\n",
                "| Encoder | Model | Dim | Embedding s / 1k docs |",
                "| :--- | :--- | :---: | :---: |",
            ]
            for k in keys:
                im = indexes[k].manifest
                parts.append(f"| {LABEL[k]} | {im['encoder']['base_model']} | {im['embedding_dim']} | {im.get('embedding_sec_per_1k')} |")
            for mode, run in merged.items():
                title = "whole corpus" if mode == "corpus" else "official CSFCube pool (re-rank each query's annotated pool)"
                parts += [f"\n## {mode}: {title}\n", format_main_table(run, args.n_bootstrap), "",
                          paired_table(run, keys, args.n_bootstrap)]
            report = "\n".join(parts) + "\n"

            md_path, json_path = out_dir / f"encoder_benchmark_{name}.md", out_dir / f"encoder_benchmark_{name}.json"
            md_path.write_text(report, encoding="utf-8")
            json_path.write_text(json.dumps({
                "benchmark": name,
                "encoders": {k: indexes[k].manifest for k in keys},
                "runs": {m: {"systems": r.systems, "aggregate": r.aggregate(), "per_query": r.per_query}
                         for m, r in merged.items()},
            }), encoding="utf-8")
            outputs += [md_path, json_path]
            summary[name] = {m: r.aggregate() for m, r in merged.items()}
            print(report)

        for f in outputs:
            shutil.copy2(f, manifest.run_dir / f.name)
        manifest.complete(metadata={"outputs": [str(f.relative_to(root)) for f in outputs], "summary": summary,
                                    "encoders": keys})
    except BaseException as exc:
        manifest.complete(status="FAILED", metadata={"error": repr(exc)})
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
