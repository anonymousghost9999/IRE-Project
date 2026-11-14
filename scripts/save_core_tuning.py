#!/usr/bin/env python3
"""Write the Core tuning file under its configuration-signature name (tuning split only; the test split is not touched).

Usage:
    python scripts/save_core_tuning.py [--overwrite]

Older runs used a shared data/artifacts/tuning_results.json, written before tuning files were named by configuration
and never committed. This recomputes the Core §9 grid and the Hybrid-Persona baseline tuning on the 80 tuning authors,
checks the optimum against that file when it exists (weights and lambda_div must agree), and saves
data/artifacts/tuning_results_<hash>.json, which every script prefers over the shared file. Commit it with -f.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paperfeed.common.config import load_config
from paperfeed.common.manifest import RunManifest
from paperfeed.data.cohort import clean_id
from paperfeed.eval.tuning import (
    grid_per_user_ndcg,
    precompute_tuning_cache,
    tune_hybrid_persona,
    tuning_artifact_path,
    tuning_result_from_points,
    tuning_signature,
)
from paperfeed.rec.recommender import RecommendConfig, RecommenderResources


def main() -> int:
    parser = argparse.ArgumentParser(description="Save the Core tuning file under its signature name.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = load_config()
    artifacts = root / cfg["paths"]["artifacts_dir"]
    processed = root / cfg["paths"]["processed_dir"]
    cohort = [json.loads(l) for l in open(processed / "cohort.jsonl", encoding="utf-8") if l.strip()]
    tuning_authors = [a for a in cohort if a.get("split") == "tuning"]
    discovery = {}
    for line in open(processed / "ground_truth.jsonl", encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            discovery[clean_id(row["author_id"])] = [clean_id(p) for p in row["discovery_in_corpus"]]
    rc = RecommendConfig.from_config(cfg)
    sig = tuning_signature(rc, [a["author_id"] for a in tuning_authors])
    out = tuning_artifact_path(artifacts, sig)
    if out.exists() and not args.overwrite:
        print(f"{out} already exists.", file=sys.stderr)
        return 1

    run_id = f"core_tuning_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    manifest = RunManifest(run_id, cfg, runs_dir=root / "results" / "runs")
    try:
        resources = RecommenderResources.load(cfg, root, load_encoder=True, use_facets=True)
        if resources.facet_store is None:
            raise RuntimeError("facet_canon is empty: run on the machine with the full facets, as for Track 1.")
        cached = precompute_tuning_cache(resources, tuning_authors, discovery, rc)
        if len(cached) != len(tuning_authors):
            raise RuntimeError(f"Only {len(cached)} of {len(tuning_authors)} tuning authors could be cached.")
        points = grid_per_user_ndcg(cached, resources, adaptive_graph=rc.adaptive_graph, base_weights=rc.weights)
        result = tuning_result_from_points(points, len(cached), sig, rc.weights)
        result.hybrid_persona = tune_hybrid_persona(cached, resources, n_titles=rc.n_titles)

        legacy = artifacts / "tuning_results.json"
        check = None
        if legacy.exists():
            old = json.loads(legacy.read_text(encoding="utf-8"))
            same = (all(abs(old["best_weights"].get(k, -1) - getattr(result.best_weights, k)) < 1e-9
                        for k in ("w_cite", "w_facet", "w_nov")) and abs(old.get("best_lambda_div", -1) - result.best_lambda_div) < 1e-9)
            check = {"legacy_file": legacy.name, "same_optimum": same, "legacy_best_mean_ndcg": old.get("best_mean_ndcg"),
                     "recomputed_best_mean_ndcg": result.best_mean_ndcg}
            if not same:
                raise RuntimeError(f"The recomputed Core optimum differs from {legacy.name}: {check}")
        result.save(out)
        shutil.copy2(out, manifest.run_dir / out.name)
        manifest.complete(metadata={"output": str(out), "best_mean_ndcg": result.best_mean_ndcg, "legacy_check": check})
        print(f"Saved {out.name}: best NDCG@10 {result.best_mean_ndcg:.4f}, lambda_div {result.best_lambda_div}; legacy check {check}")
    except BaseException as exc:
        manifest.complete(status="FAILED", metadata={"error": repr(exc)})
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
