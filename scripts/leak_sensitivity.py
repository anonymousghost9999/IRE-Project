#!/usr/bin/env python3
"""Leak-free sensitivity analysis of the published results (design §18a): no re-runs, saved per-author results only.

Usage:
    python scripts/leak_sensitivity.py [--n-bootstrap 10000]

Needs the ORIGINAL cohort (data/processed/cohort.jsonl from before the 2026-11-15 rebuild; the script checks that its
test split is exactly the one in the published results and refuses otherwise), data/candidate_authors.jsonl (titles of
each author's paper B) and the database. Reads results/track1_lfo_results_sota_comparison.json (Core, its ablations, the
five baselines, the anchors and the Phase 11 systems, all on the original cohort; it reproduces the original Core run
exactly) and the four Phase 10 variant runs. Writes data/artifacts/leaked_authors_original_cohort.json,
results/leak_sensitivity.md / .json and a run manifest.
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
from paperfeed.data.db import DatabaseManager
from paperfeed.data.leakage import preprint_leaked_authors
from paperfeed.eval.leak_sensitivity import format_report, load_rows, run_sensitivity

VARIANT_RUNS = {
    "v1": "track1_lfo_results_encoder_bge.json",
    "v2": "track1_lfo_results_cross_encoder.json",
    "v3": "track1_lfo_results_learned_ranker.json",
    "v4": "track1_lfo_results_graph_channel.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Leak-free sensitivity analysis of the published results.")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = load_config()
    results = root / "results"
    run_id = f"leak_sensitivity_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    manifest = RunManifest(run_id, cfg, runs_dir=results / "runs")
    try:
        core_rows = load_rows(results / "track1_lfo_results_sota_comparison.json")
        variant_rows = {k: load_rows(results / f) for k, f in VARIANT_RUNS.items()}
        cohort = [json.loads(l) for l in open(root / cfg["paths"]["processed_dir"] / "cohort.jsonl", encoding="utf-8") if l.strip()]
        local_test = {clean_id(a["author_id"]) for a in cohort if a.get("split") == "test"}
        if local_test != {clean_id(a) for a in core_rows}:
            raise ValueError("data/processed/cohort.jsonl is not the original cohort of the published results "
                             "(its test split differs); this analysis needs the pre-rebuild cohort.")
        b_titles = {clean_id(c["author_id"]): c.get("target_paper_b_title", "")
                    for c in map(json.loads, open(root / "data" / "candidate_authors.jsonl", encoding="utf-8"))}
        missing = [clean_id(a["author_id"]) for a in cohort if not b_titles.get(clean_id(a["author_id"]))]
        if missing:
            raise ValueError(f"No title of paper B for {len(missing)} cohort authors (e.g. {missing[0]}).")
        db = DatabaseManager(root / cfg["paths"]["sqlite_db"], cutoff_year=int(cfg["project"]["cutoff_year"]))
        leaked = preprint_leaked_authors(db, cohort, b_titles)

        s = run_sensitivity(core_rows, variant_rows, sorted(leaked), n_resamples=args.n_bootstrap)
        report = format_report(s, leaked)
        artifacts = root / cfg["paths"]["artifacts_dir"]
        leaked_path = artifacts / "leaked_authors_original_cohort.json"
        leaked_path.write_text(json.dumps(leaked, indent=2), encoding="utf-8")
        md, js = results / "leak_sensitivity.md", results / "leak_sensitivity.json"
        md.write_text(report, encoding="utf-8")
        js.write_text(json.dumps(s, indent=2), encoding="utf-8")
        for f in (md, js, leaked_path):
            shutil.copy2(f, manifest.run_dir / f.name)
        changed = [r["comparison"] for k in ("h1", "h2", "variants", "sota") for r in s[k] if r["conclusion_changed"]]
        manifest.complete(metadata={"n_leaked": len(leaked), "n_test_clean": s["n_test_clean"], "conclusions_changed": changed,
                                    "outputs": [str(md), str(js), str(leaked_path)]})
        print(report)
    except BaseException as exc:
        manifest.complete(status="FAILED", metadata={"error": repr(exc)})
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
