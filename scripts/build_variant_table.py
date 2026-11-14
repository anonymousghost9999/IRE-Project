#!/usr/bin/env python3
"""Phase 10 final variant table (plan 10.5), built from saved Track 1 results and the frozen decisions only.

Usage:
    python scripts/build_variant_table.py [--n-bootstrap 10000]

Reads results/track1_lfo_results*.json (Core, --encoder bge, --cross-encoder, --learned-ranker, --graph-channel) and
data/artifacts/variant_v*_decision.json. Writes results/variant_table.md / .json and a run manifest.
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
from paperfeed.eval.variant_table import VARIANTS, build_variant_table, format_variant_table


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Phase 10 final variant table.")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = load_config()
    results_dir = root / args.out_dir
    run_id = f"variant_table_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    manifest = RunManifest(run_id, cfg, runs_dir=root / "results" / "runs")
    manifest.extra_metadata["arguments"] = vars(args)
    try:
        table = build_variant_table(results_dir, root / cfg["paths"]["artifacts_dir"], args.n_bootstrap)
        md = format_variant_table(table)
        md_path, json_path = results_dir / "variant_table.md", results_dir / "variant_table.json"
        md_path.write_text(md, encoding="utf-8")
        json_path.write_text(json.dumps(table, indent=2), encoding="utf-8")
        for f in (md_path, json_path):
            shutil.copy2(f, manifest.run_dir / f.name)
        manifest.complete(metadata={"inputs": [v.results_file for v in VARIANTS] + [v.decision_file for v in VARIANTS if v.decision_file],
                                    "comparisons": table["comparisons"]})
        print(md)
    except BaseException as exc:
        manifest.complete(status="FAILED", metadata={"error": repr(exc)})
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
