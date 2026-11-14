#!/usr/bin/env python3
"""Phase 11b: tune RP3β on the Track 1 tuning split (docs/design.md §13.9 Phase 11 specification point 4).

Usage:
    python scripts/tune_rp3beta.py [--overwrite]

Grid fixed in advance: alpha in {0.5, 1.0, 1.5}, beta in {0, 0.25, 0.5, 0.75} (beta = 0 is P3α). Same eligible
papers, E_u and NDCG@10 as Track 1; 80 tuning authors only. Needs only the database (no facets, no encoder).
Outputs data/artifacts/sota_rp3beta_tuning.json (frozen; --overwrite to redo) and results/sota_rp3beta_tuning.md.
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

from paperfeed.common.config import load_config
from paperfeed.common.manifest import RunManifest
from paperfeed.data.cohort import clean_id
from paperfeed.data.db import DatabaseManager
from paperfeed.eval.sota import RP3_GRID, CitationGraph, RP3Author, tune_rp3beta
from paperfeed.rec.candidates import eligible_paper_ids
from paperfeed.user.exclusion import load_exclusion_sources

NAME = "sota_rp3beta_tuning"


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune RP3β on the tuning split.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = load_config()
    artifacts = root / cfg["paths"]["artifacts_dir"]
    out_json = artifacts / f"{NAME}.json"
    if out_json.exists() and not args.overwrite:
        print(f"{out_json} already exists: the tuning is frozen. Pass --overwrite only if it must be redone.", file=sys.stderr)
        return 1
    run_id = f"{NAME}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    manifest = RunManifest(run_id, cfg, runs_dir=root / "results" / "runs")
    manifest.extra_metadata["arguments"] = vars(args)
    try:
        cutoff = int(cfg["project"]["cutoff_year"])
        db = DatabaseManager(root / cfg["paths"]["sqlite_db"], cutoff_year=cutoff)
        processed = root / cfg["paths"]["processed_dir"]
        cohort = [json.loads(l) for l in open(processed / "cohort.jsonl", encoding="utf-8") if l.strip()]
        discovery = {}
        for line in open(processed / "ground_truth.jsonl", encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                discovery[clean_id(row["author_id"])] = [clean_id(p) for p in row["discovery_in_corpus"]]

        t0 = time.time()
        graph = CitationGraph.from_db(db, cutoff)
        allowed = eligible_paper_ids(db, cutoff)
        authors = []
        for a in cohort:
            if a.get("split") != "tuning":
                continue
            aid = clean_id(a["author_id"])
            authored = [clean_id(p) for p in a.get("authored_paper_ids_T", [])]
            excluded = load_exclusion_sources(db, authored).build(diagnostic=False, zero_pub=False)
            authors.append(RP3Author(aid, authored, excluded, discovery.get(aid, [])))
        res = tune_rp3beta(graph, authors, allowed)
        res["graph"] = {"n_papers": len(graph.ids), "n_edges": int(graph.A.nnz), "cutoff_year": cutoff}
        res["seconds"] = round(time.time() - t0, 1)

        lines = ["# Phase 11b: RP3β Tuning (tuning split)\n",
                 f"Grid fixed in advance (design §13.9): alpha {RP3_GRID['alpha']}, beta {RP3_GRID['beta']} (beta = 0 is "
                 f"P3α). {res['n_users']} tuning authors; citation graph of {len(graph.ids):,} papers and "
                 f"{graph.A.nnz:,} edges (citing year <= {cutoff}). The test split is not touched.\n",
                 "| alpha | beta | Tuning NDCG@10 |", "| :---: | :---: | :---: |"]
        for r in res["points"]:
            mark = " **(chosen)**" if r == res["best"] else ""
            lines.append(f"| {r['alpha']} | {r['beta']} | {r['mean_ndcg']:.4f}{mark} |")
        report = "\n".join(lines) + "\n"
        out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
        out_md = root / args.out_dir / f"{NAME}.md"
        out_md.write_text(report, encoding="utf-8")
        for f in (out_json, out_md):
            shutil.copy2(f, manifest.run_dir / f.name)
        manifest.complete(metadata={"best": res["best"], "n_users": res["n_users"], "outputs": [str(out_json), str(out_md)]})
        print(report)
    except BaseException as exc:
        manifest.complete(status="FAILED", metadata={"error": repr(exc)})
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
