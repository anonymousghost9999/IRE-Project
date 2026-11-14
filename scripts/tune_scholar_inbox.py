#!/usr/bin/env python3
"""Phase 11c: tune the Scholar Inbox-style classifier's C on the tuning split (design §13.9 Phase 11 spec point 4).

Usage:
    python scripts/build_encoder_index.py --comparison --encoder gte     # once: GTE-large full-corpus vectors
    python scripts/tune_scholar_inbox.py [--overwrite]

GTE-large vectors of the corpus, PCA to 256 dimensions (fitted on the whole corpus, seed 42), one logistic regression
per tuning author (positives: E_u papers with a vector; 10 seeded eligible negatives per positive). Grid fixed in
advance: C in {0.01, 0.1, 1, 10}. Needs only the database and the GTE index (no facets). Outputs
data/artifacts/sota_scholar_inbox_tuning.json (frozen; --overwrite to redo) and results/sota_scholar_inbox_tuning.md.
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

import numpy as np

from paperfeed.common.config import load_config
from paperfeed.common.manifest import RunManifest
from paperfeed.data.cohort import clean_id
from paperfeed.data.db import DatabaseManager
from paperfeed.eval.metrics import ndcg_at_k
from paperfeed.eval.sota import NEGATIVES_PER_POSITIVE, PCA_DIM, SCHOLAR_INBOX_C, fit_pca_space, scholar_inbox_recommend
from paperfeed.rec.candidates import eligible_paper_ids
from paperfeed.rec.recommender import load_encoder_index
from paperfeed.user.exclusion import load_exclusion_sources

NAME = "sota_scholar_inbox_tuning"


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune the Scholar Inbox-style classifier on the tuning split.")
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
    try:
        t0 = time.time()
        cutoff = int(cfg["project"]["cutoff_year"])
        db = DatabaseManager(root / cfg["paths"]["sqlite_db"], cutoff_year=cutoff)
        processed = root / cfg["paths"]["processed_dir"]
        store, _ = load_encoder_index(processed, "gte")
        ids = list(store.paper_id_index)
        space = fit_pca_space(ids, store.select(ids))
        eligible = eligible_paper_ids(db, cutoff)
        allowed_idx = np.array(sorted(space.index[p] for p in eligible if p in space.index), dtype=np.int64)

        cohort = [json.loads(l) for l in open(processed / "cohort.jsonl", encoding="utf-8") if l.strip()]
        discovery = {}
        for line in open(processed / "ground_truth.jsonl", encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                discovery[clean_id(row["author_id"])] = [clean_id(p) for p in row["discovery_in_corpus"]]
        authors = []
        for a in cohort:
            if a.get("split") == "tuning":
                aid = clean_id(a["author_id"])
                authored = [clean_id(p) for p in a.get("authored_paper_ids_T", [])]
                authors.append((aid, load_exclusion_sources(db, authored).build(), discovery.get(aid, [])))

        points = []
        for C in SCHOLAR_INBOX_C:
            nd = [ndcg_at_k(scholar_inbox_recommend(space, aid, excl, allowed_idx, C), tg, k=10) for aid, excl, tg in authors]
            points.append({"C": C, "mean_ndcg": float(np.mean(nd))})
            print(f"C={C}: {points[-1]['mean_ndcg']:.4f}", flush=True)
        best = max(points, key=lambda r: r["mean_ndcg"])  # ties: the first (smallest C)
        res = {"grid": {"C": SCHOLAR_INBOX_C}, "points": points, "best": best, "n_users": len(authors),
               "embedding": {"model": "thenlper/gte-large", "pca_dim": PCA_DIM, "explained_variance": space.explained_variance,
                             "n_papers": len(ids)}, "negatives_per_positive": NEGATIVES_PER_POSITIVE,
               "seconds": round(time.time() - t0, 1)}
        lines = ["# Phase 11c: Scholar Inbox-style Classifier Tuning (tuning split)\n",
                 f"Grid fixed in advance (design §13.9): C {SCHOLAR_INBOX_C}. GTE-large vectors of {len(ids):,} papers, "
                 f"PCA-{PCA_DIM} (explained variance {space.explained_variance:.1%}); per author a logistic regression on E_u "
                 f"positives vs {NEGATIVES_PER_POSITIVE} seeded eligible negatives each. {len(authors)} tuning authors; the "
                 "test split is not touched.\n", "| C | Tuning NDCG@10 |", "| :---: | :---: |"]
        lines += [f"| {p['C']} | {p['mean_ndcg']:.4f}{' **(chosen)**' if p is best else ''} |" for p in points]
        report = "\n".join(lines) + "\n"
        out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
        out_md = root / args.out_dir / f"{NAME}.md"
        out_md.write_text(report, encoding="utf-8")
        for f in (out_json, out_md):
            shutil.copy2(f, manifest.run_dir / f.name)
        manifest.complete(metadata={"best": best, "n_users": len(authors), "outputs": [str(out_json), str(out_md)]})
        print(report)
    except BaseException as exc:
        manifest.complete(status="FAILED", metadata={"error": repr(exc)})
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
