#!/usr/bin/env python3
"""Phase 11a: decide the candidate-pool size on the tuning split (docs/design.md §13.9 Phase 11 specification point 2).

Usage:
    python scripts/decide_pool_size.py [--n-bootstrap 10000] [--overwrite]
    python scripts/decide_pool_size.py --dry-run 10       # smoke test: first 10 tuning authors, writes nothing

Rule (fixed before implementation): the larger pool (M = 250, M_channel = 500, no popularity prior) replaces the Core
pool (M = 100, M_channel = 200) only if the paired-bootstrap 95% CI of its tuning-split NDCG@10 gain, each arm at its
own §9 grid optimum, lies above zero. The "enhanced" Track 1 run also used a popularity prior, which §9 excludes, so it
is never a selection candidate.

Only the 80 tuning authors are touched. Outputs:
    data/artifacts/pool_size_decision.json     the frozen decision (refuses to overwrite without --overwrite)
    data/artifacts/tuning_results_<hash>.json  the larger pool's tuning file (used if it is adopted)
    results/pool_size_decision.md              the readable table
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
from paperfeed.common.logger import get_logger
from paperfeed.common.manifest import RunManifest
from paperfeed.data.cohort import clean_id
from paperfeed.eval.tuning import (
    POOL_ALTERNATIVE,
    decide_pool_size,
    find_tuning_artifact,
    grid_per_user_ndcg,
    precompute_tuning_cache,
    tune_hybrid_persona,
    tuning_artifact_path,
    tuning_result_from_points,
    tuning_signature,
)
from paperfeed.rec.recommender import RecommendConfig, RecommenderResources

logger = get_logger("scripts.decide_pool_size")

DECISION_NAME = "pool_size_decision"


def _fmt(p: dict) -> str:
    return f"w_cite={p['w_cite']}, w_facet={p['w_facet']}, w_nov={p['w_nov']}, lambda_div={p['lambda_div']}"


def format_report(d: dict) -> str:
    g, r = d["ndcg_gain"], d["candidate_recall_gain"]
    rel = f"{g['relative']:+.1%}" if g["relative"] is not None else "n/a"
    core, alt = d["core"], d["larger_pool"]
    lines = [
        "# Phase 11a: Candidate-Pool Size, Tuning-Split Decision\n",
        f"Rule (design §13.9, fixed before implementation): {d['rule']['statement']} {d['n_users']} tuning authors, "
        f"{d['rule']['n_resamples']:,} paired-bootstrap resamples. The test split is not touched.\n",
        "| Arm | Pool | Grid optimum | Tuning NDCG@10 | Candidate Recall@k |",
        "| :--- | :--- | :--- | :---: | :---: |",
        f"| Core | M = 100, M_channel = 200 | {_fmt(core)} | {core['mean_ndcg']:.4f} | {core['candidate_recall_at_100']:.4f} |",
        f"| Larger pool | M = 250, M_channel = 500, w_pop = 0 | {_fmt(alt)} | {alt['mean_ndcg']:.4f} | "
        f"{alt['candidate_recall_at_100']:.4f} |",
        "",
        f"- NDCG@10 gain (larger - Core): {g['mean_diff']:+.4f} ({rel}), 95% CI [{g['ci'][0]:+.4f}, {g['ci'][1]:+.4f}], "
        f"p = {g['p_value']:.4f}, Cohen's d = {g['cohens_d']:.3f}",
        f"- Candidate recall gain (mechanism check; the pool sizes differ): {r['mean_diff']:+.4f}, 95% CI "
        f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]",
        "",
        "**Decision: the larger pool replaces the Core pool.**" if d["adopt_larger_pool"] else
        "**Decision: keep the Core pool** (the CI of the NDCG@10 gain does not lie above zero).",
        "",
        "*This is a single change, so it is also the §13.9 assembled-system check. The decision goes into "
        "`configs/best.yaml` (scripts/freeze_best_config.py) before any comparison system runs on the test split.*",
    ]
    if d.get("core_matches_track1_tuning") is False:
        lines.append(f"\n> **Warning:** the Core optimum here differs from {d.get('core_tuning_file')}; check data and code.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 11a: decide the candidate-pool size on the tuning split.")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing frozen decision")
    parser.add_argument("--dry-run", type=int, default=None, metavar="N",
                        help="Smoke test on the first N tuning authors: prints the report, writes no artifact")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = load_config()
    artifacts = root / cfg["paths"]["artifacts_dir"]
    decision_path = artifacts / f"{DECISION_NAME}.json"
    if args.dry_run is None and decision_path.exists() and not args.overwrite:
        print(f"{decision_path} already exists: the decision is frozen. Pass --overwrite only if it must be redone "
              "(and say so in the report).", file=sys.stderr)
        return 1

    manifest = None
    if args.dry_run is None:
        run_id = f"{DECISION_NAME}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        manifest = RunManifest(run_id, cfg, runs_dir=root / "results" / "runs")
        manifest.extra_metadata["arguments"] = vars(args)
    try:
        processed = root / cfg["paths"]["processed_dir"]
        cohort = [json.loads(l) for l in open(processed / "cohort.jsonl", encoding="utf-8") if l.strip()]
        tuning_authors = [a for a in cohort if a.get("split") == "tuning"]
        if args.dry_run is not None:
            tuning_authors = tuning_authors[: args.dry_run]
        discovery = {}
        for line in open(processed / "ground_truth.jsonl", encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                discovery[clean_id(row["author_id"])] = [clean_id(p) for p in row["discovery_in_corpus"]]

        core_rc = RecommendConfig.from_config(cfg)
        if core_rc.weights.w_pop != 0.0:
            raise ValueError("configs/default.yaml has a popularity prior; the Core score must not (design §9).")
        alt_rc = core_rc.with_overrides(**POOL_ALTERNATIVE)
        author_ids = [a["author_id"] for a in tuning_authors]
        core_sig, alt_sig = tuning_signature(core_rc, author_ids), tuning_signature(alt_rc, author_ids)

        resources = RecommenderResources.load(cfg, root, load_encoder=True, use_facets=True)
        if resources.facet_store is None and args.dry_run is None:
            raise RuntimeError("facet_canon is empty: run on the machine with the full facets, as for Track 1.")

        arms = {}
        for name, rc in (("core", core_rc), ("larger_pool", alt_rc)):
            logger.info("[%s] caching pools (M=%s, M_channel=%s) for %d tuning authors...", name, rc.pool_size,
                        rc.channel_depth, len(tuning_authors))
            cached = precompute_tuning_cache(resources, tuning_authors, discovery, rc)
            if len(cached) != len(tuning_authors):
                raise RuntimeError(f"[{name}] only {len(cached)} of {len(tuning_authors)} tuning authors could be cached.")
            arms[name] = (cached, grid_per_user_ndcg(cached, resources, adaptive_graph=rc.adaptive_graph,
                                                     base_weights=rc.weights))

        decision = decide_pool_size(arms["core"][1], arms["larger_pool"][1], arms["core"][0], arms["larger_pool"][0],
                                    n_resamples=args.n_bootstrap)
        decision["config_signature"] = {"core": core_sig, "larger_pool": alt_sig}
        core_file = find_tuning_artifact(artifacts, core_sig)
        hybrid = None
        if core_file is not None:
            saved = json.loads(core_file.read_text(encoding="utf-8"))
            decision["core_tuning_file"] = core_file.name
            decision["core_matches_track1_tuning"] = (
                all(abs(saved["best_weights"].get(k, -1) - decision["core"][k]) < 1e-9 for k in ("w_cite", "w_facet", "w_nov"))
                and abs(saved.get("best_lambda_div", -1) - decision["core"]["lambda_div"]) < 1e-9
            )
            hybrid = saved.get("hybrid_persona")
        else:
            decision["core_tuning_file"], decision["core_matches_track1_tuning"] = None, None

        report = format_report(decision)
        print(report)
        if args.dry_run is not None:
            logger.info("Dry run: nothing written.")
            return 0

        if hybrid is None:
            hybrid = tune_hybrid_persona(arms["core"][0], resources, n_titles=core_rc.n_titles)
        alt_tuning = tuning_result_from_points(arms["larger_pool"][1], len(arms["larger_pool"][0]), alt_sig, alt_rc.weights)
        alt_tuning.hybrid_persona = hybrid
        alt_file = tuning_artifact_path(artifacts, alt_sig)
        alt_tuning.save(alt_file)
        decision["larger_pool_tuning_file"] = alt_file.name

        decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
        out_md = root / args.out_dir / f"{DECISION_NAME}.md"
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(report, encoding="utf-8")
        for f in (decision_path, out_md, alt_file):
            shutil.copy2(f, manifest.run_dir / f.name)
        manifest.complete(metadata={"adopt_larger_pool": decision["adopt_larger_pool"], "core": decision["core"],
                                    "larger_pool": decision["larger_pool"], "ndcg_gain": decision["ndcg_gain"],
                                    "outputs": [str(decision_path), str(out_md), str(alt_file)]})
    except BaseException as exc:
        if manifest is not None:
            manifest.complete(status="FAILED", metadata={"error": repr(exc)})
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
