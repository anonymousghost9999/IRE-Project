#!/usr/bin/env python3
"""Tune the V4 citation-graph channel and freeze the Core-vs-V4 decision on the tuning split (plan 10.4b-c).

Usage:
    python scripts/decide_graph_channel.py [--n-bootstrap 10000] [--overwrite]
    python scripts/decide_graph_channel.py --dry-run 15      # smoke test: first 15 tuning authors, writes nothing

Rule (design §13.8, fixed before this script existed): V4 replaces the Core candidate pool only if the
paired-bootstrap 95% CI of its tuning-split mean Discovery NDCG@10 gain over Core, each at its own grid optimum,
lies above zero. Candidate Recall@100 is reported as the mechanism check.

Only the 80 tuning authors are touched. Outputs:
    data/artifacts/variant_v4_decision.json        the frozen decision (refuses to overwrite without --overwrite)
    data/artifacts/tuning_results_<hash>.json      the V4 tuning file, reused by run_track1_eval.py --graph-channel
    results/variant_v4_decision.md                 the readable table
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
    decide_graph_channel,
    find_tuning_artifact,
    grid_per_user_ndcg,
    precompute_tuning_cache,
    tune_hybrid_persona,
    tuning_artifact_path,
    tuning_result_from_points,
    tuning_signature,
)
from paperfeed.rec.recommender import RecommendConfig, RecommenderResources

logger = get_logger("scripts.decide_graph_channel")

DECISION_NAME = "variant_v4_decision"


def _fmt_point(p: dict) -> str:
    return f"w_cite={p['w_cite']}, w_facet={p['w_facet']}, w_nov={p['w_nov']}, lambda_div={p['lambda_div']}"


def format_report(decision: dict) -> str:
    g, r = decision["ndcg_gain"], decision["candidate_recall_gain"]
    core, v4 = decision["core"], decision["v4"]
    rel = f"{g['relative']:+.1%}" if g["relative"] is not None else "n/a"
    lines = [
        "# V4 Citation-Graph Channel: Tuning-Split Decision\n",
        f"Rule (design §13.8, fixed before implementation): {decision['rule']['statement']} "
        f"{decision['n_users']} tuning authors, {decision['rule']['n_resamples']:,} paired-bootstrap resamples. "
        "The test split is not touched.\n",
        "| Arm | Grid optimum | Tuning NDCG@10 | Candidate Recall@100 |",
        "| :--- | :--- | :---: | :---: |",
        f"| Core (text channels) | {_fmt_point(core)} | {core['mean_ndcg']:.4f} | {core['candidate_recall_at_100']:.4f} |",
        f"| V4 (+ citation-graph channel) | {_fmt_point(v4)} | {v4['mean_ndcg']:.4f} | {v4['candidate_recall_at_100']:.4f} |",
        "",
        f"- NDCG@10 gain (V4 - Core): {g['mean_diff']:+.4f} ({rel}), 95% CI [{g['ci'][0]:+.4f}, {g['ci'][1]:+.4f}], "
        f"p = {g['p_value']:.4f}, Cohen's d = {g['cohens_d']:.3f}",
        f"- Candidate Recall@100 gain (mechanism check): {r['mean_diff']:+.4f}, 95% CI [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]",
        "",
        "**Decision: V4 replaces the Core candidate pool.**" if decision["adopt_v4"] else
        "**Decision: keep the Core candidate pool** (the CI of the NDCG@10 gain does not lie above zero).",
        "",
        "*Either way V4 gets one Track 1 test run (`run_track1_eval.py --graph-channel`) as a labelled exploratory "
        "variant (plan 10.4d); the Core test run remains the answer to H1.*",
    ]
    if decision.get("core_matches_track1_tuning") is False:
        lines.append(
            "\n> **Warning:** the Core grid optimum found here differs from the Core Track 1 tuning file "
            f"({decision.get('core_tuning_file')}); check that the data and code match the Track 1 run."
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the V4 graph-channel decision on the tuning split.")
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

        # Same base configuration as the headline Track 1 run (run_track1_eval.py without options)
        core_rc = RecommendConfig.from_config(cfg)
        v4_rc = core_rc.with_overrides(graph_channel=True)
        author_ids = [a["author_id"] for a in tuning_authors]
        core_sig, v4_sig = tuning_signature(core_rc, author_ids), tuning_signature(v4_rc, author_ids)

        resources = RecommenderResources.load(cfg, root, load_encoder=True, use_facets=True)
        if resources.facet_store is None and args.dry_run is None:
            raise RuntimeError("facet_canon is empty: tune on the machine with the full facets, as for Track 1.")

        arms = {}
        for name, rc in (("core", core_rc), ("v4", v4_rc)):
            logger.info("[%s] caching candidate pools and features for %d tuning authors...", name, len(tuning_authors))
            cached = precompute_tuning_cache(resources, tuning_authors, discovery, rc)
            if len(cached) != len(tuning_authors):
                raise RuntimeError(f"[{name}] only {len(cached)} of {len(tuning_authors)} tuning authors could be cached.")
            points = grid_per_user_ndcg(cached, resources, adaptive_graph=rc.adaptive_graph, base_weights=rc.weights)
            arms[name] = (cached, points)

        decision = decide_graph_channel(
            arms["core"][1], arms["v4"][1], arms["core"][0], arms["v4"][0], n_resamples=args.n_bootstrap
        )
        decision["config_signature"] = {"core": core_sig, "v4": v4_sig}

        # Sanity check: the Core optimum must reproduce the Core Track 1 tuning (same data, same code path)
        core_file = find_tuning_artifact(artifacts, core_sig)
        if core_file is not None:
            saved = json.loads(core_file.read_text(encoding="utf-8"))
            same = (
                all(abs(saved["best_weights"].get(k, -1) - decision["core"][k]) < 1e-9 for k in ("w_cite", "w_facet", "w_nov"))
                and abs(saved.get("best_lambda_div", -1) - decision["core"]["lambda_div"]) < 1e-9
            )
            decision["core_tuning_file"], decision["core_matches_track1_tuning"] = core_file.name, same
            hybrid = saved.get("hybrid_persona")
        else:
            decision["core_tuning_file"], decision["core_matches_track1_tuning"] = None, None
            hybrid = None

        report = format_report(decision)
        print(report)
        if args.dry_run is not None:
            logger.info("Dry run: nothing written.")
            return 0

        # V4 tuning file, so run_track1_eval.py --graph-channel reuses exactly these weights. The Hybrid-Persona
        # baseline has no candidate pool of its own to change, so it keeps the Core run's tuned values.
        if hybrid is None:
            logger.info("No Core tuning file with Hybrid-Persona values; tuning the baseline on the Core cache...")
            hybrid = tune_hybrid_persona(arms["core"][0], resources, n_titles=core_rc.n_titles)
        v4_tuning = tuning_result_from_points(arms["v4"][1], len(arms["v4"][0]), v4_sig, v4_rc.weights)
        v4_tuning.hybrid_persona = hybrid
        v4_file = tuning_artifact_path(artifacts, v4_sig)
        v4_tuning.save(v4_file)
        decision["v4_tuning_file"] = v4_file.name

        decision_path.parent.mkdir(parents=True, exist_ok=True)
        decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
        out_md = root / args.out_dir / f"{DECISION_NAME}.md"
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(report, encoding="utf-8")
        for f in (decision_path, out_md, v4_file):
            shutil.copy2(f, manifest.run_dir / f.name)
        manifest.complete(metadata={
            "adopt_v4": decision["adopt_v4"],
            "core": decision["core"],
            "v4": decision["v4"],
            "ndcg_gain": decision["ndcg_gain"],
            "outputs": [str(decision_path), str(out_md), str(v4_file)],
        })
    except BaseException as exc:
        if manifest is not None:
            manifest.complete(status="FAILED", metadata={"error": repr(exc)})
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
