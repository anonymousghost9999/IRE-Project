#!/usr/bin/env python3
"""Tune the V2 cross-encoder feature and freeze the Core-vs-V2 decision on the tuning split (plan 10.2b-c).

Usage:
    python scripts/decide_cross_encoder.py [--n-bootstrap 10000] [--overwrite]
    python scripts/decide_cross_encoder.py --dry-run 10     # smoke test: first 10 tuning authors, writes nothing

Rule (design §13.8, V2 specification, fixed before implementation): both rerankers (configs/default.yaml
models.cross_encoder_primary / _fallback) are tuned on the §6.3 grid x w_ce in {0, 0.25, 0.5, 1}; the one with the
higher tuning-split NDCG@10 at its optimum is the V2 candidate, and it replaces Core only if the paired-bootstrap 95%
CI of its NDCG@10 gain over Core lies above zero (never with w_ce = 0). The reranker only rescores the Core pool.

Only the 80 tuning authors are touched. Outputs:
    data/artifacts/variant_v2_decision.json        the frozen decision (refuses to overwrite without --overwrite)
    data/artifacts/tuning_results_<hash>.json      the chosen reranker's tuning file, reused by --cross-encoder runs
    results/variant_v2_decision.md                 the readable table, with latency per feed
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paperfeed.common.config import load_config
from paperfeed.common.logger import get_logger
from paperfeed.common.manifest import RunManifest
from paperfeed.data.cohort import clean_id
from paperfeed.eval.tuning import (
    V2_GRID,
    decide_cross_encoder,
    find_tuning_artifact,
    grid_per_user_ndcg,
    latency_summary,
    precompute_tuning_cache,
    tune_hybrid_persona,
    tuning_artifact_path,
    tuning_result_from_points,
    tuning_signature,
    with_reranker_scores,
)
from paperfeed.rec.recommender import RecommendConfig, RecommenderResources
from paperfeed.rec.rerank import CrossEncoderScorer

logger = get_logger("scripts.decide_cross_encoder")

DECISION_NAME = "variant_v2_decision"


def hardware() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"GPU {torch.cuda.get_device_name(0)}"
    except Exception:  # torch missing or broken: report the CPU
        pass
    return f"CPU {platform.processor() or platform.machine()}"


def _fmt(p: dict) -> str:
    return (f"w_cite={p['w_cite']}, w_facet={p['w_facet']}, w_nov={p['w_nov']}, w_ce={p.get('w_ce', 0.0)}, "
            f"lambda_div={p['lambda_div']}")


def format_report(decision: dict) -> str:
    g = decision["ndcg_gain"]
    rel = f"{g['relative']:+.1%}" if g["relative"] is not None else "n/a"
    lines = [
        "# V2 Cross-Encoder Feature: Tuning-Split Decision\n",
        f"Rule (design §13.8, fixed before implementation): {decision['rule']['statement']} "
        f"{decision['n_users']} tuning authors, {decision['rule']['n_resamples']:,} paired-bootstrap resamples. "
        "The test split is not touched. Picking the better of two rerankers before the test slightly favours V2.\n",
        "| Arm | Grid optimum | Tuning NDCG@10 | Reranker latency per feed (mean / p95) |",
        "| :--- | :--- | :---: | :---: |",
        f"| Core | {_fmt(decision['core'])} | {decision['core']['mean_ndcg']:.4f} | - |",
    ]
    for model, p in decision["per_model_optimum"].items():
        lat = decision["latency"][model]
        mark = " (chosen)" if model == decision["chosen_model"] else ""
        lines.append(f"| V2: {model}{mark} | {_fmt(p)} | {p['mean_ndcg']:.4f} | {lat['mean_s']:.2f} s / {lat['p95_s']:.2f} s |")
    lines += [
        "",
        f"- NDCG@10 gain of the chosen reranker over Core: {g['mean_diff']:+.4f} ({rel}), 95% CI "
        f"[{g['ci'][0]:+.4f}, {g['ci'][1]:+.4f}], p = {g['p_value']:.4f}, Cohen's d = {g['cohens_d']:.3f}",
        f"- Latency: {decision['latency_hardware']}, {decision['pool_size']} candidates per feed",
        "",
        f"**Decision: V2 ({decision['chosen_model']}) replaces Core.**" if decision["adopt_v2"] else
        "**Decision: keep Core** (" + ("the chosen optimum does not use the reranker, w_ce = 0" if decision["v2"].get("w_ce", 0) == 0
                                       else "the CI of the NDCG@10 gain does not lie above zero") + ").",
        "",
        "*Either way V2 gets one Track 1 test run (`run_track1_eval.py --cross-encoder`) as a labelled exploratory "
        "variant (plan 10.2d), unless the chosen optimum has w_ce = 0, in which case it is Core and there is nothing "
        "to run.*",
    ]
    if decision.get("core_matches_track1_tuning") is False:
        lines.append(
            "\n> **Warning:** the Core grid optimum found here differs from the Core Track 1 tuning file "
            f"({decision.get('core_tuning_file')}); check that the data and code match the Track 1 run."
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the V2 cross-encoder decision on the tuning split.")
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

        core_rc = RecommendConfig.from_config(cfg)  # the headline Track 1 configuration
        author_ids = [a["author_id"] for a in tuning_authors]
        core_sig = tuning_signature(core_rc, author_ids)
        models = [cfg["models"]["cross_encoder_primary"], cfg["models"]["cross_encoder_fallback"]]

        resources = RecommenderResources.load(cfg, root, load_encoder=True, use_facets=True)
        if resources.facet_store is None and args.dry_run is None:
            raise RuntimeError("facet_canon is empty: tune on the machine with the full facets, as for Track 1.")

        logger.info("[core] caching candidate pools and features for %d tuning authors...", len(tuning_authors))
        core_cached = precompute_tuning_cache(resources, tuning_authors, discovery, core_rc)
        if len(core_cached) != len(tuning_authors):
            raise RuntimeError(f"Only {len(core_cached)} of {len(tuning_authors)} tuning authors could be cached.")
        core_points = grid_per_user_ndcg(core_cached, resources, adaptive_graph=core_rc.adaptive_graph,
                                         base_weights=core_rc.weights)

        model_cached, model_points, latency = {}, {}, {}
        for model in models:
            logger.info("[%s] scoring the cached pools of %d tuning authors...", model, len(core_cached))
            cached, seconds = with_reranker_scores(core_cached, resources, CrossEncoderScorer(model))
            latency[model] = latency_summary(seconds)
            logger.info("[%s] grid of %d points...", model,
                        len(V2_GRID["w_cite"]) * len(V2_GRID["w_facet"]) * len(V2_GRID["w_nov"]) * len(V2_GRID["w_ce"]) * len(V2_GRID["lambda_div"]))
            model_cached[model] = cached
            model_points[model] = grid_per_user_ndcg(cached, resources, grid=V2_GRID,
                                                     adaptive_graph=core_rc.adaptive_graph, base_weights=core_rc.weights)

        decision = decide_cross_encoder(core_points, model_points, core_cached, model_cached, n_resamples=args.n_bootstrap)
        decision["latency"] = latency
        decision["latency_hardware"] = hardware()
        decision["pool_size"] = core_rc.pool_size or resources.retriever.candidate_pool_size
        chosen = decision["chosen_model"]
        v2_rc = core_rc.with_overrides(cross_encoder=chosen)
        v2_sig = tuning_signature(v2_rc, author_ids)
        decision["config_signature"] = {"core": core_sig, "v2": v2_sig}

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

        # The chosen reranker's tuning file, so run_track1_eval.py --cross-encoder reuses exactly these weights.
        if hybrid is None:
            logger.info("No Core tuning file with Hybrid-Persona values; tuning the baseline on the Core cache...")
            hybrid = tune_hybrid_persona(core_cached, resources, n_titles=core_rc.n_titles)
        v2_tuning = tuning_result_from_points(model_points[chosen], len(core_cached), v2_sig, core_rc.weights)
        v2_tuning.hybrid_persona = hybrid
        v2_file = tuning_artifact_path(artifacts, v2_sig)
        v2_tuning.save(v2_file)
        decision["v2_tuning_file"] = v2_file.name

        decision_path.parent.mkdir(parents=True, exist_ok=True)
        decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
        out_md = root / args.out_dir / f"{DECISION_NAME}.md"
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(report, encoding="utf-8")
        for f in (decision_path, out_md, v2_file):
            shutil.copy2(f, manifest.run_dir / f.name)
        manifest.complete(metadata={
            "adopt_v2": decision["adopt_v2"],
            "chosen_model": chosen,
            "core": decision["core"],
            "v2": decision["v2"],
            "ndcg_gain": decision["ndcg_gain"],
            "latency": latency,
            "outputs": [str(decision_path), str(out_md), str(v2_file)],
        })
    except BaseException as exc:
        if manifest is not None:
            manifest.complete(status="FAILED", metadata={"error": repr(exc)})
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
