#!/usr/bin/env python3
"""Cross-validate the V3 learned ranker against the linear scorer and freeze the decision (plan 10.3b-c).

Usage:
    python scripts/decide_learned_ranker.py [--n-bootstrap 10000] [--overwrite]
    python scripts/decide_learned_ranker.py --dry-run 15      # smoke test: first 15 tuning authors, writes nothing

Rule (design §13.8, V3 specification, fixed before implementation): 5 author folds inside the tuning split; in each,
the linear scorer (§9 grid) and V3 (LightGBM LambdaMART with fixed settings, lambda_div chosen on the training folds)
are fitted on the same training authors and scored on the held-out ones. V3 replaces the linear scorer only if the
paired-bootstrap 95% CI of its out-of-fold NDCG@10 gain lies above zero.

Only the 80 tuning authors are touched. Outputs:
    data/artifacts/variant_v3_decision.json   the frozen decision (refuses to overwrite without --overwrite)
    data/artifacts/variant_v3_model.txt       the final model, trained on all tuning authors (LightGBM text format)
    results/variant_v3_decision.md            the readable table, per fold
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paperfeed.common.config import load_config
from paperfeed.common.logger import get_logger
from paperfeed.common.manifest import RunManifest
from paperfeed.data.cohort import clean_id
from paperfeed.eval.learned_cv import cross_validate, decide_learned_ranker, fit_v3
from paperfeed.eval.tuning import DEFAULT_TRACK1_GRID, find_tuning_artifact, precompute_tuning_cache, tuning_signature
from paperfeed.rec.learned import V3_FEATURES, V3_PARAMS
from paperfeed.rec.recommender import RecommendConfig, RecommenderResources

logger = get_logger("scripts.decide_learned_ranker")

DECISION_NAME = "variant_v3_decision"
MODEL_NAME = "variant_v3_model.txt"


def format_report(decision: dict) -> str:
    g = decision["ndcg_gain"]
    rel = f"{g['relative']:+.1%}" if g["relative"] is not None else "n/a"
    lines = [
        "# V3 Learned Ranker: Tuning-Split Decision\n",
        f"Rule (design §13.8, fixed before implementation): {decision['rule']['statement']} "
        f"{decision['n_users']} tuning authors, {decision['rule']['n_resamples']:,} paired-bootstrap resamples. "
        "The test split is not touched.\n",
        f"- Model: LightGBM lambdarank with fixed settings {json.dumps({k: v for k, v in decision['params'].items() if k not in ('verbose', 'n_jobs', 'force_row_wise')})}",
        f"- Features: {', '.join(decision['features'])}",
        "",
        "| Fold | Held-out authors | Linear optimum (fitted on the other folds) | V3 lambda_div | Linear NDCG@10 | V3 NDCG@10 |",
        "| :---: | :---: | :--- | :---: | :---: | :---: |",
    ]
    for f in decision["folds"]:
        o = f["linear_optimum"]
        lines.append(
            f"| {f['fold']} | {len(f['held_out_authors'])} | w_cite={o['w_cite']}, w_facet={o['w_facet']}, w_nov={o['w_nov']}, "
            f"lambda_div={o['lambda_div']} | {f['v3_lambda_div']} | {f['linear_mean_ndcg']:.4f} | {f['v3_mean_ndcg']:.4f} |"
        )
    lines += [
        "",
        f"- Out-of-fold NDCG@10: linear {decision['linear_oof_ndcg']:.4f}, V3 {decision['v3_oof_ndcg']:.4f}",
        f"- Gain (V3 - linear): {g['mean_diff']:+.4f} ({rel}), 95% CI [{g['ci'][0]:+.4f}, {g['ci'][1]:+.4f}], "
        f"p = {g['p_value']:.4f}, Cohen's d = {g['cohens_d']:.3f}",
        f"- Final model (all {decision['n_users']} tuning authors): lambda_div = {decision['final_lambda_div']}, "
        f"feature importance (splits) {decision['final_feature_importance']}",
        "",
        "**Decision: V3 replaces the linear scorer.**" if decision["adopt_v3"] else
        "**Decision: keep the linear scorer** (the CI of the out-of-fold NDCG@10 gain does not lie above zero).",
        "",
        "*Either way V3 gets one Track 1 test run (`run_track1_eval.py --learned-ranker`) as a labelled exploratory "
        "variant (plan 10.3d), with the final model above.*",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the V3 learned-ranker decision on the tuning split.")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing frozen decision")
    parser.add_argument("--dry-run", type=int, default=None, metavar="N",
                        help="Smoke test on the first N tuning authors: prints the report, writes no artifact")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = load_config()
    artifacts = root / cfg["paths"]["artifacts_dir"]
    decision_path, model_path = artifacts / f"{DECISION_NAME}.json", artifacts / MODEL_NAME
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

        core_rc = RecommendConfig.from_config(cfg)  # the headline Track 1 configuration (Core pool and features)
        author_ids = [a["author_id"] for a in tuning_authors]
        resources = RecommenderResources.load(cfg, root, load_encoder=True, use_facets=True)
        if resources.facet_store is None and args.dry_run is None:
            raise RuntimeError("facet_canon is empty: run on the machine with the full facets, as for Track 1.")
        channel_depth = core_rc.channel_depth or resources.retriever.channel_depth

        logger.info("Caching candidate pools and features for %d tuning authors...", len(tuning_authors))
        cached = precompute_tuning_cache(resources, tuning_authors, discovery, core_rc)
        if len(cached) != len(tuning_authors):
            raise RuntimeError(f"Only {len(cached)} of {len(tuning_authors)} tuning authors could be cached.")

        cv = cross_validate(cached, resources, channel_depth, core_rc.weights, adaptive_graph=core_rc.adaptive_graph)
        decision = decide_learned_ranker(cv, n_resamples=args.n_bootstrap)
        model, lam = fit_v3(cached, resources.store, channel_depth, DEFAULT_TRACK1_GRID["lambda_div"])
        decision.update({
            "features": list(V3_FEATURES),
            "params": dict(V3_PARAMS),
            "channel_depth": channel_depth,
            "final_lambda_div": lam,
            "final_feature_importance": {f: int(v) for f, v in zip(V3_FEATURES, model.feature_importances_)},
            "config_signature": tuning_signature(core_rc, author_ids),
        })
        core_file = find_tuning_artifact(artifacts, decision["config_signature"])
        decision["core_tuning_file"] = core_file.name if core_file else None

        report = format_report(decision)
        print(report)
        if args.dry_run is not None:
            logger.info("Dry run: nothing written.")
            return 0

        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(model_path))
        decision["model_file"] = model_path.name
        decision["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
        decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
        out_md = root / args.out_dir / f"{DECISION_NAME}.md"
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(report, encoding="utf-8")
        for f in (decision_path, model_path, out_md):
            shutil.copy2(f, manifest.run_dir / f.name)
        manifest.complete(metadata={
            "adopt_v3": decision["adopt_v3"],
            "linear_oof_ndcg": decision["linear_oof_ndcg"],
            "v3_oof_ndcg": decision["v3_oof_ndcg"],
            "ndcg_gain": decision["ndcg_gain"],
            "final_lambda_div": lam,
            "model_sha256": decision["model_sha256"],
            "outputs": [str(decision_path), str(model_path), str(out_md)],
        })
    except BaseException as exc:
        if manifest is not None:
            manifest.complete(status="FAILED", metadata={"error": repr(exc)})
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
