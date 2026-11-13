#!/usr/bin/env python3
"""Phase 8 explanation faithfulness audit (docs/plan.md §8 / docs/design.md §13.6).

Usage:
    python scripts/audit_faithfulness.py [--enhanced] [--facet-active] [--filter-popular THRESHOLD]
                                         [--sample-size 100] [--seed 42]

Audits deterministic recommendations against two criteria (prob.md §2.4, §6E.5):
1. Extractive Grounding Rate: Displayed facet spans and evidence sentences exist verbatim at their character offsets
   in the candidate paper's abstract text (target: 100%); citation edges are verified in SQLite under year <= T.
2. Counterfactual Necessity (Drop-out Rate): When the explained feature contribution is set to 0 (DeltaScore = 0),
   the paper leaves the Top-10 or falls >= 3 ranks under MMR re-diversification (verifying the explanation highlights
   a necessary ranking signal).
3. Explanation Coverage: Percentage of Top-10 recommendations successfully annotated with a grounded explanation (target >= 90%).
4. Template Distribution: Proportions of explanation templates across the evaluated recommendations.
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

from paperfeed.common.config import find_project_root, load_config
from paperfeed.common.logger import get_logger
from paperfeed.common.manifest import RunManifest
from paperfeed.eval.faithfulness import run_faithfulness_audit
from paperfeed.eval.tuning import (
    default_facet_selection_path,
    load_facet_selection,
    tuning_signature,
)
from paperfeed.rec.recommender import RecommendConfig, RecommenderResources
from paperfeed.rec.scoring import ScoringWeights

logger = get_logger("scripts.audit_faithfulness")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8 Explanation Faithfulness Audit.")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of explanations to sample (default: 100)")
    parser.add_argument("--max-authors", type=int, default=None, help="Limit number of test authors to process (default: all)")
    parser.add_argument("--enhanced", action="store_true", help="Use Track 1 enhanced parameters (M=250, M_channel=500, w_pop=0.20)")
    parser.add_argument("--facet-active", action="store_true", help="Audit the facet-active variant frozen by select_facet_config.py")
    parser.add_argument("--filter-popular", type=int, default=None, metavar="THRESHOLD", help="Filter popular papers from discovery")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument("--skip-tuning", action="store_true", help="Use the configured weights instead of the Track 1 tuned ones")
    parser.add_argument("--report-name", default=None, help="Markdown report filename")
    parser.add_argument("--results-name", default=None, help="JSON output filename")
    parser.add_argument("--out-dir", default="results", help="Output directory (default: results)")
    return parser.parse_args()


def audit_config(cfg, cohort, args, artifact_path):
    """The configuration of the Track 1 run being audited: config defaults, the --enhanced overrides, then the tuned
    weights, but only from a tuning cache whose signature matches this configuration (as run_track1_eval.py checks)."""
    rec_cfg = RecommendConfig.from_config(cfg)
    filter_popular = getattr(args, "filter_popular", None)
    if getattr(args, "enhanced", False):
        rec_cfg = rec_cfg.with_overrides(
            pool_size=250, channel_depth=500, weights=rec_cfg.weights.with_overrides(w_pop=0.20)
        )
    if getattr(args, "skip_tuning", False):
        if getattr(args, "facet_active", False):
            raise ValueError("--facet-active audits the frozen selection; it cannot be combined with --skip-tuning.")
        return rec_cfg, "configured weights (--skip-tuning)"
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"{artifact_path} not found: run scripts/run_track1_eval.py with the same options first, or pass --skip-tuning."
        )
    saved = json.loads(artifact_path.read_text(encoding="utf-8"))
    expected = tuning_signature(
        rec_cfg, [a["author_id"] for a in cohort if a.get("split") == "tuning"], filter_popular=filter_popular
    )
    if saved.get("config_signature") != expected:
        found_cache = False
        for cand_path in artifact_path.parent.glob("tuning_results*.json"):
            if cand_path == artifact_path:
                continue
            try:
                c_saved = json.loads(cand_path.read_text(encoding="utf-8"))
                if c_saved.get("config_signature") == expected:
                    saved = c_saved
                    found_cache = True
                    break
            except Exception:
                pass
        if not found_cache:
            raise ValueError(
                f"{artifact_path} was tuned for a different configuration ({saved.get('config_signature')} != {expected}); "
                "re-run scripts/run_track1_eval.py with the same options (e.g. --enhanced) or pass --skip-tuning."
            )
    tuned = ScoringWeights(**{k: v for k, v in saved["best_weights"].items() if k in ScoringWeights.__dataclass_fields__})
    if getattr(args, "enhanced", False):
        tuned = tuned.with_overrides(w_pop=0.20)  # as run_track1_eval.py applies --enhanced after tuning
    rec_cfg = rec_cfg.with_overrides(weights=tuned, lambda_div=saved.get("best_lambda_div", rec_cfg.lambda_div))
    if getattr(args, "facet_active", False):
        selection = default_facet_selection_path(artifact_path.parent, getattr(args, "enhanced", False))
        fa_weights, fa_lambda = load_facet_selection(selection, expected)
        return rec_cfg.with_overrides(weights=fa_weights, lambda_div=fa_lambda), f"facet-active variant from {selection.name} (signature matched)"
    return rec_cfg, f"tuned weights from {artifact_path.name} (signature matched)"


def main() -> int:
    args = parse_args()
    if args.facet_active and args.filter_popular is not None:
        raise ValueError("The facet-active variant does not support --filter-popular.")

    root = find_project_root()
    cfg = load_config()

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine default report and result names matching Track 1 conventions
    if args.report_name:
        report_name = args.report_name
    elif args.enhanced and args.facet_active:
        report_name = "faithfulness_audit_enhanced_facet_active_report.md"
    elif args.enhanced:
        report_name = "faithfulness_audit_enhanced_report.md"
    elif args.facet_active:
        report_name = "faithfulness_audit_facet_active_report.md"
    elif args.filter_popular is not None:
        report_name = "faithfulness_audit_report_filtered.md"
    else:
        report_name = "faithfulness_audit_report.md"

    if args.results_name:
        results_name = args.results_name
    elif args.enhanced and args.facet_active:
        results_name = "faithfulness_audit_enhanced_facet_active_report.json"
    elif args.enhanced:
        results_name = "faithfulness_audit_enhanced_report.json"
    elif args.facet_active:
        results_name = "faithfulness_audit_facet_active_report.json"
    elif args.filter_popular is not None:
        results_name = "faithfulness_audit_report_filtered.json"
    else:
        results_name = "faithfulness_audit_report.json"

    report_path = out_dir / report_name
    results_path = out_dir / results_name

    run_id = f"faithfulness_audit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    manifest = RunManifest(run_id, cfg, runs_dir=root / "results" / "runs")
    manifest.extra_metadata["arguments"] = vars(args)

    try:
        _run(args, root, cfg, manifest, report_path, results_path)
    except BaseException as exc:
        manifest.complete(status="FAILED", metadata={"error": repr(exc)})
        raise
    return 0


def _run(args, root, cfg, manifest, report_path, results_path):
    logger.info("Initializing RecommenderResources from %s...", root / cfg["paths"]["processed_dir"])
    resources = RecommenderResources.load(cfg, root, load_encoder=True, use_facets=True)

    cohort_path = root / cfg["paths"]["processed_dir"] / "cohort.jsonl"
    with open(cohort_path, "r", encoding="utf-8") as f:
        cohort = [json.loads(line) for line in f if line.strip()]
    test_authors = [a for a in cohort if a.get("split") == "test"]
    logger.info("Found %d test authors in cohort.", len(test_authors))

    rec_cfg, weights_source = audit_config(cfg, cohort, args, root / cfg["paths"]["artifacts_dir"] / "tuning_results.json")
    logger.info("Auditing with %s: %s, lambda_div=%.2f", weights_source, rec_cfg.weights, rec_cfg.lambda_div)

    t_start = time.time()
    audit_result = run_faithfulness_audit(
        resources=resources,
        test_authors=test_authors,
        config=rec_cfg,
        sample_size=args.sample_size,
        seed=args.seed,
        max_authors=args.max_authors,
    )
    elapsed = time.time() - t_start
    logger.info("Faithfulness audit completed in %.1fs.", elapsed)

    # Save JSON and Markdown artifacts
    audit_result.save(results_path)
    report_md = audit_result.to_markdown()
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    for f in (report_path, results_path):
        shutil.copy2(f, manifest.run_dir / f.name)
    manifest.complete(metadata={
        "weights_source": weights_source,
        "weights": vars(rec_cfg.weights),
        "lambda_div": rec_cfg.lambda_div,
        "pool_size": rec_cfg.pool_size,
        "n_span_explanations_audited": audit_result.n_span_explanations,
        "template_distribution": audit_result.template_distribution,
        "extractive_grounding_rate": audit_result.extractive_grounding_rate,
        "counterfactual_dropout_rate": audit_result.counterfactual_dropout_rate,
        "explanation_coverage": audit_result.explanation_coverage,
        "n_sampled": audit_result.n_sampled,
        "n_users": audit_result.n_users_in_sample,
        "elapsed_seconds": elapsed,
    })

    print(report_md)
    print(f"[+] Audit complete! Report written to {report_path}")
    print(f"[+] Raw JSON results written to {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
