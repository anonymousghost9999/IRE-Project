#!/usr/bin/env python3
"""Phase 8 explanation faithfulness audit (docs/plan.md §8 / docs/design.md §13.6).

Usage:
    python scripts/audit_faithfulness.py [--enhanced] [--sample-size 100] [--seed 42]

Audits deterministic recommendations against two criteria (prob.md §2.4, §6E.5):
1. Extractive Grounding Rate: Displayed facet spans and evidence sentences exist verbatim at their character offsets
   in the candidate paper's abstract text (target: 100%); citation edges are verified in SQLite under year <= T.
2. Counterfactual Necessity (Drop-out Rate): When the explained feature contribution is set to 0 (DeltaScore = 0),
   the paper leaves the Top-10 or falls >= 3 ranks under MMR re-diversification (verifying the explanation highlights
   a necessary ranking signal).
3. Explanation Coverage: Percentage of Top-10 recommendations successfully annotated with a grounded explanation (target >= 90%).
4. Template Distribution: Proportions of explanation templates across the evaluated recommendations.

Generates:
    results/faithfulness_audit_report.md
    results/faithfulness_audit_report.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paperfeed.common.config import find_project_root, load_config
from paperfeed.common.logger import get_logger
from paperfeed.common.manifest import RunManifest
from paperfeed.eval.faithfulness import run_faithfulness_audit
from paperfeed.rec.recommender import RecommendConfig, RecommenderResources
from paperfeed.rec.scoring import ScoringWeights

logger = get_logger("scripts.audit_faithfulness")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8 Explanation Faithfulness Audit.")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of explanations to sample (default: 100)")
    parser.add_argument("--max-authors", type=int, default=None, help="Limit number of test authors to process (default: all)")
    parser.add_argument("--enhanced", action="store_true", help="Use Track 1 enhanced parameters (M=250, M_channel=500, w_pop=0.20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument("--report-name", default="faithfulness_audit_report.md", help="Markdown report filename")
    parser.add_argument("--results-name", default="faithfulness_audit_report.json", help="JSON output filename")
    parser.add_argument("--out-dir", default="results", help="Output directory (default: results)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = find_project_root()
    cfg = load_config()

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / args.report_name
    results_path = out_dir / args.results_name

    run_id = f"faithfulness_audit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    manifest = RunManifest(run_id, cfg, runs_dir=root / "results" / "runs")
    manifest.extra_metadata["arguments"] = vars(args)

    logger.info("Initializing RecommenderResources from %s...", root / cfg["paths"]["processed_dir"])
    resources = RecommenderResources.load(cfg, root, load_encoder=False, use_facets=True)

    # Load cohort
    cohort_path = root / cfg["paths"]["processed_dir"] / "cohort.jsonl"
    with open(cohort_path, "r", encoding="utf-8") as f:
        cohort = [json.loads(line) for line in f if line.strip()]

    test_authors = [a for a in cohort if a.get("split") == "test"]
    logger.info("Found %d test authors in cohort.", len(test_authors))

    # Configuration setup
    rec_cfg = RecommendConfig.from_config(cfg)
    tuning_artifact = root / cfg["paths"]["artifacts_dir"] / "tuning_results.json"
    if tuning_artifact.exists():
        try:
            with open(tuning_artifact, "r", encoding="utf-8") as f:
                saved_tuning = json.load(f)
            bw = saved_tuning.get("best_weights", {})
            tuned_weights = ScoringWeights(**{k: v for k, v in bw.items() if k in ScoringWeights.__dataclass_fields__})
            rec_cfg = rec_cfg.with_overrides(
                weights=tuned_weights,
                lambda_div=saved_tuning.get("best_lambda_div", rec_cfg.lambda_div),
            )
            logger.info("Loaded tuned weights from %s", tuning_artifact)
        except Exception as exc:
            logger.warning("Could not load tuning artifact: %s", exc)

    if args.enhanced:
        rec_cfg = rec_cfg.with_overrides(
            pool_size=250,
            channel_depth=500,
            weights=rec_cfg.weights.with_overrides(w_pop=0.20),
        )
        logger.info("Applied enhanced parameters: M=250, M_channel=500, w_pop=0.20")

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

    manifest.complete(status="SUCCESS", metadata={
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

