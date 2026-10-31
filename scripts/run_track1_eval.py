"""Main entry-point script to run Track 1 Leave-Future-Out (LFO) Evaluation (Phase 6 / plan.md §6).

Executes:
1. Loads 400-researcher cohort, ground truth 2025 citations, and candidate indexes.
2. (Optional/Cached) Grid-search tuning on the 80 tuning authors.
3. Evaluates full Paper Feed system, ablations, and 5 baselines on the 320 test authors.
4. Paired bootstrap hypothesis tests with Benjamini-Hochberg FDR correction.
5. Slice analysis (Warm, Moderate, Sparse, Zero-Pub).
6. Generates full markdown and LaTeX tables and writes results to ``results/track1_lfo_report.md``.

Usage:
    python scripts/run_track1_eval.py [--sample 10] [--skip-tuning]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import clean_id
from paperfeed.eval.lfo import run_lfo_evaluation
from paperfeed.eval.report import generate_full_evaluation_report
from paperfeed.eval.slices import slice_by_history_volume
from paperfeed.eval.stats import run_track1_hypothesis_tests
from paperfeed.eval.tuning import evaluate_parameter_grid, precompute_tuning_cache
from paperfeed.rec.recommender import RecommendConfig, RecommenderResources
from paperfeed.rec.scoring import ScoringWeights

logger = get_logger("scripts.run_track1_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Track 1 Leave-Future-Out Evaluation")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--test-sample", type=int, default=None, help="Sample N test authors (default: all)")
    parser.add_argument("--tuning-sample", type=int, default=None, help="Sample N tuning authors (default: all)")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip tuning and use default scoring weights")
    parser.add_argument("--n-bootstrap", type=int, default=10000, help="Number of bootstrap resamples (default: 10000)")
    parser.add_argument("--out-dir", default="results", help="Directory to save report and artifacts")
    return parser.parse_args()


def load_cohort_and_ground_truth(processed_dir: Path):
    cohort_path = processed_dir / "cohort.jsonl"
    gt_path = processed_dir / "ground_truth.jsonl"

    cohort = []
    with open(cohort_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cohort.append(json.loads(line))

    discovery_map = {}
    recitation_map = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                aid = clean_id(row["author_id"])
                # Use in-corpus discovery targets as headline relevance
                disc = row.get("discovery_in_corpus", []) or row.get("discovery", [])
                recit = row.get("recitation_in_corpus", []) or row.get("recitation", [])
                discovery_map[aid] = [clean_id(p) for p in disc]
                recitation_map[aid] = [clean_id(p) for p in recit]

    return cohort, discovery_map, recitation_map


def build_metadata_maps(db, facet_store):
    """Load topic map and facet sets for metrics computation."""
    topic_map = {}
    try:
        with db.get_connection() as conn:
            rows = conn.execute("SELECT paper_id, primary_topic_id FROM papers").fetchall()
            for r in rows:
                topic_map[clean_id(r[0])] = r[1] or "UNKNOWN"
    except Exception as exc:
        logger.warning("Failed to load topic map: %s", exc)

    paper_facet_map = {}
    try:
        with db.get_connection() as conn:
            rows = conn.execute("SELECT paper_id, canonical_id FROM paper_facets").fetchall()
            for r in rows:
                pid = clean_id(r[0])
                cid = r[1]
                if pid not in paper_facet_map:
                    paper_facet_map[pid] = set()
                if cid:
                    paper_facet_map[pid].add(cid)
    except Exception as exc:
        logger.warning("Failed to load paper facet map: %s", exc)

    return topic_map, paper_facet_map


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    with open(root / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    processed_dir = root / cfg["paths"]["processed_dir"]
    logger.info("Loading recommender resources from %s...", processed_dir)
    resources = RecommenderResources.load(cfg, root, load_encoder=False, use_facets=True)

    cohort, discovery_map, recitation_map = load_cohort_and_ground_truth(processed_dir)
    topic_map, paper_facet_map = build_metadata_maps(resources.db, resources.facet_store)

    tuning_authors = [a for a in cohort if a.get("split") == "tuning"]
    test_authors = [a for a in cohort if a.get("split") == "test"]

    if args.tuning_sample:
        tuning_authors = tuning_authors[: args.tuning_sample]
    if args.test_sample:
        test_authors = test_authors[: args.test_sample]

    logger.info("Cohort partition: %d tuning authors, %d test authors.", len(tuning_authors), len(test_authors))

    tuned_config = RecommendConfig.from_config(cfg)

    # 1. Tuning stage
    tuning_artifact = root / "data" / "artifacts" / "tuning_results.json"
    if not args.skip_tuning and tuning_authors:
        if tuning_artifact.exists():
            logger.info("Loading existing tuned parameters from %s", tuning_artifact)
            with open(tuning_artifact, "r", encoding="utf-8") as f:
                saved = json.load(f)
            bw = saved.get("best_weights", {})
            tuned_weights = ScoringWeights(**{k: v for k, v in bw.items() if k in ScoringWeights.__dataclass_fields__})
            tuned_config = tuned_config.with_overrides(
                weights=tuned_weights, lambda_div=saved.get("best_lambda_div", 0.2)
            )
        else:
            logger.info("Running hyperparameter tuning on %d tuning authors...", len(tuning_authors))
            t0 = time.time()
            cached = precompute_tuning_cache(resources, tuning_authors, discovery_map, tuned_config)
            t_res = evaluate_parameter_grid(cached, resources)
            t_res.save(tuning_artifact)
            logger.info(
                "Tuning complete in %.1fs. Best Mean NDCG@10: %.4f (w_cite=%.2f, w_facet=%.2f, w_nov=%.2f, lambda=%.2f)",
                time.time() - t0,
                t_res.best_mean_ndcg,
                t_res.best_weights.w_cite,
                t_res.best_weights.w_facet,
                t_res.best_weights.w_nov,
                t_res.best_lambda_div,
            )
            tuned_config = tuned_config.with_overrides(
                weights=t_res.best_weights, lambda_div=t_res.best_lambda_div
            )

    # 2. Main LFO evaluation run on test split
    logger.info("Running Track 1 LFO evaluation across %d test authors...", len(test_authors))
    t0 = time.time()
    lfo_results = run_lfo_evaluation(
        resources=resources,
        cohort=test_authors,
        ground_truth_map=discovery_map,
        recitation_map=recitation_map,
        topic_map=topic_map,
        paper_id_index=resources.store._row,
        embedding_matrix=resources.store.vectors,
        paper_facet_map=paper_facet_map,
        config=tuned_config,
        k=10,
        include_anchors=True,
    )
    logger.info("Evaluation loop finished in %.1fs.", time.time() - t0)

    # 3. Statistical hypothesis testing
    # Convert per_user_results to per-system series
    system_series = {}
    for r in lfo_results.per_user_results:
        for sys_name, m_dict in r.system_metrics.items():
            if sys_name not in system_series:
                system_series[sys_name] = {}
            for m_name, val in m_dict.items():
                if m_name not in system_series[sys_name]:
                    system_series[sys_name][m_name] = []
                system_series[sys_name][m_name].append(val)

    hypo_report = run_track1_hypothesis_tests(system_series, n_resamples=args.n_bootstrap)

    # 4. Slices
    slice_breakdown = slice_by_history_volume(lfo_results.per_user_results)

    # 5. Generate and save report
    report_text = generate_full_evaluation_report(
        system_aggregates=lfo_results.system_aggregates,
        hypothesis_report=hypo_report,
        slice_results=slice_breakdown,
    )

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "track1_lfo_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    json_file = out_dir / "track1_lfo_results.json"
    lfo_results.save(json_file)

    logger.info("[+] Evaluation complete! Report written to %s", report_file)
    print("\n" + "=" * 90)
    print(report_text)
    print("=" * 90)


if __name__ == "__main__":
    main()
