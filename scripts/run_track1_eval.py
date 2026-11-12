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
    parser.add_argument("--w-pop", type=float, default=None, help="Popularity prior weight w_pop (default: 0.0 or from config)")
    parser.add_argument("--pool-size", type=int, default=None, help="Candidate pool size M (default: 100 or from config)")
    parser.add_argument("--channel-depth", type=int, default=None, help="Retrieval channel depth M_channel (default: 200 or from config)")
    parser.add_argument("--report-name", default="track1_lfo_report.md", help="Filename for generated markdown report")
    parser.add_argument("--results-name", default="track1_lfo_results.json", help="Filename for saved JSON results")
    parser.add_argument("--enhanced", action="store_true", help="Run with Option 2 & 3 enhancements (M=250, M_chan=500, w_pop=0.20, saved to enhanced files)")
    parser.add_argument("--filter-popular", type=int, default=None, metavar="THRESHOLD", help="Exclude papers with citation_count_T >= THRESHOLD from ground truth targets (popularity-filtered evaluation)")
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



def _load_citation_counts(db, paper_ids):
    """Load citation_count_T for a set of paper IDs (bare OpenAlex IDs) from the database."""
    from paperfeed.data.ground_truth import to_uri
    counts = {}
    uris = {to_uri(pid): pid for pid in paper_ids}
    uri_list = list(uris.keys())
    with db.get_connection() as conn:
        for i in range(0, len(uri_list), 500):
            chunk = uri_list[i : i + 500]
            marks = ",".join("?" for _ in chunk)
            for row in conn.execute(
                f"SELECT paper_id, citation_count_T FROM papers WHERE paper_id IN ({marks})",
                chunk,
            ):
                bare = uris.get(row[0])
                if bare is not None:
                    counts[bare] = row[1] or 0
    return counts


def filter_popular_from_discovery(discovery_map, db, threshold):
    """Remove papers with citation_count_T >= threshold from discovery ground truth.

    The system can still recommend these papers in the feed — they just receive
    no evaluation credit, so the metrics measure genuine research discovery only.

    Returns:
        filtered_map: dict mapping author_id -> filtered discovery list
        stats: dict with filtering statistics for the report
    """
    # Collect all unique target paper IDs
    all_target_ids = set()
    for targets in discovery_map.values():
        all_target_ids.update(targets)

    # Load citation counts from DB
    cite_counts = _load_citation_counts(db, all_target_ids)

    # Filter
    filtered_map = {}
    total_before = 0
    total_after = 0
    n_removed = 0
    removed_examples = []

    for aid, targets in discovery_map.items():
        total_before += len(targets)
        filtered = []
        for pid in targets:
            cc = cite_counts.get(pid, 0)
            if cc < threshold:
                filtered.append(pid)
            else:
                n_removed += 1
                if len(removed_examples) < 10:
                    removed_examples.append((pid, cc))
        filtered_map[aid] = filtered
        total_after += len(filtered)

    stats = {
        "threshold": threshold,
        "total_targets_before": total_before,
        "total_targets_after": total_after,
        "n_removed": n_removed,
        "pct_removed": round(100.0 * n_removed / max(total_before, 1), 2),
        "n_authors_with_targets_before": sum(1 for t in discovery_map.values() if t),
        "n_authors_with_targets_after": sum(1 for t in filtered_map.values() if t),
        "removed_examples": removed_examples,
    }

    logger.info(
        "Popularity filter (threshold=%d): %d/%d targets removed (%.1f%%). "
        "%d authors still have targets.",
        threshold, n_removed, total_before, stats["pct_removed"],
        stats["n_authors_with_targets_after"],
    )

    return filtered_map, stats


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    with open(root / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    processed_dir = root / cfg["paths"]["processed_dir"]
    logger.info("Loading recommender resources from %s...", processed_dir)
    resources = RecommenderResources.load(cfg, root, load_encoder=True, use_facets=True)

    cohort, discovery_map, recitation_map = load_cohort_and_ground_truth(processed_dir)
    topic_map, paper_facet_map = build_metadata_maps(resources.db, resources.facet_store)

    # Apply --enhanced defaults before resolving filtered filenames
    if args.enhanced:
        if args.w_pop is None:
            args.w_pop = 0.20
        if args.pool_size is None:
            args.pool_size = 250
        if args.channel_depth is None:
            args.channel_depth = 500
        if args.report_name == "track1_lfo_report.md":
            args.report_name = "track1_lfo_enhanced_report.md"
        if args.results_name == "track1_lfo_results.json":
            args.results_name = "track1_lfo_enhanced_results.json"

    # Popularity-filtered evaluation: exclude foundational/popular papers from ground truth
    pop_filter_stats = None
    if args.filter_popular is not None:
        discovery_map, pop_filter_stats = filter_popular_from_discovery(
            discovery_map, resources.db, threshold=args.filter_popular
        )
        # Default output filenames for filtered runs
        if args.report_name == "track1_lfo_report.md":
            args.report_name = "track1_lfo_filtered_report.md"
        elif args.report_name == "track1_lfo_enhanced_report.md":
            args.report_name = "track1_lfo_enhanced_filtered_report.md"
        if args.results_name == "track1_lfo_results.json":
            args.results_name = "track1_lfo_filtered_results.json"
        elif args.results_name == "track1_lfo_enhanced_results.json":
            args.results_name = "track1_lfo_enhanced_filtered_results.json"

    tuning_authors = [a for a in cohort if a.get("split") == "tuning"]
    test_authors = [a for a in cohort if a.get("split") == "test"]

    if args.tuning_sample:
        tuning_authors = tuning_authors[: args.tuning_sample]
    if args.test_sample:
        test_authors = test_authors[: args.test_sample]

    logger.info("Cohort partition: %d tuning authors, %d test authors.", len(tuning_authors), len(test_authors))

    tuned_config = RecommendConfig.from_config(cfg)
    overrides = {}
    if args.pool_size is not None:
        overrides["pool_size"] = args.pool_size
    if args.channel_depth is not None:
        overrides["channel_depth"] = args.channel_depth
    if overrides:
        tuned_config = tuned_config.with_overrides(**overrides)
    if args.w_pop is not None:
        tuned_config = tuned_config.with_overrides(
            weights=tuned_config.weights.with_overrides(w_pop=args.w_pop)
        )

    # 1. Tuning stage
    tuning_sig = {
        "pool_size": tuned_config.pool_size,
        "channel_depth": tuned_config.channel_depth,
        "w_pop": tuned_config.weights.w_pop,
        "adaptive_graph": tuned_config.adaptive_graph,
        "filter_popular": args.filter_popular,
    }
    tuning_artifact = root / "data" / "artifacts" / "tuning_results.json"
    if not args.skip_tuning and tuning_authors:
        loaded_saved = None
        if tuning_artifact.exists():
            with open(tuning_artifact, "r", encoding="utf-8") as f:
                candidate_saved = json.load(f)
            if candidate_saved.get("config_signature") == tuning_sig:
                loaded_saved = candidate_saved
            else:
                logger.info(
                    "Tuning cache %s signature mismatch (%s != %s); re-tuning for current config.",
                    tuning_artifact,
                    candidate_saved.get("config_signature"),
                    tuning_sig,
                )

        if loaded_saved is not None:
            logger.info("Loading matching tuned parameters from %s", tuning_artifact)
            bw = loaded_saved.get("best_weights", {})
            tuned_weights = ScoringWeights(**{k: v for k, v in bw.items() if k in ScoringWeights.__dataclass_fields__})
            if args.w_pop is not None:
                tuned_weights = tuned_weights.with_overrides(w_pop=args.w_pop)
            tuned_config = tuned_config.with_overrides(
                weights=tuned_weights, lambda_div=loaded_saved.get("best_lambda_div", 0.2)
            )
        else:
            logger.info("Running hyperparameter tuning on %d tuning authors...", len(tuning_authors))
            t0 = time.time()
            cached = precompute_tuning_cache(resources, tuning_authors, discovery_map, tuned_config)
            t_res = evaluate_parameter_grid(
                cached,
                resources,
                adaptive_graph=tuned_config.adaptive_graph,
                config_signature=tuning_sig,
            )
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
            tuned_weights = t_res.best_weights
            if args.w_pop is not None:
                tuned_weights = tuned_weights.with_overrides(w_pop=args.w_pop)
            tuned_config = tuned_config.with_overrides(
                weights=tuned_weights, lambda_div=t_res.best_lambda_div
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
        paper_id_index=resources.store.paper_id_index,
        embedding_matrix=resources.store.vectors,
        paper_facet_map=paper_facet_map,
        config=tuned_config,
        k=10,
        include_anchors=True,
        filter_empty_targets=(args.filter_popular is not None),
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
    run_notes = (
        f"> [!NOTE]\n"
        f"> **Evaluation Run Configuration:**\n"
        f"> - Mode: {'Enhanced (Option 2 & Option 3)' if (tuned_config.weights.w_pop > 0 or (tuned_config.pool_size and tuned_config.pool_size > 100)) else 'Baseline'}\n"
        f"> - Candidate Pool Size (M): {tuned_config.pool_size or 100}\n"
        f"> - Channel Depth (M_channel): {tuned_config.channel_depth or 200}\n"
        f"> - Popularity Prior Weight (w_pop): {tuned_config.weights.w_pop:.2f}\n"
        f"> - Scoring Weights: w_pers={tuned_config.weights.w_pers:.2f}, w_cite={tuned_config.weights.w_cite:.2f}, "
        f"w_facet={tuned_config.weights.w_facet:.2f}, w_nov={tuned_config.weights.w_nov:.2f}, "
        f"lambda_div={tuned_config.lambda_div:.2f}\n"
        f"> - Report: `{args.report_name}` | Raw JSON: `{args.results_name}`"
    )
    if pop_filter_stats is not None:
        n_eval = lfo_results.n_authors
        run_notes += (
            f"\n> \n"
            f"> **Popularity Filter Active:**\n"
            f"> - Threshold: citation_count_T >= {pop_filter_stats['threshold']}\n"
            f"> - Targets removed: {pop_filter_stats['n_removed']}/{pop_filter_stats['total_targets_before']} "
            f"({pop_filter_stats['pct_removed']}%)\n"
            f"> - Authors with remaining targets: {pop_filter_stats['n_authors_with_targets_after']}\n"
            f"> - Evaluated cohort: Macro averages and hypothesis testing over {n_eval} active authors with targets"
        )

    report_text = generate_full_evaluation_report(
        system_aggregates=lfo_results.system_aggregates,
        hypothesis_report=hypo_report,
        slice_results=slice_breakdown,
        run_notes=run_notes,
    )

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / args.report_name
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    json_file = out_dir / args.results_name
    lfo_results.save(json_file)

    logger.info("[+] Evaluation complete! Report written to %s", report_file)
    print("\n" + "=" * 90)
    print(report_text)
    print("=" * 90)


if __name__ == "__main__":
    main()
