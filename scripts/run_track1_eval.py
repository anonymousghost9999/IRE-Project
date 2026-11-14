"""Main entry-point script to run Track 1 Leave-Future-Out (LFO) Evaluation (Phase 6 / plan.md §6).

Executes:
1. Loads 400-researcher cohort, ground truth 2025 citations, and candidate indexes.
2. (Optional/Cached) Grid-search tuning on the 80 tuning authors.
3. Evaluates full Paper Feed system, ablations, and 5 baselines on the 320 test authors.
4. Paired bootstrap hypothesis tests with Benjamini-Hochberg FDR correction.
5. Slices: Warm / Moderate / Sparse, the Zero-Publication Seed-Only re-run, and Item Cold-Start (fresh vs
   established targets), each with its n.
6. Writes the report and raw results to ``results/`` and a run manifest (git commit, config, arguments, tuned values,
   input-file hashes, timings) with copies of both outputs to ``results/runs/<run_id>/``.

Usage:
    python scripts/run_track1_eval.py [--test-sample 10] [--skip-tuning] [--skip-zero-pub]
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import yaml

# Ensure src/ is on sys.path (like the other scripts), so the script runs without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paperfeed.common.logger import get_logger
from paperfeed.common.manifest import RunManifest
from paperfeed.data.cohort import clean_id, load_topic_names
from paperfeed.data.ground_truth import to_uri
from paperfeed.eval.lfo import ZERO_PUB_BASELINES, run_lfo_evaluation, run_zero_pub_evaluation
from paperfeed.eval.report import format_cold_start_table, format_zero_pub_section, generate_full_evaluation_report
from paperfeed.eval.slices import classify_cold_start_targets, item_cold_start_breakdown, slice_by_history_volume
from paperfeed.eval.stats import H1_METRICS, compare_system_to_baselines, run_track1_hypothesis_tests
from paperfeed.eval.tuning import (
    load_v2_decision,
    default_facet_selection_path,
    evaluate_parameter_grid,
    find_tuning_artifact,
    load_facet_selection,
    precompute_tuning_cache,
    tune_hybrid_persona,
    tuning_artifact_path,
    tuning_signature,
)
from paperfeed.rec.recommender import RecommendConfig, RecommenderResources
from paperfeed.eval.learned_cv import load_v3_decision
from paperfeed.rec.learned import load_ranker
from paperfeed.rec.rerank import CrossEncoderScorer
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
    parser.add_argument(
        "--graph-channel",
        action="store_true",
        help="Run the V4 citation-graph channel variant (design §13.8, plan 10.4d); needs the frozen decision from "
        "scripts/decide_graph_channel.py",
    )
    parser.add_argument(
        "--cross-encoder",
        action="store_true",
        help="Run the V2 cross-encoder variant (design §13.8, plan 10.2d) with the reranker and weights frozen by "
        "scripts/decide_cross_encoder.py",
    )
    parser.add_argument(
        "--sota-comparison",
        action="store_true",
        help="Phase 11e: run the frozen best configuration (configs/best.yaml) together with the Phase 11 comparison "
        "systems (RP3beta, Scholar Inbox-style classifier, SciNCL centroid kNN, RankGPT reranker) in one test run",
    )
    parser.add_argument(
        "--encoder",
        default=None,
        metavar="KEY",
        help="Run the V1 encoder candidate (design §13.8, plan 10.1d) frozen in data/artifacts/variant_v1_decision.json, "
        "with its full-corpus index from scripts/build_encoder_index.py; its grid is tuned on the tuning split first",
    )
    parser.add_argument(
        "--learned-ranker",
        action="store_true",
        help="Run the V3 learned ranker (design §13.8, plan 10.3d) with the model frozen by "
        "scripts/decide_learned_ranker.py",
    )
    parser.add_argument("--facet-active", action="store_true", help="Run the facet-active variant frozen by scripts/select_facet_config.py (design §9) instead of the grid optimum")
    parser.add_argument("--skip-zero-pub", action="store_true", help="Skip the Zero-Publication Seed-Only slice (a full extra pass over the test cohort)")
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
    zero_pub_map = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                aid = clean_id(row["author_id"])
                # In-corpus targets only: an empty list stays empty instead of switching that author to the
                # full-reference definition (which includes papers the system can never recommend).
                discovery_map[aid] = [clean_id(p) for p in row["discovery_in_corpus"]]
                recitation_map[aid] = [clean_id(p) for p in row["recitation_in_corpus"]]
                zero_pub_map[aid] = [clean_id(p) for p in row["zero_pub_in_corpus"]]

    return cohort, discovery_map, recitation_map, zero_pub_map


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


def _series(rows):
    """Per-user metric rows -> {system: {metric: [value per user]}} for the paired tests."""
    series = {}
    for r in rows:
        for sys_name, m_dict in r.system_metrics.items():
            for m_name, val in m_dict.items():
                series.setdefault(sys_name, {}).setdefault(m_name, []).append(val)
    return series


def _sha256(path: Path):
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed_topic_subfields(db, raw_dir: Path, authors):
    """Each author's seed-topic subfield: the Popularity domain in the zero-publication slice (design §13.1).

    Seed topics are OpenAlex Topic display names; they map back to Topic IDs through the cached raw responses and to a
    subfield through the papers table (the modal subfield of papers with that primary Topic). Among an author's seed
    topics the most common subfield wins, ties going to the earlier (more frequent) seed topic.
    """
    ids_by_name = {}
    for tid, name in load_topic_names(raw_dir).items():
        ids_by_name.setdefault(name, []).append(tid)
    subfield_of_topic = {}
    with db.get_connection() as conn:
        rows = conn.execute(
            "SELECT primary_topic_id, subfield_id, COUNT(*) AS n FROM papers "
            "WHERE primary_topic_id IS NOT NULL AND subfield_id IS NOT NULL GROUP BY 1, 2 ORDER BY 1, n DESC, 2"
        ).fetchall()
    for topic_id, subfield_id, _n in rows:
        subfield_of_topic.setdefault(topic_id, subfield_id)

    out = {}
    for a in authors:
        subfields = [
            subfield_of_topic[tid]
            for name in a.get("seed_topics") or []
            for tid in ids_by_name.get(name, [])
            if tid in subfield_of_topic
        ]
        counts = Counter(subfields)
        out[clean_id(a["author_id"])] = max(subfields, key=lambda sf: (counts[sf], -subfields.index(sf))) if subfields else None
    return out


def cold_start_classes(db, target_ids, cutoff_year):
    """fresh / established class of every target paper (design §12), from citation_count_T and year."""
    counts, years = {}, {}
    uris = [to_uri(p) for p in sorted(target_ids)]
    with db.get_connection() as conn:
        for i in range(0, len(uris), 500):
            chunk = uris[i : i + 500]
            marks = ",".join("?" for _ in chunk)
            for pid, count, year in conn.execute(
                f"SELECT paper_id, COALESCE(citation_count_T, 0), year FROM papers WHERE paper_id IN ({marks})", chunk
            ):
                counts[clean_id(pid)] = int(count)
                years[clean_id(pid)] = int(year) if year is not None else None
    return classify_cold_start_targets(counts, {p: y for p, y in years.items() if y is not None}, cutoff_year)


def build_sota_systems(resources, root, cfg):
    """The four Phase 11 comparison systems with their frozen tuning (plan 11e)."""
    import numpy as np

    from paperfeed.eval.encoder_study import paper_texts
    from paperfeed.eval.rankgpt import LocalRanker, RerankStats
    from paperfeed.eval.sota import (
        CitationGraph,
        fit_pca_space,
        make_centroid_system,
        make_rankgpt_system,
        make_rp3beta_system,
        make_scholar_inbox_system,
    )
    from paperfeed.index.encoders import make_encoder
    from paperfeed.rec.recommender import load_encoder_index

    artifacts = root / cfg["paths"]["artifacts_dir"]
    processed = root / cfg["paths"]["processed_dir"]
    rp3 = json.loads((artifacts / "sota_rp3beta_tuning.json").read_text(encoding="utf-8"))["best"]
    si = json.loads((artifacts / "sota_scholar_inbox_tuning.json").read_text(encoding="utf-8"))["best"]
    texts = lambda ids: paper_texts(resources.db, ids)  # noqa: E731

    logger.info("Phase 11e: loading the citation graph for RP3beta...")
    graph = CitationGraph.from_db(resources.db, resources.cutoff_year)
    logger.info("Phase 11e: GTE-large vectors and PCA for the Scholar Inbox-style classifier...")
    gte_store, _ = load_encoder_index(processed, "gte")
    gte_ids = list(gte_store.paper_id_index)
    space = fit_pca_space(gte_ids, gte_store.select(gte_ids))
    allowed_idx = np.array(sorted(space.index[p] for p in resources.eligible if p in space.index), dtype=np.int64)
    logger.info("Phase 11e: SciNCL vectors for the centroid kNN...")
    scincl_store, scincl_dense = load_encoder_index(processed, "scincl")
    logger.info("Phase 11e: loading the RankGPT reranker...")
    ranker = LocalRanker()
    stats = RerankStats()
    systems = {
        "RP3beta": make_rp3beta_system(graph, resources.eligible, rp3["alpha"], rp3["beta"]),
        "Scholar Inbox (GTE-large + LR)": make_scholar_inbox_system(space, allowed_idx, si["C"]),
        "SciNCL centroid kNN": make_centroid_system(scincl_store, scincl_dense, resources.eligible, texts,
                                                    make_encoder("scincl", track1=True)),
        "RankGPT (Qwen2.5-3B)": make_rankgpt_system(ranker, texts, stats),
    }
    info = {
        "RP3beta": {"alpha": rp3["alpha"], "beta": rp3["beta"], "tuning_ndcg": rp3["mean_ndcg"],
                    "graph_papers": len(graph.ids), "graph_edges": int(graph.A.nnz)},
        "Scholar Inbox (GTE-large + LR)": {"C": si["C"], "tuning_ndcg": si["mean_ndcg"], "pca_explained_variance":
                                           space.explained_variance, "model": "thenlper/gte-large"},
        "SciNCL centroid kNN": {"model": "malteos/scincl", "n_papers": len(scincl_store)},
        "RankGPT (Qwen2.5-3B)": {"model": ranker.model_name, "precision": ranker.precision, "window": 20, "step": 10},
    }
    return systems, info, stats


def phase11_sections(lfo_results, best_cfg, sota_info, rankgpt_stats, n_bootstrap):
    """The plan 11.5 tables and the best-vs-comparison tests (BH over this family)."""
    import numpy as np

    from paperfeed.eval.report import format_best_config_table, format_comparisons_markdown, format_sota_table

    rows = lfo_results.per_user_results
    latency = {}
    for name in {n for r in rows for n in r.latency_seconds}:
        vals = [r.latency_seconds[name] for r in rows if name in r.latency_seconds]
        latency[name] = float(np.mean(vals))
    cov = [r.explanation_coverage for r in rows if r.explanation_coverage is not None]
    coverage = float(np.mean(cov)) if cov else None
    sota_names = [n for n in lfo_results.system_aggregates if n.startswith("SOTA: ")]
    comparisons = compare_system_to_baselines(_series(rows), "Paper Feed (Full)", sota_names, metrics=H1_METRICS,
                                              n_resamples=n_bootstrap)
    params = "\n".join(f"- **{name}:** {json.dumps(v)}" for name, v in sota_info.items())
    rg = rankgpt_stats
    body = "\n".join([
        "Plan 11.3-11.5 / design §13.9. The best configuration (`configs/best.yaml`) is Core, so this run compares Core "
        "with the comparison systems fixed by the literature review of 2026-11-14 (design §13.9 Phase 11 specification). "
        "Every system uses the same eligible papers, exclusion set E_u and test authors; each was tuned on the tuning "
        "split with a grid fixed in advance. The test split was already used once for the Phase 10 variant table, so "
        "this run is confirmatory. Published numbers from other datasets are context only.\n",
        format_sota_table(lfo_results.system_aggregates, latency, coverage),
        "",
        "Paper Feed against each comparison system (positive = Paper Feed better; BH over this family of "
        f"{len(comparisons)} comparisons):\n",
        format_comparisons_markdown(comparisons),
        "",
        "Comparison systems and their frozen settings:\n",
        params,
        "",
        f"RankGPT: {rg.windows} windows, {rg.unparsable} unparsable (kept their input order)." if rg else "",
        "",
        "Excluded (design §13.9 Phase 11 spec point 5): LightGCN-style graph neural recommenders (compute and a "
        "collaborative user base; RP3beta represents graph recommenders); hosted feeds such as Semantic Scholar, Google "
        "Scholar and Scholar Inbox itself (cannot be restricted to <= T); SPECTER2 user-centroid kNN is the Dense "
        "Centroid baseline. The zero-publication slice below is Core only: the comparison systems need authored papers.",
    ])
    return [("Phase 11: Comparison with SOTA Paper Recommenders", body),
            ("Phase 11: Best Configuration", format_best_config_table(best_cfg))]


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    with open(root / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # The report name is suffixed later (in run()), so tag the variant here to keep the run directories apart
    variant = "".join([
        "_enhanced" if args.enhanced else "",
        "_filtered" if args.filter_popular is not None else "",
        "_facet_active" if args.facet_active else "",
        "_graph_channel" if args.graph_channel else "",
        "_cross_encoder" if args.cross_encoder else "",
        "_learned_ranker" if args.learned_ranker else "",
        f"_encoder_{args.encoder}" if args.encoder else "",
        "_sota_comparison" if args.sota_comparison else "",
    ])
    run_id = f"track1_{Path(args.report_name).stem}{variant}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    manifest = RunManifest(run_id, cfg, runs_dir=root / "results" / "runs")
    manifest.extra_metadata["arguments"] = vars(args)
    try:
        run(args, root, cfg, manifest)
    except BaseException as exc:
        manifest.complete(status="FAILED", metadata={"error": repr(exc)})
        raise


def run(args, root, cfg, manifest):
    t_start = time.time()
    processed_dir = root / cfg["paths"]["processed_dir"]
    logger.info("Loading recommender resources from %s...", processed_dir)
    best_cfg = None
    if args.sota_comparison:
        if (args.enhanced or args.filter_popular is not None or args.facet_active or args.graph_channel
                or args.cross_encoder or args.learned_ranker or args.encoder or args.skip_tuning):
            raise ValueError("--sota-comparison runs the frozen best configuration; drop the variant options and "
                             "--skip-tuning.")
        best_path = root / "configs" / "best.yaml"
        if not best_path.exists():
            raise FileNotFoundError(f"{best_path} not found: freeze it with scripts/freeze_best_config.py (plan 11a).")
        best_cfg = yaml.safe_load(best_path.read_text(encoding="utf-8"))
        if best_cfg.get("best_configuration") != "core":
            raise ValueError("configs/best.yaml is not Core; run the assembled configuration with its own options.")
        for name in ("sota_rp3beta_tuning.json", "sota_scholar_inbox_tuning.json"):
            if not (root / cfg["paths"]["artifacts_dir"] / name).exists():
                raise FileNotFoundError(f"{name} not found: freeze the comparison systems' tuning first (plan 11b-11c).")
        for attr in ("report_name", "results_name"):
            name = Path(getattr(args, attr))
            if name.stem.startswith("track1_lfo") and not name.stem.endswith("_sota_comparison"):
                setattr(args, attr, f"{name.stem}_sota_comparison{name.suffix}")

    v1_decision = None
    if args.encoder:
        # V1 was decided for the headline configuration only, and only its frozen candidate gets a test run
        if (args.enhanced or args.filter_popular is not None or args.facet_active or args.graph_channel
                or args.cross_encoder or args.learned_ranker or args.skip_tuning):
            raise ValueError("--encoder runs the V1 candidate in the headline configuration with its own tuning; drop "
                             "--enhanced, --filter-popular, --facet-active, --graph-channel, --cross-encoder, "
                             "--learned-ranker and --skip-tuning.")
        decision_path = root / cfg["paths"]["artifacts_dir"] / "variant_v1_decision.json"
        if not decision_path.exists():
            raise FileNotFoundError(f"{decision_path} not found: freeze the V1 decision with scripts/decide_encoder.py "
                                    "before the test run (plan 10.1c).")
        v1_decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if args.encoder != v1_decision["chosen_encoder"]:
            raise ValueError(f"The frozen V1 candidate is {v1_decision['chosen_encoder']!r}, not {args.encoder!r}.")
        for attr in ("report_name", "results_name"):
            name = Path(getattr(args, attr))
            if name.stem.startswith("track1_lfo") and not name.stem.endswith(f"_encoder_{args.encoder}"):
                setattr(args, attr, f"{name.stem}_encoder_{args.encoder}{name.suffix}")
    resources = RecommenderResources.load(cfg, root, load_encoder=True, use_facets=True,
                                          encoder_key=args.encoder or "specter2")

    cohort, discovery_map, recitation_map, zero_pub_map = load_cohort_and_ground_truth(processed_dir)
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

    # V4 was tuned and decided for the headline configuration only (plan 10.4b-c)
    v4_decision = None
    if args.graph_channel:
        if args.enhanced or args.filter_popular is not None or args.facet_active:
            raise ValueError("--graph-channel was tuned for the headline configuration only; drop --enhanced, "
                             "--filter-popular and --facet-active.")
        decision_path = root / cfg["paths"]["artifacts_dir"] / "variant_v4_decision.json"
        if not decision_path.exists():
            raise FileNotFoundError(f"{decision_path} not found: freeze the V4 decision with "
                                    "scripts/decide_graph_channel.py before the test run (plan 10.4c).")
        v4_decision = json.loads(decision_path.read_text(encoding="utf-8"))
        for attr in ("report_name", "results_name"):
            name = Path(getattr(args, attr))
            if name.stem.startswith("track1_lfo") and not name.stem.endswith("_graph_channel"):
                setattr(args, attr, f"{name.stem}_graph_channel{name.suffix}")

    # V2 was tuned and decided for the headline configuration only (plan 10.2b-c)
    v2_decision = None
    if args.cross_encoder:
        if args.enhanced or args.filter_popular is not None or args.facet_active or args.graph_channel:
            raise ValueError("--cross-encoder was tuned for the headline configuration only; drop --enhanced, "
                             "--filter-popular, --facet-active and --graph-channel.")
        v2_decision = load_v2_decision(root / cfg["paths"]["artifacts_dir"])
        resources = replace(resources, reranker=CrossEncoderScorer(v2_decision["chosen_model"]))
        for attr in ("report_name", "results_name"):
            name = Path(getattr(args, attr))
            if name.stem.startswith("track1_lfo") and not name.stem.endswith("_cross_encoder"):
                setattr(args, attr, f"{name.stem}_cross_encoder{name.suffix}")

    # V3 was cross-validated for the headline configuration only (plan 10.3b-c)
    v3_decision = None
    if args.learned_ranker:
        if args.enhanced or args.filter_popular is not None or args.facet_active or args.graph_channel or args.cross_encoder:
            raise ValueError("--learned-ranker was decided for the headline configuration only; drop --enhanced, "
                             "--filter-popular, --facet-active, --graph-channel and --cross-encoder.")
        v3_decision = load_v3_decision(root / cfg["paths"]["artifacts_dir"])
        resources = replace(resources, ranker=load_ranker(v3_decision["model_path"], v3_decision["channel_depth"]))
        for attr in ("report_name", "results_name"):
            name = Path(getattr(args, attr))
            if name.stem.startswith("track1_lfo") and not name.stem.endswith("_learned_ranker"):
                setattr(args, attr, f"{name.stem}_learned_ranker{name.suffix}")

    # The facet-active variant writes its own files next to the run it varies
    if args.facet_active:
        for attr in ("report_name", "results_name"):
            name = Path(getattr(args, attr))
            if name.stem.startswith("track1_lfo") and not name.stem.endswith("_facet_active"):
                setattr(args, attr, f"{name.stem}_facet_active{name.suffix}")

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
    if args.graph_channel:
        tuned_config = tuned_config.with_overrides(graph_channel=True)  # part of the tuning signature
    if args.cross_encoder:
        tuned_config = tuned_config.with_overrides(cross_encoder=v2_decision["chosen_model"])  # in the signature
    if args.encoder:
        tuned_config = tuned_config.with_overrides(dense_encoder=args.encoder)  # its own tuning file

    # 1. Tuning stage
    tuning_sig = tuning_signature(
        tuned_config, [a["author_id"] for a in tuning_authors], filter_popular=args.filter_popular
    )
    # Hybrid-Persona baseline: configured values unless tuned below (plan §6.3)
    hybrid_params = {
        "gamma_rec": tuned_config.gamma_rec,
        "n_terms": tuned_config.n_terms,
        "n_titles": tuned_config.n_titles,
    }
    artifacts_dir = root / cfg["paths"]["artifacts_dir"]
    # One tuning file per configuration (signature); older shared files are reused only on an exact signature match
    tuning_artifact = None if args.skip_tuning else find_tuning_artifact(artifacts_dir, tuning_sig)
    if args.cross_encoder and (tuning_artifact is None or tuning_artifact.name != v2_decision.get("v2_tuning_file")):
        raise RuntimeError(
            f"The V2 test run must use the tuning file frozen with the decision ({v2_decision.get('v2_tuning_file')}), "
            f"found {tuning_artifact.name if tuning_artifact else 'none'} (--skip-tuning is not allowed either)."
        )
    if args.graph_channel and (tuning_artifact is None or tuning_artifact.name != v4_decision.get("v4_tuning_file")):
        # The V4 weights were frozen with the decision; re-tuning here would let the test run pick its own
        raise RuntimeError(
            f"The V4 test run must use the tuning file frozen with the decision ({v4_decision.get('v4_tuning_file')}), "
            f"found {tuning_artifact.name if tuning_artifact else 'none'} (--skip-tuning is not allowed either)."
        )
    if not args.skip_tuning and tuning_authors:
        loaded_saved = None
        if tuning_artifact is not None:
            with open(tuning_artifact, "r", encoding="utf-8") as f:
                candidate_saved = json.load(f)
            if candidate_saved.get("hybrid_persona"):
                loaded_saved = candidate_saved
            else:
                logger.info("Tuning cache %s has no Hybrid-Persona tuning; re-tuning.", tuning_artifact)

        if loaded_saved is not None:
            logger.info("Loading matching tuned parameters from %s", tuning_artifact)
            manifest.extra_metadata["tuning_artifact"] = tuning_artifact.name
            bw = loaded_saved.get("best_weights", {})
            tuned_weights = ScoringWeights(**{k: v for k, v in bw.items() if k in ScoringWeights.__dataclass_fields__})
            if args.w_pop is not None:
                tuned_weights = tuned_weights.with_overrides(w_pop=args.w_pop)
            tuned_config = tuned_config.with_overrides(
                weights=tuned_weights, lambda_div=loaded_saved.get("best_lambda_div", 0.2)
            )
            hp = loaded_saved["hybrid_persona"]
            hybrid_params = {"gamma_rec": hp["gamma_rec"], "n_terms": hp["n_terms"], "n_titles": hp["n_titles"]}
        else:
            logger.info("Running hyperparameter tuning on %d tuning authors...", len(tuning_authors))
            t0 = time.time()
            cached = precompute_tuning_cache(resources, tuning_authors, discovery_map, tuned_config)
            if len(cached) != len(tuning_authors):  # precompute logs and skips failures; tuning must not shrink silently
                raise RuntimeError(f"Tuning cache built for only {len(cached)} of {len(tuning_authors)} tuning authors.")
            t_res = evaluate_parameter_grid(
                cached,
                resources,
                adaptive_graph=tuned_config.adaptive_graph,
                config_signature=tuning_sig,
                base_weights=tuned_config.weights,
            )
            logger.info("Tuning the Hybrid-Persona baseline (gamma_rec, pseudo-query length) on the same authors...")
            t_res.hybrid_persona = tune_hybrid_persona(cached, resources, n_titles=tuned_config.n_titles)
            hybrid_params = {
                "gamma_rec": t_res.hybrid_persona["gamma_rec"],
                "n_terms": t_res.hybrid_persona["n_terms"],
                "n_titles": t_res.hybrid_persona["n_titles"],
            }
            tuning_artifact = tuning_artifact_path(artifacts_dir, tuning_sig)
            t_res.save(tuning_artifact)
            manifest.extra_metadata["tuning_artifact"] = tuning_artifact.name
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

    variant = "grid optimum (headline)"
    if args.facet_active:
        selection_path = default_facet_selection_path(root / cfg["paths"]["artifacts_dir"], args.enhanced)
        fa_weights, fa_lambda = load_facet_selection(selection_path, tuning_sig)
        tuned_config = tuned_config.with_overrides(weights=fa_weights, lambda_div=fa_lambda)
        variant = f"facet-active variant (design §9), frozen in {selection_path.name}"
        logger.info("Running the %s: %s, lambda_div=%.2f", variant, fa_weights, fa_lambda)
    if args.graph_channel:
        variant = (
            "V4 citation-graph channel (design §13.8; exploratory, plan 10.4d); tuning-split decision: "
            + ("adopt V4" if v4_decision.get("adopt_v4") else "keep Core")
        )
        manifest.extra_metadata["v4_decision"] = {k: v4_decision.get(k) for k in ("adopt_v4", "core", "v4", "ndcg_gain")}
    if args.cross_encoder:
        variant = (
            f"V2 cross-encoder feature, {v2_decision['chosen_model']} (design §13.8; exploratory, plan 10.2d); "
            "tuning-split decision: " + ("adopt V2" if v2_decision.get("adopt_v2") else "keep Core")
        )
        manifest.extra_metadata["v2_decision"] = {
            k: v2_decision.get(k) for k in ("adopt_v2", "chosen_model", "core", "v2", "ndcg_gain", "latency")
        }
    if args.encoder:
        variant = (
            f"V1 dense encoder {args.encoder} (design §13.8; exploratory, plan 10.1d); tuning-split decision: "
            + (f"adopt {args.encoder}" if v1_decision.get("adopt_v1") else "keep SPECTER2")
            + "; weights re-tuned on the full tuning split with this encoder; the Dense Centroid and Hybrid-Persona "
            "baselines also use it"
        )
        manifest.extra_metadata["v1_decision"] = {
            k: v1_decision.get(k) for k in ("adopt_v1", "chosen_encoder", "per_encoder", "ndcg_gain", "models")
        }
    if args.learned_ranker:
        # The linear tuned weights stay for the zero-publication slice, which V3 does not rank (design §13.8 V3 point 8)
        tuned_config = tuned_config.with_overrides(
            learned_ranker=resources.ranker.name, v3_lambda_div=float(v3_decision["final_lambda_div"])
        )
        variant = (
            "V3 learned ranker, LightGBM LambdaMART (design §13.8; exploratory, plan 10.3d); tuning-split decision: "
            + ("adopt V3" if v3_decision.get("adopt_v3") else "keep the linear scorer")
            + "; the zero-publication slice keeps the linear scorer"
        )
        manifest.extra_metadata["v3_decision"] = {
            k: v3_decision.get(k) for k in ("adopt_v3", "linear_oof_ndcg", "v3_oof_ndcg", "ndcg_gain", "final_lambda_div",
                                            "model_file", "model_sha256")
        }
    manifest.extra_metadata["variant"] = variant

    # Phase 11e: the comparison systems, built from their frozen tuning (design §13.9 Phase 11 spec points 4 and 6)
    extra_systems, sota_info, rankgpt_stats = None, {}, None
    if args.sota_comparison:
        extra_systems, sota_info, rankgpt_stats = build_sota_systems(resources, root, cfg)
        manifest.extra_metadata["sota_systems"] = sota_info
        manifest.extra_metadata["best_config_sha256"] = _sha256(root / "configs" / "best.yaml")
        variant = "frozen best configuration (configs/best.yaml: Core) with the Phase 11 comparison systems"
        manifest.extra_metadata["variant"] = variant

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
        baseline_kwargs={"hybrid_persona": hybrid_params},
        extra_systems=extra_systems,
    )
    logger.info("Evaluation loop finished in %.1fs.", time.time() - t0)

    # 3. Statistical hypothesis testing
    hypo_report = run_track1_hypothesis_tests(_series(lfo_results.per_user_results), n_resamples=args.n_bootstrap)

    # 4. Slices: history volume, item cold start, zero-publication re-run
    slice_breakdown = slice_by_history_volume(lfo_results.per_user_results)

    evaluated_targets = {r.author_id: discovery_map.get(r.author_id, []) for r in lfo_results.per_user_results}
    target_class = cold_start_classes(
        resources.db, {t for ts in evaluated_targets.values() for t in ts}, resources.cutoff_year
    )
    cold_start = item_cold_start_breakdown(lfo_results.per_user_results, evaluated_targets, target_class)
    extra_sections = [("Item Cold-Start Slice", format_cold_start_table(cold_start))]
    if args.sota_comparison:
        extra_sections += phase11_sections(lfo_results, best_cfg, sota_info, rankgpt_stats, args.n_bootstrap)

    zero_pub_results = None
    if not args.skip_zero_pub:
        logger.info("Running the Zero-Publication Seed-Only slice over %d test authors...", len(test_authors))
        t0 = time.time()
        seed_subfields = seed_topic_subfields(resources.db, root / cfg["paths"]["raw_dir"], test_authors)
        zero_pub_results = run_zero_pub_evaluation(
            resources=resources,
            cohort=test_authors,
            zero_pub_ground_truth=zero_pub_map,
            topic_map=topic_map,
            paper_id_index=resources.store.paper_id_index,
            embedding_matrix=resources.store.vectors,
            paper_facet_map=paper_facet_map,
            seed_subfields=seed_subfields,
            config=tuned_config,
            k=10,
            baseline_kwargs={"hybrid_persona": hybrid_params},
        )
        zero_pub_comparisons = compare_system_to_baselines(
            _series(zero_pub_results.per_user_results),
            "Paper Feed (Full)",
            [f"Baseline: {b}" for b in ZERO_PUB_BASELINES],
            metrics=H1_METRICS,
            n_resamples=args.n_bootstrap,
        )
        extra_sections.append((
            "Zero-Publication Seed-Only Slice",
            format_zero_pub_section(zero_pub_results.system_aggregates, zero_pub_comparisons, zero_pub_results.n_authors),
        ))
        manifest.extra_metadata["zero_pub_seed_subfield_found"] = sum(1 for v in seed_subfields.values() if v)
        logger.info("Zero-publication slice finished in %.1fs.", time.time() - t0)

    # 5. Generate and save report
    run_notes = (
        f"> [!NOTE]\n"
        f"> **Evaluation Run Configuration:**\n"
        f"> - Mode: {'Enhanced (Option 2 & Option 3)' if (tuned_config.weights.w_pop > 0 or (tuned_config.pool_size and tuned_config.pool_size > 100)) else 'Baseline'}\n"
        f"> - Candidate Pool Size (M): {tuned_config.pool_size or 100}\n"
        f"> - Candidate Channels: {'SPECTER2 + BM25s + citation graph (V4)' if tuned_config.graph_channel else 'SPECTER2 + BM25s'}"
        f"{' (dense channel: ' + tuned_config.dense_encoder + ', V1)' if tuned_config.dense_encoder != 'specter2' else ''}\n"
        + (f"> - Reranker Feature (V2): {tuned_config.cross_encoder}, w_ce={tuned_config.weights.w_ce:.2f}\n" if tuned_config.cross_encoder else "")
        + (f"> - Scorer (V3): learned ranker {tuned_config.learned_ranker}, its lambda_div={tuned_config.v3_lambda_div:.2f} "
           "(the linear weights below are used only for the zero-publication slice)\n" if tuned_config.learned_ranker else "")
        + f"> - Channel Depth (M_channel): {tuned_config.channel_depth or 200}\n"
        f"> - Popularity Prior Weight (w_pop): {tuned_config.weights.w_pop:.2f}\n"
        f"> - Scoring Weights: w_pers={tuned_config.weights.w_pers:.2f}, w_cite={tuned_config.weights.w_cite:.2f}, "
        f"w_facet={tuned_config.weights.w_facet:.2f}, w_nov={tuned_config.weights.w_nov:.2f}, "
        f"lambda_div={tuned_config.lambda_div:.2f}\n"
        f"> - System variant: {variant}\n"
        f"> - Hybrid-Persona baseline: gamma_rec={hybrid_params['gamma_rec']:.2f}, n_terms={hybrid_params['n_terms']}\n"
        f"> - Report: `{args.report_name}` | Raw JSON: `{args.results_name}` | Run manifest: `results/runs/{manifest.run_id}/manifest.json`"
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
        extra_sections=extra_sections,
        graph_channel=tuned_config.graph_channel,
    )

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / args.report_name
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    json_file = out_dir / args.results_name
    lfo_results.save(json_file)
    zero_pub_file = None
    if zero_pub_results is not None:
        zero_pub_file = out_dir / f"{Path(args.results_name).stem}_zero_pub.json"
        zero_pub_results.save(zero_pub_file)

    # Every table above comes from this one run: keep copies next to its manifest.
    for f in (report_file, json_file, zero_pub_file):
        if f is not None:
            shutil.copy2(f, manifest.run_dir / f.name)
    manifest.complete(metadata={
        "outputs": [
            str(f.relative_to(root)) if f.is_relative_to(root) else str(f)  # --out-dir may point outside the repo
            for f in (report_file, json_file, zero_pub_file) if f is not None
        ],
        "n_test_authors_evaluated": lfo_results.n_authors,
        "n_zero_pub_authors_evaluated": zero_pub_results.n_authors if zero_pub_results is not None else None,
        "tuned_weights": vars(tuned_config.weights),
        "tuned_lambda_div": tuned_config.lambda_div,
        "hybrid_persona_params": hybrid_params,
        "tuning_signature": tuning_sig,
        "popularity_filter": pop_filter_stats,
        "h1_supported": hypo_report.h1_supported,
        "h2_supported": hypo_report.h2_supported,
        "input_sha256": {
            name: _sha256(path)
            for name, path in {
                "cohort.jsonl": processed_dir / "cohort.jsonl",
                "ground_truth.jsonl": processed_dir / "ground_truth.jsonl",
                "split.json": processed_dir / "split.json",
                **({tuning_artifact.name: tuning_artifact} if tuning_artifact is not None else {}),
                "embedding_ids.json": processed_dir / "embedding_ids.json",
                **({f"encoder_{args.encoder}/embedding_ids.json": processed_dir / f"encoder_{args.encoder}" / "embedding_ids.json"}
                   if args.encoder else {}),
                "index_manifest.json": processed_dir / "index_manifest.json",
            }.items()
        },
        "runtime_seconds": round(time.time() - t_start, 1),
    })
    logger.info("Run manifest: %s", manifest.manifest_path)

    logger.info("[+] Evaluation complete! Report written to %s", report_file)
    print("\n" + "=" * 90)
    print(report_text)
    print("=" * 90)


if __name__ == "__main__":
    main()
