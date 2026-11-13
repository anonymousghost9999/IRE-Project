#!/usr/bin/env python3
"""Select the facet-active variant on the tuning split (design §9 / plan §6.3). Run before any test-split run.

Usage:
    python scripts/select_facet_config.py [--enhanced] [--max-relative-loss 0.05] [--min-facet-share 0.10]

Rule (fixed before the test split is touched): among grid points with w_facet > 0, in order of tuning-split mean
Discovery NDCG@10, select the first whose relative NDCG@10 loss against the grid optimum has a paired-bootstrap 95% CI
upper end <= --max-relative-loss AND whose tuning authors' Top-10 items have >= --min-facet-share facet or new-method
explanations. The result (a selection, or "none") is frozen in data/artifacts/facet_config_selection[_enhanced].json
together with the configuration signature; ``run_track1_eval.py --facet-active`` and
``audit_faithfulness.py --facet-active`` refuse a selection made for another configuration.
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
    default_facet_selection_path,
    grid_per_user_ndcg,
    precompute_tuning_cache,
    select_facet_active_config,
    tuning_signature,
)
from paperfeed.rec.recommender import RecommendConfig, RecommenderResources

logger = get_logger("scripts.select_facet_config")


def run_config(cfg, args) -> RecommendConfig:
    """The same configuration run_track1_eval.py builds before tuning (keep the two in step)."""
    rc = RecommendConfig.from_config(cfg)
    if args.enhanced:
        rc = rc.with_overrides(pool_size=250, channel_depth=500, weights=rc.weights.with_overrides(w_pop=0.20))
    return rc


def format_report(sel: dict, signature: dict) -> str:
    rule = sel["rule"]
    lines = [
        "# Facet-Active Variant Selection (tuning split)\n",
        f"Rule (design §9): among grid points with w_facet > 0 in order of tuning-split NDCG@10, the first whose relative "
        f"NDCG@10 loss against the optimum has a 95% paired-bootstrap CI upper end <= {rule['max_relative_loss']:.0%} and "
        f"whose Top-{rule['k']} items have >= {rule['min_facet_share']:.0%} facet or new-method explanations. "
        f"{sel['n_users']} tuning authors, {rule['n_resamples']:,} resamples.\n",
        f"- Configuration: pool {signature['pool_size']}, channel depth {signature['channel_depth']}, w_pop {signature['w_pop']}, "
        f"adaptive_graph {signature['adaptive_graph']}",
        f"- Grid optimum (the headline system): w_cite={sel['best']['w_cite']}, w_facet={sel['best']['w_facet']}, "
        f"w_nov={sel['best']['w_nov']}, lambda_div={sel['best']['lambda_div']}, NDCG@10 {sel['best']['mean_ndcg']:.4f}, "
        f"templates {', '.join(f'{k} {v:.0%}' for k, v in sel['best']['template_shares'].items())}",
        "",
        "| w_cite | w_facet | w_nov | lambda_div | NDCG@10 | Rel. loss [95% CI] | Facet / new-method share | Non-inferior | Enough facets | Accepted |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for c in sel["candidates"]:
        lo, hi = c["relative_loss_ci"]
        lines.append(
            f"| {c['w_cite']} | {c['w_facet']} | {c['w_nov']} | {c['lambda_div']} | {c['mean_ndcg']:.4f} | "
            f"{c['relative_loss']:+.1%} [{lo:+.1%}, {hi:+.1%}] | {c['facet_share']:.1%} | {'yes' if c['non_inferior'] else 'no'} | "
            f"{'yes' if c['enough_facet_explanations'] else 'no'} | {'**yes**' if c['accepted'] else 'no'} |"
        )
    chosen = sel["selected"]
    lines.append("")
    lines.append(
        f"**Selected:** w_cite={chosen['w_cite']}, w_facet={chosen['w_facet']}, w_nov={chosen['w_nov']}, "
        f"lambda_div={chosen['lambda_div']} (run it with `--facet-active`)." if chosen else
        "**Selected: none.** No facet-active configuration met both conditions; report this, with the accuracy or "
        "explainability cost that ruled each candidate out."
    )
    lines.append("\n*Template shares count the dominant feature of each Top-10 item, i.e. the explanation the system "
                 "would show; grounding of those explanations is checked by the faithfulness audit.*")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Select the facet-active variant on the tuning split.")
    parser.add_argument("--enhanced", action="store_true", help="Select for the --enhanced Track 1 configuration")
    parser.add_argument("--max-relative-loss", type=float, default=0.05)
    parser.add_argument("--min-facet-share", type=float, default=0.10)
    parser.add_argument("--n-candidates", type=int, default=10)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = load_config()
    run_id = f"facet_config_selection_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    manifest = RunManifest(run_id, cfg, runs_dir=root / "results" / "runs")
    manifest.extra_metadata["arguments"] = vars(args)
    try:
        processed = root / cfg["paths"]["processed_dir"]
        cohort = [json.loads(l) for l in open(processed / "cohort.jsonl", encoding="utf-8") if l.strip()]
        tuning_authors = [a for a in cohort if a.get("split") == "tuning"]
        discovery = {}
        for line in open(processed / "ground_truth.jsonl", encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                discovery[clean_id(row["author_id"])] = [clean_id(p) for p in row["discovery_in_corpus"]]

        rc = run_config(cfg, args)
        signature = tuning_signature(rc, [a["author_id"] for a in tuning_authors])
        resources = RecommenderResources.load(cfg, root, load_encoder=True, use_facets=True)
        if resources.facet_store is None:
            raise RuntimeError("facet_canon is empty: facet features are off, so there is nothing to select.")

        logger.info("Caching candidate pools and features for %d tuning authors...", len(tuning_authors))
        cached = precompute_tuning_cache(resources, tuning_authors, discovery, rc)
        if len(cached) != len(tuning_authors):
            raise RuntimeError(f"Only {len(cached)} of {len(tuning_authors)} tuning authors could be cached.")
        points = grid_per_user_ndcg(cached, resources, adaptive_graph=rc.adaptive_graph, base_weights=rc.weights)
        sel = select_facet_active_config(
            points, cached, resources, adaptive_graph=rc.adaptive_graph, max_relative_loss=args.max_relative_loss,
            min_facet_share=args.min_facet_share, n_candidates=args.n_candidates, n_resamples=args.n_bootstrap,
        )
        sel["config_signature"] = signature

        out_json = default_facet_selection_path(root / cfg["paths"]["artifacts_dir"], args.enhanced)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(sel, indent=2), encoding="utf-8")
        out_md = root / args.out_dir / f"{out_json.stem}.md"
        out_md.parent.mkdir(parents=True, exist_ok=True)
        report = format_report(sel, signature)
        out_md.write_text(report, encoding="utf-8")
        for f in (out_json, out_md):
            shutil.copy2(f, manifest.run_dir / f.name)
        manifest.complete(metadata={"selected": sel["selected"], "best": sel["best"], "outputs": [str(out_json), str(out_md)]})
        print(report)
    except BaseException as exc:
        manifest.complete(status="FAILED", metadata={"error": repr(exc)})
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
