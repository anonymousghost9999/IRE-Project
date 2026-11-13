#!/usr/bin/env python3
"""V1 encoder study on the tuning split: build the 20k subset, compare encoders, freeze the decision (plan 10.1c).

Usage:
    python scripts/decide_encoder.py [--n-bootstrap 10000] [--overwrite]
    python scripts/decide_encoder.py --dry-run 10 --size 3000    # smoke test: writes no artifact

Rule (design §13.8, V1 specification, fixed before implementation): SPECTER2, SciNCL and BGE are each embedded on the
same encoder-neutral 20k-paper subset around the tuning authors and grid-tuned on the 80 tuning authors; the
alternative with the higher full-system NDCG@10 at its optimum replaces SPECTER2 only if the paired-bootstrap 95% CI
of its gain lies above zero. Qwen3-Embedding is never used here (released after T).

Only the 80 tuning authors are touched. Outputs:
    data/artifacts/variant_v1_decision.json    the frozen decision (refuses to overwrite without --overwrite)
    data/artifacts/variant_v1_subset_ids.json  the subset, for reproducibility
    results/variant_v1_decision.md             the readable table
Subset vectors are cached in <processed_dir>/encoder_subset/<encoder>/ (resumable).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paperfeed.common.config import load_config
from paperfeed.common.logger import get_logger
from paperfeed.common.manifest import RunManifest
from paperfeed.data.cohort import clean_id
from paperfeed.eval.encoder_study import (
    SUBSET_SIZE,
    build_subset_lexical,
    decide_encoder,
    embed_subset,
    evaluate_arm,
    graph_subset,
    paper_texts,
    subset_fingerprint,
    subset_resources,
)
from paperfeed.index.encoders import ENCODER_SPECS, make_encoder
from paperfeed.index.store import EmbeddingStore
from paperfeed.rec.recommender import RecommendConfig, RecommenderResources

logger = get_logger("scripts.decide_encoder")

DECISION_NAME = "variant_v1_decision"
ALTERNATIVES = ("scincl", "bge")  # design §13.8 V1 point 1
LABEL = {"specter2": "SPECTER2", "scincl": "SciNCL", "bge": "BGE"}


def _fmt(p: dict) -> str:
    return f"w_cite={p['w_cite']}, w_facet={p['w_facet']}, w_nov={p['w_nov']}, lambda_div={p['lambda_div']}"


def format_report(d: dict) -> str:
    g = d["ndcg_gain"]
    rel = f"{g['relative']:+.1%}" if g["relative"] is not None else "n/a"
    s = d["subset"]
    lines = [
        "# V1 Encoder Study: Tuning-Split Decision\n",
        f"Rule (design §13.8, fixed before implementation): {d['rule']['statement']} {d['n_users']} tuning authors, "
        f"{d['rule']['n_resamples']:,} paired-bootstrap resamples. The test split is not touched.\n",
        f"- Subset: {s['size']:,} papers ({s['n_seeds']:,} tuning-author papers, {s['n_hop1']:,} one-hop and "
        f"{s['n_hop2']:,} two-hop citation neighbours, chosen from the <= T graph only); "
        f"{s['target_share']:.1%} of the tuning authors' Discovery targets fall inside it. Candidates are restricted to "
        "the subset, so absolute numbers are not comparable with Track 1.",
        "- BM25s and facets are shared by all encoders; Qwen3-Embedding (post-T) is excluded from this decision.",
        "",
        "| Encoder | Grid optimum | Full system NDCG@10 | Dense Centroid NDCG@10 | Candidate Recall@100 |",
        "| :--- | :--- | :---: | :---: | :---: |",
    ]
    for key, p in d["per_encoder"].items():
        mark = " (V1 candidate)" if key == d["chosen_encoder"] else ""
        lines.append(f"| {LABEL[key]}{mark} | {_fmt(p)} | {p['mean_ndcg']:.4f} | {p['dense_centroid_ndcg']:.4f} | "
                     f"{p['candidate_recall_at_100']:.4f} |")
    lines += [
        "",
        f"- Full-system NDCG@10 gain of {LABEL[d['chosen_encoder']]} over SPECTER2: {g['mean_diff']:+.4f} ({rel}), 95% CI "
        f"[{g['ci'][0]:+.4f}, {g['ci'][1]:+.4f}], p = {g['p_value']:.4f}, Cohen's d = {g['cohens_d']:.3f}",
    ]
    for key, c in d["dense_centroid_gain"].items():
        lines.append(f"- Dense Centroid gain of {LABEL[key]} over SPECTER2: {c['mean_diff']:+.4f}, 95% CI "
                     f"[{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}] (reported, not decisive)")
    lines += [
        "",
        f"**Decision: {LABEL[d['chosen_encoder']]} replaces SPECTER2.**" if d["adopt_v1"] else
        "**Decision: keep SPECTER2** (the CI of the full-system NDCG@10 gain does not lie above zero).",
        "",
        f"*Either way the V1 candidate ({LABEL[d['chosen_encoder']]}) gets the full corpus embedded, its grid re-tuned "
        "on the full tuning split and one labelled Track 1 test run (plan 10.1d).*",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the V1 encoder decision on the tuning split.")
    parser.add_argument("--size", type=int, default=SUBSET_SIZE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing frozen decision")
    parser.add_argument("--dry-run", type=int, default=None, metavar="N",
                        help="Smoke test on the first N tuning authors: prints the report, writes no artifact")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = load_config()
    artifacts = root / cfg["paths"]["artifacts_dir"]
    processed = root / cfg["paths"]["processed_dir"]
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
        cohort = [json.loads(l) for l in open(processed / "cohort.jsonl", encoding="utf-8") if l.strip()]
        tuning_authors = [a for a in cohort if a.get("split") == "tuning"]
        if args.dry_run is not None:
            tuning_authors = tuning_authors[: args.dry_run]
        discovery = {}
        for line in open(processed / "ground_truth.jsonl", encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                discovery[clean_id(row["author_id"])] = [clean_id(p) for p in row["discovery_in_corpus"]]

        base = RecommenderResources.load(cfg, root, load_encoder=True, use_facets=True)
        if base.facet_store is None and args.dry_run is None:
            raise RuntimeError("facet_canon is empty: run on the machine with the full facets, as for Track 1.")
        core_rc = RecommendConfig.from_config(cfg)

        # 1. The subset: eligible papers with a SPECTER2 vector (so every encoder embeds the same papers), plus seeds
        stored = set(base.store.paper_id_index)
        seeds = [clean_id(p) for a in tuning_authors for p in a.get("authored_paper_ids_T", [])]
        allowed = (set(base.eligible) & stored) | (set(seeds) & stored)
        subset, info = graph_subset(base.db, seeds, allowed, base.cutoff_year, size=args.size)
        targets = {t for a in tuning_authors for t in discovery.get(clean_id(a["author_id"]), [])}
        info["target_share"] = len(targets & set(subset)) / len(targets) if targets else 0.0
        info["sha256"] = subset_fingerprint(subset)
        logger.info("Subset: %s", info)
        texts = paper_texts(base.db, subset)
        lexical = build_subset_lexical(texts)

        # 2. One arm per encoder; SPECTER2 reuses its stored vectors
        arms = {}
        spec_store = EmbeddingStore(subset, base.store.select(subset))
        arms["specter2"] = evaluate_arm(subset_resources(base, spec_store, lexical, base.encoder, subset),
                                        tuning_authors, discovery, core_rc)
        for key in ALTERNATIVES:
            encoder = make_encoder(key, track1=True)
            cache = processed / "encoder_subset" / (key if args.dry_run is None else f"{key}_dry_run")
            store = embed_subset(encoder, ENCODER_SPECS[key].model_name, texts, cache, args.batch_size)
            arms[key] = evaluate_arm(subset_resources(base, store, lexical, encoder, subset), tuning_authors, discovery,
                                     core_rc)
            del encoder
            gc.collect()

        decision = decide_encoder(arms, n_resamples=args.n_bootstrap)
        decision["subset"] = info
        decision["models"] = {k: (cfg["models"]["dense_encoder"] if k == "specter2" else ENCODER_SPECS[k].model_name)
                              for k in arms}
        report = format_report(decision)
        print(report)
        if args.dry_run is not None:
            logger.info("Dry run: nothing written.")
            return 0

        subset_path = artifacts / "variant_v1_subset_ids.json"
        subset_path.write_text(json.dumps({"sha256": info["sha256"], "ids": subset}), encoding="utf-8")
        decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
        out_md = root / args.out_dir / f"{DECISION_NAME}.md"
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(report, encoding="utf-8")
        for f in (decision_path, out_md):
            shutil.copy2(f, manifest.run_dir / f.name)
        manifest.complete(metadata={
            "adopt_v1": decision["adopt_v1"],
            "chosen_encoder": decision["chosen_encoder"],
            "per_encoder": decision["per_encoder"],
            "ndcg_gain": decision["ndcg_gain"],
            "subset": info,
            "outputs": [str(decision_path), str(subset_path), str(out_md)],
        })
    except BaseException as exc:
        if manifest is not None:
            manifest.complete(status="FAILED", metadata={"error": repr(exc)})
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
