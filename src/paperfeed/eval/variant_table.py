"""Final variant table (plan 10.5 / docs/design.md §13.8): every variant's single test run next to Core.

Built only from saved outputs: the Track 1 results JSON of each run and the frozen tuning-split decisions. Each
variant is compared with Core on the same 320 test authors (paired bootstrap, two-sided), with Benjamini-Hochberg
correction across the whole family (variants x metrics). The table reports; it never selects (design §13.8: choices
come from tuning evidence only, Phase 11).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from paperfeed.eval.stats import benjamini_hochberg_correction, paired_bootstrap_test

FULL = "Paper Feed (Full)"
COMPARED_METRICS = ("ndcg_at_10", "recall_at_10")
TABLE_METRICS = ("ndcg_at_10", "recall_at_10", "hit_rate_at_10", "mrr", "recall_at_100", "topic_entropy", "ild")


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    change: str
    results_file: str
    decision_file: Optional[str]
    adopt_key: Optional[str]


VARIANTS: Sequence[Variant] = (
    Variant("core", "Core", "linear score over the SPECTER2 + BM25s pool", "track1_lfo_results.json", None, None),
    Variant("v1", "V1 encoder (BGE)", "dense encoder bge-base-en-v1.5; grid re-tuned with it",
            "track1_lfo_results_encoder_bge.json", "variant_v1_decision.json", "adopt_v1"),
    Variant("v2", "V2 cross-encoder", "+ w_ce * reranker score (ms-marco-MiniLM-L-6-v2)",
            "track1_lfo_results_cross_encoder.json", "variant_v2_decision.json", "adopt_v2"),
    Variant("v3", "V3 learned ranker", "LightGBM LambdaMART instead of the linear score",
            "track1_lfo_results_learned_ranker.json", "variant_v3_decision.json", "adopt_v3"),
    Variant("v4", "V4 citation channel", "+ citation-graph candidate channel in RRF",
            "track1_lfo_results_graph_channel.json", "variant_v4_decision.json", "adopt_v4"),
)


def load_run(path: Path) -> Dict[str, Dict[str, Any]]:
    """author_id -> that author's row (system_metrics, n_targets) from a Track 1 results JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {row["author_id"]: row for row in data["per_user_results"]}


def check_same_test_set(runs: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> List[str]:
    """The paired comparison needs every run to cover the same authors with the same number of targets."""
    keys = list(runs)
    ref = runs[keys[0]]
    for k in keys[1:]:
        if set(runs[k]) != set(ref):
            raise ValueError(f"Run {k!r} covers different test authors than {keys[0]!r}.")
        bad = [a for a in ref if runs[k][a].get("n_targets") != ref[a].get("n_targets")]
        if bad:
            raise ValueError(f"Run {k!r} has different targets for {len(bad)} authors (e.g. {bad[0]}).")
    return sorted(ref)


def series(run: Mapping[str, Mapping[str, Any]], authors: Sequence[str], metric: str, system: str = FULL) -> np.ndarray:
    return np.array([run[a]["system_metrics"][system][metric] for a in authors], dtype=np.float64)


def compare_with_core(runs, authors, variant_keys, metrics=COMPARED_METRICS, n_resamples=10000, seed=42) -> List[Dict[str, Any]]:
    """Each variant minus Core on each metric, paired over authors; BH across the whole family."""
    rows = []
    for key in variant_keys:
        for metric in metrics:
            v, c = series(runs[key], authors, metric), series(runs["core"], authors, metric)
            diff, lo, hi, p, d = paired_bootstrap_test(v, c, n_resamples=n_resamples, seed=seed)
            rows.append({"variant": key, "metric": metric, "variant_mean": float(v.mean()), "core_mean": float(c.mean()),
                         "diff": diff, "relative": diff / c.mean() if c.mean() > 0 else None, "ci": [lo, hi],
                         "p_value": p, "cohens_d": d})
    for row, (p_bh, sig) in zip(rows, benjamini_hochberg_correction([r["p_value"] for r in rows])):
        row["p_bh"], row["significant"] = p_bh, bool(sig)
    return rows


def decision_summary(decision: Mapping[str, Any], adopt_key: str) -> Dict[str, Any]:
    g = decision["ndcg_gain"]
    return {"gain": g["mean_diff"], "relative": g.get("relative"), "ci": g["ci"], "adopted": bool(decision[adopt_key])}


def build_variant_table(results_dir: Path, artifacts_dir: Path, n_resamples: int = 10000) -> Dict[str, Any]:
    runs = {v.key: load_run(Path(results_dir) / v.results_file) for v in VARIANTS}
    authors = check_same_test_set(runs)
    decisions = {
        v.key: decision_summary(json.loads((Path(artifacts_dir) / v.decision_file).read_text(encoding="utf-8")), v.adopt_key)
        for v in VARIANTS if v.decision_file
    }
    means = {v.key: {m: float(series(runs[v.key], authors, m).mean()) for m in TABLE_METRICS} for v in VARIANTS}
    citation_only = {m: float(series(runs["core"], authors, m, "Baseline: citation_only").mean()) for m in TABLE_METRICS}
    comparisons = compare_with_core(runs, authors, [v.key for v in VARIANTS if v.key != "core"], n_resamples=n_resamples)
    return {"n_authors": len(authors), "means": means, "citation_only": citation_only, "decisions": decisions,
            "comparisons": comparisons, "n_resamples": n_resamples}


def _ci(lo: float, hi: float) -> str:
    return f"[{lo:+.4f}, {hi:+.4f}]"


def format_variant_table(t: Mapping[str, Any]) -> str:
    labels = {v.key: v for v in VARIANTS}
    lines = [
        "# Phase 10 — Final Variant Table (plan 10.5)\n",
        "> [!NOTE]",
        f"> - Every variant was decided on the 80 tuning authors before its single test run on the same {t['n_authors']} "
        "test authors (design §13.8). This table reports; it is never used to choose a variant (Phase 11 uses tuning "
        "evidence only).",
        f"> - Variant vs Core: paired bootstrap over authors ({t['n_resamples']:,} resamples), two-sided, Benjamini-Hochberg "
        "across all variant x metric comparisons. All variants are exploratory; the Core run answers H1.",
        "> - ILD is computed with each run's own embeddings, so V1 (BGE) ILD is not comparable with the SPECTER2 rows. "
        "V1's Dense Centroid and Hybrid-Persona baselines also use BGE. V3 ranks authors with papers only; its "
        "zero-publication slice is the linear scorer.\n",
        "## 1. Variants\n",
        "| Variant | What changes | Tuning-split NDCG@10 gain [95% CI] | Decision | NDCG@10 | Recall@10 | HitRate@10 | MRR | "
        "Pool Recall@100 | Topic Entropy | ILD |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for key, m in t["means"].items():
        v = labels[key]
        if key == "core":
            tuning, decision = "-", "reference"
        else:
            d = t["decisions"][key]
            rel = f" ({d['relative']:+.0%})" if d["relative"] is not None else ""
            tuning = f"{d['gain']:+.4f}{rel} {_ci(*d['ci'])}"
            decision = "adopted" if d["adopted"] else "not adopted"
        ild = f"{m['ild']:.4f}" + ("*" if key == "v1" else "")
        lines.append(
            f"| {'**' + v.label + '**' if key == 'core' else v.label} | {v.change} | {tuning} | {decision} | "
            f"{m['ndcg_at_10']:.4f} | {m['recall_at_10']:.4f} | {m['hit_rate_at_10']:.4f} | {m['mrr']:.4f} | "
            f"{m['recall_at_100']:.4f} | {m['topic_entropy']:.4f} | {ild} |"
        )
    c = t["citation_only"]
    lines += [
        f"| *Baseline: Citation-only* | graph signal over the whole corpus | - | - | {c['ndcg_at_10']:.4f} | "
        f"{c['recall_at_10']:.4f} | {c['hit_rate_at_10']:.4f} | {c['mrr']:.4f} | - | {c['topic_entropy']:.4f} | - |",
        "",
        "*\\* not comparable: computed with BGE embeddings. Pool Recall@100 for Citation-only is not defined (it has no pool).*",
        "",
        "## 2. Each variant against Core on the test split\n",
        "| Variant | Metric | Variant | Core | Difference [95% CI] | Rel. | p (raw) | p (BH) | Cohen's d | Significant? |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in t["comparisons"]:
        rel = f"{r['relative']:+.1%}" if r["relative"] is not None else "-"
        lines.append(
            f"| {labels[r['variant']].label} | {r['metric']} | {r['variant_mean']:.4f} | {r['core_mean']:.4f} | "
            f"{r['diff']:+.4f} {_ci(*r['ci'])} | {rel} | {r['p_value']:.4f} | {r['p_bh']:.4f} | {r['cohens_d']:+.3f} | "
            f"{'YES' if r['significant'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"
