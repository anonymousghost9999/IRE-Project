"""Leak-free sensitivity analysis of the published (original-cohort) results (docs/design.md §18a).

A post-hoc audit found that some authors' history <= T contained a version of their own target paper B (a preprint
under another ID). Instead of re-running every experiment on a rebuilt cohort, every comparison is recomputed from the
saved per-author results on the authors without that leak. Every system was frozen before its test run, so dropping
authors leaves a valid paired comparison of the same frozen systems; the tuning authors' leak (which could only have
influenced the chosen weights) is reported separately as a limitation.

Families and statistics are exactly the published ones: H1/H2 (``run_track1_hypothesis_tests``), the Phase 10 variant
table (each variant vs Core, BH over 8) and the Phase 11 comparison (Paper Feed vs each comparison system, BH over 8).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from paperfeed.eval.stats import H1_METRICS, compare_system_to_baselines, run_track1_hypothesis_tests
from paperfeed.eval.variant_table import compare_with_core

FULL = "Paper Feed (Full)"


def load_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    """author_id -> row (system_metrics, ...) of a saved Track 1 results JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {r["author_id"]: r for r in data["per_user_results"]}


def series(rows: Mapping[str, Mapping[str, Any]], authors: Sequence[str]) -> Dict[str, Dict[str, List[float]]]:
    """{system: {metric: [value per author in ``authors`` order]}} for the paired tests."""
    out: Dict[str, Dict[str, List[float]]] = {}
    for a in authors:
        for sysname, metrics in rows[a]["system_metrics"].items():
            for m, v in metrics.items():
                out.setdefault(sysname, {}).setdefault(m, []).append(v)
    return out


def _cmp_record(c) -> Dict[str, Any]:
    return {"system": c.mean_system, "other": c.mean_baseline, "diff": c.mean_diff, "ci": [c.ci_lower, c.ci_upper],
            "p_bh": c.p_value_corrected, "significant": bool(c.significant_at_05)}


def h1_h2(rows, authors, n_resamples: int) -> Dict[str, Any]:
    rep = run_track1_hypothesis_tests(series(rows, authors), n_resamples=n_resamples)
    return {
        "n_authors": len(authors),
        "h1_supported": rep.h1_supported,
        "h1": {f"{c.baseline_name} | {c.metric_name}": _cmp_record(c) for c in rep.h1_comparisons},
        "h2_supported": rep.h2_supported,
        "h2_entropy": _cmp_record(rep.h2_mmr_entropy),
        "h2_ndcg": _cmp_record(rep.h2_mmr_ndcg_retention),
    }


def sota_family(rows, authors, n_resamples: int) -> Dict[str, Any]:
    names = sorted({s for a in authors for s in rows[a]["system_metrics"] if s.startswith("SOTA: ")})
    comps = compare_system_to_baselines(series(rows, authors), FULL, names, metrics=H1_METRICS, n_resamples=n_resamples)
    return {f"{c.baseline_name} | {c.metric_name}": _cmp_record(c) for c in comps}


def variant_family(core_rows, variant_rows: Mapping[str, Mapping[str, Any]], authors, n_resamples: int) -> Dict[str, Any]:
    runs = {"core": core_rows, **variant_rows}
    out = {}
    for r in compare_with_core(runs, list(authors), list(variant_rows), n_resamples=n_resamples):
        out[f"{r['variant']} | {r['metric']}"] = {"system": r["variant_mean"], "other": r["core_mean"], "diff": r["diff"],
                                                 "ci": r["ci"], "p_bh": r["p_bh"], "significant": r["significant"]}
    return out


def compare_sets(full: Mapping[str, Any], clean: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """One row per comparison: all authors vs leak-free authors, flagging any change of significance or sign."""
    rows = []
    for key, f in full.items():
        c = clean[key]
        changed = f["significant"] != c["significant"] or (f["significant"] and np.sign(f["diff"]) != np.sign(c["diff"]))
        rows.append({"comparison": key, "all": f, "clean": c, "conclusion_changed": bool(changed)})
    return rows


def run_sensitivity(core_rows, variant_rows, leaked: Sequence[str], n_resamples: int = 10000) -> Dict[str, Any]:
    all_authors = sorted(core_rows)
    for name, rows in variant_rows.items():
        if set(rows) != set(all_authors):
            raise ValueError(f"Variant run {name!r} covers different test authors than the Core run.")
    leaked_test = sorted(set(leaked) & set(all_authors))
    clean_authors = [a for a in all_authors if a not in set(leaked_test)]
    out: Dict[str, Any] = {"n_test_all": len(all_authors), "n_test_leaked": len(leaked_test), "n_test_clean": len(clean_authors),
                           "leaked_test_authors": leaked_test, "n_resamples": n_resamples}
    hh_all, hh_clean = h1_h2(core_rows, all_authors, n_resamples), h1_h2(core_rows, clean_authors, n_resamples)
    out["h1"] = compare_sets(hh_all["h1"], hh_clean["h1"])
    out["h1_supported"] = {"all": hh_all["h1_supported"], "clean": hh_clean["h1_supported"]}
    out["h2"] = compare_sets({"entropy": hh_all["h2_entropy"], "ndcg retention": hh_all["h2_ndcg"]},
                             {"entropy": hh_clean["h2_entropy"], "ndcg retention": hh_clean["h2_ndcg"]})
    out["h2_supported"] = {"all": hh_all["h2_supported"], "clean": hh_clean["h2_supported"]}
    out["variants"] = compare_sets(variant_family(core_rows, variant_rows, all_authors, n_resamples),
                                   variant_family(core_rows, variant_rows, clean_authors, n_resamples))
    out["sota"] = compare_sets(sota_family(core_rows, all_authors, n_resamples),
                               sota_family(core_rows, clean_authors, n_resamples))
    return out


def _row(r: Mapping[str, Any]) -> str:
    def cell(x):
        sig = "**sig.**" if x["significant"] else "n.s."
        return f"{x['system']:.4f} vs {x['other']:.4f}, diff {x['diff']:+.4f} [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}], p_BH {x['p_bh']:.3f} ({sig})"
    return f"| {r['comparison']} | {cell(r['all'])} | {cell(r['clean'])} | {'**CHANGED**' if r['conclusion_changed'] else 'same'} |"


def format_report(s: Mapping[str, Any], leaked_info: Mapping[str, Mapping[str, str]]) -> str:
    n_changed = sum(r["conclusion_changed"] for key in ("h1", "h2", "variants", "sota") for r in s[key])
    tuning_leaked = sorted(a for a, v in leaked_info.items() if v.get("split") == "tuning")
    head = "| Comparison | All test authors (n = {a}) | Leak-free test authors (n = {c}) | Conclusion |".format(
        a=s["n_test_all"], c=s["n_test_clean"])
    sep = "| :--- | :--- | :--- | :---: |"
    lines = [
        "# Leak-Free Sensitivity Analysis of the Published Results\n",
        "> [!NOTE]",
        f"> - A post-hoc audit (2026-11-15, design §18a) found that {len(leaked_info)} of the 400 authors of the original "
        f"cohort had a version of their own target paper B (same title, another OpenAlex ID, e.g. a preprint) in their "
        f"history <= T: {s['n_test_leaked']} test and {len(tuning_leaked)} tuning authors. Check: "
        "`paperfeed.data.leakage.titles_match` (exact after normalisation, containment, or token Jaccard >= 0.75).",
        f"> - Every comparison below is recomputed from the saved per-author results on the {s['n_test_clean']} test authors "
        "without the leak. Every system was frozen before its test run, so this is a valid paired comparison of the same "
        "systems; nothing was re-tuned or re-run. Families, bootstrap and BH correction are the published ones "
        f"({s['n_resamples']:,} resamples).",
        f"> - The {len(tuning_leaked)} leaked tuning authors could only have influenced which weights were chosen; that "
        "influence is not removed here and is reported as a limitation.",
        f"> - **{n_changed} conclusion(s) change** when the leaked authors are removed.\n",
        f"## 1. H1: Paper Feed vs the five baselines (H1 supported: all {s['h1_supported']['all']}, leak-free {s['h1_supported']['clean']})\n",
        head, sep, *[_row(r) for r in s["h1"]],
        f"\n## 2. H2: MMR diversity with non-inferiority (H2 supported: all {s['h2_supported']['all']}, leak-free {s['h2_supported']['clean']})\n",
        head, sep, *[_row(r) for r in s["h2"]],
        "\n## 3. Phase 10: each variant vs Core (BH over the variant family)\n",
        head, sep, *[_row(r) for r in s["variants"]],
        "\n## 4. Phase 11: Paper Feed vs each comparison system (BH over the comparison family)\n",
        head, sep, *[_row(r) for r in s["sota"]],
        "\n*RankGPT ran with Qwen2.5-1.5B in fp16 on the run machine (design §18a); its saved label says 3B.*",
        "\n## 5. Leaked authors\n",
        "| Author | Split | History paper | Its title | Title of B |", "| :--- | :---: | :--- | :--- | :--- |",
        *[f"| {a} | {v['split']} | {v['history_paper']} | {v['history_title'][:80]} | {v['b_title'][:80]} |"
          for a, v in sorted(leaked_info.items(), key=lambda kv: (kv[1]['split'], kv[0]))],
    ]
    return "\n".join(lines) + "\n"
