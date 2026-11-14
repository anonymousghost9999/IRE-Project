"""Phase 10 final variant table (plan 10.5)."""

import json

import numpy as np
import pytest

from paperfeed.eval.variant_table import VARIANTS, build_variant_table, check_same_test_set, compare_with_core, format_variant_table

METRICS = {"ndcg_at_10": 0.0, "recall_at_10": 0.0, "hit_rate_at_10": 0.0, "mrr": 0.0, "recall_at_100": 0.0,
           "topic_entropy": 1.0, "ild": 0.1}


def _run(n, ndcg, n_targets=5):
    rows = []
    for i in range(n):
        full = {**METRICS, "ndcg_at_10": ndcg(i), "recall_at_10": ndcg(i) / 2}
        rows.append({"author_id": f"A{i}", "n_targets": n_targets,
                     "system_metrics": {"Paper Feed (Full)": full, "Baseline: citation_only": {**METRICS, "ndcg_at_10": 0.3}}})
    return {"n_authors": n, "per_user_results": rows}


def _write(tmp_path, n=40, ndcgs=None):
    ndcgs = ndcgs or {"core": lambda i: 0.2, "v1": lambda i: 0.3 + 0.01 * (i % 3), "v2": lambda i: 0.2,
                      "v3": lambda i: 0.2 - 0.01 * (i % 2), "v4": lambda i: 0.21}
    res, art = tmp_path / "results", tmp_path / "artifacts"
    res.mkdir(), art.mkdir()
    for v in VARIANTS:
        (res / v.results_file).write_text(json.dumps(_run(n, ndcgs[v.key])))
        if v.decision_file:
            (art / v.decision_file).write_text(json.dumps({"ndcg_gain": {"mean_diff": 0.01, "relative": 0.1, "ci": [-0.01, 0.02]},
                                                           v.adopt_key: False}))
    return res, art


def test_variant_table_compares_every_variant_with_core(tmp_path):
    res, art = _write(tmp_path)
    t = build_variant_table(res, art, n_resamples=300)
    assert t["n_authors"] == 40 and len(t["comparisons"]) == 8  # 4 variants x 2 metrics, one BH family
    by = {(r["variant"], r["metric"]): r for r in t["comparisons"]}
    assert by[("v1", "ndcg_at_10")]["significant"] and by[("v1", "ndcg_at_10")]["diff"] > 0
    assert by[("v2", "ndcg_at_10")]["diff"] == 0.0 and not by[("v2", "ndcg_at_10")]["significant"]
    assert all(r["p_bh"] >= r["p_value"] for r in t["comparisons"])
    assert t["citation_only"]["ndcg_at_10"] == pytest.approx(0.3)
    md = format_variant_table(t)
    assert "| V1 encoder (BGE) |" in md and "not adopted" in md and "0.1000*" in md  # V1 ILD flagged
    assert "never used to choose" in md


def test_paired_comparison_needs_the_same_test_set():
    a = {"A1": {"n_targets": 3}, "A2": {"n_targets": 4}}
    assert check_same_test_set({"core": a, "v1": dict(a)}) == ["A1", "A2"]
    with pytest.raises(ValueError, match="different test authors"):
        check_same_test_set({"core": a, "v1": {"A1": {"n_targets": 3}}})
    with pytest.raises(ValueError, match="different targets"):
        check_same_test_set({"core": a, "v1": {"A1": {"n_targets": 3}, "A2": {"n_targets": 9}}})
