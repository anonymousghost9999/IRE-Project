"""Unit tests for Phase 6 evaluation metrics (tests/test_eval.py)."""

import math
import numpy as np
import pytest

from paperfeed.eval.metrics import (
    aggregate_metrics,
    facet_novelty,
    hit_rate_at_k,
    intra_list_distance,
    mrr,
    ndcg_at_k,
    oracle_at_k,
    per_user_metrics,
    recall_at_100,
    recall_at_k,
    topic_entropy,
    topic_entropy_normalised,
)


def test_ndcg_at_k_perfect():
    rec = ["p1", "p2", "p3", "p4", "p5"]
    gt = ["p1", "p2", "p3"]
    # At k=5, first 3 items are hits. Perfect ranking -> 1.0
    assert pytest.approx(ndcg_at_k(rec, gt, k=5), 1e-6) == 1.0


def test_ndcg_at_k_zero_hits():
    rec = ["p1", "p2", "p3"]
    gt = ["x1", "x2"]
    assert ndcg_at_k(rec, gt, k=10) == 0.0


def test_ndcg_at_k_empty_gt():
    rec = ["p1", "p2", "p3"]
    assert ndcg_at_k(rec, [], k=10) == 0.0


def test_ndcg_at_k_hand_computed():
    rec = ["miss", "hit"]
    gt = ["hit"]
    # rank 0: miss -> 0
    # rank 1: hit -> 1 / log2(1 + 2) = 1 / log2(3) = 0.63092975
    # ideal: rank 0 hit -> 1 / log2(0 + 2) = 1 / 1 = 1.0
    expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert pytest.approx(ndcg_at_k(rec, gt, k=2), 1e-6) == expected


def test_recall_and_hit_rate_at_k():
    rec = ["p1", "p2", "p3", "p4"]
    gt = ["p2", "p4", "p5", "p6"]  # 4 targets, 2 hits in top 4
    assert pytest.approx(recall_at_k(rec, gt, k=4), 1e-6) == 2.0 / 4.0
    assert hit_rate_at_k(rec, gt, k=4) == 1.0
    assert hit_rate_at_k(["other"], gt, k=4) == 0.0
    assert recall_at_k(["other"], gt, k=4) == 0.0
    assert recall_at_k(rec, [], k=4) == 0.0


def test_mrr():
    assert pytest.approx(mrr(["miss", "miss", "hit"], ["hit"]), 1e-6) == 1.0 / 3.0
    assert pytest.approx(mrr(["hit", "other"], ["hit"]), 1e-6) == 1.0
    assert mrr(["a", "b"], ["c"]) == 0.0
    assert mrr([], ["c"]) == 0.0


def test_oracle_at_k():
    assert oracle_at_k(["p1", "p2"], k=10) == 1.0
    assert oracle_at_k([], k=10) == 0.0


def test_recall_at_100():
    cands = [f"c{i}" for i in range(100)]
    gt = ["c5", "c99", "c999"]  # 2 in cands, 1 out of pool
    assert pytest.approx(recall_at_100(cands, gt), 1e-6) == 2.0 / 3.0


def test_topic_entropy():
    # All same topic -> entropy 0
    topic_map = {"p1": "T1", "p2": "T1", "p3": "T1"}
    assert topic_entropy(["p1", "p2", "p3"], topic_map, k=3) == 0.0
    assert topic_entropy_normalised(["p1", "p2", "p3"], topic_map, k=3) == 0.0

    # 4 distinct topics -> entropy = log2(4) = 2.0
    topic_map_distinct = {"p1": "T1", "p2": "T2", "p3": "T3", "p4": "T4"}
    assert pytest.approx(topic_entropy(["p1", "p2", "p3", "p4"], topic_map_distinct, k=4), 1e-6) == 2.0
    assert pytest.approx(topic_entropy_normalised(["p1", "p2", "p3", "p4"], topic_map_distinct, k=4), 1e-6) == 1.0

    # Missing topic gets UNKNOWN fallback
    assert pytest.approx(topic_entropy(["p1", "p2"], {"p1": "T1"}, k=2), 1e-6) == 1.0


def test_intra_list_distance():
    # 2 orthogonal vectors -> cosine similarity = 0 -> distance = 1.0
    mat = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    id2idx = {"p1": 0, "p2": 1}
    assert pytest.approx(intra_list_distance(["p1", "p2"], mat, id2idx, k=2), 1e-6) == 1.0

    # 2 identical vectors -> cosine similarity = 1 -> distance = 0.0
    mat2 = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    assert pytest.approx(intra_list_distance(["p1", "p2"], mat2, id2idx, k=2), 1e-6) == 0.0

    # Less than 2 items -> 0.0
    assert intra_list_distance(["p1"], mat, id2idx, k=2) == 0.0


def test_facet_novelty():
    user_facets = {"f1", "f2"}
    paper_facet_map = {
        "p1": {"f1", "f3"},
        "p2": {"f4"},
    }
    # Unique facets in top-k: {f1, f3, f4} (3 facets). Novel: {f3, f4} (2 facets). Novelty = 2/3.
    assert pytest.approx(facet_novelty(["p1", "p2"], user_facets, paper_facet_map, k=2), 1e-6) == 2.0 / 3.0

    # No facets in papers -> 0.0
    assert facet_novelty(["p1"], user_facets, {}, k=1) == 0.0


def test_aggregate_and_per_user_metrics():
    row1 = {"ndcg_at_10": 0.5, "recall_at_10": 0.2}
    row2 = {"ndcg_at_10": 0.3, "recall_at_10": 0.4}
    agg = aggregate_metrics([row1, row2])
    assert pytest.approx(agg["ndcg_at_10"], 1e-6) == 0.4
    assert pytest.approx(agg["recall_at_10"], 1e-6) == 0.3

    res = per_user_metrics(
        recommended_ids=["p1", "p2"],
        ground_truth_ids=["p2"],
        candidate_ids=["p1", "p2", "p3"],
        topic_map={"p1": "T1", "p2": "T2"},
    )
    assert res["hit_rate_at_10"] == 1.0
    assert res["mrr"] == 0.5
    assert res["n_targets"] == 1.0


from unittest.mock import MagicMock
from paperfeed.eval.baselines import (
    BASELINE_NAMES,
    _eligible_not_excluded,
    random_anchor_ids,
    random_baseline,
    run_baseline,
)


def test_eligible_not_excluded():
    eligible = {"p1", "p2", "p3", "p4"}
    excluded = {"p2", "p5"}
    assert _eligible_not_excluded(eligible, excluded) == ["p1", "p3", "p4"]


def test_random_baseline_and_anchor():
    resources = MagicMock()
    resources.eligible = {f"p{i}" for i in range(20)}
    profile = MagicMock()
    profile.zero_pub = False
    profile.exclusion_sources.build.return_value = {"p0", "p1"}

    picks = random_baseline(resources, profile, k=5)
    assert len(picks) == 5
    assert len(set(picks)) == 5
    assert "p0" not in picks and "p1" not in picks

    draws = random_anchor_ids(resources, profile, k=5, n_draws=10)
    assert len(draws) == 10
    for d in draws:
        assert len(d) == 5
        assert "p0" not in d and "p1" not in d


def test_run_baseline_dispatch():
    resources = MagicMock()
    resources.eligible = {"p1", "p2", "p3"}
    profile = MagicMock()
    profile.zero_pub = False
    profile.exclusion_sources.build.return_value = set()

    picks = run_baseline("random", resources, profile, k=2)
    assert len(picks) == 2

    with pytest.raises(ValueError, match="Unknown baseline name"):
        run_baseline("unknown_model", resources, profile)
