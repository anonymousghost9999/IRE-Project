"""V3 learned ranker on toy pools (plan 10.3a)."""

import numpy as np
import pytest

pytest.importorskip("lightgbm")

from paperfeed.rec.features import CandidateFeatures, CitationSignal, FacetMatch, NewMethod
from paperfeed.rec.learned import (
    V3_FEATURES,
    V3_PARAMS,
    TrainingQuery,
    dominant_explainable,
    feature_matrix,
    labels,
    model_scored_pool,
    rank_with_model,
    ranks_outweigh_explanation,
    shap_contributions,
    train_ranker,
)
from paperfeed.rec.scoring import ScoringWeights, score_candidates


def _feat(pid, persona, bc):
    return CandidateFeatures(paper_id=pid, persona_cos=persona, session_cos=None, citation=CitationSignal(bc=bc),
                             facet=FacetMatch(), new_method=NewMethod(), penalty=0.0)


def _query(u, rng, n=30):
    """Targets are the papers with many shared references; persona is pure noise, so only citation separates them."""
    feats, targets, ranks = [], [], {}
    for i in range(n):
        pid = f"u{u}p{i}"
        is_target = i < 3
        feats.append(_feat(pid, persona=float(rng.uniform(0.2, 0.9)), bc=int(rng.integers(8, 12) if is_target else rng.integers(0, 3))))
        ranks[pid] = (i + 1, None if i % 2 else i + 1)
        if is_target:
            targets.append(pid)
    order = rng.permutation(n)
    feats = [feats[j] for j in order]
    return TrainingQuery(score_candidates(feats, ScoringWeights()), ranks, targets)


class _Store:
    def select(self, ids):
        return np.eye(len(ids), 4, dtype=np.float32) + 1e-3  # near-orthogonal: MMR barely matters


def test_feature_matrix_and_labels_follow_the_pool_order():
    rng = np.random.default_rng(0)
    q = _query(0, rng)
    X = feature_matrix(q.scored, q.channel_ranks, channel_depth=200)
    assert X.shape == (30, len(V3_FEATURES))
    for row, s in zip(X, q.scored):
        assert row[0] == s.scaled["persona"] and row[1] == s.scaled["citation"]
        dense, lexical = q.channel_ranks[s.paper_id]
        assert row[4] == dense and row[5] == (lexical if lexical is not None else 201)  # missed channel: depth + 1
    y = labels(q.scored, q.targets)
    assert y.sum() == 3 and {s.paper_id for s, l in zip(q.scored, y) if l} == set(q.targets)


def test_ranker_learns_the_separating_feature_and_is_deterministic():
    rng = np.random.default_rng(1)
    train = [_query(u, rng) for u in range(12)]
    held_out = [_query(100 + u, rng) for u in range(4)]
    m1, m2 = train_ranker(train, channel_depth=200), train_ranker(train, channel_depth=200)
    for q in held_out:
        X = feature_matrix(q.scored, q.channel_ranks, 200)
        np.testing.assert_array_equal(m1.predict(X), m2.predict(X))  # same data, same model
        picks = rank_with_model(m1, q.scored, q.channel_ranks, 200, _Store(), k=3, lambda_div=0.0)
        assert {p.scored.paper_id for p in picks} == set(q.targets)
    imp = dict(zip(V3_FEATURES, m1.feature_importances_))
    assert imp["citation"] > 0 and imp["citation"] >= max(v for k, v in imp.items() if k != "citation")


def test_model_scores_are_pool_scaled_and_shap_adds_up():
    rng = np.random.default_rng(2)
    model = train_ranker([_query(u, rng) for u in range(10)], channel_depth=200)
    q = _query(50, rng)
    pool = model_scored_pool(model, q.scored, q.channel_ranks, 200)
    scores = np.array([s.score for s in pool])
    assert scores.min() == pytest.approx(0.0) and scores.max() == pytest.approx(1.0)
    X = feature_matrix(q.scored, q.channel_ranks, 200)
    full = np.asarray(model.predict(X, pred_contrib=True))
    raw = model.predict(X)
    np.testing.assert_allclose(full.sum(axis=1), raw, rtol=1e-6, atol=1e-6)  # SHAP + bias = model output
    np.testing.assert_allclose(shap_contributions(model, X), full[:, :-1])
    assert [s.paper_id for s in pool] == [s.paper_id for s in q.scored]
    assert set(pool[0].contributions) == set(V3_FEATURES) and "model_score" in pool[0].extra
    assert q.scored[0].contributions.get("dense_rank") is None  # the input pool is not modified


def test_explainable_dominance_and_rank_check():
    c = {"persona": 0.1, "citation": 0.4, "facet": 0.0, "new_method": -0.2, "dense_rank": 0.3, "lexical_rank": 0.0}
    assert dominant_explainable(c) == "citation" and not ranks_outweigh_explanation(c)
    c2 = {**c, "dense_rank": 0.9}
    assert dominant_explainable(c2) == "citation" and ranks_outweigh_explanation(c2)
    assert dominant_explainable({"persona": -0.1, "dense_rank": 0.5}) is None


def test_training_needs_a_non_empty_query():
    with pytest.raises(ValueError, match="non-empty"):
        train_ranker([TrainingQuery([], {}, [])], channel_depth=200)


# ---------------------------------------------------------------------------
# Cross-validation and decision (plan 10.3b-c)
# ---------------------------------------------------------------------------

from types import SimpleNamespace

from paperfeed.eval.learned_cv import author_folds, cross_validate, decide_learned_ranker
from paperfeed.eval.tuning import CachedUserCandidateData


def _cached(u, rng, n=30):
    feats, targets, ranks = [], [], {}
    for i in range(n):
        pid = f"u{u}p{i}"
        is_target = i < 3
        # persona points the wrong way (targets look less similar), citation separates them
        feats.append(_feat(pid, persona=float(rng.uniform(0.2, 0.4) if is_target else rng.uniform(0.5, 0.9)),
                           bc=int(rng.integers(8, 12) if is_target else rng.integers(0, 3))))
        ranks[pid] = (i + 1, i + 1)
        if is_target:
            targets.append(pid)
    profile = SimpleNamespace(authored_papers=[{}] * 5, zero_pub=False)
    return CachedUserCandidateData(f"A{u}", profile, [f.paper_id for f in feats], feats, targets, channel_ranks=ranks)


def test_author_folds_partition_every_author_once():
    folds = author_folds(23, n_folds=5, seed=42)
    assert len(folds) == 5 and sorted(np.concatenate(folds).tolist()) == list(range(23))
    assert [len(f) for f in folds] == [5, 5, 5, 4, 4]
    assert all((a == b).all() for a, b in zip(folds, author_folds(23, 5, 42)))  # seeded
    with pytest.raises(ValueError):
        author_folds(3, n_folds=5)


def test_cross_validation_scores_every_author_out_of_fold_and_decides():
    rng = np.random.default_rng(3)
    cached = [_cached(u, rng) for u in range(20)]
    resources = SimpleNamespace(store=_Store())
    grid = {"w_cite": [0.0, 0.25], "w_facet": [0.0], "w_nov": [0.0], "lambda_div": [0.0]}
    cv = cross_validate(cached, resources, channel_depth=200, base_weights=ScoringWeights(), grid=grid,
                        params={**V3_PARAMS, "min_child_samples": 5})
    assert cv["author_ids"] == [u.author_id for u in cached] and len(cv["folds"]) == 5
    held = sorted(a for f in cv["folds"] for a in f["held_out_authors"])
    assert held == sorted(u.author_id for u in cached)  # every author held out exactly once
    # the linear grid caps w_cite at 0.25 against w_pers = 1, so persona (pointing away from the targets) wins;
    # the model learns citation and should do clearly better out of fold
    assert cv["oof_v3"].mean() > cv["oof_linear"].mean()
    d = decide_learned_ranker(cv, n_resamples=500)
    assert d["adopt_v3"] is True and d["ndcg_gain"]["ci"][0] > 0 and d["n_users"] == 20


def test_v3_not_adopted_when_the_gain_is_not_reliable():
    lin = np.full(30, 0.2)
    v3 = lin.copy()
    v3[0] = 0.9  # one lucky author
    d = decide_learned_ranker({"oof_linear": lin, "oof_v3": v3, "folds": [], "n_folds": 5, "fold_seed": 42}, n_resamples=500)
    assert d["v3_oof_ndcg"] > d["linear_oof_ndcg"] and d["adopt_v3"] is False
