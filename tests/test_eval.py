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


from paperfeed.eval.tuning import (
    CachedUserCandidateData,
    TuningResult,
    evaluate_parameter_grid,
)
from paperfeed.rec.features import CandidateFeatures


def test_evaluate_parameter_grid():
    store = MagicMock()
    store.get.side_effect = lambda pid: np.array([1.0, 0.0] if pid == "p1" else [0.0, 1.0], dtype=np.float32)
    store.select.side_effect = lambda pids: np.stack([store.get(p) for p in pids])
    resources = MagicMock()
    resources.store = store

    from paperfeed.rec.features import CitationSignal, FacetMatch, NewMethod
    f1 = CandidateFeatures(
        paper_id="p1",
        persona_cos=0.8,
        session_cos=None,
        citation=CitationSignal(bc=2, cc=1),
        facet=FacetMatch(s_task=0.3, s_dataset=0.2),
        new_method=NewMethod(flag=True),
        penalty=0.0,
    )
    f2 = CandidateFeatures(
        paper_id="p2",
        persona_cos=0.4,
        session_cos=None,
        citation=CitationSignal(bc=0, cc=0),
        facet=FacetMatch(s_task=0.1, s_dataset=0.0),
        new_method=NewMethod(flag=False),
        penalty=0.0,
    )

    user = CachedUserCandidateData(
        author_id="A1",
        profile=MagicMock(),
        candidate_ids=["p1", "p2"],
        features=[f1, f2],
        targets=["p1"],
    )

    grid = {
        "w_cite": [0.0, 1.0],
        "w_facet": [0.0],
        "w_nov": [0.0],
        "lambda_div": [0.0, 0.2],
    }

    res = evaluate_parameter_grid([user], resources, grid=grid, k=2)
    assert isinstance(res, TuningResult)
    assert res.grid_size == 4
    assert res.n_users == 1
    assert res.best_mean_ndcg > 0.0
    assert len(res.top_configurations) <= 4

    d = res.to_dict()
    assert "best_weights" in d
    assert "best_lambda_div" in d


from paperfeed.eval.lfo import UserEvaluationRow
from paperfeed.eval.report import (
    format_ablation_table_markdown,
    format_main_table_markdown,
    generate_full_evaluation_report,
)
from paperfeed.eval.slices import slice_by_history_volume
from paperfeed.eval.stats import (
    HypothesisTestReport,
    benjamini_hochberg_correction,
    paired_bootstrap_test,
    paired_cohens_d,
    run_track1_hypothesis_tests,
)


def test_paired_bootstrap_and_cohens_d():
    sys_scores = [0.5, 0.65, 0.7, 0.85, 0.9]
    base_scores = [0.2, 0.3, 0.4, 0.5, 0.6]

    d = paired_cohens_d(sys_scores, base_scores)
    assert d > 0.0

    mean_diff, lo, hi, p, effect_d = paired_bootstrap_test(
        sys_scores, base_scores, n_resamples=500, seed=42
    )
    assert pytest.approx(mean_diff, 1e-6) == 0.32
    assert lo > 0.0  # complete separation
    assert p < 0.05


def test_benjamini_hochberg_correction():
    p_vals = [0.01, 0.04, 0.03, 0.20]
    bh = benjamini_hochberg_correction(p_vals, fdr=0.05)
    assert len(bh) == 4
    # All adjusted values should be <= 1.0 and non-negative
    for adj_p, is_sig in bh:
        assert 0.0 <= adj_p <= 1.0


def test_run_track1_hypothesis_tests():
    data = {
        "Paper Feed (Full)": {
            "ndcg_at_10": [0.5, 0.6, 0.7, 0.8],
            "topic_entropy": [1.5, 1.6, 1.7, 1.8],
        },
        "Ablation: No-MMR": {
            "ndcg_at_10": [0.51, 0.61, 0.71, 0.81],
            "topic_entropy": [1.0, 1.1, 1.2, 1.3],
        },
        "Baseline: popularity": {"ndcg_at_10": [0.2, 0.2, 0.2, 0.2]},
        "Baseline: citation_only": {"ndcg_at_10": [0.1, 0.1, 0.1, 0.1]},
        "Baseline: bm25_author": {"ndcg_at_10": [0.25, 0.25, 0.25, 0.25]},
        "Baseline: dense_centroid": {"ndcg_at_10": [0.3, 0.3, 0.3, 0.3]},
        "Baseline: hybrid_persona": {"ndcg_at_10": [0.35, 0.35, 0.35, 0.35]},
    }
    for name, series in data.items():  # Recall@10 mirrors NDCG@10 here
        series["recall_at_10"] = list(series["ndcg_at_10"])
    rep = run_track1_hypothesis_tests(data, n_resamples=500)
    assert isinstance(rep, HypothesisTestReport)
    assert len(rep.h1_comparisons) == 10  # 5 baselines x {NDCG@10, Recall@10}, one BH family
    assert {c.metric_name for c in rep.h1_comparisons} == {"ndcg_at_10", "recall_at_10"}
    assert all(c.mde >= 0 for c in rep.h1_comparisons)
    assert rep.h1_supported is True
    assert rep.h2_supported is True


def test_slice_by_history_volume():
    r1 = UserEvaluationRow(
        author_id="A1",
        split="test",
        history_slice="warm",
        n_authored=10,
        n_targets=5,
        system_metrics={"PF": {"ndcg_at_10": 0.5}},
    )
    r2 = UserEvaluationRow(
        author_id="A2",
        split="test",
        history_slice="sparse",
        n_authored=1,
        n_targets=2,
        system_metrics={"PF": {"ndcg_at_10": 0.3}},
    )
    slices = slice_by_history_volume([r1, r2])
    assert "warm" in slices
    assert "sparse" in slices
    assert slices["warm"]["PF"]["ndcg_at_10"] == 0.5
    assert slices["sparse"]["PF"]["ndcg_at_10"] == 0.3


def test_reporting_formatting():
    aggs = {
        "Paper Feed (Full)": {"ndcg_at_10": 0.45, "recall_at_10": 0.25, "topic_entropy": 1.5, "ild": 0.3},
        "Ablation: No-MMR": {"ndcg_at_10": 0.46, "recall_at_10": 0.25, "topic_entropy": 1.2},
    }
    main_tbl = format_main_table_markdown(aggs)
    assert "| **Paper Feed (Full)** |" in main_tbl
    assert "0.4500" in main_tbl

    abl_tbl = format_ablation_table_markdown(aggs)
    assert "No-MMR" in abl_tbl


def _h2_data(no_mmr_ndcg, no_mmr_entropy):
    data = {
        "Paper Feed (Full)": {"ndcg_at_10": [0.5, 0.6, 0.7, 0.8], "topic_entropy": [1.5, 1.6, 1.7, 1.8]},
        "Ablation: No-MMR": {"ndcg_at_10": no_mmr_ndcg, "topic_entropy": no_mmr_entropy},
    }
    for b in ("popularity", "citation_only", "bm25_author", "dense_centroid", "hybrid_persona"):
        data[f"Baseline: {b}"] = {"ndcg_at_10": [0.1] * 4}
    for series in data.values():
        series["recall_at_10"] = list(series["ndcg_at_10"])
    return data


def test_h2_is_not_testable_when_mmr_is_off():
    """Tuned lambda_div = 0 makes No-MMR identical to the full system: H2 is untestable, not rejected."""
    from paperfeed.eval.report import format_hypothesis_markdown

    rep = run_track1_hypothesis_tests(_h2_data([0.5, 0.6, 0.7, 0.8], [1.5, 1.6, 1.7, 1.8]), n_resamples=200)
    assert rep.h2_testable is False and rep.h2_supported is False
    md = format_hypothesis_markdown(rep)
    assert "NOT TESTABLE" in md and "REJECTED" not in md.split("Hypothesis H2")[1]


def test_h2_report_shows_the_sign_of_the_ndcg_change():
    """MMR that raises NDCG must not be printed as a loss (the old report showed abs())."""
    from paperfeed.eval.report import format_hypothesis_markdown

    rep = run_track1_hypothesis_tests(_h2_data([0.49, 0.59, 0.69, 0.79], [1.0, 1.1, 1.2, 1.3]), n_resamples=200)
    assert rep.h2_testable is True
    h2 = format_hypothesis_markdown(rep).split("Hypothesis H2")[1]
    assert "NDCG@10 change with MMR = +" in h2
    assert "worst-case relative loss from the CI = 0.00%" in h2


def test_report_keeps_diagnostic_out_of_main_table_and_shows_coverage_ceiling():
    from paperfeed.eval.report import generate_full_evaluation_report

    aggs = {
        "Paper Feed (Full)": {"ndcg_at_10": 0.02, "recall_at_10": 0.015, "recall_at_100": 0.05},
        "Diagnostic: Re-Citation": {"ndcg_at_10": 0.04, "recall_at_10": 0.04, "recall_at_100": 0.08},
        "Baseline: citation_only": {"ndcg_at_10": 0.03, "recall_at_10": 0.027},
    }
    rep = run_track1_hypothesis_tests(_h2_data([0.49, 0.59, 0.69, 0.79], [1.0, 1.1, 1.2, 1.3]), n_resamples=100)
    md = generate_full_evaluation_report(aggs, rep)
    main = md.split("## 1. Main Benchmark Results")[1].split("###")[0]
    assert "Re-Citation" not in main
    assert "### Diagnostic: Re-Citation" in md
    assert "| Paper Feed (Full) | 0.0500 | 0.0150 | 30.0% |" in md


# ---------------------------------------------------------------------------
# Baseline set, no silent fallbacks, Citation-only, tuning with the configured weights
# ---------------------------------------------------------------------------

from types import SimpleNamespace

from paperfeed.data.db import DatabaseManager
from paperfeed.eval.baselines import (
    bm25_author_baseline,
    citation_only_baseline,
    popularity_baseline,
)
from paperfeed.rec.features import build_citation_context
from paperfeed.rec.scoring import ScoringWeights
from paperfeed.user.exclusion import build_exclusion_sources


def _uri(pid):
    return f"https://openalex.org/{pid}"


def _paper(pid, year=2022, subfield="https://openalex.org/subfields/1702", cites=0):
    return {
        "paper_id": _uri(pid), "title": f"Title {pid}", "abstract": "x", "authors": "[]", "year": year,
        "publication_date": None, "venue": None, "doi": None, "url": None, "citation_count_T": cites,
        "primary_topic_id": None, "subfield_id": subfield, "field_id": None, "in_candidates": 1,
    }


def test_baseline_registry_matches_the_plan_and_h1_family():
    from paperfeed.eval.stats import H1_BASELINES

    assert BASELINE_NAMES == ["popularity", "bm25_author", "dense_centroid", "citation_only", "hybrid_persona"]
    assert tuple(BASELINE_NAMES) == H1_BASELINES  # Random is an anchor, never an H1 baseline


def test_h1_refuses_to_run_with_a_missing_baseline():
    data = {
        "Paper Feed (Full)": {"ndcg_at_10": [0.5, 0.6], "topic_entropy": [1.0, 1.0]},
        "Ablation: No-MMR": {"ndcg_at_10": [0.5, 0.6], "topic_entropy": [1.0, 1.0]},
        "Baseline: popularity": {"ndcg_at_10": [0.2, 0.2]},
    }
    with pytest.raises(ValueError, match="missing"):
        run_track1_hypothesis_tests(data, n_resamples=50)


@pytest.fixture
def pop_world(tmp_path):
    """Two subfields; the user's papers are in 1702. P1-P3 are popular in 1702, Q1 is more popular but in 9999."""
    db = DatabaseManager(db_path=tmp_path / "pop.sqlite3", cutoff_year=2024)
    db.insert_papers_batch([
        _paper("U1"), _paper("U2"),
        _paper("P1", cites=500), _paper("P2", cites=300), _paper("P3", cites=100),
        _paper("OLD", year=2005, cites=900),  # before the counts_by_year window: never ranked
        _paper("Q1", subfield="https://openalex.org/subfields/9999", cites=1000),
    ])
    eligible = {"U1", "U2", "P1", "P2", "P3", "OLD", "Q1"}
    resources = SimpleNamespace(db=db, eligible=eligible, cutoff_year=2024)
    profile = SimpleNamespace(
        authored_papers=[{"paper_id": "U1", "title": "Title U1", "abstract": "x"}, {"paper_id": "U2", "title": "Title U2", "abstract": "x"}],
        zero_pub=False,
        exclusion_sources=build_exclusion_sources(["U1", "U2"], ["P2"]),
    )
    return resources, profile


def test_popularity_baseline_is_domain_conditioned(pop_world):
    resources, profile = pop_world
    assert popularity_baseline(resources, profile, k=10) == ["P1", "P3"]  # P2 cited by the user, Q1 other subfield


def test_popularity_baseline_never_falls_back_to_random(pop_world):
    resources, profile = pop_world
    resources.eligible = {"U1", "U2"}  # nothing recommendable
    assert popularity_baseline(resources, profile, k=10) == []  # an honest empty list, not a random sample

    class BrokenDB:
        def get_connection(self):
            raise RuntimeError("database is locked")

    resources.db = BrokenDB()
    with pytest.raises(RuntimeError, match="locked"):
        popularity_baseline(resources, profile, k=10)


def test_bm25_baseline_uses_the_full_author_text():
    seen = {}

    class Lexical:
        def search(self, query, k, exclusion_set=None, candidate_filter=None):
            seen["query"] = query
            return [("P9", 1.0)]

    long_abstract = " ".join(f"word{i}" for i in range(1000))
    resources = SimpleNamespace(retriever=SimpleNamespace(lexical_index=Lexical()), eligible={"P9"})
    profile = SimpleNamespace(
        authored_papers=[{"paper_id": "U1", "title": "First", "abstract": long_abstract}, {"paper_id": "U2", "title": "Last", "abstract": ""}],
        zero_pub=False,
        exclusion_sources=build_exclusion_sources(["U1", "U2"], []),
    )
    assert bm25_author_baseline(resources, profile, k=5) == ["P9"]
    assert "word999" in seen["query"] and seen["query"].endswith("Last")  # nothing cut off
    profile.authored_papers = [{"paper_id": "U1", "title": "", "abstract": ""}]
    assert bm25_author_baseline(resources, profile, k=5) == []  # no text: empty, not random


@pytest.fixture
def cite_world(tmp_path):
    """Authored A1 refs {R1, R2}; A2 refs {R2, R3}.
    P refs {R2, R3}: BC 2.  Q refs {R1}: BC 1, and the third party C1 cites A1 and Q: CC 1.  S refs {R9}: 0.
    The user already cites R1-R3 (excluded); X is from after T and cites R1-R3 (its edges are never counted)."""
    db = DatabaseManager(db_path=tmp_path / "cite.sqlite3", cutoff_year=2024)
    ids = ["A1", "A2", "P", "Q", "S", "C1", "R1", "R2", "R3", "R9"]
    db.insert_papers_batch([_paper(i) for i in ids] + [_paper("X", year=2025)])
    edges = [("A1", "R1"), ("A1", "R2"), ("A2", "R2"), ("A2", "R3"), ("P", "R2"), ("P", "R3"), ("Q", "R1"),
             ("S", "R9"), ("C1", "A1"), ("C1", "Q")]
    db.insert_citations_batch([(_uri(a), _uri(b), 2022) for a, b in edges])
    with db.get_connection() as conn:
        conn.executemany(
            "INSERT INTO paper_citations (source_paper_id, target_paper_id, source_year) VALUES (?, ?, 2025)",
            [(_uri("X"), _uri(r)) for r in ("R1", "R2", "R3")],
        )
    eligible = set(ids) | {"X"}
    resources = SimpleNamespace(db=db, eligible=eligible, cutoff_year=2024)
    ctx = build_citation_context(db, ["A1", "A2"], 2024)
    profile = SimpleNamespace(
        citation_ctx=ctx, zero_pub=False,
        exclusion_sources=build_exclusion_sources(["A1", "A2"], sorted(ctx.ref_to_authored)),
    )
    return resources, profile


def test_citation_only_baseline_ranks_by_bc_plus_cc(cite_world):
    resources, profile = cite_world
    # P: BC 2. Q: BC 1 + CC 1 = 2 (tie with P, broken by id). C1 cites A1 but shares no reference: 0. X: post-T.
    assert citation_only_baseline(resources, profile, k=10) == ["P", "Q"]
    assert citation_only_baseline(resources, profile, k=1) == ["P"]


def test_citation_only_baseline_is_not_applicable_to_zero_publication_users(cite_world):
    resources, profile = cite_world
    profile.citation_ctx = build_citation_context(resources.db, [], 2024)
    assert citation_only_baseline(resources, profile, k=10) == []


def test_tuning_grid_scores_with_the_configured_weights():
    seen = []

    def fake_score(features, weights):
        seen.append(weights)
        from paperfeed.rec.scoring import score_candidates as real
        return real(features, weights)

    from paperfeed.rec.features import CitationSignal, FacetMatch, NewMethod
    f = CandidateFeatures(paper_id="p1", persona_cos=0.8, session_cos=None, citation=CitationSignal(),
                          facet=FacetMatch(), new_method=NewMethod(), penalty=0.0)
    user = CachedUserCandidateData(author_id="A1", profile=MagicMock(), candidate_ids=["p1"], features=[f], targets=["p1"])
    store = MagicMock()
    store.select.side_effect = lambda pids: np.ones((len(pids), 2), dtype=np.float32)
    resources = SimpleNamespace(store=store)
    import paperfeed.eval.tuning as tuning
    orig = tuning.score_candidates
    tuning.score_candidates = fake_score
    try:
        res = evaluate_parameter_grid(
            [user], resources, grid={"w_cite": [0.5], "w_facet": [0.0], "w_nov": [0.0], "lambda_div": [0.0]},
            k=1, adaptive_graph=False, base_weights=ScoringWeights(w_pop=0.2, lambda_neg=0.7),
        )
    finally:
        tuning.score_candidates = orig
    assert seen and all(w.w_pop == 0.2 and w.lambda_neg == 0.7 for w in seen)  # not silently reset to 0
    assert res.best_weights.w_pop == 0.2


def test_hybrid_persona_rebuilds_the_persona_with_its_own_gamma():
    from paperfeed.eval.baselines import _persona_vector_for_gamma
    from paperfeed.index.store import EmbeddingStore
    from paperfeed.user.persona import Persona

    store = EmbeddingStore(["U1", "U2"], np.array([[1, 0], [0, 1]], dtype=np.float32))
    papers = [{"paper_id": "U1", "title": "a", "abstract": "x", "year": 2024}, {"paper_id": "U2", "title": "b", "abstract": "x", "year": 2014}]
    own = Persona(vector=np.array([1.0, 0.0], dtype=np.float32), source="authored", paper_ids=["U1", "U2"])
    profile = SimpleNamespace(persona=own, authored_papers=papers, zero_pub=False)
    resources = SimpleNamespace(store=store, encoder=None, cutoff_year=2024)

    np.testing.assert_array_equal(_persona_vector_for_gamma(resources, profile, None), own.vector)
    np.testing.assert_allclose(_persona_vector_for_gamma(resources, profile, 1.0), [2 ** -0.5, 2 ** -0.5], atol=1e-6)
    profile.zero_pub = True  # a seed-topic persona has no papers to re-weight
    np.testing.assert_array_equal(_persona_vector_for_gamma(resources, profile, 1.0), own.vector)


def test_tune_hybrid_persona_picks_the_best_grid_point(monkeypatch):
    import paperfeed.eval.baselines as baselines
    from paperfeed.eval.tuning import tune_hybrid_persona

    def fake_hybrid(resources, profile, k, n_terms, n_titles, gamma_rec):
        return ["hit"] if (gamma_rec, n_terms) == (0.8, 20) else ["miss"]

    monkeypatch.setattr(baselines, "hybrid_persona_baseline", fake_hybrid)
    users = [CachedUserCandidateData(author_id=f"A{i}", profile=None, candidate_ids=[], features=[], targets=["hit"]) for i in range(3)]
    out = tune_hybrid_persona(users, resources=None, grid={"gamma_rec": [0.7, 0.8], "n_terms": [10, 20]}, n_titles=3)
    assert (out["gamma_rec"], out["n_terms"], out["mean_ndcg"]) == (0.8, 20, 1.0)
    assert out["grid_size"] == 4 and out["n_users"] == 3 and out["n_titles"] == 3



def test_h1_is_supported_only_if_both_metrics_win():
    n = 30
    data = {
        "Paper Feed (Full)": {"ndcg_at_10": [0.6] * n, "recall_at_10": [0.1] * n, "topic_entropy": [1.0] * n},
        "Ablation: No-MMR": {"ndcg_at_10": [0.6] * n, "recall_at_10": [0.1] * n, "topic_entropy": [1.0] * n},
    }
    for b in ("popularity", "bm25_author", "dense_centroid", "citation_only", "hybrid_persona"):
        data[f"Baseline: {b}"] = {"ndcg_at_10": [0.2] * n, "recall_at_10": [0.3] * n}  # wins NDCG, loses Recall
    rep = run_track1_hypothesis_tests(data, n_resamples=200)
    assert not rep.h1_supported


def test_bootstrap_mde_matches_the_normal_approximation_and_shrinks_with_n():
    from paperfeed.eval.stats import bootstrap_mde

    rng = np.random.default_rng(0)
    small = rng.normal(0.0, 1.0, size=50)
    large = rng.normal(0.0, 1.0, size=800)
    mde_small = bootstrap_mde(small, np.zeros_like(small), n_resamples=4000)
    mde_large = bootstrap_mde(large, np.zeros_like(large), n_resamples=4000)
    analytic = 2.8016 * np.std(small, ddof=1) / np.sqrt(len(small))  # (1.96 + 0.84) * SE
    assert mde_small == pytest.approx(analytic, rel=0.1)
    assert mde_large < mde_small / 3  # ~ 1/sqrt(16) = 1/4 of the small-N MDE
    assert bootstrap_mde([], []) == 0.0


def test_h2_refuses_to_run_without_the_no_mmr_ablation():
    n = 5
    data = {"Paper Feed (Full)": {"ndcg_at_10": [0.5] * n, "recall_at_10": [0.1] * n, "topic_entropy": [1.0] * n}}
    for b in ("popularity", "bm25_author", "dense_centroid", "citation_only", "hybrid_persona"):
        data[f"Baseline: {b}"] = {"ndcg_at_10": [0.2] * n, "recall_at_10": [0.05] * n}
    with pytest.raises(ValueError, match="No-MMR"):
        run_track1_hypothesis_tests(data, n_resamples=50)


# ---------------------------------------------------------------------------
# Item cold-start slice, zero-publication baselines, report sections
# ---------------------------------------------------------------------------

from paperfeed.eval.report import format_cold_start_table, format_comparisons_markdown, format_zero_pub_section
from paperfeed.eval.slices import classify_cold_start_targets, item_cold_start_breakdown


def test_classify_cold_start_targets_follows_design_section_12():
    counts = {"new": 0, "new1": 1, "old_uncited": 0, "busy": 50, "old_busy": 50, "middle": 5, "no_year": 0}
    years = {"new": 2024, "new1": 2023, "old_uncited": 2005, "busy": 2015, "old_busy": 2008, "middle": 2023}
    classes = classify_cold_start_targets(counts, years, cutoff_year=2024)
    assert classes == {"new": "fresh", "new1": "fresh", "busy": "established"}
    # an old paper with no counted citations is NOT fresh (its citations predate the 2012 counts_by_year window),
    # an established paper must be from 2012 on, and unknown years are left out


def test_item_cold_start_breakdown_hand_computed():
    rows = [
        UserEvaluationRow("A1", "test", "warm", 5, 3, {}, {"PF": ["f1", "x"], "BL": ["e1"], "Diagnostic: Re-Citation": ["f1"]}),
        UserEvaluationRow("A2", "test", "warm", 5, 1, {}, {"PF": ["e2"], "BL": ["y"]}),
    ]
    targets = {"A1": ["f1", "f2", "e1"], "A2": ["e2"]}
    classes = {"f1": "fresh", "f2": "fresh", "e1": "established", "e2": "established"}
    out = item_cold_start_breakdown(rows, targets, classes, k=10)
    assert out["fresh"]["PF"] == {"recall_at_10": 0.5, "n_authors": 1.0, "n_targets": 2.0}  # A2 has no fresh target
    assert out["fresh"]["BL"]["recall_at_10"] == 0.0
    assert out["established"]["PF"]["recall_at_10"] == pytest.approx(0.5)  # A1: 0/1, A2: 1/1
    assert out["established"]["BL"]["recall_at_10"] == pytest.approx(0.5)  # A1: 1/1, A2: 0/1
    assert out["established"]["PF"]["n_authors"] == 2.0
    assert "Diagnostic: Re-Citation" not in out["fresh"]  # its targets are re-citations, not Discovery
    table = format_cold_start_table(out)
    assert "fresh (n = 1 authors)" in table and "established (n = 2 authors)" in table


def test_zero_pub_bm25_queries_with_the_seed_topics():
    seen = {}

    class Lexical:
        def search(self, query, k, exclusion_set=None, candidate_filter=None):
            seen["query"] = query
            return []

    resources = SimpleNamespace(retriever=SimpleNamespace(lexical_index=Lexical()), eligible=set())
    profile = SimpleNamespace(authored_papers=[], zero_pub=True, seed_topics=["Dense Retrieval", "Recommender Systems"],
                              exclusion_sources=build_exclusion_sources([], []))
    bm25_author_baseline(resources, profile, k=5)
    assert seen["query"] == "Dense Retrieval Recommender Systems"


def test_popularity_subfield_override_for_the_zero_pub_slice(pop_world):
    resources, profile = pop_world
    assert popularity_baseline(resources, profile, k=10, subfield="https://openalex.org/subfields/9999") == ["Q1"]
    assert run_baseline("popularity", resources, profile, k=10, baseline_kwargs={"subfield": "https://openalex.org/subfields/9999"}) == ["Q1"]


def test_report_sections_render():
    from paperfeed.eval.stats import BootstrapComparison

    comp = BootstrapComparison("Paper Feed (Full)", "Baseline: popularity", "recall_at_10", 0.2, 0.1, 0.1, 100.0,
                               0.05, 0.15, 0.001, 0.002, 0.5, True, mde=0.04)
    table = format_comparisons_markdown([comp])
    assert "| Baseline: popularity | recall_at_10 |" in table and "| 0.0400 |" in table
    section = format_zero_pub_section({"Paper Feed (Full)": {"ndcg_at_10": 0.1}}, [comp], n_authors=320)
    assert "n = 320" in section and "Citation-only is not applicable" in section


# ---------------------------------------------------------------------------
# Facet-active variant selection (design §9)
# ---------------------------------------------------------------------------

import json as _json

from paperfeed.eval.tuning import (
    GridPoint,
    best_grid_point,
    default_facet_selection_path,
    explanation_template_shares,
    load_facet_selection,
    relative_loss_ci,
    select_facet_active_config,
    tuning_signature,
)
from paperfeed.rec.recommender import RecommendConfig


def _gp(w_facet, ndcgs, w_cite=1.0, lam=0.0):
    params = {"w_cite": w_cite, "w_facet": w_facet, "w_nov": 0.0, "w_pop": 0.0, "lambda_div": lam}
    return GridPoint(params, ScoringWeights(w_cite=w_cite, w_facet=w_facet, w_nov=0.0), lam, np.asarray(ndcgs, dtype=float))


def test_relative_loss_ci():
    best = np.array([0.2, 0.4, 0.6, 0.8])
    assert relative_loss_ci(best, best, n_resamples=500) == (0.0, 0.0, 0.0)
    point, lo, hi = relative_loss_ci(best, best / 2, n_resamples=500)
    assert point == pytest.approx(0.5) and lo == pytest.approx(0.5) and hi == pytest.approx(0.5)  # same ratio per user
    point, lo, hi = relative_loss_ci(np.array([0.1, 0.5, 0.3, 0.9, 0.2] * 10), np.array([0.2, 0.3, 0.3, 0.7, 0.2] * 10), n_resamples=2000)
    assert lo < point < hi
    with pytest.raises(ValueError):
        relative_loss_ci(np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError):
        relative_loss_ci(best, best[:2])


def test_best_grid_point_breaks_ties_by_grid_order():
    first, second = _gp(0.0, [0.5, 0.5]), _gp(0.5, [0.5, 0.5])
    assert best_grid_point([first, second]) is first
    assert best_grid_point([]) is None


def test_facet_active_rule_needs_both_non_inferiority_and_facet_explanations(monkeypatch):
    import paperfeed.eval.tuning as tuning

    rng = np.random.default_rng(0)
    base = rng.uniform(0.2, 0.8, size=80)
    points = [
        _gp(0.0, base),                      # grid optimum, no facets
        _gp(0.5, base * 0.995, lam=0.1),     # non-inferior but its explanations are not facet ones
        _gp(1.0, base * 0.80, lam=0.2),      # many facet explanations but 20% worse
        _gp(0.25, base * 0.99, lam=0.3),     # non-inferior and explains with facets -> selected
    ]
    shares = {0.1: {"persona": 0.97, "facet": 0.03}, 0.2: {"facet": 0.5, "persona": 0.5}, 0.3: {"facet": 0.12, "persona": 0.88},
              0.0: {"persona": 1.0}}
    monkeypatch.setattr(tuning, "explanation_template_shares", lambda cached, res, w, lam, k, ag: shares[lam])
    sel = select_facet_active_config(points, cached_data=[], resources=None, n_resamples=500)
    assert [c["w_facet"] for c in sel["candidates"]] == [0.5, 0.25, 1.0]  # w_facet > 0, by NDCG
    by_facet = {c["w_facet"]: c for c in sel["candidates"]}
    assert by_facet[0.5]["non_inferior"] and not by_facet[0.5]["enough_facet_explanations"]
    assert not by_facet[1.0]["non_inferior"] and by_facet[1.0]["enough_facet_explanations"]
    assert sel["selected"]["w_facet"] == 0.25 and sel["selected"]["accepted"]
    assert sel["best"]["w_facet"] == 0.0 and sel["rule"]["max_relative_loss"] == 0.05

    monkeypatch.setattr(tuning, "explanation_template_shares", lambda *a: {"persona": 1.0})
    assert select_facet_active_config(points, [], None, n_resamples=200)["selected"] is None  # reported as "none"


def test_explanation_template_shares_count_dominant_features():
    from paperfeed.rec.features import CitationSignal, FacetMatch, NewMethod

    feats = [
        CandidateFeatures("p1", 0.1, None, CitationSignal(), FacetMatch(s_task=1.0, s_dataset=1.0), NewMethod(), 0.0),
        CandidateFeatures("p2", 0.9, None, CitationSignal(), FacetMatch(), NewMethod(), 0.0),
    ]
    user = CachedUserCandidateData("A1", SimpleNamespace(authored_papers=[{}] * 5, zero_pub=False), ["p1", "p2"], feats, [])
    store = MagicMock()
    store.select.side_effect = lambda pids: np.eye(2, dtype=np.float32)[: len(pids)]
    shares = explanation_template_shares([user], SimpleNamespace(store=store), ScoringWeights(w_facet=2.0), 0.0, k=2)
    assert shares == {"facet": 0.5, "persona": 0.5}


def test_load_facet_selection_checks_the_configuration(tmp_path):
    sig = tuning_signature(RecommendConfig(pool_size=100, channel_depth=200), ["A1"])
    path = default_facet_selection_path(tmp_path)
    assert path.name == "facet_config_selection.json"
    assert default_facet_selection_path(tmp_path, enhanced=True).name == "facet_config_selection_enhanced.json"
    with pytest.raises(FileNotFoundError):
        load_facet_selection(path, sig)
    chosen = {"weights": {"w_pers": 1.0, "w_cite": 1.0, "w_facet": 0.25, "w_nov": 0.0, "w_sess": 1.0, "lambda_neg": 0.5, "w_pop": 0.0},
              "lambda_div": 0.3}
    path.write_text(_json.dumps({"config_signature": sig, "selected": chosen}))
    weights, lam = load_facet_selection(path, sig)
    assert weights.w_facet == 0.25 and lam == 0.3
    with pytest.raises(ValueError, match="different configuration"):
        load_facet_selection(path, {**sig, "pool_size": 250})
    path.write_text(_json.dumps({"config_signature": sig, "selected": None}))
    with pytest.raises(ValueError, match="no grid point met"):
        load_facet_selection(path, sig)
