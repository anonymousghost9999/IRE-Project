"""Unit tests for Phase 5 Recommendation Engine (docs/plan.md §5, docs/design.md §8-§11)."""

import numpy as np
import pytest

from paperfeed.data.db import DatabaseManager
from paperfeed.index.dense import DenseIndex
from paperfeed.index.lexical import LexicalIndex
from paperfeed.index.retriever import HybridRetriever


def _unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# 5.1 Candidate generation
# ---------------------------------------------------------------------------

from paperfeed.rec.candidates import (
    eligible_paper_ids,
    feed_candidates,
    query_vector,
    retrieval_settings,
    search_candidates,
)


def test_query_vector_empty_session_is_the_persona():
    persona = _unit(1, 0)
    np.testing.assert_allclose(query_vector(persona, None), persona)
    np.testing.assert_allclose(query_vector(persona, np.zeros(2)), persona)


def test_query_vector_blends_session_and_persona():
    persona, session = _unit(1, 0), _unit(0, 1)
    q = query_vector(persona, session, rho=0.3)
    expected = 0.7 * session + 0.3 * persona
    np.testing.assert_allclose(q, expected / np.linalg.norm(expected), atol=1e-6)
    assert np.linalg.norm(q) == pytest.approx(1.0)


def test_query_vector_rho_extremes():
    persona, session = _unit(1, 0), _unit(0, 1)
    np.testing.assert_allclose(query_vector(persona, session, rho=1.0), persona, atol=1e-6)
    np.testing.assert_allclose(query_vector(persona, session, rho=0.0), session, atol=1e-6)
    with pytest.raises(ValueError):
        query_vector(persona, session, rho=1.5)


def test_query_vector_uses_raw_session_norm_unless_asked_to_normalise():
    persona, session = _unit(1, 0), np.array([0, 0.5], dtype=np.float32)  # EMA after one click: norm = 1 - alpha
    literal = query_vector(persona, session, rho=0.3)
    rescaled = query_vector(persona, session, rho=0.3, normalize_session=True)
    assert literal[0] / literal[1] > rescaled[0] / rescaled[1]  # the short session vector pulls less in the literal formula


def test_query_vector_cancelling_session_falls_back_to_persona():
    persona, session = _unit(1, 0), _unit(-1, 0)
    np.testing.assert_allclose(query_vector(persona, session, rho=0.5), persona)


@pytest.fixture
def retriever():
    """Dense axis: 'x' papers point along +x, 'y' along +y. Text: IR papers vs astronomy."""
    ids = ["P_IR1", "P_IR2", "P_IR3", "P_AST1", "P_AST2", "P_FUTURE", "P_AUTHORED"]
    vecs = np.array(
        [[1, 0.1], [1, 0.2], [1, 0.3], [0.1, 1], [0.2, 1], [1, 0.05], [1, 0.0]], dtype=np.float32
    )
    dense = DenseIndex(dimension=2)
    dense.add_documents(ids, vecs / np.linalg.norm(vecs, axis=1, keepdims=True))
    lex = LexicalIndex(stopwords="en")
    lex.index_documents(
        ids,
        [
            "information retrieval ranking", "information retrieval systems", "retrieval evaluation metrics",
            "astronomy galaxies", "astronomy telescopes", "information retrieval future", "information retrieval authored",
        ],
        show_progress=False,
    )
    return HybridRetriever(dense, lex, channel_depth=5, candidate_pool_size=4)


def test_feed_candidates_fuse_channels_and_report_ranks(retriever):
    cands = feed_candidates(retriever, _unit(1, 0.1), "information retrieval")
    assert 0 < len(cands) <= 4
    assert all(a.rrf_score >= b.rrf_score for a, b in zip(cands, cands[1:]))  # sorted by fused score
    top = cands[0]
    assert top.dense_rank is not None and top.lexical_rank is not None  # in both channels -> boosted to the top
    assert top.rrf_score == pytest.approx(1 / (60 + top.dense_rank) + 1 / (60 + top.lexical_rank))
    assert {c.paper_id for c in cands} <= {"P_IR1", "P_IR2", "P_IR3", "P_FUTURE", "P_AUTHORED", "P_AST1", "P_AST2"}


def test_feed_candidates_single_channel_hit_has_no_rank_in_the_other(retriever):
    # P_AST* match the dense query (y axis) but not the text query
    cands = feed_candidates(retriever, _unit(0, 1), "retrieval", exclusion_set=None)
    ast = [c for c in cands if c.paper_id.startswith("P_AST")]
    assert ast and all(c.lexical_rank is None and c.lexical_score is None and c.dense_rank is not None for c in ast)


def test_feed_candidates_apply_exclusion_before_fusion(retriever):
    cands = feed_candidates(retriever, _unit(1, 0), "information retrieval", exclusion_set={"P_AUTHORED", "W_UNKNOWN"})
    assert "P_AUTHORED" not in {c.paper_id for c in cands}
    # exclusion also applies to a URI-form ID
    cands = feed_candidates(retriever, _unit(1, 0), "information retrieval", exclusion_set={"https://openalex.org/P_IR1"})
    assert "P_IR1" not in {c.paper_id for c in cands}


def test_feed_candidates_candidate_filter_removes_future_papers(retriever):
    eligible = {"P_IR1", "P_IR2", "P_IR3", "P_AST1", "P_AST2", "P_AUTHORED"}  # P_FUTURE (year > T) is not eligible
    cands = feed_candidates(retriever, _unit(1, 0), "information retrieval", candidate_filter=eligible)
    assert "P_FUTURE" not in {c.paper_id for c in cands}


def test_feed_candidates_session_shifts_the_dense_channel(retriever):
    persona, session = _unit(1, 0.1), _unit(0.1, 1)
    persona_only = feed_candidates(retriever, persona, "")
    with_session = feed_candidates(retriever, persona, "", session_vec=session, rho=0.1)
    assert persona_only[0].paper_id.startswith("P_IR")
    assert with_session[0].paper_id.startswith("P_AST")  # the session dominates q_t at small rho


def test_feed_candidates_empty_pseudo_query_uses_dense_only(retriever):
    cands = feed_candidates(retriever, _unit(1, 0.1), "")
    assert cands and all(c.lexical_rank is None for c in cands)


def test_retriever_retrieve_still_returns_fused_pairs(retriever):
    pairs = retriever.retrieve(dense_query_vec=_unit(1, 0.1), lexical_query="information retrieval", top_m=3)
    cands = retriever.retrieve_candidates(dense_query_vec=_unit(1, 0.1), lexical_query="information retrieval", top_m=3)
    assert pairs == [(c.paper_id, c.rrf_score) for c in cands] and len(pairs) == 3


class FakeQueryEncoder:
    def __init__(self, vec):
        self.vec, self.seen = vec, []

    def embed_query(self, query):
        self.seen.append(query)
        return self.vec


def test_search_candidates_uses_query_embedding_and_text(retriever):
    enc = FakeQueryEncoder(_unit(0.1, 1))
    cands = search_candidates(retriever, enc, "  astronomy galaxies ")
    assert enc.seen == ["astronomy galaxies"]
    assert cands[0].paper_id == "P_AST1"  # dense and lexical agree
    assert cands[0].dense_rank is not None and cands[0].lexical_rank is not None


def test_search_candidates_empty_query_returns_nothing(retriever):
    enc = FakeQueryEncoder(_unit(1, 0))
    assert search_candidates(retriever, enc, "   ") == []
    assert enc.seen == []


def test_search_candidates_respect_exclusions(retriever):
    cands = search_candidates(retriever, FakeQueryEncoder(_unit(0.1, 1)), "astronomy", exclusion_set={"P_AST1"})
    assert "P_AST1" not in {c.paper_id for c in cands}


def test_eligible_paper_ids_filters_year_and_in_candidates(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "r.sqlite3", cutoff_year=2024)

    def paper(pid, year, inc=1):
        return {
            "paper_id": f"https://openalex.org/{pid}", "title": pid, "abstract": "x", "authors": "[]", "year": year,
            "publication_date": None, "venue": None, "doi": None, "url": None, "citation_count_T": 0,
            "primary_topic_id": None, "subfield_id": None, "field_id": None, "in_candidates": inc,
        }

    db.insert_papers_batch([paper("W1", 2023), paper("W2", 2024), paper("W3", 2025), paper("W4", 2020, inc=0)])
    assert eligible_paper_ids(db, 2024) == {"W1", "W2"}


def test_retrieval_settings_from_config():
    cfg = {"retrieval": {"channel_depth": 50, "rrf_k": 30, "candidate_pool_size": 20}, "user_model": {"rho_persona_weight": 0.4}}
    assert retrieval_settings(cfg) == {"channel_depth": 50, "rrf_k": 30, "candidate_pool_size": 20, "rho": 0.4}
    assert retrieval_settings({}) == {"channel_depth": 200, "rrf_k": 60, "candidate_pool_size": 100, "rho": 0.3}


# ---------------------------------------------------------------------------
# 5.2 Features
# ---------------------------------------------------------------------------

import math

from paperfeed.facets.store import FacetStore
from paperfeed.index.store import EmbeddingStore
from paperfeed.rec.features import (
    CitationSignal,
    build_citation_context,
    citation_signals,
    compute_features,
    facet_coverage,
    load_candidate_facets,
    new_method_signal,
    penalty_values,
)
from paperfeed.user.facets import UserFacets


def _paper_row(pid, year=2023, title=None):
    return {
        "paper_id": f"https://openalex.org/{pid}", "title": title or pid, "abstract": "x", "authors": "[]", "year": year,
        "publication_date": None, "venue": None, "doi": None, "url": None, "citation_count_T": 0,
        "primary_topic_id": None, "subfield_id": None, "field_id": None, "in_candidates": 1,
    }


def _uri(pid):
    return f"https://openalex.org/{pid}"


@pytest.fixture
def cite_db(tmp_path):
    """Authored A1 refs {R1, R2}; A2 refs {R2, R3}. Candidate P refs {R2, R3, R9}, so BC = |{R2, R3}| = 2.
    Third parties: C1 cites A1 + P, C2 cites A2 + P, C3 cites P only, C4 cites A1 only.  CC(P) = |{C1, C2}| = 2."""
    db = DatabaseManager(db_path=tmp_path / "c.sqlite3", cutoff_year=2024)
    ids = ["A1", "A2", "P", "Q", "C1", "C2", "C3", "C4", "C5"]
    db.insert_papers_batch([_paper_row(i, 2025 if i == "C5" else 2022) for i in ids])
    edges = [
        ("A1", "R1"), ("A1", "R2"), ("A2", "R2"), ("A2", "R3"),
        ("P", "R2"), ("P", "R3"), ("P", "R9"),
        ("C1", "A1"), ("C1", "P"), ("C2", "A2"), ("C2", "P"), ("C3", "P"), ("C4", "A1"),
        ("A2", "P"),  # an authored paper citing P is not a third party
        ("Q", "R5"),
    ]
    db.insert_citations_batch([(_uri(s), _uri(t), 2022) for s, t in edges])
    # a post-T citing paper that cites both A1 and P: must be ignored whatever the table holds
    with db.get_connection() as conn:
        conn.executemany(
            "INSERT INTO paper_citations (source_paper_id, target_paper_id, source_year) VALUES (?, ?, 2025)",
            [(_uri("C5"), _uri("A1")), (_uri("C5"), _uri("P"))],
        )
    return db


def test_bc_and_cc_hand_computed(cite_db):
    ctx = build_citation_context(cite_db, ["A1", "A2"], cutoff_year=2024)
    sig = citation_signals(cite_db, ctx, ["P", "Q"], cutoff_year=2024)
    p = sig["P"]
    assert p.bc == 2 and p.cc == 2  # C5 (year 2025) and the authored A2 are not counted
    assert p.log_cite == pytest.approx(math.log(5))
    assert (p.bc_paper, p.bc_paper_shared) == ("A2", 2)  # A2 shares R2 and R3, A1 only R2
    assert (p.cc_paper, p.cc_paper_shared) == ("A1", 1)  # C1 -> A1, C2 -> A2: tie, broken by id
    assert sig["Q"] == CitationSignal()  # shares nothing


def test_citation_signal_zero_for_zero_publication_users(cite_db):
    ctx = build_citation_context(cite_db, [], cutoff_year=2024)
    sig = citation_signals(cite_db, ctx, ["P"], cutoff_year=2024)
    assert sig["P"].bc == 0 and sig["P"].cc == 0 and sig["P"].log_cite == 0.0


def test_citation_signal_ignores_edges_after_cutoff(cite_db):
    # with T = 2021 nothing (all edges are from 2022 on) can count
    ctx = build_citation_context(cite_db, ["A1", "A2"], cutoff_year=2021)
    assert citation_signals(cite_db, ctx, ["P"], cutoff_year=2021)["P"] == CitationSignal()


@pytest.fixture
def user_facets():
    # 2 authored papers: t1 in both, t2 and d1 and m1 in one each
    return UserFacets(
        counts={"t1": 2, "t2": 1, "d1": 1, "m1": 1},
        types={"t1": "task", "t2": "task", "d1": "dataset", "m1": "method"},
        names={"t1": "task one", "t2": "task two", "d1": "dataset one", "m1": "method one"},
        paper_ids=["A1", "A2"],
    )


def test_user_facets_freq_is_share_of_papers(user_facets):
    assert user_facets.freq("t1") == 1.0 and user_facets.freq("t2") == 0.5 and user_facets.freq("nope") == 0.0


def test_facet_coverage_hand_computed(user_facets):
    # tasks {t1, t3}: only t1 shared (freq 1.0) -> S_task = 1.0 / 2; datasets {d1}: freq 0.5 / 1
    fm = facet_coverage({"task": {"t1", "t3"}, "dataset": {"d1"}, "method": set()}, user_facets)
    assert fm.s_task == pytest.approx(0.5) and fm.s_dataset == pytest.approx(0.5)
    assert fm.s_facet == pytest.approx(0.5)
    assert fm.matched == [("t1", 1.0), ("d1", 0.5)]  # best first, the evidence for the explainer


def test_facet_coverage_empty_sets_and_bounds(user_facets):
    assert facet_coverage({}, user_facets).s_facet == 0.0
    assert facet_coverage({"task": {"x"}, "dataset": set()}, user_facets).s_facet == 0.0
    full = facet_coverage({"task": {"t1"}, "dataset": {"d1"}}, UserFacets(
        counts={"t1": 2, "d1": 2}, types={"t1": "task", "d1": "dataset"}, paper_ids=["A1", "A2"]))
    assert full.s_facet == pytest.approx(1.0)  # upper bound: every candidate facet is in every authored paper


def test_new_method_indicator_cases(user_facets):
    # shared task + a method the user does not have -> 1
    nm = new_method_signal({"task": {"t1"}, "method": {"m1", "m_new"}}, user_facets)
    assert nm.flag and nm.shared_tasks == ["t1"] and nm.new_methods == ["m_new"]
    # shared task but only known methods -> 0
    assert not new_method_signal({"task": {"t1"}, "method": {"m1"}}, user_facets).flag
    # a new method on a task the user does not work on -> 0
    assert not new_method_signal({"task": {"t9"}, "method": {"m_new"}}, user_facets).flag
    # shared task, no methods at all -> 0
    assert not new_method_signal({"task": {"t1"}, "method": set()}, user_facets).flag


def test_penalty_is_max_cosine_to_skipped_papers():
    cands = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    np.testing.assert_allclose(penalty_values(cands, None), [0, 0, 0])
    np.testing.assert_allclose(penalty_values(cands, np.empty((0, 2), dtype=np.float32)), [0, 0, 0])
    negs = np.array([[1, 0], [0.6, 0.8]], dtype=np.float32)
    np.testing.assert_allclose(penalty_values(cands, negs), [1.0, 0.8, 2 ** -0.5 * 1.4], atol=1e-6)


@pytest.fixture
def facet_store_for_features(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "ff.sqlite3", cutoff_year=2024)
    db.insert_papers_batch([_paper_row(p) for p in ("P1", "P2", "P3")])
    store = FacetStore(db.db_path)

    def f(pid, ftype, cid):
        return {"paper_id": _uri(pid), "facet_type": ftype, "facet_value": "v", "canonical_id": cid,
                "evidence_sentence": "s", "start_char": 0, "end_char": 1, "confidence": 0.9}

    store.save_facets([
        f("P1", "task", "t1"), f("P1", "task", "t3"), f("P1", "dataset", "d1"), f("P1", "method", "m_new"),
        f("P1", "metric", "x1"), f("P2", "method", "m1"), f("P2", "task", None),
    ])
    return store


def test_load_candidate_facets(facet_store_for_features):
    out = load_candidate_facets(facet_store_for_features, ["P1", "P2", "P3"])
    assert out["P1"] == {"task": {"t1", "t3"}, "method": {"m_new"}, "dataset": {"d1"}}  # metric ignored
    assert out["P2"] == {"task": set(), "method": {"m1"}, "dataset": set()}  # uncanonicalized ignored
    assert out["P3"] == {"task": set(), "method": set(), "dataset": set()}
    none_store = load_candidate_facets(None, ["P1"])
    assert none_store["P1"]["task"] == set()


def test_compute_features_end_to_end(cite_db, user_facets, facet_store_for_features):
    ids = ["P", "Q", "A1"]
    emb = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)
    store = EmbeddingStore(ids, emb)
    ctx = build_citation_context(cite_db, ["A1", "A2"], 2024)

    feats = compute_features(
        ["P", "Q"], store, persona_vec=_unit(1, 0), user_facets=user_facets, citation_ctx=ctx, db=cite_db,
        cutoff_year=2024, session_vec=None, negative_ids=["A1"],
    )
    p, q = feats
    assert (p.paper_id, q.paper_id) == ("P", "Q")
    assert p.persona_cos == pytest.approx(1.0) and q.persona_cos == pytest.approx(0.0, abs=1e-6)
    assert p.session_cos is None and q.session_cos is None  # empty session: the feature is dropped
    assert p.citation.bc == 2 and p.log_cite == pytest.approx(math.log(5))
    assert p.penalty == pytest.approx(0.6) and q.penalty == pytest.approx(0.8)  # cosine to the skipped A1 = [0.6, 0.8]
    assert p.facet.s_facet == 0.0 and not p.new_method.flag  # no facet store given

    with_session = compute_features(
        ["P"], store, _unit(1, 0), user_facets, ctx, cite_db, 2024, session_vec=np.array([0, 0.5], dtype=np.float32),
    )
    assert with_session[0].session_cos == pytest.approx(0.0, abs=1e-6)
    assert with_session[0].penalty == 0.0  # nothing skipped


def test_compute_features_with_facets(cite_db, user_facets, facet_store_for_features):
    store = EmbeddingStore(["P1"], np.array([[1, 0]], dtype=np.float32))
    ctx = build_citation_context(cite_db, [], 2024)
    (f,) = compute_features(
        ["P1"], store, _unit(1, 0), user_facets, ctx, cite_db, 2024, facet_store=facet_store_for_features,
    )
    assert f.facet.s_task == pytest.approx(0.5) and f.facet.s_dataset == pytest.approx(0.5)
    assert f.new_method.flag and f.new_method.new_methods == ["m_new"]
    assert f.citation == CitationSignal()  # zero-publication style: no authored papers


def test_compute_features_empty_and_missing_embedding(cite_db, user_facets):
    ctx = build_citation_context(cite_db, [], 2024)
    store = EmbeddingStore(["P"], np.array([[1, 0]], dtype=np.float32))
    assert compute_features([], store, _unit(1, 0), user_facets, ctx, cite_db, 2024) == []
    with pytest.raises(KeyError):
        compute_features(["unknown"], store, _unit(1, 0), user_facets, ctx, cite_db, 2024)


# ---------------------------------------------------------------------------
# 5.3 Scaling and score
# ---------------------------------------------------------------------------

from paperfeed.rec.features import CandidateFeatures, FacetMatch, NewMethod
from paperfeed.rec.scoring import FEATURE_NAMES, ScoringWeights, min_max, rank_candidates, score_candidates


def _feat(pid, persona=0.5, session=None, bc=0, cc=0, s_task=0.0, s_dataset=0.0, new_method=False, penalty=0.0):
    return CandidateFeatures(
        paper_id=pid, persona_cos=persona, session_cos=session,
        citation=CitationSignal(bc=bc, cc=cc), facet=FacetMatch(s_task=s_task, s_dataset=s_dataset),
        new_method=NewMethod(flag=new_method), penalty=penalty,
    )


def test_min_max_scaling():
    np.testing.assert_allclose(min_max([2.0, 4.0, 3.0]), [0.0, 1.0, 0.5])
    np.testing.assert_allclose(min_max([0.7, 0.7, 0.7]), [0, 0, 0])  # constant -> 0
    np.testing.assert_allclose(min_max([5.0]), [0.0])  # a one-paper pool
    assert min_max([]).size == 0
    np.testing.assert_allclose(min_max([-1.0, 0.0, 1.0]), [0.0, 0.5, 1.0])


def test_score_hand_computed():
    pool = [
        _feat("A", persona=0.2, session=0.9, bc=0, s_task=0.0),
        _feat("B", persona=0.6, session=0.1, bc=3, s_task=1.0, s_dataset=1.0, new_method=True, penalty=0.4),
        _feat("C", persona=1.0, session=0.5, bc=1, s_task=0.5),
    ]
    w = ScoringWeights(w_pers=1.0, w_sess=1.0, w_cite=0.5, w_facet=0.5, w_nov=0.5, lambda_neg=0.5)
    a, b, c = score_candidates(pool, w)

    # persona mm: [0, .5, 1]; session mm: [1, 0, .5]; log1p(BC): [0, ln4, ln2] -> mm [0, 1, .5]; s_facet: [0, 1, .25] -> mm same
    assert a.scaled["persona"] == 0.0 and b.scaled["persona"] == pytest.approx(0.5) and c.scaled["persona"] == 1.0
    assert [a.scaled["session"], b.scaled["session"], c.scaled["session"]] == pytest.approx([1.0, 0.0, 0.5])
    assert [a.scaled["citation"], b.scaled["citation"], c.scaled["citation"]] == pytest.approx([0.0, 1.0, math.log(2) / math.log(4)])
    assert [a.scaled["facet"], b.scaled["facet"], c.scaled["facet"]] == pytest.approx([0.0, 1.0, 0.125 / 0.5])
    assert a.score == pytest.approx(0 + 1.0 + 0 + 0 + 0 - 0)
    assert b.score == pytest.approx(0.5 + 0.0 + 0.5 + 0.5 + 0.5 - 0.5 * 0.4)  # penalty enters raw, not scaled
    assert c.score == pytest.approx(1.0 + 0.5 + 0.5 * 0.5 + 0.5 * 0.25 + 0.0 - 0.0)


def test_contributions_sum_to_score_and_penalty_is_negative():
    pool = [_feat("A", 0.1, 0.2, bc=1, penalty=0.3), _feat("B", 0.9, 0.8, bc=4, s_task=1.0, new_method=True)]
    for s in score_candidates(pool, ScoringWeights()):
        assert set(s.contributions) == set(FEATURE_NAMES) == set(s.scaled)
        assert s.score == pytest.approx(sum(s.contributions.values()))
        assert s.contributions["penalty"] <= 0 and all(v >= 0 for k, v in s.contributions.items() if k != "penalty")
    a = score_candidates(pool, ScoringWeights(lambda_neg=2.0))[0]
    assert a.contributions["penalty"] == pytest.approx(-0.6)


def test_empty_session_drops_the_session_term():
    pool = [_feat("A", persona=0.2, session=None), _feat("B", persona=0.8, session=None)]
    a, b = score_candidates(pool, ScoringWeights(w_sess=5.0))
    assert not a.session_active and a.contributions["session"] == 0.0 and b.contributions["session"] == 0.0
    assert b.score == pytest.approx(1.0)  # persona term only


def test_constant_features_contribute_nothing_and_binary_is_not_scaled():
    # every candidate has the same persona cosine and BC: scaled to 0; the lone new-method candidate keeps the full w_nov
    pool = [_feat("A", persona=0.5, bc=2), _feat("B", persona=0.5, bc=2, new_method=True), _feat("C", persona=0.5, bc=2)]
    a, b, c = score_candidates(pool, ScoringWeights(w_nov=0.7))
    assert a.score == 0.0 and c.score == 0.0
    assert b.scaled["new_method"] == 1.0 and b.score == pytest.approx(0.7)  # not z-scored: a rare indicator does not explode


def test_single_candidate_pool():
    (s,) = score_candidates([_feat("A", persona=0.9, session=0.9, bc=5, s_task=1.0, new_method=True, penalty=0.2)], ScoringWeights())
    assert s.scaled["persona"] == s.scaled["session"] == s.scaled["citation"] == s.scaled["facet"] == 0.0
    assert s.score == pytest.approx(0.5 - 0.5 * 0.2)
    assert score_candidates([], ScoringWeights()) == []


def test_zero_weight_switches_a_term_off():
    pool = [_feat("A", 0.1, bc=0), _feat("B", 0.9, bc=9)]
    full = score_candidates(pool, ScoringWeights(w_cite=1.0))
    no_cite = score_candidates(pool, ScoringWeights(w_cite=0.0))
    assert full[1].score - no_cite[1].score == pytest.approx(1.0)
    assert no_cite[1].contributions["citation"] == 0.0


def test_rank_candidates_is_deterministic_with_ties():
    pool = [_feat("B", 0.5), _feat("A", 0.5), _feat("C", 0.9)]
    ranked = rank_candidates(score_candidates(pool, ScoringWeights()))
    assert [s.paper_id for s in ranked] == ["C", "A", "B"]  # tie between A and B broken by id
    assert [s.paper_id for s in rank_candidates(score_candidates(pool[::-1], ScoringWeights()))] == ["C", "A", "B"]


def test_scoring_weights_validation_and_config():
    with pytest.raises(ValueError, match="w_cite"):
        ScoringWeights(w_cite=-0.1)
    cfg = {"scoring_weights": {"w_pers": 1.0, "w_sess": 1.5, "w_cite": 0.25, "lambda_div": 0.2, "lambda_neg": 1.0}}
    w = ScoringWeights.from_config(cfg)
    assert (w.w_sess, w.w_cite, w.w_facet, w.lambda_neg) == (1.5, 0.25, 0.5, 1.0)  # unset w_facet keeps its default
    assert ScoringWeights.from_config({}) == ScoringWeights()
    from paperfeed.common.config import load_config
    assert ScoringWeights.from_config(load_config()) == ScoringWeights()  # the defaults in default.yaml
    assert w.with_overrides(w_nov=0.0).w_nov == 0.0 and w.w_nov == 0.5


# ---------------------------------------------------------------------------
# 5.4 MMR
# ---------------------------------------------------------------------------

from paperfeed.rec.mmr import mmr_select, mmr_settings, select_top_k
from paperfeed.rec.scoring import ScoredCandidate


def _sc(pid, score):
    return ScoredCandidate(paper_id=pid, features=_feat(pid), scaled={}, contributions={}, score=score)


def _pool():
    """A and B are near-duplicates (same direction); C is on another axis."""
    scored = [_sc("A", 1.0), _sc("B", 0.95), _sc("C", 0.6)]
    vecs = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    return scored, vecs


def test_mmr_lambda_zero_is_pure_score_order():
    scored, vecs = _pool()
    assert [p.paper_id for p in mmr_select(scored, vecs, k=3, lambda_div=0.0)] == ["A", "B", "C"]


def test_mmr_diversifies_hand_computed():
    scored, vecs = _pool()
    picks = mmr_select(scored, vecs, k=3, lambda_div=0.5)
    # 1st: A (1.0). 2nd: B = 0.95 - 0.5 * 1 = 0.45 vs C = 0.6 - 0.5 * 0 = 0.6 -> C. 3rd: B = 0.45.
    assert [p.paper_id for p in picks] == ["A", "C", "B"]
    assert [p.rank for p in picks] == [1, 2, 3]
    assert picks[0].max_similarity == 0.0 and picks[0].mmr_value == pytest.approx(1.0)
    assert picks[1].mmr_value == pytest.approx(0.6)
    assert picks[2].max_similarity == pytest.approx(1.0) and picks[2].mmr_value == pytest.approx(0.45)


def test_mmr_small_lambda_keeps_the_relevant_duplicate():
    scored, vecs = _pool()
    assert [p.paper_id for p in mmr_select(scored, vecs, k=2, lambda_div=0.01)] == ["A", "B"]  # 0.94 still beats C


def test_mmr_uses_max_similarity_over_the_picked_set():
    scored = [_sc("A", 1.0), _sc("B", 0.9), _sc("C", 0.8), _sc("D", 0.79)]
    vecs = np.array([[1, 0], [0, 1], [1, 0.1], [0.5, 0.5]], dtype=np.float32)
    picks = mmr_select(scored, vecs, k=4, lambda_div=1.0)
    # after A, B: C is near A (cos ~ 0.995), D is 0.707 from both; pick order is A, B, then D (0.79 - 0.707 > 0.8 - 0.995)
    assert [p.paper_id for p in picks] == ["A", "B", "D", "C"]
    assert picks[2].max_similarity == pytest.approx(0.5 ** 0.5, abs=1e-5)


def test_mmr_returns_k_unique_papers():
    rng = np.random.default_rng(0)
    scored = [_sc(f"P{i:03d}", float(rng.random())) for i in range(100)]
    vecs = rng.normal(size=(100, 8)).astype(np.float32)
    picks = mmr_select(scored, vecs, k=10, lambda_div=0.2)
    assert len(picks) == 10 and len({p.paper_id for p in picks}) == 10
    assert picks[0].paper_id == max(scored, key=lambda s: s.score).paper_id  # the first pick is the top-scored paper


def test_mmr_k_larger_than_pool_and_edge_cases():
    scored, vecs = _pool()
    assert len(mmr_select(scored, vecs, k=10)) == 3
    assert mmr_select(scored, vecs, k=0) == []
    assert mmr_select([], np.empty((0, 2)), k=5) == []
    with pytest.raises(ValueError, match="mismatch"):
        mmr_select(scored, vecs[:2])
    with pytest.raises(ValueError, match="unique"):
        mmr_select([_sc("A", 1.0), _sc("A", 0.5)], np.eye(2, dtype=np.float32))
    with pytest.raises(ValueError):
        mmr_select(scored, vecs, lambda_div=-1)


def test_mmr_ties_break_by_paper_id_and_ignore_input_order():
    scored = [_sc("B", 0.5), _sc("A", 0.5), _sc("C", 0.5)]
    vecs = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    order = [p.paper_id for p in mmr_select(scored, vecs, k=3, lambda_div=0.0)]
    assert order == ["A", "B", "C"]
    perm = [2, 0, 1]
    shuffled = mmr_select([scored[i] for i in perm], vecs[perm], k=3, lambda_div=0.0)
    assert [p.paper_id for p in shuffled] == order


def test_mmr_negative_similarity_is_a_bonus_not_clamped_to_zero():
    # after picking A, B is opposite to A (cos = -1): with lambda 0.5 its value rises above its raw score
    scored = [_sc("A", 1.0), _sc("B", 0.5), _sc("C", 0.6)]
    vecs = np.array([[1, 0], [-1, 0], [0, 1]], dtype=np.float32)
    picks = mmr_select(scored, vecs, k=2, lambda_div=0.5)
    assert [p.paper_id for p in picks] == ["A", "B"]  # B: 0.5 + 0.5 = 1.0 beats C: 0.6


def test_select_top_k_reads_vectors_from_the_store():
    scored, vecs = _pool()
    store = EmbeddingStore(["A", "B", "C", "X"], np.vstack([vecs, [[1, 1]]]).astype(np.float32))
    assert [p.paper_id for p in select_top_k(scored, store, k=3, lambda_div=0.5)] == ["A", "C", "B"]
    assert select_top_k([], store) == []


def test_mmr_settings_from_config():
    assert mmr_settings({"retrieval": {"feed_size": 5}, "scoring_weights": {"lambda_div": 0.3}}) == {"k": 5, "lambda_div": 0.3}
    assert mmr_settings({}) == {"k": 10, "lambda_div": 0.2}


# ---------------------------------------------------------------------------
# 5.5 Explainer
# ---------------------------------------------------------------------------

from paperfeed.rec.explain import (
    EXPLAINABLE_FEATURES,
    Explainer,
    dominant_feature,
    explanation_coverage,
    span_is_grounded,
)
from paperfeed.user.session import SessionState

ABSTRACT = "We propose BERT for passage retrieval. Results on MS MARCO improve."


def _span(text, sub):
    i = text.index(sub)
    return i, i + len(sub)


def _scored(pid, contributions, **feat_kwargs):
    feats = _feat(pid, **{k: v for k, v in feat_kwargs.items() if k not in ("facet", "citation", "new_method")})
    if "facet" in feat_kwargs:
        feats.facet = feat_kwargs["facet"]
    if "citation" in feat_kwargs:
        feats.citation = feat_kwargs["citation"]
    if "new_method" in feat_kwargs:
        feats.new_method = feat_kwargs["new_method"]
    full = {name: 0.0 for name in ("persona", "session", "citation", "facet", "new_method", "penalty")}
    full.update(contributions)
    return ScoredCandidate(paper_id=pid, features=feats, scaled={}, contributions=full, score=sum(full.values()))


def test_dominant_feature_picks_largest_positive_contribution():
    assert dominant_feature({"persona": 0.3, "citation": 0.9, "facet": 0.2}) == "citation"
    assert dominant_feature({"persona": 0.3, "penalty": -5.0}) == "persona"  # a deduction is never explained
    assert dominant_feature({"penalty": 9.0, "persona": 0.1}) == "persona"  # the penalty key is not explainable at all
    assert dominant_feature({"persona": 0.0, "facet": 0.0}) is None
    assert dominant_feature({}) is None
    # ties go to the earlier feature of EXPLAINABLE_FEATURES
    assert dominant_feature({"persona": 0.5, "session": 0.5, "citation": 0.5}) == "citation"
    assert dominant_feature({"facet": 0.5, "new_method": 0.5}) == "facet"
    assert EXPLAINABLE_FEATURES == ("facet", "new_method", "citation", "session", "persona")


def test_span_is_grounded_checks_offsets_and_sentence():
    s, e = _span(ABSTRACT, "passage retrieval")
    row = {"facet_value": "passage retrieval", "start_char": s, "end_char": e, "evidence_sentence": "We propose BERT for passage retrieval."}
    assert span_is_grounded(ABSTRACT, row)
    assert not span_is_grounded(ABSTRACT, {**row, "start_char": s + 1, "end_char": e + 1})  # wrong offsets
    assert not span_is_grounded(ABSTRACT, {**row, "evidence_sentence": "Results on MS MARCO improve."})  # span outside its sentence
    assert not span_is_grounded(ABSTRACT, {**row, "evidence_sentence": "We never wrote this sentence."})
    assert not span_is_grounded(ABSTRACT, {**row, "facet_value": "dense retrieval"})
    assert not span_is_grounded("", row) and not span_is_grounded(ABSTRACT, {**row, "start_char": None})


@pytest.fixture
def facet_env(tmp_path):
    """Candidate P1 with abstract ABSTRACT and canonical facets; user works on t1 (2 of 2 papers) and d1 (1 of 2)."""
    db = DatabaseManager(db_path=tmp_path / "x.sqlite3", cutoff_year=2024)
    row = _paper_row("P1", title="A Candidate")
    row["abstract"] = ABSTRACT
    db.insert_papers_batch([row])
    store = FacetStore(db.db_path)

    def f(ftype, cid, sub, sentence):
        s, e = _span(ABSTRACT, sub)
        return {"paper_id": _uri("P1"), "facet_type": ftype, "facet_value": sub, "canonical_id": cid,
                "evidence_sentence": sentence, "start_char": s, "end_char": e, "confidence": 0.9}

    s1, s2 = "We propose BERT for passage retrieval.", "Results on MS MARCO improve."
    store.save_facets([f("task", "t1", "passage retrieval", s1), f("method", "m_new", "BERT", s1), f("dataset", "d1", "MS MARCO", s2)])
    user_facets = UserFacets(
        counts={"t1": 2, "d1": 1}, types={"t1": "task", "d1": "dataset"}, names={"t1": "retrieval", "d1": "ms marco"},
        paper_ids=["A1", "A2"],
    )
    return db, store, user_facets


def _explainer(db, user_facets, store=None, facet_store=None, session=None, ctx=None):
    return Explainer(user_facets, db, 2024, ctx or build_citation_context(db, [], 2024), store=store, facet_store=facet_store, session=session)


def test_facet_explanation_uses_best_matched_facet_and_verbatim_span(facet_env):
    db, fstore, uf = facet_env
    ex = _explainer(db, uf, facet_store=fstore)
    both = _scored("P1", {"facet": 0.8, "persona": 0.1}, facet=FacetMatch(s_task=1.0, s_dataset=0.5, matched=[("t1", 1.0), ("d1", 0.5)]))
    r = ex.explain(both)
    assert r.dominant_feature == "facet" and r.covered
    assert r.explanation.text == "Addresses **passage retrieval**, studied in 2 of your papers"
    assert r.explanation.evidence["span"] == "passage retrieval" and r.explanation.evidence["user_papers_with_facet"] == 2

    only_dataset = _scored("P1", {"facet": 0.8}, facet=FacetMatch(s_dataset=0.5, matched=[("d1", 0.5)]))
    assert ex.explain(only_dataset).explanation.text == "Evaluates on **MS MARCO**, used in 1 of your papers"


def test_facet_explanation_refused_when_span_is_not_grounded(facet_env):
    db, fstore, uf = facet_env
    with db.get_connection() as conn:  # corrupt the stored offsets of every facet of P1
        conn.execute("UPDATE paper_facets SET start_char = start_char + 3, end_char = end_char + 3")
    ex = _explainer(db, uf, facet_store=fstore)
    sc = _scored("P1", {"facet": 0.8, "persona": 0.7}, persona=0.9, facet=FacetMatch(s_task=1.0, matched=[("t1", 1.0)]))
    r = ex.explain(sc)
    assert r.dominant_feature == "facet" and not r.covered and r.explanation is None
    assert "grounded" in r.failure  # and it does NOT fall back to the persona explanation although persona contributes 0.7


def test_new_method_explanation(facet_env):
    db, fstore, uf = facet_env
    ex = _explainer(db, uf, facet_store=fstore)
    sc = _scored("P1", {"new_method": 0.5}, new_method=NewMethod(flag=True, shared_tasks=["t1"], new_methods=["m_new"]))
    r = ex.explain(sc)
    assert r.explanation.text == "Proposes **BERT**, which is new to your work, for **passage retrieval** that you work on"
    assert r.explanation.evidence["method"]["span"] == "BERT" and r.explanation.evidence["task"]["span"] == "passage retrieval"
    # the new method is not in the abstract's facets -> cannot be grounded
    missing = _scored("P1", {"new_method": 0.5}, new_method=NewMethod(flag=True, shared_tasks=["t1"], new_methods=["m_unknown"]))
    assert not ex.explain(missing).covered


def test_facet_features_without_a_facet_store_are_not_explained(facet_env):
    db, _, uf = facet_env
    r = _explainer(db, uf).explain(_scored("P1", {"facet": 0.4}, facet=FacetMatch(s_task=1.0, matched=[("t1", 1.0)])))
    assert not r.covered and r.dominant_feature == "facet"


def test_citation_explanation_bc_and_cc_are_edge_verified(cite_db, user_facets):
    ctx = build_citation_context(cite_db, ["A1", "A2"], 2024)
    ex = _explainer(cite_db, user_facets, ctx=ctx)

    bc = _scored("P", {"citation": 0.5}, citation=CitationSignal(bc=2, cc=2, bc_paper="A2", bc_paper_shared=2, cc_paper="A1", cc_paper_shared=1))
    r = ex.explain(bc)
    assert r.explanation.text == "Shares **2** references with your paper *A2*"
    assert r.explanation.evidence["kind"] == "bibliographic_coupling" and r.explanation.evidence["shared_references"] == 2

    cc = _scored("P", {"citation": 0.5}, citation=CitationSignal(bc=0, cc=2, cc_paper="A1", cc_paper_shared=1))
    r = ex.explain(cc)
    assert r.explanation.text == "Frequently cited together with your paper *A1*"
    assert r.explanation.evidence["co_citing_papers"] == 1  # C1 only: C5 is post-T, the authored A2 is not a third party


def test_citation_explanation_refused_when_edges_do_not_verify(cite_db, user_facets):
    ctx = build_citation_context(cite_db, ["A1", "A2"], 2024)
    ex = _explainer(cite_db, user_facets, ctx=ctx)
    wrong_count = _scored("P", {"citation": 0.5}, citation=CitationSignal(bc=3, bc_paper="A2", bc_paper_shared=3))
    r = ex.explain(wrong_count)
    assert not r.covered and "verified" in r.failure
    not_a_cocitation = _scored("Q", {"citation": 0.5}, citation=CitationSignal(bc=0, cc=1, cc_paper="A1", cc_paper_shared=1))
    assert not ex.explain(not_a_cocitation).covered  # Q has no citing papers at all
    no_paper = _scored("P", {"citation": 0.5}, citation=CitationSignal())
    assert not ex.explain(no_paper).covered


def test_citation_edges_after_cutoff_are_not_evidence(cite_db, user_facets):
    ctx = build_citation_context(cite_db, ["A1", "A2"], 2024)
    ex = Explainer(user_facets, cite_db, 2021, ctx)  # with T = 2021 every 2022 edge is "future": nothing can verify
    sc = _scored("P", {"citation": 0.5}, citation=CitationSignal(bc=2, bc_paper="A2", bc_paper_shared=2))
    assert not ex.explain(sc).covered


def test_session_explanation_names_the_most_similar_session_paper(cite_db, user_facets):
    store = EmbeddingStore(["A1", "A2", "P"], np.array([[1, 0], [0, 1], [0.1, 1]], dtype=np.float32))
    session = SessionState(dim=2)
    session.click("A1", np.array([1, 0], dtype=np.float32))
    session.save("A2", np.array([0, 1], dtype=np.float32))
    ex = _explainer(cite_db, user_facets, store=store, session=session)
    r = ex.explain(_scored("P", {"session": 0.6, "persona": 0.2}))
    assert r.explanation.text == "Similar to **A2** you opened"  # P points along +y like A2
    assert r.explanation.evidence["session_paper_id"] == "A2"
    # empty session / no store: nothing to refer to
    assert not _explainer(cite_db, user_facets, store=store, session=SessionState(dim=2)).explain(_scored("P", {"session": 0.6})).covered
    assert not _explainer(cite_db, user_facets, session=session).explain(_scored("P", {"session": 0.6})).covered


def test_persona_explanation(cite_db, user_facets):
    ex = _explainer(cite_db, user_facets)
    r = ex.explain(_scored("P", {"persona": 1.0}, persona=0.8))
    assert r.explanation.text == "Close to your overall research profile"
    assert not ex.explain(_scored("P", {"persona": 1.0}, persona=-0.1)).covered


def test_no_positive_contribution_means_no_explanation(cite_db, user_facets):
    r = _explainer(cite_db, user_facets).explain(_scored("P", {"penalty": -0.4}))
    assert r.dominant_feature is None and not r.covered and r.failure


def test_explanation_coverage_counts_ungrounded_as_not_covered(cite_db, user_facets):
    ex = _explainer(cite_db, user_facets)
    results = ex.explain_all([_scored("P", {"persona": 1.0}, persona=0.8), _scored("Q", {"citation": 1.0}, citation=CitationSignal())])
    assert [r.covered for r in results] == [True, False]
    assert explanation_coverage(results) == 0.5
    assert explanation_coverage([]) == 0.0


def test_explained_feature_always_matches_the_scored_dominant_contribution(cite_db, user_facets):
    pool = [
        _feat("P", persona=0.9, bc=2, s_task=0.0), _feat("Q", persona=0.2, bc=0), _feat("A1", persona=0.5, bc=1),
    ]
    scored = score_candidates(pool, ScoringWeights(w_pers=1.0, w_cite=3.0))
    ex = _explainer(cite_db, user_facets)
    for s in scored:
        assert ex.explain(s).dominant_feature == dominant_feature(s.contributions)


# ---------------------------------------------------------------------------
# 5.6 Recommender interface
# ---------------------------------------------------------------------------

import json

from paperfeed.rec.recommender import (
    RecommendConfig,
    RecommenderResources,
    UserProfile,
    _check_output,
    build_user_profile,
    recommend,
)

IR_TEXT = {
    "U1": "information retrieval authored alpha", "U2": "information retrieval authored beta",
    "I1": "We study passage retrieval with dense models.", "I2": "information retrieval ranking gamma",
    "I3": "information retrieval evaluation delta", "I4": "information retrieval systems epsilon",
    "R1": "information retrieval cited zeta", "F": "information retrieval future eta", "N": "information retrieval noncandidate theta",
}
AST_TEXT = {"A1": "astronomy galaxies one", "A2": "astronomy telescopes two", "A3": "astronomy stars three"}
VECS = {
    "U1": (1, 0.1), "U2": (1, 0.2), "I1": (1, 0.3), "I2": (1, 0.4), "I3": (1, 0.5), "I4": (1, 0.6), "R1": (1, 0.35),
    "F": (1, 0.15), "N": (1, 0.12), "A1": (0.1, 1), "A2": (0.2, 1), "A3": (0.3, 1),
}


class WorldEncoder:
    """Fake SPECTER2: every query embeds to the +x ('IR') axis unless it mentions astronomy."""

    def embed_query(self, query):
        return _unit(0.1, 1) if "astronomy" in query else _unit(1, 0.2)

    def embed_queries(self, queries):
        return np.vstack([self.embed_query(q) for q in queries])

    def embed_papers(self, papers, batch_size=32, show_progress=False):
        return np.vstack([_unit(1, 0.25) for _ in papers])


@pytest.fixture
def world(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "w.sqlite3", cutoff_year=2024)
    texts = {**IR_TEXT, **AST_TEXT}
    rows = []
    for pid, text in texts.items():
        row = _paper_row(pid, year=2025 if pid == "F" else 2022, title=f"Title {pid}")
        row["abstract"] = text
        row["in_candidates"] = 0 if pid == "N" else 1
        rows.append(row)
    db.insert_papers_batch(rows)
    db.insert_citations_batch([(_uri(s), _uri(t), 2022) for s, t in [("U1", "R1"), ("U1", "R9"), ("I1", "R9"), ("U2", "R9")]])

    ids = list(texts)
    vecs = np.array([VECS[i] for i in ids], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    dense = DenseIndex(dimension=2)
    dense.add_documents(ids, vecs)
    lex = LexicalIndex(stopwords="en")
    lex.index_documents(ids, [texts[i] for i in ids], show_progress=False)
    retriever = HybridRetriever(dense, lex, channel_depth=12, candidate_pool_size=12)
    return RecommenderResources(
        db=db, store=EmbeddingStore(ids, vecs), retriever=retriever, cutoff_year=2024,
        eligible=eligible_paper_ids(db, 2024), facet_store=None, encoder=WorldEncoder(),
    )


CFG = RecommendConfig(k=4, lambda_div=0.0)


def _profile(world, authored=("U1", "U2")):
    return build_user_profile(world, "u1", authored_ids=list(authored), config=CFG)


def test_feed_recommendations_respect_every_filter(world):
    prof = _profile(world)
    res = recommend(world, prof, config=CFG)
    ids = res.paper_ids
    assert 0 < len(ids) <= 4 and len(set(ids)) == len(ids)
    assert not {"U1", "U2"} & set(ids)  # authored
    assert "R1" not in ids  # cited by the user
    assert "F" not in ids  # published after T
    assert "N" not in ids  # not a candidate
    assert [i.rank for i in res.items] == list(range(1, len(ids) + 1))
    assert res.n_excluded == len({"U1", "U2", "R1", "R9"}) and not res.session_active
    assert all(i.title.startswith("Title") and i.score is not None and i.dense_rank for i in res.items)


def test_recommend_is_deterministic(world):
    prof = _profile(world)
    a = json.dumps(recommend(world, prof, config=CFG).to_dict(), sort_keys=True)
    b = json.dumps(recommend(world, _profile(world), config=CFG).to_dict(), sort_keys=True)
    assert a == b


def test_diagnostic_mode_makes_cited_papers_eligible_again(world):
    prof = _profile(world)
    assert "R1" not in recommend(world, prof, config=CFG.with_overrides(k=12)).paper_ids
    diag = recommend(world, prof, config=CFG.with_overrides(k=12, diagnostic=True))
    assert "R1" in diag.paper_ids and not {"U1", "U2"} & set(diag.paper_ids)  # authored stay excluded


def test_zero_pub_switch_empties_the_exclusion_set(world):
    res = recommend(world, _profile(world), config=CFG.with_overrides(k=12, zero_pub=True))
    assert res.n_excluded == 0 and {"U1", "U2", "R1"} <= set(res.paper_ids)
    assert "F" not in res.paper_ids and "N" not in res.paper_ids  # the eligibility filter still applies


def test_session_shifts_recommendations_and_saved_papers_are_excluded(world):
    prof = _profile(world)
    cfg = CFG.with_overrides(k=3, rho=0.1, weights={"w_sess": 3.0})
    base = recommend(world, prof, config=cfg)
    assert not any(p.startswith("A") for p in base.paper_ids)

    session = cfg.make_session(dim=2)
    session.save("A1", world.store.select(["A1"])[0])
    session.click("A2", world.store.select(["A2"])[0])
    shifted = recommend(world, prof, session=session, config=cfg)
    assert shifted.session_active and any(p.startswith("A") for p in shifted.paper_ids)
    assert "A1" not in shifted.paper_ids  # saved -> SessionConsumed -> excluded
    assert shifted.n_excluded == base.n_excluded + 1
    assert any(i.contributions["session"] > 0 for i in shifted.items)


def test_skipped_papers_are_penalised(world):
    prof = _profile(world)
    cfg = CFG.with_overrides(k=12)
    session = cfg.make_session(dim=2)
    session.skip("I2")
    res = recommend(world, prof, session=session, config=cfg)
    assert not res.session_active  # a skip alone does not start a session vector
    by_id = {i.paper_id: i for i in res.items}
    assert by_id["I3"].contributions["penalty"] < 0  # I3 is close to the skipped I2
    assert by_id["A3"].contributions["penalty"] > by_id["I3"].contributions["penalty"]  # far from it: smaller deduction


def test_ablation_switches(world):
    prof = _profile(world)
    no_cite = recommend(world, prof, config=CFG.with_overrides(k=12, weights={"w_cite": 0.0}))
    assert all(i.contributions["citation"] == 0.0 for i in no_cite.items)
    with_cite = recommend(world, prof, config=CFG.with_overrides(k=12, weights={"w_cite": 2.0}))
    assert any(i.contributions["citation"] > 0 for i in with_cite.items)  # I1 shares R9 with the user's papers

    assert len(recommend(world, prof, config=CFG.with_overrides(k=2)).items) == 2
    small_pool = recommend(world, prof, config=CFG.with_overrides(k=10, pool_size=3))
    assert small_pool.n_candidates == 3 and len(small_pool.items) == 3

    # lambda_div = 0 is the pure score order
    ranked = recommend(world, prof, config=CFG.with_overrides(k=12, lambda_div=0.0)).items
    assert [i.score for i in ranked] == sorted([i.score for i in ranked], reverse=True)


def test_explanations_can_be_switched_off(world):
    prof = _profile(world)
    on = recommend(world, prof, config=CFG)
    off = recommend(world, prof, config=CFG.with_overrides(explain=False))
    assert all(i.explanation is not None for i in on.items) and all(i.explanation is None for i in off.items)
    assert on.paper_ids == off.paper_ids  # explaining never changes the ranking
    assert all(i.explanation.dominant_feature in {"persona", "citation"} for i in on.items)  # no facets or session here


def test_pseudo_query_comes_from_recent_authored_titles(world):
    res = recommend(world, _profile(world), config=CFG)
    assert "Title U1" in res.pseudo_query and "Title U2" in res.pseudo_query


def test_search_mode(world):
    prof = _profile(world)
    res = recommend(world, prof, mode="search", query="astronomy galaxies", config=CFG)
    assert res.mode == "search" and res.paper_ids[0] == "A1"
    assert all(i.score is None and i.explanation is None and i.contributions == {} for i in res.items)
    assert res.pseudo_query == "astronomy galaxies"
    ir = recommend(world, prof, mode="search", query="information retrieval", config=CFG.with_overrides(k=12))
    assert not {"U1", "U2", "R1", "F", "N"} & set(ir.paper_ids)  # E_u and the eligibility filter apply here too
    assert recommend(world, prof, mode="search", query="   ", config=CFG).items == []
    with pytest.raises(ValueError, match="Unknown mode"):
        recommend(world, prof, mode="browse")
    world.encoder = None
    with pytest.raises(ValueError, match="encoder"):
        recommend(world, prof, mode="search", query="x")


def test_zero_publication_profile_and_recommendations(world):
    prof = build_user_profile(world, "z1", seed_topics=["information retrieval", "dense retrieval"], config=CFG)
    assert prof.zero_pub and prof.persona.source == "seed_topics" and prof.exclusion_sources.build() == set()
    res = recommend(world, prof, config=CFG.with_overrides(k=12))
    assert res.n_excluded == 0 and "F" not in res.paper_ids and len(res.items) > 0
    assert all(i.contributions["citation"] == 0.0 for i in res.items)  # BC = CC = 0 without authored papers
    world.encoder = None
    with pytest.raises(ValueError, match="encoder"):
        build_user_profile(world, "z2", seed_topics=["x"])


def test_profile_embeds_authored_papers_missing_from_the_store(world):
    world.db.insert_papers_batch([{**_paper_row("U3", 2023, "Title U3"), "abstract": ""}])  # title-only, not in the store
    prof = build_user_profile(world, "u3", authored_ids=["U1", "U3"], config=CFG)
    assert prof.persona.n_embedded_on_the_fly == 1 and prof.persona.paper_ids == ["U1", "U3"]


def test_facets_feed_the_score_when_available(world, tmp_path):
    fstore = FacetStore(world.db.db_path)

    def f(pid, ftype, cid, sub, text):
        i = text.index(sub)
        return {"paper_id": _uri(pid), "facet_type": ftype, "facet_value": sub, "canonical_id": cid,
                "evidence_sentence": text, "start_char": i, "end_char": i + len(sub), "confidence": 0.9}

    fstore.save_facets([
        f("U1", "task", "t1", "information retrieval", IR_TEXT["U1"]), f("U2", "task", "t1", "information retrieval", IR_TEXT["U2"]),
        f("I1", "task", "t1", "passage retrieval", IR_TEXT["I1"]),
    ])
    fstore.save_canonical_clusters([{"canonical_id": "t1", "facet_type": "task", "canonical_name": "information retrieval"}])
    world.facet_store = fstore
    prof = _profile(world)
    assert prof.user_facets.ids == {"t1"} and prof.user_facets.freq("t1") == 1.0
    res = recommend(world, prof, config=CFG.with_overrides(k=12))
    by_id = {i.paper_id: i for i in res.items}
    assert by_id["I1"].contributions["facet"] > 0 and by_id["I2"].contributions["facet"] == 0
    assert "information retrieval" in res.pseudo_query  # facet terms enter the lexical query


def test_output_guard_rejects_forbidden_papers(world):
    _check_output(["I1", "I2"], world, {"U1"})
    with pytest.raises(RuntimeError, match="excluded"):
        _check_output(["I1", "U1"], world, {"U1"})
    with pytest.raises(RuntimeError, match="ineligible"):
        _check_output(["F"], world, set())
    with pytest.raises(RuntimeError, match="duplicate"):
        _check_output(["I1", "I1"], world, set())


def test_recommend_config_from_defaults_and_overrides():
    from paperfeed.common.config import load_config

    cfg = RecommendConfig.from_config(load_config())
    assert (cfg.k, cfg.lambda_div, cfg.rho, cfg.n_terms, cfg.n_titles) == (10, 0.2, 0.3, 10, 3)
    assert (cfg.channel_depth, cfg.pool_size, cfg.alpha_sess, cfg.use_gate, cfg.gamma_rec) == (200, 100, 0.5, False, 0.9)
    assert cfg.weights == ScoringWeights()
    changed = cfg.with_overrides(k=5, weights={"w_nov": 0.0}, use_gate=True)
    assert changed.k == 5 and changed.weights.w_nov == 0.0 and changed.weights.w_cite == 0.5 and cfg.k == 10
    session = changed.make_session(dim=2)
    assert session.use_gate and session.alpha_sess == 0.5


def test_result_is_json_serialisable(world):
    res = recommend(world, _profile(world), config=CFG)
    payload = json.loads(json.dumps(res.to_dict()))
    assert payload["mode"] == "feed" and len(payload["items"]) == len(res.items)
    assert payload["items"][0]["explanation"] == res.items[0].explanation.explanation.text


def test_unpenalised_paper_has_positive_zero_penalty_contribution():
    (s,) = score_candidates([_feat("A", penalty=0.0)], ScoringWeights(lambda_neg=0.5))
    assert str(s.contributions["penalty"]) == "0.0"  # not "-0.0", which would leak into JSON run logs
