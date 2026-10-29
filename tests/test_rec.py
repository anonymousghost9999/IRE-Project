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
