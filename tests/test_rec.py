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
