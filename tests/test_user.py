"""Unit tests for Phase 4 User Model (docs/plan.md §4, docs/design.md §7)."""

import numpy as np
import pytest

from paperfeed.index.store import EmbeddingStore
from paperfeed.user.persona import (
    build_persona,
    build_seed_persona,
    load_authored_papers,
    persona_vector,
    recency_weights,
    seed_topic_persona,
)


# ---------------------------------------------------------------------------
# 4.1 Persona vector
# ---------------------------------------------------------------------------

def test_recency_weights_hand_computed():
    w = recency_weights([2024, 2023, 2022], cutoff_year=2024, gamma_rec=0.9)
    np.testing.assert_allclose(w, [1.0, 0.9, 0.81])


def test_recency_weights_reject_papers_after_cutoff():
    with pytest.raises(ValueError, match="year > T"):
        recency_weights([2024, 2025], cutoff_year=2024)


def test_persona_vector_is_weighted_sum_then_normalised():
    # Orthogonal unit vectors: the persona direction is exactly the weights.
    emb = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    vec = persona_vector(emb, [2024, 2023], cutoff_year=2024, gamma_rec=0.9)
    expected = np.array([1.0, 0.9, 0.0])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(vec, expected, atol=1e-6)
    assert np.linalg.norm(vec) == pytest.approx(1.0)


def test_persona_vector_recent_paper_dominates():
    emb = np.array([[1, 0], [0, 1]], dtype=np.float32)
    vec = persona_vector(emb, [2024, 2014], cutoff_year=2024, gamma_rec=0.9)
    assert vec[0] > vec[1]


def test_persona_vector_gamma_one_is_plain_mean_direction():
    emb = np.array([[1, 0], [0, 1]], dtype=np.float32)
    vec = persona_vector(emb, [2024, 2010], cutoff_year=2024, gamma_rec=1.0)
    np.testing.assert_allclose(vec, [2 ** -0.5, 2 ** -0.5], atol=1e-6)


def test_persona_vector_input_validation():
    with pytest.raises(ValueError):
        persona_vector(np.empty((0, 3)), [], cutoff_year=2024)
    with pytest.raises(ValueError, match="mismatch"):
        persona_vector(np.ones((2, 3)), [2024], cutoff_year=2024)
    # Equal and opposite vectors with equal weights cancel to zero
    with pytest.raises(ValueError, match="zero vector"):
        persona_vector(np.array([[1, 0], [-1, 0]]), [2024, 2024], cutoff_year=2024)


def test_seed_topic_persona_is_unweighted_mean_normalised():
    seeds = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    vec = seed_topic_persona(seeds)
    mean = seeds.mean(axis=0)
    np.testing.assert_allclose(vec, mean / np.linalg.norm(mean), atol=1e-6)
    with pytest.raises(ValueError):
        seed_topic_persona(np.empty((0, 2)))


class FakeEncoder:
    """Deterministic stand-in for SPECTER2Encoder: the vector encodes whether an abstract was used."""

    def __init__(self):
        self.papers_seen = []
        self.queries_seen = []

    def embed_papers(self, papers, batch_size=32, show_progress=False):
        self.papers_seen.extend(papers)
        return np.array([[0, 1] if p.get("abstract") else [0, -1] for p in papers], dtype=np.float32)

    def embed_queries(self, queries, batch_size=32):
        self.queries_seen.extend(queries)
        return np.array([[1, 0] for _ in queries], dtype=np.float32)


def _papers():
    return [
        {"paper_id": "W1", "title": "A", "abstract": "abs", "year": 2024},
        {"paper_id": "W2", "title": "B", "abstract": "", "year": 2024},  # title-only, not in store
    ]


def test_build_persona_uses_stored_vectors_and_embeds_the_rest():
    store = EmbeddingStore(["W1"], np.array([[1, 0]], dtype=np.float32))
    enc = FakeEncoder()
    persona = build_persona(_papers(), store, enc, cutoff_year=2024)

    assert persona.source == "authored"
    assert persona.paper_ids == ["W1", "W2"]
    assert persona.n_embedded_on_the_fly == 1
    assert [p["paper_id"] for p in enc.papers_seen] == ["W2"]  # only the paper without a stored vector
    # W1 stored [1,0] + W2 title-only [0,-1], equal weights
    np.testing.assert_allclose(persona.vector, [2 ** -0.5, -(2 ** -0.5)], atol=1e-6)


def test_build_persona_without_encoder_needs_all_vectors_stored():
    store = EmbeddingStore(["W1"], np.array([[1, 0]], dtype=np.float32))
    with pytest.raises(ValueError, match="no encoder"):
        build_persona(_papers(), store, None, cutoff_year=2024)
    persona = build_persona(_papers()[:1], store, None, cutoff_year=2024)
    np.testing.assert_allclose(persona.vector, [1, 0])


def test_build_persona_skips_empty_papers_and_rejects_none_left():
    store = EmbeddingStore(["W1"], np.array([[1, 0]], dtype=np.float32))
    papers = _papers()[:1] + [{"paper_id": "W9", "title": " ", "abstract": "", "year": 2024}]
    persona = build_persona(papers, store, None, cutoff_year=2024)
    assert persona.paper_ids == ["W1"]
    with pytest.raises(ValueError, match="No authored papers"):
        build_persona([], store, None, cutoff_year=2024)


def test_build_seed_persona_uses_query_adapter():
    enc = FakeEncoder()
    persona = build_seed_persona(["dense retrieval", " ", "recommender systems"], enc)
    assert persona.source == "seed_topics" and persona.paper_ids == []
    assert enc.queries_seen == ["dense retrieval", "recommender systems"]
    np.testing.assert_allclose(persona.vector, [1, 0])
    with pytest.raises(ValueError):
        build_seed_persona([], enc)


def test_load_authored_papers_enforces_cutoff(tmp_path):
    from paperfeed.data.db import DatabaseManager

    db = DatabaseManager(db_path=tmp_path / "t.sqlite3", cutoff_year=2024)

    def row(pid, year, abstract="x"):
        return {
            "paper_id": f"https://openalex.org/{pid}", "title": pid, "abstract": abstract, "authors": "[]", "year": year,
            "publication_date": None, "venue": None, "doi": None, "url": None, "citation_count_T": 0,
            "primary_topic_id": None, "subfield_id": None, "field_id": None, "in_candidates": 1,
        }

    db.insert_papers_batch([row("W1", 2023), row("W2", 2024, abstract=None), row("W3", 2025)])

    papers = load_authored_papers(db, ["W1", "https://openalex.org/W2", "W404"], cutoff_year=2024)
    assert [p["paper_id"] for p in papers] == ["W1", "W2"]  # W404 absent from the DB: skipped
    assert papers[1]["abstract"] == ""

    with pytest.raises(ValueError, match="year > T"):
        load_authored_papers(db, ["W1", "W3"], cutoff_year=2024)


# ---------------------------------------------------------------------------
# 4.2 User facet set
# ---------------------------------------------------------------------------

from paperfeed.data.db import DatabaseManager
from paperfeed.facets.store import FacetStore
from paperfeed.index.dense import DenseIndex
from paperfeed.user.facets import UserFacets, build_user_facets, build_zero_pub_facets


@pytest.fixture
def facet_store(tmp_path):
    """Three papers with canonical facets; W3 has none and W4 is only reachable through the dense index."""
    db = DatabaseManager(db_path=tmp_path / "f.sqlite3", cutoff_year=2024)

    def paper(pid):
        return {
            "paper_id": f"https://openalex.org/{pid}", "title": pid, "abstract": "x", "authors": "[]", "year": 2023,
            "publication_date": None, "venue": None, "doi": None, "url": None, "citation_count_T": 0,
            "primary_topic_id": None, "subfield_id": None, "field_id": None, "in_candidates": 1,
        }

    db.insert_papers_batch([paper(p) for p in ("W1", "W2", "W3", "W4")])
    store = FacetStore(db.db_path)

    def facet(pid, ftype, value, canon):
        return {
            "paper_id": f"https://openalex.org/{pid}", "facet_type": ftype, "facet_value": value, "canonical_id": canon,
            "evidence_sentence": "s", "start_char": 0, "end_char": 1, "confidence": 0.9,
        }

    store.save_facets([
        facet("W1", "task", "retrieval", "canon_task_a"),
        facet("W1", "task", "Retrieval", "canon_task_a"),  # same canonical facet twice in one paper
        facet("W1", "method", "BM25", "canon_method_b"),
        facet("W1", "metric", "nDCG", "canon_metric_c"),  # not a core type
        facet("W2", "task", "document retrieval", "canon_task_a"),
        facet("W2", "dataset", "MS MARCO", "canon_dataset_d"),
        facet("W2", "method", "uncanonicalized", None),
        facet("W4", "method", "BM25", "canon_method_b"),
    ])
    store.save_canonical_clusters([
        {"canonical_id": "canon_task_a", "facet_type": "task", "canonical_name": "document retrieval"},
        {"canonical_id": "canon_method_b", "facet_type": "method", "canonical_name": "BM25"},
        {"canonical_id": "canon_metric_c", "facet_type": "metric", "canonical_name": "nDCG"},
        {"canonical_id": "canon_dataset_d", "facet_type": "dataset", "canonical_name": "MS MARCO"},
    ])
    return store


def test_build_user_facets_counts_papers_and_keeps_core_types_only(facet_store):
    uf = build_user_facets(["W1", "https://openalex.org/W2", "W3"], facet_store)
    assert uf.ids == {"canon_task_a", "canon_method_b", "canon_dataset_d"}
    assert uf.counts["canon_task_a"] == 2  # two papers, counted once each despite two mentions in W1
    assert uf.counts["canon_method_b"] == 1
    assert uf.types["canon_dataset_d"] == "dataset"
    assert uf.names["canon_task_a"] == "document retrieval"
    assert uf.paper_ids == ["W1", "W2", "W3"] and uf.source == "authored"


def test_build_user_facets_ignores_uncanonicalized_and_empty_input(facet_store):
    assert len(build_user_facets([], facet_store)) == 0
    assert len(build_user_facets(["W3"], facet_store)) == 0
    # all facet types when asked
    uf = build_user_facets(["W1"], facet_store, facet_types=("task", "method", "dataset", "metric"))
    assert "canon_metric_c" in uf.ids


def test_user_facets_top_is_deterministic_and_filterable(facet_store):
    uf = build_user_facets(["W1", "W2"], facet_store)
    assert uf.top() == [("canon_task_a", 2), ("canon_dataset_d", 1), ("canon_method_b", 1)]  # ties by ID
    assert uf.top(1) == [("canon_task_a", 2)]
    assert uf.top_names(facet_type="method") == ["BM25"]


def test_canonical_id_without_cluster_row_is_dropped(facet_store):
    facet_store.save_facets([{
        "paper_id": "https://openalex.org/W3", "facet_type": "task", "facet_value": "x", "canonical_id": "canon_task_ghost",
        "evidence_sentence": "s", "start_char": 0, "end_char": 1, "confidence": 0.9,
    }])
    assert "canon_task_ghost" not in build_user_facets(["W3"], facet_store).ids


def test_zero_pub_facets_come_from_nearest_papers(facet_store):
    index = DenseIndex(dimension=2)
    index.add_documents(
        ["W1", "W2", "W4"],
        np.array([[0, 1], [0.1, 1], [1, 0]], dtype=np.float32),  # W4 is far from the persona
    )
    persona = np.array([0, 1], dtype=np.float32)

    uf = build_zero_pub_facets(persona, index, facet_store, k=2)
    assert uf.source == "nearest_papers"
    assert sorted(uf.paper_ids) == ["W1", "W2"]  # W4 excluded: not among the 2 nearest
    assert uf.ids == {"canon_task_a", "canon_method_b", "canon_dataset_d"}
    assert uf.counts["canon_method_b"] == 1  # from W1 only, W4's BM25 not counted

    assert len(build_zero_pub_facets(persona, DenseIndex(dimension=2), facet_store)) == 0


# ---------------------------------------------------------------------------
# 4.3 Exclusion set
# ---------------------------------------------------------------------------

from paperfeed.user.exclusion import build_exclusion_sources, load_exclusion_sources


@pytest.fixture
def sources():
    return build_exclusion_sources(
        authored_ids=["W1", "https://openalex.org/W2"],
        citations=["W10", "https://openalex.org/W11", "W1"],  # W1: an author citing their own paper
    )


def test_exclusion_standard_mode_is_union_of_all_three_sources(sources):
    assert sources.build(session_consumed=["W20"]) == {"W1", "W2", "W10", "W11", "W20"}


def test_exclusion_diagnostic_mode_drops_citations_only(sources):
    excluded = sources.build(session_consumed=["https://openalex.org/W20"], diagnostic=True)
    assert excluded == {"W1", "W2", "W20"}  # authored and session-consumed stay, W10/W11 become eligible
    assert "W10" not in excluded and "W11" not in excluded


def test_exclusion_zero_pub_mode_is_empty(sources):
    assert sources.build(zero_pub=True) == set()
    assert sources.build(session_consumed=["W20"], zero_pub=True) == set()
    assert sources.build(diagnostic=True, zero_pub=True) == set()  # zero_pub wins


def test_exclusion_session_consumed_grows_without_mutating_sources(sources):
    first = sources.build()
    second = sources.build(session_consumed=["W20"])
    assert "W20" not in first and "W20" in second
    assert sources.authored == {"W1", "W2"}


def test_exclusion_empty_user(sources):
    empty = build_exclusion_sources([], [])
    assert empty.build() == set()
    assert empty.build(session_consumed=["W5"]) == {"W5"}


def test_load_exclusion_sources_reads_citations_from_db(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "e.sqlite3", cutoff_year=2024)

    def paper(pid, year):
        return {
            "paper_id": f"https://openalex.org/{pid}", "title": pid, "abstract": "x", "authors": "[]", "year": year,
            "publication_date": None, "venue": None, "doi": None, "url": None, "citation_count_T": 0,
            "primary_topic_id": None, "subfield_id": None, "field_id": None, "in_candidates": 1,
        }

    db.insert_papers_batch([paper("W1", 2023), paper("W2", 2022), paper("W3", 2021)])
    db.insert_citations_batch([
        ("https://openalex.org/W1", "https://openalex.org/W10", 2023),
        ("https://openalex.org/W1", "https://openalex.org/W11", 2023),
        ("https://openalex.org/W2", "https://openalex.org/W10", 2022),  # shared reference
        ("https://openalex.org/W3", "https://openalex.org/W99", 2021),  # W3 is not authored by this user
        ("https://openalex.org/W1", "https://openalex.org/W12", 2025),  # post-T edge: rejected by the DB layer
    ])

    src = load_exclusion_sources(db, ["W1", "W2"])
    assert src.authored == {"W1", "W2"}
    assert src.citations == {"W10", "W11"}
    assert src.build() == {"W1", "W2", "W10", "W11"}
    assert src.build(diagnostic=True) == {"W1", "W2"}


# ---------------------------------------------------------------------------
# 4.4 Session state
# ---------------------------------------------------------------------------

from paperfeed.user.session import SessionState


def _e(*xs):
    return np.array(xs, dtype=np.float32)


def test_session_starts_empty():
    s = SessionState(dim=2)
    assert s.is_empty
    np.testing.assert_array_equal(s.vector, [0, 0])
    assert s.negatives == set() and s.consumed == set() and s.recent_positives() == []


def test_session_ema_update_hand_computed():
    s = SessionState(alpha_sess=0.5, dim=2)
    s.click("W1", _e(1, 0))
    np.testing.assert_allclose(s.vector, [0.5, 0.0])  # 0.5 * 0 + 0.5 * e1
    s.click("W2", _e(0, 1))
    np.testing.assert_allclose(s.vector, [0.25, 0.5])  # 0.5 * [0.5, 0] + 0.5 * e2
    assert not s.is_empty


def test_session_ema_equals_exponentially_decayed_sum():
    alpha = 0.7
    embs = [_e(1, 0), _e(0, 1), _e(1, 1), _e(-1, 0.5)]
    s = SessionState(alpha_sess=alpha, dim=2)
    for i, e in enumerate(embs):
        s.click(f"W{i}", e)
    t = len(embs)
    expected = sum((1 - alpha) * alpha ** (t - 1 - i) * e for i, e in enumerate(embs))  # newest item weighs most
    np.testing.assert_allclose(s.vector, expected, atol=1e-6)


def test_session_alpha_extremes():
    keep = SessionState(alpha_sess=1.0, dim=2)  # alpha = 1: the vector never moves off zero
    keep.click("W1", _e(1, 0))
    np.testing.assert_array_equal(keep.vector, [0, 0])
    latest = SessionState(alpha_sess=0.0, dim=2)  # alpha = 0: the vector is the last clicked embedding
    latest.click("W1", _e(1, 0))
    latest.click("W2", _e(0, 1))
    np.testing.assert_allclose(latest.vector, [0, 1])


def test_save_adds_to_consumed_but_click_does_not():
    s = SessionState(dim=2)
    s.click("W1", _e(1, 0))
    s.save("W2", _e(0, 1))
    assert s.consumed == {"W2"}
    assert s.recent_positives() == ["W2", "W1"]  # both moved the vector; most recent first
    np.testing.assert_allclose(s.vector, [0.25, 0.5])  # save follows the same EMA as click


def test_skip_adds_negative_and_leaves_vector_alone():
    s = SessionState(dim=2)
    s.click("W1", _e(1, 0))
    before = s.vector
    s.skip("W9")
    assert s.negatives == {"W9"} and s.consumed == set()
    np.testing.assert_array_equal(s.vector, before)
    assert s.recent_positives() == ["W1"]


def test_recent_positives_limit():
    s = SessionState(dim=2)
    for i in range(5):
        s.click(f"W{i}", _e(1, 0))
    assert s.recent_positives(3) == ["W4", "W3", "W2"]


def test_pivot_gate_resets_on_divergent_click():
    s = SessionState(alpha_sess=0.5, use_gate=True, gamma_pivot=0.35, dim=2)
    s.click("W1", _e(1, 0))
    s.click("W2", _e(0, 1))  # cos = 0, delta = 1 > 0.35: hard reset to the clicked embedding
    np.testing.assert_allclose(s.vector, [0, 1])
    assert s.n_gate_resets == 1


def test_pivot_gate_keeps_blending_for_similar_click():
    s = SessionState(alpha_sess=0.5, use_gate=True, gamma_pivot=0.35, dim=2)
    s.click("W1", _e(1, 0))
    s.click("W2", _e(1, 0.1))  # cos ~ 0.995, delta ~ 0.005 < 0.35: normal EMA
    np.testing.assert_allclose(s.vector, 0.5 * np.array([0.5, 0]) + 0.5 * np.array([1, 0.1]), atol=1e-6)
    assert s.n_gate_resets == 0


def test_pivot_gate_threshold_is_strict_and_first_click_is_not_gated():
    # delta exactly equal to gamma is not > gamma, so no reset
    s = SessionState(alpha_sess=0.5, use_gate=True, gamma_pivot=1.0, dim=2)
    s.click("W1", _e(1, 0))
    s.click("W2", _e(0, 1))  # delta = 1.0, not > 1.0
    np.testing.assert_allclose(s.vector, [0.25, 0.5])
    # first click: nothing to diverge from
    first = SessionState(use_gate=True, gamma_pivot=0.0, dim=2)
    first.click("W1", _e(1, 0))
    assert first.n_gate_resets == 0
    np.testing.assert_allclose(first.vector, [0.5, 0])


def test_gate_off_by_default_never_resets():
    s = SessionState(alpha_sess=0.5, dim=2)
    s.click("W1", _e(1, 0))
    s.click("W2", _e(0, 1))
    np.testing.assert_allclose(s.vector, [0.25, 0.5])
    assert s.n_gate_resets == 0


def test_reset_clears_session():
    s = SessionState(dim=2)
    s.save("W1", _e(1, 0))
    s.skip("W2")
    s.reset()
    assert s.is_empty and s.negatives == set() and s.consumed == set() and s.recent_positives() == []
    np.testing.assert_array_equal(s.vector, [0, 0])
    s.click("W3", _e(0, 1))  # usable again after a reset
    np.testing.assert_allclose(s.vector, [0, 0.5])


def test_apply_dispatch_and_validation():
    s = SessionState(dim=2)
    s.apply("click", "W1", _e(1, 0))
    s.apply("save", "W2", _e(0, 1))
    s.apply("skip", "W3")
    assert s.consumed == {"W2"} and s.negatives == {"W3"}
    with pytest.raises(ValueError, match="needs the paper embedding"):
        s.apply("click", "W4")
    with pytest.raises(ValueError, match="Unknown action"):
        s.apply("like", "W4", _e(1, 0))
    with pytest.raises(ValueError, match="shape"):
        s.click("W5", np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError, match="alpha_sess"):
        SessionState(alpha_sess=1.5)


def test_vector_property_is_a_copy():
    s = SessionState(dim=2)
    s.click("W1", _e(1, 0))
    s.vector[:] = 99
    np.testing.assert_allclose(s.vector, [0.5, 0])


def test_session_from_config():
    cfg = {"user_model": {"alpha_session": 0.3, "use_pivot_gate": True, "gamma_pivot": 0.25}}
    s = SessionState.from_config(cfg, dim=2)
    assert (s.alpha_sess, s.use_gate, s.gamma_pivot) == (0.3, True, 0.25)
    d = SessionState.from_config({}, dim=2)
    assert (d.alpha_sess, d.use_gate, d.gamma_pivot) == (0.5, False, 0.35)


# ---------------------------------------------------------------------------
# 4.5 Pseudo-query builder
# ---------------------------------------------------------------------------

from paperfeed.user.query import (
    build_feed_query,
    build_pseudo_query,
    most_recent_titles,
    query_settings,
    titles_for,
)


@pytest.fixture
def user_facets():
    return UserFacets(
        counts={"canon_task_a": 3, "canon_method_b": 2, "canon_dataset_d": 2, "canon_task_e": 1},
        types={"canon_task_a": "task", "canon_method_b": "method", "canon_dataset_d": "dataset", "canon_task_e": "task"},
        names={
            "canon_task_a": "document retrieval", "canon_method_b": "BM25",
            "canon_dataset_d": "MS MARCO", "canon_task_e": "query expansion",
        },
    )


def test_pseudo_query_terms_most_frequent_first_then_titles(user_facets):
    q = build_pseudo_query(user_facets, ["Dense Passage Retrieval", "ColBERT"], n_terms=3)
    # ties (BM25 / MS MARCO, both 2) broken by canonical id: canon_dataset_d before canon_method_b
    assert q == "document retrieval MS MARCO BM25 Dense Passage Retrieval ColBERT"


def test_pseudo_query_length_parameter(user_facets):
    assert build_pseudo_query(user_facets, n_terms=1) == "document retrieval"
    assert build_pseudo_query(user_facets, n_terms=0, title_texts=["Only A Title"]) == "Only A Title"
    assert len(build_pseudo_query(user_facets, n_terms=100).split()) == 7  # all four facets, 2+1+2+2 words


def test_pseudo_query_empty_and_cleaning(user_facets):
    assert build_pseudo_query(UserFacets(), []) == ""
    q = build_pseudo_query(UserFacets(), ["  A Title ", "a title", "", "Untitled", "Another"])
    assert q == "A Title Another"  # whitespace stripped, duplicate / placeholder / empty titles dropped


def test_most_recent_titles_orders_by_year_then_id():
    papers = [
        {"paper_id": "W3", "title": "Old", "year": 2018},
        {"paper_id": "W2", "title": "Newer B", "year": 2023},
        {"paper_id": "W1", "title": "Newer A", "year": 2023},
        {"paper_id": "W4", "title": "Newest", "year": 2024},
    ]
    assert most_recent_titles(papers, 3) == ["Newest", "Newer A", "Newer B"]
    assert most_recent_titles([], 3) == []


@pytest.fixture
def title_db(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "q.sqlite3", cutoff_year=2024)

    def paper(pid, title):
        return {
            "paper_id": f"https://openalex.org/{pid}", "title": title, "abstract": "x", "authors": "[]", "year": 2023,
            "publication_date": None, "venue": None, "doi": None, "url": None, "citation_count_T": 0,
            "primary_topic_id": None, "subfield_id": None, "field_id": None, "in_candidates": 1,
        }

    db.insert_papers_batch([paper("W1", "First Paper"), paper("W2", "Second Paper"), paper("W3", "Third Paper")])
    return db


def test_titles_for_keeps_requested_order_and_skips_missing(title_db):
    assert titles_for(title_db, ["W3", "https://openalex.org/W1", "W404"]) == ["Third Paper", "First Paper"]
    assert titles_for(title_db, []) == []


def test_feed_query_uses_session_titles_when_session_active(user_facets, title_db):
    session = SessionState(dim=2)
    session.click("W1", _e(1, 0))
    session.save("W3", _e(0, 1))
    authored = [{"paper_id": "W2", "title": "Authored Recent", "year": 2024}]
    q = build_feed_query(user_facets, session, title_db, authored, n_terms=1)
    assert q == "document retrieval Third Paper First Paper"  # most recent session paper first, not the authored title


def test_feed_query_falls_back_to_recent_authored_when_session_empty(user_facets, title_db):
    authored = [
        {"paper_id": f"W{i}", "title": f"Authored {i}", "year": 2020 + i} for i in range(1, 6)
    ]
    expected = "document retrieval Authored 5 Authored 4 Authored 3"
    assert build_feed_query(user_facets, SessionState(dim=2), title_db, authored, n_terms=1) == expected
    assert build_feed_query(user_facets, None, title_db, authored, n_terms=1) == expected  # Track 1: no session at all


def test_feed_query_title_count_parameter(user_facets, title_db):
    authored = [{"paper_id": f"W{i}", "title": f"Authored {i}", "year": 2020 + i} for i in range(1, 6)]
    q = build_feed_query(user_facets, None, title_db, authored, n_terms=0, n_titles=1)
    assert q == "Authored 5"


def test_query_settings_from_config():
    assert query_settings({"user_model": {"pseudo_query_terms": 6, "pseudo_query_titles": 2}}) == {"n_terms": 6, "n_titles": 2}
    assert query_settings({}) == {"n_terms": 10, "n_titles": 3}
    from paperfeed.common.config import load_config
    assert query_settings(load_config()) == {"n_terms": 10, "n_titles": 3}


def test_session_positives_lists_every_click_and_save_oldest_first():
    s = SessionState(dim=2)
    s.click("W1", _e(1, 0))
    s.skip("W9")
    s.save("W2", _e(0, 1))
    s.click("W3", _e(1, 1))
    assert s.positives == ["W1", "W2", "W3"]
    s.positives.append("hack")  # a copy
    assert s.positives == ["W1", "W2", "W3"]
