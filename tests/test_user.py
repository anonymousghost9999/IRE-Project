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
