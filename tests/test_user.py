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
