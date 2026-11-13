"""V1 encoder study on the tuning split (plan 10.1c): subset rule, per-encoder resources, cached vectors, decision."""

from types import SimpleNamespace

import numpy as np
import pytest

from paperfeed.data.db import DatabaseManager
from paperfeed.eval.encoder_study import (
    build_subset_lexical,
    decide_encoder,
    embed_subset,
    graph_subset,
    paper_texts,
    subset_fingerprint,
    subset_resources,
)
from paperfeed.eval.tuning import CachedUserCandidateData, GridPoint
from paperfeed.index.store import EmbeddingStore
from paperfeed.rec.scoring import ScoringWeights


def _uri(pid):
    return f"https://openalex.org/{pid}"


def _paper(pid, year=2022):
    return {
        "paper_id": _uri(pid), "title": f"Title {pid}", "abstract": f"About {pid}", "authors": "[]", "year": year,
        "publication_date": None, "venue": None, "doi": None, "url": None, "citation_count_T": 0,
        "primary_topic_id": None, "subfield_id": None, "field_id": None, "in_candidates": 1,
    }


@pytest.fixture
def graph_db(tmp_path):
    """Seeds S1, S2. One hop: A (2 links), B (1), C (1, cites S1), Z (2024 citer). Two hop: D (links to A and B), E (to A).
    F cites S1 but only in 2025 (after T): never a neighbour."""
    db = DatabaseManager(db_path=tmp_path / "g.sqlite3", cutoff_year=2024)
    ids = ["S1", "S2", "A", "B", "C", "D", "E", "F", "Z"]
    db.insert_papers_batch([_paper(i) for i in ids])
    edges = [("S1", "A"), ("S2", "A"), ("S1", "B"), ("C", "S1"), ("D", "A"), ("D", "B"), ("E", "A")]
    db.insert_citations_batch([(_uri(a), _uri(b), 2022) for a, b in edges] + [(_uri("Z"), _uri("S2"), 2024)])
    with db.get_connection() as conn:
        conn.execute("INSERT INTO paper_citations (source_paper_id, target_paper_id, source_year) VALUES (?, ?, 2025)",
                     (_uri("F"), _uri("S1")))
    return db, set(ids)


def test_subset_rule_seeds_then_one_hop_then_two_hop(graph_db):
    db, allowed = graph_db
    ids, info = graph_subset(db, ["S1", "S2"], allowed, 2024, size=6)
    assert ids[:2] == ["S1", "S2"]
    assert ids[2] == "A"  # two links to the seeds
    assert set(ids[3:6]) == {"B", "C", "Z"} and ids[3:6] == ["B", "C", "Z"]  # one link each, ties by ID
    assert info == {"size": 6, "n_seeds": 2, "n_hop1": 4, "n_hop1_available": 4, "n_hop2": 0, "rule": "design §13.8 V1 point 4"}
    full, info = graph_subset(db, ["S1", "S2"], allowed, 2024, size=8)
    assert full[6:] == ["D", "E"] and info["n_hop2"] == 2  # D links to two chosen papers, E to one
    assert "F" not in full  # its citation is from after T
    assert graph_subset(db, ["S1", "S2"], allowed, 2024, size=8) == (full, info)  # deterministic


def test_subset_respects_allowed_papers_and_size(graph_db):
    db, allowed = graph_db
    ids, _ = graph_subset(db, ["S1", "S2"], allowed - {"A"}, 2024, size=4)
    assert "A" not in ids and len(ids) == 4
    ids, _ = graph_subset(db, ["S1", "S2"], allowed, 2024, size=3)
    assert ids == ["S1", "S2", "A"]
    with pytest.raises(ValueError, match="do not fit"):
        graph_subset(db, ["S1", "S2"], allowed, 2024, size=1)


class _CountingEncoder:
    def __init__(self, seed=0):
        self.n, self.seed = 0, seed

    def embed_papers(self, papers, batch_size=32, show_progress=False):
        self.n += len(papers)
        rng = np.random.default_rng(self.seed)
        v = rng.normal(size=(len(papers), 4)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_subset_vectors_are_cached_per_subset_and_model(graph_db, tmp_path):
    db, _ = graph_db
    texts = paper_texts(db, ["A", "B", "C"])
    assert texts["A"] == {"title": "Title A", "abstract": "About A"}
    enc = _CountingEncoder()
    s1 = embed_subset(enc, "m1", texts, tmp_path / "c", checkpoint_every=2)
    assert enc.n == 3 and len(s1) == 3
    again = _CountingEncoder()
    s2 = embed_subset(again, "m1", texts, tmp_path / "c")
    assert again.n == 0 and np.allclose(s1.select(["A", "C"]), s2.select(["A", "C"]))  # resumed from the cache
    other = _CountingEncoder()
    embed_subset(other, "m2", texts, tmp_path / "c")
    assert other.n == 3  # another model: the cache is discarded
    assert subset_fingerprint(["B", "A"]) == subset_fingerprint(["A", "B"])


def test_subset_resources_restrict_candidates_and_use_the_encoders_vectors(graph_db):
    db, _ = graph_db
    ids = ["A", "B", "C"]
    texts = paper_texts(db, ids)
    store = EmbeddingStore(ids, np.eye(3, 4, dtype=np.float32))
    retriever = SimpleNamespace(channel_depth=200, rrf_k=60, candidate_pool_size=100)
    base = SimpleNamespace(retriever=retriever, eligible={"A", "B", "X"}, store=None, encoder=None)
    from dataclasses import dataclass

    @dataclass
    class Res:
        retriever: object
        eligible: set
        store: object
        encoder: object

    r = subset_resources(Res(**vars(base)), store, build_subset_lexical(texts), "enc", ids)
    assert r.eligible == {"A", "B"} and r.encoder == "enc" and r.store is store
    assert r.retriever.dense_index.search(np.array([0, 1, 0, 0], dtype=np.float32), k=1)[0][0] == "B"
    assert r.retriever.lexical_index.search("About C", k=1)[0][0] == "C"
    assert (r.retriever.channel_depth, r.retriever.candidate_pool_size) == (200, 100)


def _arm(ndcg, centroid, n=30):
    users = [CachedUserCandidateData(f"A{i}", None, ["t"], [], ["t"]) for i in range(n)]
    pt = GridPoint({"w_cite": 1.0, "w_facet": 0.0, "w_nov": 0.0, "w_pop": 0.0, "lambda_div": 0.0}, ScoringWeights(), 0.0,
                   np.asarray(ndcg, dtype=float))
    return {"cached": users, "points": [pt], "dense_centroid": np.asarray(centroid, dtype=float)}


def test_encoder_decision_picks_the_better_alternative_and_applies_the_ci_rule():
    n = 30
    arms = {
        "specter2": _arm([0.2] * n, [0.1] * n),
        "scincl": _arm([0.21] * n, [0.1] * n),
        "bge": _arm([0.3 + 0.01 * (i % 3) for i in range(n)], [0.15] * n),
    }
    d = decide_encoder(arms, n_resamples=500)
    assert d["chosen_encoder"] == "bge" and d["adopt_v1"] is True and d["ndcg_gain"]["ci"][0] > 0
    assert d["per_encoder"]["bge"]["candidate_recall_at_100"] == 1.0
    assert d["dense_centroid_gain"]["bge"]["mean_diff"] == pytest.approx(0.05)

    lucky = [0.2] * n
    lucky[0] = 0.9
    d = decide_encoder({"specter2": _arm([0.2] * n, [0.1] * n), "bge": _arm(lucky, [0.1] * n)}, n_resamples=500)
    assert d["adopt_v1"] is False  # one lucky author is not a reliable gain
    with pytest.raises(ValueError):
        decide_encoder({"bge": _arm([0.2] * n, [0.1] * n)})
