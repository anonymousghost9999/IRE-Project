"""Phase 11 comparison systems: RP3β on a toy citation graph (plan 11b)."""

import numpy as np
import pytest

pytest.importorskip("scipy")

from paperfeed.eval.sota import CitationGraph, RP3Author, rp3beta_recommend, tune_rp3beta

# The author wrote U, which cites R1 and R2. Citing papers C1 (R1, X, Y), C2 (R1, R2, X), C3 (R2, Z) and P1..P4 (Y).
# Walk from {R1, R2}: X is reached through C1 and C2, Y through C1, Z through C3. Y is popular (cited by 5 papers).
EDGES = [("U", "R1"), ("U", "R2"), ("C1", "R1"), ("C1", "X"), ("C1", "Y"), ("C2", "R1"), ("C2", "R2"), ("C2", "X"),
         ("C3", "R2"), ("C3", "Z")] + [(f"P{i}", "Y") for i in range(4)]


@pytest.fixture
def graph():
    return CitationGraph.from_edges([(f"https://openalex.org/{a}", b) for a, b in EDGES] + [("C1", "X")])  # dup edge


def _score(graph, pid, alpha, beta):
    s = graph.walk_scores(graph.references(["U"]), alpha)
    i = graph.index[pid]
    return s[i] / graph.indeg[i] ** beta


def test_graph_counts_and_profile(graph):
    assert graph.A.nnz == len(EDGES)  # IDs are cleaned and the duplicate edge counts once
    assert graph.indeg[graph.index["Y"]] == 5 and graph.outdeg[graph.index["C1"]] == 3
    assert [graph.ids[i] for i in graph.references(["U"])] == ["R1", "R2"]
    assert graph.references(["unknown"]) == []


def test_walk_matches_the_formula(graph):
    # alpha = 1: indeg(R1) = 3 (U, C1, C2), indeg(R2) = 3 (U, C2, C3); outdeg C1 = 3, C2 = 3, C3 = 2, U = 2
    # X: via C1 from R1: 1/3 * 1/3; via C2 from R1 and R2: (1/3 + 1/3) * 1/3  -> 1/9 + 2/9 = 1/3
    assert _score(graph, "X", 1.0, 0.0) == pytest.approx(1 / 3)
    # Z: via C3 from R2: 1/3 * 1/2
    assert _score(graph, "Z", 1.0, 0.0) == pytest.approx(1 / 6)
    # Y: via C1 from R1: 1/3 * 1/3; beta = 0.5 divides by sqrt(5)
    assert _score(graph, "Y", 1.0, 0.5) == pytest.approx((1 / 9) / np.sqrt(5))
    # alpha raises every transition probability: Z with alpha = 2 -> (1/3)^2 * (1/2)^2
    assert _score(graph, "Z", 2.0, 0.0) == pytest.approx(1 / 36)


def test_recommend_filters_and_popularity_penalty(graph):
    allowed = {"X", "Y", "Z", "R1", "R2", "C1"}
    excluded = {"U", "R1", "R2"}  # authored and cited by the author (E_u)
    recs = rp3beta_recommend(graph, ["U"], allowed, excluded, alpha=1.0, beta=0.0, k=10)
    assert recs[0] == "X" and set(recs) == {"X", "Y", "Z"}  # C1 has no incoming walk; E_u removed
    assert rp3beta_recommend(graph, ["U"], allowed - {"X"}, excluded, 1.0, 0.0, k=1) == ["Z"]  # Z 1/6 > Y 1/9
    assert rp3beta_recommend(graph, ["U"], allowed, excluded, 1.0, 0.75, k=3)[-1] == "Y"  # popularity penalty
    assert rp3beta_recommend(graph, [], allowed, excluded, 1.0, 0.0) == []  # zero-publication: not applicable


def test_tuning_picks_the_best_grid_point_with_ties_to_the_earlier(graph):
    allowed = {"X", "Y", "Z"}
    au = RP3Author("A1", ["U"], {"U", "R1", "R2"}, targets=["Z"])
    res = tune_rp3beta(graph, [au], allowed, grid={"alpha": [1.0, 2.0], "beta": [0.0, 0.75]}, k=1)
    assert res["n_users"] == 1 and len(res["points"]) == 4
    # alpha 1: X (1/3) beats Z (1/6). alpha 2, beta 0.75: X = (1/27) / 2^0.75 ~ 0.022 < Z = 1/36 ~ 0.028, so Z ranks first
    assert res["best"] == {"alpha": 2.0, "beta": 0.75, "mean_ndcg": 1.0}
    nobody = RP3Author("A1", ["U"], {"U", "R1", "R2"}, targets=["never-reached"])
    tied = tune_rp3beta(graph, [nobody], allowed, grid={"alpha": [1.0, 2.0], "beta": [0.0, 0.75]}, k=1)
    assert tied["best"] == {"alpha": 1.0, "beta": 0.0, "mean_ndcg": 0.0}  # all equal: the first grid point


# ---------------------------------------------------------------------------
# Scholar Inbox-style classifier and centroid kNN (plan 11c)
# ---------------------------------------------------------------------------

from paperfeed.eval.sota import author_seed, centroid_recommend, fit_pca_space, scholar_inbox_recommend


def _space(n=400, d=32, seed=0):
    """Papers on a 'topic axis': ids T* lie along +axis0 (the author's topic), O* elsewhere."""
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, d)).astype(np.float32)
    ids = [f"T{i}" if i < 60 else f"O{i}" for i in range(n)]
    vecs[:60, 0] += 6.0
    return ids, vecs


def test_pca_space_and_classifier_find_the_authors_topic():
    ids, vecs = _space()
    space = fit_pca_space(ids, vecs, dim=8)
    assert space.Z.shape == (400, 8) and 0 < space.explained_variance <= 1
    excluded = {f"T{i}" for i in range(10)} | {"O100"}  # E_u: authored + cited (positives); never recommended
    allowed = np.arange(400)
    recs = scholar_inbox_recommend(space, "A1", excluded, allowed, C=1.0, k=20)
    assert not set(recs) & excluded and len(recs) == 20
    assert sum(r.startswith("T") for r in recs) >= 15  # the remaining topic papers rank first
    assert recs == scholar_inbox_recommend(space, "A1", excluded, allowed, C=1.0, k=20)  # seeded: deterministic
    assert scholar_inbox_recommend(space, "A1", set(), allowed, C=1.0) == []  # no positives: not applicable
    only = np.array([space.index["O200"], space.index["T50"]])
    assert set(scholar_inbox_recommend(space, "A1", excluded, only, C=1.0, k=5)) == {"O200", "T50"}  # eligibility
    assert author_seed("https://openalex.org/A1") == author_seed("A1")


def test_centroid_knn_embeds_missing_authored_papers():
    from paperfeed.index.dense import DenseIndex
    from paperfeed.index.store import EmbeddingStore

    ids = ["P1", "P2", "P3", "P4"]
    vecs = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    store, dense = EmbeddingStore(ids, vecs), DenseIndex(dimension=2)
    dense.add_documents(ids, vecs)
    assert centroid_recommend(store, dense, ["P1"], {"P1"}, set(ids), k=1) == ["P2"]
    assert centroid_recommend(store, dense, ["NEW"], set(), set(ids), k=1) == []  # no vector, no encoder

    class Enc:
        def embed_papers(self, papers):
            return np.array([[0.0, 1.0]], dtype=np.float32)

    texts = {"NEW": {"title": "t", "abstract": "a"}}
    assert centroid_recommend(store, dense, ["NEW"], {"P3"}, set(ids), k=1, encoder=Enc(), texts=texts) == ["P4"]
