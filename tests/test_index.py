"""Unit tests for Phase 2 Representations and Indexing modules.

Governed by docs/plan.md §2 and docs/design.md §5, §8:
- RRF hand-computed verification
- DenseIndex (FAISS FlatIP) functionality, filtering, and persistence
- LexicalIndex (BM25s) retrieval, filtering, and persistence
- HybridRetriever fusion
- SPECTER2Encoder formatting and normalization logic
"""

from pathlib import Path
import numpy as np
import pytest

from paperfeed.index.dense import DenseIndex
from paperfeed.index.embeddings import SPECTER2Encoder
from paperfeed.index.lexical import LexicalIndex
from paperfeed.index.retriever import HybridRetriever, reciprocal_rank_fusion


def test_reciprocal_rank_fusion_hand_computed():
    """
    Unit-test RRF on a small hand-computed example (docs/plan.md §2.5).
    List 1: ["A", "B", "C"]
    List 2: ["B", "A", "D"]
    With k_rrf = 60:
    Rank 1 contributes: 1 / (60 + 1) = 1/61
    Rank 2 contributes: 1 / (60 + 2) = 1/62
    Rank 3 contributes: 1 / (60 + 3) = 1/63
    
    A: 1/61 + 1/62 ≈ 0.01639344 + 0.01612903 = 0.03252247
    B: 1/62 + 1/61 ≈ 0.03252247 (tied with A, tie-break by ID -> A before B)
    C: 1/63 ≈ 0.01587302
    D: 1/63 ≈ 0.01587302 (tied with C, tie-break by ID -> C before D)
    """
    lists = [["A", "B", "C"], ["B", "A", "D"]]
    fused = reciprocal_rank_fusion(lists, k_rrf=60)

    assert len(fused) == 4
    ids = [item[0] for item in fused]
    assert ids == ["A", "B", "C", "D"]

    score_a = 1.0 / 61.0 + 1.0 / 62.0
    score_c = 1.0 / 63.0

    assert pytest.approx(fused[0][1], rel=1e-5) == score_a
    assert pytest.approx(fused[1][1], rel=1e-5) == score_a
    assert pytest.approx(fused[2][1], rel=1e-5) == score_c
    assert pytest.approx(fused[3][1], rel=1e-5) == score_c


def test_dense_index_search_and_filters(tmp_path: Path):
    """Test FAISS IndexFlatIP top-k search, exclusion sets, and candidate filters."""
    dim = 4
    idx = DenseIndex(dimension=dim)

    # 4 orthogonal / distinct unit vectors
    pids = ["W1", "W2", "W3", "W4"]
    vecs = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.7071, 0.7071, 0.0, 0.0],
    ], dtype=np.float32)

    idx.add_documents(pids, vecs)
    assert len(idx) == 4

    # Search with vector close to W1
    q = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    hits = idx.search(q, k=2)
    assert len(hits) == 2
    assert hits[0][0] == "W1"

    # Search with exclusion set excluding W1
    hits_excluded = idx.search(q, k=2, exclusion_set={"W1"})
    result_ids = [pid for pid, _ in hits_excluded]
    assert "W1" not in result_ids
    assert result_ids[0] == "W4"

    # Search with candidate filter whitelist
    hits_filtered = idx.search(q, k=2, candidate_filter={"W2", "W3"})
    result_ids_fil = [pid for pid, _ in hits_filtered]
    assert set(result_ids_fil).issubset({"W2", "W3"})

    # Persistence roundtrip
    idx_file = tmp_path / "test.faiss"
    map_file = tmp_path / "test_map.json"
    idx.save(idx_file, map_file)

    loaded_idx = DenseIndex.load(idx_file, map_file)
    assert len(loaded_idx) == 4
    loaded_hits = loaded_idx.search(q, k=2)
    assert loaded_hits[0][0] == "W1"


def test_lexical_index_search_and_filters(tmp_path: Path):
    """Test BM25s indexing, query retrieval, exclusions, and persistence."""
    idx = LexicalIndex(stopwords="en")
    pids = ["P1", "P2", "P3"]
    docs = [
        "quantum computing and entanglement physics",
        "deep neural networks and transformer architecture",
        "graph convolutional networks and relational databases",
    ]

    idx.index_documents(pids, docs, show_progress=False)
    assert len(idx) == 3

    # Query matching P2
    hits = idx.search("transformer deep learning", k=2)
    assert len(hits) > 0
    assert hits[0][0] == "P2"

    # Query with exclusion set
    hits_excluded = idx.search("transformer deep learning", k=2, exclusion_set={"P2"})
    result_ids = [pid for pid, _ in hits_excluded]
    assert "P2" not in result_ids

    # Persistence roundtrip
    save_dir = tmp_path / "bm25s_dir"
    idx.save(save_dir)

    loaded_idx = LexicalIndex.load(save_dir)
    assert len(loaded_idx) == 3
    loaded_hits = loaded_idx.search("quantum entanglement", k=1)
    assert loaded_hits[0][0] == "P1"


def test_hybrid_retriever_fusion():
    """Test multi-channel hybrid retrieval combining dense and lexical hits via RRF."""
    dim = 2
    dense_idx = DenseIndex(dimension=dim)
    dense_idx.add_documents(
        ["P_DENSE", "P_SHARED", "P_OTHER"],
        np.array([[1.0, 0.0], [0.7071, 0.7071], [0.0, 1.0]], dtype=np.float32),
    )

    lex_idx = LexicalIndex(stopwords="en")
    lex_idx.index_documents(
        ["P_LEX", "P_SHARED", "P_OTHER"],
        ["information retrieval ranking", "information retrieval systems", "astronomy galaxies"],
        show_progress=False,
    )

    retriever = HybridRetriever(dense_index=dense_idx, lexical_index=lex_idx, candidate_pool_size=3)

    # P_SHARED appears in both dense and lexical lists -> should receive highest fused RRF score
    q_vec = np.array([0.8, 0.6], dtype=np.float32)
    q_text = "information retrieval"
    fused = retriever.retrieve(dense_query_vec=q_vec, lexical_query=q_text, top_m=3)

    assert len(fused) == 3
    # P_SHARED is boosted by appearing in both channels
    top_pid = fused[0][0]
    assert top_pid == "P_SHARED"


def test_embeddings_format_paper_text():
    """Test text concatenation logic for title and abstract."""
    encoder = SPECTER2Encoder.__new__(SPECTER2Encoder)
    encoder._model = None
    class DummyTokenizer:
        sep_token = "[SEP]"
    encoder._tokenizer = DummyTokenizer()

    text_with_abs = encoder.format_paper_text("Attention Is All You Need", "We propose the Transformer model.")
    assert text_with_abs == "Attention Is All You Need [SEP] We propose the Transformer model."

    text_title_only = encoder.format_paper_text("Attention Is All You Need", None)
    assert text_title_only == "Attention Is All You Need"


def _many_unit_vectors(n: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(0)
    v = rng.normal(size=(n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_empty_candidate_filter_allows_nothing():
    """An empty whitelist must return no hits, not behave like 'no filter'."""
    idx = DenseIndex(dimension=8)
    vecs = _many_unit_vectors(10)
    idx.add_documents([f"W{i}" for i in range(10)], vecs)
    assert idx.search(vecs[0], k=3, candidate_filter=set()) == []

    lex = LexicalIndex()
    lex.index_documents(["P1", "P2"], ["graph neural networks", "graph databases"], show_progress=False)
    assert lex.search("graph", k=2, candidate_filter=set()) == []


def test_narrow_whitelist_widens_fetch_depth():
    """With a whitelist of far-away papers, the default over-fetch (~200) is not enough; search must widen."""
    n = 1000
    vecs = _many_unit_vectors(n)
    idx = DenseIndex(dimension=8)
    ids = [f"W{i}" for i in range(n)]
    idx.add_documents(ids, vecs)
    q = vecs[0]
    order = np.argsort(-(vecs @ q))
    worst = {ids[i] for i in order[-5:]}  # the 5 least similar papers
    hits = idx.search(q, k=5, candidate_filter=worst)
    assert {pid for pid, _ in hits} == worst

    # An explicit fetch_depth is respected (no widening).
    assert len(idx.search(q, k=5, candidate_filter=worst, fetch_depth=50)) < 5


def test_lexical_narrow_whitelist_widens_fetch_depth():
    n = 600
    docs = [f"filler text number {i} about topic{i % 7}" for i in range(n - 1)] + ["unique zebra quantum"]
    ids = [f"P{i}" for i in range(n)]
    lex = LexicalIndex()
    lex.index_documents(ids, docs, show_progress=False)
    hits = lex.search("filler text topic3", k=1, candidate_filter={"P599"})
    assert [pid for pid, _ in hits] == ["P599"]


def test_embedding_store_roundtrip_select_and_legacy(tmp_path: Path):
    from paperfeed.index.store import EmbeddingStore, ids_sha256

    vecs = _many_unit_vectors(5)
    store = EmbeddingStore()
    store.add(["W1", "W2", "W3"], vecs[:3])
    store.add(["W4", "W5"], vecs[3:])
    assert store.missing(["W1", "W9"]) == ["W9"]
    assert np.allclose(store.select(["W5", "W2"]), vecs[[4, 1]])
    with pytest.raises(KeyError):
        store.select(["W1", "W9"])

    vpath, ipath = tmp_path / "e.npy", tmp_path / "e_ids.json"
    store.save(vpath, ipath)
    loaded = EmbeddingStore.load(vpath, ipath)
    assert loaded.ids == store.ids and np.allclose(loaded.vectors, store.vectors)

    # Legacy layout: vectors without an ID file adopt the index's paper_ids.json order, else refuse.
    ipath.unlink()
    with pytest.raises(FileNotFoundError):
        EmbeddingStore.load(vpath, ipath)
    legacy = tmp_path / "paper_ids.json"
    legacy.write_text('{"id_to_paper": ["W1", "W2", "W3", "W4", "W5"]}')
    assert EmbeddingStore.load(vpath, ipath, legacy_ids_path=legacy).ids == store.ids

    assert ids_sha256(["a", "b"]) != ids_sha256(["b", "a"])
