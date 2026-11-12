"""Unit tests for the Phase 7 retrieval benchmarks (docs/plan.md §7, docs/design.md §13.3)."""

import json
from pathlib import Path

import pytest

from paperfeed.bench.retrieval import (
    BenchmarkQuery,
    RetrievalBenchmark,
    _csfcube_text,
    load_benchmark,
    prepare_csfcube,
    prepare_litsearch,
    save_benchmark,
)


# ---------------------------------------------------------------------------
# 7.1 Datasets
# ---------------------------------------------------------------------------

@pytest.fixture
def csfcube_raw(tmp_path):
    """A two-query CSFCube release in the official file layout."""
    raw = tmp_path / "csfcube"
    raw.mkdir()
    papers = {
        "100": ("Query paper", "['First sentence.', 'Second sentence.']"),
        "200": ("Relevant one", "['Uses the same method.']"),
        "300": ("Somewhat related", "['Partly related.']"),
        "400": ("Unrelated", "['Different topic.']"),
        "500": ("Other query", "['Another query paper.']"),
    }
    with open(raw / "abstracts-csfcube-preds.jsonl", "w", encoding="utf-8") as f:
        for pid, (title, abstract) in papers.items():
            f.write(json.dumps({"paper_id": pid, "title": title, "abstract": abstract, "metadata": "{}",
                                "pred_labels": "[]", "pred_labels_truncated": "False"}) + "\n")
    (raw / "queries-release.csv").write_text(
        "pid,facet,year,title,paper type\n100,method,2017,Query paper,data-driven approach\n500,result,2019,Other query,empirical\n",
        encoding="utf-8",
    )
    anns = {
        "background": {},
        "method": {"100": {"cands": ["200", "300", "400"], "relevance_adju": [3, 1, 0], "relevance_ann1": [3, 1, 0]}},
        "result": {"500": {"cands": ["200", "400"], "relevance_adju": [2, 0], "relevance_ann1": [2, 0]}},
    }
    for facet, d in anns.items():
        (raw / f"test-pid2anns-csfcube-{facet}.json").write_text(json.dumps(d), encoding="utf-8")
    return raw


def test_csfcube_text_parses_stringified_sentence_lists():
    assert _csfcube_text("['A b.', 'C d.']") == "A b. C d."
    assert _csfcube_text(["A b.", " C d."]) == "A b. C d."
    assert _csfcube_text("Plain title") == "Plain title"
    assert _csfcube_text("[not a list") == "[not a list"
    assert _csfcube_text("[__import__('os')]") == "[__import__('os')]"  # never evaluated: left as text
    assert _csfcube_text(None) == ""


def test_prepare_csfcube(csfcube_raw):
    bench = prepare_csfcube(csfcube_raw)
    assert bench.name == "csfcube" and bench.query_adapter == "proximity" and bench.relevance_threshold == 2
    assert bench.corpus["100"] == {"title": "Query paper", "abstract": "First sentence. Second sentence."}
    assert [q.qid for q in bench.queries] == ["100_method", "500_result"]
    q = bench.queries[0]
    assert q.kind == "paper" and q.doc_id == "100" and q.facet == "method"
    assert q.qrels == {"200": 3, "300": 1, "400": 0} and q.pool == ["200", "300", "400"]
    assert q.relevant(2) == ["200"]  # grade 1 is not relevant under the official threshold
    assert q.meta == {"year": "2017", "paper_type": "data-driven approach"}


def test_csfcube_rejects_mismatched_grades(csfcube_raw):
    (csfcube_raw / "test-pid2anns-csfcube-method.json").write_text(
        json.dumps({"100": {"cands": ["200", "300"], "relevance_adju": [3]}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="candidates vs"):
        prepare_csfcube(csfcube_raw)


@pytest.fixture
def litsearch_raw(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    raw = tmp_path / "litsearch"
    raw.mkdir()
    pq.write_table(pa.table({
        "query_set": ["inline_acl", "manual_iclr"],
        "query": ["Papers on distilling language models?", " Which work studies mode collapse? "],
        "specificity": [0, 1],
        "quality": [2, 1],
        "corpusids": [[11], [22, 33]],
    }), raw / "query.parquet")
    for i, rows in enumerate([[(11, "Distillation", "We distil."), (22, "GAN modes", None)], [(33, "Mode collapse", "Study."), (44, "Other", "X.")]]):
        pq.write_table(pa.table({
            "corpusid": [r[0] for r in rows], "title": [r[1] for r in rows], "abstract": [r[2] for r in rows],
            "citations": [[] for _ in rows], "full_paper": ["long text" for _ in rows],
        }), raw / f"corpus_clean_{i}.parquet")
    return raw


def test_prepare_litsearch(litsearch_raw):
    bench = prepare_litsearch(litsearch_raw)
    assert bench.name == "litsearch" and bench.query_adapter == "adhoc_query" and bench.relevance_threshold == 1
    assert set(bench.corpus) == {"11", "22", "33", "44"}
    assert bench.corpus["22"] == {"title": "GAN modes", "abstract": ""}  # missing abstract: title only
    q0, q1 = bench.queries
    assert (q0.qid, q0.kind, q0.text, q0.qrels) == ("litsearch_0000", "text", "Papers on distilling language models?", {"11": 1})
    assert q1.text == "Which work studies mode collapse?" and q1.qrels == {"22": 1, "33": 1}
    assert q1.pool is None and q1.meta["query_set"] == "manual_iclr"


def test_check_catches_judged_documents_missing_from_the_corpus():
    bench = RetrievalBenchmark(
        name="x", corpus={"a": {"title": "A", "abstract": ""}},
        queries=[BenchmarkQuery(qid="q", kind="text", text="t", qrels={"a": 1, "zzz": 1})],
        relevance_threshold=1, query_adapter="adhoc_query",
    )
    with pytest.raises(ValueError, match="not in the corpus"):
        bench.check()
    bench.queries = [BenchmarkQuery(qid="q", kind="paper", qrels={"a": 1})]
    with pytest.raises(ValueError, match="no doc_id"):
        bench.check()


def test_save_and_load_round_trip(csfcube_raw, tmp_path):
    bench = prepare_csfcube(csfcube_raw)
    d = save_benchmark(bench, tmp_path / "out")
    assert sorted(p.name for p in d.iterdir()) == ["benchmark.json", "corpus.jsonl", "queries.jsonl"]
    meta = json.loads((d / "benchmark.json").read_text())
    assert (meta["n_docs"], meta["n_queries"], meta["n_docs_without_abstract"]) == (5, 2, 0)
    again = load_benchmark(d)
    assert again.corpus == bench.corpus and again.queries == bench.queries
    assert (again.relevance_threshold, again.query_adapter, again.license) == (2, "proximity", "CC BY-NC 4.0")


# ---------------------------------------------------------------------------
# 7.1 Per-benchmark indexes
# ---------------------------------------------------------------------------

import numpy as np

from paperfeed.bench.retrieval_index import build_benchmark_index, corpus_fingerprint, lexical_text, load_benchmark_index


class FakeEncoder:
    """Deterministic stand-in for SPECTER2Encoder that counts what it embeds."""

    base_model_name = "fake-base"
    dense_adapter_name = "fake-proximity"
    query_adapter_name = "fake-adhoc"

    def __init__(self):
        self.papers_embedded = 0
        self.queries_embedded = []

    @staticmethod
    def _vec(text):
        rng = np.random.default_rng(abs(hash(text)) % (2 ** 32))
        v = rng.normal(size=8).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_papers(self, papers, batch_size=32, show_progress=False):
        self.papers_embedded += len(papers)
        return np.vstack([self._vec(p["title"] + p["abstract"]) for p in papers])

    def embed_queries(self, queries, batch_size=32):
        self.queries_embedded.extend(queries)
        return np.vstack([self._vec("q:" + q) for q in queries])


def _text_bench():
    corpus = {
        "1": {"title": "Knowledge distillation", "abstract": "Compressing language models."},
        "2": {"title": "Mode collapse in GANs", "abstract": ""},
        "3": {"title": "Graph neural networks", "abstract": "Message passing."},
    }
    queries = [BenchmarkQuery(qid="t0", kind="text", text="distilling language models", qrels={"1": 1})]
    return RetrievalBenchmark(name="tiny", corpus=corpus, queries=queries, relevance_threshold=1, query_adapter="adhoc_query")


def test_lexical_text_matches_the_main_index_format():
    assert lexical_text({"title": "T", "abstract": "A"}) == "T A"
    assert lexical_text({"title": "T", "abstract": ""}) == "T"


def test_build_and_load_text_benchmark_index(tmp_path):
    bench, enc = _text_bench(), FakeEncoder()
    manifest = build_benchmark_index(bench, enc, tmp_path, batch_size=2, checkpoint_every=2)
    assert (manifest["n_docs"], manifest["n_queries"], manifest["query_adapter"]) == (3, 1, "adhoc_query")
    assert manifest["encoder"] == {"base_model": "fake-base", "dense_adapter": "fake-proximity", "query_adapter": "fake-adhoc"}
    assert enc.papers_embedded == 3 and enc.queries_embedded == ["distilling language models"]

    idx = load_benchmark_index(bench, tmp_path)
    np.testing.assert_allclose(idx.query_vectors["t0"], FakeEncoder._vec("q:distilling language models"))
    assert idx.dense.search(FakeEncoder._vec("Knowledge distillationCompressing language models."), k=1)[0][0] == "1"
    assert idx.lexical.search("mode collapse", k=1)[0][0] == "2"  # title-only document is searchable


def test_paper_queries_reuse_the_query_papers_own_vector(csfcube_raw, tmp_path):
    bench, enc = prepare_csfcube(csfcube_raw), FakeEncoder()
    build_benchmark_index(bench, enc, tmp_path)
    assert enc.queries_embedded == []  # query-by-example: no adhoc_query encoding
    idx = load_benchmark_index(bench, tmp_path)
    np.testing.assert_allclose(idx.query_vectors["100_method"], idx.store.select(["100"])[0])


def test_rebuild_resumes_and_discards_vectors_after_a_corpus_change(tmp_path):
    bench, enc = _text_bench(), FakeEncoder()
    build_benchmark_index(bench, enc, tmp_path)
    again = FakeEncoder()
    m2 = build_benchmark_index(bench, again, tmp_path)
    assert again.papers_embedded == 0 and m2["embedded_docs_timed"] == 3  # resumed: nothing re-embedded

    bench.corpus["2"]["abstract"] = "Now with an abstract."
    changed = FakeEncoder()
    build_benchmark_index(bench, changed, tmp_path)
    assert changed.papers_embedded == 3  # fingerprint changed: every vector rebuilt, never mixed


def test_load_refuses_an_index_built_from_another_corpus(tmp_path):
    bench = _text_bench()
    build_benchmark_index(bench, FakeEncoder(), tmp_path)
    old_fp = corpus_fingerprint(bench)
    bench.corpus["4"] = {"title": "New", "abstract": ""}
    assert corpus_fingerprint(bench) != old_fp
    with pytest.raises(ValueError, match="different corpus"):
        load_benchmark_index(bench, tmp_path)
    with pytest.raises(FileNotFoundError):
        load_benchmark_index(bench, tmp_path / "nowhere")


def test_an_interrupted_build_resumes_from_its_checkpoint(tmp_path):
    bench = _text_bench()

    class CrashingEncoder(FakeEncoder):
        def embed_papers(self, papers, batch_size=32, show_progress=False):
            if self.papers_embedded:  # the second checkpoint chunk never completes
                raise KeyboardInterrupt
            return super().embed_papers(papers, batch_size, show_progress)

    with pytest.raises(KeyboardInterrupt):
        build_benchmark_index(bench, CrashingEncoder(), tmp_path, checkpoint_every=2)
    resumed = FakeEncoder()
    manifest = build_benchmark_index(bench, resumed, tmp_path, checkpoint_every=2)
    assert resumed.papers_embedded == 1  # only the document after the checkpoint
    assert manifest["n_docs"] == 3 and "status" not in manifest
