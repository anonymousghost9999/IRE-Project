"""Unit tests for Phase 3.4 SciERC Benchmark Evaluation Module."""

from pathlib import Path
import json
import tempfile
from unittest.mock import MagicMock
import pytest

from paperfeed.bench.scierc import (
    DEFAULT_LABEL_MAP,
    EVALUATION_LABELS,
    NATIVE_LABEL_MAP,
    BenchmarkResults,
    EntitySpan,
    LabelMetrics,
    SciERCDocument,
    SciERCEvaluator,
    char_to_token_offsets,
    compute_span_overlap_iou,
    evaluate_document_spans,
    filter_predictions_by_threshold,
    format_markdown_report,
    load_scierc_dataset,
)


@pytest.fixture
def sample_scierc_record() -> dict:
    """Return a minimal valid SciERC record fixture."""
    return {
        "doc_key": "DOC_001",
        "sentences": [
            ["We", "propose", "BERT", "for", "text", "classification", "."],
            ["Experiments", "on", "GLUE", "show", "92", "%", "accuracy", "."],
        ],
        "ner": [
            # Sentence 0 (tokens 0..6): "BERT" [2, 2, 'Method'], "text classification" [4, 5, 'Task']
            [[2, 2, "Method"], [4, 5, "Task"]],
            # Sentence 1 (tokens 7..14): "GLUE" [9, 9, 'Material'], "accuracy" [13, 13, 'Metric']
            [[9, 9, "Material"], [13, 13, "Metric"]],
        ],
    }


def test_char_to_token_offsets():
    """Verify mapping of character offsets to token index spans."""
    tokens = ["Deep", "neural", "networks", "for", "NLP"]
    # Char spans:
    # Deep: (0, 4)
    # neural: (5, 11)
    # networks: (12, 20)
    # for: (21, 24)
    # NLP: (25, 28)
    spans = [(0, 4), (5, 11), (12, 20), (21, 24), (25, 28)]

    # Exact token span for "Deep neural networks" (char 0 to 20)
    assert char_to_token_offsets(0, 20, spans) == (0, 2)
    # Single token "NLP" (char 25 to 28)
    assert char_to_token_offsets(25, 28, spans) == (4, 4)
    # Partial character overlap with "neural"
    assert char_to_token_offsets(6, 10, spans) == (1, 1)
    # Out-of-bounds char range
    assert char_to_token_offsets(30, 40, spans) is None


def test_span_overlap_iou():
    """Verify token overlap and IoU calculation."""
    s1 = EntitySpan(start_token=2, end_token=5, label="Method")  # length 4
    s2 = EntitySpan(start_token=4, end_token=7, label="Method")  # length 4

    overlap, iou = compute_span_overlap_iou(s1, s2)
    # Overlap tokens: 4, 5 -> overlap = 2
    # Union tokens: 2, 3, 4, 5, 6, 7 -> union = 6
    assert overlap == 2
    assert pytest.approx(iou, rel=1e-3) == (2.0 / 6.0)

    # Disjoint spans
    s3 = EntitySpan(start_token=8, end_token=10, label="Method")
    overlap_disjoint, iou_disjoint = compute_span_overlap_iou(s1, s3)
    assert overlap_disjoint == 0
    assert iou_disjoint == 0.0


def test_label_metrics():
    """Verify precision, recall, and F1 computation."""
    m = LabelMetrics(label="Task", tp=8, fp=2, fn=2).compute()
    assert pytest.approx(m.precision) == 0.8
    assert pytest.approx(m.recall) == 0.8
    assert pytest.approx(m.f1) == 0.8

    # Edge cases (zero division)
    zero_m = LabelMetrics(label="Task", tp=0, fp=0, fn=0).compute()
    assert zero_m.precision == 0.0
    assert zero_m.recall == 0.0
    assert zero_m.f1 == 0.0


def test_strict_and_relaxed_evaluation_logic():
    """Test strict vs relaxed span matching across different prediction scenarios."""
    gold = [
        EntitySpan(start_token=0, end_token=2, label="Task"),    # "information retrieval systems"
        EntitySpan(start_token=5, end_token=5, label="Method"),  # "BM25"
        EntitySpan(start_token=8, end_token=9, label="Metric"),  # "NDCG score"
    ]

    preds = [
        # Scenario 1: Exact strict match
        EntitySpan(start_token=0, end_token=2, label="Task"),
        # Scenario 2: Partial relaxed match ("BM25 model" vs "BM25")
        EntitySpan(start_token=5, end_token=6, label="Method"),
        # Scenario 3: Spurious prediction
        EntitySpan(start_token=15, end_token=16, label="Material"),
    ]

    eval_labels = ["Task", "Method", "Metric", "Material"]
    strict_counts, relaxed_counts = evaluate_document_spans(gold, preds, eval_labels)

    # Task: exact match -> Strict TP=1, Relaxed TP=1
    assert strict_counts["Task"] == (1, 0, 0)
    assert relaxed_counts["Task"] == (1, 0, 0)

    # Method: relaxed match only -> Strict TP=0, FP=1, FN=1; Relaxed TP=1, FP=0, FN=0
    assert strict_counts["Method"] == (0, 1, 1)
    assert relaxed_counts["Method"] == (1, 0, 0)

    # Metric: completely missed -> Strict TP=0, FP=0, FN=1; Relaxed TP=0, FP=0, FN=1
    assert strict_counts["Metric"] == (0, 0, 1)
    assert relaxed_counts["Metric"] == (0, 0, 1)

    # Material: spurious -> Strict TP=0, FP=1, FN=0; Relaxed TP=0, FP=1, FN=0
    assert strict_counts["Material"] == (0, 1, 0)
    assert relaxed_counts["Material"] == (0, 1, 0)


def test_strict_multiset_matching_duplicate_spans():
    """Verify multiset strict matching correctly handles duplicate predictions and gold spans."""
    # Scenario: two identical predictions against one gold span -> TP=1, FP=1, FN=0
    gold = [EntitySpan(start_token=2, end_token=2, label="Method")]
    preds = [
        EntitySpan(start_token=2, end_token=2, label="Method"),
        EntitySpan(start_token=2, end_token=2, label="Method"),
    ]
    strict_counts, relaxed_counts = evaluate_document_spans(gold, preds, ["Method"])
    assert strict_counts["Method"] == (1, 1, 0)
    assert relaxed_counts["Method"] == (1, 1, 0)

    # Scenario: two identical predictions against two identical gold spans -> TP=2, FP=0, FN=0
    gold_dup = [
        EntitySpan(start_token=2, end_token=2, label="Method"),
        EntitySpan(start_token=2, end_token=2, label="Method"),
    ]
    strict_counts_dup, relaxed_counts_dup = evaluate_document_spans(gold_dup, preds, ["Method"])
    assert strict_counts_dup["Method"] == (2, 0, 0)
    assert relaxed_counts_dup["Method"] == (2, 0, 0)


def test_load_scierc_dataset_json_and_jsonl(tmp_path: Path, sample_scierc_record: dict):
    """Test loader handles both single-array and JSONL formats."""
    # Test JSONL
    jsonl_file = tmp_path / "test.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample_scierc_record) + "\n")

    docs = load_scierc_dataset(jsonl_file)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.doc_key == "DOC_001"
    assert len(doc.sentences) == 2
    assert doc.flat_tokens == [
        "We", "propose", "BERT", "for", "text", "classification", ".",
        "Experiments", "on", "GLUE", "show", "92", "%", "accuracy", ".",
    ]
    assert doc.sentence_offsets == [0, 7]

    gold_spans = doc.get_gold_spans()
    assert len(gold_spans) == 4
    labels = {s.label for s in gold_spans}
    assert labels == {"Method", "Task", "Material", "Metric"}

    # Test JSON array
    json_array_file = tmp_path / "test.json"
    with open(json_array_file, "w", encoding="utf-8") as f:
        f.write(json.dumps([sample_scierc_record]))

    docs_arr = load_scierc_dataset(json_array_file)
    assert len(docs_arr) == 1
    assert docs_arr[0].doc_key == "DOC_001"


def test_load_scierc_dataset_raises_on_malformed_json(tmp_path: Path):
    """Verify load_scierc_dataset raises ValueError on malformed array or JSONL files."""
    # Malformed JSON array (starts with [ but invalid syntax)
    bad_array = tmp_path / "bad_array.json"
    bad_array.write_text("[{broken json", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed SciERC JSON array"):
        load_scierc_dataset(bad_array)

    # Malformed JSONL
    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text('{"doc_key": "OK"}\n{invalid line\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed SciERC JSONL"):
        load_scierc_dataset(bad_jsonl)


def test_scierc_evaluator_end_to_end(sample_scierc_record: dict):
    """Test full evaluator run with a mock extractor and label mapping."""
    doc = SciERCDocument(
        doc_key=sample_scierc_record["doc_key"],
        sentences=sample_scierc_record["sentences"],
        gold_ner=sample_scierc_record["ner"],
    )

    # Mock extractor simulating GLiNER returning lowercase production facets:
    # Sentence 0: "We propose BERT for text classification ."
    # Sentence 1: "Experiments on GLUE show 92 % accuracy ."
    mock_extractor = MagicMock()
    mock_extractor.extract_from_abstract.side_effect = [
        # Sentence 0 output:
        [
            # "BERT" starts at char 11, end at 15
            {"facet_type": "method", "facet_value": "BERT", "start_char": 11, "end_char": 15, "confidence": 0.95},
            # "text classification" starts at char 20, end at 39
            {"facet_type": "task", "facet_value": "text classification", "start_char": 20, "end_char": 39, "confidence": 0.91},
        ],
        # Sentence 1 output:
        [
            # "GLUE" starts at char 15, end at 19
            {"facet_type": "dataset", "facet_value": "GLUE", "start_char": 15, "end_char": 19, "confidence": 0.88},
            # "accuracy" starts at char 30, end at 38
            {"facet_type": "metric", "facet_value": "accuracy", "start_char": 30, "end_char": 38, "confidence": 0.89},
        ],
    ]

    evaluator = SciERCEvaluator(
        label_map=DEFAULT_LABEL_MAP,
        eval_labels=EVALUATION_LABELS,
        extractor=mock_extractor,
    )

    results = evaluator.evaluate([doc])

    assert results.total_documents == 1
    assert results.total_gold_entities == 4
    assert results.total_predicted_entities == 4

    # All 4 predictions matched exactly and mapped correctly
    assert results.strict_micro.precision == 1.0
    assert results.strict_micro.recall == 1.0
    assert results.strict_micro.f1 == 1.0

    assert results.relaxed_micro.f1 == 1.0
    assert results.strict_macro_f1 > 0.0

    # Check Markdown report generation
    report = format_markdown_report(results)
    assert "SciERC Extraction Benchmark Report" in report
    assert "Strict Span Matching" in report
    assert "Relaxed Span Matching" in report
    assert "100.00%" in report


def test_native_label_prompts_map_to_scierc_labels():
    """Native-label prompts are lowercase (FacetExtractor lowercases labels) and cover every SciERC label."""
    assert all(k == k.lower() for k in NATIVE_LABEL_MAP)
    assert set(NATIVE_LABEL_MAP.values()) == set(EVALUATION_LABELS)

    doc = SciERCDocument(
        doc_key="D",
        sentences=[["We", "use", "BLEU", "."]],
        gold_ner=[[(2, 2, "Metric"), ]],
    )
    extractor = MagicMock()
    extractor.extract_from_abstract.return_value = [
        {"facet_type": "metric", "facet_value": "BLEU", "start_char": 7, "end_char": 11, "confidence": 0.9},
    ]
    results = SciERCEvaluator(label_map=NATIVE_LABEL_MAP, extractor=extractor).evaluate([doc])
    assert results.strict_per_label["Metric"].tp == 1

    # A prompt such as "other scientific term" must map back to OtherScientificTerm
    doc2 = SciERCDocument("D2", [["a", "b"]], [[(0, 0, "OtherScientificTerm")]])
    extractor.extract_from_abstract.return_value = [
        {"facet_type": "other scientific term", "facet_value": "a", "start_char": 0, "end_char": 1, "confidence": 0.8},
    ]
    results = SciERCEvaluator(label_map=NATIVE_LABEL_MAP, extractor=extractor).evaluate([doc2])
    assert results.strict_per_label["OtherScientificTerm"].tp == 1


def test_evaluator_with_precomputed_predictions(sample_scierc_record: dict):
    """Test evaluate with explicit precomputed EntitySpans."""
    doc = SciERCDocument(
        doc_key=sample_scierc_record["doc_key"],
        sentences=sample_scierc_record["sentences"],
        gold_ner=sample_scierc_record["ner"],
    )

    precomputed = {
        "DOC_001": [
            EntitySpan(start_token=2, end_token=2, label="Method", text="BERT"),
            EntitySpan(start_token=4, end_token=5, label="Task", text="text classification"),
        ]
    }

    evaluator = SciERCEvaluator(eval_labels=EVALUATION_LABELS)
    results = evaluator.evaluate([doc], predictions=precomputed)

    assert results.total_documents == 1
    assert results.total_gold_entities == 4
    assert results.total_predicted_entities == 2

    assert results.strict_per_label["Method"].tp == 1
    assert results.strict_per_label["Task"].tp == 1
    assert results.strict_per_label["Material"].fn == 1
    assert results.strict_per_label["Metric"].fn == 1

    assert pytest.approx(results.strict_micro.precision) == 1.0
    assert pytest.approx(results.strict_micro.recall) == 0.5
    assert pytest.approx(results.strict_micro.f1) == (2 * 1.0 * 0.5 / 1.5)


def test_generic_gold_entities_ignored():
    """Verify predictions overlapping gold Generic spans are ignored without false positives."""
    doc = SciERCDocument(
        doc_key="DOC_GENERIC",
        sentences=[["This", "approach", "solves", "the", "problem", "."]],
        gold_ner=[
            # "This approach" [0, 1] is Generic; "the problem" [3, 4] is Task
            [[0, 1, "Generic"], [3, 4, "Task"]]
        ],
    )

    # Predictions:
    # 1. "This approach" [0, 1] predicted as Method (overlaps gold Generic)
    # 2. "the problem" [3, 4] predicted as Task (true positive)
    predictions = {
        "DOC_GENERIC": [
            EntitySpan(start_token=0, end_token=1, label="Method", text="This approach"),
            EntitySpan(start_token=3, end_token=4, label="Task", text="the problem"),
        ]
    }

    evaluator = SciERCEvaluator(eval_labels=EVALUATION_LABELS, ignore_generic=True)
    results = evaluator.evaluate([doc], predictions=predictions)

    # "This approach" is ignored because it overlaps a gold Generic span
    assert results.metadata["predictions_ignored_generic"] == 1
    assert results.total_predicted_entities == 1
    assert results.total_gold_entities == 1  # only "Task" (Generic is not an eval label)

    # Method has 0 FP because it was ignored
    assert results.strict_per_label["Method"].tp == 0
    assert results.strict_per_label["Method"].fp == 0
    assert results.strict_per_label["Method"].fn == 0

    # Task is exact match
    assert results.strict_per_label["Task"].tp == 1
    assert results.strict_per_label["Task"].fp == 0
    assert results.strict_per_label["Task"].fn == 0

    # Precision should be 1.0 (not depressed to 0.5 by Generic)
    assert results.strict_micro.precision == 1.0
    assert results.strict_micro.recall == 1.0
    assert results.strict_micro.f1 == 1.0

    # If ignore_generic is False, Method counts as FP
    results_no_ignore = evaluator.evaluate([doc], predictions=predictions, ignore_generic=False)
    assert results_no_ignore.metadata["predictions_ignored_generic"] == 0
    assert results_no_ignore.total_predicted_entities == 2
    assert results_no_ignore.strict_per_label["Method"].fp == 1
    assert results_no_ignore.strict_micro.precision == 0.5


def test_batched_matches_per_sentence_predictions(sample_scierc_record: dict):
    """Verify batched and per-sentence predictions produce identical EntitySpans on a mock extractor."""
    doc = SciERCDocument(
        doc_key=sample_scierc_record["doc_key"],
        sentences=sample_scierc_record["sentences"],
        gold_ner=sample_scierc_record["ner"],
    )

    mapping = {
        "We propose BERT for text classification .": [
            {"start": 11, "end": 15, "label": "method", "text": "BERT", "score": 0.95},
            {"start": 20, "end": 39, "label": "task", "text": "text classification", "score": 0.91},
        ],
        "Experiments on GLUE show 92 % accuracy .": [
            {"start": 15, "end": 19, "label": "dataset", "text": "GLUE", "score": 0.88},
            {"start": 30, "end": 38, "label": "metric", "text": "accuracy", "score": 0.89},
        ],
    }

    class MockGLiNER:
        def predict_entities(self, text, labels, threshold=0.5):
            return mapping.get(text, [])

        def batch_predict_entities(self, texts, labels, threshold=0.5, batch_size=32):
            return [mapping.get(t, []) for t in texts]

    mock = MockGLiNER()
    evaluator = SciERCEvaluator(
        label_map=DEFAULT_LABEL_MAP,
        eval_labels=EVALUATION_LABELS,
        extractor=mock,
    )

    # 1. Per-sentence prediction
    per_sentence_spans = evaluator.predict_document_spans(doc)
    assert len(per_sentence_spans) == 4

    # 2. Batched prediction with chunk sizes 1 and 32
    batched_preds_1 = evaluator.predict_batch_documents([doc], batch_size=1)
    batched_preds_32 = evaluator.predict_batch_documents([doc], batch_size=32)

    assert doc.doc_key in batched_preds_1
    assert doc.doc_key in batched_preds_32
    assert per_sentence_spans == batched_preds_1[doc.doc_key]
    assert per_sentence_spans == batched_preds_32[doc.doc_key]


def test_extractor_errors_not_swallowed(sample_scierc_record: dict):
    """Verify that an extractor raising on every sentence aborts the benchmark with a clear error."""
    doc = SciERCDocument(
        doc_key=sample_scierc_record["doc_key"],
        sentences=sample_scierc_record["sentences"],
        gold_ner=sample_scierc_record["ner"],
    )

    class BrokenExtractor:
        def predict_entities(self, text, labels, threshold=0.5):
            raise RuntimeError("CUDA out of memory error during extraction")

    broken = BrokenExtractor()
    evaluator = SciERCEvaluator(
        label_map=DEFAULT_LABEL_MAP,
        eval_labels=EVALUATION_LABELS,
        extractor=broken,
    )

    # Calling evaluate should raise RuntimeError and abort rather than returning 0 F1 silently
    with pytest.raises(RuntimeError) as exc_info:
        evaluator.evaluate([doc])
    assert "Benchmark aborted" in str(exc_info.value) or "failed" in str(exc_info.value)

    # When strict_errors=True, it fails immediately on the sentence
    evaluator_strict = SciERCEvaluator(
        label_map=DEFAULT_LABEL_MAP,
        eval_labels=EVALUATION_LABELS,
        extractor=broken,
        strict_errors=True,
    )
    with pytest.raises(RuntimeError) as exc_info_strict:
        evaluator_strict.evaluate([doc])
    assert "CUDA out of memory error" in str(exc_info_strict.value)


def test_report_flags_low_confidence_and_3label_averages(sample_scierc_record: dict):
    """Verify markdown report flags low-confidence facets with footnotes and includes 3-label core averages."""
    doc = SciERCDocument(
        doc_key=sample_scierc_record["doc_key"],
        sentences=sample_scierc_record["sentences"],
        gold_ner=sample_scierc_record["ner"],
    )

    precomputed = {
        "DOC_001": [
            EntitySpan(start_token=2, end_token=2, label="Method", text="BERT"),
            EntitySpan(start_token=4, end_token=5, label="Task", text="text classification"),
            EntitySpan(start_token=9, end_token=9, label="Material", text="GLUE"),
        ]
    }

    evaluator = SciERCEvaluator(eval_labels=EVALUATION_LABELS)
    results = evaluator.evaluate([doc], predictions=precomputed)
    report = format_markdown_report(results)

    # 1. Footnote mark on Material and OtherScientificTerm
    assert "`Material` †" in report
    assert "`OtherScientificTerm` †" in report
    assert "† *Note: `Material` and `OtherScientificTerm`" in report

    # 2. 3-label core averages row present
    assert "Core 3 Labels: Task, Method, Metric" in report
    assert results.core_strict_micro.f1 > 0.0
    assert "core_3_labels" in results.to_dict()["strict_span_matching"]


def test_threshold_sweep_filtering():
    """Verify filtering predictions at a higher threshold never increases the prediction count."""
    predictions = {
        "DOC_1": [
            EntitySpan(0, 1, "Method", "BERT", confidence=0.35),
            EntitySpan(2, 3, "Task", "NER", confidence=0.55),
            EntitySpan(4, 5, "Metric", "F1", confidence=0.75),
            EntitySpan(6, 7, "Material", "SciERC", confidence=0.92),
        ]
    }

    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    prev_count = float("inf")

    for t in thresholds:
        filtered = filter_predictions_by_threshold(predictions, t)
        count = len(filtered["DOC_1"])
        assert count <= prev_count, f"Count increased at threshold {t}: {count} > {prev_count}"
        prev_count = count

    assert len(filter_predictions_by_threshold(predictions, 0.2)["DOC_1"]) == 4
    assert len(filter_predictions_by_threshold(predictions, 0.4)["DOC_1"]) == 3
    assert len(filter_predictions_by_threshold(predictions, 0.6)["DOC_1"]) == 2
    assert len(filter_predictions_by_threshold(predictions, 0.8)["DOC_1"]) == 1
    assert len(filter_predictions_by_threshold(predictions, 0.95)["DOC_1"]) == 0


def test_macro_average_with_zero_gold_labels():
    """Verify macro average only computes over labels with gold > 0 (e.g. under --limit)."""
    doc = SciERCDocument(
        doc_key="D_ZERO_GOLD",
        sentences=[["Run", "BERT", "on", "classification", "."]],
        gold_ner=[[[1, 1, "Method"], [3, 3, "Task"]]],
    )
    predictions = {
        "D_ZERO_GOLD": [
            EntitySpan(1, 1, "Method", "BERT"),
            EntitySpan(3, 3, "Task", "classification"),
        ]
    }
    evaluator = SciERCEvaluator(eval_labels=EVALUATION_LABELS)
    results = evaluator.evaluate([doc], predictions=predictions)

    # Method and Task each have F1=1.0. The other 3 labels have gold=0.
    # Macro F1 should be 1.0 (averaged over active labels with gold > 0), not depressed to 0.4.
    assert pytest.approx(results.strict_macro_f1) == 1.0
    assert pytest.approx(results.relaxed_macro_f1) == 1.0


def test_reproducibility_git_commit_helper():
    """Verify get_git_commit returns a valid git hash string or fallback."""
    import sys
    sys.path.insert(0, "scripts")
    from evaluate_scierc import get_git_commit
    commit = get_git_commit()
    assert isinstance(commit, str)
    assert len(commit) >= 4 or commit == "unknown"



def test_real_scierc_test_file_if_present():
    """Verify parsing and structure of the real official SciERC test split if present on disk."""
    real_path = Path("data/raw/scierc/test.json")
    if not real_path.exists():
        pytest.skip("data/raw/scierc/test.json not found on disk")

    docs = load_scierc_dataset(real_path)
    assert len(docs) == 100
    for doc in docs:
        assert doc.doc_key
        assert len(doc.sentences) > 0
        gold_spans = doc.get_gold_spans(target_labels=set(EVALUATION_LABELS))
        assert isinstance(gold_spans, list)

