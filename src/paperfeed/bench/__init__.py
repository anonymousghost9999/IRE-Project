"""Paper Feed benchmarks subpackage (Phase 3.4 & Phase 7)."""

from paperfeed.bench.scierc import (
    DEFAULT_LABEL_MAP,
    EVALUATION_LABELS,
    BenchmarkResults,
    EntitySpan,
    LabelMetrics,
    SciERCDocument,
    SciERCEvaluator,
    evaluate_document_spans,
    format_markdown_report,
    load_scierc_dataset,
)

__all__ = [
    "DEFAULT_LABEL_MAP",
    "EVALUATION_LABELS",
    "BenchmarkResults",
    "EntitySpan",
    "LabelMetrics",
    "SciERCDocument",
    "SciERCEvaluator",
    "evaluate_document_spans",
    "format_markdown_report",
    "load_scierc_dataset",
]
