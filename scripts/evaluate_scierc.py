#!/usr/bin/env python3
"""CLI script to evaluate GLiNER on the official SciERC Gold Standard (Phase 3.4 / docs/plan.md §3.4).

Usage:
    python scripts/evaluate_scierc.py [--data-path data/raw/scierc/test.json] [--out-dir results]

Outputs:
    - results/scierc_benchmark_report.json: Machine-readable Strict & Relaxed F1 metrics
    - results/scierc_benchmark_report.md: Formatted GitHub-style summary table
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

# Add project root and src/ to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from paperfeed.bench.scierc import (
    DEFAULT_LABEL_MAP,
    EVALUATION_LABELS,
    NATIVE_LABEL_MAP,
    SciERCEvaluator,
    format_markdown_report,
    load_scierc_dataset,
)
from paperfeed.common.config import load_config
from paperfeed.facets.extractor import FacetExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_scierc")


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    fcfg = cfg.get("facets", {})
    default_model = cfg.get("models", {}).get("gliner_model", "urchade/gliner_medium-v2.1")

    parser = argparse.ArgumentParser(description="Evaluate GLiNER on SciERC gold standard test split.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=project_root / "data" / "raw" / "scierc" / "test.json",
        help="Path to SciERC test split (test.json)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=project_root / "results",
        help="Output directory for benchmark reports",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="Model path or HuggingFace ID for GLiNER",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=fcfg.get("threshold", 0.5),
        help="Confidence threshold for GLiNER entity prediction",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda' or 'cpu'). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to evaluate (for testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Sentence batch size for GLiNER inference (default: 32)",
    )
    parser.add_argument(
        "--native-labels",
        action="store_true",
        help="Prompt GLiNER directly with SciERC's native labels (Task, Method, Metric, Material, OtherScientificTerm)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.data_path.exists():
        logger.error("SciERC dataset file not found at: %s", args.data_path)
        logger.error(
            "Please ensure SciERC test split is available at %s.",
            args.data_path,
        )
        return 1

    logger.info("Loading SciERC dataset from: %s", args.data_path)
    docs = load_scierc_dataset(args.data_path)
    if not docs:
        logger.error("No documents loaded from %s", args.data_path)
        return 1

    if args.limit and args.limit > 0:
        docs = docs[: args.limit]
        logger.info("Evaluation limited to first %d documents.", len(docs))

    # Determine labels and label mapping
    if args.native_labels:
        logger.info("Using SciERC native labels directly (Task, Method, Metric, Material, OtherScientificTerm)")
        label_map = NATIVE_LABEL_MAP
        active_labels = list(label_map)
    else:
        logger.info("Using production facet labels (task, method, dataset, metric, topic) mapped to SciERC ontology")
        active_labels = ["task", "method", "dataset", "metric", "topic"]
        label_map = DEFAULT_LABEL_MAP

    logger.info(
        "Initializing FacetExtractor (model=%s, threshold=%.2f, device=%s)...",
        args.model,
        args.threshold,
        args.device or "auto",
    )
    extractor = FacetExtractor(
        model_name_or_path=args.model,
        labels=active_labels,
        threshold=args.threshold,
        device=args.device,
    )

    evaluator = SciERCEvaluator(
        label_map=label_map,
        eval_labels=EVALUATION_LABELS,
        extractor=extractor,
    )

    logger.info("Running SciERC evaluation across %d documents...", len(docs))
    t0 = time.time()
    results = evaluator.evaluate(
        docs,
        batch_size=args.batch_size,
        metadata={
            "model": args.model,
            "threshold": args.threshold,
            "device": extractor.device,
            "batch_size": args.batch_size,
            "native_labels_prompted": args.native_labels,
            "evaluation_time_seconds": 0.0,
        },
    )
    elapsed = time.time() - t0
    results.metadata["evaluation_time_seconds"] = round(elapsed, 2)
    logger.info("Evaluation finished in %.2fs (%.2fs/doc)", elapsed, elapsed / max(len(docs), 1))

    # Write JSON report
    json_path = args.out_dir / "scierc_benchmark_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results.to_dict(), f, indent=2)
    logger.info("Saved machine-readable benchmark report: %s", json_path)

    # Write Markdown report
    md_content = format_markdown_report(results)
    md_path = args.out_dir / "scierc_benchmark_report.md"
    md_path.write_text(md_content, encoding="utf-8")
    logger.info("Saved formatted Markdown report: %s", md_path)

    # Print summary to stdout
    print("\n" + "=" * 70)
    print("PHASE 3.4 — SCIERC EXTRACTION BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Total Documents: {results.total_documents}")
    print(f"Total Gold Entities (Target Labels): {results.total_gold_entities}")
    print(f"Total Extracted Entities:            {results.total_predicted_entities}")
    print("-" * 70)
    print(f"Strict Span Micro-F1:   {results.strict_micro.f1 * 100:6.2f}% (P: {results.strict_micro.precision * 100:.2f}%, R: {results.strict_micro.recall * 100:.2f}%)")
    print(f"Strict Span Macro-F1:   {results.strict_macro_f1 * 100:6.2f}% (P: {results.strict_macro_precision * 100:.2f}%, R: {results.strict_macro_recall * 100:.2f}%)")
    print("-" * 70)
    print(f"Relaxed Span Micro-F1:  {results.relaxed_micro.f1 * 100:6.2f}% (P: {results.relaxed_micro.precision * 100:.2f}%, R: {results.relaxed_micro.recall * 100:.2f}%)")
    print(f"Relaxed Span Macro-F1:  {results.relaxed_macro_f1 * 100:6.2f}% (P: {results.relaxed_macro_precision * 100:.2f}%, R: {results.relaxed_macro_recall * 100:.2f}%)")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

