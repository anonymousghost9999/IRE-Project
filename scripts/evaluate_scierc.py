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
from typing import Any, Dict

# Add project root and src/ to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from paperfeed.bench.scierc import (
    DEFAULT_LABEL_MAP,
    EVALUATION_LABELS,
    NATIVE_LABEL_MAP,
    SciERCEvaluator,
    filter_predictions_by_threshold,
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


def get_git_commit() -> str:
    """Return the current git commit short hash, or 'unknown'."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    fcfg = cfg.get("facets", {})
    local_model = project_root / "models" / "gliner_medium-v2.1"
    has_local_weights = local_model.exists() and any(
        (local_model / fname).exists()
        for fname in ("gliner_config.json", "model.safetensors", "pytorch_model.bin")
    )
    default_model = str(local_model) if has_local_weights else cfg.get("models", {}).get("gliner_model", "urchade/gliner_medium-v2.1")

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
    parser.add_argument(
        "--strict-errors",
        action="store_true",
        help="Abort benchmark immediately if any sentence extraction fails",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Comma-separated confidence thresholds to sweep (e.g. '0.3,0.4,0.5,0.6,0.7')",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.data_path.exists():
        logger.error("SciERC dataset file not found at: %s", args.data_path)
        logger.error(
            "\n"
            "================================================================================\n"
            "SciERC Dataset Not Found\n"
            "================================================================================\n"
            "The benchmark evaluation requires the official SciERC test split (Luan et al., 2018).\n"
            "\n"
            "Instructions to obtain and place the SciERC dataset:\n"
            "1. Download the processed SciERC dataset:\n"
            "   - Official source (UW): http://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz\n"
            "   - Or Hugging Face mirror: https://huggingface.co/datasets/sthoran/scierc_processed_data\n"
            "2. Place the test file at the expected location:\n"
            f"   {args.data_path}\n"
            "   (e.g., mkdir -p data/raw/scierc && curl -L -o data/raw/scierc/test.json \\\n"
            "    https://huggingface.co/datasets/sthoran/scierc_processed_data/resolve/main/test.json)\n"
            "3. Re-run: python scripts/evaluate_scierc.py\n"
            "================================================================================\n"
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

    # Check if a threshold sweep was requested
    sweep_thresholds = None
    initial_threshold = args.threshold
    if args.sweep:
        sweep_thresholds = sorted([float(x.strip()) for x in args.sweep.split(",") if x.strip()])
        if sweep_thresholds:
            initial_threshold = sweep_thresholds[0]
            logger.info("Threshold sweep requested across: %s (min: %.2f)", sweep_thresholds, initial_threshold)

    logger.info(
        "Initializing FacetExtractor (model=%s, threshold=%.2f, device=%s)...",
        args.model,
        initial_threshold,
        args.device or "auto",
    )
    extractor = FacetExtractor(
        model_name_or_path=args.model,
        labels=active_labels,
        threshold=initial_threshold,
        device=args.device,
        strict_errors=args.strict_errors,
    )

    evaluator = SciERCEvaluator(
        label_map=label_map,
        eval_labels=EVALUATION_LABELS,
        extractor=extractor,
        strict_errors=args.strict_errors,
    )

    logger.info("Running SciERC evaluation across %d documents...", len(docs))
    t0 = time.time()

    git_commit = get_git_commit()
    # Normalize model path to a portable relative path or HuggingFace ID
    model_repr = str(args.model)
    try:
        mpath = Path(model_repr)
        if mpath.is_absolute():
            try:
                rel = mpath.relative_to(project_root)
                model_repr = str(rel).replace("\\", "/")
            except ValueError:
                pass
    except Exception:
        pass

    base_metadata: Dict[str, Any] = {
        "model": model_repr,
        "device": extractor.device,
        "split_file": str(args.data_path.name),
        "document_count": len(docs),
        "git_commit": git_commit,
        "label_prompts": active_labels,
        "label_map": label_map,
        "batch_size": args.batch_size,
        "native_labels_prompted": args.native_labels,
    }

    if sweep_thresholds:
        # Extract base predictions once at minimum threshold
        logger.info("Extracting base predictions once at minimum threshold %.2f...", initial_threshold)
        base_predictions = evaluator.predict_batch_documents(docs, batch_size=args.batch_size)
        sweep_rows = []
        primary_results = None

        for t in sweep_thresholds:
            t_preds = filter_predictions_by_threshold(base_predictions, t)
            res_t = evaluator.evaluate(
                docs,
                predictions=t_preds,
                metadata={
                    **base_metadata,
                    "threshold": t,
                },
            )
            sweep_rows.append({
                "threshold": t,
                "strict_precision": round(res_t.strict_micro.precision, 4),
                "strict_recall": round(res_t.strict_micro.recall, 4),
                "strict_f1": round(res_t.strict_micro.f1, 4),
                "relaxed_precision": round(res_t.relaxed_micro.precision, 4),
                "relaxed_recall": round(res_t.relaxed_micro.recall, 4),
                "relaxed_f1": round(res_t.relaxed_micro.f1, 4),
            })
            if abs(t - args.threshold) < 1e-4 or primary_results is None:
                primary_results = res_t

        results = primary_results or res_t
        results.threshold_sweep = sweep_rows
    else:
        results = evaluator.evaluate(
            docs,
            batch_size=args.batch_size,
            metadata={
                **base_metadata,
                "threshold": args.threshold,
                "evaluation_time_seconds": 0.0,
            },
        )
    elapsed = time.time() - t0
    results.metadata["evaluation_time_seconds"] = round(elapsed, 2)
    logger.info("Evaluation finished in %.2fs (%.2fs/doc)", elapsed, elapsed / max(len(docs), 1))

    # Write JSON report
    report_stem = "scierc_benchmark_report_native" if args.native_labels else "scierc_benchmark_report"
    json_path = args.out_dir / f"{report_stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results.to_dict(), f, indent=2)
    logger.info("Saved machine-readable benchmark report: %s", json_path)

    # Write Markdown report
    md_content = format_markdown_report(results)
    md_path = args.out_dir / f"{report_stem}.md"
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
