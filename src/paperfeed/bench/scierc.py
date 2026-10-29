"""SciERC Gold Standard Benchmark Module (Phase 3.4 / docs/plan.md §3.4 / docs/design.md §13.4).

Evaluates GLiNER (and optional LLM baselines) on the official SciERC release (Luan et al., 2018):
- Loads the official SciERC test (or dev/train) split.
- Maps extraction labels to SciERC ontology:
    task   -> Task
    method -> Method
    metric -> Metric
    dataset -> Material
    topic  -> OtherScientificTerm
- Computes Strict and Relaxed Span F1 per label, as well as Macro- and Micro-averaged F1.
- Generates structured JSON and Markdown benchmark reports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

# Official label mapping defined in docs/plan.md §3.4 & docs/design.md §13.4
DEFAULT_LABEL_MAP: Dict[str, str] = {
    "task": "Task",
    "method": "Method",
    "metric": "Metric",
    "dataset": "Material",
    "topic": "OtherScientificTerm",
}

# Natural-language prompts for SciERC's native labels (lowercase, as FacetExtractor lowercases labels;
# GLiNER works better with spaced words than with the CamelCase "OtherScientificTerm").
NATIVE_LABEL_MAP: Dict[str, str] = {
    "task": "Task",
    "method": "Method",
    "metric": "Metric",
    "material": "Material",
    "other scientific term": "OtherScientificTerm",
}

# The 5 primary SciERC evaluation target labels
EVALUATION_LABELS: List[str] = [
    "Task",
    "Method",
    "Metric",
    "Material",
    "OtherScientificTerm",
]


@dataclass
class EntitySpan:
    """Represents an extracted or gold scientific entity span."""

    start_token: int  # 0-based inclusive token offset
    end_token: int    # 0-based inclusive token offset
    label: str        # e.g. "Task", "Method", "Material", etc.
    text: str = ""    # verbatim span text
    confidence: Optional[float] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntitySpan":
        return cls(**data)


@dataclass
class SciERCDocument:
    """Structured representation of one SciERC annotated document/abstract."""

    doc_key: str
    sentences: List[List[str]]  # List of sentences, each is a list of token strings
    gold_ner: List[List[Tuple[int, int, str]]]  # List of per-sentence [start_tok, end_tok, label]
    relations: Optional[List[Any]] = None
    clusters: Optional[List[Any]] = None

    @property
    def flat_tokens(self) -> List[str]:
        """Flat list of all document tokens."""
        return [tok for s in self.sentences for tok in s]

    @property
    def sentence_offsets(self) -> List[int]:
        """Cumulative start token index for each sentence in document token coordinates."""
        offsets = []
        curr = 0
        for s in self.sentences:
            offsets.append(curr)
            curr += len(s)
        return offsets

    def get_gold_spans(self, target_labels: Optional[Set[str]] = None) -> List[EntitySpan]:
        """Extract all gold EntitySpan objects across sentences with flat document token indices."""
        flat = self.flat_tokens
        spans: List[EntitySpan] = []
        for s_ner in self.gold_ner:
            for s_idx, e_idx, lbl in s_ner:
                if target_labels and lbl not in target_labels:
                    continue
                # Gold span tokens (inclusive e_idx)
                span_tokens = flat[s_idx : e_idx + 1] if s_idx < len(flat) else []
                span_text = " ".join(span_tokens)
                spans.append(
                    EntitySpan(
                        start_token=s_idx,
                        end_token=e_idx,
                        label=lbl,
                        text=span_text,
                    )
                )
        return spans


def load_scierc_dataset(file_path: Union[str, Path]) -> List[SciERCDocument]:
    """Load SciERC dataset from a JSON/JSONLines file.

    Handles both:
    1. Line-delimited JSON objects (.jsonl / .json with one object per line)
    2. A single JSON array of document objects
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"SciERC dataset file not found at {path}")

    docs: List[SciERCDocument] = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text.startswith("["):
        try:
            records = json.loads(text)
            for rec in records:
                docs.append(_parse_scierc_record(rec))
            return docs
        except json.JSONDecodeError:
            pass

    # Fallback to line-delimited JSON
    for line_idx, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            docs.append(_parse_scierc_record(rec))
        except json.JSONDecodeError as e:
            logger.warning("Error parsing SciERC line %d: %s", line_idx, e)

    logger.info("Loaded %d SciERC documents from %s", len(docs), path)
    return docs


def _parse_scierc_record(rec: Dict[str, Any]) -> SciERCDocument:
    """Helper to convert a raw dictionary into a SciERCDocument."""
    doc_key = str(rec.get("doc_key", ""))
    sentences = rec.get("sentences", [])
    raw_ner = rec.get("ner", [])

    gold_ner: List[List[Tuple[int, int, str]]] = []
    for sent_ner in raw_ner:
        parsed_sent: List[Tuple[int, int, str]] = []
        for item in sent_ner:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                s_tok, e_tok, lbl = int(item[0]), int(item[1]), str(item[2])
                parsed_sent.append((s_tok, e_tok, lbl))
        gold_ner.append(parsed_sent)

    return SciERCDocument(
        doc_key=doc_key,
        sentences=sentences,
        gold_ner=gold_ner,
        relations=rec.get("relations"),
        clusters=rec.get("clusters"),
    )


def char_to_token_offsets(
    char_start: int,
    char_end: int,
    token_char_spans: List[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    """Map character offsets [char_start, char_end) to 0-based inclusive token span (start_tok, end_tok)."""
    overlapping = [
        i
        for i, (ts, te) in enumerate(token_char_spans)
        if max(char_start, ts) < min(char_end, te)
    ]
    if not overlapping:
        return None
    return (overlapping[0], overlapping[-1])


@dataclass
class LabelMetrics:
    """Precision, Recall, and F1 counts and scores for a single entity label."""

    label: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    def compute(self) -> "LabelMetrics":
        """Compute precision, recall, and f1 from counts."""
        p_denom = self.tp + self.fp
        r_denom = self.tp + self.fn
        self.precision = self.tp / p_denom if p_denom > 0 else 0.0
        self.recall = self.tp / r_denom if r_denom > 0 else 0.0
        f1_denom = self.precision + self.recall
        self.f1 = (2.0 * self.precision * self.recall / f1_denom) if f1_denom > 0 else 0.0
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class BenchmarkResults:
    """Complete evaluation report holding strict and relaxed metrics."""

    strict_per_label: Dict[str, LabelMetrics] = field(default_factory=dict)
    relaxed_per_label: Dict[str, LabelMetrics] = field(default_factory=dict)
    strict_micro: LabelMetrics = field(default_factory=lambda: LabelMetrics("micro"))
    relaxed_micro: LabelMetrics = field(default_factory=lambda: LabelMetrics("micro"))
    strict_macro_f1: float = 0.0
    relaxed_macro_f1: float = 0.0
    strict_macro_precision: float = 0.0
    relaxed_macro_precision: float = 0.0
    strict_macro_recall: float = 0.0
    relaxed_macro_recall: float = 0.0
    total_gold_entities: int = 0
    total_predicted_entities: int = 0
    total_documents: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_documents": self.total_documents,
            "total_gold_entities": self.total_gold_entities,
            "total_predicted_entities": self.total_predicted_entities,
            "strict_span_matching": {
                "micro": self.strict_micro.to_dict(),
                "macro_f1": round(self.strict_macro_f1, 4),
                "macro_precision": round(self.strict_macro_precision, 4),
                "macro_recall": round(self.strict_macro_recall, 4),
                "per_label": {lbl: m.to_dict() for lbl, m in self.strict_per_label.items()},
            },
            "relaxed_span_matching": {
                "micro": self.relaxed_micro.to_dict(),
                "macro_f1": round(self.relaxed_macro_f1, 4),
                "macro_precision": round(self.relaxed_macro_precision, 4),
                "macro_recall": round(self.relaxed_macro_recall, 4),
                "per_label": {lbl: m.to_dict() for lbl, m in self.relaxed_per_label.items()},
            },
            "metadata": self.metadata,
        }


def compute_span_overlap_iou(span1: EntitySpan, span2: EntitySpan) -> Tuple[int, float]:
    """Compute token-level overlap and Intersection over Union (IoU) between two spans."""
    overlap = max(0, min(span1.end_token, span2.end_token) - max(span1.start_token, span2.start_token) + 1)
    if overlap <= 0:
        return 0, 0.0
    len1 = span1.end_token - span1.start_token + 1
    len2 = span2.end_token - span2.start_token + 1
    union = len1 + len2 - overlap
    iou = overlap / union if union > 0 else 0.0
    return overlap, iou


def evaluate_document_spans(
    gold_spans: List[EntitySpan],
    pred_spans: List[EntitySpan],
    eval_labels: List[str],
) -> Tuple[Dict[str, Tuple[int, int, int]], Dict[str, Tuple[int, int, int]]]:
    """Evaluate strict and relaxed span matching for one document.

    Returns:
        (strict_counts, relaxed_counts) where each is {label: (tp, fp, fn)}
    """
    strict_counts: Dict[str, Tuple[int, int, int]] = {}
    relaxed_counts: Dict[str, Tuple[int, int, int]] = {}

    for lbl in eval_labels:
        gold_lbl = [g for g in gold_spans if g.label == lbl]
        pred_lbl = [p for p in pred_spans if p.label == lbl]

        # --- 1. Strict Span Matching (exact token boundary + label) ---
        gold_strict_keys = {(g.start_token, g.end_token) for g in gold_lbl}
        pred_strict_keys = {(p.start_token, p.end_token) for p in pred_lbl}

        strict_tp = len(gold_strict_keys & pred_strict_keys)
        strict_fp = len(pred_lbl) - strict_tp
        strict_fn = len(gold_lbl) - strict_tp
        strict_counts[lbl] = (strict_tp, strict_fp, strict_fn)

        # --- 2. Relaxed Span Matching (overlapping token boundary + label) ---
        # Bipartite greedy matching by IoU to ensure each gold entity matches at most 1 pred
        candidate_pairs: List[Tuple[float, int, int]] = []
        for p_idx, p in enumerate(pred_lbl):
            for g_idx, g in enumerate(gold_lbl):
                overlap, iou = compute_span_overlap_iou(p, g)
                if overlap > 0:
                    candidate_pairs.append((iou, p_idx, g_idx))

        # Sort descending by IoU
        candidate_pairs.sort(key=lambda x: x[0], reverse=True)

        matched_preds: Set[int] = set()
        matched_golds: Set[int] = set()
        relaxed_tp = 0

        for iou, p_idx, g_idx in candidate_pairs:
            if p_idx not in matched_preds and g_idx not in matched_golds:
                matched_preds.add(p_idx)
                matched_golds.add(g_idx)
                relaxed_tp += 1

        relaxed_fp = len(pred_lbl) - relaxed_tp
        relaxed_fn = len(gold_lbl) - relaxed_tp
        relaxed_counts[lbl] = (relaxed_tp, relaxed_fp, relaxed_fn)

    return strict_counts, relaxed_counts


class SciERCEvaluator:
    """Benchmark harness for evaluating scientific entity extraction on SciERC."""

    def __init__(
        self,
        label_map: Optional[Dict[str, str]] = None,
        eval_labels: Optional[List[str]] = None,
        extractor: Optional[Any] = None,
    ) -> None:
        self.label_map = label_map or DEFAULT_LABEL_MAP
        self.eval_labels = eval_labels or EVALUATION_LABELS
        self.extractor = extractor

    def predict_document_spans(self, doc: SciERCDocument) -> List[EntitySpan]:
        """Extract entities from a SciERCDocument using the attached extractor.

        Processes sentence-by-sentence to preserve SciERC token alignment.
        """
        if self.extractor is None:
            raise ValueError("No extractor provided to SciERCEvaluator.")

        predicted_spans: List[EntitySpan] = []
        flat = doc.flat_tokens
        offsets = doc.sentence_offsets

        for s_idx, (tokens, s_offset) in enumerate(zip(doc.sentences, offsets)):
            if not tokens:
                continue

            sent_text = " ".join(tokens)
            token_char_spans: List[Tuple[int, int]] = []
            curr = 0
            for tok in tokens:
                start = curr
                end = curr + len(tok)
                token_char_spans.append((start, end))
                curr = end + 1

            # Extract raw entities from this sentence
            raw_entities = []
            if hasattr(self.extractor, "extract_from_abstract"):
                # Use FacetExtractor API
                facets = self.extractor.extract_from_abstract(f"{doc.doc_key}_s{s_idx}", sent_text)
                for f in facets:
                    raw_entities.append({
                        "start": f["start_char"],
                        "end": f["end_char"],
                        "label": f["facet_type"],
                        "text": f["facet_value"],
                        "confidence": f.get("confidence"),
                    })
            elif hasattr(self.extractor, "predict_entities"):
                # Direct GLiNER model API
                labels_to_query = list(self.label_map.keys())
                ents = self.extractor.predict_entities(sent_text, labels_to_query)
                for e in ents:
                    raw_entities.append({
                        "start": e["start"],
                        "end": e["end"],
                        "label": e["label"],
                        "text": e["text"],
                        "confidence": e.get("score"),
                    })
            elif callable(self.extractor):
                # Generic callable fn(text) -> list of entity dicts
                raw_entities = self.extractor(sent_text)
            else:
                raise TypeError(f"Unsupported extractor type: {type(self.extractor)}")

            for ent in raw_entities:
                c_start = ent["start"]
                c_end = ent["end"]
                raw_label = ent["label"].lower().strip()

                # Map to SciERC ontology
                mapped_label = self.label_map.get(raw_label, raw_label)
                if mapped_label not in self.eval_labels:
                    continue

                tok_span = char_to_token_offsets(c_start, c_end, token_char_spans)
                if tok_span is None:
                    continue

                rel_tok_start, rel_tok_end = tok_span
                doc_tok_start = s_offset + rel_tok_start
                doc_tok_end = s_offset + rel_tok_end

                span_tokens = flat[doc_tok_start : doc_tok_end + 1]
                verbatim_text = " ".join(span_tokens)

                predicted_spans.append(
                    EntitySpan(
                        start_token=doc_tok_start,
                        end_token=doc_tok_end,
                        label=mapped_label,
                        text=verbatim_text,
                        confidence=ent.get("confidence"),
                    )
                )

        return predicted_spans

    def evaluate(
        self,
        docs: List[SciERCDocument],
        predictions: Optional[Dict[str, List[EntitySpan]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResults:
        """Run full evaluation on a list of SciERCDocuments."""
        strict_accum: Dict[str, List[int]] = {lbl: [0, 0, 0] for lbl in self.eval_labels}
        relaxed_accum: Dict[str, List[int]] = {lbl: [0, 0, 0] for lbl in self.eval_labels}

        total_gold = 0
        total_pred = 0

        target_labels_set = set(self.eval_labels)

        for doc in docs:
            gold_spans = doc.get_gold_spans(target_labels=target_labels_set)
            total_gold += len(gold_spans)

            if predictions is not None and doc.doc_key in predictions:
                pred_spans = predictions[doc.doc_key]
            else:
                pred_spans = self.predict_document_spans(doc)

            # Filter pred_spans to target labels
            pred_spans = [p for p in pred_spans if p.label in target_labels_set]
            total_pred += len(pred_spans)

            s_counts, r_counts = evaluate_document_spans(gold_spans, pred_spans, self.eval_labels)

            for lbl in self.eval_labels:
                tp_s, fp_s, fn_s = s_counts[lbl]
                strict_accum[lbl][0] += tp_s
                strict_accum[lbl][1] += fp_s
                strict_accum[lbl][2] += fn_s

                tp_r, fp_r, fn_r = r_counts[lbl]
                relaxed_accum[lbl][0] += tp_r
                relaxed_accum[lbl][1] += fp_r
                relaxed_accum[lbl][2] += fn_r

        # Compute per-label metrics
        strict_per_label: Dict[str, LabelMetrics] = {}
        relaxed_per_label: Dict[str, LabelMetrics] = {}

        for lbl in self.eval_labels:
            tp_s, fp_s, fn_s = strict_accum[lbl]
            m_s = LabelMetrics(label=lbl, tp=tp_s, fp=fp_s, fn=fn_s).compute()
            strict_per_label[lbl] = m_s

            tp_r, fp_r, fn_r = relaxed_accum[lbl]
            m_r = LabelMetrics(label=lbl, tp=tp_r, fp=fp_r, fn=fn_r).compute()
            relaxed_per_label[lbl] = m_r

        # Micro metrics
        tp_s_tot = sum(m.tp for m in strict_per_label.values())
        fp_s_tot = sum(m.fp for m in strict_per_label.values())
        fn_s_tot = sum(m.fn for m in strict_per_label.values())
        strict_micro = LabelMetrics("micro", tp_s_tot, fp_s_tot, fn_s_tot).compute()

        tp_r_tot = sum(m.tp for m in relaxed_per_label.values())
        fp_r_tot = sum(m.fp for m in relaxed_per_label.values())
        fn_r_tot = sum(m.fn for m in relaxed_per_label.values())
        relaxed_micro = LabelMetrics("micro", tp_r_tot, fp_r_tot, fn_r_tot).compute()

        # Macro metrics (average across evaluation labels)
        strict_macro_f1 = (
            sum(m.f1 for m in strict_per_label.values()) / len(strict_per_label)
            if strict_per_label
            else 0.0
        )
        strict_macro_p = (
            sum(m.precision for m in strict_per_label.values()) / len(strict_per_label)
            if strict_per_label
            else 0.0
        )
        strict_macro_r = (
            sum(m.recall for m in strict_per_label.values()) / len(strict_per_label)
            if strict_per_label
            else 0.0
        )

        relaxed_macro_f1 = (
            sum(m.f1 for m in relaxed_per_label.values()) / len(relaxed_per_label)
            if relaxed_per_label
            else 0.0
        )
        relaxed_macro_p = (
            sum(m.precision for m in relaxed_per_label.values()) / len(relaxed_per_label)
            if relaxed_per_label
            else 0.0
        )
        relaxed_macro_r = (
            sum(m.recall for m in relaxed_per_label.values()) / len(relaxed_per_label)
            if relaxed_per_label
            else 0.0
        )

        return BenchmarkResults(
            strict_per_label=strict_per_label,
            relaxed_per_label=relaxed_per_label,
            strict_micro=strict_micro,
            relaxed_micro=relaxed_micro,
            strict_macro_f1=strict_macro_f1,
            relaxed_macro_f1=relaxed_macro_f1,
            strict_macro_precision=strict_macro_p,
            relaxed_macro_precision=relaxed_macro_p,
            strict_macro_recall=strict_macro_r,
            relaxed_macro_recall=relaxed_macro_r,
            total_gold_entities=total_gold,
            total_predicted_entities=total_pred,
            total_documents=len(docs),
            metadata=metadata or {},
        )


def format_markdown_report(results: BenchmarkResults) -> str:
    """Format benchmark results into a GitHub Flavored Markdown report."""
    lines: List[str] = [
        "# SciERC Extraction Benchmark Report (Phase 3.4)",
        "",
        "> **Dataset:** SciERC (Luan et al., 2018) Official Evaluation Split  ",
        f"> **Total Documents:** {results.total_documents}  ",
        f"> **Total Gold Entities (Evaluated Labels):** {results.total_gold_entities}  ",
        f"> **Total Extracted Entities:** {results.total_predicted_entities}  ",
        "",
        "## 1. Summary Performance Metrics",
        "",
        "| Metric Track | Strict Precision | Strict Recall | Strict F1 | Relaxed Precision | Relaxed Recall | Relaxed F1 |",
        "|---|---|---|---|---|---|---|",
        (
            f"| **Micro-Average** | {results.strict_micro.precision * 100:.2f}% | "
            f"{results.strict_micro.recall * 100:.2f}% | **{results.strict_micro.f1 * 100:.2f}%** | "
            f"{results.relaxed_micro.precision * 100:.2f}% | {results.relaxed_micro.recall * 100:.2f}% | "
            f"**{results.relaxed_micro.f1 * 100:.2f}%** |"
        ),
        (
            f"| **Macro-Average** | {results.strict_macro_precision * 100:.2f}% | "
            f"{results.strict_macro_recall * 100:.2f}% | **{results.strict_macro_f1 * 100:.2f}%** | "
            f"{results.relaxed_macro_precision * 100:.2f}% | {results.relaxed_macro_recall * 100:.2f}% | "
            f"**{results.relaxed_macro_f1 * 100:.2f}%** |"
        ),
        "",
        "---",
        "",
        "## 2. Per-Label Breakdown",
        "",
        "### A. Strict Span Matching (Exact Token Boundaries)",
        "",
        "| Label | Gold Count | Pred Count | TP | FP | FN | Precision | Recall | Strict F1 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for lbl in EVALUATION_LABELS:
        m = results.strict_per_label.get(lbl, LabelMetrics(lbl))
        gold_c = m.tp + m.fn
        pred_c = m.tp + m.fp
        lines.append(
            f"| `{lbl}` | {gold_c} | {pred_c} | {m.tp} | {m.fp} | {m.fn} | "
            f"{m.precision * 100:.2f}% | {m.recall * 100:.2f}% | **{m.f1 * 100:.2f}%** |"
        )

    lines.extend([
        "",
        "### B. Relaxed Span Matching (Overlapping Token Boundaries)",
        "",
        "| Label | Gold Count | Pred Count | TP | FP | FN | Precision | Recall | Relaxed F1 |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    for lbl in EVALUATION_LABELS:
        m = results.relaxed_per_label.get(lbl, LabelMetrics(lbl))
        gold_c = m.tp + m.fn
        pred_c = m.tp + m.fp
        lines.append(
            f"| `{lbl}` | {gold_c} | {pred_c} | {m.tp} | {m.fp} | {m.fn} | "
            f"{m.precision * 100:.2f}% | {m.recall * 100:.2f}% | **{m.f1 * 100:.2f}%** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Notes & Methodology",
        "",
        "> [!NOTE]",
        "> - **Label Mapping:** `task -> Task`, `method -> Method`, `metric -> Metric`, `dataset -> Material`, `topic -> OtherScientificTerm`.",
        "> - **Ontology Nuance:** As noted in `docs/design.md §13.4`, `Material` and `OtherScientificTerm` in SciERC are broader than our production `dataset` and `topic` facets.",
        "> - **SciERC `Generic` Category:** SciERC includes anaphoric/pronoun references (`Generic`, e.g. *'this method'*), which are omitted from scientific entity extraction.",
        "",
    ])

    return "\n".join(lines)

