"""GLiNER Facet Extractor for scientific abstracts (Phase 3.1 / docs/plan.md §3.1).

Extracts scientific entities (task, method, dataset, metric, topic) from abstracts
with verbatim character offsets and sentence-level evidence for grounded explainability.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Default facet labels according to docs/plan.md §3.1 & docs/design.md §6.2
DEFAULT_FACET_LABELS = ["task", "method", "dataset", "metric", "topic"]
DEFAULT_MODEL_NAME = "urchade/gliner_medium-v2.1"
LOCAL_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models" / "gliner_medium-v2.1"


# A period after one of these does not end a sentence ("e.g. the", "et al. show", "Fig. 3").
_ABBREVIATIONS = frozenset({
    "e.g", "i.e", "et al", "al", "fig", "figs", "eq", "eqs", "sec", "secs", "tab", "vs", "cf", "etc",
    "approx", "resp", "no", "nos", "dr", "prof", "ref", "refs", "ca", "viz", "inc", "ltd", "st",
})
_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*\s+")
_SENTENCE_START = re.compile(r"[A-Z0-9\"'(\[]")


def split_sentences(text: str) -> List[Tuple[str, int, int]]:
    """Split text into sentences with exact character offsets: ``text[start:end] == sentence``.

    A rule-based splitter with no optional dependencies, so every machine produces the same sentences
    (and therefore the same facets). A boundary is terminal punctuation followed by whitespace and a
    sentence-initial character, unless the word before it is a known abbreviation or a single initial.
    """
    if not text or not text.strip():
        return []

    sentences: List[Tuple[str, int, int]] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        end = m.start() + len(m.group().rstrip())
        nxt = text[m.end() : m.end() + 1]
        if not nxt or not _SENTENCE_START.match(nxt):
            continue
        word = re.split(r"\s+", text[start:m.start()].rstrip().lower())[-1] if text[start:m.start()].strip() else ""
        word = word.strip("([\"'")
        # Single capital initial ("J. Smith") or a known abbreviation: not a boundary.
        if m.group().startswith(".") and (word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha())):
            continue
        sentences.append((start, end))
        start = m.end()
    sentences.append((start, len(text)))

    result: List[Tuple[str, int, int]] = []
    for s_start, s_end in sentences:
        chunk = text[s_start:s_end]
        stripped = chunk.strip()
        if not stripped:
            continue
        lead = len(chunk) - len(chunk.lstrip())
        result.append((stripped, s_start + lead, s_start + lead + len(stripped)))
    return result


class FacetExtractor:
    """Extracts typed scientific facets from paper abstracts using GLiNER."""

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        labels: Optional[List[str]] = None,
        threshold: float = 0.5,
        device: Optional[str] = None,
        strict_errors: bool = False,
    ):
        self.labels = [lbl.lower().strip() for lbl in (labels or DEFAULT_FACET_LABELS)]
        self.threshold = float(threshold)
        self.strict_errors = bool(strict_errors)
        self.error_count: int = 0
        self.grounding_dropped_count: int = 0

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if model_name_or_path:
            self.model_path = model_name_or_path
        elif LOCAL_MODEL_DIR.exists() and any(
            (LOCAL_MODEL_DIR / fname).exists()
            for fname in ("gliner_config.json", "model.safetensors", "pytorch_model.bin")
        ):
            self.model_path = str(LOCAL_MODEL_DIR)
        else:
            self.model_path = DEFAULT_MODEL_NAME

        self._model = None

    def _ensure_model_loaded(self):
        if self._model is not None:
            return

        from gliner import GLiNER

        logger.info("Loading GLiNER model from %s onto %s...", self.model_path, self.device)
        self._model = GLiNER.from_pretrained(self.model_path)
        self._model.to(self.device)
        self._model.eval()

    def batch_predict_entities(
        self,
        sentences: List[str],
        labels: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        batch_size: int = 32,
    ) -> List[List[Dict[str, Any]]]:
        """Run batched entity prediction on a list of sentences using GLiNER.

        Errors propagate: the caller decides whether to count, skip or abort (the SciERC evaluator counts
        failed sentences itself).
        """
        if not sentences:
            return []
        self._ensure_model_loaded()
        return self._model.batch_predict_entities(
            sentences,
            labels=labels or self.labels,
            threshold=self.threshold if threshold is None else threshold,
            batch_size=batch_size,
        )

    def extract_from_abstract(
        self,
        paper_id: str,
        abstract: str,
    ) -> List[Dict[str, Any]]:
        """Extract typed facets from a single paper abstract.

        Returns a list of dicts conforming to SQLite `paper_facets` schema:
        - paper_id: str
        - facet_type: str
        - facet_value: str (verbatim span)
        - canonical_id: None (populated during canonicalization)
        - evidence_sentence: str
        - start_char: int (offset in full abstract)
        - end_char: int (offset in full abstract)
        - confidence: float
        """
        if not abstract or not abstract.strip():
            return []

        self._ensure_model_loaded()

        sentences = split_sentences(abstract)
        if not sentences:
            return []

        facets: List[Dict[str, Any]] = []
        seen_spans = set()

        for sent_text, sent_start, sent_end in sentences:
            if not sent_text.strip():
                continue

            try:
                ents = self._model.predict_entities(
                    sent_text,
                    self.labels,
                    threshold=self.threshold,
                )
            except Exception as e:
                self.error_count += 1
                if self.strict_errors:
                    raise
                logger.warning("GLiNER error predicting entities for paper %s: %s", paper_id, e)
                continue

            for ent in ents:
                span_text = ent.get("text", "")
                rel_start = ent.get("start", 0)
                rel_end = ent.get("end", len(span_text))
                label = ent.get("label", "").lower().strip()
                score = float(ent.get("score", 0.0))

                if not span_text or label not in self.labels:
                    continue

                abs_start = sent_start + rel_start
                abs_end = sent_start + rel_end

                # Verbatim span alignment guard: ensure exact substring match
                actual_span = abstract[abs_start:abs_end]
                if actual_span != span_text:
                    # In case of minor offset misalignment, search in sentence
                    idx = sent_text.find(span_text)
                    if idx != -1:
                        abs_start = sent_start + idx
                        abs_end = abs_start + len(span_text)
                        actual_span = abstract[abs_start:abs_end]

                # Exact verbatim grounding check
                if actual_span != span_text:
                    self.grounding_dropped_count += 1
                    continue

                # Deduplicate identical (facet_type, abs_start, abs_end)
                span_key = (label, abs_start, abs_end)
                if span_key in seen_spans:
                    continue
                seen_spans.add(span_key)

                facets.append({
                    "paper_id": paper_id,
                    "facet_type": label,
                    "facet_value": span_text,
                    "canonical_id": None,
                    "evidence_sentence": sent_text.strip(),
                    "start_char": abs_start,
                    "end_char": abs_end,
                    "confidence": round(score, 4),
                })

        return facets

    def extract_from_papers(
        self,
        papers: List[Dict[str, str]],
        sentence_batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """Extract facets for a list of papers [{'paper_id': ..., 'abstract': ...}] using batched GPU inference."""
        if not papers:
            return []

        self._ensure_model_loaded()

        # 1. Split all abstracts into sentences and map to paper metadata
        # Item: (paper_id, abstract, sent_text, sent_start, sent_end)
        sent_items = []
        for p in papers:
            p_id = p.get("paper_id", "")
            abstract = p.get("abstract", "")
            if not p_id or not abstract or not abstract.strip():
                continue
            sentences = split_sentences(abstract)
            for sent_text, sent_start, sent_end in sentences:
                if sent_text.strip():
                    sent_items.append((p_id, abstract, sent_text, sent_start, sent_end))

        if not sent_items:
            return []

        # 2. Run batched entity prediction across all sentences
        all_sentences = [item[2] for item in sent_items]
        try:
            # Batch predict on GPU
            ents_per_sent = self._model.batch_predict_entities(
                all_sentences,
                self.labels,
                threshold=self.threshold,
                batch_size=sentence_batch_size,
            )
        except Exception as e:
            if self.strict_errors:
                raise
            logger.warning("Error during batched entity prediction: %s. Falling back to sequential.", e)
            return [f for p in papers for f in self.extract_from_abstract(p.get("paper_id", ""), p.get("abstract", ""))]

        # 3. Map entities back to verbatim abstract coordinates
        all_facets: List[Dict[str, Any]] = []
        seen_spans_by_paper = defaultdict(set)

        for (p_id, abstract, sent_text, sent_start, sent_end), ents in zip(sent_items, ents_per_sent):
            for ent in ents:
                span_text = ent.get("text", "")
                rel_start = ent.get("start", 0)
                rel_end = ent.get("end", len(span_text))
                label = ent.get("label", "").lower().strip()
                score = float(ent.get("score", 0.0))

                if not span_text or label not in self.labels:
                    continue

                abs_start = sent_start + rel_start
                abs_end = sent_start + rel_end

                # Verbatim span alignment guard
                actual_span = abstract[abs_start:abs_end]
                if actual_span != span_text:
                    idx = sent_text.find(span_text)
                    if idx != -1:
                        abs_start = sent_start + idx
                        abs_end = abs_start + len(span_text)
                        actual_span = abstract[abs_start:abs_end]

                if actual_span != span_text:
                    self.grounding_dropped_count += 1
                    continue

                span_key = (label, abs_start, abs_end)
                if span_key in seen_spans_by_paper[p_id]:
                    continue
                seen_spans_by_paper[p_id].add(span_key)

                all_facets.append({
                    "paper_id": p_id,
                    "facet_type": label,
                    "facet_value": span_text,
                    "canonical_id": None,
                    "evidence_sentence": sent_text.strip(),
                    "start_char": abs_start,
                    "end_char": abs_end,
                    "confidence": round(score, 4),
                })

        return all_facets

