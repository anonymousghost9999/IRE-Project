"""GLiNER Facet Extractor for scientific abstracts (Phase 3.1 / docs/plan.md §3.1).

Extracts scientific entities (task, method, dataset, metric, topic) from abstracts
with verbatim character offsets and sentence-level evidence for grounded explainability.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Default facet labels according to docs/plan.md §3.1 & docs/design.md §6.2
DEFAULT_FACET_LABELS = ["task", "method", "dataset", "metric", "topic"]
DEFAULT_MODEL_NAME = "urchade/gliner_medium-v2.1"
LOCAL_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models" / "gliner_medium-v2.1"


def split_sentences_spacy(text: str) -> List[Tuple[str, int, int]]:
    """Split text into sentences preserving verbatim character offsets.

    Returns a list of (sentence_text, start_char, end_char) tuples.
    """
    if not text or not text.strip():
        return []

    try:
        import spacy
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        doc = nlp(text)
        sentences = [(sent.text, sent.start_char, sent.end_char) for sent in doc.sents]
        if sentences:
            return sentences
    except Exception as e:
        logger.debug("spacy sentencizer fallback: %s", e)

    # Fallback rule-based splitter preserving offsets
    sentences = []
    # Match sentences ending with period, exclamation, or question mark followed by whitespace or end
    pattern = re.compile(r'([^.!?]+[.!?]+(?:\s+|$)|[^.!?]+$)')
    for m in pattern.finditer(text):
        s_text = m.group()
        if s_text.strip():
            sentences.append((s_text, m.start(), m.end()))

    if not sentences:
        sentences = [(text, 0, len(text))]
    return sentences


class FacetExtractor:
    """Extracts typed scientific facets from paper abstracts using GLiNER."""

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        labels: Optional[List[str]] = None,
        threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        self.labels = [lbl.lower().strip() for lbl in (labels or DEFAULT_FACET_LABELS)]
        self.threshold = float(threshold)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if model_name_or_path:
            self.model_path = model_name_or_path
        elif LOCAL_MODEL_DIR.exists() and (LOCAL_MODEL_DIR / "model.safetensors").exists():
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

        sentences = split_sentences_spacy(abstract)
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
    ) -> List[Dict[str, Any]]:
        """Extract facets for a list of papers [{'paper_id': ..., 'abstract': ...}]."""
        all_facets: List[Dict[str, Any]] = []
        for p in papers:
            p_id = p.get("paper_id", "")
            abstract = p.get("abstract", "")
            if not p_id or not abstract:
                continue
            facets = self.extract_from_abstract(p_id, abstract)
            all_facets.extend(facets)
        return all_facets

