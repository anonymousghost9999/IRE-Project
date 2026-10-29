"""Unit tests for Phase 3 scientific facet extraction, canonicalization, and persistence."""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paperfeed.facets.canonical import FacetCanonicalizer, normalize_span
from paperfeed.facets.extractor import FacetExtractor, split_sentences_spacy
from paperfeed.facets.store import FacetStore


def test_split_sentences_spacy_grounding():
    """Verify sentence splitter produces exact verbatim offsets."""
    text = (
        "We introduce Graph Convolutional Networks (GCN) for semi-supervised classification. "
        "Our model scales linearly in the number of graph edges! "
        "Experiments on Cora demonstrate state-of-the-art accuracy."
    )
    sents = split_sentences_spacy(text)
    assert len(sents) >= 3

    for sent_text, start_char, end_char in sents:
        assert text[start_char:end_char] == sent_text
        assert len(sent_text.strip()) > 0


def test_normalize_span():
    """Verify span normalization strips punctuation and lowercases cleanly."""
    assert normalize_span("  BERT Model.  ") == "bert model"
    assert normalize_span('"Dense Passage Retriever"') == "dense passage retriever"
    assert normalize_span("(SimCLR)") == "simclr"
    assert normalize_span("Contrastive Learning") == "contrastive learning"


def test_facet_canonicalizer_mock():
    """Verify canonicalizer clusters identical and variant spans under one canonical_id."""
    canonicalizer = FacetCanonicalizer(distance_threshold=0.3)
    # Stub embedder to return deterministic embeddings
    mock_embedder = MagicMock()
    # "bert" and "bert model" close; "resnet" far
    def fake_encode(spans, **kwargs):
        import numpy as np
        vecs = []
        for s in spans:
            if "bert" in s:
                vecs.append([1.0, 0.0, 0.0])
            elif "resnet" in s:
                vecs.append([0.0, 1.0, 0.0])
            else:
                vecs.append([0.0, 0.0, 1.0])
        return np.array(vecs, dtype=np.float32)

    mock_embedder.encode.side_effect = fake_encode
    canonicalizer._embedder = mock_embedder

    facets = [
        {"paper_id": "W1", "facet_type": "method", "facet_value": "BERT", "confidence": 0.95},
        {"paper_id": "W2", "facet_type": "method", "facet_value": "bert", "confidence": 0.88},
        {"paper_id": "W3", "facet_type": "method", "facet_value": "BERT model", "confidence": 0.91},
        {"paper_id": "W4", "facet_type": "method", "facet_value": "ResNet", "confidence": 0.92},
        {"paper_id": "W1", "facet_type": "dataset", "facet_value": "SQuAD", "confidence": 0.96},
    ]

    updated_facets, canon_defs = canonicalizer.canonicalize(facets)

    # 1. Check all facets have canonical_id populated
    for f in updated_facets:
        assert f["canonical_id"] is not None
        assert f["canonical_id"].startswith("canon_")

    # 2. Check "BERT", "bert", and "BERT model" share the same canonical_id
    bert_canon_ids = {
        f["canonical_id"] for f in updated_facets if "bert" in f["facet_value"].lower()
    }
    assert len(bert_canon_ids) == 1, f"Expected 1 cluster for BERT variants, got {bert_canon_ids}"

    # 3. Check ResNet has a distinct canonical_id
    resnet_canon_id = next(f["canonical_id"] for f in updated_facets if f["facet_value"] == "ResNet")
    assert resnet_canon_id not in bert_canon_ids

    # 4. Check dataset facet is grouped under dataset namespace
    squad_facet = next(f for f in updated_facets if f["facet_value"] == "SQuAD")
    assert squad_facet["canonical_id"].startswith("canon_dataset_")


def test_facet_store_crud():
    """Verify SQLite round-trip storage and retrieval for paper_facets and facet_canon."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_paperfeed.sqlite3"
        schema_path = Path(__file__).resolve().parent.parent / "src" / "paperfeed" / "data" / "schema.sql"

        # Initialize schema
        conn = sqlite3.connect(str(db_path))
        with open(schema_path, "r", encoding="utf-8") as f:
            conn.executescript(f.read())

        # Insert dummy paper
        conn.execute("""
            INSERT INTO papers (paper_id, title, abstract, year, in_candidates)
            VALUES ('W100', 'Test Title', 'Test Abstract text for validation.', 2024, 1)
        """)
        conn.commit()
        conn.close()

        store = FacetStore(db_path)

        # 1. Save canonical clusters
        canon_defs = [
            {"canonical_id": "canon_method_001", "facet_type": "method", "canonical_name": "BERT"},
            {"canonical_id": "canon_task_001", "facet_type": "task", "canonical_name": "Question Answering"},
        ]
        store.save_canonical_clusters(canon_defs)
        assert store.count_canonical() == 2

        clusters = store.get_canonical_clusters()
        assert "canon_method_001" in clusters
        assert clusters["canon_method_001"]["canonical_name"] == "BERT"

        # 2. Save paper facets
        facets = [
            {
                "paper_id": "W100",
                "facet_type": "method",
                "facet_value": "BERT",
                "canonical_id": "canon_method_001",
                "evidence_sentence": "Test Abstract text for validation.",
                "start_char": 5,
                "end_char": 9,
                "confidence": 0.94,
            },
            {
                "paper_id": "W100",
                "facet_type": "task",
                "facet_value": "Question Answering",
                "canonical_id": "canon_task_001",
                "evidence_sentence": "Test Abstract text for validation.",
                "start_char": 10,
                "end_char": 28,
                "confidence": 0.89,
            },
        ]
        store.save_facets(facets)
        assert store.count_facets() == 2

        # 3. Retrieve by paper_id
        retrieved = store.get_facets_by_paper_id("W100")
        assert len(retrieved) == 2
        assert retrieved[0]["facet_value"] == "BERT"
        assert retrieved[0]["canonical_id"] == "canon_method_001"

        # 4. Retrieve by batch paper_ids
        batch_retrieved = store.get_facets_by_paper_ids(["W100", "W999"])
        assert len(batch_retrieved["W100"]) == 2
        assert len(batch_retrieved["W999"]) == 0


def test_extractor_verbatim_grounding():
    """Verify FacetExtractor guarantees 100% extractive grounding on verbatim spans."""
    abstract = (
        "We introduce SimCLR for contrastive learning of visual representations. "
        "We evaluate on the ImageNet dataset with top-1 classification accuracy."
    )

    extractor = FacetExtractor(threshold=0.5)

    # Stub GLiNER model
    mock_model = MagicMock()
    mock_model.predict_entities.side_effect = [
        # Sentence 1: "We introduce SimCLR for contrastive learning of visual representations. "
        [
            {"text": "SimCLR", "start": 13, "end": 19, "label": "method", "score": 0.96},
            {"text": "contrastive learning", "start": 24, "end": 44, "label": "task", "score": 0.91},
        ],
        # Sentence 2: "We evaluate on the ImageNet dataset with top-1 classification accuracy."
        [
            {"text": "ImageNet", "start": 19, "end": 27, "label": "dataset", "score": 0.95},
            {"text": "top-1 classification accuracy", "start": 41, "end": 70, "label": "metric", "score": 0.88},
        ],
    ]
    extractor._model = mock_model

    facets = extractor.extract_from_abstract("W200", abstract)
    assert len(facets) == 4

    # Strict grounding check: abstract[start_char:end_char] MUST equal facet_value
    for f in facets:
        val = f["facet_value"]
        start = f["start_char"]
        end = f["end_char"]
        evidence = f["evidence_sentence"]

        assert abstract[start:end] == val, f"Grounding mismatch: '{abstract[start:end]}' != '{val}'"
        assert evidence in abstract, f"Evidence sentence '{evidence}' not in abstract"

