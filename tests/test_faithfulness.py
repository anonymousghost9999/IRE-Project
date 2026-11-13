"""Unit tests for Phase 8 explanation faithfulness audit (tests/test_faithfulness.py)."""

from pathlib import Path
import sqlite3
import numpy as np
import pytest

from paperfeed.eval.faithfulness import (
    AuditedExplanation,
    ExplanationSample,
    FaithfulnessAuditResult,
    compute_counterfactual_impact,
    sample_stratified_explanations,
    verify_grounding,
)
from paperfeed.rec.explain import Explanation
from paperfeed.rec.scoring import CandidateFeatures, ScoredCandidate, ScoringWeights


class MockDB:
    def __init__(self, papers=None, edges=None):
        self.papers = papers or {}
        self.edges_list = edges or []

    def get_paper(self, paper_id):
        return self.papers.get(paper_id)

    def get_connection(self):
        # In-memory SQLite for testing citation edges
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE paper_citations (source_paper_id TEXT, target_paper_id TEXT, source_year INTEGER)")
        for src, tgt, yr in self.edges_list:
            conn.execute("INSERT INTO paper_citations VALUES (?, ?, ?)", (src, tgt, yr))
        conn.commit()
        return conn


class MockEmbeddingStore:
    def __init__(self, vectors):
        self.vectors = vectors

    def select(self, ids):
        return np.array([self.vectors.get(i, [1.0, 0.0]) for i in ids], dtype=np.float32)


# ---------------------------------------------------------------------------
# Test Grounding Verification
# ---------------------------------------------------------------------------

def test_verify_grounding_facet_success():
    abstract = "We propose an approach for Semantic Parsing using deep models."
    db = MockDB(papers={"P1": {"abstract": abstract}})
    # "Semantic Parsing" starts at 27, ends at 43
    exp = Explanation(
        feature="facet",
        text="Addresses **Semantic Parsing**",
        evidence={
            "span": "Semantic Parsing",
            "start_char": 27,
            "end_char": 43,
            "evidence_sentence": "We propose an approach for Semantic Parsing using deep models.",
        },
    )
    is_valid, details = verify_grounding("P1", exp, db, cutoff_year=2024)
    assert is_valid is True
    assert details["exact_span_match"] is True
    assert details["span_in_sentence"] is True
    assert details["sentence_in_abstract"] is True


def test_verify_grounding_facet_mismatch():
    abstract = "We propose an approach for Semantic Parsing using deep models."
    db = MockDB(papers={"P1": {"abstract": abstract}})
    # Deliberately wrong offsets
    exp = Explanation(
        feature="facet",
        text="Addresses **Semantic Parsing**",
        evidence={
            "span": "Semantic Parsing",
            "start_char": 10,
            "end_char": 26,
            "evidence_sentence": "We propose an approach for Semantic Parsing using deep models.",
        },
    )
    is_valid, details = verify_grounding("P1", exp, db, cutoff_year=2024)
    assert is_valid is False
    assert details["exact_span_match"] is False


def test_verify_grounding_new_method():
    abstract = "We introduce Transformer-XL for Language Modeling with long dependencies."
    db = MockDB(papers={"P1": {"abstract": abstract}})
    # "Transformer-XL" start 13 end 27, "Language Modeling" start 32 end 49
    exp = Explanation(
        feature="new_method",
        text="Proposes **Transformer-XL** for **Language Modeling**",
        evidence={
            "method": {
                "span": "Transformer-XL",
                "start_char": 13,
                "end_char": 27,
                "evidence_sentence": "We introduce Transformer-XL for Language Modeling with long dependencies.",
            },
            "task": {
                "span": "Language Modeling",
                "start_char": 32,
                "end_char": 49,
                "evidence_sentence": "We introduce Transformer-XL for Language Modeling with long dependencies.",
            },
        },
    )
    is_valid, details = verify_grounding("P1", exp, db, cutoff_year=2024)
    assert is_valid is True
    assert details["method_grounded"] is True
    assert details["task_grounded"] is True


def test_verify_grounding_citation_bc():
    from paperfeed.data.ground_truth import to_uri
    # P1 (cand) and Auth1 share ref R1 and R2
    edges_data = [
        (to_uri("P1"), to_uri("R1"), 2020),
        (to_uri("P1"), to_uri("R2"), 2021),
        (to_uri("Auth1"), to_uri("R1"), 2019),
        (to_uri("Auth1"), to_uri("R2"), 2018),
        (to_uri("Auth1"), to_uri("R3"), 2018),
    ]
    db = MockDB(papers={"P1": {"abstract": "test"}}, edges=edges_data)
    exp = Explanation(
        feature="citation",
        text="Shares 2 references with your paper",
        evidence={
            "kind": "bibliographic_coupling",
            "authored_paper_id": "Auth1",
            "shared_references": 2,
        },
    )
    is_valid, details = verify_grounding("P1", exp, db, cutoff_year=2024)
    assert is_valid is True
    assert details["actual_refs"] == 2


def test_verify_grounding_persona():
    db = MockDB()
    exp = Explanation(
        feature="persona",
        text="Close to your overall research profile",
        evidence={"persona_cosine": 0.82},
    )
    is_valid, details = verify_grounding("P1", exp, db, cutoff_year=2024)
    assert is_valid is True
    assert details["persona_cosine"] == 0.82


# ---------------------------------------------------------------------------
# Test Counterfactual Impact
# ---------------------------------------------------------------------------

def test_counterfactual_impact_leaves_top_k():
    store = MockEmbeddingStore({f"P{i}": [1.0, 0.0] for i in range(1, 15)})

    # Build candidates where P1 was top because of a massive citation contribution
    scored_cands = []
    for i in range(1, 15):
        pid = f"P{i}"
        if pid == "P1":
            contribs = {"citation": 5.0, "persona": 0.1}
            score = 5.1
        else:
            contribs = {"citation": 0.0, "persona": 4.0 - 0.1 * i}
            score = 4.0 - 0.1 * i
        scored_cands.append(
            ScoredCandidate(
                paper_id=pid,
                features=None,
                scaled={},
                contributions=contribs,
                score=score,
            )
        )

    sample = ExplanationSample(
        author_id="A1",
        paper_id="P1",
        rank=1,
        dominant_feature="citation",
        explanation=Explanation("citation", "text"),
        scored_candidates=scored_cands,
    )

    cf_rank, rank_drop, is_necessary = compute_counterfactual_impact(
        sample, store, k=10, lambda_div=0.0
    )
    # When citation (5.0) is zeroed, P1 score drops from 5.1 to 0.1, falling to the bottom of the pool (> 10)
    assert is_necessary is True
    assert cf_rank is None or cf_rank > 10


def test_counterfactual_impact_retained():
    store = MockEmbeddingStore({f"P{i}": [1.0, 0.0] for i in range(1, 15)})

    # P1 has a tiny dominant feature contribution, but its base persona score dominates everyone
    scored_cands = []
    for i in range(1, 15):
        pid = f"P{i}"
        if pid == "P1":
            contribs = {"citation": 0.05, "persona": 10.0}
            score = 10.05
        else:
            contribs = {"citation": 0.0, "persona": 1.0}
            score = 1.0
        scored_cands.append(
            ScoredCandidate(
                paper_id=pid,
                features=None,
                scaled={},
                contributions=contribs,
                score=score,
            )
        )

    sample = ExplanationSample(
        author_id="A1",
        paper_id="P1",
        rank=1,
        dominant_feature="citation",
        explanation=Explanation("citation", "text"),
        scored_candidates=scored_cands,
    )

    cf_rank, rank_drop, is_necessary = compute_counterfactual_impact(
        sample, store, k=10, lambda_div=0.0
    )
    # P1 still has score 10.0, stays rank 1, so rank_drop = 0 and is_necessary is False
    assert is_necessary is False
    assert cf_rank == 1
    assert rank_drop == 0


# ---------------------------------------------------------------------------
# Test Stratified Sampling
# ---------------------------------------------------------------------------

def test_sample_stratified_explanations():
    candidates = []
    # 50 persona items across 10 users
    for i in range(50):
        candidates.append(
            ExplanationSample(
                author_id=f"User_{i % 10}",
                paper_id=f"P_{i}",
                rank=(i % 10) + 1,
                dominant_feature="persona",
                explanation=Explanation("persona", "text"),
                scored_candidates=[],
            )
        )
    # 20 citation items across 5 users
    for i in range(20):
        candidates.append(
            ExplanationSample(
                author_id=f"User_Cite_{i % 5}",
                paper_id=f"P_Cite_{i}",
                rank=(i % 10) + 1,
                dominant_feature="citation",
                explanation=Explanation("citation", "text"),
                scored_candidates=[],
            )
        )
    # 5 facet items
    for i in range(5):
        candidates.append(
            ExplanationSample(
                author_id=f"User_Facet_{i}",
                paper_id=f"P_Facet_{i}",
                rank=1,
                dominant_feature="facet",
                explanation=Explanation("facet", "text"),
                scored_candidates=[],
            )
        )

    sampled = sample_stratified_explanations(candidates, target_n=25, seed=42)
    assert len(sampled) == 25
    templates = {s.dominant_feature for s in sampled}
    assert "persona" in templates
    assert "citation" in templates
    assert "facet" in templates


# ---------------------------------------------------------------------------
# Test Report & Serialization
# ---------------------------------------------------------------------------

def test_faithfulness_audit_result_serialization(tmp_path):
    result = FaithfulnessAuditResult(
        n_sampled=100,
        n_users_in_sample=40,
        total_evaluated_recommendations=1000,
        total_covered_recommendations=980,
        explanation_coverage=0.98,
        extractive_grounding_rate=1.0,
        counterfactual_dropout_rate=0.88,
        left_top_k_count=75,
        dropped_gte_3_count=13,
        retained_near_rank_count=12,
        mean_rank_drop=6.4,
        median_rank_drop=7.0,
        template_distribution={"persona": 60, "citation": 40},
        template_proportions={"persona": 0.60, "citation": 0.40},
        audited_explanations=[],
    )
    d = result.to_dict()
    assert d["n_sampled"] == 100
    assert d["extractive_grounding_rate"] == 1.0

    md = result.to_markdown()
    assert "100.00%" in md
    assert "PASS" in md

    save_path = tmp_path / "report.json"
    result.save(save_path)
    assert save_path.exists()
