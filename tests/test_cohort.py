"""Unit tests for candidate evaluation author and cohort selection (Phase 1 Step 1.2 & 1.3)."""

from pathlib import Path
from typing import Any, Dict, List
import pytest
from paperfeed.data.cohort import (
    CandidateAuthor,
    CohortMiner,
    classify_history_slice,
    clean_id,
    extract_first_author,
    filter_author_plausibility,
    load_candidate_authors,
    save_candidate_authors,
    select_target_paper_b,
)


def test_classify_history_slice():
    assert classify_history_slice(10) == "warm"
    assert classify_history_slice(5) == "warm"
    assert classify_history_slice(4) == "moderate"
    assert classify_history_slice(3) == "moderate"
    assert classify_history_slice(2) == "sparse"
    assert classify_history_slice(1) == "sparse"
    assert classify_history_slice(0) == "zero"


def test_clean_id():
    assert clean_id("https://openalex.org/W12345") == "W12345"
    assert clean_id("A9876") == "A9876"
    assert clean_id("") == ""


def test_extract_first_author():
    # Multi-author paper
    work_multi = {
        "id": "W1",
        "authorships": [
            {"author": {"id": "https://openalex.org/A1", "display_name": "Author One"}, "author_position": "first"},
            {"author": {"id": "https://openalex.org/A2", "display_name": "Author Two"}, "author_position": "last"},
        ],
    }
    first = extract_first_author(work_multi)
    assert first is not None
    assert first["id"] == "A1"
    assert first["display_name"] == "Author One"

    # Single-author paper
    work_solo = {
        "id": "W2",
        "authorships": [
            {"author": {"id": "https://openalex.org/A3", "display_name": "Solo Author"}, "author_position": "middle"},
        ],
    }
    solo = extract_first_author(work_solo)
    assert solo is not None
    assert solo["id"] == "A3"

    # Empty authorships
    assert extract_first_author({}) is None


def test_select_target_paper_b():
    works = [
        {"id": "https://openalex.org/W300", "publication_date": "2024-06-01", "title": "Paper Later"},
        {"id": "https://openalex.org/W200", "publication_date": "2024-01-15", "title": "Paper Early 2"},
        {"id": "https://openalex.org/W100", "publication_date": "2024-01-15", "title": "Paper Early 1"},
    ]
    # Should pick earliest date (2024-01-15) and tie-break on lower ID ("W100" < "W200")
    selected = select_target_paper_b(works)
    assert selected["id"] == "https://openalex.org/W100"
    assert selected["title"] == "Paper Early 1"


def test_filter_author_plausibility_temporal_guard():
    # Contains future paper > 2024 -> Must reject
    leaked_history = [
        {"id": "W1", "publication_year": 2023, "primary_topic": {"field": {"id": "fields/17"}}},
        {"id": "W2", "publication_year": 2025, "primary_topic": {"field": {"id": "fields/17"}}},
    ]
    assert filter_author_plausibility(leaked_history, cutoff_year=2024) is False


def test_filter_author_plausibility_volume_and_domain():
    # Implausible productivity (>15 papers/year over career span)
    spam_history = [
        {"id": f"W_{i}", "publication_year": 2023, "primary_topic": {"field": {"id": "fields/17"}}}
        for i in range(25)
    ]
    assert filter_author_plausibility(spam_history, max_papers_per_year=15.0) is False

    # Non-CS author
    bio_history = [
        {
            "id": "W1",
            "publication_year": 2021,
            "primary_topic": {"field": {"id": "fields/13", "display_name": "Medicine"}},
        }
    ]
    assert filter_author_plausibility(bio_history, require_cs_topic=True) is False

    # Valid author
    valid_history = [
        {
            "id": "W1",
            "publication_year": 2020,
            "primary_topic": {"field": {"id": "fields/17", "display_name": "Computer Science"}},
        },
        {
            "id": "W2",
            "publication_year": 2022,
            "primary_topic": {"field": {"id": "fields/17", "display_name": "Computer Science"}},
        },
    ]
    assert filter_author_plausibility(valid_history) is True


def test_candidate_author_serialization(tmp_path: Path):
    candidates = [
        CandidateAuthor(
            author_id="A101",
            author_name="Alice Researcher",
            target_paper_b_id="W999",
            target_paper_b_title="Future Work 2024",
            target_paper_b_date="2024-02-10",
            target_paper_b_references=["W1", "W2"],
            target_paper_b_topics=["Information Retrieval"],
            authored_paper_ids_T=["W1", "W3", "W4"],
            selection_rule="earliest_publication_date_then_lowest_id",
            history_size_slice="moderate",
        )
    ]

    out_file = tmp_path / "candidates.jsonl"
    save_candidate_authors(candidates, out_file)
    assert out_file.exists()

    loaded = load_candidate_authors(out_file)
    assert len(loaded) == 1
    assert loaded[0].author_id == "A101"
    assert loaded[0].target_paper_b_id == "W999"
    assert loaded[0].history_size_slice == "moderate"
    assert loaded[0].authored_paper_ids_T == ["W1", "W3", "W4"]


class MockOpenAlexClient:
    """Mock OpenAlex client yielding canned responses for offline testing."""

    def __init__(self, target_works: List[Dict[str, Any]], history_works_by_author: Dict[str, List[Dict[str, Any]]]):
        self.target_works = target_works
        self.history_works_by_author = history_works_by_author

    def paginate_works(self, filter_str: str, **kwargs):
        if "publication_year:2025" in filter_str:
            for w in self.target_works:
                yield w
        elif "author.id:" in filter_str:
            parts = filter_str.split(",")
            author_part = [p for p in parts if p.startswith("author.id:")][0]
            author_id = author_part.split(":")[1]
            for w in self.history_works_by_author.get(author_id, []):
                yield w


def test_cohort_miner_pipeline(tmp_path: Path):
    mock_target_works = [
        {
            "id": "https://openalex.org/W_TARGET_1",
            "title": "2025 Target Paper A",
            "publication_year": 2025,
            "publication_date": "2025-03-01",
            "authorships": [{"author": {"id": "https://openalex.org/A100", "display_name": "Author 100"}, "author_position": "first"}],
            "referenced_works": ["https://openalex.org/W_REF_1"],
            "primary_topic": {"display_name": "Natural Language Processing", "field": {"id": "fields/17"}},
        }
    ]

    mock_history = {
        "A100": [
            {
                "id": "https://openalex.org/W_HIST_1",
                "publication_year": 2022,
                "primary_topic": {"field": {"id": "fields/17", "display_name": "Computer Science"}},
            },
            {
                "id": "https://openalex.org/W_HIST_2",
                "publication_year": 2024,
                "primary_topic": {"field": {"id": "fields/17", "display_name": "Computer Science"}},
            },
        ]
    }

    mock_client = MockOpenAlexClient(mock_target_works, mock_history)
    miner = CohortMiner(client=mock_client, cutoff_year=2024, target_year=2025)

    out_file = tmp_path / "candidate_authors.jsonl"
    candidates = miner.mine_candidate_authors(max_candidates=5, output_file=out_file)

    assert len(candidates) == 1
    c = candidates[0]
    assert c.author_id == "A100"
    assert c.target_paper_b_id == "W_TARGET_1"
    assert c.history_size_slice == "sparse"  # 2 papers <= T
    assert out_file.exists()

