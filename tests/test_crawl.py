"""Unit tests for Corpus Expansion Crawler (Phase 1, Step 1.4)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest
from paperfeed.data.cohort import CandidateAuthor
from paperfeed.data.crawl import CorpusCrawler
from paperfeed.data.db import DatabaseManager


class MockOpenAlexCrawlerClient:
    """Mock OpenAlex client for crawler testing without live network calls."""

    def __init__(self, works_db: Dict[str, Dict[str, Any]], raw_cache_dir: Optional[Path] = None) -> None:
        self.works_db = works_db
        self.raw_cache_dir = raw_cache_dir or Path("/dummy/raw")

    def fetch_work(self, work_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        clean = work_id.replace("https://openalex.org/", "").strip()
        return self.works_db.get(clean) or self.works_db.get(work_id)

    def fetch_works_batch(self, work_ids: List[str], **kwargs):
        for wid in work_ids:
            work = self.fetch_work(wid)
            if work:
                yield work

    def paginate_works(self, filter_str: str, **kwargs):
        for w in self.works_db.values():
            if "cites:" in filter_str:
                # Mock citing works
                if "W_AUTH_1" in filter_str and w.get("id") == "https://openalex.org/W_HOP1_1":
                    yield w
            elif "primary_topic.id:" in filter_str:
                yield w
            elif "primary_topic.subfield.id:" in filter_str or "primary_topic.field.id:" in filter_str:
                yield w


@pytest.fixture
def sample_candidate_authors_file(tmp_path: Path) -> Path:
    candidates = [
        CandidateAuthor(
            author_id="A501",
            author_name="Alice Researcher",
            target_paper_b_id="W_FUTURE_TARGET_B",
            target_paper_b_title="Target Future Paper 2025",
            target_paper_b_date="2025-02-15",
            target_paper_b_references=["W_AUTH_1", "W_FOUNDATION_1"],
            target_paper_b_topics=["Information Retrieval"],
            authored_paper_ids_T=["W_AUTH_1"],
            selection_rule="earliest_publication_date_then_lowest_id",
            history_size_slice="sparse",
        )
    ]
    file_path = tmp_path / "candidate_authors.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c.to_dict()) + "\n")
    return file_path


@pytest.fixture
def mock_crawler_corpus() -> Dict[str, Dict[str, Any]]:
    return {
        "W_AUTH_1": {
            "id": "https://openalex.org/W_AUTH_1",
            "title": "Authored Paper 2023",
            "publication_year": 2023,
            "publication_date": "2023-04-10",
            "type": "article",
            "language": "en",
            "authorships": [{"author": {"id": "https://openalex.org/A501", "display_name": "Alice Researcher"}, "author_position": "first"}],
            "referenced_works": ["https://openalex.org/W_FOUNDATION_1"],
            "counts_by_year": [{"year": 2023, "cited_by_count": 5}],
            "primary_topic": {"id": "https://openalex.org/T10286", "display_name": "Information Retrieval"},
            "abstract_inverted_index": {"Search": [0], "ranking": [1]},
        },
        "W_HOP1_1": {
            "id": "https://openalex.org/W_HOP1_1",
            "title": "Hop 1 Citing Paper 2024",
            "publication_year": 2024,
            "publication_date": "2024-03-12",
            "type": "article",
            "language": "en",
            "authorships": [{"author": {"id": "https://openalex.org/A502", "display_name": "Bob Colleague"}, "author_position": "first"}],
            "referenced_works": ["https://openalex.org/W_AUTH_1"],
            "counts_by_year": [{"year": 2024, "cited_by_count": 2}],
            "primary_topic": {"id": "https://openalex.org/T10203", "display_name": "Recommender Systems"},
            "abstract_inverted_index": {"We": [0], "cite": [1]},
        },
        "W_FOUNDATION_1": {
            "id": "https://openalex.org/W_FOUNDATION_1",
            "title": "Foundational Classic 2020",
            "publication_year": 2020,
            "publication_date": "2020-01-01",
            "type": "article",
            "language": "en",
            "authorships": [{"author": {"id": "https://openalex.org/A999", "display_name": "Pioneer"}, "author_position": "first"}],
            "referenced_works": [],
            "counts_by_year": [{"year": 2020, "cited_by_count": 50}],
            "primary_topic": {"id": "https://openalex.org/T12072", "display_name": "Machine Learning"},
            "abstract_inverted_index": {"Foundation": [0]},
        },
        "W_FUTURE_TARGET_B": {
            "id": "https://openalex.org/W_FUTURE_TARGET_B",
            "title": "Forbidden Future Paper 2025",
            "publication_year": 2025,
            "publication_date": "2025-02-15",
            "type": "article",
            "language": "en",
            "referenced_works": ["https://openalex.org/W_AUTH_1"],
        },
    }


def test_is_valid_work_filters(tmp_path: Path):
    db = DatabaseManager(db_path=tmp_path / "test.sqlite3", cutoff_year=2024)
    client = MockOpenAlexCrawlerClient({})
    crawler = CorpusCrawler(client=client, db=db, checkpoint_dir=tmp_path / "ckpts")

    # Forbidden Paper B
    crawler.set_forbidden_paper_b_ids({"W_LEAK_TARGET"})
    assert crawler.is_valid_work({"id": "https://openalex.org/W_LEAK_TARGET", "publication_year": 2024}) is False

    # Leaked future year (> 2024)
    assert crawler.is_valid_work({"id": "W_2025", "publication_year": 2025}) is False

    # Disallowed type (e.g. editorial)
    assert crawler.is_valid_work({"id": "W_EDIT", "publication_year": 2023, "type": "editorial"}) is False

    # Non-English language
    assert crawler.is_valid_work({"id": "W_FR", "publication_year": 2023, "language": "fr"}) is False

    # Valid paper
    assert crawler.is_valid_work({"id": "W_VALID", "publication_year": 2024, "type": "article", "language": "en"}) is True


def test_crawler_full_pipeline_offline(
    tmp_path: Path,
    sample_candidate_authors_file: Path,
    mock_crawler_corpus: Dict[str, Dict[str, Any]],
):
    db_file = tmp_path / "test_corpus.sqlite3"
    db = DatabaseManager(db_path=db_file, cutoff_year=2024)
    mock_client = MockOpenAlexCrawlerClient(mock_crawler_corpus)

    crawler = CorpusCrawler(
        client=mock_client,
        db=db,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    summary = crawler.run_pipeline(
        candidate_authors_path=sample_candidate_authors_file,
        target_size=10,
    )

    assert summary["final_paper_count"] > 0
    assert summary["forbidden_target_paper_b_count"] == 1

    # Assert forbidden Paper B was NEVER ingested into the SQLite database
    assert db.get_paper("W_FUTURE_TARGET_B") is None
    assert db.get_paper("https://openalex.org/W_FUTURE_TARGET_B") is None

    # Assert valid papers ARE in database
    auth_paper = db.get_paper("https://openalex.org/W_AUTH_1")
    assert auth_paper is not None
    assert auth_paper["year"] == 2023
    assert auth_paper["citation_count_T"] == 5

    # Check that summary manifest exists on disk
    summary_file = tmp_path / "checkpoints" / "crawl_summary.json"
    assert summary_file.exists()

