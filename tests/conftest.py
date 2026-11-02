"""Shared pytest fixtures for Paper Feed test suite."""

import pytest
from pathlib import Path
from typing import Any, Dict, List


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Return minimal mock configuration for unit testing."""
    return {
        "project": {"name": "paperfeed_test", "cutoff_year": 2024, "target_year": 2025},
        "retrieval": {"candidate_pool_size": 100, "feed_size": 10, "rrf_k": 60},
        "scoring_weights": {"w_pers": 1.0, "w_cite": 0.5, "w_facet": 0.5, "w_nov": 0.5},
    }


@pytest.fixture
def fake_openalex_works() -> List[Dict[str, Any]]:
    """Return fixture list of fake OpenAlex works for offline unit testing."""
    return [
        {
            "id": "https://openalex.org/W1001",
            "doi": "https://doi.org/10.1001/fake1",
            "title": "Dense Retrieval with Transformer Adapters",
            "publication_year": 2022,
            "publication_date": "2022-05-10",
            "authorships": [{"author": {"id": "https://openalex.org/A501", "display_name": "Alice Smith"}, "author_position": "first"}],
            "referenced_works": ["https://openalex.org/W9001", "https://openalex.org/W9002"],
            "counts_by_year": [{"year": 2022, "cited_by_count": 4}, {"year": 2023, "cited_by_count": 8}],
            "primary_topic": {"id": "https://openalex.org/T10001", "display_name": "Information Retrieval"},
            "abstract_inverted_index": {"We": [0], "study": [1], "dense": [2], "retrieval": [3], "models.": [4]},
        },
        {
            "id": "https://openalex.org/W1002",
            "doi": "https://doi.org/10.1001/fake2",
            "title": "Facet Extraction in Scientific Papers",
            "publication_year": 2023,
            "publication_date": "2023-09-15",
            "authorships": [{"author": {"id": "https://openalex.org/A501", "display_name": "Alice Smith"}, "author_position": "first"}],
            "referenced_works": ["https://openalex.org/W9001", "https://openalex.org/W9003"],
            "counts_by_year": [{"year": 2023, "cited_by_count": 3}],
            "primary_topic": {"id": "https://openalex.org/T10001", "display_name": "Information Retrieval"},
            "abstract_inverted_index": {"Scientific": [0], "facets": [1], "improve": [2], "recommendation.": [3]},
        },
    ]

