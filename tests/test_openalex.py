"""Unit tests for Phase 1.1 (OpenAlex client and abstract reconstruction)."""

from unittest.mock import MagicMock
import httpx
from paperfeed.openalex.abstracts import reconstruct_abstract, has_valid_abstract
from paperfeed.openalex.client import OpenAlexClient


def test_reconstruct_abstract_basic():
    inverted = {
        "Dense": [0],
        "passage": [1],
        "retrieval": [2],
        "is": [3],
        "effective.": [4],
    }
    expected = "Dense passage retrieval is effective."
    assert reconstruct_abstract(inverted) == expected
    assert has_valid_abstract(inverted) is True


def test_reconstruct_abstract_disordered_and_multiword():
    inverted = {
        "model": [1, 5],
        "The": [0],
        "learns": [2],
        "representations": [3],
        "efficiently.": [4],
        "evaluated.": [6],
    }
    expected = "The model learns representations efficiently. model evaluated."
    assert reconstruct_abstract(inverted) == expected


def test_reconstruct_abstract_empty_or_none():
    assert reconstruct_abstract(None) == ""
    assert reconstruct_abstract({}) == ""
    assert has_valid_abstract(None) is False
    assert has_valid_abstract({}) is False


def test_client_init_and_headers():
    client = OpenAlexClient(email="test_user@university.edu", api_key="secret_key_123")
    assert client.email == "test_user@university.edu"
    assert client.api_key == "secret_key_123"
    assert "mailto:test_user@university.edu" in client.session.headers["User-Agent"]
    assert client.session.headers["api_key"] == "secret_key_123"
    client.close()


def test_client_mock_pagination(tmp_path):
    client = OpenAlexClient(raw_cache_dir=tmp_path)
    
    # Mock _get method to return 2 pages of 1 item each
    page1 = {
        "meta": {"next_cursor": "cursor_page_2"},
        "results": [{"id": "https://openalex.org/W1", "title": "Paper One"}],
    }
    page2 = {
        "meta": {"next_cursor": None},
        "results": [{"id": "https://openalex.org/W2", "title": "Paper Two"}],
    }
    
    client._get = MagicMock(side_effect=[page1, page2])
    
    works = list(client.paginate_works(filter_str="publication_year:2023", per_page=1))
    assert len(works) == 2
    assert works[0]["title"] == "Paper One"
    assert works[1]["title"] == "Paper Two"
    assert client._get.call_count == 2
    client.close()

