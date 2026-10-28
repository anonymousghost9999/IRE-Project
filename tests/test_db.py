"""Unit tests for Phase 1.2 (SQLite database manager and schema)."""

import json
from paperfeed.data.db import DatabaseManager


def test_init_db(tmp_path):
    db_file = tmp_path / "test.sqlite3"
    db = DatabaseManager(db_path=db_file, cutoff_year=2023)
    assert db.db_path.exists()

    with db.get_connection() as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    assert "papers" in tables
    assert "paper_citations" in tables
    assert "paper_facets" in tables
    assert "users" in tables
    assert "user_interactions" in tables


def test_parse_openalex_work_and_temporal_guard():
    raw_work = {
        "id": "https://openalex.org/W123",
        "title": "Dual-Horizon Recommender",
        "publication_year": 2022,
        "publication_date": "2022-04-15",
        "authorships": [
            {"author": {"id": "https://openalex.org/A1", "display_name": "Bob"}, "author_position": "first"}
        ],
        "primary_location": {"source": {"display_name": "SIGIR"}, "landing_page_url": "https://example.com/p1"},
        "doi": "10.123/sigir.1",
        "referenced_works": ["https://openalex.org/W999", "https://openalex.org/W888"],
        "counts_by_year": [
            {"year": 2022, "cited_by_count": 5},
            {"year": 2023, "cited_by_count": 15},
            {"year": 2024, "cited_by_count": 50},  # Must NOT be counted for citation_count_T
        ],
        "primary_topic": {
            "id": "https://openalex.org/T1",
            "display_name": "Information Retrieval",
            "subfield": {"id": "subfields/1701"},
            "field": {"id": "fields/17"},
        },
        "abstract_inverted_index": {"We": [0], "evaluate": [1], "recommendations.": [2]},
    }

    paper_row, citations = DatabaseManager.parse_openalex_work(raw_work, cutoff_year=2023)
    assert paper_row["paper_id"] == "https://openalex.org/W123"
    assert paper_row["citation_count_T"] == 20  # 5 + 15, excluding 2024 count
    assert paper_row["in_candidates"] == 1
    assert len(citations) == 2
    assert citations[0] == ("https://openalex.org/W123", "https://openalex.org/W999", 2022)


def test_insert_and_query_papers(tmp_path, fake_openalex_works):
    db_file = tmp_path / "test.sqlite3"
    db = DatabaseManager(db_path=db_file, cutoff_year=2023)

    rows = []
    all_citations = []
    for w in fake_openalex_works:
        p_row, c_list = DatabaseManager.parse_openalex_work(w, cutoff_year=2023)
        rows.append(p_row)
        all_citations.extend(c_list)

    inserted_papers = db.insert_papers_batch(rows)
    inserted_citations = db.insert_citations_batch(all_citations)

    assert inserted_papers == 2
    assert inserted_citations == 4
    assert db.count_papers() == 2
    assert db.count_citations() == 4

    p1 = db.get_paper("https://openalex.org/W1001")
    assert p1 is not None
    assert p1["title"] == "Dense Retrieval with Transformer Adapters"
    assert p1["year"] == 2022
    assert p1["citation_count_T"] == 12  # 4 + 8


def test_temporal_citation_leakage_rejection(tmp_path):
    db_file = tmp_path / "test.sqlite3"
    db = DatabaseManager(db_path=db_file, cutoff_year=2023)

    # Citations from source papers published in 2024 must be rejected
    citations = [
        ("https://openalex.org/W_2022", "https://openalex.org/W_ref1", 2022),
        ("https://openalex.org/W_2024", "https://openalex.org/W_ref2", 2024),  # Must be filtered out
    ]
    inserted = db.insert_citations_batch(citations)
    assert inserted == 1  # Only the 2022 citation was inserted
    assert db.count_citations() == 1



def test_reingesting_a_paper_keeps_its_facets(tmp_path):
    """INSERT OR REPLACE would delete the row and cascade-delete its facets; the upsert must not."""
    db = DatabaseManager(db_path=tmp_path / "t.sqlite3", cutoff_year=2024)
    row = {"paper_id": "https://openalex.org/W1", "title": "old", "abstract": "a", "authors": "[]", "year": 2020,
           "publication_date": None, "venue": None, "doi": None, "url": None, "citation_count_T": 1,
           "primary_topic_id": None, "subfield_id": None, "field_id": None, "in_candidates": 1}
    db.insert_papers_batch([row])
    with db.get_connection() as conn:
        conn.execute("INSERT INTO paper_facets (paper_id, facet_type, facet_value) VALUES (?,?,?)", (row["paper_id"], "task", "retrieval"))
    db.insert_papers_batch([dict(row, title="new", citation_count_T=9)])
    with db.get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM paper_facets").fetchone()[0] == 1
    paper = db.get_paper("W1")
    assert paper["title"] == "new" and paper["citation_count_T"] == 9
