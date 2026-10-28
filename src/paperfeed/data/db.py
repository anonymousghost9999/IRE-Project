"""SQLite database manager and ingestion engine for Paper Feed (Phase 1.2)."""

import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
from paperfeed.common.config import find_project_root, load_config
from paperfeed.common.logger import get_logger
from paperfeed.openalex.abstracts import reconstruct_abstract

logger = get_logger("data.db")


class DatabaseManager:
    """Manages SQLite connection, schema lifecycle, batch insertions, and queries."""

    def __init__(self, db_path: Optional[str | Path] = None, cutoff_year: Optional[int] = None) -> None:
        cfg = load_config()
        if db_path is None:
            db_path = cfg.get("paths", {}).get("sqlite_db", "data/processed/paperfeed.sqlite3")
        
        self.db_path = Path(db_path)
        if not self.db_path.is_absolute():
            self.db_path = find_project_root() / self.db_path

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cutoff_year = cutoff_year if cutoff_year is not None else cfg.get("project", {}).get("cutoff_year", 2024)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Create a connection with foreign keys and row factory enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def init_db(self) -> None:
        """Initialize database schema from schema.sql."""
        schema_path = Path(__file__).parent / "schema.sql"
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found at {schema_path}")

        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        with self.get_connection() as conn:
            conn.executescript(schema_sql)

    @staticmethod
    def parse_openalex_work(work: Dict[str, Any], cutoff_year: int = 2023) -> Tuple[Dict[str, Any], List[Tuple[str, str, int]]]:
        """
        Extract clean paper row and citation edges from OpenAlex JSON record.
        Enforces time-bounded citation counts (counts_by_year <= cutoff_year).
        """
        paper_id = work.get("id", "").strip()
        title = work.get("title") or "Untitled"
        abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
        
        # Authors
        authors = []
        for a in work.get("authorships", []):
            author_info = a.get("author", {})
            if author_info:
                authors.append({
                    "id": author_info.get("id"),
                    "name": author_info.get("display_name"),
                    "position": a.get("author_position", "middle"),
                })
        authors_json = json.dumps(authors)

        year = int(work.get("publication_year") or 0)
        pub_date = work.get("publication_date")
        
        # Venue
        loc = work.get("primary_location") or {}
        src = loc.get("source") or {}
        venue = src.get("display_name")
        doi = work.get("doi")
        url = loc.get("landing_page_url") or work.get("id")

        # Time-bounded citations <= cutoff_year
        citation_count_T = 0
        for entry in work.get("counts_by_year", []):
            if int(entry.get("year", 0)) <= cutoff_year:
                citation_count_T += int(entry.get("cited_by_count", 0))

        # Topics
        primary_topic = work.get("primary_topic") or {}
        topic_id = primary_topic.get("id")
        subfield_id = (primary_topic.get("subfield") or {}).get("id")
        field_id = (primary_topic.get("field") or {}).get("id")

        if not topic_id:
            topics = work.get("topics", [])
            if topics and isinstance(topics[0], dict):
                topic_id = topics[0].get("id")
                subfield_id = (topics[0].get("subfield") or {}).get("id")
                field_id = (topics[0].get("field") or {}).get("id")
            else:
                topic_id = "UNKNOWN"

        in_candidates = 1 if len(abstract.strip()) > 0 else 0

        paper_row = {
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "authors": authors_json,
            "year": year,
            "publication_date": pub_date,
            "venue": venue,
            "doi": doi,
            "url": url,
            "citation_count_T": citation_count_T,
            "primary_topic_id": topic_id,
            "subfield_id": subfield_id,
            "field_id": field_id,
            "in_candidates": in_candidates,
        }

        # Outgoing citation edges (from this paper to referenced works)
        # Only valid if source paper year <= cutoff_year
        citations: List[Tuple[str, str, int]] = []
        if year <= cutoff_year:
            for ref_id in work.get("referenced_works", []):
                if ref_id and ref_id != paper_id:
                    citations.append((paper_id, ref_id.strip(), year))

        return paper_row, citations

    def insert_papers_batch(self, papers: List[Dict[str, Any]]) -> int:
        """Batch insert or replace paper records."""
        if not papers:
            return 0

        sql = """
            INSERT OR REPLACE INTO papers (
                paper_id, title, abstract, authors, year, publication_date,
                venue, doi, url, citation_count_T, primary_topic_id,
                subfield_id, field_id, in_candidates
            ) VALUES (
                :paper_id, :title, :abstract, :authors, :year, :publication_date,
                :venue, :doi, :url, :citation_count_T, :primary_topic_id,
                :subfield_id, :field_id, :in_candidates
            )
        """
        with self.get_connection() as conn:
            conn.executemany(sql, papers)
        return len(papers)

    def insert_citations_batch(self, citations: List[Tuple[str, str, int]]) -> int:
        """Batch insert citation edges, filtering strictly to source_year <= cutoff_year."""
        valid_citations = [c for c in citations if c[2] <= self.cutoff_year]
        if not valid_citations:
            return 0

        sql = """
            INSERT OR IGNORE INTO paper_citations (source_paper_id, target_paper_id, source_year)
            VALUES (?, ?, ?)
        """
        with self.get_connection() as conn:
            conn.executemany(sql, valid_citations)
        return len(valid_citations)

    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Fetch single paper row as dict, matching both bare ID and full OpenAlex URI."""
        bare_id = paper_id.replace("https://openalex.org/", "").strip()
        full_uri = f"https://openalex.org/{bare_id}"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM papers WHERE paper_id = ? OR paper_id = ?", (full_uri, bare_id))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_papers_batch(self, paper_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch multiple papers by ID preserving requested list."""
        if not paper_ids:
            return []
        placeholders = ",".join("?" for _ in paper_ids)
        sql = f"SELECT * FROM papers WHERE paper_id IN ({placeholders})"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, paper_ids)
            return [dict(r) for r in cursor.fetchall()]

    def count_papers(self, in_candidates_only: bool = False) -> int:
        """Return total count of indexed papers."""
        sql = "SELECT COUNT(*) FROM papers"
        if in_candidates_only:
            sql += " WHERE in_candidates = 1"
        with self.get_connection() as conn:
            return conn.execute(sql).fetchone()[0]

    def count_citations(self) -> int:
        """Return total count of stored citation edges."""
        with self.get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM paper_citations").fetchone()[0]

