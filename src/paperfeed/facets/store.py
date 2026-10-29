"""SQLite persistence for scientific facets and canonical clusters (Phase 3).

Stores and queries extracted paper facets (`paper_facets`) and their canonical
cluster assignments (`facet_canon`).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)


class FacetStore:
    """Manages SQLite storage for paper facets and canonical facet mappings."""

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}")
        self._ensure_progress_table()

    def _ensure_progress_table(self) -> None:
        """`facet_progress` records every paper the extractor has finished, including papers with no facets,
        so a resumed run does not re-extract them."""
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS facet_progress (
                        paper_id TEXT PRIMARY KEY,
                        FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
                    )
                    """
                )
                # Databases extracted before this table existed: papers that already have facets are done.
                conn.execute("INSERT OR IGNORE INTO facet_progress SELECT DISTINCT paper_id FROM paper_facets")
        finally:
            conn.close()

    def clear_all(self) -> None:
        """Remove all facets, canonical clusters and extraction progress."""
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("DELETE FROM paper_facets")
                conn.execute("DELETE FROM facet_canon")
                conn.execute("DELETE FROM facet_progress")
        finally:
            conn.close()

    def count_processed(self) -> int:
        conn = self._get_connection()
        try:
            return conn.execute("SELECT COUNT(*) FROM facet_progress").fetchone()[0]
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def save_facets(
        self,
        facets: Sequence[Dict[str, Any]],
        clear_existing: bool = False,
        processed_paper_ids: Optional[Sequence[str]] = None,
    ) -> int:
        """Batch insert facets into `paper_facets`.

        Args:
            facets: List of facet dicts with keys:
                paper_id, facet_type, facet_value, canonical_id,
                evidence_sentence, start_char, end_char, confidence.
            clear_existing: If True, clears all rows in `paper_facets` first.
            processed_paper_ids: Papers this batch finished (with or without facets); recorded in
                `facet_progress` in the same transaction as the facets.

        Returns:
            Number of rows inserted.
        """
        if not facets and not processed_paper_ids:
            return 0

        rows = [
            (
                f["paper_id"],
                f["facet_type"],
                f["facet_value"],
                f.get("canonical_id"),
                f.get("evidence_sentence"),
                f.get("start_char"),
                f.get("end_char"),
                f.get("confidence"),
            )
            for f in facets
        ]

        query = """
            INSERT INTO paper_facets (
                paper_id, facet_type, facet_value, canonical_id,
                evidence_sentence, start_char, end_char, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        conn = self._get_connection()
        try:
            with conn:
                cur = conn.cursor()
                if clear_existing:
                    cur.execute("DELETE FROM paper_facets")
                cur.executemany(query, rows)
                if processed_paper_ids:
                    cur.executemany(
                        "INSERT OR IGNORE INTO facet_progress (paper_id) VALUES (?)",
                        [(pid,) for pid in processed_paper_ids],
                    )
        finally:
            conn.close()

        logger.info("Saved %d facets to paper_facets in %s", len(rows), self.db_path)
        return len(rows)

    def save_canonical_clusters(
        self,
        canonical_defs: Sequence[Dict[str, Any]],
        clear_existing: bool = False,
    ) -> int:
        """Batch insert/replace canonical cluster entries into `facet_canon`.

        Args:
            canonical_defs: List of dicts with keys:
                canonical_id, facet_type, canonical_name.
            clear_existing: If True, clears `facet_canon` first.

        Returns:
            Number of rows saved.
        """
        if not canonical_defs:
            return 0

        rows = [
            (c["canonical_id"], c["facet_type"], c["canonical_name"])
            for c in canonical_defs
        ]

        query = """
            INSERT OR REPLACE INTO facet_canon (
                canonical_id, facet_type, canonical_name
            ) VALUES (?, ?, ?)
        """

        conn = self._get_connection()
        try:
            with conn:
                cur = conn.cursor()
                if clear_existing:
                    cur.execute("DELETE FROM facet_canon")
                cur.executemany(query, rows)
        finally:
            conn.close()

        logger.info("Saved %d canonical definitions to facet_canon in %s", len(rows), self.db_path)
        return len(rows)

    def get_facets_by_paper_id(self, paper_id: str) -> List[Dict[str, Any]]:
        """Retrieve all extracted facets for a single paper."""
        query = """
            SELECT facet_id, paper_id, facet_type, facet_value, canonical_id,
                   evidence_sentence, start_char, end_char, confidence
            FROM paper_facets
            WHERE paper_id = ?
            ORDER BY facet_id ASC
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(query, (paper_id,))
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

    def get_facets_by_paper_ids(
        self,
        paper_ids: Sequence[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve facets for multiple papers grouped by paper_id."""
        if not paper_ids:
            return {}

        results: Dict[str, List[Dict[str, Any]]] = {pid: [] for pid in paper_ids}
        chunk_size = 500

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            for i in range(0, len(paper_ids), chunk_size):
                chunk = paper_ids[i : i + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                query = f"""
                    SELECT facet_id, paper_id, facet_type, facet_value, canonical_id,
                           evidence_sentence, start_char, end_char, confidence
                    FROM paper_facets
                    WHERE paper_id IN ({placeholders})
                    ORDER BY facet_id ASC
                """
                cur.execute(query, tuple(chunk))
                for row in cur.fetchall():
                    pid = row["paper_id"]
                    if pid in results:
                        results[pid].append(dict(row))
            return results
        finally:
            conn.close()

    def get_canonical_clusters(self) -> Dict[str, Dict[str, str]]:
        """Retrieve all canonical cluster mappings {canonical_id: {'facet_type': ..., 'canonical_name': ...}}."""
        query = "SELECT canonical_id, facet_type, canonical_name FROM facet_canon"
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(query)
            return {
                r["canonical_id"]: {
                    "facet_type": r["facet_type"],
                    "canonical_name": r["canonical_name"],
                }
                for r in cur.fetchall()
            }
        finally:
            conn.close()

    def get_canonical_names(self, canonical_ids: Sequence[str]) -> Dict[str, Dict[str, str]]:
        """Canonical cluster entries ``{canonical_id: {'facet_type', 'canonical_name'}}`` for the given IDs only."""
        ids = list(dict.fromkeys(canonical_ids))
        results: Dict[str, Dict[str, str]] = {}
        conn = self._get_connection()
        try:
            for i in range(0, len(ids), 500):
                chunk = ids[i : i + 500]
                marks = ",".join("?" for _ in chunk)
                for r in conn.execute(
                    f"SELECT canonical_id, facet_type, canonical_name FROM facet_canon WHERE canonical_id IN ({marks})", chunk
                ):
                    results[r["canonical_id"]] = {"facet_type": r["facet_type"], "canonical_name": r["canonical_name"]}
            return results
        finally:
            conn.close()

    def count_facets(self) -> int:
        """Return total row count in paper_facets."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM paper_facets")
            return cur.fetchone()[0]
        finally:
            conn.close()

    def count_canonical(self) -> int:
        """Return total row count in facet_canon."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM facet_canon")
            return cur.fetchone()[0]
        finally:
            conn.close()
