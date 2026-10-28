"""Persistent disk cache for intermediate model outputs and API responses (Phase 0.5)."""

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Optional


class DiskCache:
    """Thread-safe SQLite-backed key-value disk cache keyed by (item_id, model, config_hash)."""

    def __init__(self, db_path: str | Path = "data/artifacts/cache.sqlite3") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS disk_cache (
                    cache_key TEXT PRIMARY KEY,
                    item_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_item ON disk_cache(item_id)")

    @staticmethod
    def make_key(item_id: str, model_id: str, config_hash: str) -> str:
        """Create deterministic key from item, model, and configuration hash."""
        raw = f"{item_id}::{model_id}::{config_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, item_id: str, model_id: str, config_hash: str) -> Optional[Any]:
        """Retrieve cached payload if present, else None."""
        key = self.make_key(item_id, model_id, config_hash)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT payload_json FROM disk_cache WHERE cache_key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None

    def set(self, item_id: str, model_id: str, config_hash: str, payload: Any) -> None:
        """Store payload in cache."""
        key = self.make_key(item_id, model_id, config_hash)
        payload_str = json.dumps(payload)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO disk_cache (cache_key, item_id, model_id, config_hash, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (key, item_id, model_id, config_hash, payload_str),
            )

    def has(self, item_id: str, model_id: str, config_hash: str) -> bool:
        """Check if an item exists in the cache."""
        key = self.make_key(item_id, model_id, config_hash)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM disk_cache WHERE cache_key = ?", (key,))
            return cursor.fetchone() is not None

