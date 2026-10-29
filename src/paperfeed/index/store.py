"""Embedding store: the SPECTER2 vectors computed so far, each row tied to a paper ID (Phase 2).

``embeddings.npy`` carries no IDs, so the row order is stored next to it in ``embedding_ids.json``.
The store may hold more papers than the current index (for example after a filter change); the index
selects the rows it needs by ID, so vectors can never drift out of line with the paper list.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from paperfeed.common.logger import get_logger

logger = get_logger("index.store")


def ids_sha256(paper_ids: Sequence[str]) -> str:
    """Order-sensitive fingerprint of a paper-ID list."""
    h = hashlib.sha256()
    for pid in paper_ids:
        h.update(pid.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _atomic_write(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


class EmbeddingStore:
    def __init__(self, ids: Optional[List[str]] = None, vectors: Optional[np.ndarray] = None) -> None:
        self.ids: List[str] = ids or []
        self.vectors: np.ndarray = vectors if vectors is not None else np.empty((0, 768), dtype=np.float32)
        if len(self.ids) != len(self.vectors):
            raise ValueError(f"Store mismatch: {len(self.ids)} IDs vs {len(self.vectors)} vectors.")
        self._row: Dict[str, int] = {pid: i for i, pid in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)

    def missing(self, paper_ids: Sequence[str]) -> List[str]:
        return [p for p in paper_ids if p not in self._row]

    def add(self, paper_ids: Sequence[str], vectors: np.ndarray) -> None:
        if len(paper_ids) != len(vectors):
            raise ValueError(f"Length mismatch: {len(paper_ids)} IDs vs {len(vectors)} vectors.")
        start = len(self.ids)
        self.ids.extend(paper_ids)
        self._row.update({pid: start + i for i, pid in enumerate(paper_ids)})
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)]) if len(self.vectors) else vectors.astype(np.float32)

    def select(self, paper_ids: Sequence[str]) -> np.ndarray:
        """Vectors for ``paper_ids``, in that order. Raises if any is absent."""
        absent = self.missing(paper_ids)
        if absent:
            raise KeyError(f"{len(absent)} papers have no stored embedding, e.g. {absent[:3]}")
        return self.vectors[[self._row[p] for p in paper_ids]]

    def save(self, vectors_path: Path, ids_path: Path) -> None:
        vectors_path, ids_path = Path(vectors_path), Path(ids_path)

        def write_vectors(tmp: Path) -> None:
            with open(tmp, "wb") as f:  # a file handle stops np.save from appending ".npy"
                np.save(f, self.vectors)

        _atomic_write(vectors_path, write_vectors)
        _atomic_write(ids_path, lambda t: Path(t).write_text(json.dumps({"ids": self.ids}), encoding="utf-8"))

    @classmethod
    def load(cls, vectors_path: Path, ids_path: Path, legacy_ids_path: Optional[Path] = None) -> "EmbeddingStore":
        """Load the store. ``legacy_ids_path`` (a ``paper_ids.json`` written by the first index build, where the
        index and the embeddings had identical order) is used once when no ``embedding_ids.json`` exists yet."""
        vectors_path, ids_path = Path(vectors_path), Path(ids_path)
        if not vectors_path.exists():
            return cls()
        vectors = np.load(vectors_path)
        if ids_path.exists():
            ids = json.loads(ids_path.read_text(encoding="utf-8"))["ids"]
        elif legacy_ids_path is not None and Path(legacy_ids_path).exists():
            ids = json.loads(Path(legacy_ids_path).read_text(encoding="utf-8"))["id_to_paper"]
            logger.warning("No embedding_ids.json; adopting paper_ids.json as the row order of embeddings.npy.")
        else:
            raise FileNotFoundError(f"{vectors_path} exists but its row IDs ({ids_path}) do not; cannot align safely.")
        return cls(ids, vectors)
