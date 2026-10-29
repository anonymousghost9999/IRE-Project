"""FAISS Dense Index Module (Phase 2, Step 2.3).

Governed by docs/prob.md §6D and docs/design.md §5, §8:
- Index type: FAISS IndexFlatIP (exact inner product on L2-normalized vectors = cosine similarity)
- Dimension: 768
- Filtering: Hard exclusion of user's past citations / authored papers (exclusion_set)
- ID mapping: Bi-directional paper_id <-> integer internal ID mapping
"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import faiss
import numpy as np

from paperfeed.common.logger import get_logger

logger = get_logger("index.dense")


def clean_id(uri_or_id: str) -> str:
    """Normalize OpenAlex URI to bare ID."""
    return uri_or_id.replace("https://openalex.org/", "").strip()


class DenseIndex:
    """
    FAISS IndexFlatIP wrapper for exact dense retrieval over normalized paper embeddings.
    """

    def __init__(self, dimension: int = 768) -> None:
        self.dimension = dimension
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self.id_to_paper: List[str] = []
        self.paper_to_id: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.id_to_paper)

    def add_documents(self, paper_ids: List[str], vectors: np.ndarray) -> None:
        """
        Add L2-normalized paper embeddings with associated paper identifiers.
        """
        if len(paper_ids) != len(vectors):
            raise ValueError(f"Length mismatch: {len(paper_ids)} paper IDs vs {len(vectors)} vectors.")

        if len(paper_ids) == 0:
            return

        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}.")

        # Clean paper IDs
        cleaned_ids = [clean_id(pid) for pid in paper_ids]
        start_idx = len(self.id_to_paper)

        for i, pid in enumerate(cleaned_ids):
            internal_idx = start_idx + i
            self.id_to_paper.append(pid)
            self.paper_to_id[pid] = internal_idx

        # Ensure vectors are contiguous C-order for FAISS
        if not vectors.flags["C_CONTIGUOUS"]:
            vectors = np.ascontiguousarray(vectors)

        self.index.add(vectors)
        logger.info(f"Added {len(cleaned_ids)} vectors to FAISS index. Total: {self.index.ntotal}")

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 100,
        exclusion_set: Optional[Union[Set[str], Iterable[str]]] = None,
        candidate_filter: Optional[Union[Set[str], Iterable[str]]] = None,
        fetch_depth: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k nearest papers by cosine similarity (inner product).
        
        Args:
            query_vector: 1D (768,) or 2D (1, 768) L2-normalized vector.
            k: Number of candidate papers to return.
            exclusion_set: Paper IDs to exclude (e.g. authored papers, past citations).
            candidate_filter: Optional whitelist of allowable candidate paper IDs.
            fetch_depth: Number of candidates to retrieve internally before filtering.
        """
        if self.index.ntotal == 0:
            return []

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)

        # Normalize query vector if needed
        norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
        if norm[0, 0] > 0:
            query_vector = query_vector / norm

        if not query_vector.flags["C_CONTIGUOUS"]:
            query_vector = np.ascontiguousarray(query_vector)

        clean_exclusions = {clean_id(x) for x in exclusion_set} if exclusion_set else set()
        clean_candidates = {clean_id(x) for x in candidate_filter} if candidate_filter else None

        # Determine over-fetch depth to account for exclusions
        if fetch_depth is None:
            depth = max(k * 2, k + len(clean_exclusions), 200)
        else:
            depth = fetch_depth
        depth = min(depth, self.index.ntotal)

        distances, indices = self.index.search(query_vector, depth)

        results: List[Tuple[str, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.id_to_paper):
                continue
            paper_id = self.id_to_paper[idx]
            if paper_id in clean_exclusions:
                continue
            if clean_candidates is not None and paper_id not in clean_candidates:
                continue
            results.append((paper_id, float(dist)))
            if len(results) >= k:
                break

        return results

    def save(self, index_path: Union[str, Path], id_map_path: Union[str, Path]) -> None:
        """Persist FAISS index and paper ID mapping."""
        index_path = Path(index_path)
        id_map_path = Path(id_map_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        id_map_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        with open(id_map_path, "w", encoding="utf-8") as f:
            json.dump({"id_to_paper": self.id_to_paper}, f)
        logger.info(f"Saved FAISS index ({self.index.ntotal} items) to {index_path} and ID map to {id_map_path}")

    @classmethod
    def load(cls, index_path: Union[str, Path], id_map_path: Union[str, Path]) -> "DenseIndex":
        """Load persisted FAISS index and paper ID mapping."""
        index_path = Path(index_path)
        id_map_path = Path(id_map_path)

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")
        if not id_map_path.exists():
            raise FileNotFoundError(f"ID map file not found: {id_map_path}")

        raw_index = faiss.read_index(str(index_path))
        dim = raw_index.d
        instance = cls(dimension=dim)
        instance.index = raw_index

        with open(id_map_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            instance.id_to_paper = data.get("id_to_paper", [])
            instance.paper_to_id = {pid: i for i, pid in enumerate(instance.id_to_paper)}

        logger.info(f"Loaded FAISS index ({instance.index.ntotal} items, dim={dim}) from {index_path}")
        return instance

