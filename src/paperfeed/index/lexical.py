"""BM25s Lexical Index Module (Phase 2, Step 2.4).

Governed by docs/prob.md §6D and docs/design.md §5, §8:
- Engine: bm25s (pure Python, fast sparse retrieval)
- Tokenization: lowercase, stop-word removal ('en')
- Input: title + abstract
- Filtering: Hard exclusion of user's past citations / authored papers (exclusion_set)
"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import bm25s

from paperfeed.common.logger import get_logger

logger = get_logger("index.lexical")


def clean_id(uri_or_id: str) -> str:
    """Normalize OpenAlex URI to bare ID."""
    return uri_or_id.replace("https://openalex.org/", "").strip()


class LexicalIndex:
    """
    BM25s lexical index over title + abstract of candidate papers.
    """

    def __init__(self, stopwords: str = "en") -> None:
        self.stopwords = stopwords
        self.retriever: Optional[bm25s.BM25] = None
        self.id_to_paper: List[str] = []
        self.paper_to_id: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.id_to_paper)

    def index_documents(
        self,
        paper_ids: List[str],
        documents: List[str],
        show_progress: bool = True,
    ) -> None:
        """
        Tokenize and index text documents using BM25s.
        """
        if len(paper_ids) != len(documents):
            raise ValueError(f"Length mismatch: {len(paper_ids)} paper IDs vs {len(documents)} documents.")

        if not paper_ids:
            return

        cleaned_ids = [clean_id(pid) for pid in paper_ids]
        self.id_to_paper = cleaned_ids
        self.paper_to_id = {pid: i for i, pid in enumerate(cleaned_ids)}

        logger.info(f"Tokenizing {len(documents)} documents for BM25s...")
        corpus_tokens = bm25s.tokenize(
            documents,
            stopwords=self.stopwords,
            show_progress=show_progress,
        )

        logger.info("Building BM25s index...")
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens, show_progress=show_progress)
        logger.info(f"BM25s indexed {len(self.id_to_paper)} documents.")

    def search(
        self,
        query: str,
        k: int = 100,
        exclusion_set: Optional[Union[Set[str], Iterable[str]]] = None,
        candidate_filter: Optional[Union[Set[str], Iterable[str]]] = None,
        fetch_depth: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k papers matching the text query using BM25.
        
        Args:
            query: Query string (e.g. search query or user pseudo-query).
            k: Number of candidates to return.
            exclusion_set: Paper IDs to exclude.
            candidate_filter: Whitelist of candidate IDs.
            fetch_depth: Over-fetch depth before filtering.
        """
        if self.retriever is None or len(self.id_to_paper) == 0:
            return []

        query = (query or "").strip()
        if not query:
            return []

        query_tokens = bm25s.tokenize([query], stopwords=self.stopwords, show_progress=False)

        clean_exclusions = {clean_id(x) for x in exclusion_set} if exclusion_set else set()
        if candidate_filter is not None:
            sample = next(iter(candidate_filter), None) if candidate_filter else None
            clean_candidates = candidate_filter if isinstance(candidate_filter, set) and (sample is None or not sample.startswith("http")) else {clean_id(x) for x in candidate_filter}
        else:
            clean_candidates = None

        # Same adaptive over-fetch as the dense channel (see DenseIndex.search).
        adaptive = fetch_depth is None
        depth = fetch_depth if fetch_depth is not None else max(k * 2, k + len(clean_exclusions), 200)
        depth = min(depth, len(self.id_to_paper))

        while True:
            docs, scores = self.retriever.retrieve(query_tokens, k=depth, show_progress=False)
            results: List[Tuple[str, float]] = []
            for idx, score in zip(docs[0], scores[0]):
                if idx < 0 or idx >= len(self.id_to_paper):
                    continue
                paper_id = self.id_to_paper[idx]
                if paper_id in clean_exclusions:
                    continue
                if clean_candidates is not None and paper_id not in clean_candidates:
                    continue
                results.append((paper_id, float(score)))
                if len(results) >= k:
                    break
            if len(results) >= k or not adaptive or depth >= len(self.id_to_paper):
                break
            depth = min(depth * 4, len(self.id_to_paper))

        return results

    def save(self, index_dir: Union[str, Path]) -> None:
        """Persist BM25s index and paper ID mapping."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        if self.retriever is not None:
            self.retriever.save(str(index_dir))

        id_map_path = index_dir / "id_map.json"
        with open(id_map_path, "w", encoding="utf-8") as f:
            json.dump({"id_to_paper": self.id_to_paper, "stopwords": self.stopwords}, f)

        logger.info(f"Saved BM25s index ({len(self.id_to_paper)} items) to {index_dir}")

    @classmethod
    def load(cls, index_dir: Union[str, Path]) -> "LexicalIndex":
        """Load persisted BM25s index and paper ID mapping."""
        index_dir = Path(index_dir)
        id_map_path = index_dir / "id_map.json"

        if not id_map_path.exists():
            raise FileNotFoundError(f"BM25s ID map not found: {id_map_path}")

        with open(id_map_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        instance = cls(stopwords=meta.get("stopwords", "en"))
        instance.id_to_paper = meta.get("id_to_paper", [])
        instance.paper_to_id = {pid: i for i, pid in enumerate(instance.id_to_paper)}
        instance.retriever = bm25s.BM25.load(str(index_dir), load_corpus=False)

        logger.info(f"Loaded BM25s index ({len(instance.id_to_paper)} items) from {index_dir}")
        return instance

