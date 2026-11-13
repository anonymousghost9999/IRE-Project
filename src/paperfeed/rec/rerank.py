"""V2 cross-encoder feature in feed mode (Extended; docs/design.md §13.8 "V2 specification", plan 10.2a).

The reranker scores (profile query, candidate) pairs over the Core candidate pool. The profile query is the feed
pseudo-query of §8.1, each candidate is its title + abstract. The raw score becomes one more feature of the linear
score (``CandidateFeatures.ce_score``, scaled like the others in ``rec.scoring``); it never changes the pool. The score
has no grounded explanation template, so the explainer ignores it (``rec.explain.EXPLAINABLE_FEATURES``).
"""

from __future__ import annotations

from dataclasses import replace
import time
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import clean_id
from paperfeed.data.ground_truth import to_uri
from paperfeed.rec.features import CandidateFeatures

logger = get_logger("rec.rerank")


class FeedReranker(Protocol):
    """What the recommender needs from a reranker: its name and a score per (query, text) pair."""

    model_name: str

    def score(self, query: str, texts: Sequence[str]) -> List[float]: ...


class CrossEncoderScorer:
    """sentence-transformers cross-encoder (``bge-reranker-base`` / ``ms-marco-MiniLM-L-6-v2``)."""

    def __init__(self, model_name: str, batch_size: int = 32, max_length: int = 512, device: Optional[str] = None):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        if not texts:
            return []
        out = self.model.predict([(query, t) for t in texts], batch_size=self.batch_size, show_progress_bar=False)
        return [float(x) for x in out]


def candidate_text(row: Mapping[str, object]) -> str:
    """Title + abstract, the document side of every reranker pair."""
    title = str(row.get("title") or "").strip()
    abstract = str(row.get("abstract") or "").strip()
    return f"{title}. {abstract}".strip(" .") if abstract else title


def candidate_texts(db, paper_ids: Sequence[str]) -> Dict[str, str]:
    """Bare paper ID -> reranker text, for every ID (a paper missing from the database gets "")."""
    rows = db.get_papers_batch([to_uri(p) for p in paper_ids]) if paper_ids else []
    texts = {clean_id(r["paper_id"]): candidate_text(r) for r in rows}
    return {clean_id(p): texts.get(clean_id(p), "") for p in paper_ids}


def add_reranker_scores(
    features: Sequence[CandidateFeatures],
    query: str,
    texts: Mapping[str, str],
    reranker: FeedReranker,
) -> Tuple[List[CandidateFeatures], float]:
    """Copies of ``features`` with ``ce_score`` filled in, and the reranker time in seconds.

    An empty query carries no signal, so every candidate gets 0.0 (a constant feature scales to 0, i.e. no effect).
    """
    if not features:
        return [], 0.0
    query = (query or "").strip()
    if not query:
        return [replace(f, ce_score=0.0) for f in features], 0.0
    t0 = time.perf_counter()
    scores = reranker.score(query, [texts.get(f.paper_id, "") for f in features])
    elapsed = time.perf_counter() - t0
    if len(scores) != len(features):
        raise RuntimeError(f"Reranker {reranker.model_name} returned {len(scores)} scores for {len(features)} pairs.")
    return [replace(f, ce_score=float(s)) for f, s in zip(features, scores)], elapsed
