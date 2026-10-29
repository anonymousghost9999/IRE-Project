"""Scientific Facet Canonicalization Pipeline (Phase 3.2 / docs/plan.md §3.2).

Normalizes and clusters extracted facet spans per facet type using sentence embeddings and a
cosine distance threshold. Assigns stable canonical IDs and selects the most frequent surface
form as the canonical name.

Clustering is greedy leader clustering over exact nearest-neighbour lists: spans are visited from
the most to the least frequent, and each unassigned span becomes a cluster leader that absorbs every
still-unassigned span within the distance threshold of it. Unlike average-linkage agglomerative
clustering, which needs an n x n distance matrix (hundreds of GB for the full corpus), this needs
only n x k neighbour lists, is deterministic, and never chains unrelated spans together.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default cosine distance (1 - cos_sim) between a span and its cluster leader.
DEFAULT_DISTANCE_THRESHOLD = 0.25
# Nearest neighbours examined per span. A cluster larger than this is finished by later leaders.
DEFAULT_NEIGHBOURS = 64
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def normalize_span(span: str) -> str:
    """Normalize a facet span for canonicalization (lowercase, strip noise)."""
    if not span:
        return ""
    text = span.lower().strip()
    # Strip surrounding quotation marks, brackets, periods, and trailing punctuation
    text = re.sub(r'^[\'"“‘\(\[\{\.,;:\s]+|[\'"”’\)\]\}\.,;:\s]+$', '', text)
    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def cluster_spans(
    embeddings: np.ndarray,
    visit_order: List[int],
    distance_threshold: float,
    n_neighbours: int = DEFAULT_NEIGHBOURS,
    block_size: int = 8192,
) -> np.ndarray:
    """Greedy leader clustering. Returns, for every span, the index of its cluster leader.

    ``visit_order`` lists span indices from highest to lowest priority (frequency); ``embeddings`` are
    L2-normalised rows, so the inner product is the cosine similarity.
    """
    import faiss

    n = len(embeddings)
    vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    k = min(n_neighbours, n)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    sims = np.empty((n, k), dtype=np.float32)
    nbrs = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, block_size):
        sims[start : start + block_size], nbrs[start : start + block_size] = index.search(vectors[start : start + block_size], k)

    min_sim = 1.0 - distance_threshold
    leader = np.full(n, -1, dtype=np.int64)
    for i in visit_order:
        if leader[i] >= 0:
            continue
        leader[i] = i
        near = nbrs[i][(sims[i] >= min_sim) & (nbrs[i] >= 0)]
        near = near[leader[near] < 0]
        leader[near] = i
    return leader


class FacetCanonicalizer:
    """Clusters and canonicalizes facet spans per facet type."""

    def __init__(
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
        device: Optional[str] = None,
        n_neighbours: int = DEFAULT_NEIGHBOURS,
    ):
        self.model_name = embedding_model_name
        self.distance_threshold = float(distance_threshold)
        self.device = device
        self.n_neighbours = int(n_neighbours)
        self._embedder = None

    def _ensure_embedder(self):
        if self._embedder is not None:
            return

        from sentence_transformers import SentenceTransformer

        logger.info("Loading sentence transformer %s for facet canonicalization...", self.model_name)
        self._embedder = SentenceTransformer(self.model_name, device=self.device)

    def canonicalize(
        self,
        facets: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Canonicalize a list of facet dicts.

        Args:
            facets: List of facet dicts with keys:
                'facet_type', 'facet_value', 'paper_id', etc.

        Returns:
            Tuple of:
            - updated_facets: Same list of facet dicts with 'canonical_id' populated.
            - canonical_definitions: List of dicts for `facet_canon` table:
                [{'canonical_id': ..., 'facet_type': ..., 'canonical_name': ...}]
        """
        if not facets:
            return [], []

        # 1. Group spans and surface forms by facet_type
        # type -> normalized_span -> Counter of raw surface forms
        type_spans: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))

        for f in facets:
            ftype = f["facet_type"].lower().strip()
            raw_val = f["facet_value"].strip()
            norm_val = normalize_span(raw_val)
            if not norm_val:
                norm_val = raw_val.lower().strip()
            type_spans[ftype][norm_val][raw_val] += 1

        # Global registry of span -> canonical_id
        # (facet_type, norm_span) -> canonical_id
        span_to_canon_id: Dict[Tuple[str, str], str] = {}
        canonical_defs: List[Dict[str, Any]] = []

        for ftype, span_dict in sorted(type_spans.items()):
            unique_norm_spans = sorted(span_dict.keys())
            if not unique_norm_spans:
                continue

            self._ensure_embedder()
            embeddings = self._embedder.encode(
                unique_norm_spans,
                batch_size=256,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            freq = [sum(span_dict[s].values()) for s in unique_norm_spans]
            # Most frequent first; ties broken alphabetically (unique_norm_spans is sorted), so runs are reproducible.
            order = sorted(range(len(unique_norm_spans)), key=lambda i: (-freq[i], i))
            leaders = cluster_spans(np.asarray(embeddings), order, self.distance_threshold, self.n_neighbours)

            members: Dict[int, List[str]] = defaultdict(list)
            for idx, lead in enumerate(leaders):
                members[int(lead)].append(unique_norm_spans[idx])

            for lead, member_spans in members.items():
                surface_counter: Counter = Counter()
                for span in member_spans:
                    surface_counter.update(span_dict[span])
                canonical_name = surface_counter.most_common(1)[0][0]

                # The leader's normalized span is unique within the facet type, so the ID cannot collide
                # between clusters and does not change when other spans are added to the cluster.
                digest = hashlib.sha256(unique_norm_spans[lead].encode("utf-8")).hexdigest()[:12]
                canon_id = f"canon_{ftype}_{digest}"
                for span in member_spans:
                    span_to_canon_id[(ftype, span)] = canon_id
                canonical_defs.append({
                    "canonical_id": canon_id,
                    "facet_type": ftype,
                    "canonical_name": canonical_name,
                })

        # 2. Populate canonical_id into facets
        updated_facets: List[Dict[str, Any]] = []
        for f in facets:
            ftype = f["facet_type"].lower().strip()
            raw_val = f["facet_value"].strip()
            norm_val = normalize_span(raw_val)
            if not norm_val:
                norm_val = raw_val.lower().strip()

            canon_id = span_to_canon_id.get((ftype, norm_val))
            f_copy = dict(f)
            f_copy["canonical_id"] = canon_id
            updated_facets.append(f_copy)

        if len({c["canonical_id"] for c in canonical_defs}) != len(canonical_defs):
            raise RuntimeError("Canonical ID collision: two clusters hashed to the same canonical_id.")
        return updated_facets, canonical_defs
