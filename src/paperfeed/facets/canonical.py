"""Scientific Facet Canonicalization Pipeline (Phase 3.2 / docs/plan.md §3.2).

Normalizes and clusters extracted facet spans per facet type using sentence embeddings
and agglomerative clustering with cosine distance threshold. Assigns stable canonical IDs
and selects the most frequent surface form as the canonical name.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default distance threshold for cosine distance (1 - cos_sim)
# Threshold ~0.25-0.3 strikes a balance: merges "bert" and "BERT model"
# while keeping distinct methods (e.g. "bert" vs "roberta") separate.
DEFAULT_DISTANCE_THRESHOLD = 0.25
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


class FacetCanonicalizer:
    """Clusters and canonicalizes facet spans per facet type."""

    def __init__(
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
        device: Optional[str] = None,
    ):
        self.model_name = embedding_model_name
        self.distance_threshold = float(distance_threshold)
        self.device = device
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

        from sklearn.cluster import AgglomerativeClustering

        for ftype, span_dict in sorted(type_spans.items()):
            unique_norm_spans = sorted(span_dict.keys())
            if not unique_norm_spans:
                continue

            if len(unique_norm_spans) == 1:
                # Single span in this facet type
                span = unique_norm_spans[0]
                most_freq_surface = span_dict[span].most_common(1)[0][0]
                canon_id = f"canon_{ftype}_{hashlib.sha256(span.encode('utf-8')).hexdigest()[:8]}"
                span_to_canon_id[(ftype, span)] = canon_id
                canonical_defs.append({
                    "canonical_id": canon_id,
                    "facet_type": ftype,
                    "canonical_name": most_freq_surface,
                })
                continue

            # Multi-span clustering with sentence embeddings
            self._ensure_embedder()
            embeddings = self._embedder.encode(
                unique_norm_spans,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=self.distance_threshold,
            )
            labels = clustering.fit_predict(embeddings)

            # Group spans by cluster label
            clusters: Dict[int, List[str]] = defaultdict(list)
            for idx, label in enumerate(labels):
                clusters[label].append(unique_norm_spans[idx])

            # For each cluster, pick canonical name and generate canonical_id
            for cluster_idx, member_spans in clusters.items():
                # Aggregate frequency across all surface forms in the cluster
                total_surface_counter: Counter = Counter()
                for span in member_spans:
                    total_surface_counter.update(span_dict[span])

                canonical_name = total_surface_counter.most_common(1)[0][0]

                # Deterministic canonical ID based on facet_type and primary surface form
                canon_id = f"canon_{ftype}_{hashlib.sha256(canonical_name.lower().strip().encode('utf-8')).hexdigest()[:8]}"

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

        # Deduplicate canonical definitions by canonical_id
        unique_canonical_defs = {}
        for c in canonical_defs:
            unique_canonical_defs[c["canonical_id"]] = c

        return updated_facets, list(unique_canonical_defs.values())
