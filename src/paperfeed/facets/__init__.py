"""Scientific facets extraction and canonicalization module (Phase 3)."""

from paperfeed.facets.canonical import FacetCanonicalizer, normalize_span
from paperfeed.facets.extractor import FacetExtractor, split_sentences
from paperfeed.facets.store import FacetStore

__all__ = [
    "FacetExtractor",
    "split_sentences",
    "FacetCanonicalizer",
    "normalize_span",
    "FacetStore",
]
