"""User facet set F_u (Phase 4.2 / docs/plan.md §4.2 / docs/design.md §7.1).

F_u is the set of canonical task, method and dataset facets of the user's authored papers. Each facet keeps
the number of those papers that carry it, which the pseudo-query builder (4.5) uses to pick the most frequent
terms. Zero-publication users have no papers, so F_u comes from the corpus papers nearest to their persona vector.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import clean_id
from paperfeed.data.ground_truth import to_uri
from paperfeed.facets.store import FacetStore
from paperfeed.index.dense import DenseIndex

logger = get_logger("user.facets")

CORE_FACET_TYPES: Tuple[str, ...] = ("task", "method", "dataset")
ZERO_PUB_NEIGHBOURS = 20


@dataclass
class UserFacets:
    """F_u: canonical facets with their paper counts."""

    counts: Dict[str, int] = field(default_factory=dict)  # canonical_id -> number of source papers with that facet
    types: Dict[str, str] = field(default_factory=dict)  # canonical_id -> facet type
    names: Dict[str, str] = field(default_factory=dict)  # canonical_id -> canonical name
    source: str = "authored"  # "authored" or "nearest_papers"
    paper_ids: List[str] = field(default_factory=list)  # papers the facets were taken from (bare IDs)

    @property
    def ids(self) -> Set[str]:
        return set(self.counts)

    def __len__(self) -> int:
        return len(self.counts)

    def freq(self, canonical_id: str) -> float:
        """freq_u(f): the share of the source papers (the user's authored papers, or the 20 nearest ones) with facet f."""
        n_papers = len(self.paper_ids) or max(self.counts.values(), default=1)
        return self.counts.get(canonical_id, 0) / n_papers

    def top(self, n: Optional[int] = None, facet_type: Optional[str] = None) -> List[Tuple[str, int]]:
        """Most frequent facets as ``(canonical_id, count)``; ties broken by canonical_id so results are reproducible."""
        items = [(cid, c) for cid, c in self.counts.items() if facet_type is None or self.types[cid] == facet_type]
        items.sort(key=lambda kv: (-kv[1], kv[0]))
        return items if n is None else items[:n]

    def top_names(self, n: Optional[int] = None, facet_type: Optional[str] = None) -> List[str]:
        return [self.names[cid] for cid, _ in self.top(n, facet_type)]


def build_user_facets(
    paper_ids: Sequence[str],
    facet_store: FacetStore,
    facet_types: Sequence[str] = CORE_FACET_TYPES,
    source: str = "authored",
) -> UserFacets:
    """Canonical facets of the given papers (bare or full-URI IDs).

    A facet is counted once per paper, however many times it is mentioned in the abstract. Facets without a
    canonical ID (canonicalization not yet run) and facets of other types (metric, topic) are ignored.
    """
    bare_ids = list(dict.fromkeys(clean_id(p) for p in paper_ids))
    if not bare_ids:
        return UserFacets(source=source)

    wanted = {t.lower() for t in facet_types}
    by_uri = facet_store.get_facets_by_paper_ids([to_uri(p) for p in bare_ids])

    counts: Counter = Counter()
    for rows in by_uri.values():
        per_paper = {r["canonical_id"] for r in rows if r["canonical_id"] and r["facet_type"].lower() in wanted}
        counts.update(per_paper)

    canon_entries = facet_store.get_canonical_names(list(counts))
    names = {
        cid: info["canonical_name"] for cid, info in canon_entries.items()
    }
    types = {
        cid: info["facet_type"].lower() for cid, info in canon_entries.items()
    }

    # A canonical ID without a cluster row cannot be named; keep it out so every member of F_u is explainable.
    unnamed = [cid for cid in counts if cid not in names]
    if unnamed:
        logger.warning("Dropping %d canonical facets with no facet_canon row (e.g. %s).", len(unnamed), unnamed[0])
        for cid in unnamed:
            del counts[cid]
            types.pop(cid, None)

    return UserFacets(counts=dict(counts), types=types, names=names, source=source, paper_ids=bare_ids)


def build_zero_pub_facets(
    persona_vector: np.ndarray,
    dense_index: DenseIndex,
    facet_store: FacetStore,
    k: int = ZERO_PUB_NEIGHBOURS,
    facet_types: Sequence[str] = CORE_FACET_TYPES,
) -> UserFacets:
    """F_u for a zero-publication user: facets of the ``k`` corpus papers nearest to the persona vector."""
    hits = dense_index.search(persona_vector, k=k)
    neighbour_ids = [pid for pid, _ in hits]
    if not neighbour_ids:
        logger.warning("No corpus papers found near the persona vector; F_u is empty.")
    return build_user_facets(neighbour_ids, facet_store, facet_types, source="nearest_papers")
