"""Exclusion set E_u (Phase 4.3 / docs/plan.md §4.3 / docs/design.md §7.1, §8).

    E_u = Authored(u, <= T)  U  Citations(u, <= T)  U  SessionConsumed(u)

Papers in E_u are removed from candidate generation before scoring. Two switches:
- ``diagnostic``: drop Citations(u, <= T), for the re-citation diagnostic run (papers the user already cited become
  eligible, so the run measures how well the system re-finds them).
- ``zero_pub``: E_u is empty (a zero-publication user has no history to exclude).
Skipped papers (S^-_u) are not excluded; they only add a score penalty (design §9).
"""

from dataclasses import dataclass, field
from typing import FrozenSet, Iterable, Sequence, Set

from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import clean_id
from paperfeed.data.ground_truth import citations_of_authored

logger = get_logger("user.exclusion")


def _bare(ids: Iterable[str]) -> FrozenSet[str]:
    return frozenset(clean_id(i) for i in ids if i)


@dataclass(frozen=True)
class ExclusionSources:
    """The static parts of E_u: a user's authored papers and the papers they cite (bare OpenAlex IDs, all <= T)."""

    authored: FrozenSet[str] = field(default_factory=frozenset)
    citations: FrozenSet[str] = field(default_factory=frozenset)

    def build(
        self,
        session_consumed: Iterable[str] = (),
        diagnostic: bool = False,
        zero_pub: bool = False,
    ) -> Set[str]:
        """E_u for the current moment. ``zero_pub`` wins over ``diagnostic`` and returns the empty set."""
        if zero_pub:
            return set()
        excluded = set(self.authored)
        if not diagnostic:
            excluded |= self.citations
        excluded |= _bare(session_consumed)
        return excluded


def build_exclusion_sources(authored_ids: Sequence[str], citations: Iterable[str]) -> ExclusionSources:
    """Sources from explicit ID collections (IDs may be bare or full URIs)."""
    return ExclusionSources(authored=_bare(authored_ids), citations=_bare(citations))


def load_exclusion_sources(db, authored_ids: Sequence[str]) -> ExclusionSources:
    """Sources for an author from the database: Citations(u, <= T) are the works referenced by their authored papers.

    The citation table only holds edges whose source paper has year <= T, so nothing past the cutoff can enter.
    """
    authored = _bare(authored_ids)
    return ExclusionSources(authored=authored, citations=frozenset(citations_of_authored(db, sorted(authored))))
