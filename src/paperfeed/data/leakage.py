"""Automated temporal-leakage checker (Phase 1, Step 1.10).

Governed by docs/design.md §4.4 and docs/plan.md §1.10. Run in tests and before every evaluation.
Nothing here modifies data; every check returns violations instead of raising, so a run can report
all of them at once, and ``LeakageReport.assert_clean()`` raises when anything is found.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from paperfeed.data.cohort import CandidateAuthor, clean_id
from paperfeed.data.db import DatabaseManager
from paperfeed.data.ground_truth import GroundTruth, paper_status, to_uri

MAX_LISTED = 10


class LeakageError(AssertionError):
    """Raised by LeakageReport.assert_clean() when a temporal-leakage check fails."""


@dataclass
class LeakageReport:
    """Check name -> up to MAX_LISTED offending examples (empty dict means clean)."""

    violations: Dict[str, List[str]] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not self.violations

    def add(self, check: str, items: Iterable[str]) -> None:
        items = list(items)
        if items:
            self.violations[check] = items[:MAX_LISTED]
            self.counts[check] = len(items)

    def merge(self, other: "LeakageReport") -> "LeakageReport":
        self.violations.update(other.violations)
        self.counts.update(other.counts)
        return self

    def assert_clean(self) -> None:
        if not self.clean:
            lines = [f"  {k}: {self.counts[k]} violation(s), e.g. {v[:3]}" for k, v in self.violations.items()]
            raise LeakageError("Temporal leakage detected:\n" + "\n".join(lines))


def find_targets_cited_before_publication(db: DatabaseManager, target_ids: Iterable[str], cutoff_year: int) -> Set[str]:
    """Target papers B that papers published <= T already cite.

    A genuine T+1 paper cannot be cited by a paper from <= T. When this happens, OpenAlex has dated an
    old work (a reissue, a merged record) to T+1, so it is not a future paper and its author must
    not be evaluated.
    """
    ids = sorted({clean_id(t) for t in target_ids})
    bad: Set[str] = set()
    with db.get_connection() as conn:
        for i in range(0, len(ids), 500):
            chunk = [to_uri(x) for x in ids[i : i + 500]]
            marks = ",".join("?" for _ in chunk)
            for (target,) in conn.execute(
                f"SELECT DISTINCT target_paper_id FROM paper_citations WHERE target_paper_id IN ({marks}) AND source_year <= ?",
                chunk + [cutoff_year],
            ):
                bad.add(clean_id(target))
    return bad


def check_corpus(
    db: DatabaseManager,
    cutoff_year: int,
    forbidden_paper_ids: Iterable[str],
    cited_forbidden_ids: Optional[Iterable[str]] = None,
) -> LeakageReport:
    """Corpus-level checks: no paper or edge from after T, no target paper B anywhere.

    ``forbidden_paper_ids`` must be absent from the papers table and never appear as an edge source.
    ``cited_forbidden_ids`` (default: the same set) must additionally not be the target of any edge;
    pass the validated cohort targets here, since targets dropped for predating T are cited by design.
    """
    report = LeakageReport()
    forbidden = {clean_id(p) for p in forbidden_paper_ids}
    forbidden_uris = sorted(to_uri(p) for p in forbidden)
    cited = forbidden if cited_forbidden_ids is None else {clean_id(p) for p in cited_forbidden_ids}
    cited_uris = sorted(to_uri(p) for p in cited)

    with db.get_connection() as conn:
        report.add(
            "paper_year_after_T",
            (r[0] for r in conn.execute("SELECT paper_id FROM papers WHERE year > ?", (cutoff_year,))),
        )
        report.add(
            "paper_date_after_T",
            (
                r[0]
                for r in conn.execute(
                    "SELECT paper_id FROM papers WHERE publication_date > ?", (f"{cutoff_year}-12-31",)
                )
            ),
        )
        report.add(
            "citation_edge_source_year_after_T",
            (
                f"{r[0]}->{r[1]}"
                for r in conn.execute(
                    "SELECT source_paper_id, target_paper_id FROM paper_citations WHERE source_year > ?",
                    (cutoff_year,),
                )
            ),
        )
        for i in range(0, len(forbidden_uris), 500):
            chunk = forbidden_uris[i : i + 500]
            marks = ",".join("?" for _ in chunk)
            report.add(
                "target_paper_B_in_papers",
                (r[0] for r in conn.execute(f"SELECT paper_id FROM papers WHERE paper_id IN ({marks})", chunk)),
            )
            report.add(
                "target_paper_B_is_edge_source",
                (r[0] for r in conn.execute(f"SELECT DISTINCT source_paper_id FROM paper_citations WHERE source_paper_id IN ({marks})", chunk)),
            )
        for i in range(0, len(cited_uris), 500):
            chunk = cited_uris[i : i + 500]
            marks = ",".join("?" for _ in chunk)
            report.add(
                "target_paper_B_is_cited",
                (r[0] for r in conn.execute(f"SELECT DISTINCT target_paper_id FROM paper_citations WHERE target_paper_id IN ({marks})", chunk)),
            )
    return report


def normalize_title(s: Optional[str]) -> str:
    """Normalize string for robust title comparison."""
    if not s:
        return ""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def titles_match(t1: Optional[str], t2: Optional[str]) -> bool:
    """True if two titles match exactly after normalization, or have high token similarity / containment."""
    if not t1 or not t2:
        return False
    n1, n2 = normalize_title(t1), normalize_title(t2)
    if not n1 or not n2:
        return False
    if n1 == n2:
        return True
    if len(n1) > 20 and len(n2) > 20 and (n1 in n2 or n2 in n1):
        return True
    s1 = set(re.sub(r"[^a-z0-9\s]", " ", t1.lower()).split())
    s2 = set(re.sub(r"[^a-z0-9\s]", " ", t2.lower()).split())
    if len(s1) >= 3 and len(s2) >= 3:
        if len(s1 & s2) / len(s1 | s2) >= 0.75:
            return True
    return False


def check_users(
    db: DatabaseManager,
    cohort: List[CandidateAuthor],
    truths: List[GroundTruth],
    cutoff_year: int,
) -> LeakageReport:
    """Per-user checks: profile inputs (authored papers) are <= T and never include B (or B's preprint); targets are <= T."""
    report = LeakageReport()
    paper_b = {clean_id(c.target_paper_b_id) for c in cohort}
    late_authored: List[str] = []
    b_authored: List[str] = []
    b_preprint: List[str] = []

    with db.get_connection() as conn:
        for c in cohort:
            authored = [clean_id(a) for a in c.authored_paper_ids_T]
            b_authored += [f"{c.author_id}:{a}" for a in authored if a in paper_b]
            for pid, s in paper_status(db, authored).items():
                if s["year"] > cutoff_year:
                    late_authored.append(f"{c.author_id}:{pid}")

            # Check for preprint leakage (matching title of target paper B in authored history)
            b_title = getattr(c, "target_paper_b_title", "")
            if b_title:
                marks = ",".join("?" for _ in authored)
                if marks:
                    chunk = [to_uri(a) for a in authored]
                    for row in conn.execute(
                        f"SELECT paper_id, title FROM papers WHERE paper_id IN ({marks})", chunk
                    ):
                        if row[1] and titles_match(row[1], b_title):
                            b_preprint.append(f"{c.author_id}:{clean_id(row[0])} -> {row[1]!r}")

    report.add("authored_paper_year_after_T", late_authored)
    report.add("target_paper_B_in_authored_history", b_authored)
    report.add("target_paper_B_preprint_in_authored_history", b_preprint)

    bad_targets: List[str] = []
    for gt in truths:
        targets: Set[str] = set(gt.discovery) | set(gt.recitation) | set(gt.zero_pub)
        if clean_id(gt.paper_b_id) in targets:
            bad_targets.append(f"{gt.author_id}:{gt.paper_b_id}")
        for pid, s in paper_status(db, sorted(targets)).items():
            if s["year"] > cutoff_year:
                bad_targets.append(f"{gt.author_id}:{pid}")
    report.add("ground_truth_target_after_T_or_is_B", bad_targets)
    return report


def check_paper_ids(db: DatabaseManager, paper_ids: Iterable[str], cutoff_year: int, forbidden: Iterable[str]) -> LeakageReport:
    """Hook for later phases: assert that the papers used by a feature, index entry, pseudo-query or
    recommended list are all <= T and none is a target paper B."""
    report = LeakageReport()
    ids = [clean_id(p) for p in paper_ids]
    forbidden_ids = {clean_id(f) for f in forbidden}
    report.add("uses_target_paper_B", [p for p in ids if p in forbidden_ids])
    report.add("uses_paper_after_T", [p for p, s in paper_status(db, ids).items() if s["year"] > cutoff_year])
    return report


def check_all(
    db: DatabaseManager,
    cohort: List[CandidateAuthor],
    truths: List[GroundTruth],
    cutoff_year: int,
    forbidden_paper_ids: Iterable[str],
    cited_forbidden_ids: Optional[Iterable[str]] = None,
) -> LeakageReport:
    """Corpus checks plus per-user checks."""
    return check_corpus(db, cutoff_year, forbidden_paper_ids, cited_forbidden_ids).merge(
        check_users(db, cohort, truths, cutoff_year)
    )
