"""Ground truth for the Leave-Future-Out evaluation (Phase 1, Step 1.8).

Governed by docs/prob.md §4 and docs/design.md §13.1. For an author u with target paper B (year T+1):

    Ref_<=T(B)       references of B published <= T
    Citations(u)     papers cited by any paper u authored with year <= T
    Discovery        Ref_<=T(B) \\ (Citations(u) U Authored(u))        (headline target)
    ReCitation       Ref_<=T(B) & Citations(u) \\ Authored(u)          (diagnostic run only)
    ZeroPub          Ref_<=T(B) \\ Authored(u)                         (zero-publication slice)

Each set is stored twice: the full set, and its in-corpus subset (stored in the DB with year <= T and
an abstract). A reference that is not in the DB has an unknown year; it is assumed to be <= T, since
B was published in T+1 and nearly all of its references predate it. A reference that is in the DB with
year > T (cannot happen through the crawler) is dropped.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Set

from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import CandidateAuthor, clean_id
from paperfeed.data.db import DatabaseManager

logger = get_logger("data.ground_truth")

_CHUNK = 500


def to_uri(paper_id: str) -> str:
    """Bare OpenAlex ID -> the full-URI form used as primary key in the database."""
    return f"https://openalex.org/{clean_id(paper_id)}"


@dataclass
class GroundTruth:
    """Ground-truth sets for one evaluation author (all IDs are bare OpenAlex IDs)."""

    author_id: str
    paper_b_id: str
    refs_all: List[str]
    n_authored: int
    n_citations_u: int
    discovery: List[str] = field(default_factory=list)
    recitation: List[str] = field(default_factory=list)
    zero_pub: List[str] = field(default_factory=list)
    discovery_in_corpus: List[str] = field(default_factory=list)
    recitation_in_corpus: List[str] = field(default_factory=list)
    zero_pub_in_corpus: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundTruth":
        return cls(**data)


def citations_of_authored(db: DatabaseManager, authored_ids: List[str]) -> Set[str]:
    """Citations(u, <=T): every paper referenced by an authored paper (bare IDs)."""
    cited: Set[str] = set()
    uris = [to_uri(a) for a in authored_ids]
    with db.get_connection() as conn:
        for i in range(0, len(uris), _CHUNK):
            chunk = uris[i : i + _CHUNK]
            marks = ",".join("?" for _ in chunk)
            for (target,) in conn.execute(
                f"SELECT DISTINCT target_paper_id FROM paper_citations WHERE source_paper_id IN ({marks})",
                chunk,
            ):
                cited.add(clean_id(target))
    return cited


def paper_status(db: DatabaseManager, paper_ids: List[str]) -> Dict[str, Dict[str, int]]:
    """For the papers present in the DB: {bare_id: {"year": y, "in_candidates": 0/1}}."""
    status: Dict[str, Dict[str, int]] = {}
    uris = [to_uri(p) for p in paper_ids]
    with db.get_connection() as conn:
        for i in range(0, len(uris), _CHUNK):
            chunk = uris[i : i + _CHUNK]
            marks = ",".join("?" for _ in chunk)
            for pid, year, inc in conn.execute(
                f"SELECT paper_id, year, in_candidates FROM papers WHERE paper_id IN ({marks})", chunk
            ):
                status[clean_id(pid)] = {"year": year, "in_candidates": inc}
    return status


def compute_ground_truth(db: DatabaseManager, author: CandidateAuthor, cutoff_year: int) -> GroundTruth:
    """Compute the Discovery / ReCitation / zero-publication target sets for one author."""
    authored = {clean_id(a) for a in author.authored_paper_ids_T}
    citations_u = citations_of_authored(db, sorted(authored))

    b_id = clean_id(author.target_paper_b_id)
    # OpenAlex occasionally lists a work among its own references; B can never be its own target.
    refs = [r for r in dict.fromkeys(clean_id(r) for r in author.target_paper_b_references if r) if r != b_id]
    status = paper_status(db, refs)

    # Keep refs of unknown year (not in DB) and refs known to be <= T.
    refs_le_t = [r for r in refs if r not in status or status[r]["year"] <= cutoff_year]

    def in_corpus(pid: str) -> bool:
        s = status.get(pid)
        return bool(s and s["in_candidates"] and s["year"] <= cutoff_year)

    discovery = [r for r in refs_le_t if r not in authored and r not in citations_u]
    recitation = [r for r in refs_le_t if r in citations_u and r not in authored]
    zero_pub = [r for r in refs_le_t if r not in authored]

    return GroundTruth(
        author_id=author.author_id,
        paper_b_id=author.target_paper_b_id,
        refs_all=refs_le_t,
        n_authored=len(authored),
        n_citations_u=len(citations_u),
        discovery=discovery,
        recitation=recitation,
        zero_pub=zero_pub,
        discovery_in_corpus=[r for r in discovery if in_corpus(r)],
        recitation_in_corpus=[r for r in recitation if in_corpus(r)],
        zero_pub_in_corpus=[r for r in zero_pub if in_corpus(r)],
    )


def save_ground_truth(truths: List[GroundTruth], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for gt in truths:
            f.write(json.dumps(gt.to_dict(), ensure_ascii=False) + "\n")
    logger.info(f"Saved ground truth for {len(truths)} authors to {path}")


def load_ground_truth(input_path: str | Path) -> List[GroundTruth]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [GroundTruth.from_dict(json.loads(line)) for line in f if line.strip()]
