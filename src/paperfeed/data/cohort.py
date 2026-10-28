"""Candidate author and evaluation cohort selection module (Phase 1, Step 1.2 & 1.3).

Governed by docs/prob.md §4 and docs/design.md §4.1, §13.1.
Enforces:
1. Target paper B selection strictly in T+1 (2024) in Computer Science.
2. First/sole author extraction with deterministic tie-breaking (earliest date, lowest ID).
3. Author history retrieval strictly <= T (2023) with temporal assertion guards.
4. Anomaly filtering (implausible productivity > 15 papers/year, non-CS profile).
5. History slice stratification (Warm >= 5, Moderate 3-4, Sparse 1-2).
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from paperfeed.common.logger import get_logger
from paperfeed.openalex.client import OpenAlexClient

logger = get_logger("data.cohort")


def clean_id(uri_or_id: str) -> str:
    """Normalize OpenAlex URI to bare identifier (e.g. 'https://openalex.org/W123' -> 'W123')."""
    if not uri_or_id:
        return ""
    return uri_or_id.replace("https://openalex.org/", "").strip()


def classify_history_slice(num_authored_T: int) -> str:
    """
    Classify author by their publication volume up to T (docs/design.md §13.1).
    - Warm: >= 5 prior papers
    - Moderate: 3-4 prior papers
    - Sparse: 1-2 prior papers
    - Zero: 0 prior papers (zero-publication cold-start slice)
    """
    if num_authored_T >= 5:
        return "warm"
    elif 3 <= num_authored_T <= 4:
        return "moderate"
    elif 1 <= num_authored_T <= 2:
        return "sparse"
    return "zero"


@dataclass
class CandidateAuthor:
    """Structured record for an evaluation cohort candidate author and target paper B."""

    author_id: str
    author_name: str
    target_paper_b_id: str
    target_paper_b_title: str
    target_paper_b_date: str
    target_paper_b_references: List[str]
    target_paper_b_topics: List[str]
    authored_paper_ids_T: List[str]
    selection_rule: str
    history_size_slice: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateAuthor":
        """Reconstruct instance from dictionary."""
        return cls(**data)


def extract_first_author(work: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Extract first author or sole author from OpenAlex work authorships.
    Returns dictionary with 'id' and 'display_name', or None if missing.
    """
    authorships = work.get("authorships", [])
    if not authorships:
        return None

    # Search for author marked as "first"
    for authorship in authorships:
        if authorship.get("author_position") == "first":
            author_obj = authorship.get("author") or {}
            raw_id = author_obj.get("id")
            if raw_id:
                return {
                    "id": clean_id(raw_id),
                    "display_name": author_obj.get("display_name", "").strip(),
                }

    # If single-author paper where author_position might be omitted or "solo"
    if len(authorships) == 1:
        author_obj = authorships[0].get("author") or {}
        raw_id = author_obj.get("id")
        if raw_id:
            return {
                "id": clean_id(raw_id),
                "display_name": author_obj.get("display_name", "").strip(),
            }

    return None


def select_target_paper_b(
    candidate_works: List[Dict[str, Any]],
    rule_name: str = "earliest_publication_date_then_lowest_id",
) -> Dict[str, Any]:
    """
    Select deterministic target paper B for an author with 1+ candidate 2024 papers.
    Rule: Earliest publication_date, broken by lowest lexicographical OpenAlex ID (D§4.1, P§4).
    """
    if not candidate_works:
        raise ValueError("Cannot select target paper B from empty work list.")

    def sort_key(work: Dict[str, Any]):
        pub_date = work.get("publication_date") or "9999-99-99"
        work_id = clean_id(work.get("id", ""))
        return (pub_date, work_id)

    sorted_works = sorted(candidate_works, key=sort_key)
    return sorted_works[0]


def filter_author_plausibility(
    authored_works_T: List[Dict[str, Any]],
    cutoff_year: int = 2023,
    max_papers_per_year: float = 15.0,
    require_cs_topic: bool = True,
) -> bool:
    """
    Validate author profile to filter out merged profiles, automated scrapers,
    or authors without genuine CS historical presence.
    """
    if not authored_works_T:
        return False

    # Guard: Assert no work leaks beyond cutoff_year T
    for work in authored_works_T:
        year = work.get("publication_year")
        if year is not None and year > cutoff_year:
            logger.warning(
                f"Author history contains leaked post-{cutoff_year} work {work.get('id')} ({year}). Rejecting."
            )
            return False

    years = [w.get("publication_year") for w in authored_works_T if w.get("publication_year")]
    if not years:
        return False

    min_year = min(years)
    max_year = max(years)
    career_span = max(max_year - min_year + 1, 1)

    # Implausible volume check: average papers per career year
    avg_per_year = len(authored_works_T) / career_span
    if avg_per_year > max_papers_per_year:
        logger.debug(
            f"Author rejected: volume {len(authored_works_T)} papers in {career_span} years ({avg_per_year:.1f}/yr) > {max_papers_per_year}"
        )
        return False

    # Check for excessive spikes in any single year
    year_counts: Dict[int, int] = {}
    for y in years:
        year_counts[y] = year_counts.get(y, 0) + 1
    if any(cnt > max_papers_per_year * 1.5 for cnt in year_counts.values()):
        logger.debug(f"Author rejected: single-year spike detected ({year_counts}).")
        return False

    # Check CS domain presence in history
    if require_cs_topic:
        has_cs = False
        for work in authored_works_T:
            primary_topic = work.get("primary_topic") or {}
            field = primary_topic.get("field") or {}
            field_id = clean_id(field.get("id", ""))
            field_name = (field.get("display_name") or "").lower()
            if field_id == "fields/17" or "computer science" in field_name:
                has_cs = True
                break
            # Also inspect topics array if present
            topics = work.get("topics", [])
            for t in topics:
                t_field = t.get("field") or {}
                if clean_id(t_field.get("id", "")) == "fields/17" or "computer science" in (
                    t_field.get("display_name") or ""
                ).lower():
                    has_cs = True
                    break
            if has_cs:
                break

        if not has_cs:
            logger.debug("Author rejected: no historical Computer Science works found.")
            return False

    return True


def save_candidate_authors(candidates: List[CandidateAuthor], output_path: str | Path) -> None:
    """Save candidate authors to a JSONL file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(candidates)} candidate authors to {path}")


def load_candidate_authors(input_path: str | Path) -> List[CandidateAuthor]:
    """Load candidate authors from a JSONL file."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Cohort candidate file not found: {path}")

    candidates: List[CandidateAuthor] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(CandidateAuthor.from_dict(json.loads(line)))
    return candidates


class CohortMiner:
    """Automates candidate author discovery, filtering, and history assembly."""

    def __init__(
        self,
        client: OpenAlexClient,
        cutoff_year: int = 2023,
        target_year: int = 2024,
        max_papers_per_year: float = 15.0,
    ) -> None:
        self.client = client
        self.cutoff_year = cutoff_year
        self.target_year = target_year
        self.max_papers_per_year = max_papers_per_year

    def mine_candidate_authors(
        self,
        max_candidates: int = 1000,
        max_target_works_to_scan: int = 2000,
        output_file: Optional[str | Path] = None,
    ) -> List[CandidateAuthor]:
        """
        Mine candidate evaluation authors:
        1. Fetch 2024 CS papers.
        2. Identify first author.
        3. Pick deterministic target paper B.
        4. Fetch <= 2023 history and check plausibility.
        5. Stop once target number of qualified candidate authors is reached.
        """
        logger.info(
            f"Mining up to {max_candidates} candidate authors (Target year: {self.target_year}, Cutoff: {self.cutoff_year})"
        )

        filter_str = f"publication_year:{self.target_year},primary_topic.field.id:fields/17"
        works_by_author: Dict[str, List[Dict[str, Any]]] = {}
        author_names: Dict[str, str] = {}

        # 1. Collect candidate works grouped by author
        for work in self.client.paginate_works(
            filter_str=filter_str,
            max_results=max_target_works_to_scan,
            cache_prefix=f"target_works_{self.target_year}",
        ):
            first_auth = extract_first_author(work)
            if not first_auth or not first_auth["id"]:
                continue

            a_id = first_auth["id"]
            if a_id not in works_by_author:
                works_by_author[a_id] = []
                author_names[a_id] = first_auth["display_name"]
            works_by_author[a_id].append(work)

        logger.info(
            f"Found {len(works_by_author)} unique first authors across scanned {self.target_year} works."
        )

        candidates: List[CandidateAuthor] = []

        # 2. For each author, select paper B and retrieve past history <= T
        for a_id, author_works_2024 in works_by_author.items():
            if len(candidates) >= max_candidates:
                break

            paper_b = select_target_paper_b(author_works_2024)
            b_id = clean_id(paper_b.get("id", ""))
            b_title = paper_b.get("title", "")
            b_date = paper_b.get("publication_date", "")
            b_refs = [clean_id(ref) for ref in paper_b.get("referenced_works", [])]

            topics: List[str] = []
            if paper_b.get("primary_topic"):
                topics.append(paper_b["primary_topic"].get("display_name", ""))
            for t in paper_b.get("topics", []):
                name = t.get("display_name")
                if name and name not in topics:
                    topics.append(name)

            # Retrieve author history <= T (to_publication_date: cutoff_year-12-31)
            history_filter = f"author.id:{a_id},to_publication_date:{self.cutoff_year}-12-31"
            authored_works_T: List[Dict[str, Any]] = []
            try:
                for past_work in self.client.paginate_works(
                    filter_str=history_filter,
                    cache_prefix=f"author_history_{a_id}",
                    max_results=200,
                ):
                    authored_works_T.append(past_work)
            except Exception as e:
                logger.warning(f"Error fetching history for author {a_id}: {e}")
                continue

            # Validate author plausibility & domain check
            if not filter_author_plausibility(
                authored_works_T=authored_works_T,
                cutoff_year=self.cutoff_year,
                max_papers_per_year=self.max_papers_per_year,
            ):
                continue

            authored_ids = [clean_id(w.get("id", "")) for w in authored_works_T if w.get("id")]
            slice_name = classify_history_slice(len(authored_ids))

            candidate = CandidateAuthor(
                author_id=a_id,
                author_name=author_names.get(a_id, ""),
                target_paper_b_id=b_id,
                target_paper_b_title=b_title,
                target_paper_b_date=b_date,
                target_paper_b_references=b_refs,
                target_paper_b_topics=topics,
                authored_paper_ids_T=authored_ids,
                selection_rule="earliest_publication_date_then_lowest_id",
                history_size_slice=slice_name,
            )
            candidates.append(candidate)

        logger.info(f"Successfully qualified {len(candidates)} candidate authors.")

        if output_file:
            save_candidate_authors(candidates, output_file)

        return candidates

