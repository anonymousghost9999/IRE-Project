"""Track 2 persona and topic generation (Plan 9b / docs/design.md §13.2 point 4).

Generates 30 evaluation personas (from the Track 1 test split) and 10 disjoint pilot personas
(from the tuning split) using real OpenAlex author histories.
Topic A = primary subfield (modal subfield of authored papers with >= 500 corpus papers).
Topic B = a different CS/AI subfield with >= 500 corpus papers, chosen with a per-persona seed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import sqlite3
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import clean_id

logger = get_logger("sim.persona")

MIN_AUTHORED_PAPERS = 5
MIN_SUBFIELD_CORPUS_PAPERS = 500
PERSONA_SEED = 42


@dataclass(frozen=True)
class SimPersona:
    author_id: str
    author_name: str
    split: str  # "evaluation" or "pilot"
    persona_text: str
    topic_a_id: str
    topic_a_name: str
    topic_a_query: str
    topic_b_id: str
    topic_b_name: str
    topic_b_query: str
    recent_titles: Tuple[str, ...]
    n_authored_T: int
    authored_paper_ids: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["recent_titles"] = list(self.recent_titles)
        d["authored_paper_ids"] = list(self.authored_paper_ids)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SimPersona:
        d = dict(data)
        d["recent_titles"] = tuple(d.get("recent_titles", []))
        d["authored_paper_ids"] = tuple(d.get("authored_paper_ids", []))
        return cls(**d)
        d["recent_titles"] = tuple(d.get("recent_titles", []))
        return cls(**d)


def get_eligible_subfields(
    db_path: str | Path,
    cutoff_year: int = 2024,
    min_papers: int = MIN_SUBFIELD_CORPUS_PAPERS,
) -> Dict[str, int]:
    """Find subfield IDs in the candidate corpus with at least min_papers papers."""
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        rows = cur.execute(
            """
            SELECT subfield_id, count(*)
            FROM papers
            WHERE year <= ? AND in_candidates = 1
            GROUP BY subfield_id
            HAVING count(*) >= ?
            ORDER BY count(*) DESC
            """,
            (cutoff_year, min_papers),
        ).fetchall()
        return {r[0]: r[1] for r in rows if r[0]}
    finally:
        con.close()


def load_subfield_names(
    raw_dir: str | Path,
    target_ids: Set[str],
) -> Dict[str, str]:
    """Extract display names for target subfield IDs from raw OpenAlex work json files."""
    names: Dict[str, str] = {}
    for fpath in Path(raw_dir).glob("*.json"):
        if len(names) >= len(target_ids):
            break
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            results = data.get("results", [])
            for work in results:
                pt = work.get("primary_topic")
                if pt and pt.get("subfield"):
                    sf = pt["subfield"]
                    sf_id = sf.get("id", "")
                    if sf_id in target_ids and sf_id not in names:
                        names[sf_id] = sf.get("display_name", "").strip()
                for t in work.get("topics", []):
                    sf = t.get("subfield")
                    if sf and sf.get("id") in target_ids and sf.get("id") not in names:
                        names[sf["id"]] = sf.get("display_name", "").strip()
        except Exception:
            continue
    return names


def build_persona_for_author(
    author_id: str,
    author_name: str,
    split_name: str,
    raw_dir: str | Path,
    eligible_subfields: Mapping[str, int],
    subfield_names: Mapping[str, str],
    cutoff_year: int = 2024,
) -> Optional[SimPersona]:
    """Build a persona from author history <= cutoff_year. Returns None if ineligible."""
    hpath = Path(raw_dir) / f"author_history_{author_id}_start.json"
    if not hpath.exists():
        return None

    try:
        with open(hpath, "r", encoding="utf-8") as f:
            works = json.load(f).get("results", [])
    except Exception as exc:
        logger.warning("Error reading history for author %s: %s", author_id, exc)
        return None

    # Filter papers <= cutoff_year
    valid_works = []
    for w in works:
        yr = w.get("publication_year")
        pdate = w.get("publication_date") or ""
        y = yr if yr is not None else (int(pdate[:4]) if len(pdate) >= 4 and pdate[:4].isdigit() else 9999)
        if y <= cutoff_year:
            valid_works.append(w)

    if len(valid_works) < MIN_AUTHORED_PAPERS:
        return None

    # Modal subfield of author's papers
    sf_counts: Counter[str] = Counter()
    for w in valid_works:
        pt = w.get("primary_topic")
        if pt and pt.get("subfield"):
            sf_id = pt["subfield"].get("id")
            if sf_id:
                sf_counts[sf_id] += 1

    if not sf_counts:
        return None

    # Sort by count descending, then ID ascending for determinism
    sorted_sf = sorted(sf_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    modal_sf = sorted_sf[0][0]
    if modal_sf not in eligible_subfields:
        return None

    topic_a_id = modal_sf
    topic_a_name = subfield_names.get(topic_a_id, "Computer Science")

    # Pick Topic B: a different subfield among eligible ones, using per-persona deterministic seed
    available_b = sorted([sf for sf in eligible_subfields if sf != topic_a_id])
    if not available_b:
        return None
    rng_b = random.Random(f"topic_b_{author_id}")
    topic_b_id = rng_b.choice(available_b)
    topic_b_name = subfield_names.get(topic_b_id, "Information Systems")

    # 3 most recent authored titles
    def w_sort_key(w: Dict[str, Any]) -> Tuple[str, str]:
        return (w.get("publication_date") or "0000-00-00", w.get("id") or "")

    sorted_works = sorted(valid_works, key=w_sort_key, reverse=True)
    recent_titles = [w.get("title", "").strip() for w in sorted_works[:3] if w.get("title")]
    all_authored_ids = [clean_id(w.get("id", "")) for w in valid_works if w.get("id")]

    persona_text = (
        f"Researcher in {topic_a_name}. "
        f"Recent work: " + "; ".join(f"\"{t}\"" for t in recent_titles) + "."
    )

    return SimPersona(
        author_id=author_id,
        author_name=author_name,
        split=split_name,
        persona_text=persona_text,
        topic_a_id=topic_a_id,
        topic_a_name=topic_a_name,
        topic_a_query=topic_a_name,
        topic_b_id=topic_b_id,
        topic_b_name=topic_b_name,
        topic_b_query=topic_b_name,
        recent_titles=tuple(recent_titles),
        n_authored_T=len(valid_works),
        authored_paper_ids=tuple(all_authored_ids),
    )


def generate_track2_personas(
    db_path: str | Path,
    raw_dir: str | Path,
    cohort_path: str | Path,
    split_path: str | Path,
    n_eval: int = 30,
    n_pilot: int = 10,
    seed: int = PERSONA_SEED,
) -> Dict[str, List[SimPersona]]:
    """Generate evaluation and pilot personas for Track 2."""
    eligible_subfields = get_eligible_subfields(db_path)
    logger.info("Found %d eligible subfields with >= %d corpus papers.", len(eligible_subfields), MIN_SUBFIELD_CORPUS_PAPERS)

    subfield_names = load_subfield_names(raw_dir, set(eligible_subfields.keys()))
    logger.info("Loaded %d subfield display names.", len(subfield_names))

    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    test_ids = set(split_data.get("test", []))
    tuning_ids = set(split_data.get("tuning", []))

    author_names: Dict[str, str] = {}
    with open(cohort_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                author_names[row["author_id"]] = row.get("author_name", "")

    # Collect eligible test authors
    eval_candidates: List[SimPersona] = []
    for aid in sorted(test_ids):
        p = build_persona_for_author(
            author_id=aid,
            author_name=author_names.get(aid, aid),
            split_name="evaluation",
            raw_dir=raw_dir,
            eligible_subfields=eligible_subfields,
            subfield_names=subfield_names,
        )
        if p is not None:
            eval_candidates.append(p)

    # Collect eligible tuning authors
    pilot_candidates: List[SimPersona] = []
    for aid in sorted(tuning_ids):
        p = build_persona_for_author(
            author_id=aid,
            author_name=author_names.get(aid, aid),
            split_name="pilot",
            raw_dir=raw_dir,
            eligible_subfields=eligible_subfields,
            subfield_names=subfield_names,
        )
        if p is not None:
            pilot_candidates.append(p)

    logger.info(
        "Candidate personas: %d evaluation candidates (need %d), %d pilot candidates (need %d).",
        len(eval_candidates), n_eval, len(pilot_candidates), n_pilot
    )

    rng = random.Random(seed)
    rng.shuffle(eval_candidates)
    eval_selected = eval_candidates[:n_eval]

    rng_pilot = random.Random(seed)
    rng_pilot.shuffle(pilot_candidates)
    pilot_selected = pilot_candidates[:n_pilot]

    return {
        "evaluation": eval_selected,
        "pilot": pilot_selected,
    }


def save_personas_artifact(
    personas: Mapping[str, Sequence[SimPersona]],
    output_path: str | Path,
) -> None:
    """Save personas dictionary to JSON artifact."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "track2-v1",
        "n_evaluation": len(personas.get("evaluation", [])),
        "n_pilot": len(personas.get("pilot", [])),
        "evaluation": [x.to_dict() for x in personas.get("evaluation", [])],
        "pilot": [x.to_dict() for x in personas.get("pilot", [])],
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved Track 2 personas artifact to %s", p)
