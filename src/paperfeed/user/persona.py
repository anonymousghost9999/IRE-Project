"""Long-term persona vector v_pers_u (Phase 4.1 / docs/plan.md §4.1 / docs/design.md §7.1).

    v_pers_u = normalize( sum_p gamma_rec^(T - year(p)) * e_p )     over Authored(u, <= T)

Authored papers use the stored SPECTER2 'proximity' vectors; a paper without a stored vector (no abstract, or
an abstract too short to be indexed) is embedded on the fly, title-only when it has no abstract.
Zero-publication users get the mean of the 'adhoc_query' embeddings of their seed topics instead.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np

from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import clean_id
from paperfeed.data.ground_truth import to_uri
from paperfeed.index.store import EmbeddingStore

logger = get_logger("user.persona")

DEFAULT_GAMMA_REC = 0.9


class PaperEncoder(Protocol):
    """The slice of ``SPECTER2Encoder`` used here (also easy to fake in tests)."""

    def embed_papers(self, papers: List[Dict[str, Any]], batch_size: int = 32, show_progress: bool = False) -> np.ndarray: ...

    def embed_queries(self, queries: List[str], batch_size: int = 32) -> np.ndarray: ...


@dataclass
class Persona:
    """A user's persona vector and where it came from."""

    vector: np.ndarray
    source: str  # "authored" or "seed_topics"
    paper_ids: List[str]  # authored papers that contributed (empty for seed topics)
    n_embedded_on_the_fly: int = 0  # authored papers that had no stored vector


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero vector (the embeddings cancel out or are all zero).")
    return (vec / norm).astype(np.float32)


def recency_weights(years: Sequence[int], cutoff_year: int, gamma_rec: float = DEFAULT_GAMMA_REC) -> np.ndarray:
    """gamma_rec^(T - year) for each paper. A year after T is rejected: nothing past the cutoff may enter a profile."""
    years_arr = np.asarray(years, dtype=np.int64)
    if (years_arr > cutoff_year).any():
        bad = sorted({int(y) for y in years_arr[years_arr > cutoff_year]})
        raise ValueError(f"Authored papers with year > T={cutoff_year} cannot enter the persona (years {bad}).")
    return np.power(float(gamma_rec), (cutoff_year - years_arr).astype(np.float64))


def persona_vector(
    embeddings: np.ndarray,
    years: Sequence[int],
    cutoff_year: int,
    gamma_rec: float = DEFAULT_GAMMA_REC,
) -> np.ndarray:
    """L2-normalised recency-weighted sum of the authored-paper embeddings (shape ``(d,)``)."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("persona_vector needs a non-empty (n, d) embedding matrix.")
    if len(embeddings) != len(years):
        raise ValueError(f"Length mismatch: {len(embeddings)} embeddings vs {len(years)} years.")
    weights = recency_weights(years, cutoff_year, gamma_rec).astype(np.float32)
    return _l2_normalize(weights @ embeddings)


def seed_topic_persona(seed_embeddings: np.ndarray) -> np.ndarray:
    """Zero-publication persona: unweighted mean of the seed-topic ``adhoc_query`` embeddings.

    The mean is L2-normalised so it has the same scale as an authored persona in q_t = normalize((1-rho) v_sess + rho v_pers).
    """
    seed_embeddings = np.asarray(seed_embeddings, dtype=np.float32)
    if seed_embeddings.ndim != 2 or len(seed_embeddings) == 0:
        raise ValueError("seed_topic_persona needs at least one seed-topic embedding.")
    return _l2_normalize(seed_embeddings.mean(axis=0))


def load_authored_papers(db, authored_ids: Sequence[str], cutoff_year: int) -> List[Dict[str, Any]]:
    """Authored papers (bare IDs) with title, abstract and year, from the database.

    IDs missing from the database are skipped with a warning. A paper with year > T raises: the author history
    is defined as year <= T and a violation means the cohort or database is wrong.
    """
    bare_ids = list(dict.fromkeys(clean_id(a) for a in authored_ids))
    rows = db.get_papers_batch([to_uri(a) for a in bare_ids])
    papers = [
        {"paper_id": clean_id(r["paper_id"]), "title": r["title"] or "", "abstract": r["abstract"] or "", "year": int(r["year"])}
        for r in rows
    ]
    if len(papers) < len(bare_ids):
        logger.warning("%d of %d authored papers are not in the database.", len(bare_ids) - len(papers), len(bare_ids))
    late = [p["paper_id"] for p in papers if p["year"] > cutoff_year]
    if late:
        raise ValueError(f"Authored papers with year > T={cutoff_year}: {late[:5]}")
    papers.sort(key=lambda p: p["paper_id"])
    return papers


def build_persona(
    papers: Sequence[Dict[str, Any]],
    store: EmbeddingStore,
    encoder: Optional[PaperEncoder],
    cutoff_year: int,
    gamma_rec: float = DEFAULT_GAMMA_REC,
) -> Persona:
    """Persona of an author from their papers (dicts with ``paper_id``, ``title``, ``abstract``, ``year``).

    Stored vectors are used where present. The rest are embedded with ``encoder`` (title + abstract, or title only
    without an abstract); a paper with neither title nor abstract is skipped.
    """
    usable = [p for p in papers if (p.get("title") or "").strip() or (p.get("abstract") or "").strip()]
    if len(usable) < len(papers):
        logger.warning("Skipping %d authored papers with no title and no abstract.", len(papers) - len(usable))
    if not usable:
        raise ValueError("No authored papers to build a persona from; use seed_topic_persona for zero-publication users.")

    missing_ids = set(store.missing([p["paper_id"] for p in usable]))
    vectors: Dict[str, np.ndarray] = {}
    for p in usable:
        if p["paper_id"] not in missing_ids:
            vectors[p["paper_id"]] = store.select([p["paper_id"]])[0]

    fresh = [p for p in usable if p["paper_id"] in missing_ids]
    if fresh:
        if encoder is None:
            raise ValueError(f"{len(fresh)} authored papers have no stored embedding and no encoder was given.")
        for p, vec in zip(fresh, encoder.embed_papers(fresh)):
            vectors[p["paper_id"]] = vec

    matrix = np.vstack([vectors[p["paper_id"]] for p in usable])
    vec = persona_vector(matrix, [p["year"] for p in usable], cutoff_year, gamma_rec)
    return Persona(vector=vec, source="authored", paper_ids=[p["paper_id"] for p in usable], n_embedded_on_the_fly=len(fresh))


def build_seed_persona(seed_topics: Sequence[str], encoder: PaperEncoder) -> Persona:
    """Zero-publication persona from onboarding seed topics (embedded with the ``adhoc_query`` adapter)."""
    topics = [t.strip() for t in seed_topics if t and t.strip()]
    if not topics:
        raise ValueError("A zero-publication user needs at least one seed topic.")
    return Persona(vector=seed_topic_persona(encoder.embed_queries(topics)), source="seed_topics", paper_ids=[])
