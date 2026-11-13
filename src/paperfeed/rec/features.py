"""Candidate features (Phase 5.2 / docs/plan.md §5.2 / docs/design.md §9).

Raw (unscaled) features for each candidate paper p of user u:
- persona cosine cos(e_p, v_pers) and session cosine cos(e_p, v_sess) (``None`` while the session is empty);
- citation signal BC(p|u) = |Refs(p) ∩ Refs(Authored(u))| and CC(p|u) = third-party papers citing both p and an authored
  paper, both from citations with source year <= T, used as the count log(1 + BC + CC);
- facet coverage S_facet = (S_task + S_dataset) / 2 with S_tau = sum of freq_u(f) over f in F_p^tau ∩ F_u^tau divided by |F_p^tau|;
- new-method indicator: p shares a task with the user and has a method that is not in F_u;
- penalty: the maximum cosine to a skipped paper (0 when nothing was skipped).
Scaling and the weighted sum are 5.3. Alongside each number the evidence the explainer (5.5) needs is recorded: the authored
paper behind the largest BC / CC contribution, the matched facets and the new methods.
"""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import clean_id
from paperfeed.data.ground_truth import to_uri
from paperfeed.facets.store import FacetStore
from paperfeed.index.store import EmbeddingStore
from paperfeed.user.facets import CORE_FACET_TYPES, UserFacets

logger = get_logger("rec.features")

_CHUNK = 500

# paper id -> facet type -> canonical ids
CandidateFacets = Dict[str, Set[str]]


# ---------------------------------------------------------------------------
# Embedding cosines and the negative-feedback penalty
# ---------------------------------------------------------------------------

def _cosines(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine of every row of ``vectors`` with ``query`` (not assuming unit-norm rows)."""
    q = np.asarray(query, dtype=np.float32)
    q_norm = float(np.linalg.norm(q))
    row_norms = np.linalg.norm(vectors, axis=1)
    denom = np.maximum(row_norms * q_norm, 1e-12)
    return (vectors @ q) / denom


def penalty_values(candidate_vectors: np.ndarray, negative_vectors: Optional[np.ndarray]) -> np.ndarray:
    """max_n cos(e_p, e_n) over the skipped papers; zeros when nothing was skipped."""
    if negative_vectors is None or len(negative_vectors) == 0:
        return np.zeros(len(candidate_vectors), dtype=np.float32)
    cand_norm = np.maximum(np.linalg.norm(candidate_vectors, axis=1, keepdims=True), 1e-12)
    neg_norm = np.maximum(np.linalg.norm(negative_vectors, axis=1, keepdims=True), 1e-12)
    sims = (candidate_vectors / cand_norm) @ (negative_vectors / neg_norm).T
    return sims.max(axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Citation signal: BC and CC
# ---------------------------------------------------------------------------

@dataclass
class CitationSignal:
    bc: int = 0
    cc: int = 0
    bc_paper: Optional[str] = None  # authored paper sharing the most references with p
    bc_paper_shared: int = 0
    cc_paper: Optional[str] = None  # authored paper co-cited with p by the most third-party papers
    cc_paper_shared: int = 0

    @property
    def log_cite(self) -> float:
        return math.log1p(self.bc + self.cc)


@dataclass
class UserCitationContext:
    """Per-user citation data, loaded once and reused for every candidate (bare IDs)."""

    authored: Set[str] = field(default_factory=set)
    ref_to_authored: Dict[str, Set[str]] = field(default_factory=dict)  # reference -> authored papers that cite it
    citer_to_authored: Dict[str, Set[str]] = field(default_factory=dict)  # third-party citing paper -> authored papers it cites


def edges(db, column: str, ids: Sequence[str], cutoff_year: int) -> List[Tuple[str, str]]:
    """(source, target) bare-ID pairs of citation edges whose ``column`` (source or target) is in ``ids``, source year <= T."""
    out: List[Tuple[str, str]] = []
    uris = [to_uri(i) for i in ids]
    with db.get_connection() as conn:
        for i in range(0, len(uris), _CHUNK):
            chunk = uris[i : i + _CHUNK]
            marks = ",".join("?" for _ in chunk)
            for src, tgt in conn.execute(
                f"SELECT source_paper_id, target_paper_id FROM paper_citations WHERE {column} IN ({marks}) AND source_year <= ?",
                [*chunk, cutoff_year],
            ):
                out.append((clean_id(src), clean_id(tgt)))
    return out

_edges = edges


def build_citation_context(db, authored_ids: Sequence[str], cutoff_year: int) -> UserCitationContext:
    """Citation data of a user's authored papers: what they cite and who cites them (third-party citers only)."""
    authored = {clean_id(a) for a in authored_ids}
    ctx = UserCitationContext(authored=authored)
    if not authored:
        return ctx
    ref_to_authored: Dict[str, Set[str]] = defaultdict(set)
    for src, tgt in _edges(db, "source_paper_id", sorted(authored), cutoff_year):
        ref_to_authored[tgt].add(src)
    citer_to_authored: Dict[str, Set[str]] = defaultdict(set)
    for src, tgt in _edges(db, "target_paper_id", sorted(authored), cutoff_year):
        if src not in authored:  # co-citation counts third parties only
            citer_to_authored[src].add(tgt)
    ctx.ref_to_authored, ctx.citer_to_authored = dict(ref_to_authored), dict(citer_to_authored)
    return ctx


def corpus_citation_scores(db, ctx: UserCitationContext, cutoff_year: int) -> Dict[str, float]:
    """log(1 + BC + CC) for every paper with BC + CC > 0, over the whole graph (no eligibility or exclusion filter).

    BC(p) counts p's references shared with the authored papers, CC(p) the third-party papers (year <= T) citing both
    p and an authored paper: the definitions of the citation feature (D§4.4), but computed for every paper instead of
    a candidate pool. Shared by the Citation-only baseline and the V4 graph candidate channel (D§13.8), so the two
    rank identically. Zero-publication users get {}.
    """
    if not ctx.authored:
        return {}
    bc: Counter = Counter()  # paper -> references it shares with the authored papers
    for src, _tgt in _edges(db, "target_paper_id", sorted(ctx.ref_to_authored), cutoff_year):
        bc[src] += 1
    cc: Counter = Counter()  # paper -> third-party papers co-citing it with an authored paper
    for _src, tgt in _edges(db, "source_paper_id", sorted(ctx.citer_to_authored), cutoff_year):
        cc[tgt] += 1
    return {p: math.log1p(bc[p] + cc[p]) for p in set(bc) | set(cc)}


def rank_by_citation(
    scores: Mapping[str, float],
    eligible: Optional[Set[str]] = None,
    excluded: Optional[Iterable[str]] = None,
    k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Papers of ``scores`` that are eligible and not excluded, by descending score, ties by paper ID, top ``k``."""
    banned = {clean_id(e) for e in excluded} if excluded else set()
    kept = [(p, s) for p, s in scores.items() if (eligible is None or p in eligible) and p not in banned]
    kept.sort(key=lambda kv: (-kv[1], kv[0]))
    return kept if k is None else kept[:k]


def _best(counter: Counter) -> Tuple[Optional[str], int]:
    if not counter:
        return None, 0
    paper, n = min(counter.items(), key=lambda kv: (-kv[1], kv[0]))  # ties by paper id
    return paper, n


def citation_signals(db, ctx: UserCitationContext, candidate_ids: Sequence[str], cutoff_year: int) -> Dict[str, CitationSignal]:
    """BC and CC (with the authored paper behind each) for every candidate. Zero-publication users get all zeros."""
    ids = [clean_id(c) for c in candidate_ids]
    signals = {c: CitationSignal() for c in ids}
    if not ctx.authored or not ids:
        return signals

    refs_of: Dict[str, Set[str]] = defaultdict(set)
    for src, tgt in _edges(db, "source_paper_id", ids, cutoff_year):
        refs_of[src].add(tgt)
    citers_of: Dict[str, Set[str]] = defaultdict(set)
    for src, tgt in _edges(db, "target_paper_id", ids, cutoff_year):
        citers_of[tgt].add(src)

    for pid in ids:
        sig = signals[pid]
        shared_refs = refs_of.get(pid, set()) & ctx.ref_to_authored.keys()
        sig.bc = len(shared_refs)
        sig.bc_paper, sig.bc_paper_shared = _best(Counter(a for r in shared_refs for a in ctx.ref_to_authored[r]))

        co_citers = citers_of.get(pid, set()) & ctx.citer_to_authored.keys()
        co_citers.discard(pid)
        sig.cc = len(co_citers)
        sig.cc_paper, sig.cc_paper_shared = _best(Counter(a for c in co_citers for a in ctx.citer_to_authored[c]))
    return signals


# ---------------------------------------------------------------------------
# Facet coverage and the new-method indicator
# ---------------------------------------------------------------------------

def load_candidate_facets(
    facet_store: Optional[FacetStore],
    paper_ids: Sequence[str],
    facet_types: Sequence[str] = CORE_FACET_TYPES,
) -> Dict[str, CandidateFacets]:
    """Canonical task / method / dataset ids of each candidate (empty sets when facets are unavailable)."""
    bare = [clean_id(p) for p in paper_ids]
    result: Dict[str, CandidateFacets] = {p: {t: set() for t in facet_types} for p in bare}
    if facet_store is None or not bare:
        return result
    wanted = {t.lower() for t in facet_types}
    for uri, rows in facet_store.get_facets_by_paper_ids([to_uri(p) for p in bare]).items():
        sets = result[clean_id(uri)]
        for r in rows:
            ftype = r["facet_type"].lower()
            if r["canonical_id"] and ftype in wanted:
                sets[ftype].add(r["canonical_id"])
    return result


@dataclass
class FacetMatch:
    s_task: float = 0.0
    s_dataset: float = 0.0
    matched: List[Tuple[str, float]] = field(default_factory=list)  # shared task/dataset (canonical_id, freq_u), best first

    @property
    def s_facet(self) -> float:
        return 0.5 * (self.s_task + self.s_dataset)


def facet_coverage(cand_facets: CandidateFacets, user_facets: UserFacets) -> FacetMatch:
    """S_tau = sum_{f in F_p^tau ∩ F_u^tau} freq_u(f) / |F_p^tau| for tau in {task, dataset}; 0 when p has no such facet."""
    scores, matched = {}, []
    for tau in ("task", "dataset"):
        f_p = cand_facets.get(tau, set())
        shared = {f for f in f_p if f in user_facets.counts}
        scores[tau] = sum(user_facets.freq(f) for f in shared) / len(f_p) if f_p else 0.0
        matched += [(f, user_facets.freq(f)) for f in shared]
    matched.sort(key=lambda fm: (-fm[1], fm[0]))
    return FacetMatch(s_task=scores["task"], s_dataset=scores["dataset"], matched=matched)


@dataclass
class NewMethod:
    flag: bool = False
    shared_tasks: List[str] = field(default_factory=list)  # tasks of p that the user already works on
    new_methods: List[str] = field(default_factory=list)  # methods of p that are not in F_u


def new_method_signal(cand_facets: CandidateFacets, user_facets: UserFacets) -> NewMethod:
    """1[exists task in F_p ∩ F_u and exists method in F_p with method not in F_u]."""
    shared_tasks = sorted(f for f in cand_facets.get("task", set()) if f in user_facets.counts)
    new_methods = sorted(f for f in cand_facets.get("method", set()) if f not in user_facets.counts)
    return NewMethod(flag=bool(shared_tasks and new_methods), shared_tasks=shared_tasks, new_methods=new_methods)


# ---------------------------------------------------------------------------
# All features for a candidate list
# ---------------------------------------------------------------------------

@dataclass
class CandidateFeatures:
    paper_id: str
    persona_cos: float
    session_cos: Optional[float]  # None while the session is empty (the feature is dropped)
    citation: CitationSignal
    facet: FacetMatch
    new_method: NewMethod
    penalty: float
    citation_count: int = 0
    ce_score: Optional[float] = None  # V2 cross-encoder score (rec.rerank); None when the reranker is off (Core)

    @property
    def log_cite(self) -> float:
        return self.citation.log_cite


def load_citation_counts(db: Any, paper_ids: Sequence[str]) -> Dict[str, int]:
    """Batch-fetch COALESCE(citation_count_T, 0) for candidate IDs (clean ID -> count)."""
    if not paper_ids:
        return {}
    bare = [clean_id(p) for p in paper_ids]
    counts: Dict[str, int] = {p: 0 for p in bare}
    uris = [to_uri(p) for p in bare]
    try:
        with db.get_connection() as conn:
            for i in range(0, len(uris), _CHUNK):
                chunk = uris[i : i + _CHUNK]
                marks = ",".join("?" for _ in chunk)
                for pid, cnt in conn.execute(
                    f"SELECT paper_id, COALESCE(citation_count_T, 0) FROM papers WHERE paper_id IN ({marks})",
                    chunk,
                ):
                    counts[clean_id(pid)] = int(cnt or 0)
    except Exception as exc:
        logger.warning("Could not batch-load citation counts: %s", exc)
    return counts


def compute_features(
    paper_ids: Sequence[str],
    store: EmbeddingStore,
    persona_vec: np.ndarray,
    user_facets: UserFacets,
    citation_ctx: UserCitationContext,
    db,
    cutoff_year: int,
    session_vec: Optional[np.ndarray] = None,
    negative_ids: Iterable[str] = (),
    facet_store: Optional[FacetStore] = None,
) -> List[CandidateFeatures]:
    """Raw features for ``paper_ids`` (candidates, in the given order)."""
    ids = [clean_id(p) for p in paper_ids]
    if not ids:
        return []
    vectors = store.select(ids)

    persona_cos = _cosines(vectors, persona_vec)
    session_active = session_vec is not None and float(np.linalg.norm(session_vec)) > 1e-12
    session_cos = _cosines(vectors, session_vec) if session_active else None

    neg_ids = [clean_id(n) for n in negative_ids]
    known_neg = [n for n in neg_ids if not store.missing([n])]
    if len(known_neg) < len(neg_ids):
        logger.warning("%d skipped papers have no stored embedding and are ignored by the penalty.", len(neg_ids) - len(known_neg))
    penalties = penalty_values(vectors, store.select(known_neg) if known_neg else None)

    citations = citation_signals(db, citation_ctx, ids, cutoff_year)
    cand_facets = load_candidate_facets(facet_store, ids)
    counts = load_citation_counts(db, ids)

    return [
        CandidateFeatures(
            paper_id=pid,
            persona_cos=float(persona_cos[i]),
            session_cos=float(session_cos[i]) if session_cos is not None else None,
            citation=citations[pid],
            facet=facet_coverage(cand_facets[pid], user_facets),
            new_method=new_method_signal(cand_facets[pid], user_facets),
            penalty=float(penalties[i]),
            citation_count=counts.get(pid, 0),
        )
        for i, pid in enumerate(ids)
    ]
