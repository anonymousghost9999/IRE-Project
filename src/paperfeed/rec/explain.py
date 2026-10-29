"""Deterministic, grounded explanations (Phase 5.5 / docs/plan.md §5.5 / docs/design.md §11).

The explanation is a template filled from extracted facets (verbatim spans and evidence sentences), verified citation
edges and the session; there is no LLM paraphrasing and no external API. Faithfulness rule: the explained feature is the
one with the largest positive scaled contribution w_k * mm_k among {facet, new_method, citation, session, persona}
(ties in that order). If that feature cannot be grounded, no explanation is shown and the paper counts as not covered;
the explainer never falls back to a weaker feature, because then the text would not match what drove the ranking.

Grounding checks before display:
- facet spans: the abstract holds the verbatim span at its stored offsets, inside its (verbatim) evidence sentence;
- citations: the claimed shared references / co-citing papers are real edges with source year <= T;
- session: the named paper was clicked or saved in this session; persona: the persona similarity is positive.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import clean_id
from paperfeed.data.ground_truth import to_uri
from paperfeed.facets.store import FacetStore
from paperfeed.index.store import EmbeddingStore
from paperfeed.rec.features import UserCitationContext, _edges
from paperfeed.rec.scoring import ScoredCandidate
from paperfeed.user.facets import UserFacets
from paperfeed.user.session import SessionState

logger = get_logger("rec.explain")

# Features the explainer may show, in tie-break order. The penalty is a deduction, never a reason to recommend.
EXPLAINABLE_FEATURES: Tuple[str, ...] = ("facet", "new_method", "citation", "session", "persona")


@dataclass
class Explanation:
    feature: str
    text: str  # markdown: grounded content in **bold**, paper titles in *italics*
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplanationResult:
    paper_id: str
    dominant_feature: Optional[str]  # the feature that should be explained (None when nothing contributes)
    explanation: Optional[Explanation]  # None when it could not be grounded
    failure: Optional[str] = None  # why there is no explanation

    @property
    def covered(self) -> bool:
        return self.explanation is not None


def dominant_feature(contributions: Mapping[str, float]) -> Optional[str]:
    """The explainable feature with the largest positive contribution; ties go to the earlier one in EXPLAINABLE_FEATURES."""
    best, best_value = None, 0.0
    for name in EXPLAINABLE_FEATURES:
        value = contributions.get(name, 0.0)
        if value > best_value + 1e-12:
            best, best_value = name, value
    return best


def explanation_coverage(results: Sequence[ExplanationResult]) -> float:
    """Explanation Coverage: the share of recommendations shown with a grounded explanation (target >= 90%)."""
    return sum(r.covered for r in results) / len(results) if results else 0.0


# ---------------------------------------------------------------------------
# Grounding checks
# ---------------------------------------------------------------------------

def span_is_grounded(abstract: str, row: Mapping[str, Any]) -> bool:
    """The verbatim span is in the abstract at its offsets and lies within its evidence sentence."""
    span, start, end = row.get("facet_value"), row.get("start_char"), row.get("end_char")
    sentence = (row.get("evidence_sentence") or "").strip()
    if not abstract or not span or start is None or end is None or not sentence:
        return False
    if abstract[start:end] != span or span not in sentence:
        return False
    pos = abstract.find(sentence)
    while pos != -1:  # the sentence may occur more than once; the span must sit inside one of the occurrences
        if pos <= start and end <= pos + len(sentence):
            return True
        pos = abstract.find(sentence, pos + 1)
    return False


def _best_row(rows: Sequence[Mapping[str, Any]], canonical_id: str, abstract: str) -> Optional[Mapping[str, Any]]:
    """The grounded facet row of a canonical facet with the highest confidence (ties by position)."""
    candidates = [r for r in rows if r["canonical_id"] == canonical_id and span_is_grounded(abstract, r)]
    return min(candidates, key=lambda r: (-(r["confidence"] or 0.0), r["start_char"])) if candidates else None


def _span_evidence(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "facet_type": row["facet_type"], "canonical_id": row["canonical_id"], "span": row["facet_value"],
        "start_char": row["start_char"], "end_char": row["end_char"], "evidence_sentence": row["evidence_sentence"],
    }


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------

class Explainer:
    """Builds the explanation of one scored candidate for one user."""

    def __init__(
        self,
        user_facets: UserFacets,
        db,
        cutoff_year: int,
        citation_ctx: UserCitationContext,
        store: Optional[EmbeddingStore] = None,
        facet_store: Optional[FacetStore] = None,
        session: Optional[SessionState] = None,
    ) -> None:
        self.user_facets = user_facets
        self.db = db
        self.cutoff_year = cutoff_year
        self.citation_ctx = citation_ctx
        self.store = store
        self.facet_store = facet_store
        self.session = session

    # -- public --------------------------------------------------------------------------------------------------

    def explain(self, scored: ScoredCandidate) -> ExplanationResult:
        feature = dominant_feature(scored.contributions)
        if feature is None:
            return ExplanationResult(scored.paper_id, None, None, "no positive contribution to explain")
        builder = {
            "facet": self._facet, "new_method": self._new_method, "citation": self._citation,
            "session": self._session, "persona": self._persona,
        }[feature]
        explanation, failure = builder(scored)
        if explanation is None:
            logger.info("No grounded explanation for %s (%s): %s", scored.paper_id, feature, failure)
        return ExplanationResult(scored.paper_id, feature, explanation, failure)

    def explain_all(self, scored: Sequence[ScoredCandidate]) -> List[ExplanationResult]:
        return [self.explain(s) for s in scored]

    # -- lookups -------------------------------------------------------------------------------------------------

    def _abstract_and_rows(self, paper_id: str) -> Tuple[str, List[Mapping[str, Any]]]:
        paper = self.db.get_paper(paper_id)
        abstract = (paper or {}).get("abstract") or ""
        rows = self.facet_store.get_facets_by_paper_id(to_uri(paper_id)) if self.facet_store is not None else []
        return abstract, rows

    def _title(self, paper_id: str) -> Optional[str]:
        title = ((self.db.get_paper(paper_id) or {}).get("title") or "").strip()
        return title if title and title.lower() != "untitled" else None

    # -- templates -----------------------------------------------------------------------------------------------

    def _facet(self, scored: ScoredCandidate) -> Tuple[Optional[Explanation], Optional[str]]:
        matched = scored.features.facet.matched  # shared task / dataset facets, highest freq_u first
        if not matched:
            return None, "facet contribution without a matched facet"
        abstract, rows = self._abstract_and_rows(scored.paper_id)
        for cid, _freq in matched:  # the best matched facet that can be grounded in this abstract
            row = _best_row(rows, cid, abstract)
            if row is None:
                continue
            n = self.user_facets.counts[cid]
            if row["facet_type"].lower() == "dataset":
                text = f"Evaluates on **{row['facet_value']}**, used in {n} of your papers"
            else:
                text = f"Addresses **{row['facet_value']}**, studied in {n} of your papers"
            return Explanation("facet", text, {"user_papers_with_facet": n, **_span_evidence(row)}), None
        return None, "no matched facet could be grounded in the abstract"

    def _new_method(self, scored: ScoredCandidate) -> Tuple[Optional[Explanation], Optional[str]]:
        nm = scored.features.new_method
        if not nm.flag:
            return None, "new-method contribution without the indicator"
        abstract, rows = self._abstract_and_rows(scored.paper_id)
        method_rows = [r for r in (_best_row(rows, c, abstract) for c in nm.new_methods) if r is not None]
        task_rows = [r for r in (_best_row(rows, c, abstract) for c in nm.shared_tasks) if r is not None]
        if not method_rows or not task_rows:
            return None, "the new method or the shared task could not be grounded in the abstract"
        method = min(method_rows, key=lambda r: (-(r["confidence"] or 0.0), r["start_char"]))
        task = min(task_rows, key=lambda r: (-self.user_facets.freq(r["canonical_id"]), -(r["confidence"] or 0.0), r["start_char"]))
        text = f"Proposes **{method['facet_value']}**, which is new to your work, for **{task['facet_value']}** that you work on"
        return Explanation("new_method", text, {"method": _span_evidence(method), "task": _span_evidence(task)}), None

    def _citation(self, scored: ScoredCandidate) -> Tuple[Optional[Explanation], Optional[str]]:
        sig = scored.features.citation
        use_bc = sig.bc >= sig.cc  # the template follows whichever of BC / CC contributes more
        paper_id, claimed = (sig.bc_paper, sig.bc_paper_shared) if use_bc else (sig.cc_paper, sig.cc_paper_shared)
        if paper_id is None or claimed <= 0:
            return None, "citation contribution without an authored paper behind it"
        title = self._title(paper_id)
        if title is None:
            return None, f"authored paper {paper_id} has no title to show"

        cand = clean_id(scored.paper_id)
        if use_bc:
            refs: Dict[str, set] = {cand: set(), paper_id: set()}
            for src, tgt in _edges(self.db, "source_paper_id", [cand, paper_id], self.cutoff_year):
                refs[src].add(tgt)
            verified = len(refs[cand] & refs[paper_id])
            if verified != claimed:
                return None, f"shared references could not be verified (claimed {claimed}, found {verified})"
            text = f"Shares **{verified}** references with your paper *{title}*"
            evidence = {"kind": "bibliographic_coupling", "authored_paper_id": paper_id, "title": title, "shared_references": verified}
        else:
            citers: Dict[str, set] = {cand: set(), paper_id: set()}
            for src, tgt in _edges(self.db, "target_paper_id", [cand, paper_id], self.cutoff_year):
                citers[tgt].add(src)
            verified = len((citers[cand] & citers[paper_id]) - self.citation_ctx.authored)
            if verified != claimed:
                return None, f"co-citation could not be verified (claimed {claimed}, found {verified})"
            text = f"Frequently cited together with your paper *{title}*"
            evidence = {"kind": "co_citation", "authored_paper_id": paper_id, "title": title, "co_citing_papers": verified}
        return Explanation("citation", text, evidence), None

    def _session(self, scored: ScoredCandidate) -> Tuple[Optional[Explanation], Optional[str]]:
        if self.session is None or self.session.is_empty or self.store is None:
            return None, "no active session to refer to"
        known = [p for p in dict.fromkeys(self.session.positives) if not self.store.missing([p])]
        if not known:
            return None, "no session paper has a stored embedding"
        cand_vec = self.store.select([clean_id(scored.paper_id)])[0]
        vecs = self.store.select(known)
        sims = (vecs @ cand_vec) / np.maximum(np.linalg.norm(vecs, axis=1) * np.linalg.norm(cand_vec), 1e-12)
        best = min(range(len(known)), key=lambda i: (-round(float(sims[i]), 12), known[i]))
        title = self._title(known[best])
        if title is None:
            return None, f"session paper {known[best]} has no title to show"
        text = f"Similar to **{title}** you opened"
        return Explanation("session", text, {"session_paper_id": known[best], "title": title, "cosine": float(sims[best])}), None

    def _persona(self, scored: ScoredCandidate) -> Tuple[Optional[Explanation], Optional[str]]:
        if scored.features.persona_cos <= 0:
            return None, "persona contribution without a positive similarity"
        return Explanation("persona", "Close to your overall research profile", {"persona_cosine": scored.features.persona_cos}), None
