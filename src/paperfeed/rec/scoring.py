"""Feature scaling and the linear score (Phase 5.3 / docs/plan.md §5.3 / docs/design.md §9).

    Score(p) = w_pers mm[cos_pers] + w_sess mm[cos_sess] + w_cite mm[log(1+BC+CC)] + w_facet mm[S_facet]
               + w_nov 1_new-method - lambda_neg Penalty

Continuous features are min-max scaled to [0, 1] within the candidate pool (a constant feature scales to 0, so a
one-paper pool scores 0 on all of them). The new-method indicator stays 0/1 and the penalty is not scaled. While the
session is empty the session term is dropped (contribution 0). The per-feature contributions w_k * mm_k are kept so
the explainer (5.5) and the faithfulness audit read the same numbers the score was built from.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from paperfeed.rec.features import CandidateFeatures

# Names of the score terms, in the order of the formula.
FEATURE_NAMES = ("persona", "session", "citation", "facet", "new_method", "penalty")


@dataclass(frozen=True)
class ScoringWeights:
    """Weights of the linear score. Setting one to 0 switches that term off (ablations)."""

    w_pers: float = 1.0
    w_sess: float = 1.0
    w_cite: float = 0.5
    w_facet: float = 0.5
    w_nov: float = 0.5
    lambda_neg: float = 0.5

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"Scoring weight {name} must be >= 0, got {value}.")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "ScoringWeights":
        """Build from the ``scoring_weights`` section of configs/default.yaml (missing keys keep the defaults)."""
        section = cfg.get("scoring_weights", {})
        known = {k: float(section[k]) for k in cls.__dataclass_fields__ if k in section}
        return cls(**known)

    def with_overrides(self, **changes: float) -> "ScoringWeights":
        return replace(self, **changes)


def min_max(values: Sequence[float]) -> np.ndarray:
    """Min-max scale to [0, 1]; a constant (or single-element) input gives zeros."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


@dataclass
class ScoredCandidate:
    paper_id: str
    features: CandidateFeatures
    scaled: Dict[str, float]  # mm value per term (new_method raw 0/1, penalty raw cosine, session 0 when inactive)
    contributions: Dict[str, float]  # signed term contribution to the score; the penalty is negative
    score: float
    session_active: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)  # room for channel ranks etc.


def score_candidates(
    features: Sequence[CandidateFeatures],
    weights: Optional[ScoringWeights] = None,
) -> List[ScoredCandidate]:
    """Scale the pool's features and compute each candidate's score, in the order given."""
    w = weights or ScoringWeights()
    if not features:
        return []

    session_active = all(f.session_cos is not None for f in features)
    mm_pers = min_max([f.persona_cos for f in features])
    mm_sess = min_max([f.session_cos for f in features]) if session_active else np.zeros(len(features))
    mm_cite = min_max([f.log_cite for f in features])
    mm_facet = min_max([f.facet.s_facet for f in features])

    scored: List[ScoredCandidate] = []
    for i, f in enumerate(features):
        scaled = {
            "persona": float(mm_pers[i]),
            "session": float(mm_sess[i]),
            "citation": float(mm_cite[i]),
            "facet": float(mm_facet[i]),
            "new_method": 1.0 if f.new_method.flag else 0.0,
            "penalty": float(f.penalty),
        }
        contributions = {
            "persona": w.w_pers * scaled["persona"],
            "session": w.w_sess * scaled["session"] if session_active else 0.0,
            "citation": w.w_cite * scaled["citation"],
            "facet": w.w_facet * scaled["facet"],
            "new_method": w.w_nov * scaled["new_method"],
            "penalty": -w.lambda_neg * scaled["penalty"],
        }
        scored.append(
            ScoredCandidate(
                paper_id=f.paper_id,
                features=f,
                scaled=scaled,
                contributions=contributions,
                score=float(sum(contributions.values())),
                session_active=session_active,
            )
        )
    return scored


def rank_candidates(scored: Sequence[ScoredCandidate]) -> List[ScoredCandidate]:
    """Highest score first; ties broken by paper_id so the order is deterministic."""
    return sorted(scored, key=lambda s: (-s.score, s.paper_id))
