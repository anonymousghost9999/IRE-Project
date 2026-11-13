"""V3 learned ranker (Extended; docs/design.md §9.1 and §13.8 "V3 specification", plan 10.3a).

LightGBM LambdaMART over six features of the Core pool: the min-max-scaled persona, citation and facet scores, the
new-method indicator, and the dense and lexical channel ranks. Settings are fixed in advance (``V3_PARAMS``); there is
no hyperparameter search on the 80 tuning authors. Model scores are min-max scaled within each pool and then
diversified by the same MMR as the linear scorer. Per-candidate SHAP contributions come from LightGBM's
``pred_contrib`` and are stored as the candidate's ``contributions``, so the explainer and the faithfulness audit read
the numbers the rank was built from.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from paperfeed.rec.mmr import Selection, select_top_k
from paperfeed.rec.scoring import ScoredCandidate, min_max

# Feature columns, in model order. The first four are explainable (they have templates), the channel ranks are not.
V3_FEATURES: Tuple[str, ...] = ("persona", "citation", "facet", "new_method", "dense_rank", "lexical_rank")
V3_EXPLAINABLE: Tuple[str, ...] = ("persona", "citation", "facet", "new_method")

# design §13.8, V3 point 3: fixed before implementation
V3_PARAMS: Dict[str, object] = {
    "objective": "lambdarank",
    "n_estimators": 100,
    "learning_rate": 0.05,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 20,
    "reg_lambda": 1.0,
    "colsample_bytree": 0.8,
    "subsample": 1.0,
    "deterministic": True,
    "force_row_wise": True,
    "n_jobs": 1,
    "random_state": 42,
    "verbose": -1,
}

ChannelRanks = Mapping[str, Tuple[Optional[int], Optional[int]]]


def feature_matrix(scored: Sequence[ScoredCandidate], channel_ranks: ChannelRanks, channel_depth: int) -> np.ndarray:
    """One row per candidate (pool order), columns ``V3_FEATURES``. ``scored`` gives the pool-scaled feature values
    (``ScoredCandidate.scaled``, independent of the weights); a channel that missed a paper gives channel_depth + 1."""
    missing = float(channel_depth + 1)
    rows = []
    for s in scored:
        dense, lexical = channel_ranks.get(s.paper_id, (None, None))
        rows.append([
            s.scaled["persona"],
            s.scaled["citation"],
            s.scaled["facet"],
            s.scaled["new_method"],
            float(dense) if dense is not None else missing,
            float(lexical) if lexical is not None else missing,
        ])
    return np.asarray(rows, dtype=np.float64).reshape(len(rows), len(V3_FEATURES))


def labels(scored: Sequence[ScoredCandidate], targets: Sequence[str]) -> np.ndarray:
    wanted = set(targets)
    return np.array([1 if s.paper_id in wanted else 0 for s in scored], dtype=np.int32)


@dataclass
class TrainingQuery:
    """One author's pool for training or scoring: the pool-scaled candidates, their channel ranks and targets."""

    scored: List[ScoredCandidate]
    channel_ranks: ChannelRanks
    targets: List[str]


def train_ranker(queries: Sequence[TrainingQuery], channel_depth: int, params: Optional[Mapping[str, object]] = None):
    """Fit LambdaMART with the fixed settings on the non-empty queries. Returns the fitted ``LGBMRanker``."""
    from lightgbm import LGBMRanker

    used = [q for q in queries if q.scored]
    if not used:
        raise ValueError("No non-empty query to train on.")
    X = np.vstack([feature_matrix(q.scored, q.channel_ranks, channel_depth) for q in used])
    y = np.concatenate([labels(q.scored, q.targets) for q in used])
    model = LGBMRanker(**dict(params or V3_PARAMS))
    model.fit(X, y, group=[len(q.scored) for q in used])
    return model


def shap_contributions(model, X: np.ndarray) -> np.ndarray:
    """Per-row SHAP contributions of each feature (LightGBM ``pred_contrib``; the bias column is dropped)."""
    if X.shape[0] == 0:
        return np.zeros((0, len(V3_FEATURES)))
    contrib = np.asarray(model.predict(X, pred_contrib=True), dtype=np.float64)
    return contrib[:, : len(V3_FEATURES)]


# The "off" value of each explainable feature (design §13.8, V3 point 7): scaled value 0, indicator 0.
OFF_VALUE = 0.0


def switch_off(X: np.ndarray, features: Sequence[str], rows: Optional[Sequence[int]] = None) -> np.ndarray:
    """A copy of ``X`` with ``features`` set to their off value (all rows, or only ``rows``)."""
    X = np.array(X, dtype=np.float64, copy=True)
    cols = [V3_FEATURES.index(f) for f in features]
    if cols:
        idx = slice(None) if rows is None else list(rows)
        for c in cols:
            X[idx, c] = OFF_VALUE
    return X


def model_scored_pool(
    model, scored: Sequence[ScoredCandidate], channel_ranks: ChannelRanks, channel_depth: int,
    off_features: Sequence[str] = (),
) -> List[ScoredCandidate]:
    """The pool re-scored by the model: ``score`` is the model output min-max scaled within the pool (the scale MMR
    expects) and ``contributions`` are the SHAP values per feature. ``extra["model_score"]`` keeps the raw output and
    ``extra["v3_row"]`` the model input, for the counterfactual. ``off_features`` switches inputs off (ablations)."""
    if not scored:
        return []
    X = switch_off(feature_matrix(scored, channel_ranks, channel_depth), off_features)
    raw = np.asarray(model.predict(X), dtype=np.float64)
    scaled = min_max(raw)
    shap = shap_contributions(model, X)
    out = []
    for i, s in enumerate(scored):
        contributions = {name: float(shap[i, j]) for j, name in enumerate(V3_FEATURES)}
        out.append(replace(s, score=float(scaled[i]), contributions=contributions,
                           extra={**s.extra, "model_score": float(raw[i]), "v3_row": X[i].tolist()}))
    return out


def counterfactual_pool(model, pool: Sequence[ScoredCandidate], paper_id: str, feature: str) -> List[ScoredCandidate]:
    """The model-scored ``pool`` after switching ``feature`` off for ``paper_id`` only: re-predict, rescale, keep the
    rest (design §13.8, V3 point 7). The caller re-runs MMR on it."""
    rows = [s.extra.get("v3_row") for s in pool]
    if any(r is None for r in rows):
        raise ValueError("counterfactual_pool needs a pool scored by model_scored_pool.")
    ids = [s.paper_id for s in pool]
    X = switch_off(np.asarray(rows, dtype=np.float64), [feature], rows=[ids.index(paper_id)])
    scaled = min_max(np.asarray(model.predict(X), dtype=np.float64))
    return [replace(s, score=float(scaled[i])) for i, s in enumerate(pool)]


@dataclass
class LoadedRanker:
    """The frozen V3 model as the recommender uses it (``RecommenderResources.ranker``)."""

    name: str
    booster: object
    channel_depth: int

    def predict(self, X: np.ndarray, pred_contrib: bool = False) -> np.ndarray:
        return np.asarray(self.booster.predict(X, pred_contrib=pred_contrib), dtype=np.float64)


def load_ranker(path, channel_depth: int) -> LoadedRanker:
    """A LightGBM text model file as a ``LoadedRanker`` named after the file."""
    from pathlib import Path

    from lightgbm import Booster

    p = Path(path)
    return LoadedRanker(name=p.name, booster=Booster(model_file=str(p)), channel_depth=int(channel_depth))


def rank_with_model(
    model, scored: Sequence[ScoredCandidate], channel_ranks: ChannelRanks, channel_depth: int, store, k: int,
    lambda_div: float, off_features: Sequence[str] = (),
) -> List[Selection]:
    """Top-k by the model's pool-scaled score, diversified by MMR."""
    pool = model_scored_pool(model, scored, channel_ranks, channel_depth, off_features)
    return select_top_k(pool, store, k=k, lambda_div=lambda_div)


def dominant_explainable(contributions: Mapping[str, float]) -> Optional[str]:
    """The explainable feature with the largest positive SHAP contribution (ties: earlier in ``V3_EXPLAINABLE``)."""
    best, best_value = None, 0.0
    for name in V3_EXPLAINABLE:
        value = contributions.get(name, 0.0)
        if value > best_value + 1e-12:
            best, best_value = name, value
    return best


def ranks_outweigh_explanation(contributions: Mapping[str, float]) -> bool:
    """True when a channel rank's SHAP contribution exceeds every explainable one (design §13.8, V3 point 7)."""
    explainable = max((contributions.get(f, 0.0) for f in V3_EXPLAINABLE), default=0.0)
    return max(contributions.get("dense_rank", 0.0), contributions.get("lexical_rank", 0.0)) > explainable + 1e-12
