"""V3 learned ranker vs the linear scorer: author-fold cross-validation on the tuning split (plan 10.3b-c).

design §13.8, V3 point 5: the 80 tuning authors are split into 5 folds by author (fixed seed). In each fold the
linear scorer is tuned on the §9 grid and V3 is trained (and its lambda_div chosen) on the same training authors, and
both are scored on the held-out authors. The out-of-fold NDCG@10 of every author under both scorers is compared with a
paired bootstrap; V3 replaces the linear scorer only if the 95% CI of its gain lies above zero.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from paperfeed.common.logger import get_logger
from paperfeed.eval.metrics import ndcg_at_k
from paperfeed.eval.tuning import (
    DEFAULT_TRACK1_GRID,
    CachedUserCandidateData,
    _ranked_ids,
    best_grid_point,
    grid_per_user_ndcg,
)
from paperfeed.rec.learned import TrainingQuery, rank_with_model, train_ranker
from paperfeed.rec.scoring import ScoringWeights, score_candidates

logger = get_logger("eval.learned_cv")

N_FOLDS = 5
FOLD_SEED = 42


def author_folds(n_users: int, n_folds: int = N_FOLDS, seed: int = FOLD_SEED) -> List[np.ndarray]:
    """Indices of the held-out authors of each fold: a seeded permutation cut into ``n_folds`` near-equal parts."""
    if n_users < n_folds:
        raise ValueError(f"Need at least {n_folds} authors for {n_folds} folds, got {n_users}.")
    perm = np.random.default_rng(seed).permutation(n_users)
    return [np.sort(part) for part in np.array_split(perm, n_folds)]


def to_query(u: CachedUserCandidateData) -> TrainingQuery:
    """The V3 view of a cached author: pool-scaled features (independent of the weights), channel ranks, targets."""
    return TrainingQuery(score_candidates(u.features, ScoringWeights()), dict(u.channel_ranks), list(u.targets))


def _model_ndcg(model, u: CachedUserCandidateData, store, channel_depth: int, lambda_div: float, k: int) -> float:
    if not u.features:
        return 0.0
    q = to_query(u)
    picks = rank_with_model(model, q.scored, q.channel_ranks, channel_depth, store, k, lambda_div)
    return ndcg_at_k([p.scored.paper_id for p in picks], u.targets, k=k)


def choose_lambda(model, users: Sequence[CachedUserCandidateData], store, channel_depth: int,
                  lambdas: Sequence[float], k: int = 10) -> Tuple[float, float]:
    """The lambda_div with the highest mean NDCG@k on ``users`` (ties: the earlier, i.e. smaller, value)."""
    best_lam, best_mean = float(lambdas[0]), -1.0
    for lam in lambdas:
        mean = float(np.mean([_model_ndcg(model, u, store, channel_depth, lam, k) for u in users])) if users else 0.0
        if mean > best_mean + 1e-12:
            best_lam, best_mean = float(lam), mean
    return best_lam, best_mean


def _linear_ndcg(u: CachedUserCandidateData, resources, weights, lambda_div: float, adaptive_graph: bool, k: int) -> float:
    if not u.features:
        return 0.0
    _, picks = _ranked_ids(u, resources, weights, lambda_div, k, adaptive_graph)
    return ndcg_at_k([p.paper_id for p in picks], u.targets, k=k)


def fit_v3(users: Sequence[CachedUserCandidateData], store, channel_depth: int,
           lambdas: Sequence[float], k: int = 10, params: Optional[Mapping[str, object]] = None):
    """Train V3 on ``users`` and choose its lambda_div on the same users. Returns (model, lambda_div)."""
    model = train_ranker([to_query(u) for u in users], channel_depth, params)
    lam, _ = choose_lambda(model, users, store, channel_depth, lambdas, k)
    return model, lam


def cross_validate(
    cached: Sequence[CachedUserCandidateData],
    resources,
    channel_depth: int,
    base_weights: ScoringWeights,
    adaptive_graph: bool = True,
    grid: Optional[Mapping[str, Sequence[float]]] = None,
    k: int = 10,
    n_folds: int = N_FOLDS,
    seed: int = FOLD_SEED,
    params: Optional[Mapping[str, object]] = None,
) -> Dict[str, Any]:
    """Out-of-fold NDCG@k of every cached author under the linear scorer and V3, fitted on the same training folds."""
    g = dict(grid or DEFAULT_TRACK1_GRID)
    lambdas = list(g.get("lambda_div", [0.0]))
    n = len(cached)
    oof_linear, oof_v3 = np.zeros(n), np.zeros(n)
    folds_out = []
    for f, held in enumerate(author_folds(n, n_folds, seed)):
        held_set = set(int(i) for i in held)
        train = [u for i, u in enumerate(cached) if i not in held_set]
        test_idx = [int(i) for i in held]

        best = best_grid_point(grid_per_user_ndcg(train, resources, g, k, adaptive_graph, base_weights))
        model, lam = fit_v3(train, resources.store, channel_depth, lambdas, k, params)
        for i in test_idx:
            oof_linear[i] = _linear_ndcg(cached[i], resources, best.weights, best.lambda_div, adaptive_graph, k)
            oof_v3[i] = _model_ndcg(model, cached[i], resources.store, channel_depth, lam, k)
        folds_out.append({
            "fold": f,
            "held_out_authors": [cached[i].author_id for i in test_idx],
            "linear_optimum": {**best.params, "train_mean_ndcg": best.mean_ndcg},
            "v3_lambda_div": lam,
            "linear_mean_ndcg": float(oof_linear[test_idx].mean()),
            "v3_mean_ndcg": float(oof_v3[test_idx].mean()),
        })
        logger.info("Fold %d: linear %.4f, V3 %.4f on %d held-out authors", f, folds_out[-1]["linear_mean_ndcg"],
                    folds_out[-1]["v3_mean_ndcg"], len(test_idx))
    return {"author_ids": [u.author_id for u in cached], "oof_linear": oof_linear, "oof_v3": oof_v3, "folds": folds_out,
            "n_folds": n_folds, "fold_seed": seed}


def decide_learned_ranker(cv: Mapping[str, Any], n_resamples: int = 10000, seed: int = 42, alpha: float = 0.05) -> Dict[str, Any]:
    """Pre-registered V3 rule (D§13.8, V3 point 5) on the out-of-fold NDCG@10 of ``cross_validate``."""
    from paperfeed.eval.stats import paired_bootstrap_test

    lin, v3 = np.asarray(cv["oof_linear"], dtype=np.float64), np.asarray(cv["oof_v3"], dtype=np.float64)
    diff, lo, hi, p_value, d = paired_bootstrap_test(v3, lin, n_resamples=n_resamples, alpha=alpha, seed=seed)
    lin_mean = float(lin.mean())
    return {
        "rule": {
            "statement": "V3 replaces the linear scorer only if the paired-bootstrap 95% CI of its out-of-fold "
            "NDCG@10 gain (5 author folds inside the tuning split, both scorers fitted on the same training folds) "
            "lies above zero (design §13.8, V3 point 5).",
            "n_folds": cv["n_folds"],
            "fold_seed": cv["fold_seed"],
            "n_resamples": n_resamples,
            "alpha": alpha,
            "seed": seed,
        },
        "n_users": int(lin.size),
        "linear_oof_ndcg": lin_mean,
        "v3_oof_ndcg": float(v3.mean()),
        "ndcg_gain": {
            "mean_diff": diff,
            "relative": diff / lin_mean if lin_mean > 0 else None,
            "ci": [lo, hi],
            "p_value": p_value,
            "cohens_d": d,
        },
        "folds": cv["folds"],
        "adopt_v3": bool(lo > 0),
    }


def load_v3_decision(artifacts_dir) -> Dict[str, Any]:
    """The frozen V3 decision (plan 10.3c), required before any --learned-ranker test run or audit. The model file
    must be the one frozen with it (same SHA-256), so the test run cannot use a retrained model."""
    import hashlib
    import json
    from pathlib import Path

    path = Path(artifacts_dir) / "variant_v3_decision.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: freeze the V3 decision with scripts/decide_learned_ranker.py "
                                "before the test run (plan 10.3c).")
    decision = json.loads(path.read_text(encoding="utf-8"))
    model_path = Path(artifacts_dir) / decision.get("model_file", "")
    if not decision.get("model_file") or not model_path.is_file():
        raise FileNotFoundError(f"The V3 model file {model_path} frozen with the decision is missing.")
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != decision.get("model_sha256"):
        raise ValueError(
            f"{model_path} is not the model frozen with the decision (SHA-256 differs). If it was committed from "
            "another machine, git may have converted its line endings: .gitattributes marks data/artifacts/*.txt as "
            "-text so the exact bytes are stored; re-add the original file there."
        )
    decision["model_path"] = str(model_path)
    return decision
