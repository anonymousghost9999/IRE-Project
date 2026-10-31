"""Stratified slice evaluation for Track 1 (Phase 6.4 & 6.6 / plan.md §6.4 / design.md §13.1).

Computes metric breakdowns across:
1. User history volume slices:
   - Warm (>= 5 authored papers)
   - Moderate (3-4 authored papers)
   - Sparse (1-2 authored papers)
   - Zero-Pub (0 authored papers, seed topics only)
2. Item cold-start slices:
   - Discovery targets with citation_count_T <= 1 (new / unestablished papers)
   - Discovery targets with citation_count_T >= 20 (established papers)
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Set

import numpy as np

from paperfeed.eval.lfo import UserEvaluationRow
from paperfeed.eval.metrics import aggregate_metrics


def slice_by_history_volume(
    rows: Sequence[UserEvaluationRow],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Breakdown metrics by history slice (Warm, Moderate, Sparse, Zero-Pub).

    Returns:
        slice_name -> system_name -> metric_name -> average_value
    """
    slice_groups: Dict[str, List[UserEvaluationRow]] = {
        "warm": [],
        "moderate": [],
        "sparse": [],
        "zero": [],
    }

    for r in rows:
        s = str(r.history_slice).lower()
        if s not in slice_groups:
            slice_groups[s] = []
        slice_groups[s].append(r)

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for s_name, group in slice_groups.items():
        if not group:
            continue
        results[s_name] = {}
        system_names = list(group[0].system_metrics.keys())

        for sys_name in system_names:
            dicts = [u.system_metrics[sys_name] for u in group if sys_name in u.system_metrics]
            agg = aggregate_metrics(dicts)
            agg["n_authors"] = float(len(group))
            results[s_name][sys_name] = agg

    return results


def item_cold_start_breakdown(
    recommended_by_user: Mapping[str, Sequence[str]],
    targets_by_user: Mapping[str, Sequence[str]],
    citation_count_map: Mapping[str, int],
    k: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Evaluate target recall separated into low-citation (<= 1) vs high-citation (>= 20) items."""
    cold_recalls: List[float] = []
    warm_recalls: List[float] = []

    for uid, rec_ids in recommended_by_user.items():
        all_targets = targets_by_user.get(uid, [])
        cold_targets = {t for t in all_targets if citation_count_map.get(t, 0) <= 1}
        warm_targets = {t for t in all_targets if citation_count_map.get(t, 0) >= 20}

        rec_k = set(rec_ids[:k])

        if cold_targets:
            cold_recalls.append(len(rec_k & cold_targets) / len(cold_targets))
        if warm_targets:
            warm_recalls.append(len(rec_k & warm_targets) / len(warm_targets))

    return {
        "cold_start_targets (cite <= 1)": {
            "recall_at_10": float(np.mean(cold_recalls)) if cold_recalls else 0.0,
            "n_eval_authors": float(len(cold_recalls)),
        },
        "established_targets (cite >= 20)": {
            "recall_at_10": float(np.mean(warm_recalls)) if warm_recalls else 0.0,
            "n_eval_authors": float(len(warm_recalls)),
        },
    }

