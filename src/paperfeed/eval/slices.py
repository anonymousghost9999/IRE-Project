"""Stratified slice evaluation for Track 1 (Phase 6.4 & 6.6 / plan.md §6.4 / design.md §13.1).

Computes metric breakdowns across:
1. User history volume slices:
   - Warm (>= 5 authored papers)
   - Moderate (3-4 authored papers)
   - Sparse (1-2 authored papers)
   - Zero-Pub (0 authored papers, seed topics only)
2. Item cold-start slices (design §12):
   - fresh targets: citation_count_T <= 1 and published in [T-1, T]
   - established targets: citation_count_T >= 20 and published since 2012
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


FRESH = "fresh"
ESTABLISHED = "established"


def classify_cold_start_targets(
    citation_counts: Mapping[str, int],
    years: Mapping[str, int],
    cutoff_year: int,
) -> Dict[str, str]:
    """Item cold-start classes of design §12 for each paper with a known count and year.

    - fresh: citation_count_T <= 1 and published in [T-1, T] (a new paper, not an old one whose citations predate the
      2012 ``counts_by_year`` window);
    - established: citation_count_T >= 20 and published since 2012.
    Papers in neither class are left out.
    """
    classes: Dict[str, str] = {}
    for pid, count in citation_counts.items():
        year = years.get(pid)
        if year is None:
            continue
        if count <= 1 and cutoff_year - 1 <= year <= cutoff_year:
            classes[pid] = FRESH
        elif count >= 20 and year >= 2012:
            classes[pid] = ESTABLISHED
    return classes


def item_cold_start_breakdown(
    rows: Sequence[UserEvaluationRow],
    targets_by_user: Mapping[str, Sequence[str]],
    target_class: Mapping[str, str],
    k: int = 10,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Recall@k restricted to fresh vs established targets, per system (plan §6.4 / design §12).

    For each class, a user counts only if they have at least one target of that class; ``n_authors`` reports how many
    did. Uses the top-k lists saved in each row (``system_recommendations``).

    Returns:
        class -> system_name -> {"recall_at_10", "n_authors", "n_targets"}
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cls in (FRESH, ESTABLISHED):
        per_system: Dict[str, List[float]] = {}
        n_targets = 0
        for r in rows:
            cls_targets = {t for t in targets_by_user.get(r.author_id, []) if target_class.get(t) == cls}
            if not cls_targets:
                continue
            n_targets += len(cls_targets)
            for sys_name, rec_ids in r.system_recommendations.items():
                if sys_name.startswith("Diagnostic:"):  # scored against re-citation targets, not Discovery
                    continue
                hits = len(set(rec_ids[:k]) & cls_targets)
                per_system.setdefault(sys_name, []).append(hits / len(cls_targets))
        out[cls] = {
            sys_name: {"recall_at_10": float(np.mean(vals)), "n_authors": float(len(vals)), "n_targets": float(n_targets)}
            for sys_name, vals in per_system.items()
        }
    return out
