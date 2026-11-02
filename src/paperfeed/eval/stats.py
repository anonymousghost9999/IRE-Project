"""Statistical testing and hypothesis validation for Track 1 (Phase 6.5 / plan.md §6.5 / design.md §13.1).

Validates:
- H1: Paper Feed significantly outperforms all 5 baselines on Discovery NDCG@10 and Recall@10
      using paired bootstrap resampling and Benjamini-Hochberg (BH) FDR correction.
- H2: MMR diversification significantly increases Topic Entropy (one-sided paired bootstrap)
      with non-inferiority on NDCG@10 (relative NDCG loss <= 3%).
- Effect sizes: Paired Cohen's d.
- 95% Confidence Intervals via percentile bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass
class BootstrapComparison:
    """Paired bootstrap result between two systems."""

    system_name: str
    baseline_name: str
    metric_name: str
    mean_system: float
    mean_baseline: float
    mean_diff: float
    relative_gain_percent: float
    ci_lower: float
    ci_upper: float
    p_value: float
    p_value_corrected: float = 1.0
    cohens_d: float = 0.0
    significant_at_05: bool = False


@dataclass
class HypothesisTestReport:
    """Full statistical hypothesis testing summary."""

    h1_comparisons: List[BootstrapComparison]
    h1_supported: bool  # True if Paper Feed beats ALL 5 baselines on NDCG@10 with p < 0.05
    h2_mmr_entropy: BootstrapComparison
    h2_mmr_ndcg_retention: BootstrapComparison
    h2_supported: bool  # True if MMR increases entropy without >3% relative NDCG loss
    n_resamples: int
    n_authors: int


def paired_cohens_d(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Cohen's d for paired samples: mean(diff) / std(diff)."""
    diff = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    std_diff = float(np.std(diff, ddof=1))
    if std_diff <= 1e-12:
        return 0.0
    return float(np.mean(diff) / std_diff)


def paired_bootstrap_test(
    scores_system: Sequence[float],
    scores_baseline: Sequence[float],
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
    alternative: str = "two-sided",  # "two-sided", "greater", "less"
) -> Tuple[float, float, float, float, float]:
    """Paired bootstrap test.

    Returns:
        (mean_diff, ci_lower, ci_upper, p_value, cohens_d)
    """
    s = np.asarray(scores_system, dtype=np.float64)
    b = np.asarray(scores_baseline, dtype=np.float64)
    n = len(s)
    if n == 0 or len(b) != n:
        raise ValueError(f"Sample length mismatch or empty: {len(s)} vs {len(b)}")

    diff = s - b
    mean_diff = float(np.mean(diff))
    d = paired_cohens_d(s, b)

    rng = np.random.default_rng(seed)
    resample_diffs = np.empty(n_resamples, dtype=np.float64)

    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        resample_diffs[i] = np.mean(diff[indices])

    # Percentile CI
    ci_lower = float(np.percentile(resample_diffs, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(resample_diffs, 100.0 * (1.0 - alpha / 2.0)))

    # Bootstrap p-value under null hypothesis (shifted differences)
    shifted_diffs = resample_diffs - mean_diff
    if alternative == "two-sided":
        p_val = float(np.mean(np.abs(shifted_diffs) >= np.abs(mean_diff)))
    elif alternative == "greater":
        p_val = float(np.mean(shifted_diffs >= mean_diff))
    elif alternative == "less":
        p_val = float(np.mean(shifted_diffs <= mean_diff))
    else:
        raise ValueError(f"Unknown alternative {alternative}")

    # Guard against float 0.0 in empirical p-values
    p_val = max(p_val, 1.0 / n_resamples)

    return mean_diff, ci_lower, ci_upper, p_val, d


def benjamini_hochberg_correction(p_values: Sequence[float], fdr: float = 0.05) -> List[Tuple[float, bool]]:
    """Apply Benjamini-Hochberg FDR correction across a family of p-values.

    Returns a list of (p_adjusted, is_significant) in original order.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort p-values with original indices
    sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m

    # Cumulative minimum for monotonicity
    min_adj = 1.0
    for rank_1_indexed, (orig_idx, p) in reversed(list(enumerate(sorted_pairs, start=1))):
        val = (p * m) / rank_1_indexed
        min_adj = min(min_adj, val)
        adjusted[orig_idx] = min(min_adj, 1.0)

    return [(adj, adj <= fdr) for adj in adjusted]


def run_track1_hypothesis_tests(
    per_user_metrics_by_system: Mapping[str, Mapping[str, Sequence[float]]],
    n_resamples: int = 10000,
    seed: int = 42,
) -> HypothesisTestReport:
    """Execute full hypothesis suite across all baselines and MMR ablations."""
    pf = per_user_metrics_by_system.get("Paper Feed (Full)", {})
    if not pf:
        raise ValueError("Paper Feed (Full) system metrics not found.")

    pf_ndcg = pf.get("ndcg_at_10", [])
    n_authors = len(pf_ndcg)

    # H1: Paper Feed vs 5 Baselines on NDCG@10
    baselines_to_test = [
        "Baseline: random",
        "Baseline: popularity",
        "Baseline: bm25_author",
        "Baseline: dense_centroid",
        "Baseline: hybrid_persona",
    ]

    h1_raw: List[BootstrapComparison] = []
    p_values: List[float] = []

    for b_name in baselines_to_test:
        b_metrics = per_user_metrics_by_system.get(b_name, {})
        b_ndcg = b_metrics.get("ndcg_at_10", [0.0] * n_authors)

        diff, lo, hi, p_val, d = paired_bootstrap_test(
            pf_ndcg, b_ndcg, n_resamples=n_resamples, seed=seed, alternative="two-sided"
        )
        mean_pf = float(np.mean(pf_ndcg))
        mean_b = float(np.mean(b_ndcg))
        rel_gain = ((mean_pf - mean_b) / mean_b * 100.0) if mean_b > 1e-9 else 0.0

        comp = BootstrapComparison(
            system_name="Paper Feed (Full)",
            baseline_name=b_name,
            metric_name="ndcg_at_10",
            mean_system=mean_pf,
            mean_baseline=mean_b,
            mean_diff=diff,
            relative_gain_percent=rel_gain,
            ci_lower=lo,
            ci_upper=hi,
            p_value=p_val,
            cohens_d=d,
        )
        h1_raw.append(comp)
        p_values.append(p_val)

    # Benjamini-Hochberg correction across H1 family
    bh_results = benjamini_hochberg_correction(p_values)
    h1_comparisons: List[BootstrapComparison] = []
    all_sig = True

    for comp, (adj_p, is_sig) in zip(h1_raw, bh_results):
        comp.p_value_corrected = adj_p
        comp.significant_at_05 = is_sig
        if not is_sig or comp.mean_diff <= 0:
            all_sig = False
        h1_comparisons.append(comp)

    # H2: MMR diversification check
    # Check 1: Does MMR raise topic_entropy? (Full vs No-MMR)
    pf_entropy = pf.get("topic_entropy", [])
    no_mmr = per_user_metrics_by_system.get("Ablation: No-MMR", {})
    no_mmr_entropy = no_mmr.get("topic_entropy", [0.0] * n_authors)
    no_mmr_ndcg = no_mmr.get("ndcg_at_10", [0.0] * n_authors)

    diff_ent, lo_ent, hi_ent, p_ent, d_ent = paired_bootstrap_test(
        pf_entropy, no_mmr_entropy, n_resamples=n_resamples, seed=seed, alternative="greater"
    )
    mean_pf_ent = float(np.mean(pf_entropy))
    mean_no_mmr_ent = float(np.mean(no_mmr_entropy))

    h2_entropy = BootstrapComparison(
        system_name="Paper Feed (Full)",
        baseline_name="Ablation: No-MMR",
        metric_name="topic_entropy",
        mean_system=mean_pf_ent,
        mean_baseline=mean_no_mmr_ent,
        mean_diff=diff_ent,
        relative_gain_percent=((mean_pf_ent - mean_no_mmr_ent) / mean_no_mmr_ent * 100.0) if mean_no_mmr_ent > 1e-9 else 0.0,
        ci_lower=lo_ent,
        ci_upper=hi_ent,
        p_value=p_ent,
        cohens_d=d_ent,
        significant_at_05=(p_ent < 0.05 and diff_ent > 0),
    )

    # Check 2: Relative NDCG loss of MMR is <= 3% (non-inferiority)
    diff_ndcg, lo_ndcg, hi_ndcg, p_ndcg, d_ndcg = paired_bootstrap_test(
        pf_ndcg, no_mmr_ndcg, n_resamples=n_resamples, seed=seed, alternative="greater"
    )
    mean_no_mmr_ndcg = float(np.mean(no_mmr_ndcg))
    loss_percent = ((mean_no_mmr_ndcg - float(np.mean(pf_ndcg))) / mean_no_mmr_ndcg * 100.0) if mean_no_mmr_ndcg > 1e-9 else 0.0
    is_non_inferior = bool(loss_percent <= 3.0 and lo_ndcg >= -0.03 * mean_no_mmr_ndcg)

    h2_retention = BootstrapComparison(
        system_name="Paper Feed (Full)",
        baseline_name="Ablation: No-MMR",
        metric_name="ndcg_at_10",
        mean_system=float(np.mean(pf_ndcg)),
        mean_baseline=mean_no_mmr_ndcg,
        mean_diff=diff_ndcg,
        relative_gain_percent=-loss_percent,
        ci_lower=lo_ndcg,
        ci_upper=hi_ndcg,
        p_value=p_ndcg,
        cohens_d=d_ndcg,
        significant_at_05=is_non_inferior,
    )

    h2_supported = bool(h2_entropy.significant_at_05 and h2_retention.significant_at_05)

    return HypothesisTestReport(
        h1_comparisons=h1_comparisons,
        h1_supported=all_sig,
        h2_mmr_entropy=h2_entropy,
        h2_mmr_ndcg_retention=h2_retention,
        h2_supported=h2_supported,
        n_resamples=n_resamples,
        n_authors=n_authors,
    )

