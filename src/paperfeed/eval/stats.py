"""Statistical testing and hypothesis validation for Track 1 (Phase 6.5 / plan.md §6.5 / design.md §13.1).

Validates:
- H1: Paper Feed significantly outperforms all 5 baselines on Discovery NDCG@10 and Recall@10
      using paired bootstrap resampling and Benjamini-Hochberg (BH) FDR correction.
- H2: MMR diversification significantly increases Topic Entropy (one-sided paired bootstrap)
      with non-inferiority on NDCG@10 (relative NDCG loss <= 3%).
- Effect sizes: Paired Cohen's d.
- 95% Confidence Intervals via percentile bootstrap.
- Minimum detectable effect (MDE) per comparison from the bootstrap standard error, so a non-significant result
  can be read as "inconclusive at this N" rather than "no effect" (plan §6.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from statistics import NormalDist
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# The H1 family (plan §6.2 / design §13.1); must match ``paperfeed.eval.baselines.BASELINE_NAMES``.
H1_BASELINES = ("popularity", "bm25_author", "dense_centroid", "citation_only", "hybrid_persona")
# H1 is tested on both headline metrics (plan §6.5); BH correction runs over all baseline x metric comparisons.
H1_METRICS = ("ndcg_at_10", "recall_at_10")


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
    mde: float = 0.0  # smallest true mean difference detectable with 80% power at alpha = 0.05 and this N


@dataclass
class HypothesisTestReport:
    """Full statistical hypothesis testing summary."""

    h1_comparisons: List[BootstrapComparison]
    h1_supported: bool  # True if Paper Feed beats ALL 5 baselines on NDCG@10 AND Recall@10 (BH-corrected p < 0.05)
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


def bootstrap_mde(
    scores_system: Sequence[float],
    scores_baseline: Sequence[float],
    n_resamples: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Minimum detectable effect: (z_{1-alpha/2} + z_{power}) * bootstrap standard error of the mean paired difference.

    Uses the same resamples as ``paired_bootstrap_test`` (same seed), so the two describe the same bootstrap.
    """
    diff = np.asarray(scores_system, dtype=np.float64) - np.asarray(scores_baseline, dtype=np.float64)
    n = len(diff)
    if n == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(diff[rng.integers(0, n, size=n)]) for _ in range(n_resamples)])
    se = float(np.std(means, ddof=1)) if n_resamples > 1 else 0.0
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0) + NormalDist().inv_cdf(power)
    return float(z * se)


def compare_system_to_baselines(
    per_user_metrics_by_system: Mapping[str, Mapping[str, Sequence[float]]],
    system_name: str,
    baseline_names: Sequence[str],
    metrics: Sequence[str] = H1_METRICS,
    n_resamples: int = 10000,
    seed: int = 42,
) -> List[BootstrapComparison]:
    """Paired bootstrap of ``system_name`` against every baseline on every metric, BH-corrected as one family.

    Raises if a system or baseline lacks a metric: comparing against a missing series must never default to zeros.
    """
    needed = [system_name, *baseline_names]
    missing = [(name, m) for name in needed for m in metrics if m not in per_user_metrics_by_system.get(name, {})]
    if missing:
        raise ValueError(f"Comparison needs every per-user series; missing: {missing}")

    comparisons: List[BootstrapComparison] = []
    for metric in metrics:
        sys_scores = per_user_metrics_by_system[system_name][metric]
        for b_name in baseline_names:
            b_scores = per_user_metrics_by_system[b_name][metric]
            diff, lo, hi, p_val, d = paired_bootstrap_test(
                sys_scores, b_scores, n_resamples=n_resamples, seed=seed, alternative="two-sided"
            )
            mean_s, mean_b = float(np.mean(sys_scores)), float(np.mean(b_scores))
            comparisons.append(
                BootstrapComparison(
                    system_name=system_name,
                    baseline_name=b_name,
                    metric_name=metric,
                    mean_system=mean_s,
                    mean_baseline=mean_b,
                    mean_diff=diff,
                    relative_gain_percent=((mean_s - mean_b) / mean_b * 100.0) if mean_b > 1e-9 else 0.0,
                    ci_lower=lo,
                    ci_upper=hi,
                    p_value=p_val,
                    cohens_d=d,
                    mde=bootstrap_mde(sys_scores, b_scores, n_resamples=n_resamples, seed=seed),
                )
            )

    for comp, (adj_p, is_sig) in zip(comparisons, benjamini_hochberg_correction([c.p_value for c in comparisons])):
        comp.p_value_corrected = adj_p
        comp.significant_at_05 = is_sig
    return comparisons


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

    # H1: Paper Feed vs the 5 baselines of plan §6.2 on NDCG@10 and Recall@10 (Random is an anchor, never tested).
    h1_comparisons = compare_system_to_baselines(
        per_user_metrics_by_system,
        "Paper Feed (Full)",
        [f"Baseline: {name}" for name in H1_BASELINES],
        metrics=H1_METRICS,
        n_resamples=n_resamples,
        seed=seed,
    )
    all_sig = all(c.significant_at_05 and c.mean_diff > 0 for c in h1_comparisons)

    # H2: MMR diversification check
    # Check 1: Does MMR raise topic_entropy? (Full vs No-MMR)
    no_mmr = per_user_metrics_by_system.get("Ablation: No-MMR", {})
    h2_missing = [m for m in ("topic_entropy", "ndcg_at_10") if m not in pf or m not in no_mmr]
    if h2_missing:
        raise ValueError(f"H2 needs Paper Feed (Full) and Ablation: No-MMR series for {h2_missing}")
    pf_entropy = pf["topic_entropy"]
    no_mmr_entropy = no_mmr["topic_entropy"]
    no_mmr_ndcg = no_mmr["ndcg_at_10"]

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
        mde=bootstrap_mde(pf_entropy, no_mmr_entropy, n_resamples=n_resamples, seed=seed),
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
        mde=bootstrap_mde(pf_ndcg, no_mmr_ndcg, n_resamples=n_resamples, seed=seed),
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

