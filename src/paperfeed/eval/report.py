"""Report and table generator for Track 1 Leave-Future-Out Evaluation (Phase 6.6 / plan.md §6.6 / design.md §13.1).

Renders:
1. Main Results Table (Paper Feed vs Baselines vs Anchors, with % of Oracle).
2. Component Ablations Table (Full vs No-MMR vs No-Graph vs No-Facets).
3. History Slice Breakdown Table (Warm, Moderate, Sparse, Zero-Pub).
4. Hypothesis Testing Summary (H1 and H2 with paired bootstrap CIs and Cohen's d).
Outputs both Markdown and LaTeX formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from paperfeed.eval.stats import HypothesisTestReport


def format_main_table_markdown(system_aggregates: Mapping[str, Mapping[str, float]]) -> str:
    """Format the primary comparison table in GitHub Markdown."""
    lines = [
        "| System Architecture | NDCG@10 | % of Oracle | Recall@10 | HitRate@10 | MRR | Topic Entropy | ILD |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    oracle_ndcg = (system_aggregates.get("Anchor: Oracle@10") or {}).get("ndcg_at_10", 1.0)
    if oracle_ndcg <= 1e-9:
        oracle_ndcg = 1.0

    for name, m in system_aggregates.items():
        ndcg = m.get("ndcg_at_10", 0.0)
        pct_oracle = (ndcg / oracle_ndcg) * 100.0
        rec = m.get("recall_at_10", 0.0)
        hr = m.get("hit_rate_at_10", 0.0)
        mrr_val = m.get("mrr", 0.0)
        ent = m.get("topic_entropy", 0.0)
        ild = m.get("ild", 0.0)

        bold = "**" if "Paper Feed (Full)" in name else ""
        lines.append(
            f"| {bold}{name}{bold} | {ndcg:.4f} | {pct_oracle:.1f}% | {rec:.4f} | {hr:.4f} | {mrr_val:.4f} | {ent:.4f} | {ild:.4f} |"
        )

    return "\n".join(lines)


def format_ablation_table_markdown(system_aggregates: Mapping[str, Mapping[str, float]]) -> str:
    """Format component ablations table."""
    ablation_names = [
        "Paper Feed (Full)",
        "Ablation: No-MMR",
        "Ablation: No-Graph",
        "Ablation: No-Facets",
    ]
    lines = [
        "| Configuration | NDCG@10 | Delta NDCG | Recall@10 | Topic Entropy | Delta Entropy |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    full_ndcg = (system_aggregates.get("Paper Feed (Full)") or {}).get("ndcg_at_10", 0.0)
    full_ent = (system_aggregates.get("Paper Feed (Full)") or {}).get("topic_entropy", 0.0)

    for name in ablation_names:
        m = system_aggregates.get(name, {})
        ndcg = m.get("ndcg_at_10", 0.0)
        rec = m.get("recall_at_10", 0.0)
        ent = m.get("topic_entropy", 0.0)

        delta_ndcg = ndcg - full_ndcg
        delta_ent = ent - full_ent

        lines.append(
            f"| {name} | {ndcg:.4f} | {delta_ndcg:+.4f} | {rec:.4f} | {ent:.4f} | {delta_ent:+.4f} |"
        )

    return "\n".join(lines)


def format_hypothesis_markdown(report: HypothesisTestReport) -> str:
    """Format hypothesis findings (H1 and H2)."""
    lines = [
        "### Hypothesis Testing Summary\n",
        f"**Sample Size:** N = {report.n_authors} test researchers | **Bootstrap Resamples:** B = {report.n_resamples:,}\n",
        "#### Hypothesis H1: Paper Feed Outperforms Baselines (NDCG@10)",
        f"**Verdict:** {'ACCEPTED (Supported by empirical data)' if report.h1_supported else 'REJECTED / INCONCLUSIVE'}\n",
        "| Comparison Baseline | System NDCG | Baseline NDCG | Rel. Gain | 95% CI of Difference | p-value (raw) | p-value (BH FDR) | Cohen's d | Significant? |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for c in report.h1_comparisons:
        sig = "YES (p < 0.05)" if c.significant_at_05 else "NO"
        lines.append(
            f"| {c.baseline_name} | {c.mean_system:.4f} | {c.mean_baseline:.4f} | {c.relative_gain_percent:+.2f}% | [{c.ci_lower:+.4f}, {c.ci_upper:+.4f}] | {c.p_value:.4e} | {c.p_value_corrected:.4e} | {c.cohens_d:.3f} | {sig} |"
        )

    lines.append("\n#### Hypothesis H2: MMR Diversity with Non-Inferiority")
    lines.append(f"**Verdict:** {'ACCEPTED (Supported by empirical data)' if report.h2_supported else 'REJECTED / INCONCLUSIVE'}\n")
    ent = report.h2_mmr_entropy
    ndcg = report.h2_mmr_ndcg_retention
    lines.append(
        f"- **Entropy Increase (one-sided):** Delta = {ent.mean_diff:+.4f} ({ent.relative_gain_percent:+.2f}%), "
        f"95% CI = [{ent.ci_lower:+.4f}, {ent.ci_upper:+.4f}], p = {ent.p_value:.4e} (Significant: {ent.significant_at_05})"
    )
    lines.append(
        f"- **NDCG Retention (non-inferiority <= 3% loss):** Rel. Loss = {abs(ndcg.relative_gain_percent):.2f}%, "
        f"95% CI = [{ndcg.ci_lower:+.4f}, {ndcg.ci_upper:+.4f}], p = {ndcg.p_value:.4e} (Non-inferior: {ndcg.significant_at_05})"
    )

    return "\n".join(lines)


def generate_full_evaluation_report(
    system_aggregates: Mapping[str, Mapping[str, float]],
    hypothesis_report: HypothesisTestReport,
    slice_results: Optional[Mapping[str, Mapping[str, Mapping[str, float]]]] = None,
) -> str:
    """Generate complete markdown report containing all tables and hypothesis results."""
    parts = [
        "# Paper Feed — Track 1: Leave-Future-Out Empirical Evaluation Report\n",
        "## 1. Main Benchmark Results\n",
        format_main_table_markdown(system_aggregates),
        "\n## 2. Component Ablations\n",
        format_ablation_table_markdown(system_aggregates),
        "\n## 3. Statistical Hypothesis Testing\n",
        format_hypothesis_markdown(hypothesis_report),
    ]

    if slice_results:
        parts.append("\n## 4. History Slice Stratification\n")
        lines = [
            "| Stratum | Authors (n) | Paper Feed NDCG@10 | Best Baseline NDCG@10 | Relative Gain |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
        for s_name, sys_dict in slice_results.items():
            pf = sys_dict.get("Paper Feed (Full)", {})
            pf_ndcg = pf.get("ndcg_at_10", 0.0)
            n_auth = int(pf.get("n_authors", 0))

            base_ndcgs = [
                v.get("ndcg_at_10", 0.0)
                for k, v in sys_dict.items()
                if k.startswith("Baseline:")
            ]
            best_base = max(base_ndcgs) if base_ndcgs else 0.0
            gain = ((pf_ndcg - best_base) / best_base * 100.0) if best_base > 1e-9 else 0.0
            lines.append(
                f"| {s_name.capitalize()} | {n_auth} | {pf_ndcg:.4f} | {best_base:.4f} | {gain:+.2f}% |"
            )
        parts.append("\n".join(lines))

    return "\n".join(parts)

