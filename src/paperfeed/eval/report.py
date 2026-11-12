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
from typing import Any, Dict, List, Mapping, Optional, Tuple

from paperfeed.eval.stats import BootstrapComparison, HypothesisTestReport


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
        "#### Hypothesis H1: Paper Feed Outperforms the Five Baselines (NDCG@10 and Recall@10)",
        f"**Verdict:** {'ACCEPTED (Supported by empirical data)' if report.h1_supported else 'REJECTED / INCONCLUSIVE'}\n",
        format_comparisons_markdown(report.h1_comparisons),
        "",
        "*BH correction runs over all baseline x metric comparisons. MDE: smallest true difference detectable with 80% "
        "power at alpha = 0.05 and this N; a non-significant row whose observed difference is below its MDE is "
        "inconclusive, not evidence of no effect.*",
    ]

    lines.append("\n#### Hypothesis H2: MMR Diversity with Non-Inferiority")
    lines.append(f"**Verdict:** {'ACCEPTED (Supported by empirical data)' if report.h2_supported else 'REJECTED / INCONCLUSIVE'}\n")
    ent = report.h2_mmr_entropy
    ndcg = report.h2_mmr_ndcg_retention
    lines.append(
        f"- **Entropy Increase (one-sided):** Delta = {ent.mean_diff:+.4f} ({ent.relative_gain_percent:+.2f}%), "
        f"95% CI = [{ent.ci_lower:+.4f}, {ent.ci_upper:+.4f}], p = {ent.p_value:.4e}, MDE = {ent.mde:.4f} (Significant: {ent.significant_at_05})"
    )
    lines.append(
        f"- **NDCG Retention (non-inferiority <= 3% loss):** Rel. Loss = {abs(ndcg.relative_gain_percent):.2f}%, "
        f"95% CI = [{ndcg.ci_lower:+.4f}, {ndcg.ci_upper:+.4f}], p = {ndcg.p_value:.4e}, MDE = {ndcg.mde:.4f} (Non-inferior: {ndcg.significant_at_05})"
    )

    return "\n".join(lines)


def format_comparisons_markdown(comparisons: List[BootstrapComparison]) -> str:
    """One row per (baseline, metric) paired-bootstrap comparison."""
    lines = [
        "| Comparison Baseline | Metric | System | Baseline | Rel. Gain | 95% CI of Difference | p (raw) | p (BH) | Cohen's d | MDE | Significant? |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for c in comparisons:
        sig = "YES (p < 0.05)" if c.significant_at_05 else "NO"
        lines.append(
            f"| {c.baseline_name} | {c.metric_name} | {c.mean_system:.4f} | {c.mean_baseline:.4f} | {c.relative_gain_percent:+.2f}% "
            f"| [{c.ci_lower:+.4f}, {c.ci_upper:+.4f}] | {c.p_value:.4e} | {c.p_value_corrected:.4e} | {c.cohens_d:.3f} "
            f"| {c.mde:.4f} | {sig} |"
        )
    return "\n".join(lines)


def format_zero_pub_section(
    system_aggregates: Mapping[str, Mapping[str, float]],
    comparisons: List[BootstrapComparison],
    n_authors: int,
) -> str:
    """Zero-Publication Seed-Only slice: its own systems and comparisons, never mixed with the main tables."""
    return "\n".join([
        f"Re-run of the test cohort with each profile replaced by the author's 3 seed topics (n = {n_authors}): "
        "E_u is empty and the ground truth is Ref(B) <= T minus Authored(u). Lexical and Dense baselines query with "
        "the seed topics, Popularity uses the seed topics' subfield, Citation-only is not applicable. Compare only "
        "within this table.\n",
        format_main_table_markdown(system_aggregates),
        "",
        format_comparisons_markdown(comparisons),
    ])


def format_cold_start_table(breakdown: Mapping[str, Mapping[str, Mapping[str, float]]]) -> str:
    """Recall@10 over fresh vs established targets, per system, with n per class."""
    classes = [c for c in ("fresh", "established") if c in breakdown]
    systems = list(dict.fromkeys(name for c in classes for name in breakdown[c]))
    header = "| System | " + " | ".join(
        f"Recall@10 {c} (n = {int(next(iter(breakdown[c].values()))['n_authors']) if breakdown[c] else 0} authors)" for c in classes
    ) + " |"
    lines = [header, "| :--- |" + " :---: |" * len(classes)]
    for name in systems:
        cells = [f"{breakdown[c][name]['recall_at_10']:.4f}" if name in breakdown[c] else "-" for c in classes]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "*fresh: citation_count_T <= 1 and published in [T-1, T]; established: citation_count_T >= 20 and published "
        "since 2012 (design §12). Each class averages over the authors with at least one target in it.*"
    )
    return "\n".join(lines)


def generate_full_evaluation_report(
    system_aggregates: Mapping[str, Mapping[str, float]],
    hypothesis_report: HypothesisTestReport,
    slice_results: Optional[Mapping[str, Mapping[str, Mapping[str, float]]]] = None,
    run_notes: Optional[str] = None,
    extra_sections: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Generate complete markdown report containing all tables and hypothesis results."""
    parts = [
        "# Paper Feed — Track 1: Leave-Future-Out Empirical Evaluation Report\n",
    ]
    if run_notes:
        parts.append(run_notes.strip() + "\n\n")
    parts.extend([
        "## 1. Main Benchmark Results\n",
        format_main_table_markdown(system_aggregates),
        "\n## 2. Component Ablations\n",
        format_ablation_table_markdown(system_aggregates),
        "\n## 3. Statistical Hypothesis Testing\n",
        format_hypothesis_markdown(hypothesis_report),
    ])

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

    for i, (title, body) in enumerate(extra_sections or [], start=5):
        parts.append(f"\n## {i}. {title}\n")
        parts.append(body)

    return "\n".join(parts)

