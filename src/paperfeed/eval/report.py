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
        if name.startswith("Diagnostic:"):
            continue  # different ground truth (own past references): reported in its own table
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


def format_diagnostic_table(system_aggregates: Mapping[str, Mapping[str, float]]) -> str:
    """Diagnostic re-citation run (plan 6.4): scored on ReCitation targets, so never compared with Discovery rows."""
    rows = [(n, m) for n, m in system_aggregates.items() if n.startswith("Diagnostic:")]
    if not rows:
        return ""
    lines = [
        "| Run | Ground truth | Recall@10 | NDCG@10 | HitRate@10 | Candidate Recall@100 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]
    for name, m in rows:
        lines.append(
            f"| {name} | ReCitation (own past references cited again after T) | {m.get('recall_at_10', 0.0):.4f} | "
            f"{m.get('ndcg_at_10', 0.0):.4f} | {m.get('hit_rate_at_10', 0.0):.4f} | {m.get('recall_at_100', 0.0):.4f} |"
        )
    lines.append("")
    lines.append(
        "*Citations(u, <= T) are removed from the exclusion set, so the feed may re-recommend the author's own past "
        "references. Different ground truth from every other table: not comparable with the Discovery numbers.*"
    )
    return "\n".join(lines)


def format_coverage_ceiling_table(system_aggregates: Mapping[str, Mapping[str, float]], graph_channel: bool = False) -> str:
    """Candidate coverage ceiling (plan 6.6): Recall@100 of the first-stage pool bounds what re-ranking can reach."""
    names = [n for n in ("Paper Feed (Full)", "Ablation: No-MMR", "Ablation: No-Graph", "Ablation: No-Facets") if n in system_aggregates]
    if not names:
        return ""
    lines = [
        "| System | Candidate pool Recall@100 (ceiling) | Recall@10 | Share of ceiling reached |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for name in names:
        m = system_aggregates[name]
        ceiling, r10 = m.get("recall_at_100", 0.0), m.get("recall_at_10", 0.0)
        share = f"{r10 / ceiling * 100.0:.1f}%" if ceiling > 1e-9 else "-"
        lines.append(f"| {name} | {ceiling:.4f} | {r10:.4f} | {share} |")
    cit = system_aggregates.get("Baseline: citation_only")
    lines.append("")
    if graph_channel:
        note = (
            "*The pool fuses the text channels (BM25 + SPECTER2) and the V4 citation-graph channel by RRF; the No-Graph "
            "ablation drops the graph channel too, so its ceiling is the text-only pool. Targets outside the pool cannot "
            "be recommended whatever the scoring weights."
        )
    else:
        note = (
            "*The pool comes from the text channels (BM25 + SPECTER2, RRF); targets outside it cannot be recommended "
            "whatever the scoring weights."
        )
    if cit:
        note += (
            f" For reference, Citation-only ranks the whole eligible corpus by the graph signal and reaches "
            f"Recall@10 = {cit.get('recall_at_10', 0.0):.4f}."
        )
    lines.append(note + "*")
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
    ent = report.h2_mmr_entropy
    ndcg = report.h2_mmr_ndcg_retention
    if not getattr(report, "h2_testable", True):
        lines.append(
            "**Verdict:** NOT TESTABLE: the tuned lambda_div is 0, so the full system and the No-MMR ablation are "
            "identical and there is no MMR effect to measure.\n"
        )
        return "\n".join(lines)
    lines.append(f"**Verdict:** {'ACCEPTED (Supported by empirical data)' if report.h2_supported else 'REJECTED / INCONCLUSIVE'}\n")
    lines.append(
        f"- **Entropy Increase (one-sided):** Delta = {ent.mean_diff:+.4f} ({ent.relative_gain_percent:+.2f}%), "
        f"95% CI = [{ent.ci_lower:+.4f}, {ent.ci_upper:+.4f}], p = {ent.p_value:.4e}, MDE = {ent.mde:.4f} (Significant: {ent.significant_at_05})"
    )
    # Difference is Full - No-MMR, so the CI lower end is the largest plausible NDCG loss caused by MMR.
    worst_loss = (max(0.0, -ndcg.ci_lower) / ndcg.mean_baseline * 100.0) if ndcg.mean_baseline > 1e-9 else 0.0
    lines.append(
        f"- **NDCG Retention (non-inferiority: relative loss <= 3%):** NDCG@10 change with MMR = "
        f"{ndcg.relative_gain_percent:+.2f}% (Delta = {ndcg.mean_diff:+.4f}, 95% CI = [{ndcg.ci_lower:+.4f}, "
        f"{ndcg.ci_upper:+.4f}]); worst-case relative loss from the CI = {worst_loss:.2f}%, MDE = {ndcg.mde:.4f} "
        f"(Non-inferior: {ndcg.significant_at_05})"
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
    graph_channel: bool = False,
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
    ])
    coverage = format_coverage_ceiling_table(system_aggregates, graph_channel=graph_channel)
    if coverage:
        parts.extend(["\n### Candidate Coverage Ceiling\n", coverage])
    diagnostic = format_diagnostic_table(system_aggregates)
    if diagnostic:
        parts.extend(["\n### Diagnostic: Re-Citation\n", diagnostic])
    parts.extend([
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



# ---------------------------------------------------------------------------
# Phase 11 tables (plan 11.5)
# ---------------------------------------------------------------------------

def format_sota_table(system_aggregates: Mapping[str, Mapping[str, float]], latency: Mapping[str, float],
                      coverage: Optional[float]) -> str:
    """Paper Feed, the comparison systems, the Track 1 baselines and the Random anchor on the same test authors."""
    oracle = (system_aggregates.get("Anchor: Oracle@10") or {}).get("ndcg_at_10", 1.0) or 1.0
    names = (["Paper Feed (Full)"] + [n for n in system_aggregates if n.startswith("SOTA: ")]
             + [n for n in system_aggregates if n.startswith("Baseline: ")]
             + [n for n in system_aggregates if n.startswith("Anchor: Random")])
    lines = [
        "| System | NDCG@10 | % of Oracle | Recall@10 | HitRate@10 | MRR | Topic Entropy | Explanation Coverage | "
        "Latency per feed (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name in names:
        m = system_aggregates.get(name)
        if not m:
            continue
        cov = f"{coverage:.1%}" if (name == "Paper Feed (Full)" and coverage is not None) else "none"
        lat = f"{latency[name]:.2f}" if name in latency else "-"
        bold = "**" if name == "Paper Feed (Full)" else ""
        lines.append(
            f"| {bold}{name}{bold} | {m.get('ndcg_at_10', 0.0):.4f} | {m.get('ndcg_at_10', 0.0) / oracle * 100:.1f}% | "
            f"{m.get('recall_at_10', 0.0):.4f} | {m.get('hit_rate_at_10', 0.0):.4f} | {m.get('mrr', 0.0):.4f} | "
            f"{m.get('topic_entropy', 0.0):.4f} | {cov} | {lat} |"
        )
    lines.append("")
    lines.append("*Explanation Coverage is a capability difference, not ranked against accuracy: only Paper Feed explains "
                 "its recommendations. Latency is the mean time to produce one feed on the run machine; baselines and "
                 "the anchor were not timed (\"-\"). The comparison systems' one-off setup (graph, PCA, model loading) "
                 "is excluded.*")
    return "\n".join(lines)


def format_best_config_table(best: Mapping[str, Any]) -> str:
    """configs/best.yaml as the plan's best-configuration table: part, choice, tuning evidence."""
    lines = [
        f"Best configuration: **{best.get('best_configuration')}**. {best.get('replacement_rule', '')} "
        f"{best.get('assembled_check', '')}\n",
        "| Part | Chosen option | Replaces Core | Tuning evidence | n | Rationale |",
        "| :--- | :--- | :---: | :--- | :---: | :--- |",
    ]
    for part, p in (best.get("parts") or {}).items():
        ev = p.get("evidence")
        if not ev:
            evidence, n = "none", "-"
        elif "ci95" in ev:
            evidence, n = f"gain {ev['gain']:+.4f} [{ev['ci95'][0]:+.4f}, {ev['ci95'][1]:+.4f}]", ev.get("n_users", "-")
        else:
            evidence, n = f"NDCG@10 {ev.get('value', 0.0):.4f}", ev.get("n_users", "-")
        lines.append(f"| {part} | `{p.get('choice')}` | {'yes' if p.get('replaces_core') else 'no'} | {evidence} | {n} | "
                     f"{p.get('rationale', '')} |")
    return "\n".join(lines)
