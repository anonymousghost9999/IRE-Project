"""Altair visualization helpers for PaperFeed UI."""

from typing import Any, Dict, List, Mapping, Sequence
import pandas as pd
import altair as alt


def create_facet_bar_chart(facets_data: Sequence[Mapping[str, Any]], top_n: int = 10) -> alt.Chart:
    """Create a sleek horizontal bar chart of top canonical facets."""
    if not facets_data:
        df = pd.DataFrame({"facet": ["None"], "count": [0], "type": ["default"]})
    else:
        df = pd.DataFrame(facets_data[:top_n])
        if "count" not in df.columns:
            df["count"] = 1

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X("count:Q", title="Frequency / Weight", axis=alt.Axis(labels=True, ticks=False)),
            y=alt.Y("facet:N", sort="-x", title="", axis=alt.Axis(labels=True, ticks=False)),
            color=alt.Color(
                "type:N",
                scale=alt.Scale(
                    domain=["task", "method", "dataset", "metric", "topic", "other"],
                    range=["#3B82F6", "#8B5CF6", "#10B981", "#F59E0B", "#EC4899", "#94A3B8"],
                ),
                legend=alt.Legend(title="Facet Type", orient="bottom"),
            ),
            tooltip=["facet:N", "type:N", "count:Q"],
        )
        .properties(height=max(160, len(df) * 22))
        .configure_view(strokeWidth=0)
        .configure_axis(gridColor="#2E3348", labelColor="#94A3B8", titleColor="#CBD5E1")
    )
    return chart


def create_timeline_chart(papers: Sequence[Mapping[str, Any]]) -> alt.Chart:
    """Create bar chart of authored papers across years."""
    if not papers:
        df = pd.DataFrame({"year": [2024], "count": [0]})
    else:
        counts = {}
        for p in papers:
            yr = p.get("year") or 2024
            counts[yr] = counts.get(yr, 0) + 1
        df = pd.DataFrame([{"year": int(k), "count": int(v)} for k, v in sorted(counts.items())])

    chart = (
        alt.Chart(df)
        .mark_bar(color="#4A9EFF", cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("year:O", title="Publication Year"),
            y=alt.Y("count:Q", title="Papers Authored"),
            tooltip=["year:O", "count:Q"],
        )
        .properties(height=180)
        .configure_view(strokeWidth=0)
        .configure_axis(gridColor="#2E3348", labelColor="#94A3B8", titleColor="#CBD5E1")
    )
    return chart


def create_score_breakdown_chart(contributions: Mapping[str, float]) -> alt.Chart:
    """Create horizontal bar chart breaking down linear score components."""
    rows = []
    labels = {
        "persona": "Persona Match",
        "session": "Session Focus",
        "citation": "Citation Graph",
        "facet": "Facet Match",
        "new_method": "New Method",
        "penalty": "Negative Feedback",
    }
    for k, v in contributions.items():
        if k in labels and abs(v) > 1e-4:
            rows.append({"feature": labels[k], "weight": round(float(v), 3)})

    if not rows:
        rows = [{"feature": "Persona Match", "weight": 1.0}]

    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X("weight:Q", title="Feature Contribution (w_k * mm_k)", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("feature:N", sort="-x", title=""),
            color=alt.condition(
                alt.datum.weight > 0,
                alt.value("#38BDF8"),
                alt.value("#F87171"),
            ),
            tooltip=["feature:N", "weight:Q"],
        )
        .properties(height=140)
        .configure_view(strokeWidth=0)
        .configure_axis(gridColor="#2E3348", labelColor="#94A3B8", titleColor="#CBD5E1")
    )
    return chart
