"""Facet pill badges and color mappings for PaperFeed UI."""

import re
from typing import Dict, List, Mapping, Optional, Sequence


FACET_COLORS = {
    "task": {"color": "#3B82F6", "bg": "rgba(59, 130, 246, 0.15)", "border": "rgba(59, 130, 246, 0.3)"},
    "method": {"color": "#8B5CF6", "bg": "rgba(139, 92, 246, 0.15)", "border": "rgba(139, 92, 246, 0.3)"},
    "dataset": {"color": "#10B981", "bg": "rgba(16, 185, 129, 0.15)", "border": "rgba(16, 185, 129, 0.3)"},
    "metric": {"color": "#F59E0B", "bg": "rgba(245, 158, 11, 0.15)", "border": "rgba(245, 158, 11, 0.3)"},
    "topic": {"color": "#EC4899", "bg": "rgba(236, 72, 153, 0.15)", "border": "rgba(236, 72, 153, 0.3)"},
    "citation": {"color": "#34D399", "bg": "rgba(52, 211, 153, 0.15)", "border": "rgba(52, 211, 153, 0.3)"},
    "default": {"color": "#94A3B8", "bg": "rgba(148, 163, 184, 0.15)", "border": "rgba(148, 163, 184, 0.3)"},
}

GENERIC_FACET_TERMS = {
    "method", "methods", "approach", "approaches", "technique", "techniques",
    "framework", "frameworks", "model", "models", "algorithm", "algorithms",
    "system", "systems", "manner", "strategy", "strategies", "paradigm", "paradigms",
    "process", "processes", "procedure", "procedures", "work", "study", "paper",
    "execution", "implementation", "solution", "solutions", "tool", "tools",
    "experiment", "experiments", "experimental results", "results", "baseline", "baselines",
    "baseline experiments", "evaluation", "evaluation results", "performance",
    "benchmark", "benchmarks", "state of the art", "sota", "overview", "introduction",
}

GENERIC_PATTERN = re.compile(
    r"^(this|these|that|those|our|their|its|such|each|every|all|the proposed|a proposed|the novel|a novel|novel|new|current|existing|various|different)\s+(approach|method|technique|manner|framework|model|algorithm|strategy|paradigm|work|paper|study|system|process|solution|tool)s?$",
    re.IGNORECASE,
)


def is_informative_facet(name: str) -> bool:
    """Filter out deictic pronouns, stopwords, numbers, and uninformative generic research labels."""
    if not name or not name.strip():
        return False
    norm = name.strip().lower()
    # Filter single letter noise
    if len(norm) <= 1:
        return False
    # Filter pure numbers or percentages
    if norm.isdigit() or re.match(r"^\d+(\.\d+)?%?$", norm):
        return False
    # Filter exact generic words
    if norm in GENERIC_FACET_TERMS:
        return False
    # Filter regex deictic + noun ("this approach", "each manner", "the proposed system")
    if GENERIC_PATTERN.match(norm):
        return False
    # Filter short phrases consisting solely of determiners and generic words
    if norm.startswith(("this ", "our ", "the proposed ", "a novel ", "each ", "their ", "its ", "the ")):
        words = norm.split()
        if len(words) <= 3 and all(w in GENERIC_FACET_TERMS or w in {"this", "our", "the", "a", "an", "each", "their", "its", "proposed", "novel", "new"} for w in words):
            return False
    return True


def render_facet_pill(facet_type: str, text: str, clickable: bool = False) -> str:
    """Returns HTML for a single typed facet pill."""
    style = FACET_COLORS.get(facet_type.lower(), FACET_COLORS["default"])
    cursor = "pointer" if clickable else "default"
    clean_text = str(text).replace("<", "&lt;").replace(">", "&gt;")
    clean_type = str(facet_type).upper().replace("<", "&lt;").replace(">", "&gt;")
    
    return (
        f'<span style="'
        f'display: inline-flex; align-items: center; margin: 2px 4px 2px 0;'
        f'padding: 2px 8px; border-radius: 9999px; font-size: 0.75rem;'
        f'font-weight: 500; font-family: ui-sans-serif, system-ui, sans-serif;'
        f'color: {style["color"]}; background-color: {style["bg"]};'
        f'border: 1px solid {style["border"]}; cursor: {cursor};'
        f'line-height: 1.25;">'
        f'<strong style="opacity: 0.8; margin-right: 4px; font-size: 0.65rem;">{clean_type}</strong> '
        f'{clean_text}'
        f'</span>'
    )


def render_pills_container(facets: Sequence[Mapping[str, str]]) -> str:
    """Render a flex container containing multiple facet pills, filtering out generic noise."""
    if not facets:
        return ""
    pills = [
        render_facet_pill(f.get("type", "topic"), f.get("value", ""))
        for f in facets
        if f.get("value") and is_informative_facet(f.get("value", ""))
    ]
    return f'<div style="display: flex; flex-wrap: wrap; margin-top: 6px; margin-bottom: 8px;">{"".join(pills)}</div>'
