"""Tests for Phase 12 Streamlit Demo UI bridge and components."""

import pytest
from demo.components.explanation import render_explanation_badge, render_explanation_block
from demo.components.facet_pills import render_facet_pill, render_pills_container
from demo.recommend_bridge import _format_authors
from demo.state import get_featured_authors, get_popular_seed_topics, load_best_recommend_config


def test_facet_pills_rendering():
    pill = render_facet_pill("task", "question answering")
    assert "question answering" in pill
    assert "TASK" in pill
    assert "#3B82F6" in pill  # Blue for task

    container = render_pills_container([
        {"type": "task", "value": "QA"},
        {"type": "method", "value": "BERT"},
    ])
    assert "QA" in container
    assert "BERT" in container


def test_explanation_rendering():
    badge = render_explanation_badge("facet")
    assert "Facet Match" in badge
    assert "💡" in badge

    block = render_explanation_block(
        text="Recommended because you work on **dual encoders**",
        dominant_feature="facet",
        evidence={"evidence_sentence": "We train a dual encoder for QA.", "span": "dual encoder"},
    )
    assert "dual encoders" in block
    assert "for QA." in block
    assert "dual encoder" in block


def test_format_authors():
    assert _format_authors(None) == "Unknown Authors"
    assert _format_authors("Single Author") == "Single Author"
    assert _format_authors(["Author A", "Author B"]) == "Author A, Author B"


def test_demo_state_helpers():
    seeds = get_popular_seed_topics()
    assert len(seeds) >= 10
    assert "Natural Language Processing" in seeds

    authors = get_featured_authors()
    assert len(authors) > 0
    assert any(a["author_id"] == "A5060364305" for a in authors)

    cfg = load_best_recommend_config()
    assert cfg.alpha_sess == 0.3
    assert cfg.rho == 0.3
    assert cfg.use_gate is True


def test_generate_bibtex():
    from demo.pages.library import generate_bibtex

    papers = [
        {
            "paper_id": "W12345",
            "title": "Dense Passage Retrieval",
            "authors": "Karpukhin et al.",
            "year": 2020,
            "venue": "EMNLP",
            "url": "https://openalex.org/W12345",
        }
    ]
    bib = generate_bibtex(papers)
    assert "@article{W12345," in bib
    assert "Dense Passage Retrieval" in bib
    assert "Karpukhin et al." in bib


def test_theme_and_clean_html():
    from demo.theme import clean_html, get_theme

    theme = get_theme()
    assert "bg" in theme
    assert "card_bg" in theme
    assert "accent" in theme

    indented_html = """
        <div style="margin: 0;">
            <span>Test Text</span>
        </div>
    """
    cleaned = clean_html(indented_html)
    assert cleaned.startswith('<div style="margin: 0;">')
    assert not cleaned.startswith(" ")


def test_is_informative_facet():
    from demo.components.facet_pills import is_informative_facet

    # Uninformative / generic facets that should be filtered out
    assert not is_informative_facet("This approach")
    assert not is_informative_facet("Each manner")
    assert not is_informative_facet("our method")
    assert not is_informative_facet("The proposed system")
    assert not is_informative_facet("execution")
    assert not is_informative_facet("baseline experiments")
    assert not is_informative_facet("47")

    # Informative scientific facets that should be preserved
    assert is_informative_facet("architectural design")
    assert is_informative_facet("MBSE")
    assert is_informative_facet("declarative modeling paradigms")
    assert is_informative_facet("BERT")
    assert is_informative_facet("dual encoder")
    assert is_informative_facet("6G")
    assert is_informative_facet("graph neural networks")
