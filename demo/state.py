"""Global state management and cached resource loaders for Streamlit."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import yaml

import streamlit as st

from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import clean_id
from paperfeed.data.ground_truth import to_uri
from paperfeed.rec.recommender import (
    RecommendConfig,
    RecommenderResources,
    ScoringWeights,
    UserProfile,
    build_user_profile,
)
from paperfeed.user.session import SessionState

logger = get_logger("demo.state")

ROOT_DIR = Path(__file__).resolve().parent.parent


@st.cache_resource(show_spinner="Loading indices and recommender models (one-time initialization)...")
def load_app_resources() -> RecommenderResources:
    """Load FAISS index, BM25s, SQLite DB, Facets, and SPECTER2 encoder once."""
    config_path = ROOT_DIR / "configs" / "default.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    
    # Load all resources with SPECTER2 encoder
    resources = RecommenderResources.load(
        cfg,
        root=ROOT_DIR,
        load_encoder=True,
        use_facets=True,
        encoder_key="specter2",
    )
    return resources


def load_best_recommend_config() -> RecommendConfig:
    """Build RecommendConfig populated with frozen Phase 11 best configuration and Phase 9 simulation parameters."""
    cfg_path = ROOT_DIR / "configs" / "default.yaml"
    base_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    base_rcfg = RecommendConfig.from_config(base_cfg)

    # Check for frozen best.yaml
    best_yaml_path = ROOT_DIR / "configs" / "best.yaml"
    scoring_weights = base_rcfg.weights
    lambda_div = 0.5
    if best_yaml_path.exists():
        best_data = yaml.safe_load(best_yaml_path.read_text(encoding="utf-8"))
        sw = best_data.get("scoring_weights", {})
        scoring_weights = ScoringWeights(
            w_pers=float(sw.get("w_pers", 1.0)),
            w_sess=float(sw.get("w_sess", 1.0)),
            w_cite=float(sw.get("w_cite", 1.0)),
            w_facet=float(sw.get("w_facet", 0.25)),
            w_nov=float(sw.get("w_nov", 0.25)),
            lambda_neg=float(sw.get("lambda_neg", 0.5)),
            w_pop=float(sw.get("w_pop", 0.0)),
            w_ce=float(sw.get("w_ce", 0.0)),
        )
        parts_div = best_data.get("parts", {}).get("diversification", {}).get("choice", {})
        if "lambda_div" in parts_div:
            lambda_div = float(parts_div["lambda_div"])

    # Check for frozen Phase 9 simulation parameters
    track2_config_path = ROOT_DIR / "data" / "artifacts" / "track2_config.json"
    alpha_sess = 0.3
    rho = 0.3
    gamma_pivot = 0.35
    use_gate = True

    if track2_config_path.exists():
        try:
            t2_data = json.loads(track2_config_path.read_text(encoding="utf-8"))
            opt = t2_data.get("optimal_params", {})
            alpha_sess = float(opt.get("alpha_sess", 0.3))
            rho = float(opt.get("rho", 0.3))
            gamma_pivot = float(opt.get("gamma_pivot", 0.35))
        except Exception as exc:
            logger.warning("Could not load track2_config: %s", exc)

    return base_rcfg.with_overrides(
        weights=scoring_weights,
        lambda_div=lambda_div,
        alpha_sess=alpha_sess,
        rho=rho,
        gamma_pivot=gamma_pivot,
        use_gate=use_gate,
        k=10,
        pool_size=100,
        channel_depth=200,
    )


@st.cache_data
def get_featured_authors() -> List[Dict[str, Any]]:
    """Return featured OpenAlex authors with pre-indexed papers in the database."""
    candidates_file = ROOT_DIR / "data" / "candidate_authors.jsonl"
    if not candidates_file.exists():
        return []
    
    # Priority list of rich authors in our database (ordered; Daya Guo first)
    priority_order = ["A5060364305", "A5072548276", "A5081550394", "A5036875405", "A5020661868", "A5073147712"]
    priority_ids = set(priority_order)
    featured = []
    others = []
    
    with open(candidates_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx > 150:
                break
            entry = json.loads(line)
            aid = entry.get("author_id")
            item = {
                "author_id": aid,
                "name": entry.get("author_name", aid),
                "n_papers": len(entry.get("authored_paper_ids_T", [])),
                "authored_ids": entry.get("authored_paper_ids_T", []),
                "topics": entry.get("target_paper_b_topics", [])[:2],
            }
            if aid in priority_ids:
                featured.append(item)
            elif len(item["authored_ids"]) >= 5:
                others.append(item)
                
    # Sort priority authors according to explicit priority_order
    featured.sort(key=lambda x: priority_order.index(x["author_id"]) if x["author_id"] in priority_order else 999)
    return featured + others[:10]


@st.cache_data
def get_popular_seed_topics() -> List[str]:
    """Curated list of standard CS/AI research seed topics for zero-publication mode."""
    return [
        "Natural Language Processing",
        "Large Language Models",
        "Graph Neural Networks",
        "Dense Passage Retrieval",
        "Recommender Systems",
        "Adversarial Robustness in Machine Learning",
        "Computer Vision",
        "Diffusion Models",
        "Reinforcement Learning from Human Feedback",
        "Information Retrieval",
        "Multimodal Representation Learning",
        "Few-Shot Learning",
        "Tabular Deep Learning",
        "Code Intelligence and Synthesis",
        "Scientific Document Understanding",
    ]


def init_session_state() -> None:
    """Initialize all session state defaults if not already present."""
    if "config" not in st.session_state:
        st.session_state["config"] = load_best_recommend_config()

    if "session" not in st.session_state:
        cfg: RecommendConfig = st.session_state["config"]
        st.session_state["session"] = cfg.make_session(dim=768)

    if "user_mode" not in st.session_state:
        st.session_state["user_mode"] = "author"

    if "user_id" not in st.session_state:
        st.session_state["user_id"] = "A5060364305"  # Daya Guo by default

    if "author_name" not in st.session_state:
        st.session_state["author_name"] = "Daya Guo"

    if "authored_ids" not in st.session_state:
        featured = get_featured_authors()
        matching = [a for a in featured if a.get("author_id") == st.session_state.get("user_id")]
        default_ids = matching[0]["authored_ids"] if matching else (featured[0]["authored_ids"] if featured else [])
        st.session_state["authored_ids"] = default_ids

    if "seed_topics" not in st.session_state:
        st.session_state["seed_topics"] = ["Code Intelligence and Synthesis", "Large Language Models"]

    if "profile" not in st.session_state:
        st.session_state["profile"] = None

    if "feed_items" not in st.session_state:
        st.session_state["feed_items"] = []

    if "feed_meta" not in st.session_state:
        st.session_state["feed_meta"] = {}

    if "library" not in st.session_state:
        st.session_state["library"] = []  # list of paper dicts

    if "skipped_ids" not in st.session_state:
        st.session_state["skipped_ids"] = set()

    if "pilot_ratings" not in st.session_state:
        st.session_state["pilot_ratings"] = {}  # paper_id -> {"relevance": 1-5, "clarity": 1-5, "notes": str}

    if "feed_dirty" not in st.session_state:
        st.session_state["feed_dirty"] = True

    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Feed"


def reset_user_session() -> None:
    """Clear session vector, negative interactions, and mark feed dirty."""
    cfg: RecommendConfig = st.session_state.get("config", load_best_recommend_config())
    st.session_state["session"] = cfg.make_session(dim=768)
    st.session_state["skipped_ids"] = set()
    st.session_state["feed_dirty"] = True
