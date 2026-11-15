"""Track 2 episode simulator and evaluation arms (Plan 9c / docs/design.md §13.2 points 4-5).

Implements the multi-turn session simulation protocol:
1. 5 Topic A feed turns (turns 0-4).
2. Shift turn: search query for Topic B, agent acts on results, session updates.
3. Up to 10 Topic B feed turns: measures Pivot Latency k_pivot (turns until Topic B >= 50% of Top-10, censored at 10).
4. Re-engagement turn: search query for Topic A, agent acts on results, session updates.
5. 3 Re-engagement feed turns: measures Retention@k (k=1..3: share of Topic A in Top-10).

Arms:
- persona_only: rho=1, w_sess=0, lambda_neg=0, no session input anywhere.
- session_only: rho=0, w_pers=0, skip penalty on, EMA session.
- static_blend: 50/50 blend (rho=0.5, w_pers=w_sess), un-decayed session mean, skip penalty on.
- hybrid: EMA session + persona + negative penalty (pilot-tuned alpha_sess, rho, w_sess, lambda_neg).
- hybrid_gated: hybrid with use_gate=True and gamma_pivot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import clean_id
from paperfeed.data.ground_truth import to_uri
from paperfeed.rec.recommender import (
    RecommendConfig,
    RecommendationResult,
    RecommenderResources,
    UserProfile,
    build_user_profile,
    recommend,
)
from paperfeed.rec.scoring import ScoringWeights
from paperfeed.sim.llm import PaperForRating, Rater
from paperfeed.sim.persona import SimPersona
from paperfeed.user.session import SessionState

logger = get_logger("sim.episode")

ARMS = ("persona_only", "session_only", "static_blend", "hybrid", "hybrid_gated")
N_FEED_A_TURNS = 5
MAX_FEED_B_TURNS = 10
N_REENGAGE_TURNS = 3


class StaticBlendSessionState(SessionState):
    """Session state for Static Blend arm: un-decayed uniform mean of positive embeddings."""

    def __init__(self, dim: int = 768) -> None:
        super().__init__(alpha_sess=0.0, use_gate=False, dim=dim)
        self._positive_embeddings: List[np.ndarray] = []

    def _positive(self, paper_id: str, embedding: np.ndarray) -> None:
        emb = self._check_embedding(embedding)
        clean = clean_id(paper_id)
        self._positive_embeddings.append(emb)
        self._vector = np.mean(self._positive_embeddings, axis=0).astype(np.float32)
        self._positives.append(clean)


class PersonaOnlySessionState(SessionState):
    """Session state for Persona-Only arm: session stays empty regardless of feedback."""

    def _positive(self, paper_id: str, embedding: np.ndarray) -> None:
        pass

    def click(self, paper_id: str, embedding: np.ndarray) -> None:
        pass

    def save(self, paper_id: str, embedding: np.ndarray) -> None:
        pass

    def skip(self, paper_id: str) -> None:
        pass

    def apply(self, action: str, paper_id: str, embedding: Optional[np.ndarray] = None) -> None:
        pass  # Never updates session vector, negatives or consumed items


@dataclass
class EpisodeResult:
    persona_id: str
    arm: str
    seed: int
    k_pivot: int  # 1..10 (10 = censored)
    pivot_reached: bool
    retention_at_1: float
    retention_at_2: float
    retention_at_3: float
    turn_metrics: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_paper_embedding(store: Any, paper_id: str, dim: int = 768) -> np.ndarray:
    """Retrieve embedding vector for paper_id from EmbeddingStore."""
    clean = clean_id(paper_id)
    idx = store.paper_id_index.get(clean)
    if idx is not None:
        return store.vectors[idx]
    uri = to_uri(clean)
    idx_uri = store.paper_id_index.get(uri)
    if idx_uri is not None:
        return store.vectors[idx_uri]
    return np.zeros(dim, dtype=np.float32)


def load_subfield_lookup(db_path: str | Path | Any) -> Dict[str, str]:
    """Load a mapping from paper_id (bare and URI) to subfield_id."""
    lookup: Dict[str, str] = {}
    actual_path = getattr(db_path, "db_path", db_path)
    if isinstance(actual_path, (str, Path)):
        conn = sqlite3.connect(str(actual_path))
        close_at_end = True
    else:
        conn = actual_path
        close_at_end = False

    try:
        cur = conn.cursor()
        for row in cur.execute("SELECT paper_id, subfield_id FROM papers WHERE subfield_id IS NOT NULL"):
            raw_id = row[0]
            sf = row[1]
            bare = clean_id(raw_id)
            lookup[bare] = sf
            lookup[to_uri(bare)] = sf
    finally:
        if close_at_end:
            conn.close()
    return lookup


def build_arm_config_and_session(
    arm: str,
    base_config: RecommendConfig,
    base_weights: ScoringWeights,
    dim: int = 768,
    tuned_params: Optional[Mapping[str, Any]] = None,
) -> Tuple[RecommendConfig, SessionState]:
    """Construct the arm-specific RecommendConfig and SessionState."""
    params = dict(tuned_params or {})
    alpha_sess = float(params.get("alpha_sess", 0.5))
    rho = float(params.get("rho", 0.3))
    w_sess = float(params.get("w_sess", 1.0))
    lambda_neg = float(params.get("lambda_neg", 0.5))
    gamma_pivot = float(params.get("gamma_pivot", 0.35))

    if arm == "persona_only":
        cfg = RecommendConfig(
            weights=base_weights.with_overrides(w_pers=1.0, w_sess=0.0, lambda_neg=0.0),
            lambda_div=base_config.lambda_div,
            k=base_config.k,
            rho=1.0,
            n_titles=base_config.n_titles,
            n_terms=base_config.n_terms,
            use_gate=False,
            adaptive_graph=base_config.adaptive_graph,
            explain=False,
        )
        session = PersonaOnlySessionState(dim=dim)

    elif arm == "session_only":
        cfg = RecommendConfig(
            weights=base_weights.with_overrides(w_pers=0.0, w_sess=w_sess, lambda_neg=lambda_neg),
            lambda_div=base_config.lambda_div,
            k=base_config.k,
            rho=0.0,
            alpha_sess=alpha_sess,
            n_titles=base_config.n_titles,
            n_terms=base_config.n_terms,
            use_gate=False,
            adaptive_graph=base_config.adaptive_graph,
            explain=False,
        )
        session = SessionState(alpha_sess=alpha_sess, use_gate=False, dim=dim)

    elif arm == "static_blend":
        cfg = RecommendConfig(
            weights=base_weights.with_overrides(w_pers=1.0, w_sess=1.0, lambda_neg=lambda_neg),
            lambda_div=base_config.lambda_div,
            k=base_config.k,
            rho=0.5,
            n_titles=base_config.n_titles,
            n_terms=base_config.n_terms,
            use_gate=False,
            adaptive_graph=base_config.adaptive_graph,
            explain=False,
        )
        session = StaticBlendSessionState(dim=dim)

    elif arm == "hybrid":
        cfg = RecommendConfig(
            weights=base_weights.with_overrides(w_pers=1.0, w_sess=w_sess, lambda_neg=lambda_neg),
            lambda_div=base_config.lambda_div,
            k=base_config.k,
            rho=rho,
            alpha_sess=alpha_sess,
            n_titles=base_config.n_titles,
            n_terms=base_config.n_terms,
            use_gate=False,
            adaptive_graph=base_config.adaptive_graph,
            explain=False,
        )
        session = SessionState(alpha_sess=alpha_sess, use_gate=False, dim=dim)

    elif arm == "hybrid_gated":
        cfg = RecommendConfig(
            weights=base_weights.with_overrides(w_pers=1.0, w_sess=w_sess, lambda_neg=lambda_neg),
            lambda_div=base_config.lambda_div,
            k=base_config.k,
            rho=rho,
            alpha_sess=alpha_sess,
            gamma_pivot=gamma_pivot,
            use_gate=True,
            n_titles=base_config.n_titles,
            n_terms=base_config.n_terms,
            adaptive_graph=base_config.adaptive_graph,
            explain=False,
        )
        session = SessionState(alpha_sess=alpha_sess, use_gate=True, gamma_pivot=gamma_pivot, dim=dim)

    else:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")

    return cfg, session


def fetch_papers_for_rating(
    resources: RecommenderResources,
    paper_ids: Sequence[str],
) -> List[PaperForRating]:
    """Retrieve metadata (title, abstract) for papers to be rated by LLM."""
    clean_ids = [clean_id(pid) for pid in paper_ids]
    uris = [to_uri(pid) for pid in clean_ids]
    raw_records = resources.db.get_papers_batch(uris)
    by_id = {clean_id(r["paper_id"]): r for r in raw_records}
    papers: List[PaperForRating] = []
    for cid in clean_ids:
        r = by_id.get(cid, {})
        papers.append(
            PaperForRating(
                paper_id=cid,
                title=r.get("title") or cid,
                abstract=r.get("abstract") or "",
            )
        )
    return papers


def compute_topic_share(
    paper_ids: Sequence[str],
    target_subfield_id: str,
    subfield_lookup: Mapping[str, str],
) -> float:
    """Fraction of paper_ids whose subfield matches target_subfield_id."""
    if not paper_ids:
        return 0.0
    matches = sum(1 for pid in paper_ids if subfield_lookup.get(clean_id(pid)) == target_subfield_id)
    return float(matches / len(paper_ids))


def run_episode(
    persona: SimPersona,
    arm: str,
    seed: int,
    resources: RecommenderResources,
    rater: Rater,
    subfield_lookup: Mapping[str, str],
    base_config: Optional[RecommendConfig] = None,
    base_weights: Optional[ScoringWeights] = None,
    tuned_params: Optional[Mapping[str, Any]] = None,
    rec_lock: Optional[Any] = None,
) -> EpisodeResult:
    """Run one simulation episode for a persona under a specific arm and seed."""
    def _do_recommend(*a, **kw):
        if rec_lock is not None:
            with rec_lock:
                return recommend(*a, **kw)
        return recommend(*a, **kw)

    def _do_build_profile(*a, **kw):
        if rec_lock is not None:
            with rec_lock:
                return build_user_profile(*a, **kw)
        return build_user_profile(*a, **kw)

    bcfg = base_config or RecommendConfig()
    bweights = base_weights or bcfg.weights
    dim = resources.store.vectors.shape[1] if len(resources.store.vectors) > 0 else 768

    arm_cfg, session = build_arm_config_and_session(
        arm=arm,
        base_config=bcfg,
        base_weights=bweights,
        dim=dim,
        tuned_params=tuned_params,
    )

    profile = _do_build_profile(
        resources=resources,
        user_id=persona.author_id,
        authored_ids=list(persona.authored_paper_ids),
        config=arm_cfg,
    )

    turn_metrics: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # 1. 5 Topic A Feed Turns (turns 0-4)
    # -------------------------------------------------------------------------
    goal_a = f"Find recent and relevant research in {persona.topic_a_name}."
    for turn in range(N_FEED_A_TURNS):
        rec_res = _do_recommend(resources, profile, session, mode="feed", config=arm_cfg)
        top_ids = [item.paper_id for item in rec_res.items]
        share_a = compute_topic_share(top_ids, persona.topic_a_id, subfield_lookup)
        share_b = compute_topic_share(top_ids, persona.topic_b_id, subfield_lookup)

        turn_metrics.append({
            "stage": "feed_a",
            "turn": turn,
            "share_topic_a": share_a,
            "share_topic_b": share_b,
            "top_ids": top_ids,
        })

        # LLM ratings
        rating_papers = fetch_papers_for_rating(resources, top_ids)
        ratings = rater.rate(
            persona_id=persona.author_id,
            persona_text=persona.persona_text,
            phase="topic_a",
            goal_text=goal_a,
            papers=rating_papers,
            seed=seed,
        )

        # Agent clicks/skips
        for pid in top_ids:
            r = ratings.get(clean_id(pid), 3)
            if r == 5:
                emb = get_paper_embedding(resources.store, pid, dim)
                session.apply("save", pid, emb)
            elif r == 4:
                emb = get_paper_embedding(resources.store, pid, dim)
                session.apply("click", pid, emb)
            elif r <= 2:
                session.apply("skip", pid)

    # -------------------------------------------------------------------------
    # 2. Shift Turn (Topic B search)
    # -------------------------------------------------------------------------
    goal_b = f"Explore new research in {persona.topic_b_name}."
    search_b_res = _do_recommend(
        resources, profile, session, mode="search", query=persona.topic_b_query, config=arm_cfg
    )
    search_b_ids = [item.paper_id for item in search_b_res.items]
    rating_papers_b = fetch_papers_for_rating(resources, search_b_ids)
    ratings_search_b = rater.rate(
        persona_id=persona.author_id,
        persona_text=persona.persona_text,
        phase="topic_b_search",
        goal_text=goal_b,
        papers=rating_papers_b,
        seed=seed,
    )

    # Click policy for Topic B search: clicks >= 4; if none, highest-rated Topic B paper
    clicked_b = False
    for pid in search_b_ids:
        r = ratings_search_b.get(clean_id(pid), 3)
        if r >= 4:
            emb = get_paper_embedding(resources.store, pid, dim)
            session.apply("save" if r == 5 else "click", pid, emb)
            clicked_b = True

    if not clicked_b and search_b_ids:
        # Fallback to highest-rated search result (without oracle subfield metadata lookup)
        best_pid = max(search_b_ids, key=lambda pid: ratings_search_b.get(clean_id(pid), 0))
        emb = get_paper_embedding(resources.store, best_pid, dim)
        session.apply("click", best_pid, emb)

    # -------------------------------------------------------------------------
    # 3. Up to 10 Topic B Feed Turns
    # -------------------------------------------------------------------------
    k_pivot = MAX_FEED_B_TURNS
    pivot_reached = False

    for b_turn in range(1, MAX_FEED_B_TURNS + 1):
        rec_b = _do_recommend(resources, profile, session, mode="feed", config=arm_cfg)
        top_ids_b = [item.paper_id for item in rec_b.items]
        share_a = compute_topic_share(top_ids_b, persona.topic_a_id, subfield_lookup)
        share_b = compute_topic_share(top_ids_b, persona.topic_b_id, subfield_lookup)

        turn_metrics.append({
            "stage": "feed_b",
            "b_turn": b_turn,
            "share_topic_a": share_a,
            "share_topic_b": share_b,
            "top_ids": top_ids_b,
        })

        if not pivot_reached and share_b >= 0.5:
            k_pivot = b_turn
            pivot_reached = True

        rating_papers_feed_b = fetch_papers_for_rating(resources, top_ids_b)
        ratings_feed_b = rater.rate(
            persona_id=persona.author_id,
            persona_text=persona.persona_text,
            phase="topic_b",
            goal_text=goal_b,
            papers=rating_papers_feed_b,
            seed=seed,
        )

        for pid in top_ids_b:
            r = ratings_feed_b.get(clean_id(pid), 3)
            if r == 5:
                emb = get_paper_embedding(resources.store, pid, dim)
                session.apply("save", pid, emb)
            elif r == 4:
                emb = get_paper_embedding(resources.store, pid, dim)
                session.apply("click", pid, emb)
            elif r <= 2:
                session.apply("skip", pid)

    # -------------------------------------------------------------------------
    # 4. Re-engagement Turn (Topic A search)
    # -------------------------------------------------------------------------
    goal_reengage = f"Return to my core research in {persona.topic_a_name}."
    search_a_res = _do_recommend(
        resources, profile, session, mode="search", query=persona.topic_a_query, config=arm_cfg
    )
    search_a_ids = [item.paper_id for item in search_a_res.items]
    rating_papers_reengage = fetch_papers_for_rating(resources, search_a_ids)
    ratings_search_a = rater.rate(
        persona_id=persona.author_id,
        persona_text=persona.persona_text,
        phase="topic_a_reengage_search",
        goal_text=goal_reengage,
        papers=rating_papers_reengage,
        seed=seed,
    )

    clicked_a = False
    for pid in search_a_ids:
        r = ratings_search_a.get(clean_id(pid), 3)
        if r >= 4:
            emb = get_paper_embedding(resources.store, pid, dim)
            session.apply("save" if r == 5 else "click", pid, emb)
            clicked_a = True

    if not clicked_a and search_a_ids:
        topic_a_cands = [
            pid for pid in search_a_ids
            if subfield_lookup.get(clean_id(pid)) == persona.topic_a_id
        ]
        pool = topic_a_cands if topic_a_cands else search_a_ids
        best_pid = max(pool, key=lambda pid: ratings_search_a.get(clean_id(pid), 0))
        emb = get_paper_embedding(resources.store, best_pid, dim)
        session.apply("click", best_pid, emb)

    # -------------------------------------------------------------------------
    # 5. 3 Re-engagement Feed Turns (Retention@k)
    # -------------------------------------------------------------------------
    retentions: List[float] = []
    for r_turn in range(1, N_REENGAGE_TURNS + 1):
        rec_r = _do_recommend(resources, profile, session, mode="feed", config=arm_cfg)
        top_ids_r = [item.paper_id for item in rec_r.items]
        share_a = compute_topic_share(top_ids_r, persona.topic_a_id, subfield_lookup)
        share_b = compute_topic_share(top_ids_r, persona.topic_b_id, subfield_lookup)
        retentions.append(share_a)

        turn_metrics.append({
            "stage": "feed_reengage",
            "reengage_turn": r_turn,
            "share_topic_a": share_a,
            "share_topic_b": share_b,
            "top_ids": top_ids_r,
        })

        rating_papers_r = fetch_papers_for_rating(resources, top_ids_r)
        ratings_r = rater.rate(
            persona_id=persona.author_id,
            persona_text=persona.persona_text,
            phase="topic_a_reengage",
            goal_text=goal_reengage,
            papers=rating_papers_r,
            seed=seed,
        )

        for pid in top_ids_r:
            r = ratings_r.get(clean_id(pid), 3)
            if r == 5:
                emb = get_paper_embedding(resources.store, pid, dim)
                session.apply("save", pid, emb)
            elif r == 4:
                emb = get_paper_embedding(resources.store, pid, dim)
                session.apply("click", pid, emb)
            elif r <= 2:
                session.apply("skip", pid)

    return EpisodeResult(
        persona_id=persona.author_id,
        arm=arm,
        seed=seed,
        k_pivot=k_pivot,
        pivot_reached=pivot_reached,
        retention_at_1=retentions[0] if len(retentions) >= 1 else 0.0,
        retention_at_2=retentions[1] if len(retentions) >= 2 else 0.0,
        retention_at_3=retentions[2] if len(retentions) >= 3 else 0.0,
        turn_metrics=turn_metrics,
    )
