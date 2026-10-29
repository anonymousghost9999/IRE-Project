"""The recommender interface (Phase 5.6 / docs/plan.md §5.6 / docs/design.md §8-§11).

``recommend(resources, profile, session, mode)`` is the one deterministic function behind every evaluation run, the
simulator and the UI. Feed mode runs the whole pipeline:

    exclusion set E_u  ->  pseudo-query  ->  candidates (dense on q_t + BM25s, RRF)  ->  features  ->  min-max scaled
    linear score  ->  MMR  ->  grounded explanations

Search mode returns the fused ``adhoc_query`` + BM25s list for a query text with each channel's rank (it is not
personalised and has no explanation). Every ablation is a field of ``RecommendConfig``: each weight, lambda_div, rho, K,
the exclusion switches (diagnostic, zero-publication), the pivot gate and the candidate depths.
Identical inputs give identical outputs: every sort breaks ties by paper_id.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

import numpy as np

from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import clean_id
from paperfeed.data.ground_truth import to_uri
from paperfeed.facets.store import FacetStore
from paperfeed.index.retriever import HybridRetriever, RetrievedCandidate
from paperfeed.index.store import EmbeddingStore
from paperfeed.rec.candidates import (
    DEFAULT_RHO,
    eligible_paper_ids,
    feed_candidates,
    search_candidates,
)
from paperfeed.rec.explain import Explainer, ExplanationResult
from paperfeed.rec.features import UserCitationContext, build_citation_context, compute_features
from paperfeed.rec.mmr import DEFAULT_K, DEFAULT_LAMBDA_DIV, select_top_k
from paperfeed.rec.scoring import ScoringWeights, score_candidates
from paperfeed.user.exclusion import ExclusionSources, load_exclusion_sources
from paperfeed.user.facets import UserFacets, build_user_facets, build_zero_pub_facets
from paperfeed.user.persona import DEFAULT_GAMMA_REC, Persona, build_persona, build_seed_persona, load_authored_papers
from paperfeed.user.query import DEFAULT_N_TERMS, DEFAULT_N_TITLES, build_feed_query
from paperfeed.user.session import SessionState

logger = get_logger("rec.recommender")

MODES = ("feed", "search")


# ---------------------------------------------------------------------------
# Configuration (every ablation switch)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RecommendConfig:
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    lambda_div: float = DEFAULT_LAMBDA_DIV
    k: int = DEFAULT_K
    rho: float = DEFAULT_RHO
    normalize_session: bool = False
    diagnostic: bool = False  # exclusion: drop Citations(u, <= T) (re-citation diagnostic run)
    zero_pub: bool = False  # exclusion: E_u = empty set (profiles with no authored papers are always treated this way)
    n_terms: int = DEFAULT_N_TERMS  # pseudo-query length
    n_titles: int = DEFAULT_N_TITLES
    channel_depth: Optional[int] = None  # M_channel; None keeps the retriever's value
    pool_size: Optional[int] = None  # M; None keeps the retriever's value
    alpha_sess: float = 0.5  # session parameters, used by ``make_session``
    use_gate: bool = False
    gamma_pivot: float = 0.35
    gamma_rec: float = DEFAULT_GAMMA_REC  # persona recency decay, used by ``build_user_profile``
    explain: bool = True

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "RecommendConfig":
        """Defaults from configs/default.yaml."""
        um, r, sw = cfg.get("user_model", {}), cfg.get("retrieval", {}), cfg.get("scoring_weights", {})
        return cls(
            weights=ScoringWeights.from_config(cfg),
            lambda_div=float(sw.get("lambda_div", DEFAULT_LAMBDA_DIV)),
            k=int(r.get("feed_size", DEFAULT_K)),
            rho=float(um.get("rho_persona_weight", DEFAULT_RHO)),
            n_terms=int(um.get("pseudo_query_terms", DEFAULT_N_TERMS)),
            n_titles=int(um.get("pseudo_query_titles", DEFAULT_N_TITLES)),
            channel_depth=int(r["channel_depth"]) if "channel_depth" in r else None,
            pool_size=int(r["candidate_pool_size"]) if "candidate_pool_size" in r else None,
            alpha_sess=float(um.get("alpha_session", 0.5)),
            use_gate=bool(um.get("use_pivot_gate", False)),
            gamma_pivot=float(um.get("gamma_pivot", 0.35)),
            gamma_rec=float(um.get("gamma_recency", DEFAULT_GAMMA_REC)),
        )

    def with_overrides(self, **changes: Any) -> "RecommendConfig":
        """A copy with some switches changed; ``weights=`` may also be given as a dict of ScoringWeights fields."""
        weights = changes.pop("weights", None)
        if isinstance(weights, Mapping):
            weights = self.weights.with_overrides(**weights)
        new = replace(self, **changes)
        return replace(new, weights=weights) if weights is not None else new

    def make_session(self, dim: int = 768) -> SessionState:
        return SessionState(alpha_sess=self.alpha_sess, use_gate=self.use_gate, gamma_pivot=self.gamma_pivot, dim=dim)


# ---------------------------------------------------------------------------
# Shared resources and per-user profile
# ---------------------------------------------------------------------------

@dataclass
class RecommenderResources:
    """Everything shared across users, loaded once."""

    db: Any  # DatabaseManager
    store: EmbeddingStore
    retriever: HybridRetriever
    cutoff_year: int
    eligible: Set[str]  # papers that may be recommended: in_candidates and year <= T
    facet_store: Optional[FacetStore] = None  # None until facets are extracted and canonicalized
    encoder: Any = None  # SPECTER2Encoder; needed for title-only authored papers, seed topics and search mode

    @classmethod
    def load(
        cls,
        cfg: Mapping[str, Any],
        root: Path,
        load_encoder: bool = False,
        use_facets: bool = True,
    ) -> "RecommenderResources":
        """Load the built indexes, embeddings and database from ``paths.processed_dir``."""
        from paperfeed.data.db import DatabaseManager
        from paperfeed.index.dense import DenseIndex
        from paperfeed.index.lexical import LexicalIndex

        processed = Path(root) / cfg["paths"]["processed_dir"]
        cutoff = int(cfg["project"]["cutoff_year"])
        db = DatabaseManager(Path(root) / cfg["paths"]["sqlite_db"], cutoff_year=cutoff)
        store = EmbeddingStore.load(processed / "embeddings.npy", processed / "embedding_ids.json", processed / "paper_ids.json")
        rcfg = RecommendConfig.from_config(cfg)
        retriever = HybridRetriever(
            DenseIndex.load(processed / "faiss.index", processed / "paper_ids.json"),
            LexicalIndex.load(processed / "bm25s"),
            channel_depth=rcfg.channel_depth or 200,
            rrf_k=int(cfg.get("retrieval", {}).get("rrf_k", 60)),
            candidate_pool_size=rcfg.pool_size or 100,
        )
        facet_store = None
        if use_facets:
            with db.get_connection() as conn:
                canonicalized = conn.execute("SELECT 1 FROM facet_canon LIMIT 1").fetchone() is not None
            facet_store = FacetStore(db.db_path) if canonicalized else None
            if facet_store is None:
                logger.warning("facet_canon is empty: facet features, the new-method term and facet explanations are off.")
        encoder = None
        if load_encoder:
            from paperfeed.index.embeddings import SPECTER2Encoder

            m = cfg["models"]
            encoder = SPECTER2Encoder(m["dense_encoder"], m["dense_adapter"], m["query_adapter"])
        return cls(db, store, retriever, cutoff, eligible_paper_ids(db, cutoff), facet_store, encoder)


@dataclass
class UserProfile:
    """The static part of a user model: persona, F_u and the exclusion / citation sources. Session state is separate."""

    user_id: str
    persona: Persona
    user_facets: UserFacets
    exclusion_sources: ExclusionSources
    citation_ctx: UserCitationContext
    authored_papers: List[Dict[str, Any]] = field(default_factory=list)  # dicts with paper_id, title, abstract, year
    zero_pub: bool = False  # True when the persona comes from seed topics


def build_user_profile(
    resources: RecommenderResources,
    user_id: str,
    authored_ids: Sequence[str] = (),
    seed_topics: Sequence[str] = (),
    config: Optional[RecommendConfig] = None,
) -> UserProfile:
    """Profile from authored papers (<= T), or from seed topics when there are none (zero-publication user)."""
    cfg = config or RecommendConfig()
    r = resources
    if authored_ids:
        papers = load_authored_papers(r.db, authored_ids, r.cutoff_year)
        persona = build_persona(papers, r.store, r.encoder, r.cutoff_year, cfg.gamma_rec)
        paper_ids = [p["paper_id"] for p in papers]
        facets = build_user_facets(paper_ids, r.facet_store) if r.facet_store is not None else UserFacets(paper_ids=paper_ids)
        return UserProfile(
            user_id=user_id, persona=persona, user_facets=facets,
            exclusion_sources=load_exclusion_sources(r.db, paper_ids),
            citation_ctx=build_citation_context(r.db, paper_ids, r.cutoff_year),
            authored_papers=papers,
        )
    if r.encoder is None:
        raise ValueError("A zero-publication user needs the encoder to embed their seed topics.")
    persona = build_seed_persona(seed_topics, r.encoder)
    facets = (
        build_zero_pub_facets(persona.vector, r.retriever.dense_index, r.facet_store)
        if r.facet_store is not None else UserFacets(source="nearest_papers")
    )
    return UserProfile(
        user_id=user_id, persona=persona, user_facets=facets, exclusion_sources=ExclusionSources(),
        citation_ctx=build_citation_context(r.db, [], r.cutoff_year), zero_pub=True,
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class RecommendedItem:
    rank: int
    paper_id: str
    title: str
    year: Optional[int]
    score: Optional[float]  # linear score (feed mode); None in search mode
    mmr_value: Optional[float]
    contributions: Dict[str, float]  # w_k * mm_k per score term (empty in search mode)
    rrf_score: float
    dense_rank: Optional[int]
    lexical_rank: Optional[int]
    explanation: Optional[ExplanationResult] = None

    def to_dict(self) -> Dict[str, Any]:
        expl = self.explanation
        return {
            "rank": self.rank, "paper_id": self.paper_id, "title": self.title, "year": self.year,
            "score": self.score, "mmr_value": self.mmr_value, "contributions": self.contributions,
            "rrf_score": self.rrf_score, "dense_rank": self.dense_rank, "lexical_rank": self.lexical_rank,
            "dominant_feature": expl.dominant_feature if expl else None,
            "explanation": expl.explanation.text if expl and expl.explanation else None,
            "explanation_failure": expl.failure if expl else None,
        }


@dataclass
class RecommendationResult:
    user_id: str
    mode: str
    items: List[RecommendedItem]
    n_candidates: int  # size of the fused candidate pool before scoring
    candidate_ids: List[str]  # the pool in fused order (Recall@100 and the V3 ranker need it)
    session_active: bool
    pseudo_query: str
    n_excluded: int
    config: RecommendConfig

    @property
    def paper_ids(self) -> List[str]:
        return [i.paper_id for i in self.items]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id, "mode": self.mode, "n_candidates": self.n_candidates, "session_active": self.session_active,
            "pseudo_query": self.pseudo_query, "n_excluded": self.n_excluded, "items": [i.to_dict() for i in self.items],
        }


# ---------------------------------------------------------------------------
# recommend
# ---------------------------------------------------------------------------

def _item_meta(db, paper_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    rows = db.get_papers_batch([to_uri(p) for p in paper_ids]) if paper_ids else []
    return {clean_id(r["paper_id"]): r for r in rows}


def _check_output(paper_ids: Sequence[str], resources: RecommenderResources, excluded: Set[str]) -> None:
    """Runtime guard: nothing that is excluded, from after T or outside the candidate set may be recommended."""
    bad_excluded = [p for p in paper_ids if p in excluded]
    bad_ineligible = [p for p in paper_ids if p not in resources.eligible]
    if bad_excluded or bad_ineligible:
        raise RuntimeError(
            f"Recommender returned papers it must not: excluded={bad_excluded[:3]} ineligible(year > T or not a candidate)={bad_ineligible[:3]}"
        )
    if len(set(paper_ids)) != len(paper_ids):
        raise RuntimeError("Recommender returned duplicate papers.")


def recommend(
    resources: RecommenderResources,
    profile: UserProfile,
    session: Optional[SessionState] = None,
    mode: str = "feed",
    query: Optional[str] = None,
    config: Optional[RecommendConfig] = None,
) -> RecommendationResult:
    """Top-K recommendations for one user. ``session=None`` means an empty session (Track 1)."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'; expected one of {MODES}.")
    cfg = config or RecommendConfig()
    r = resources
    session = session if session is not None else SessionState(dim=len(profile.persona.vector))
    session_active = not session.is_empty

    excluded = profile.exclusion_sources.build(
        session_consumed=session.consumed, diagnostic=cfg.diagnostic, zero_pub=cfg.zero_pub or profile.zero_pub
    )

    # -- search mode: the fused list, not personalised ----------------------------------------------------------
    if mode == "search":
        if r.encoder is None:
            raise ValueError("Search mode needs the encoder to embed the query.")
        cands = search_candidates(
            r.retriever, r.encoder, query or "", excluded, r.eligible, cfg.channel_depth, cfg.pool_size
        )
        top = cands[: cfg.k]
        meta = _item_meta(r.db, [c.paper_id for c in top])
        items = [_search_item(i, c, meta) for i, c in enumerate(top, start=1)]
        _check_output([i.paper_id for i in items], r, excluded)
        return RecommendationResult(profile.user_id, mode, items, len(cands), [c.paper_id for c in cands], session_active,
                                    (query or "").strip(), len(excluded), cfg)

    # -- feed mode ----------------------------------------------------------------------------------------------
    pseudo_query = build_feed_query(profile.user_facets, session, r.db, profile.authored_papers, cfg.n_terms, cfg.n_titles)
    session_vec = session.vector if session_active else None
    cands = feed_candidates(
        r.retriever, profile.persona.vector, pseudo_query, session_vec, excluded, r.eligible,
        cfg.rho, cfg.normalize_session, cfg.channel_depth, cfg.pool_size,
    )
    by_id: Dict[str, RetrievedCandidate] = {c.paper_id: c for c in cands}
    features = compute_features(
        [c.paper_id for c in cands], r.store, profile.persona.vector, profile.user_facets, profile.citation_ctx, r.db,
        r.cutoff_year, session_vec, session.negatives, r.facet_store,
    )
    scored = score_candidates(features, cfg.weights)
    picks = select_top_k(scored, r.store, cfg.k, cfg.lambda_div)

    explanations: Dict[str, ExplanationResult] = {}
    if cfg.explain and picks:
        explainer = Explainer(profile.user_facets, r.db, r.cutoff_year, profile.citation_ctx, r.store, r.facet_store, session)
        explanations = {e.paper_id: e for e in explainer.explain_all([p.scored for p in picks])}

    meta = _item_meta(r.db, [p.paper_id for p in picks])
    items = [
        RecommendedItem(
            rank=p.rank, paper_id=p.paper_id, title=(meta.get(p.paper_id) or {}).get("title", ""),
            year=(meta.get(p.paper_id) or {}).get("year"), score=p.scored.score, mmr_value=p.mmr_value,
            contributions=dict(p.scored.contributions), rrf_score=by_id[p.paper_id].rrf_score,
            dense_rank=by_id[p.paper_id].dense_rank, lexical_rank=by_id[p.paper_id].lexical_rank,
            explanation=explanations.get(p.paper_id),
        )
        for p in picks
    ]
    _check_output([i.paper_id for i in items], r, excluded)
    return RecommendationResult(profile.user_id, mode, items, len(cands), [c.paper_id for c in cands], session_active,
                                pseudo_query, len(excluded), cfg)


def _search_item(rank: int, c: RetrievedCandidate, meta: Mapping[str, Mapping[str, Any]]) -> RecommendedItem:
    row = meta.get(c.paper_id) or {}
    return RecommendedItem(
        rank=rank, paper_id=c.paper_id, title=row.get("title", ""), year=row.get("year"), score=None, mmr_value=None,
        contributions={}, rrf_score=c.rrf_score, dense_rank=c.dense_rank, lexical_rank=c.lexical_rank,
    )
