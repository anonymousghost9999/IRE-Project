"""Paper Feed recommendation engine (Phase 5)."""

from paperfeed.rec.candidates import feed_candidates, query_vector, search_candidates
from paperfeed.rec.explain import Explainer, Explanation, ExplanationResult
from paperfeed.rec.features import (
    CandidateFeatures,
    CitationSignal,
    FacetMatch,
    NewMethod,
    UserCitationContext,
    build_citation_context,
    compute_features,
)
from paperfeed.rec.mmr import Selection, mmr_select, select_top_k
from paperfeed.rec.recommender import (
    RecommendConfig,
    RecommendedItem,
    RecommendationResult,
    RecommenderResources,
    UserProfile,
    build_user_profile,
    recommend,
)
from paperfeed.rec.scoring import (
    ScoredCandidate,
    ScoringWeights,
    rank_candidates,
    score_candidates,
)

__all__ = [
    "CandidateFeatures",
    "CitationSignal",
    "Explainer",
    "Explanation",
    "ExplanationResult",
    "FacetMatch",
    "NewMethod",
    "RecommendConfig",
    "RecommendationResult",
    "RecommendedItem",
    "RecommenderResources",
    "ScoredCandidate",
    "ScoringWeights",
    "Selection",
    "UserCitationContext",
    "UserProfile",
    "build_citation_context",
    "build_user_profile",
    "compute_features",
    "feed_candidates",
    "mmr_select",
    "query_vector",
    "rank_candidates",
    "recommend",
    "score_candidates",
    "search_candidates",
    "select_top_k",
]
