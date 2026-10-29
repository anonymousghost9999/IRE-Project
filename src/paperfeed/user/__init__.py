"""Paper Feed user modeling subpackage (Phase 4 / docs/plan.md §4 / docs/design.md §7).

Provides:
- Persona: persona vector v_pers_u with recency decay
- UserFacets: canonical facet profile F_u
- ExclusionSources: static and dynamic exclusion set E_u
- SessionState: dynamic session state v_sess_t and negative set S^-_u
- Pseudo-query generation for lexical retrieval
"""

from paperfeed.user.exclusion import ExclusionSources, build_exclusion_sources, load_exclusion_sources
from paperfeed.user.facets import UserFacets, build_user_facets, build_zero_pub_facets
from paperfeed.user.persona import Persona, build_persona, build_seed_persona, load_authored_papers, persona_vector
from paperfeed.user.query import build_feed_query, build_pseudo_query, most_recent_titles
from paperfeed.user.session import SessionState

__all__ = [
    "Persona",
    "UserFacets",
    "ExclusionSources",
    "SessionState",
    "build_persona",
    "build_seed_persona",
    "build_user_facets",
    "build_zero_pub_facets",
    "build_exclusion_sources",
    "load_exclusion_sources",
    "load_authored_papers",
    "persona_vector",
    "build_feed_query",
    "build_pseudo_query",
    "most_recent_titles",
]
