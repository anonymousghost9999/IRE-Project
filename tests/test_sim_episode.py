"""Unit tests for Track 2 persona extraction and episode simulation (Plan 9b, 9c)."""

import json
from pathlib import Path
import numpy as np
import pytest

from paperfeed.rec.recommender import RecommendConfig
from paperfeed.rec.scoring import ScoringWeights
from paperfeed.sim.episode import (
    ARMS,
    PersonaOnlySessionState,
    StaticBlendSessionState,
    build_arm_config_and_session,
    compute_topic_share,
)
from paperfeed.sim.llm import PaperForRating, Rater, RatingCache
from paperfeed.sim.persona import (
    SimPersona,
    get_eligible_subfields,
)


def test_static_blend_session_state_un_decayed_mean():
    dim = 4
    sess = StaticBlendSessionState(dim=dim)
    assert sess.is_empty
    assert np.allclose(sess.vector, np.zeros(dim))

    emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    sess.click("P1", emb1)
    assert not sess.is_empty
    assert np.allclose(sess.vector, emb1)

    emb2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    sess.click("P2", emb2)
    expected = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
    assert np.allclose(sess.vector, expected)

    emb3 = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    sess.save("P3", emb3)
    expected3 = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
    assert np.allclose(sess.vector, expected3)
    assert "P3" in sess.consumed


def test_persona_only_session_state_ignores_actions():
    dim = 4
    sess = PersonaOnlySessionState(dim=dim)
    emb = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    sess.click("P1", emb)
    sess.save("P2", emb)
    sess.skip("P3")
    assert sess.is_empty
    assert np.allclose(sess.vector, np.zeros(dim))
    assert len(sess.consumed) == 0
    assert len(sess.negatives) == 0


@pytest.mark.parametrize("arm", ARMS)
def test_build_arm_config_and_session(arm):
    base_cfg = RecommendConfig(lambda_div=0.2, k=10)
    base_weights = ScoringWeights(w_cite=0.5, w_facet=0.5, w_nov=0.5)

    cfg, sess = build_arm_config_and_session(
        arm=arm,
        base_config=base_cfg,
        base_weights=base_weights,
        dim=768,
        tuned_params={"alpha_sess": 0.6, "rho": 0.4, "w_sess": 1.2, "lambda_neg": 0.3, "gamma_pivot": 0.4},
    )

    if arm == "persona_only":
        assert cfg.rho == 1.0
        assert cfg.weights.w_pers == 1.0
        assert cfg.weights.w_sess == 0.0
        assert cfg.weights.lambda_neg == 0.0
        assert isinstance(sess, PersonaOnlySessionState)
    elif arm == "session_only":
        assert cfg.rho == 0.0
        assert cfg.weights.w_pers == 0.0
        assert cfg.weights.w_sess == 1.2
        assert cfg.weights.lambda_neg == 0.3
    elif arm == "static_blend":
        assert cfg.rho == 0.5
        assert cfg.weights.w_pers == 1.0
        assert cfg.weights.w_sess == 1.0
        assert isinstance(sess, StaticBlendSessionState)
    elif arm == "hybrid":
        assert cfg.rho == 0.4
        assert cfg.weights.w_pers == 1.0
        assert cfg.weights.w_sess == 1.2
        assert not cfg.use_gate
    elif arm == "hybrid_gated":
        assert cfg.use_gate
        assert cfg.gamma_pivot == 0.4


def test_compute_topic_share():
    lookup = {"P1": "subfield_1", "P2": "subfield_1", "P3": "subfield_2", "P4": "subfield_3"}
    assert compute_topic_share(["P1", "P2", "P3", "P4"], "subfield_1", lookup) == 0.5
    assert compute_topic_share(["P1", "P2"], "subfield_1", lookup) == 1.0
    assert compute_topic_share(["P3", "P4"], "subfield_1", lookup) == 0.0
    assert compute_topic_share([], "subfield_1", lookup) == 0.0


def test_sim_persona_serialization():
    p = SimPersona(
        author_id="A123",
        author_name="Alice Smith",
        split="evaluation",
        persona_text="Researcher in AI",
        topic_a_id="sf1",
        topic_a_name="Artificial Intelligence",
        topic_a_query="Artificial Intelligence",
        topic_b_id="sf2",
        topic_b_name="Information Systems",
        topic_b_query="Information Systems",
        recent_titles=("Paper 1", "Paper 2"),
        n_authored_T=8,
    )
    d = p.to_dict()
    assert d["author_id"] == "A123"
    assert d["recent_titles"] == ["Paper 1", "Paper 2"]
    p2 = SimPersona.from_dict(d)
    assert p2.author_id == p.author_id
    assert p2.recent_titles == p.recent_titles
