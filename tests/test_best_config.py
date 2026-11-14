"""Phase 11a: pool-size decision and configs/best.yaml assembly."""

import json

import numpy as np
import pytest

from paperfeed.eval.best_config import DECISION_FILES, build_best_config
from paperfeed.eval.tuning import CachedUserCandidateData, GridPoint, decide_pool_size
from paperfeed.rec.scoring import ScoringWeights


def _gp(ndcgs):
    params = {"w_cite": 1.0, "w_facet": 0.25, "w_nov": 0.25, "w_pop": 0.0, "lambda_div": 0.5}
    return GridPoint(params, ScoringWeights(), 0.5, np.asarray(ndcgs, dtype=float))


def _users(n, pool):
    return [CachedUserCandidateData(f"A{i}", None, pool, [], ["t1", "t2"]) for i in range(n)]


def test_pool_size_rule():
    n = 40
    core, alt_users = [_gp([0.2] * n)], _users(n, ["t1", "t2"])
    better = [_gp([0.3 + 0.01 * (i % 3) for i in range(n)])]
    d = decide_pool_size(core, better, _users(n, ["t1"]), alt_users, n_resamples=300)
    assert d["adopt_larger_pool"] is True and d["rule"]["alternative"] == {"pool_size": 250, "channel_depth": 500}
    assert d["candidate_recall_gain"]["mean_diff"] == pytest.approx(0.5)
    lucky = [0.2] * n
    lucky[0] = 0.9
    assert decide_pool_size(core, [_gp(lucky)], _users(n, ["t1"]), alt_users, n_resamples=300)["adopt_larger_pool"] is False


def _write_decisions(tmp_path, adopt=()):
    gain = {"mean_diff": 0.01, "relative": 0.4, "ci": [-0.002, 0.02]}
    for part, (fname, key) in DECISION_FILES.items():
        d = {"ndcg_gain": gain, "n_users": 80, key: part in adopt, "chosen_encoder": "bge",
             "chosen_model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "larger_pool_tuning_file": "tuning_results_big.json"}
        (tmp_path / fname).write_text(json.dumps(d))
    for name, lam in (("tuning_results_core.json", 0.5), ("tuning_results_big.json", 0.1)):
        (tmp_path / name).write_text(json.dumps({"best_weights": {"w_cite": 1.0, "w_facet": 0.25, "w_pop": 0.0},
                                                  "best_lambda_div": lam, "best_mean_ndcg": 0.024, "n_users": 80}))


CFG = {"user_model": {"alpha_session": 0.5, "rho_persona_weight": 0.3, "use_pivot_gate": False, "gamma_pivot": 0.35}}


def test_best_config_is_core_when_nothing_cleared_its_rule(tmp_path):
    _write_decisions(tmp_path)
    best = build_best_config(tmp_path, "tuning_results_core.json", CFG)
    assert best["best_configuration"] == "core" and best["assembled_check"].startswith("Not needed")
    p = best["parts"]
    assert p["dense_encoder"]["choice"] == "specter2" and p["scorer"]["choice"] == "linear"
    assert p["feed_reranking"]["choice"] is None and p["candidate_channels"]["choice"] == ["dense", "bm25s"]
    assert p["pool_size"]["choice"] == {"pool_size": 100, "channel_depth": 200}
    assert p["dense_encoder"]["evidence"]["ci95"] == [-0.002, 0.02] and p["dense_encoder"]["evidence"]["n_users"] == 80
    assert p["diversification"]["choice"] == {"lambda_div": 0.5} and best["tuning_file"].endswith("tuning_results_core.json")
    assert p["session_model"]["evidence"] is None and "deferred" in p["session_model"]["rationale"]


def test_adopted_pool_uses_its_own_tuning_file_and_multiple_replacements_need_the_assembled_check(tmp_path):
    _write_decisions(tmp_path, adopt=("pool_size",))
    best = build_best_config(tmp_path, "tuning_results_core.json", CFG)
    assert best["best_configuration"] == "assembled" and best["assembled_check"].startswith("Single")
    assert best["parts"]["pool_size"]["choice"] == {"pool_size": 250, "channel_depth": 500}
    assert best["parts"]["diversification"]["choice"] == {"lambda_div": 0.1}
    _write_decisions(tmp_path, adopt=("pool_size", "dense_encoder"))
    assert build_best_config(tmp_path, "tuning_results_core.json", CFG)["assembled_check"].startswith("Multiple")


def test_every_decision_must_exist(tmp_path):
    _write_decisions(tmp_path)
    (tmp_path / "pool_size_decision.json").unlink()
    with pytest.raises(FileNotFoundError, match="frozen before"):
        build_best_config(tmp_path, "tuning_results_core.json", CFG)
