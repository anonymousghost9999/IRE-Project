"""Phase 11a: assemble ``configs/best.yaml`` from the frozen tuning-split decisions (docs/design.md §13.9, plan 11.1).

Every part of the architecture gets its choice, the evidence it rests on (metric, gain, 95% CI, number of tuning
authors, source file) and a one-line rationale. Only tuning-split decisions are read; the test-split variant table is
never an input. The replacement rule is the one each decision applied: an option replaces Core only if the CI of its
tuning-split NDCG@10 gain lies above zero.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

DECISION_FILES = {
    "dense_encoder": ("variant_v1_decision.json", "adopt_v1"),
    "feed_reranking": ("variant_v2_decision.json", "adopt_v2"),
    "scorer": ("variant_v3_decision.json", "adopt_v3"),
    "candidate_channels": ("variant_v4_decision.json", "adopt_v4"),
    "pool_size": ("pool_size_decision.json", "adopt_larger_pool"),
}


def _load(artifacts: Path, name: str) -> Dict[str, Any]:
    path = artifacts / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: every decision must be frozen before configs/best.yaml.")
    return json.loads(path.read_text(encoding="utf-8"))


def _evidence(d: Mapping[str, Any], source: str) -> Dict[str, Any]:
    g = d["ndcg_gain"]
    return {"metric": "tuning-split NDCG@10 gain over Core (paired bootstrap)", "gain": round(g["mean_diff"], 6),
            "relative": round(g["relative"], 4) if g.get("relative") is not None else None,
            "ci95": [round(g["ci"][0], 6), round(g["ci"][1], 6)], "n_users": d["n_users"], "source": source}


def build_best_config(artifacts_dir: Path, core_tuning_file: str, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    artifacts = Path(artifacts_dir)
    decisions = {part: (_load(artifacts, f), key, f) for part, (f, key) in DECISION_FILES.items()}
    adopted = {part: bool(d[key]) for part, (d, key, _) in decisions.items()}

    def entry(part: str, core_choice: Any, alt_choice: Any, why_kept: str, why_adopted: str) -> Dict[str, Any]:
        d, _, f = decisions[part]
        a = adopted[part]
        return {"choice": alt_choice if a else core_choice, "replaces_core": a,
                "evidence": _evidence(d, f"data/artifacts/{f}"), "rationale": why_adopted if a else why_kept}

    v1 = decisions["dense_encoder"][0]
    v2 = decisions["feed_reranking"][0]
    parts = {
        "dense_encoder": entry("dense_encoder", "specter2", v1.get("chosen_encoder"),
                               "V1 candidate's gain CI includes zero; SPECTER2 is kept.",
                               "V1 candidate's gain CI lies above zero."),
        "candidate_channels": entry("candidate_channels", ["dense", "bm25s"], ["dense", "bm25s", "citation_graph"],
                                    "V4 citation channel's gain CI includes zero; text channels only.",
                                    "V4 citation channel's gain CI lies above zero."),
        "pool_size": entry("pool_size", {"pool_size": 100, "channel_depth": 200},
                           {"pool_size": 250, "channel_depth": 500},
                           "Larger pool's gain CI includes zero; M = 100, M_channel = 200 kept.",
                           "Larger pool's gain CI lies above zero (w_pop = 0)."),
        "feed_reranking": entry("feed_reranking", None, v2.get("chosen_model"),
                                "V2 cross-encoder's gain CI includes zero; no reranker.",
                                "V2 cross-encoder's gain CI lies above zero."),
        "scorer": entry("scorer", "linear", "lambdamart",
                        "V3 out-of-fold gain CI includes zero; the linear scorer is kept.",
                        "V3 out-of-fold gain CI lies above zero."),
    }
    n_replaced = sum(p["replaces_core"] for p in parts.values())

    # weights and lambda_div: the grid optimum of the tuning file matching the chosen pool
    tuning_name = decisions["pool_size"][0]["larger_pool_tuning_file"] if adopted["pool_size"] else core_tuning_file
    tuning = _load(artifacts, tuning_name)
    weights = {k: v for k, v in tuning["best_weights"].items()}
    parts["diversification"] = {
        "choice": {"lambda_div": tuning["best_lambda_div"]}, "replaces_core": False,
        "evidence": {"metric": "tuning-split mean NDCG@10 (§9 grid optimum)", "value": tuning["best_mean_ndcg"],
                     "n_users": tuning["n_users"], "source": f"data/artifacts/{tuning_name}"},
        "rationale": "lambda_div is part of the §9 grid tuned on the tuning split.",
    }
    um = cfg.get("user_model", {})
    parts["session_model"] = {
        "choice": {"alpha_sess": um.get("alpha_session"), "rho": um.get("rho_persona_weight"),
                   "use_pivot_gate": um.get("use_pivot_gate", False), "gamma_pivot": um.get("gamma_pivot")},
        "replaces_core": False,
        "evidence": None,
        "rationale": "No evidence: Track 2 (Phase 9) is deferred, so the design defaults are kept. Track 1 feeds use an "
                     "empty session, so these values do not affect any Track 1 or Phase 11 number.",
    }
    return {
        "best_configuration": "core" if n_replaced == 0 else "assembled",
        "replacement_rule": "An option replaces the Core component only if its tuning-split NDCG@10 gain has a 95% "
                            "bootstrap CI above zero (design §13.9). Test-split results are never used.",
        "assembled_check": ("Not needed: no component was replaced, so the best configuration is Core." if n_replaced == 0
                            else "Single replacement: its own decision is the assembled-system check." if n_replaced == 1
                            else "Multiple replacements: run the §13.9 assembled-system check before freezing."),
        "scoring_weights": weights,
        "tuning_file": f"data/artifacts/{tuning_name}",
        "parts": parts,
    }
