#!/usr/bin/env python3
"""Pilot tuning for Track 2 session parameters (Plan 9d / docs/design.md §13.2 point 6).

Tunes on the 10 pilot personas across 2 seeds:
- Criterion: J = mean(Retention@3) - mean(k_pivot) / 10
- Search grid around defaults:
    alpha_sess in [0.3, 0.5, 0.7] (default: 0.5)
    rho in [0.2, 0.3, 0.5] (default: 0.3)
    w_sess in [0.5, 1.0] (default: 1.0)
    lambda_neg in [0.0, 0.5] (default: 0.5)
    gamma_pivot in [0.25, 0.35, 0.45] (default: 0.35)
Freezes optimum into data/artifacts/track2_config.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paperfeed.common.config import find_project_root, load_config, load_dotenv_file
from paperfeed.common.logger import get_logger
from paperfeed.rec.recommender import RecommendConfig, RecommenderResources
from paperfeed.sim.episode import load_subfield_lookup, run_episode
from paperfeed.sim.llm import RatingCache, Rater, make_client
from paperfeed.sim.persona import SimPersona

logger = get_logger("scripts.tune_track2")


def evaluate_pilot_grid(
    pilot_personas: Sequence[SimPersona],
    seeds: Sequence[int],
    params: Mapping[str, Any],
    arm: str,
    resources: RecommenderResources,
    rater: Rater,
    subfield_lookup: Mapping[str, str],
    base_config: RecommendConfig,
) -> float:
    """Run pilot personas over seeds and return criterion J = mean(Retention@3) - mean(k_pivot) / 10."""
    j_values = []
    k_pivots = []
    retentions = []

    count = 0
    total = len(pilot_personas) * len(seeds)
    for p in pilot_personas:
        for s in seeds:
            ep = run_episode(
                persona=p,
                arm=arm,
                seed=s,
                resources=resources,
                rater=rater,
                subfield_lookup=subfield_lookup,
                base_config=base_config,
                tuned_params=params,
            )
            k_pivots.append(ep.k_pivot)
            retentions.append(ep.retention_at_3)
            count += 1
            if count % 5 == 0 or count == total:
                logger.info("  [%s] Pilot ep %d/%d (k_pivot=%d, Ret@3=%.2f, cache=%d)", arm, count, total, ep.k_pivot, ep.retention_at_3, len(rater.cache))

    mean_ret = float(sum(retentions) / len(retentions)) if retentions else 0.0
    mean_pivot = float(sum(k_pivots) / len(k_pivots)) if k_pivots else 10.0
    j = mean_ret - (mean_pivot / 10.0)
    return j, mean_ret, mean_pivot


def main() -> int:
    parser = argparse.ArgumentParser(description="Pilot tuning for Track 2 session parameters.")
    parser.add_argument("--provider", default="nvidia")
    parser.add_argument("--model", default="minimaxai/minimax-m3.1-flash")
    parser.add_argument("--personas", default="data/artifacts/track2_personas.json")
    parser.add_argument("--cache", default="data/artifacts/track2_rating_cache.jsonl")
    parser.add_argument("--output", default="data/artifacts/track2_config.json")
    args = parser.parse_args()

    load_dotenv_file()
    root = find_project_root()
    cfg = load_config()

    personas_path = root / args.personas
    if not personas_path.exists():
        logger.error("Personas file %s not found. Run scripts/build_track2_personas.py first.", personas_path)
        return 1

    with open(personas_path, "r", encoding="utf-8") as f:
        pdata = json.load(f)
    pilot_personas = [SimPersona.from_dict(x) for x in pdata.get("pilot", [])]
    logger.info("Loaded %d pilot personas.", len(pilot_personas))

    cache = RatingCache(root / args.cache)
    logger.info("Loaded rating cache with %d cached ratings.", len(cache))

    client = make_client(args.provider, args.model)
    rater = Rater(client=client, cache=cache, temperature=0.7, max_workers=2)

    logger.info("Loading recommender resources...")
    resources = RecommenderResources.load(cfg, root, load_encoder=True)
    subfield_lookup = load_subfield_lookup(resources.db.db_path)
    base_config = RecommendConfig.from_config(cfg)

    pilot_seeds = [42, 43]
    defaults = {
        "alpha_sess": 0.5,
        "rho": 0.3,
        "w_sess": 1.0,
        "lambda_neg": 0.5,
        "gamma_pivot": 0.35,
    }
    grid = {
        "alpha_sess": [0.3, 0.5, 0.7],
        "rho": [0.2, 0.3, 0.5],
        "w_sess": [0.5, 1.0],
        "lambda_neg": [0.0, 0.5],
        "gamma_pivot": [0.25, 0.35, 0.45],
    }

    current = dict(defaults)
    history = []

    logger.info("Beginning pilot tuning coordinate search (2 passes)...")
    for pass_idx in range(1, 3):
        logger.info("--- Pass %d of 2 ---", pass_idx)
        for param, candidates in grid.items():
            best_val = current[param]
            arm = "hybrid_gated" if param == "gamma_pivot" else "hybrid"

            # Score current setting first
            j_best, ret_best, piv_best = evaluate_pilot_grid(
                pilot_personas, pilot_seeds, current, arm, resources, rater, subfield_lookup, base_config
            )
            logger.info("Param %s: current %s -> J=%.4f (Ret@3=%.3f, Pivot=%.2f)", param, best_val, j_best, ret_best, piv_best)

            for val in candidates:
                if val == best_val:
                    continue
                test_params = dict(current)
                test_params[param] = val
                j_cand, ret_cand, piv_cand = evaluate_pilot_grid(
                    pilot_personas, pilot_seeds, test_params, arm, resources, rater, subfield_lookup, base_config
                )
                logger.info("  testing %s=%s -> J=%.4f (Ret@3=%.3f, Pivot=%.2f)", param, val, j_cand, ret_cand, piv_cand)
                # Ties keep current/default
                if j_cand > j_best + 1e-5:
                    j_best = j_cand
                    best_val = val
                    logger.info("  * new best %s=%s (J=%.4f)", param, val, j_best)

            current[param] = best_val
            history.append({
                "pass": pass_idx,
                "param": param,
                "chosen": best_val,
                "j": j_best,
            })

    # Evaluate final frozen settings
    j_final_hybrid, ret_h, piv_h = evaluate_pilot_grid(
        pilot_personas, pilot_seeds, current, "hybrid", resources, rater, subfield_lookup, base_config
    )
    j_final_gated, ret_g, piv_g = evaluate_pilot_grid(
        pilot_personas, pilot_seeds, current, "hybrid_gated", resources, rater, subfield_lookup, base_config
    )

    out_artifact = {
        "version": "track2-v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "provider": args.provider,
        "n_pilot_personas": len(pilot_personas),
        "pilot_seeds": pilot_seeds,
        "criterion": "J = Retention@3 - (k_pivot / 10)",
        "optimal_params": current,
        "final_metrics": {
            "hybrid": {"j": j_final_hybrid, "mean_retention_at_3": ret_h, "mean_k_pivot": piv_h},
            "hybrid_gated": {"j": j_final_gated, "mean_retention_at_3": ret_g, "mean_k_pivot": piv_g},
        },
        "history": history,
    }

    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_artifact, f, indent=2)

    logger.info("Frozen Track 2 configuration written to %s", out_path)
    print("\n--- Pilot Tuning Complete ---")
    print("Optimal parameters:")
    for k, v in current.items():
        print(f"  {k}: {v}")
    print(f"Saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
