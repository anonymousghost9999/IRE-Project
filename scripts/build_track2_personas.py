#!/usr/bin/env python3
"""Build Track 2 evaluation and pilot personas (Plan 9b / docs/design.md §13.2 point 4).

Usage:
    python scripts/build_track2_personas.py [--output data/artifacts/track2_personas.json]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paperfeed.common.config import find_project_root, load_config
from paperfeed.common.logger import get_logger
from paperfeed.sim.persona import generate_track2_personas, save_personas_artifact

logger = get_logger("scripts.build_track2_personas")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Track 2 evaluation and pilot personas.")
    parser.add_argument("--output", default="data/artifacts/track2_personas.json", help="Path to output JSON")
    args = parser.parse_args()

    root = find_project_root()
    cfg = load_config()

    db_path = root / cfg["paths"]["sqlite_db"]
    raw_dir = root / cfg["paths"]["raw_dir"]
    cohort_path = root / cfg["paths"]["processed_dir"] / "cohort.jsonl"
    split_path = root / cfg["paths"]["processed_dir"] / "split.json"

    personas = generate_track2_personas(
        db_path=db_path,
        raw_dir=raw_dir,
        cohort_path=cohort_path,
        split_path=split_path,
        n_eval=30,
        n_pilot=10,
        seed=cfg["project"].get("random_seed", 42),
    )

    out_path = root / args.output
    save_personas_artifact(personas, out_path)

    print(f"Generated {len(personas['evaluation'])} evaluation personas and {len(personas['pilot'])} pilot personas.")
    print(f"Written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
