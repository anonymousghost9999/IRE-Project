#!/usr/bin/env python3
"""Full-corpus dense index for the V1 encoder candidate (plan 10.1d / docs/design.md §13.8 V1 point 7).

Usage:
    python scripts/build_encoder_index.py [--encoder bge] [--batch-size 32]
    python scripts/build_encoder_index.py --comparison --encoder gte      # Phase 11 comparison vectors (scincl, gte)

Embeds exactly the papers of the Core SPECTER2 store with the encoder chosen as the V1 candidate in
data/artifacts/variant_v1_decision.json (refuses any other encoder) and writes <processed_dir>/encoder_<key>/:
embeddings.npy + embedding_ids.json (checkpointed, resumable), faiss.index + paper_ids.json, index_manifest.json.
BM25s and facets are reused. Then run scripts/run_track1_eval.py --encoder <key>.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paperfeed.common.config import load_config
from paperfeed.common.logger import get_logger
from paperfeed.data.db import DatabaseManager
from paperfeed.eval.encoder_study import build_full_encoder_index
from paperfeed.index.encoders import ENCODER_SPECS, make_encoder
from paperfeed.index.store import EmbeddingStore

logger = get_logger("scripts.build_encoder_index")

PHASE11_COMPARISON = ("scincl", "gte")  # comparison systems' vectors (never used by Paper Feed itself)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the full-corpus index for the V1 encoder candidate.")
    parser.add_argument("--encoder", default=None, help="Defaults to the frozen V1 candidate")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--comparison", action="store_true",
                        help="Build a Phase 11 comparison encoder (scincl or gte, design §13.9 Phase 11 spec point 4) "
                        "instead of the V1 candidate")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = load_config()
    processed = root / cfg["paths"]["processed_dir"]
    decision_path = root / cfg["paths"]["artifacts_dir"] / "variant_v1_decision.json"
    if args.comparison:
        if args.encoder not in PHASE11_COMPARISON:
            print(f"--comparison builds one of {PHASE11_COMPARISON}.", file=sys.stderr)
            return 1
    elif not decision_path.exists():
        print(f"{decision_path} not found: freeze the V1 decision with scripts/decide_encoder.py first.", file=sys.stderr)
        return 1
    candidate = None if args.comparison else json.loads(decision_path.read_text(encoding="utf-8"))["chosen_encoder"]
    key = args.encoder or candidate
    if not args.comparison and key != candidate:
        print(f"The frozen V1 candidate is {candidate!r}; only it gets a full-corpus index and test run.", file=sys.stderr)
        return 1

    reference = EmbeddingStore.load(processed / "embeddings.npy", processed / "embedding_ids.json", processed / "paper_ids.json")
    db = DatabaseManager(root / cfg["paths"]["sqlite_db"], cutoff_year=int(cfg["project"]["cutoff_year"]))
    every = int(cfg.get("index", {}).get("embed_checkpoint_every", 10000))
    manifest = build_full_encoder_index(list(reference.paper_id_index), db, make_encoder(key, track1=True),
                                        ENCODER_SPECS[key].model_name, processed / f"encoder_{key}", args.batch_size, every)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
