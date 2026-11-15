#!/usr/bin/env python3
"""Probe an LLM provider's free tier before choosing the Track 2 simulator (plan 9a). Makes at most one rating call.

Usage (the key comes from the provider's environment variable only, e.g. NVIDIA_API_KEY, MISTRAL_API_KEY):
    python scripts/probe_llm.py --provider nvidia                       # list the models this key can call
    python scripts/probe_llm.py --provider nvidia --model <model id>   # + one tiny rating call and its limit headers

Providers: gemini (GEMINI_API_KEY) and the OpenAI-format ones in paperfeed.sim.llm.OPENAI_COMPATIBLE.
Prints the rate-limit headers the provider returns (limits and remaining quota as your key sees them) and whether
the reply parses as a valid rating. Nothing is cached or written.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paperfeed.common.config import load_dotenv_file
from paperfeed.sim.llm import OPENAI_COMPATIBLE, PaperForRating, Rater, RatingCache, make_client

load_dotenv_file()

PAPERS = [
    PaperForRating("W1", "Dense passage retrieval for open-domain question answering",
                   "We train a dual encoder to retrieve passages for questions and outperform BM25."),
    PaperForRating("W2", "Rotation curves of spiral galaxies", "We measure rotation curves of 50 nearby galaxies."),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe an LLM provider's free tier.")
    parser.add_argument("--provider", required=True, choices=["gemini", *OPENAI_COMPATIBLE])
    parser.add_argument("--model", default=None, help="Model ID to test with one rating call")
    args = parser.parse_args()

    client = make_client(args.provider, args.model or "unused")
    try:
        models = client.list_models()
        print(f"{len(models)} models available to this key:")
        for m in models:
            print("  ", m)
    except Exception as exc:  # listing is a convenience; some providers restrict it
        print(f"Could not list models: {exc}")
    if not args.model:
        print("\nRe-run with --model <id> to test one rating call and see the rate-limit headers.")
        return 0

    rater = Rater(client, RatingCache(), max_rate_limit_waits=0)
    t0 = time.time()
    try:
        ratings = rater.rate("probe", "A researcher in information retrieval.", "A",
                             "Find papers on dense retrieval.", PAPERS, seed=0)
        print(f"\nRating call OK in {time.time() - t0:.1f}s: {ratings}  (expected W1 high, W2 low)")
    except Exception as exc:
        print(f"\nRating call failed: {exc!r}")
    headers = getattr(client, "last_headers", {})
    print("\nRate-limit headers returned by the provider:")
    for k, v in sorted(headers.items()):
        print(f"   {k}: {v}")
    if not headers:
        print("   (none: this provider does not report limits in its replies; check its console)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
