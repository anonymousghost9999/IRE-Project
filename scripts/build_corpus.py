"""Main execution script to mine evaluation candidate authors and crawl the time-bounded corpus.

Usage:
    python scripts/build_corpus.py [--max-authors 1000] [--target-corpus-size 120000]
"""

import argparse
from pathlib import Path
import sys
from paperfeed.common.config import find_project_root, load_config
from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import CohortMiner, load_candidate_authors
from paperfeed.data.crawl import CorpusCrawler
from paperfeed.data.db import DatabaseManager
from paperfeed.openalex.client import OpenAlexClient

logger = get_logger("scripts.build_corpus")


def parse_args():
    parser = argparse.ArgumentParser(description="Mine candidate authors and build corpus.")
    parser.add_argument(
        "--max-authors",
        type=int,
        default=1000,
        help="Maximum candidate evaluation authors to discover (default: 1000)",
    )
    parser.add_argument(
        "--target-corpus-size",
        type=int,
        default=120000,
        help="Target total candidate papers in SQLite corpus (default: 120000)",
    )
    parser.add_argument(
        "--authors-file",
        type=str,
        default="data/candidate_authors.jsonl",
        help="Path to candidate authors JSONL file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = find_project_root()
    cfg = load_config()

    cutoff_year = cfg.get("project", {}).get("cutoff_year", 2024)
    target_year = cfg.get("project", {}).get("target_year", 2025)

    authors_path = Path(args.authors_file)
    if not authors_path.is_absolute():
        authors_path = root / authors_path

    logger.info("Initializing OpenAlex Client and SQLite DatabaseManager...")
    client = OpenAlexClient()
    db = DatabaseManager(cutoff_year=cutoff_year)

    # Step 1: Mine candidate authors if not already present
    if not authors_path.exists() or authors_path.stat().st_size == 0:
        logger.info(
            f"Step 1: Mining up to {args.max_authors} candidate authors (Target year {target_year}, Cutoff {cutoff_year})..."
        )
        miner = CohortMiner(
            client=client,
            cutoff_year=cutoff_year,
            target_year=target_year,
        )
        candidates = miner.mine_candidate_authors(
            max_candidates=args.max_authors,
            output_file=authors_path,
        )
        logger.info(f"Step 1 Complete: Saved {len(candidates)} candidate authors to {authors_path}")
    else:
        candidates = load_candidate_authors(authors_path)
        logger.info(f"Loaded {len(candidates)} existing candidate authors from {authors_path}")

    # Step 2: Run corpus crawl pipeline
    logger.info(
        f"Step 2: Starting Corpus Crawl Pipeline towards target size {args.target_corpus_size} papers..."
    )
    crawler = CorpusCrawler(
        client=client,
        db=db,
        config=cfg,
    )
    summary = crawler.run_pipeline(
        candidate_authors_path=authors_path,
        target_size=args.target_corpus_size,
    )

    logger.info("==================================================")
    logger.info("CORPUS BUILD SUMMARY:")
    logger.info(f"Total Papers in Database:     {summary.get('final_paper_count')}")
    logger.info(f"Total Citations in Database:  {summary.get('final_citation_count')}")
    logger.info(f"Authored Papers Seeded:       {summary.get('authored_papers')}")
    logger.info(f"Hop-1 Discovered Papers:      {summary.get('hop1_papers')}")
    logger.info(f"Hop-2 Discovered Papers:      {summary.get('hop2_papers')}")
    logger.info(f"Foundational Anchors Ingested:{summary.get('foundational_anchors')}")
    logger.info("==================================================")

    client.close()


if __name__ == "__main__":
    main()

