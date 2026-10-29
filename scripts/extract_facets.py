#!/usr/bin/env python3
"""CLI script to extract and canonicalize scientific facets (Phase 3 / docs/plan.md §3).

Iterates over candidate papers in `data/processed/paperfeed.sqlite3`, extracts
typed scientific facets (task, method, dataset, metric, topic) using GLiNER,
clusters and canonicalizes them, and persists into SQLite `paper_facets` and `facet_canon`.
"""

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

# Add project root and src/ to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from paperfeed.facets.canonical import FacetCanonicalizer
from paperfeed.facets.extractor import FacetExtractor
from paperfeed.facets.store import FacetStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("extract_facets")


def main():
    parser = argparse.ArgumentParser(description="Extract and canonicalize scientific facets.")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=project_root / "data" / "processed" / "paperfeed.sqlite3",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of candidate papers to process (for testing or profile audit)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for database updates and processing",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="GLiNER confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.25,
        help="Cosine distance threshold for facet canonicalization clustering (default: 0.25)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing paper_facets and facet_canon tables before running",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model path or name for GLiNER",
    )
    args = parser.parse_args()

    if not args.db_path.exists():
        logger.error("Database file not found: %s", args.db_path)
        sys.exit(1)

    store = FacetStore(args.db_path)

    # 1. Fetch candidate papers with abstracts
    conn = sqlite3.connect(str(args.db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    query = """
        SELECT paper_id, title, abstract
        FROM papers
        WHERE in_candidates = 1
          AND abstract IS NOT NULL
          AND LENGTH(abstract) >= 150
        ORDER BY year DESC, paper_id ASC
    """
    if args.limit:
        query += f" LIMIT {args.limit}"

    cur.execute(query)
    rows = cur.fetchall()
    conn.close()

    total_papers = len(rows)
    logger.info("Found %d candidate papers to process.", total_papers)
    if total_papers == 0:
        logger.info("No papers to process.")
        return

    # 2. Initialize extractor
    extractor = FacetExtractor(
        model_name_or_path=args.model,
        threshold=args.threshold,
    )

    all_extracted_facets = []
    start_time = time.time()
    last_log_time = start_time
    last_log_count = 0

    logger.info("Starting GLiNER extraction on %d papers...", total_papers)
    for idx, row in enumerate(rows, 1):
        p_id = row["paper_id"]
        abstract = row["abstract"]
        facets = extractor.extract_from_abstract(p_id, abstract)
        all_extracted_facets.extend(facets)

        if idx % args.batch_size == 0 or idx == total_papers:
            now = time.time()
            elapsed_batch = now - last_log_time
            papers_in_batch = idx - last_log_count
            sec_per_1k = (elapsed_batch / papers_in_batch) * 1000 if papers_in_batch > 0 else 0
            logger.info(
                "Processed %d/%d papers (%.1f%%) | %d facets extracted | Rate: %.2fs / 1k abstracts",
                idx,
                total_papers,
                (idx / total_papers) * 100,
                len(all_extracted_facets),
                sec_per_1k,
            )
            last_log_time = now
            last_log_count = idx

    total_extraction_time = time.time() - start_time
    rate_1k = (total_extraction_time / total_papers) * 1000 if total_papers > 0 else 0
    logger.info(
        "Extraction finished: %d papers in %.2fs (%.2fs / 1k abstracts). Total raw facets: %d",
        total_papers,
        total_extraction_time,
        rate_1k,
        len(all_extracted_facets),
    )

    if not all_extracted_facets:
        logger.warning("No facets were extracted.")
        return

    # 3. Canonicalization
    logger.info("Starting canonicalization clustering...")
    canonicalizer = FacetCanonicalizer(
        distance_threshold=args.distance_threshold,
    )
    canon_start = time.time()
    canonicalized_facets, canonical_defs = canonicalizer.canonicalize(all_extracted_facets)
    canon_time = time.time() - canon_start
    logger.info(
        "Canonicalization complete in %.2fs: %d canonical definitions created.",
        canon_time,
        len(canonical_defs),
    )

    # 4. Persistence into SQLite
    logger.info("Persisting facets into database: %s", args.db_path)
    store.save_facets(canonicalized_facets, clear_existing=args.clear)
    store.save_canonical_clusters(canonical_defs, clear_existing=args.clear)

    # 5. Summary statistics
    facet_counts_by_type = {}
    for f in canonicalized_facets:
        ftype = f["facet_type"]
        facet_counts_by_type[ftype] = facet_counts_by_type.get(ftype, 0) + 1

    canon_by_type = {}
    for c in canonical_defs:
        ftype = c["facet_type"]
        canon_by_type[ftype] = canon_by_type.get(ftype, 0) + 1

    print("\n" + "=" * 60)
    print("PHASE 3 FACET EXTRACTION & CANONICALIZATION SUMMARY")
    print("=" * 60)
    print(f"Papers processed:         {total_papers}")
    print(f"Total facets extracted:   {len(canonicalized_facets)}")
    print(f"Facets per paper:         {len(canonicalized_facets) / total_papers:.2f}")
    print(f"Extraction speed:         {rate_1k:.2f}s per 1k abstracts")
    print(f"Total canonical clusters: {len(canonical_defs)}")
    print("\nFacet distribution by type:")
    for ftype, count in sorted(facet_counts_by_type.items()):
        canon_count = canon_by_type.get(ftype, 0)
        print(f"  - {ftype:10s}: {count:6d} facets -> {canon_count:5d} canonical clusters")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

