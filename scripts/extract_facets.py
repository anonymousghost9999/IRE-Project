#!/usr/bin/env python3
"""CLI script to extract and canonicalize scientific facets (Phase 3 / docs/plan.md §3).

Iterates over candidate papers in `data/processed/paperfeed.sqlite3`, extracts
typed scientific facets (task, method, dataset, metric, topic) using batched GLiNER,
clusters and canonicalizes them, and persists into SQLite `paper_facets` and `facet_canon`.
Supports checkpointed batches and resume.
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
        default=32,
        help="Number of papers per batch for GPU inference and checkpointing (default: 32)",
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
        "--resume",
        action="store_true",
        default=True,
        help="Skip papers that already have extracted facets in paper_facets (default: True)",
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

    if args.clear:
        conn = sqlite3.connect(str(args.db_path))
        with conn:
            conn.execute("DELETE FROM paper_facets")
            conn.execute("DELETE FROM facet_canon")
        conn.close()
        logger.info("Cleared existing paper_facets and facet_canon tables.")

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
    """
    if args.resume and not args.clear:
        query += " AND paper_id NOT IN (SELECT DISTINCT paper_id FROM paper_facets)"

    query += " ORDER BY year DESC, paper_id ASC"
    if args.limit:
        query += f" LIMIT {args.limit}"

    cur.execute(query)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    total_papers = len(rows)
    logger.info("Found %d candidate papers to process.", total_papers)
    if total_papers == 0:
        logger.info("No remaining papers to extract. Running canonicalization pass...")
    else:
        # 2. Initialize extractor
        extractor = FacetExtractor(
            model_name_or_path=args.model,
            threshold=args.threshold,
        )

        all_extracted_facets = []
        start_time = time.time()
        last_log_time = start_time
        processed_count = 0

        logger.info("Starting batched GLiNER extraction on %d papers...", total_papers)
        for i in range(0, total_papers, args.batch_size):
            batch = rows[i : i + args.batch_size]
            batch_facets = extractor.extract_from_papers(batch, sentence_batch_size=32)
            all_extracted_facets.extend(batch_facets)

            # Checkpoint save to SQLite
            if batch_facets:
                store.save_facets(batch_facets)

            processed_count += len(batch)
            now = time.time()
            elapsed = now - last_log_time
            rate_1k = (elapsed / len(batch)) * 1000 if len(batch) > 0 else 0
            logger.info(
                "Processed %d/%d papers (%.1f%%) | %d batch facets | Rate: %.2fs / 1k abstracts",
                processed_count,
                total_papers,
                (processed_count / total_papers) * 100,
                len(batch_facets),
                rate_1k,
            )
            last_log_time = now

        total_time = time.time() - start_time
        overall_rate = (total_time / total_papers) * 1000 if total_papers > 0 else 0
        logger.info(
            "Extraction finished: %d papers in %.2fs (%.2fs / 1k abstracts). Total new facets: %d",
            total_papers,
            total_time,
            overall_rate,
            len(all_extracted_facets),
        )

    # 3. Canonicalization pass across all facets in database
    logger.info("Fetching all facets from database for global canonicalization clustering...")
    conn = sqlite3.connect(str(args.db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT facet_id, paper_id, facet_type, facet_value
        FROM paper_facets
    """)
    db_facets = [dict(r) for r in cur.fetchall()]
    conn.close()

    if not db_facets:
        logger.warning("No facets in database to canonicalize.")
        return

    logger.info("Canonicalizing %d total database facets...", len(db_facets))
    canonicalizer = FacetCanonicalizer(
        distance_threshold=args.distance_threshold,
    )
    canon_start = time.time()
    canonicalized_facets, canonical_defs = canonicalizer.canonicalize(db_facets)
    canon_time = time.time() - canon_start
    logger.info(
        "Canonicalization complete in %.2fs: %d canonical definitions created.",
        canon_time,
        len(canonical_defs),
    )

    # 4. Update canonical_id in database and save canonical clusters
    logger.info("Updating canonical IDs in SQLite database...")
    store.save_canonical_clusters(canonical_defs, clear_existing=True)

    conn = sqlite3.connect(str(args.db_path))
    with conn:
        update_rows = [(f["canonical_id"], f["facet_id"]) for f in canonicalized_facets]
        conn.executemany(
            "UPDATE paper_facets SET canonical_id = ? WHERE facet_id = ?",
            update_rows,
        )
    conn.close()

    # 5. Summary statistics
    facet_counts_by_type = {}
    for f in canonicalized_facets:
        ftype = f["facet_type"]
        facet_counts_by_type[ftype] = facet_counts_by_type.get(ftype, 0) + 1

    canon_by_type = {}
    for c in canonical_defs:
        ftype = c["facet_type"]
        canon_by_type[ftype] = canon_by_type.get(ftype, 0) + 1

    total_in_db = len(canonicalized_facets)
    print("\n" + "=" * 60)
    print("PHASE 3 FACET EXTRACTION & CANONICALIZATION SUMMARY")
    print("=" * 60)
    print(f"Total facets in database: {total_in_db}")
    print(f"Total canonical clusters: {len(canonical_defs)}")
    print("\nFacet distribution by type:")
    for ftype, count in sorted(facet_counts_by_type.items()):
        canon_count = canon_by_type.get(ftype, 0)
        print(f"  - {ftype:10s}: {count:6d} facets -> {canon_count:5d} canonical clusters")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
