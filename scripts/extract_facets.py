#!/usr/bin/env python3
"""CLI script to extract and canonicalize scientific facets (Phase 3 / docs/plan.md §3).

Iterates over candidate papers in `data/processed/paperfeed.sqlite3`, extracts typed scientific facets
(task, method, dataset, metric, topic) using batched GLiNER, clusters and canonicalizes them, and
persists into SQLite `paper_facets` and `facet_canon`. Settings come from the `facets` section of
configs/default.yaml (command-line flags override them).

Every finished batch is checkpointed together with the IDs of its papers (`facet_progress`), so a rerun
resumes where it stopped and never re-extracts a paper, including papers that yielded no facets. Use
--clear to start over. Writes data/processed/facet_manifest.json (settings, counts and timings).
"""

import argparse
import json
import logging
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

# Add project root and src/ to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from paperfeed.common.config import load_config
from paperfeed.facets.canonical import FacetCanonicalizer
from paperfeed.facets.extractor import FacetExtractor
from paperfeed.facets.store import FacetStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("extract_facets")


def main():
    cfg = load_config()
    fcfg = cfg.get("facets", {})
    cutoff = cfg["project"]["cutoff_year"]
    min_chars = cfg.get("index", {}).get("min_abstract_chars", 1)

    parser = argparse.ArgumentParser(description="Extract and canonicalize scientific facets.")
    parser.add_argument("--db-path", type=Path, default=project_root / cfg["paths"]["sqlite_db"], help="Path to SQLite database")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of candidate papers to process (for testing or profile audit)")
    parser.add_argument("--batch-size", type=int, default=fcfg.get("paper_batch_size", 32), help="Papers per batch for GPU inference and checkpointing")
    parser.add_argument("--threshold", type=float, default=fcfg.get("threshold", 0.5), help="GLiNER confidence threshold")
    parser.add_argument("--distance-threshold", type=float, default=fcfg.get("canon_distance_threshold", 0.25), help="Cosine distance threshold for facet canonicalization")
    parser.add_argument("--clear", action="store_true", help="Clear existing facets, canonical clusters and progress before running")
    parser.add_argument("--model", type=str, default=cfg["models"].get("gliner_model"), help="Model path or name for GLiNER")
    parser.add_argument("--manifest", type=Path, default=project_root / cfg["paths"]["processed_dir"] / "facet_manifest.json")
    args = parser.parse_args()

    if not args.db_path.exists():
        logger.error("Database file not found: %s", args.db_path)
        sys.exit(1)

    store = FacetStore(args.db_path)
    if args.clear:
        store.clear_all()
        logger.info("Cleared existing facets, canonical clusters and extraction progress.")

    old_manifest = {}
    if args.manifest.exists() and not args.clear:
        old_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    extraction_seconds = float(old_manifest.get("extraction_time_seconds", 0.0))
    papers_timed = int(old_manifest.get("papers_extracted_timed", 0))

    # 1. Candidate papers with a usable abstract that have not been processed yet
    conn = sqlite3.connect(str(args.db_path))
    conn.row_factory = sqlite3.Row
    query = """
        SELECT paper_id, title, abstract
        FROM papers
        WHERE in_candidates = 1
          AND year <= ?
          AND abstract IS NOT NULL
          AND LENGTH(abstract) >= ?
          AND paper_id NOT IN (SELECT paper_id FROM facet_progress)
        ORDER BY year DESC, paper_id ASC
    """
    params = [cutoff, min_chars]
    if args.limit:
        query += " LIMIT ?"
        params.append(args.limit)
    rows = [dict(r) for r in conn.execute(query, params).fetchall()]
    conn.close()

    total_papers = len(rows)
    logger.info("Found %d candidate papers to process (%d already done).", total_papers, store.count_processed())
    new_facets = 0
    if total_papers == 0:
        logger.info("No remaining papers to extract. Running canonicalization pass...")
    else:
        extractor = FacetExtractor(
            model_name_or_path=args.model,
            labels=fcfg.get("labels"),
            threshold=args.threshold,
        )
        sentence_batch = fcfg.get("sentence_batch_size", 32)
        start_time = time.time()
        last_log_time = start_time
        processed_count = 0

        logger.info("Starting batched GLiNER extraction on %d papers...", total_papers)
        for i in range(0, total_papers, args.batch_size):
            batch = rows[i : i + args.batch_size]
            batch_facets = extractor.extract_from_papers(batch, sentence_batch_size=sentence_batch)
            # Facets and "these papers are done" are committed together.
            store.save_facets(batch_facets, processed_paper_ids=[p["paper_id"] for p in batch])
            new_facets += len(batch_facets)

            processed_count += len(batch)
            now = time.time()
            rate_1k = ((now - last_log_time) / len(batch)) * 1000
            logger.info(
                "Processed %d/%d papers (%.1f%%) | %d batch facets | Rate: %.2fs / 1k abstracts",
                processed_count, total_papers, (processed_count / total_papers) * 100, len(batch_facets), rate_1k,
            )
            last_log_time = now

        total_time = time.time() - start_time
        extraction_seconds += total_time
        papers_timed += total_papers
        logger.info(
            "Extraction finished: %d papers in %.2fs (%.2fs / 1k abstracts). Total new facets: %d",
            total_papers, total_time, (total_time / total_papers) * 1000, new_facets,
        )

    # 2. Canonicalization pass across all facets in the database
    logger.info("Fetching all facets from database for global canonicalization clustering...")
    conn = sqlite3.connect(str(args.db_path))
    conn.row_factory = sqlite3.Row
    db_facets = [dict(r) for r in conn.execute("SELECT facet_id, paper_id, facet_type, facet_value FROM paper_facets")]
    conn.close()

    if not db_facets:
        logger.warning("No facets in database to canonicalize.")
        return

    logger.info("Canonicalizing %d total database facets...", len(db_facets))
    canonicalizer = FacetCanonicalizer(
        embedding_model_name=cfg["models"].get("facet_embed_model", "all-MiniLM-L6-v2"),
        distance_threshold=args.distance_threshold,
        n_neighbours=fcfg.get("canon_neighbours", 64),
    )
    canon_start = time.time()
    canonicalized_facets, canonical_defs = canonicalizer.canonicalize(db_facets)
    canon_time = time.time() - canon_start
    logger.info("Canonicalization complete in %.2fs: %d canonical definitions created.", canon_time, len(canonical_defs))

    # 3. Update canonical_id in database and save canonical clusters
    logger.info("Updating canonical IDs in SQLite database...")
    store.save_canonical_clusters(canonical_defs, clear_existing=True)
    conn = sqlite3.connect(str(args.db_path))
    with conn:
        conn.executemany(
            "UPDATE paper_facets SET canonical_id = ? WHERE facet_id = ?",
            [(f["canonical_id"], f["facet_id"]) for f in canonicalized_facets],
        )
    conn.close()

    # 4. Summary and manifest
    facet_counts_by_type = dict(Counter(f["facet_type"] for f in canonicalized_facets))
    canon_by_type = dict(Counter(c["facet_type"] for c in canonical_defs))
    print("\n" + "=" * 60)
    print("PHASE 3 FACET EXTRACTION & CANONICALIZATION SUMMARY")
    print("=" * 60)
    print(f"Total facets in database: {len(canonicalized_facets)}")
    print(f"Total canonical clusters: {len(canonical_defs)}")
    print("\nFacet distribution by type:")
    for ftype, count in sorted(facet_counts_by_type.items()):
        print(f"  - {ftype:10s}: {count:6d} facets -> {canon_by_type.get(ftype, 0):5d} canonical clusters")
    print("=" * 60 + "\n")

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cutoff_year": cutoff,
        "min_abstract_chars": min_chars,
        "gliner_model": args.model,
        "labels": fcfg.get("labels"),
        "gliner_threshold": args.threshold,
        "sentence_splitter": "paperfeed.facets.extractor.split_sentences",
        "canonicalizer_model": cfg["models"].get("facet_embed_model"),
        "canon_distance_threshold": args.distance_threshold,
        "canon_neighbours": fcfg.get("canon_neighbours", 64),
        "papers_processed_total": store.count_processed(),
        "facets_total": len(canonicalized_facets),
        "canonical_clusters_total": len(canonical_defs),
        "facets_by_type": facet_counts_by_type,
        "canonical_by_type": canon_by_type,
        # Extraction timing accumulates across resumed runs; canonicalization is redone each run.
        "extraction_time_seconds": round(extraction_seconds, 2),
        "papers_extracted_timed": papers_timed,
        "extraction_sec_per_1k": round(extraction_seconds / papers_timed * 1000, 2) if papers_timed else None,
        "canonicalization_time_seconds": round(canon_time, 2),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote facet manifest to %s", args.manifest)


if __name__ == "__main__":
    main()
