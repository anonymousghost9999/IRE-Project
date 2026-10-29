"""Build dense (FAISS) and lexical (BM25s) indices over the candidate corpus (Phase 2).

Usage:
    python scripts/build_index.py [--db-path data/processed/paperfeed.sqlite3] [--out-dir data/processed]

Generates:
    - data/processed/embeddings.npy + embedding_ids.json: the embedding store (L2-normalized SPECTER2 vectors and
      the paper ID of each row). Only papers without a stored vector are embedded, and the store is checkpointed
      during embedding, so an interrupted or repeated run resumes instead of starting over.
    - data/processed/faiss.index: FAISS IndexFlatIP index
    - data/processed/paper_ids.json: Internal index -> paper_id mapping
    - data/processed/bm25s/: Serialized BM25s lexical index
    - data/processed/index_manifest.json: Indexing metadata and timing statistics
"""

import argparse
import json
import time
from pathlib import Path
import sys
from typing import Any, Dict, List
import numpy as np

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paperfeed.common.config import find_project_root, load_config
from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import load_candidate_authors
from paperfeed.data.db import DatabaseManager
from paperfeed.data.leakage import check_paper_ids
from paperfeed.index.dense import DenseIndex, clean_id
from paperfeed.index.embeddings import SPECTER2Encoder
from paperfeed.index.lexical import LexicalIndex
from paperfeed.index.store import EmbeddingStore, ids_sha256

logger = get_logger("scripts.build_index")


def load_candidate_papers(db: DatabaseManager, cutoff_year: int, limit: int = 0, min_abstract_chars: int = 1) -> List[Dict[str, Any]]:
    """Fetch all candidate papers with usable abstracts up to cutoff year."""
    query = """
        SELECT paper_id, title, abstract, year
        FROM papers
        WHERE in_candidates = 1
          AND year <= ?
          AND abstract IS NOT NULL
          AND length(abstract) >= ?
        ORDER BY paper_id ASC
    """
    if limit > 0:
        query += f" LIMIT {limit}"

    papers = []
    with db.get_connection() as conn:
        cursor = conn.execute(query, (cutoff_year, min_abstract_chars))
        for row in cursor.fetchall():
            papers.append({
                "paper_id": clean_id(row[0]),
                "title": row[1] or "",
                "abstract": row[2] or "",
                "year": row[3],
            })
    return papers


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SPECTER2 and BM25s indices.")
    parser.add_argument("--db-path", default="data/processed/paperfeed.sqlite3")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="Test limit on paper count (0 = all)")
    parser.add_argument("--skip-embeddings", action="store_true", help="Never embed: use the stored vectors, fail if a paper has none")
    args = parser.parse_args()

    root = find_project_root()
    cfg = load_config()
    cutoff = cfg["project"]["cutoff_year"]

    db_path = root / args.db_path
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        logger.error(f"Database file not found: {db_path}")
        return 1

    db = DatabaseManager(db_path=db_path, cutoff_year=cutoff)
    min_chars = cfg.get("index", {}).get("min_abstract_chars", 1)
    logger.info(f"Loading candidate papers from {db_path} (cutoff={cutoff}, abstract >= {min_chars} chars)...")
    papers = load_candidate_papers(db, cutoff, limit=args.limit, min_abstract_chars=min_chars)
    total_papers = len(papers)
    logger.info(f"Found {total_papers} candidate papers eligible for indexing.")

    if total_papers == 0:
        logger.warning("No candidate papers found. Exiting.")
        return 0

    paper_ids = [p["paper_id"] for p in papers]

    # Leakage gate: nothing after T and none of the evaluation target papers may be indexed.
    candidates_path = root / "data" / "candidate_authors.jsonl"
    forbidden = [c.target_paper_b_id for c in load_candidate_authors(candidates_path)] if candidates_path.exists() else []
    if not forbidden:
        logger.warning(f"{candidates_path} not found: checking years only, not target papers.")
    leakage = check_paper_ids(db, paper_ids, cutoff, forbidden)
    if not leakage.clean:
        logger.error(f"LEAKAGE CHECK FAILED on index contents: {leakage.counts}")
        return 1

    embeddings_path = out_dir / "embeddings.npy"
    embedding_ids_path = out_dir / "embedding_ids.json"
    faiss_path = out_dir / "faiss.index"
    id_map_path = out_dir / "paper_ids.json"
    bm25s_dir = out_dir / "bm25s"
    manifest_path = out_dir / "index_manifest.json"

    old_manifest: Dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest: Dict[str, Any] = {
        "cutoff_year": cutoff,
        "total_papers": total_papers,
        "min_abstract_chars": min_chars,
        "paper_ids_sha256": ids_sha256(paper_ids),
        "dense_encoder": cfg["models"]["dense_encoder"],
        "dense_adapter": cfg["models"]["dense_adapter"],
        "query_adapter": cfg["models"]["query_adapter"],
        "embedding_dim": 768,
        "leakage_check_clean": True,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # Embedding timing accumulates across runs; the first build embedded every paper it indexed.
        "embedding_time_seconds": old_manifest.get("embedding_time_seconds", 0.0),
        "embedded_papers_timed": old_manifest.get("embedded_papers_timed", old_manifest.get("total_papers", 0) if "embedding_time_seconds" in old_manifest else 0),
        "embedding_device": old_manifest.get("embedding_device"),
    }

    # 1. Embed the papers that have no stored vector yet (checkpointing as we go)
    store = EmbeddingStore.load(embeddings_path, embedding_ids_path, legacy_ids_path=id_map_path)
    todo_ids = store.missing(paper_ids)
    logger.info(f"Embedding store holds {len(store)} vectors; {len(todo_ids)} of {total_papers} papers still need one.")
    if todo_ids and args.skip_embeddings:
        logger.error(f"--skip-embeddings given but {len(todo_ids)} papers have no stored embedding.")
        return 1
    if todo_ids:
        logger.info("Initializing SPECTER2 encoder (fp16 on CUDA if available)...")
        encoder = SPECTER2Encoder(
            base_model_name=cfg["models"]["dense_encoder"],
            dense_adapter_name=cfg["models"]["dense_adapter"],
            query_adapter_name=cfg["models"]["query_adapter"],
        )
        by_id = {p["paper_id"]: p for p in papers}
        every = cfg.get("index", {}).get("embed_checkpoint_every", 10000)
        t_embed = 0.0
        for start in range(0, len(todo_ids), every):
            chunk_ids = todo_ids[start : start + every]
            t0 = time.time()
            vecs = encoder.embed_papers([by_id[i] for i in chunk_ids], batch_size=args.batch_size, show_progress=True)
            t_embed += time.time() - t0
            store.add(chunk_ids, vecs)
            store.save(embeddings_path, embedding_ids_path)
            logger.info(f"Embedded {min(start + every, len(todo_ids))}/{len(todo_ids)}; store checkpointed.")
        manifest["embedding_time_seconds"] = round(manifest["embedding_time_seconds"] + t_embed, 2)
        manifest["embedded_papers_timed"] += len(todo_ids)
        manifest["embedding_device"] = str(encoder.device)
    if not embedding_ids_path.exists():
        store.save(embeddings_path, embedding_ids_path)  # legacy store: persist the adopted row IDs
    timed = manifest["embedded_papers_timed"]
    manifest["embedding_sec_per_1k"] = round(manifest["embedding_time_seconds"] / timed * 1000, 2) if timed else None
    embeddings = store.select(paper_ids)

    # 2. Build and save FAISS IndexFlatIP
    logger.info("Building FAISS dense index...")
    t0 = time.time()
    dense_index = DenseIndex(dimension=embeddings.shape[1])
    dense_index.add_documents(paper_ids, embeddings)
    dense_index.save(faiss_path, id_map_path)
    t_faiss = time.time() - t0
    logger.info(f"FAISS index built and saved in {t_faiss:.2f}s.")
    manifest["faiss_build_time_seconds"] = round(t_faiss, 2)

    # 3. Build and save BM25s lexical index
    logger.info("Building BM25s lexical index...")
    t0 = time.time()
    lexical_index = LexicalIndex(stopwords="en")
    paper_texts = [f"{p['title']} {p['abstract']}" for p in papers]
    lexical_index.index_documents(paper_ids, paper_texts, show_progress=True)
    lexical_index.save(bm25s_dir)
    t_bm25 = time.time() - t0
    logger.info(f"BM25s index built and saved in {t_bm25:.2f}s.")
    manifest["bm25s_build_time_seconds"] = round(t_bm25, 2)

    # Write manifest
    manifest_path = out_dir / "index_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved index manifest to {manifest_path}")

    logger.info("All Phase 2 indices successfully built!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

