"""Build dense (FAISS) and lexical (BM25s) indices over the candidate corpus (Phase 2).

Usage:
    python scripts/build_index.py [--db-path data/processed/paperfeed.sqlite3] [--out-dir data/processed]

Generates:
    - data/processed/embeddings.npy: Float32 matrix of L2-normalized SPECTER2 embeddings
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
from paperfeed.data.db import DatabaseManager
from paperfeed.index.dense import DenseIndex, clean_id
from paperfeed.index.embeddings import SPECTER2Encoder
from paperfeed.index.lexical import LexicalIndex

logger = get_logger("scripts.build_index")


def load_candidate_papers(db: DatabaseManager, cutoff_year: int, limit: int = 0) -> List[Dict[str, Any]]:
    """Fetch all candidate papers with abstracts up to cutoff year."""
    query = """
        SELECT paper_id, title, abstract, year
        FROM papers
        WHERE in_candidates = 1
          AND year <= ?
          AND abstract IS NOT NULL
          AND length(abstract) > 0
        ORDER BY paper_id ASC
    """
    if limit > 0:
        query += f" LIMIT {limit}"

    papers = []
    with db.get_connection() as conn:
        cursor = conn.execute(query, (cutoff_year,))
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
    parser.add_argument("--skip-embeddings", action="store_true", help="Reuse existing embeddings.npy")
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
    logger.info(f"Loading candidate papers from {db_path} (cutoff={cutoff})...")
    papers = load_candidate_papers(db, cutoff, limit=args.limit)
    total_papers = len(papers)
    logger.info(f"Found {total_papers} candidate papers eligible for indexing.")

    if total_papers == 0:
        logger.warning("No candidate papers found. Exiting.")
        return 0

    paper_ids = [p["paper_id"] for p in papers]
    embeddings_path = out_dir / "embeddings.npy"
    faiss_path = out_dir / "faiss.index"
    id_map_path = out_dir / "paper_ids.json"
    bm25s_dir = out_dir / "bm25s"

    manifest: Dict[str, Any] = {
        "cutoff_year": cutoff,
        "total_papers": total_papers,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # 1. Compute or load SPECTER2 embeddings
    embeddings: np.ndarray
    if args.skip_embeddings and embeddings_path.exists():
        logger.info(f"Loading precomputed embeddings from {embeddings_path}...")
        embeddings = np.load(embeddings_path)
        manifest["embedding_source"] = "reused"
    else:
        logger.info(f"Initializing SPECTER2 encoder (fp16 on CUDA if available)...")
        encoder = SPECTER2Encoder(
            base_model_name=cfg["models"]["dense_encoder"],
            dense_adapter_name=cfg["models"]["dense_adapter"],
            query_adapter_name=cfg["models"]["query_adapter"],
        )
        t0 = time.time()
        logger.info(f"Embedding {total_papers} papers in batches of {args.batch_size}...")
        embeddings = encoder.embed_papers(papers, batch_size=args.batch_size, show_progress=True)
        t_embed = time.time() - t0
        rate = (t_embed / total_papers) * 1000 if total_papers > 0 else 0
        logger.info(f"Finished embedding {total_papers} papers in {t_embed:.1f}s ({rate:.2f}s per 1k papers).")
        
        np.save(embeddings_path, embeddings)
        logger.info(f"Saved embeddings to {embeddings_path}")
        manifest["embedding_time_seconds"] = round(t_embed, 2)
        manifest["embedding_sec_per_1k"] = round(rate, 2)
        manifest["embedding_device"] = str(encoder.device)

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

