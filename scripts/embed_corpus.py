"""
Embedding & Vector Indexing Script (Phase 5):
Encodes 3,901 real scientific papers from data/corpus.jsonl into dense vectors.
Leverages CUDA GPU acceleration if available, falling back to multi-threaded CPU.
Saves:
  - data/embeddings.npy
  - data/paper_ids.json
"""

import os
import json
import time
import numpy as np

def main():
    corpus_path = "data/corpus.jsonl"
    emb_path = "data/embeddings.npy"
    id_map_path = "data/paper_ids.json"

    if not os.path.exists(corpus_path):
        print(f"Error: {corpus_path} not found. Run scripts/fetch_corpus.py first.")
        return

    print("=" * 70)
    print("STARTING REAL CORPUS EMBEDDING PIPELINE (PHASE 5)")
    print("=" * 70)

    # 1. Load corpus
    papers = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                papers.append(json.loads(line))

    print(f"[+] Loaded {len(papers)} papers from {corpus_path}")

    # 2. Format input texts
    texts = [f"Title: {p['title']} | Abstract: {p['abstract']}" for p in papers]
    paper_ids = [p["paper_id"] for p in papers]

    # 3. Import sentence-transformers & PyTorch
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] PyTorch Version: {torch.__version__}")
    print(f"[+] Execution Device: {device.upper()}")
    if device == "cuda":
        print(f"    --> GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"    --> Allocated Memory: {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB")

    # 4. Load Model (BAAI/bge-small-en-v1.5: fast, highly accurate, MTEB leader)
    model_name = "BAAI/bge-small-en-v1.5"
    print(f"[+] Loading Embedding Model: '{model_name}'...")
    start_time = time.time()
    model = SentenceTransformer(model_name, device=device)

    # 5. Encode in batches
    batch_size = 64 if device == "cuda" else 32
    print(f"[+] Encoding {len(texts)} paper abstracts (batch_size={batch_size})...")
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # Unit L2 norm for exact cosine similarity via dot product
        convert_to_numpy=True
    )
    
    elapsed = time.time() - start_time
    print(f"[+] Encoding complete in {elapsed:.2f} seconds ({len(texts)/elapsed:.1f} papers/sec)!")
    print(f"[+] Embedding Matrix Shape: {embeddings.shape} (dtype: {embeddings.dtype})")

    # 6. Save vectors and index mapping
    np.save(emb_path, embeddings.astype(np.float32))
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(paper_ids, f)

    print(f"[+] Saved embeddings to {emb_path}")
    print(f"[+] Saved paper ID map to {id_map_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
