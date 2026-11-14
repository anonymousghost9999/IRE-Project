"""V1 encoder study on the Track 1 tuning split (plan 10.1c / docs/design.md §13.8 V1 points 4-6).

1. ``graph_subset``: an encoder-neutral 20k-paper subset around the tuning authors, chosen from the <= T citation graph
   only (never from targets or embeddings).
2. ``subset_resources``: the recommender's resources restricted to the subset, with one encoder's vectors, a dense
   index over them and the shared subset BM25s. Facets do not depend on the encoder and are reused.
3. ``evaluate_arm``: the §9 grid on the 80 tuning authors, plus the Dense Centroid baseline and candidate Recall@100.
4. ``decide_encoder``: the better of SciNCL and BGE at its optimum vs SPECTER2 at its optimum; adopted only if the
   paired-bootstrap 95% CI of the NDCG@10 gain lies above zero.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import clean_id
from paperfeed.data.ground_truth import to_uri
from paperfeed.eval.metrics import ndcg_at_k
from paperfeed.eval.tuning import (
    CachedUserCandidateData,
    GridPoint,
    _compare_to_core,
    _point_record,
    best_grid_point,
    grid_per_user_ndcg,
    precompute_tuning_cache,
)
from paperfeed.index.dense import DenseIndex
from paperfeed.index.lexical import LexicalIndex
from paperfeed.index.retriever import HybridRetriever
from paperfeed.index.store import EmbeddingStore
from paperfeed.rec.features import edges

logger = get_logger("eval.encoder_study")

SUBSET_SIZE = 20000


# ---------------------------------------------------------------------------
# 1. The subset (design §13.8 V1 point 4)
# ---------------------------------------------------------------------------

def _link_counts(db, ids: Sequence[str], cutoff_year: int) -> Counter:
    """For every paper one citation hop from ``ids`` (references and citing papers, citing year <= T): how many links
    it has to ``ids``."""
    ids = sorted(set(ids))
    counts: Counter = Counter()
    for _src, tgt in edges(db, "source_paper_id", ids, cutoff_year):  # what the ids cite
        counts[tgt] += 1
    for src, _tgt in edges(db, "target_paper_id", ids, cutoff_year):  # who cites the ids
        counts[src] += 1
    return counts


def graph_subset(
    db,
    seed_ids: Iterable[str],
    allowed: Set[str],
    cutoff_year: int,
    size: int = SUBSET_SIZE,
) -> Tuple[List[str], Dict[str, Any]]:
    """The seeds (in ``allowed``) plus one-hop neighbours by links to the seeds, then two-hop neighbours by links to
    seeds and one-hop papers, up to ``size`` papers in total. ``allowed``: papers that can be in the subset (eligible
    papers with a stored vector, plus the seeds). Ties go to the smaller paper ID. Deterministic."""
    seeds = sorted({clean_id(s) for s in seed_ids} & allowed)
    if len(seeds) > size:
        raise ValueError(f"{len(seeds)} seed papers do not fit in a subset of {size}.")
    chosen: List[str] = list(seeds)
    taken = set(chosen)

    hop1 = _link_counts(db, seeds, cutoff_year)
    ranked1 = sorted((p for p in hop1 if p in allowed and p not in taken), key=lambda p: (-hop1[p], p))
    room = size - len(chosen)
    chosen += ranked1[:room]
    taken.update(ranked1[:room])
    n_hop1 = min(len(ranked1), room)

    n_hop2 = 0
    if len(chosen) < size:
        hop2 = _link_counts(db, chosen, cutoff_year)
        ranked2 = sorted((p for p in hop2 if p in allowed and p not in taken), key=lambda p: (-hop2[p], p))
        add = ranked2[: size - len(chosen)]
        chosen += add
        n_hop2 = len(add)
    info = {"size": len(chosen), "n_seeds": len(seeds), "n_hop1": n_hop1, "n_hop1_available": len(ranked1),
            "n_hop2": n_hop2, "rule": "design §13.8 V1 point 4"}
    return chosen, info


def subset_fingerprint(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 2. Per-encoder resources on the subset
# ---------------------------------------------------------------------------

def paper_texts(db, ids: Sequence[str], chunk: int = 2000) -> Dict[str, Dict[str, str]]:
    """Title and abstract of every paper ("" when missing). Fetched in chunks: one query for 20k IDs would exceed
    SQLite's limit on bound parameters."""
    ids = list(ids)
    rows: List[Dict[str, Any]] = []
    for i in range(0, len(ids), chunk):
        rows += db.get_papers_batch([to_uri(p) for p in ids[i : i + chunk]])
    out = {clean_id(r["paper_id"]): {"title": r.get("title") or "", "abstract": r.get("abstract") or ""} for r in rows}
    return {p: out.get(p, {"title": "", "abstract": ""}) for p in ids}


def build_subset_lexical(texts: Mapping[str, Mapping[str, str]]) -> LexicalIndex:
    """BM25s over "title abstract", the main index's document text, for the subset (shared by every encoder)."""
    ids = sorted(texts)
    lex = LexicalIndex(stopwords="en")
    lex.index_documents(ids, [f"{texts[i]['title']} {texts[i]['abstract']}".strip() for i in ids], show_progress=False)
    return lex


def embed_subset(
    encoder,
    model_name: str,
    texts: Mapping[str, Mapping[str, str]],
    cache_dir: Path,
    batch_size: int = 32,
    checkpoint_every: int = 5000,
) -> EmbeddingStore:
    """The subset's vectors for one encoder, checkpointed in ``cache_dir`` and resumed when the subset and model match."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb, ids_path, man = cache_dir / "embeddings.npy", cache_dir / "embedding_ids.json", cache_dir / "manifest.json"
    ids = sorted(texts)
    want = {"model": model_name, "subset_sha256": subset_fingerprint(ids)}
    if man.exists() and json.loads(man.read_text(encoding="utf-8")) != want:
        logger.warning("%s: subset or model changed; discarding the cached vectors.", cache_dir)
        emb.unlink(missing_ok=True)
        ids_path.unlink(missing_ok=True)
    man.write_text(json.dumps(want), encoding="utf-8")
    store = EmbeddingStore.load(emb, ids_path)
    todo = store.missing(ids)
    for start in range(0, len(todo), checkpoint_every):
        chunk = todo[start : start + checkpoint_every]
        vecs = encoder.embed_papers([texts[p] for p in chunk], batch_size=batch_size, show_progress=True)
        store.add(chunk, np.asarray(vecs, dtype=np.float32))
        store.save(emb, ids_path)
        logger.info("%s: embedded %d/%d subset papers.", model_name, min(start + checkpoint_every, len(todo)), len(todo))
    return EmbeddingStore(ids, store.select(ids))


def subset_resources(base, store: EmbeddingStore, lexical: LexicalIndex, encoder, subset_ids: Sequence[str]):
    """``base`` (RecommenderResources) restricted to the subset, with this encoder's vectors and dense index."""
    ids = list(subset_ids)
    vecs = store.select(ids)
    dense = DenseIndex(dimension=vecs.shape[1])
    dense.add_documents(ids, vecs)
    r = base.retriever
    retriever = HybridRetriever(dense, lexical, channel_depth=r.channel_depth, rrf_k=r.rrf_k,
                                candidate_pool_size=r.candidate_pool_size)
    return replace(base, store=store, retriever=retriever, eligible=set(base.eligible) & set(ids), encoder=encoder)


# ---------------------------------------------------------------------------
# 3. One encoder's tuning-split numbers
# ---------------------------------------------------------------------------

def evaluate_arm(resources, tuning_authors: Sequence[Mapping[str, Any]], discovery: Mapping[str, Sequence[str]],
                 config, k: int = 10) -> Dict[str, Any]:
    """Tuning cache, §9 grid, the Dense Centroid baseline and candidate Recall@100 for one encoder."""
    from paperfeed.eval.baselines import dense_centroid_baseline

    cached = precompute_tuning_cache(resources, tuning_authors, discovery, config)
    if len(cached) != len(tuning_authors):
        raise RuntimeError(f"Only {len(cached)} of {len(tuning_authors)} tuning authors could be cached.")
    points = grid_per_user_ndcg(cached, resources, adaptive_graph=config.adaptive_graph, base_weights=config.weights)
    centroid = np.array([
        ndcg_at_k(dense_centroid_baseline(resources, u.profile, k=k), u.targets, k=k) for u in cached
    ])
    return {"cached": cached, "points": points, "dense_centroid": centroid}


# ---------------------------------------------------------------------------
# 4. Decision (design §13.8 V1 point 6)
# ---------------------------------------------------------------------------

def decide_encoder(
    arms: Mapping[str, Mapping[str, Any]],
    reference: str = "specter2",
    n_resamples: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """The alternative with the higher full-system tuning NDCG@10 at its optimum (ties: the first listed) is the V1
    candidate; it replaces ``reference`` only if the paired-bootstrap 95% CI of its gain lies above zero."""
    if reference not in arms or len(arms) < 2:
        raise ValueError("Need the reference encoder and at least one alternative.")
    alternatives = [k for k in arms if k != reference]
    per_arm = {}
    for key, arm in arms.items():
        best = best_grid_point(arm["points"])
        per_arm[key] = {
            **_point_record(best),
            "dense_centroid_ndcg": float(np.mean(arm["dense_centroid"])),
            "candidate_recall_at_100": float(np.mean([
                len(set(u.targets) & set(u.candidate_ids)) / len(u.targets) if u.targets else 0.0 for u in arm["cached"]
            ])),
        }
    chosen = max(alternatives, key=lambda k: (per_arm[k]["mean_ndcg"], -alternatives.index(k)))
    cmp = _compare_to_core(arms[reference]["points"], arms[chosen]["points"], arms[reference]["cached"],
                           arms[chosen]["cached"], n_resamples, seed, alpha)
    from paperfeed.eval.stats import paired_bootstrap_test

    centroid = {}
    for key in alternatives:
        d, lo, hi, _, _ = paired_bootstrap_test(arms[key]["dense_centroid"], arms[reference]["dense_centroid"],
                                                n_resamples=n_resamples, alpha=alpha, seed=seed)
        centroid[key] = {"mean_diff": d, "ci": [lo, hi]}
    return {
        "rule": {
            "statement": "The alternative encoder with the higher full-system tuning NDCG@10 at its own grid optimum is "
            "the V1 candidate; it replaces SPECTER2 only if the paired-bootstrap 95% CI of its gain over SPECTER2 (both "
            "on the 20k subset, each at its own optimum) lies above zero (design §13.8, V1 point 6).",
            "n_resamples": n_resamples,
            "alpha": alpha,
            "seed": seed,
        },
        "n_users": cmp["n_users"],
        "per_encoder": per_arm,
        "chosen_encoder": chosen,
        "ndcg_gain": cmp["ndcg_gain"],
        "candidate_recall_gain": cmp["candidate_recall_gain"],
        "dense_centroid_gain": centroid,
        "adopt_v1": cmp["adopt"],
    }


# ---------------------------------------------------------------------------
# 5. Full-corpus index for the V1 candidate (plan 10.1d)
# ---------------------------------------------------------------------------

def build_full_encoder_index(
    reference_ids: Sequence[str],
    db,
    encoder,
    model_name: str,
    out_dir: Path,
    batch_size: int = 32,
    checkpoint_every: int = 10000,
) -> Dict[str, Any]:
    """Embed exactly the papers of the SPECTER2 store (``reference_ids``) with ``encoder`` and build its FAISS index in
    ``out_dir`` (embedding resumes from checkpoints). BM25s and facets are reused from the Core index. The manifest is
    marked complete only at the end, so a half-built index is never loaded (RecommenderResources.load checks it)."""
    import time

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "index_manifest.json"
    ids = sorted({clean_id(p) for p in reference_ids})
    manifest = {"model": model_name, "status": "embedding", "n_docs": len(ids), "ids_sha256": subset_fingerprint(ids)}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    t0 = time.time()
    texts = paper_texts(db, ids)
    store = embed_subset(encoder, model_name, texts, out_dir, batch_size, checkpoint_every)
    t_embed = time.time() - t0
    vecs = store.select(ids)
    dense = DenseIndex(dimension=vecs.shape[1])
    dense.add_documents(ids, vecs)
    dense.save(out_dir / "faiss.index", out_dir / "paper_ids.json")
    manifest.update({"status": "complete", "embedding_dim": int(vecs.shape[1]), "build_seconds": round(t_embed, 1),
                     "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "bm25s": "reused from the Core index", "same_papers_as": "the SPECTER2 embedding store"})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
