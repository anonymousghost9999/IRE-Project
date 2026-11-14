"""Phase 11 comparison systems (docs/design.md §13.9 "Phase 11 specification" point 4).

RP3β (plan 11b) on the <= T citation graph, as a bipartite citing-paper x cited-paper graph A (A[c, i] = 1 when paper c
cites paper i, citing year <= T). The author is linked to the references of their authored papers (R_u). The score of
item i is the three-step walk author -> reference j -> citing paper c -> i, each transition probability raised to
alpha, divided by popularity(i)^beta, with popularity = number of citing papers:

    s(i) = sum_{j in R_u} sum_c  (1/|R_u|)^a * (A[c,j] / indeg(j))^a * (A[c,i] / outdeg(c))^a  /  indeg(i)^b

The first factor is the same for every item of an author, so it does not change the ranking and is dropped. beta = 0
is P3α. Items are restricted to eligible papers and E_u is removed, exactly as for every other system.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import clean_id
from paperfeed.eval.metrics import ndcg_at_k

logger = get_logger("eval.sota")

RP3_GRID = {"alpha": [0.5, 1.0, 1.5], "beta": [0.0, 0.25, 0.5, 0.75]}  # design §13.9 Phase 11 spec point 4


@dataclass
class CitationGraph:
    """Sparse citing x cited matrix over every paper in the <= T edges (CSR, and its transpose)."""

    ids: List[str]  # node index -> bare paper ID (citing and cited papers share one index)
    index: Dict[str, int]
    A: object  # scipy.sparse.csr_matrix, shape (n, n): A[c, i] = 1 when c cites i
    AT: object  # A transposed, CSR
    indeg: np.ndarray  # number of papers citing each node
    outdeg: np.ndarray  # number of references of each node

    @classmethod
    def from_edges(cls, edges: Iterable[Tuple[str, str]]) -> "CitationGraph":
        from scipy.sparse import csr_matrix

        index: Dict[str, int] = {}
        rows: List[int] = []
        cols: List[int] = []
        for src, tgt in edges:
            s, t = clean_id(src), clean_id(tgt)
            if s == t:
                continue
            rows.append(index.setdefault(s, len(index)))
            cols.append(index.setdefault(t, len(index)))
        n = len(index)
        A = csr_matrix((np.ones(len(rows), dtype=np.float32), (np.asarray(rows), np.asarray(cols))), shape=(n, n))
        A.data[:] = 1.0  # a duplicated edge still counts once
        A.sum_duplicates()
        A.data[:] = 1.0
        ids = [""] * n
        for pid, i in index.items():
            ids[i] = pid
        return cls(ids, index, A, A.T.tocsr(), np.asarray(A.sum(axis=0)).ravel(), np.asarray(A.sum(axis=1)).ravel())

    @classmethod
    def from_db(cls, db, cutoff_year: int, chunk: int = 500000) -> "CitationGraph":
        def stream():
            with db.get_connection() as conn:
                cur = conn.execute("SELECT source_paper_id, target_paper_id FROM paper_citations WHERE source_year <= ?",
                                   (cutoff_year,))
                while True:
                    rows = cur.fetchmany(chunk)
                    if not rows:
                        return
                    yield from rows

        graph = cls.from_edges(stream())
        logger.info("Citation graph: %d papers, %d edges (<= %d).", len(graph.ids), graph.A.nnz, cutoff_year)
        return graph

    def references(self, paper_ids: Iterable[str]) -> List[int]:
        """Node indices of everything the given papers cite (R_u for an author's authored papers)."""
        rows = [self.index[p] for p in {clean_id(x) for x in paper_ids} if p in self.index]
        if not rows:
            return []
        return sorted(set(self.A[rows].indices.tolist()))

    def walk_scores(self, profile: Sequence[int], alpha: float) -> np.ndarray:
        """Unnormalised P3 scores (before the popularity division) for every node, for one alpha."""
        n = len(self.ids)
        if not profile:
            return np.zeros(n)
        prof = np.asarray(profile)
        with np.errstate(divide="ignore"):
            w_j = np.where(self.indeg[prof] > 0, self.indeg[prof] ** -alpha, 0.0)  # (A[c,j]/indeg(j))^a for A=1
        v = self.A[:, prof] @ w_j  # per citing paper c: sum over j in R_u that c cites
        with np.errstate(divide="ignore"):
            v = v * np.where(self.outdeg > 0, self.outdeg ** -alpha, 0.0)  # (A[c,i]/outdeg(c))^a for A=1
        return np.asarray(self.AT @ v).ravel()

    def rank(self, scores: np.ndarray, beta: float, allowed: Set[str], excluded: Set[str], k: int) -> List[str]:
        """Top-k allowed, non-excluded items by score / indeg^beta (only items with a positive score); ties by ID."""
        cand = np.flatnonzero(scores > 0)
        if cand.size == 0:
            return []
        pop = self.indeg[cand]
        s = scores[cand] / np.where(pop > 0, pop, 1.0) ** beta
        order = sorted(range(cand.size), key=lambda t: (-s[t], self.ids[cand[t]]))
        out: List[str] = []
        for t in order:
            pid = self.ids[cand[t]]
            if pid in allowed and pid not in excluded:
                out.append(pid)
                if len(out) == k:
                    break
        return out


def rp3beta_recommend(graph: CitationGraph, authored_ids: Sequence[str], allowed: Set[str], excluded: Set[str],
                      alpha: float, beta: float, k: int = 10) -> List[str]:
    """RP3β top-k for one author (empty for an author without references in the graph, e.g. zero-publication)."""
    return graph.rank(graph.walk_scores(graph.references(authored_ids), alpha), beta, allowed, excluded, k)


@dataclass
class RP3Author:
    author_id: str
    authored_ids: List[str]
    excluded: Set[str]
    targets: List[str]


def tune_rp3beta(graph: CitationGraph, authors: Sequence[RP3Author], allowed: Set[str],
                 grid: Optional[Mapping[str, Sequence[float]]] = None, k: int = 10) -> Dict[str, object]:
    """Mean NDCG@k on ``authors`` for every (alpha, beta); the best point wins, ties going to the earlier grid point."""
    g = dict(grid or RP3_GRID)
    per_point: Dict[Tuple[float, float], List[float]] = {(a, b): [] for a, b in itertools.product(g["alpha"], g["beta"])}
    for au in authors:
        profile = graph.references(au.authored_ids)
        for a in g["alpha"]:
            scores = graph.walk_scores(profile, a)
            for b in g["beta"]:
                per_point[(a, b)].append(ndcg_at_k(graph.rank(scores, b, allowed, au.excluded, k), au.targets, k=k))
    records = [{"alpha": a, "beta": b, "mean_ndcg": float(np.mean(v)) if v else 0.0} for (a, b), v in per_point.items()]
    best = max(records, key=lambda r: r["mean_ndcg"])  # max keeps the first of equal values (grid order)
    return {"grid": g, "points": records, "best": best, "n_users": len(authors)}


# ---------------------------------------------------------------------------
# Scholar Inbox-style per-author classifier (plan 11c; design §13.9 Phase 11 spec point 4)
# ---------------------------------------------------------------------------

SCHOLAR_INBOX_C = [0.01, 0.1, 1.0, 10.0]
NEGATIVES_PER_POSITIVE = 10
PCA_DIM = 256


@dataclass
class EmbeddingSpace:
    """Corpus vectors reduced by PCA (fitted once on the whole corpus, as Scholar Inbox does)."""

    ids: List[str]
    index: Dict[str, int]
    Z: np.ndarray  # (n, dim) float32
    explained_variance: float


def fit_pca_space(ids: Sequence[str], vectors: np.ndarray, dim: int = PCA_DIM, seed: int = 42) -> EmbeddingSpace:
    from sklearn.decomposition import PCA

    vectors = np.asarray(vectors, dtype=np.float32)
    dim = min(dim, vectors.shape[0], vectors.shape[1])
    pca = PCA(n_components=dim, random_state=seed, svd_solver="randomized" if vectors.shape[0] > 5000 else "full")
    Z = pca.fit_transform(vectors).astype(np.float32)
    ids = [clean_id(p) for p in ids]
    return EmbeddingSpace(ids, {p: i for i, p in enumerate(ids)}, Z, float(pca.explained_variance_ratio_.sum()))


def author_seed(author_id: str, base: int = 42) -> int:
    import zlib

    return (zlib.crc32(clean_id(author_id).encode("utf-8")) + base) % (2**31)


def scholar_inbox_recommend(space: EmbeddingSpace, author_id: str, excluded: Set[str], allowed_idx: np.ndarray,
                            C: float, k: int = 10) -> List[str]:
    """Train a logistic regression on the author's positives (E_u: authored papers and their references with a vector)
    against 10 random eligible negatives per positive, then rank the eligible, non-excluded papers by its score.
    Empty when the author has no positive with a vector (not applicable, like Citation-only for zero-publication)."""
    from sklearn.linear_model import LogisticRegression

    pos = np.array(sorted(space.index[p] for p in excluded if p in space.index), dtype=np.int64)
    if pos.size == 0:
        return []
    excl_idx = {space.index[p] for p in excluded if p in space.index}
    pool = np.array([i for i in allowed_idx if i not in excl_idx], dtype=np.int64)
    if pool.size == 0:
        return []
    rng = np.random.default_rng(author_seed(author_id))
    neg = rng.choice(pool, size=min(pool.size, NEGATIVES_PER_POSITIVE * pos.size), replace=False)
    X = np.vstack([space.Z[pos], space.Z[neg]])
    y = np.r_[np.ones(pos.size), np.zeros(neg.size)]
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(X, y)
    scores = clf.decision_function(space.Z[pool])
    order = sorted(range(pool.size), key=lambda t: (-scores[t], space.ids[pool[t]]))[:k]
    return [space.ids[pool[t]] for t in order]


# ---------------------------------------------------------------------------
# User-centroid kNN with another encoder (SciNCL; plan 11c)
# ---------------------------------------------------------------------------

def centroid_recommend(store, dense_index, authored_ids: Sequence[str], excluded: Set[str], allowed: Set[str],
                       k: int = 10, encoder=None, texts: Optional[Mapping[str, Mapping[str, str]]] = None) -> List[str]:
    """Dense kNN on the mean vector of the author's papers (the Dense Centroid baseline with another encoder).
    Authored papers without a stored vector are embedded with ``encoder`` from ``texts`` when given."""
    authored = [clean_id(p) for p in authored_ids]
    present = [p for p in authored if not store.missing([p])]
    vecs = [store.select(present)] if present else []
    missing = [p for p in authored if p not in set(present)]
    if missing and encoder is not None and texts is not None:
        papers = [texts[p] for p in missing if p in texts and texts[p].get("title")]
        if papers:
            vecs.append(np.asarray(encoder.embed_papers(papers), dtype=np.float32))
    if not vecs:
        return []
    centroid = np.vstack(vecs).mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm <= 0:
        return []
    hits = dense_index.search(centroid / norm, k=k, exclusion_set=excluded, candidate_filter=allowed)
    return [clean_id(p) for p, _ in hits][:k]
