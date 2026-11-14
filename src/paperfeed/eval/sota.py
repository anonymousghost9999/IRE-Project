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
