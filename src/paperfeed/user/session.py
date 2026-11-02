"""Session state (Phase 4.4 / docs/plan.md §4.4 / docs/design.md §7.2-7.3).

Holds the session vector v_sess_t, the negative set S^-_u and SessionConsumed(u):
- ``click`` / ``save``: v_t = alpha * v_{t-1} + (1 - alpha) * e_clicked, with v_0 = 0. ``save`` also adds the paper to
  SessionConsumed (and so to the exclusion set).
- ``skip``: the paper joins S^-_u; the vector is unchanged.
- Optional pivot gate (``use_gate``, off by default): if 1 - cos(e_clicked, v_{t-1}) > gamma_pivot the session is
  reset to v_t = e_clicked instead of blending. The first positive action is never gated (v_0 = 0 has no direction).
- ``reset()`` clears everything. The inactivity timeout is a UI concern and lives outside this class.

In Track 1 the session is never updated, so it stays empty (v = 0, no negatives).
"""

from typing import Any, Dict, List, Mapping, Optional, Set

import numpy as np

from paperfeed.common.logger import get_logger
from paperfeed.data.cohort import clean_id

logger = get_logger("user.session")

DEFAULT_DIM = 768
POSITIVE_ACTIONS = ("click", "save")


class SessionState:
    def __init__(
        self,
        alpha_sess: float = 0.5,
        use_gate: bool = False,
        gamma_pivot: float = 0.35,
        dim: int = DEFAULT_DIM,
    ) -> None:
        if not 0.0 <= alpha_sess <= 1.0:
            raise ValueError(f"alpha_sess must be in [0, 1], got {alpha_sess}.")
        self.alpha_sess = float(alpha_sess)
        self.use_gate = bool(use_gate)
        self.gamma_pivot = float(gamma_pivot)
        self.dim = int(dim)
        self._vector = np.zeros(self.dim, dtype=np.float32)
        self._negatives: Set[str] = set()
        self._consumed: Set[str] = set()
        self._positives: List[str] = []  # paper IDs of click / save actions, oldest first
        self.n_gate_resets = 0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], dim: int = DEFAULT_DIM) -> "SessionState":
        """Build from the ``user_model`` section of configs/default.yaml."""
        um = cfg.get("user_model", {})
        return cls(
            alpha_sess=um.get("alpha_session", 0.5),
            use_gate=um.get("use_pivot_gate", False),
            gamma_pivot=um.get("gamma_pivot", 0.35),
            dim=dim,
        )

    # -- read access ---------------------------------------------------------------------------------------------

    @property
    def vector(self) -> np.ndarray:
        """v_sess_t (a copy; zero when the session has no positive action yet)."""
        return self._vector.copy()

    @property
    def is_empty(self) -> bool:
        """True while there has been no positive action, i.e. the retrieval query should use the persona alone."""
        return not self._positives

    @property
    def negatives(self) -> Set[str]:
        """S^-_u: skipped paper IDs (clean bare IDs)."""
        return set(self._negatives)

    @property
    def consumed(self) -> Set[str]:
        """SessionConsumed(u): saved paper IDs (clean bare IDs)."""
        return set(self._consumed)

    @property
    def positives(self) -> List[str]:
        """Every clicked or saved paper ID, oldest first (clean bare IDs)."""
        return list(self._positives)

    def recent_positives(self, n: int = 3) -> List[str]:
        """The last ``n`` clicked or saved unique paper IDs, most recent first (for the pseudo-query, plan §4.5)."""
        unique_recent = list(dict.fromkeys(self._positives[::-1]))
        return unique_recent[:n]

    # -- actions -------------------------------------------------------------------------------------------------

    def _check_embedding(self, embedding: np.ndarray) -> np.ndarray:
        emb = np.asarray(embedding, dtype=np.float32)
        if emb.shape != (self.dim,):
            raise ValueError(f"Embedding must have shape ({self.dim},), got {emb.shape}.")
        return emb

    def _positive(self, paper_id: str, embedding: np.ndarray) -> None:
        emb = self._check_embedding(embedding)
        clean = clean_id(paper_id)
        prev = self._vector
        gated = False
        if self.use_gate and self._positives:
            norms = float(np.linalg.norm(emb)) * float(np.linalg.norm(prev))
            if norms > 1e-12:
                delta = 1.0 - float(emb @ prev) / norms
                gated = delta > self.gamma_pivot
        if gated:
            self._vector = emb.copy()
            self.n_gate_resets += 1
        else:
            self._vector = (self.alpha_sess * prev + (1.0 - self.alpha_sess) * emb).astype(np.float32)
        self._positives.append(clean)

    def click(self, paper_id: str, embedding: np.ndarray) -> None:
        """Positive action: blend the clicked paper into the session vector."""
        self._positive(paper_id, embedding)

    def save(self, paper_id: str, embedding: np.ndarray) -> None:
        """Positive action like ``click``; the paper also joins SessionConsumed."""
        self._positive(paper_id, embedding)
        self._consumed.add(clean_id(paper_id))

    def skip(self, paper_id: str) -> None:
        """Negative action: the paper joins S^-_u. The session vector does not change."""
        self._negatives.add(clean_id(paper_id))

    def apply(self, action: str, paper_id: str, embedding: Optional[np.ndarray] = None) -> None:
        """Dispatch a feedback action by name (``click``, ``save`` or ``skip``)."""
        if action == "skip":
            self.skip(paper_id)
        elif action in POSITIVE_ACTIONS:
            if embedding is None:
                raise ValueError(f"Action '{action}' needs the paper embedding.")
            (self.save if action == "save" else self.click)(paper_id, embedding)
        else:
            raise ValueError(f"Unknown action '{action}'; expected click, save or skip.")

    def reset(self) -> None:
        """Clear the session: v_sess = 0, S^-_u and SessionConsumed empty (the persona is unaffected)."""
        self._vector = np.zeros(self.dim, dtype=np.float32)
        self._negatives.clear()
        self._consumed.clear()
        self._positives.clear()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vector": self._vector.tolist(),
            "negatives": sorted(self._negatives),
            "consumed": sorted(self._consumed),
            "positives": list(self._positives),
        }
