"""
Novelty Search — Archive, k-NN Scoring & Adaptive Hybrid Manager
==================================================================
Implements Novelty Search with:
  - Behavior descriptor b(x) = (x_final, y_final)
  - k-NN novelty scoring via scipy cKDTree
  - Adaptive archive with threshold control
  - Hybrid fitness-novelty combination with adaptive λ
  - Minimal Criteria (MCNS) filtering

Reference
---------
- brainstorm/03_novelty/behavior_descriptor.md   (Decisions #16, #17)
- brainstorm/03_novelty/archive_management.md     (Decisions #18, #19)
- brainstorm/03_novelty/adaptive_lambda.md        (Decisions #20, #21)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


# ══════════════════════════════════════════════════════════════════════════
#  Novelty Archive
# ══════════════════════════════════════════════════════════════════════════

class NoveltyArchive:
    """Bounded archive of novel behavior descriptors with adaptive threshold.

    Parameters
    ----------
    k_neighbours : int
        Number of nearest neighbours for novelty score (Decision #17: 15).
    max_size : int
        Hard cap on archive size; oldest entries are evicted (FIFO).
    initial_threshold : float
        Starting novelty threshold for archive admission.
    """

    def __init__(
        self,
        k_neighbours: int = 15,
        max_size: int = 2000,
        initial_threshold: float = 0.5,
    ):
        self.k = k_neighbours
        self.max_size = max_size
        self.threshold = initial_threshold
        self._archive: list[np.ndarray] = []
        self._tree: cKDTree | None = None

        # Adaptive threshold bookkeeping
        self._added_this_gen: int = 0
        self._evaluated_this_gen: int = 0

    # ------------------------------------------------------------------ #
    #  Novelty scoring                                                     #
    # ------------------------------------------------------------------ #

    def compute_novelty(
        self,
        behaviors: np.ndarray,
        population_behaviors: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute novelty score for each behavior in *behaviors*.

        Novelty(b) = mean distance to k nearest neighbours in
        (population ∪ archive).

        Parameters
        ----------
        behaviors : ndarray, shape (N, D)
        population_behaviors : ndarray, shape (N, D), optional
            If provided, included in the neighbour pool.

        Returns
        -------
        scores : ndarray, shape (N,)
        """
        # Build combined pool
        pool_parts: list[np.ndarray] = []
        if population_behaviors is not None and len(population_behaviors):
            pool_parts.append(population_behaviors)
        if self._archive:
            pool_parts.append(np.array(self._archive))

        if not pool_parts:
            return np.zeros(len(behaviors))

        pool = np.concatenate(pool_parts, axis=0)
        tree = cKDTree(pool)

        # k+1 because the point itself may be in the pool
        k_eff = min(self.k + 1, len(pool))
        dists, _ = tree.query(behaviors, k=k_eff)

        if dists.ndim == 1:
            dists = dists[:, np.newaxis]

        # Average distance (exclude self-match at dist=0)
        scores = np.mean(dists[:, 1:], axis=1) if dists.shape[1] > 1 else dists[:, 0]
        return scores

    # ------------------------------------------------------------------ #
    #  Archive management                                                  #
    # ------------------------------------------------------------------ #

    def try_add(self, behavior: np.ndarray, novelty_score: float) -> bool:
        """Attempt to add a behavior to the archive.

        Uses adaptive threshold + 1% random admission (Decision #18).
        """
        self._evaluated_this_gen += 1

        admitted = False
        if novelty_score >= self.threshold:
            admitted = True
        elif np.random.random() < 0.01:
            admitted = True  # 1% random admission

        if admitted:
            self._archive.append(behavior.copy())
            self._added_this_gen += 1
            self._tree = None  # invalidate cached tree

            # FIFO eviction if over capacity
            if len(self._archive) > self.max_size:
                self._archive = self._archive[-self.max_size:]
                self._tree = None

        return admitted

    def end_generation(self):
        """Adapt threshold at the end of each generation (Decision #18).

        Target: ~5–10% of evaluated individuals should be archived.
        """
        if self._evaluated_this_gen > 0:
            rate = self._added_this_gen / self._evaluated_this_gen
            if rate > 0.10:
                self.threshold *= 1.05    # too many → raise bar
            elif rate < 0.05:
                self.threshold *= 0.95    # too few → lower bar
            self.threshold = max(self.threshold, 0.01)

        self._added_this_gen = 0
        self._evaluated_this_gen = 0

    @property
    def size(self) -> int:
        return len(self._archive)


# ══════════════════════════════════════════════════════════════════════════
#  Adaptive Hybrid Manager
# ══════════════════════════════════════════════════════════════════════════

class AdaptiveHybridManager:
    """Controls the hybrid fitness-novelty blend weight λ.

    Normal:     λ = 0.6  (fitness-dominant)
    Stagnation: λ = 0.1  (novelty-dominant, "trap-breaker" mode)

    Reference: brainstorm/03_novelty/adaptive_lambda.md (Decision #20)
    """

    def __init__(
        self,
        base_lambda: float = 0.6,
        exploration_lambda: float = 0.1,
        stagnation_limit: int = 5,
    ):
        self.base_lambda = base_lambda
        self.exploration_lambda = exploration_lambda
        self.stagnation_limit = stagnation_limit

        self.current_lambda: float = base_lambda
        self._best_history: list[float] = []
        self._stagnant_gens: int = 0

    def update(self, current_max_fitness: float) -> float:
        """Call once per generation with the population's best fitness."""
        if self._best_history and current_max_fitness <= max(self._best_history):
            self._stagnant_gens += 1
        else:
            self._stagnant_gens = 0

        self._best_history.append(current_max_fitness)

        if self._stagnant_gens >= self.stagnation_limit:
            self.current_lambda = self.exploration_lambda
        else:
            self.current_lambda = min(
                self.base_lambda, self.current_lambda + 0.1,
            )

        return self.current_lambda

    def combine_scores(
        self,
        raw_fitness: np.ndarray,
        raw_novelty: np.ndarray,
        steps: np.ndarray,
        collisions: np.ndarray,
        mc_threshold: int = 30,
    ) -> np.ndarray:
        """Min-Max normalise and combine fitness + novelty.

        Applies Minimal Criteria (Decision #21): genomes that collide
        before step *mc_threshold* receive only 20% of their novelty score.

        Returns
        -------
        hybrid : ndarray, shape (N,) — values ≥ 0.01
        """
        # Min-Max scaling
        f_norm = _minmax(raw_fitness)
        n_norm = _minmax(raw_novelty)

        # MCNS penalty
        mc_factor = np.where(collisions & (steps < mc_threshold), 0.2, 1.0)
        adjusted_novelty = n_norm * mc_factor

        hybrid = (
            self.current_lambda * f_norm
            + (1.0 - self.current_lambda) * adjusted_novelty
        )
        return np.maximum(hybrid, 0.01)

    @property
    def is_exploring(self) -> bool:
        return self.current_lambda <= self.exploration_lambda


# ── Helpers ──────────────────────────────────────────────────────────────

def _minmax(arr: np.ndarray) -> np.ndarray:
    ptp = np.ptp(arr)
    if ptp < 1e-6:
        return np.zeros_like(arr)
    return (arr - np.min(arr)) / ptp
