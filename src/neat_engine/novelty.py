"""
Novelty Search — Archive, k-NN Scoring & Adaptive Hybrid Manager
==================================================================
Implements Novelty Search with:
  - Behavior descriptor b(x) = vị trí cuối trên K bản đồ, nối thành vector 2K chiều
  - k-NN novelty scoring via scipy cKDTree
  - Adaptive archive with threshold control
  - Hybrid fitness-novelty combination with adaptive λ
  - Minimal Criteria (MCNS) filtering

Reference
---------
- brainstorm/03_novelty/behavior_descriptor.md   (Decisions #16, #17)
- brainstorm/03_novelty/archive_management.md     (Decisions #18, #19)
- brainstorm/03_novelty/adaptive_lambda.md        (Decisions #20, #21)

Các lỗi đã sửa (báo cáo mục 9)
-------------------------------
L06  Descriptor cũ là TRUNG BÌNH vị trí cuối qua K bản đồ khác nhau: hai hành vi
     khác hẳn có thể cùng một trung bình. Nay nối K vị trí (behavior_descriptor),
     chia cho kích thước arena để mọi chiều cùng thang [0, 1].
L07  Archive giữ vị trí đo trên bản đồ của thế hệ trước nên không so được với
     thế hệ sau. NEATRunner nay đo hành vi trên một bộ "bản đồ probe" cố định
     (novelty_space="probe", mặc định), hoặc chỉ so trong quần thể hiện tại
     (novelty_space="population", không dùng archive).
L08  Điểm hybrid chuẩn hóa min–max theo từng thế hệ nên không so được giữa các
     thế hệ (gây L02, làm stagnation vô nghĩa). Nay dùng thang cố định:
     fitness / fitness lớn nhất có thể, novelty / novelty_scale (cắt về [0, 1]).
L09  λ so với kỷ lục toàn run và không reset khi lên cấp nên kẹt ở 0,1 tới 87–88%
     số thế hệ. Nay so trung bình W thế hệ gần nhất với W thế hệ trước đó, gọi
     reset() khi lên cấp, λ khởi đầu 0,5, sàn 0,2.
L03  Thêm fixed_lambda: λ cố định (M4 "novelty thuần" dùng fixed_lambda = 0).
L14  Nhận ngẫu nhiên 1% vào archive dùng bộ sinh số có seed.

LehmanArchive (25/11/2026): luật archive của Lehman & Stanley (2011) cho novelty search thuần kiểu bài báo.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import cKDTree


def behavior_descriptor(final_positions, arena_size=(10.0, 10.0)) -> np.ndarray:
    """L06: nối vị trí cuối (x, y) trên K bản đồ thành vector 2K chiều, chia cho kích thước arena."""
    p = np.asarray(final_positions, dtype=np.float64) / np.asarray(arena_size, dtype=np.float64)
    return p.ravel()


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
        Starting novelty threshold for archive admission. Descriptor nay đã chia
        cho kích thước arena (L06) nên ngưỡng khởi đầu là 0,1 thay cho 0,5 (mét);
        ngưỡng tự điều chỉnh để nhận khoảng 5–10% số cá thể mỗi thế hệ.
    rng : numpy.random.Generator, optional
        Bộ sinh số có seed (L14) cho việc nhận ngẫu nhiên 1%.
    """

    def __init__(
        self,
        k_neighbours: int = 15,
        max_size: int = 2000,
        initial_threshold: float = 0.1,
        rng: Optional[np.random.Generator] = None,
    ):
        self.k = k_neighbours
        self.max_size = max_size
        self.threshold = initial_threshold
        self._archive: list[np.ndarray] = []
        self._rng = rng if rng is not None else np.random.default_rng()

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
        use_archive: bool = True,
    ) -> np.ndarray:
        """Novelty(b) = khoảng cách trung bình tới k láng giềng gần nhất trong
        (quần thể ∪ archive), bỏ chính nó.

        Parameters
        ----------
        behaviors : ndarray, shape (N, D)
        population_behaviors : ndarray, shape (N, D), optional
            If provided, included in the neighbour pool.
        use_archive : bool
            False: chỉ so trong quần thể hiện tại (novelty_space="population").

        Returns
        -------
        scores : ndarray, shape (N,)
        """
        pool_parts: list[np.ndarray] = []
        if population_behaviors is not None and len(population_behaviors):
            pool_parts.append(population_behaviors)
        if use_archive and self._archive:
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
        elif self._rng.random() < 0.01:
            admitted = True  # 1% random admission

        if admitted:
            self._archive.append(np.array(behavior, dtype=np.float64, copy=True))
            self._added_this_gen += 1

            # FIFO eviction if over capacity
            if len(self._archive) > self.max_size:
                self._archive = self._archive[-self.max_size:]

        return admitted

    def calibrate(self, scores: np.ndarray, target_rate: float = 0.075) -> None:
        """Đặt ngưỡng ban đầu theo phân bố novelty thực tế (gọi khi archive còn rỗng).

        Ngưỡng cố định ban đầu không khớp thang của descriptor nên thế hệ đầu nhận gần
        như mọi cá thể, và với hệ số 1,05 mỗi thế hệ phải hơn 40 thế hệ mới bắt kịp.
        Lấy phân vị (1 - target_rate) để chỉ khoảng 7,5% được nhận, rồi tự điều chỉnh.
        """
        if len(scores):
            self.threshold = max(float(np.quantile(scores, 1.0 - target_rate)), 1e-4)

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
            self.threshold = max(self.threshold, 1e-4)

        self._added_this_gen = 0
        self._evaluated_this_gen = 0

    def clear(self):
        self._archive = []

    @property
    def size(self) -> int:
        return len(self._archive)


class LehmanArchive(NoveltyArchive):
    """Archive theo luật của Lehman & Stanley (2011), dùng cho novelty search thuần (NEATRunner novelty_rule="lehman").

    Giống noveltyset.h của mã gốc (jal278/novelty-search-cpp) và lehman_maze/lehman_maze_ns.py:
      - nhận cá thể khi novelty > ngưỡng; không nhận ngẫu nhiên; archive không giới hạn kích thước;
      - archive còn rỗng mà không ai qua ngưỡng: nhận cá thể có novelty cao nhất;
      - 10 thế hệ liền không nhận ai: ngưỡng x 0,95 (không thấp hơn sàn);
      - nhận hơn 4 cá thể trong một thế hệ: ngưỡng x 1,2.
    Novelty vẫn tính bằng compute_novelty (trung bình khoảng cách tới k = 15 láng giềng gần nhất trong quần thể
    và archive, trước khi thêm cá thể của thế hệ này).

    Ngưỡng của bài báo (6 đơn vị, sàn 0,25 đơn vị của mê cung gốc) đổi sang thang descriptor của dự án:
    1 đơn vị = 0,03125 m (tỉ lệ đã dùng để dựng data/maps/test_lehman), chia cho arena 10 m, rồi nhân sqrt(K)
    với K bản đồ probe, vì khoảng cách trong 2K chiều lớn gấp sqrt(K) khi hành vi lệch như nhau trên mọi bản đồ.
    Với K = 4: ngưỡng đầu 0,0375, sàn 0,0015625.
    """

    UNIT = 0.03125 / 10.0     # một đơn vị của mê cung gốc, theo thang descriptor (mét / kích thước arena)

    def __init__(self, k_neighbours: int = 15, n_maps: int = 4, initial_units: float = 6.0, floor_units: float = 0.25):
        scale = self.UNIT * float(np.sqrt(n_maps))
        super().__init__(k_neighbours=k_neighbours, max_size=10**9, initial_threshold=initial_units * scale)
        self.floor = floor_units * scale
        self._generations_without_add = 0

    def update(self, behaviors: np.ndarray, novelty: np.ndarray) -> int:
        """Thêm cá thể của một thế hệ vào archive và chỉnh ngưỡng; trả số cá thể được nhận."""
        behaviors = np.asarray(behaviors, dtype=np.float64)
        novelty = np.asarray(novelty, dtype=np.float64)
        idx = np.flatnonzero(novelty > self.threshold)
        if not self._archive and idx.size == 0:
            idx = np.array([int(np.argmax(novelty))])
        for i in idx:
            self._archive.append(np.array(behaviors[i], dtype=np.float64, copy=True))
        added = int(idx.size)
        self._generations_without_add = 0 if added else self._generations_without_add + 1
        if self._generations_without_add == 10:
            self.threshold = max(self.threshold * 0.95, self.floor)
            self._generations_without_add = 0
        if added > 4:
            self.threshold *= 1.2
        return added


# ══════════════════════════════════════════════════════════════════════════
#  Adaptive Hybrid Manager
# ══════════════════════════════════════════════════════════════════════════

class AdaptiveHybridManager:
    """Controls the hybrid fitness-novelty blend weight λ.

    hybrid = λ · fitness_chuẩn_hóa + (1 − λ) · novelty_chuẩn_hóa

    Normal:     λ tăng dần 0,1 mỗi thế hệ tới base_lambda (0,6, fitness là chính)
    Stagnation: λ = exploration_lambda (0,2, novelty là chính, "trap-breaker")
    Fixed:      fixed_lambda ≠ None giữ λ cố định (M4: 0 = novelty thuần)

    Stagnation (L09): trung bình fitness thô tốt nhất của `window` thế hệ gần nhất
    không cao hơn trung bình của `window` thế hệ trước đó, liên tiếp
    `stagnation_limit` lần kiểm tra. Fitness mỗi thế hệ đo trên bản đồ ngẫu nhiên
    nên ồn; so với kỷ lục (code cũ) thì gần như luôn "trì trệ".

    Reference: brainstorm/03_novelty/adaptive_lambda.md (Decision #20)
    """

    def __init__(
        self,
        base_lambda: float = 0.6,
        exploration_lambda: float = 0.2,
        start_lambda: float = 0.5,
        stagnation_limit: int = 5,
        window: int = 5,
        fixed_lambda: Optional[float] = None,
        fitness_scale: float = 28.0,
        novelty_scale: float = 1.0,
    ):
        self.base_lambda = base_lambda
        self.exploration_lambda = exploration_lambda
        self.start_lambda = start_lambda
        self.stagnation_limit = stagnation_limit
        self.window = window
        self.fixed_lambda = fixed_lambda
        self.fitness_scale = fitness_scale      # L08: thang cố định của fitness
        self.novelty_scale = novelty_scale      # L08: thang cố định của novelty

        self.current_lambda: float = start_lambda
        self._best_history: list[float] = []
        self._stagnant_gens: int = 0
        self.reset()

    def reset(self):
        """Gọi khi lên cấp curriculum: thang fitness đổi nên bỏ lịch sử cũ (L09)."""
        self._best_history = []
        self._stagnant_gens = 0
        self.current_lambda = self.fixed_lambda if self.fixed_lambda is not None else self.start_lambda

    def update(self, current_max_fitness: float) -> float:
        """Call once per generation with the population's best RAW fitness."""
        if self.fixed_lambda is not None:
            self.current_lambda = self.fixed_lambda
            return self.current_lambda

        self._best_history.append(float(current_max_fitness))
        w = self.window
        if len(self._best_history) >= 2 * w:
            recent = float(np.mean(self._best_history[-w:]))
            previous = float(np.mean(self._best_history[-2 * w:-w]))
            if recent <= previous:
                self._stagnant_gens += 1
            else:
                self._stagnant_gens = 0

        if self._stagnant_gens >= self.stagnation_limit:
            self.current_lambda = self.exploration_lambda
        else:
            self.current_lambda = min(self.base_lambda, self.current_lambda + 0.1)
        self.current_lambda = max(self.current_lambda, self.exploration_lambda)
        return self.current_lambda

    def combine_scores(
        self,
        raw_fitness: np.ndarray,
        raw_novelty: np.ndarray,
        steps: np.ndarray,
        collisions: np.ndarray,
        mc_threshold: int = 30,
    ) -> np.ndarray:
        """Chuẩn hóa theo thang CỐ ĐỊNH (L08) rồi kết hợp fitness + novelty.

        Applies Minimal Criteria (Decision #21): genomes that collide
        before step *mc_threshold* receive only 20% of their novelty score.

        Returns
        -------
        hybrid : ndarray, shape (N,) — values in [0.01, 1]
        """
        f_norm = np.clip(np.asarray(raw_fitness, dtype=np.float64) / self.fitness_scale, 0.0, 1.0)
        n_norm = np.clip(np.asarray(raw_novelty, dtype=np.float64) / self.novelty_scale, 0.0, 1.0)

        # MCNS penalty
        mc_factor = np.where(np.asarray(collisions) & (np.asarray(steps) < mc_threshold), 0.2, 1.0)
        adjusted_novelty = n_norm * mc_factor

        hybrid = (
            self.current_lambda * f_norm
            + (1.0 - self.current_lambda) * adjusted_novelty
        )
        return np.maximum(hybrid, 0.01)

    @property
    def is_exploring(self) -> bool:
        return self.fixed_lambda is None and self.current_lambda <= self.exploration_lambda
