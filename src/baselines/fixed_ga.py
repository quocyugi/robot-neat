"""
Baseline B1 / B2 — Fixed-Topology MLP evolved by Genetic Algorithm
====================================================================
Evolves weights of a fixed MLP architecture using a simple real-valued GA.
Used to answer Q1: does topology evolution (NEAT) outperform weight-only?

Reference
---------
- overview.md §6.2 (B1: 1×8, B2: 2×16-8)
- brainstorm/04_experiments/baseline_fairness.md (Decision #24, #25)

Sửa cấu hình (báo cáo mục 8.13)
-------------------------------
Bản gốc: tournament k=3, lai ghép đều cho mọi con, đột biến 10% số gen với
sigma 0.5, khởi tạo N(0, 1), giữ 2 elite. Trên bản đồ tường dày, cùng fitness
geodesic và cùng 180k lượt đánh giá, bản gốc đạt trung bình 3 seed
cấp 3: 16% (4–28%), cấp 4: 14% (0–26%).

Mặc định mới theo GA của Such et al. 2017 (Deep Neuroevolution):
  - chọn lọc cắt ngọn: cha mẹ là 20% cá thể tốt nhất (truncation=0.2);
  - 1 elite chép nguyên sang thế hệ sau (được đánh giá lại trên bản đồ mới);
  - đột biến Gauss trên MỌI gen (mutation_rate=1.0);
  - lai ghép đều giữa hai cha mẹ đã chọn (crossover_rate=1.0).
Cùng điều kiện, cấu hình này đạt trung bình 3 seed cấp 3: 34% (26–42%),
cấp 4: 29% (24–34%); xem bảng mục 8.13 của báo cáo.

Sửa lỗi L24 (GA dùng tanh(z), NEAT dùng tanh(2,5·z))
-----------------------------------------------------
FixedMLP nay dùng cùng hàm kích hoạt với neat-python: tanh(2,5·z) (gain=2.5).
Vì mạng tuyến tính trước tanh, tham số θ với gain 2,5 cho ĐÚNG cùng mạng như
tham số 2,5·θ với gain 1. Do đó khởi tạo N(0; 0,2) và sigma 0,02 ở đây tương
đương chính xác với N(0; 0,5) và sigma 0,05 đã kiểm trong báo cáo (lai ghép đều
và chọn lọc không phụ thuộc thang đo), nên các kết quả trên vẫn đúng.
Muốn dùng tanh thường: FixedMLP(..., gain=1.0) và SimpleGA(init_std=0.5,
mutation_power=0.05, gain=1.0).

Tham số cũ vẫn chọn được: selection="tournament", tournament_k=3,
mutation_rate=0.1, mutation_power=0.5, init_std=1.0, elitism=2, gain=1.0.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.env.geodesic import DEFAULT_CACHE_DIR, get_geodesic
from src.env.robot_env import RobotNavEnv
from src.env.kinematics import MAX_STEPS
from src.neat_engine.evaluator import compute_episode_fitness

ACTIVATION_GAIN: float = 2.5   # = hàm "tanh" của neat-python: tanh(2.5 * z)


class FixedMLP:
    """Simple feedforward MLP with tanh activations (cùng hệ số 2,5 như NEAT)."""

    def __init__(self, layer_sizes: list[int], gain: float = ACTIVATION_GAIN):
        """
        Parameters
        ----------
        layer_sizes : list[int]
            e.g. [15, 8, 2] for B1 or [15, 16, 8, 2] for B2.
        gain : float
            Hệ số trong tanh(gain · z); 2.5 khớp neat-python (L24).
        """
        self.layer_sizes = layer_sizes
        self.gain = gain
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for i in range(len(layer_sizes) - 1):
            self.weights.append(
                np.zeros((layer_sizes[i], layer_sizes[i + 1]))
            )
            self.biases.append(np.zeros(layer_sizes[i + 1]))

    def activate(self, inputs: list[float]) -> list[float]:
        x = np.array(inputs, dtype=np.float64)
        for W, b in zip(self.weights, self.biases):
            x = np.tanh(self.gain * (x @ W + b))
        return x.tolist()

    # ── Flat parameter vector (for GA crossover / mutation) ───────────
    def get_params(self) -> np.ndarray:
        parts = []
        for W, b in zip(self.weights, self.biases):
            parts.append(W.ravel())
            parts.append(b.ravel())
        return np.concatenate(parts)

    def set_params(self, flat: np.ndarray):
        flat = np.asarray(flat, dtype=np.float64)
        offset = 0
        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            n_w = n_in * n_out
            self.weights[i] = flat[offset:offset + n_w].reshape(n_in, n_out)
            offset += n_w
            self.biases[i] = flat[offset:offset + n_out]
            offset += n_out

    @property
    def num_params(self) -> int:
        return sum(
            self.layer_sizes[i] * self.layer_sizes[i + 1] + self.layer_sizes[i + 1]
            for i in range(len(self.layer_sizes) - 1)
        )


class SimpleGA:
    """Real-valued GA for fixed-topology MLP evolution.

    Supports:
      - Truncation selection (default) or tournament selection
      - Gaussian mutation (default: every gene, sigma 0.02 với gain 2.5)
      - Uniform crossover (default: every child)
    rng: bộ sinh số có seed (L14); mặc định tạo mới không seed.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        pop_size: int = 150,
        mutation_rate: float = 1.0,
        mutation_power: float = 0.02,
        tournament_k: int = 3,
        selection: str = "truncation",
        truncation: float = 0.2,
        elitism: int = 1,
        crossover_rate: float = 1.0,
        init_std: float = 0.2,
        gain: float = ACTIVATION_GAIN,
        rng: Optional[np.random.Generator] = None,
    ):
        self.layer_sizes = layer_sizes
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.mutation_power = mutation_power
        self.tournament_k = tournament_k
        self.selection = selection
        self.n_parents = max(2, int(round(truncation * pop_size)))
        self.elitism = elitism
        self.crossover_rate = crossover_rate
        self.gain = gain
        self.rng = rng if rng is not None else np.random.default_rng()

        # Initialise population
        template = FixedMLP(layer_sizes, gain=gain)
        self.n_params = template.num_params
        self.population = self.rng.standard_normal((pop_size, self.n_params)) * init_std
        self.fitnesses = np.zeros(pop_size)

    def evaluate(self, maps: list[dict], progress: str = "geodesic", geo_cache: Optional[str] = DEFAULT_CACHE_DIR) -> float:
        """Evaluate all individuals and return best fitness (bản tuần tự).

        Every individual (elites included) is scored on the same maps, and the
        caller should pass freshly drawn maps each generation so that a lucky
        score is not carried over. scripts/run_baselines.py dùng bản song song.
        """
        geos = [get_geodesic(m, geo_cache) if progress == "geodesic" else None for m in maps]
        for i in range(self.pop_size):
            net = FixedMLP(self.layer_sizes, gain=self.gain)
            net.set_params(self.population[i])
            fits = []
            for map_data, geo in zip(maps, geos):
                env = RobotNavEnv(map_data=map_data, geodesic=geo)
                obs, _ = env.reset()
                terminated = truncated = False
                while not (terminated or truncated):
                    action = np.array(net.activate(obs.tolist()), dtype=np.float32)
                    obs, _, terminated, truncated, info = env.step(action)
                fits.append(compute_episode_fitness(info, progress=progress))
                env.close()
            self.fitnesses[i] = float(np.mean(fits))
        return float(np.max(self.fitnesses))

    def evolve(self):
        """One generation: select → crossover → mutate."""
        order = np.argsort(-self.fitnesses, kind="stable")
        new_pop = np.zeros_like(self.population)

        # Elitism: copy the best `elitism` individuals unchanged
        new_pop[:self.elitism] = self.population[order[:self.elitism]]
        parents = order[:self.n_parents]

        for i in range(self.elitism, self.pop_size):
            p1 = self._select(parents)
            if self.rng.random() < self.crossover_rate:
                child = self._crossover(p1, self._select(parents))
            else:
                child = p1
            new_pop[i] = self._mutate(child)

        self.population = new_pop

    def _select(self, parents: np.ndarray) -> np.ndarray:
        if self.selection == "tournament":
            return self._tournament()
        return self.population[parents[self.rng.integers(len(parents))]].copy()

    def _tournament(self) -> np.ndarray:
        idx = self.rng.choice(self.pop_size, size=self.tournament_k, replace=False)
        best = idx[np.argmax(self.fitnesses[idx])]
        return self.population[best].copy()

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        mask = self.rng.random(self.n_params) < 0.5
        return np.where(mask, p1, p2)

    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        mask = self.rng.random(self.n_params) < self.mutation_rate
        noise = self.rng.standard_normal(self.n_params) * self.mutation_power
        genome[mask] += noise[mask]
        return genome

    def get_best_net(self) -> FixedMLP:
        """Best individual of the last evaluated generation.

        For reporting, prefer the individual with the best success rate on the
        validation maps (as for NEAT), not the best noisy training fitness.
        """
        best_idx = int(np.argmax(self.fitnesses))
        net = FixedMLP(self.layer_sizes, gain=self.gain)
        net.set_params(self.population[best_idx])
        return net
