"""
Baseline B1 / B2 — Fixed-Topology MLP evolved by Genetic Algorithm
====================================================================
Evolves weights of a fixed MLP architecture using a simple real-valued GA.
Used to answer Q1: does topology evolution (NEAT) outperform weight-only?

Reference
---------
- overview.md §6.2 (B1: 1×8, B2: 2×16-8)
- brainstorm/04_experiments/baseline_fairness.md (Decision #24, #25)
"""

from __future__ import annotations

import numpy as np

from src.env.robot_env import RobotNavEnv
from src.env.kinematics import MAX_STEPS
from src.neat_engine.evaluator import compute_episode_fitness


class FixedMLP:
    """Simple feedforward MLP with tanh activations."""

    def __init__(self, layer_sizes: list[int]):
        """
        Parameters
        ----------
        layer_sizes : list[int]
            e.g. [15, 8, 2] for B1 or [15, 16, 8, 2] for B2.
        """
        self.layer_sizes = layer_sizes
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
            x = np.tanh(x @ W + b)
        return x.tolist()

    # ── Flat parameter vector (for GA crossover / mutation) ───────────
    def get_params(self) -> np.ndarray:
        parts = []
        for W, b in zip(self.weights, self.biases):
            parts.append(W.ravel())
            parts.append(b.ravel())
        return np.concatenate(parts)

    def set_params(self, flat: np.ndarray):
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
    """Minimalist real-valued GA for fixed-topology MLP evolution.

    Supports:
      - Tournament selection
      - Gaussian mutation
      - Uniform crossover
    """

    def __init__(
        self,
        layer_sizes: list[int],
        pop_size: int = 150,
        mutation_rate: float = 0.1,
        mutation_power: float = 0.5,
        tournament_k: int = 3,
    ):
        self.layer_sizes = layer_sizes
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.mutation_power = mutation_power
        self.tournament_k = tournament_k

        # Initialise population
        template = FixedMLP(layer_sizes)
        self.n_params = template.num_params
        self.population = np.random.randn(pop_size, self.n_params)
        self.fitnesses = np.zeros(pop_size)

    def evaluate(self, maps: list[dict]) -> float:
        """Evaluate all individuals and return best fitness."""
        for i in range(self.pop_size):
            net = FixedMLP(self.layer_sizes)
            net.set_params(self.population[i])
            fits = []
            for map_data in maps:
                env = RobotNavEnv(map_data=map_data)
                obs, _ = env.reset()
                terminated = truncated = False
                while not (terminated or truncated):
                    action = np.array(net.activate(obs.tolist()), dtype=np.float32)
                    obs, _, terminated, truncated, info = env.step(action)
                fits.append(compute_episode_fitness(info))
                env.close()
            self.fitnesses[i] = float(np.mean(fits))
        return float(np.max(self.fitnesses))

    def evolve(self):
        """One generation: select → crossover → mutate."""
        new_pop = np.zeros_like(self.population)

        # Elitism: keep top 2
        elite_idx = np.argsort(self.fitnesses)[-2:]
        new_pop[:2] = self.population[elite_idx]

        for i in range(2, self.pop_size):
            p1 = self._tournament()
            p2 = self._tournament()
            child = self._crossover(p1, p2)
            child = self._mutate(child)
            new_pop[i] = child

        self.population = new_pop

    def _tournament(self) -> np.ndarray:
        idx = np.random.choice(self.pop_size, size=self.tournament_k, replace=False)
        best = idx[np.argmax(self.fitnesses[idx])]
        return self.population[best].copy()

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        mask = np.random.random(self.n_params) < 0.5
        return np.where(mask, p1, p2)

    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        mask = np.random.random(self.n_params) < self.mutation_rate
        noise = np.random.randn(self.n_params) * self.mutation_power
        genome[mask] += noise[mask]
        return genome

    def get_best_net(self) -> FixedMLP:
        best_idx = int(np.argmax(self.fitnesses))
        net = FixedMLP(self.layer_sizes)
        net.set_params(self.population[best_idx])
        return net
