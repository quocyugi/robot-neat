"""
Baseline B3 — Random Search
=============================
Evaluates randomly initialised networks to establish a non-trivial lower bound.

Reference: overview.md §6.2 (B3)
"""

from __future__ import annotations

import numpy as np

from src.baselines.fixed_ga import FixedMLP
from src.env.robot_env import RobotNavEnv
from src.neat_engine.evaluator import compute_episode_fitness


def random_search(
    layer_sizes: list[int],
    maps: list[dict],
    n_trials: int = 1000,
) -> tuple[FixedMLP, float]:
    """Sample *n_trials* random networks and return the best one."""
    best_net = None
    best_fitness = -float("inf")

    for _ in range(n_trials):
        net = FixedMLP(layer_sizes)
        params = np.random.randn(net.num_params)
        net.set_params(params)

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

        mean_fit = float(np.mean(fits))
        if mean_fit > best_fitness:
            best_fitness = mean_fit
            best_net = net

    return best_net, best_fitness
