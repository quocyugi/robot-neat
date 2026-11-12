"""
Baseline B3 — Random Search
=============================
Evaluates randomly initialised networks to establish a non-trivial lower bound.

Reference: overview.md §6.2 (B3)

Khởi tạo N(0; 0,2) với tanh hệ số 2,5, cùng phân bố khởi tạo của GA (xem L24 trong
src/baselines/fixed_ga.py); rng có seed để tái lập (L14). scripts/run_baselines.py
chạy bản song song cùng ngân sách với NEAT.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.baselines.fixed_ga import ACTIVATION_GAIN, FixedMLP
from src.env.geodesic import DEFAULT_CACHE_DIR, get_geodesic
from src.env.robot_env import RobotNavEnv
from src.neat_engine.evaluator import compute_episode_fitness


def random_search(
    layer_sizes: list[int],
    maps: list[dict],
    n_trials: int = 1000,
    init_std: float = 0.2,
    rng: Optional[np.random.Generator] = None,
    progress: str = "geodesic",
    geo_cache: Optional[str] = DEFAULT_CACHE_DIR,
) -> tuple[FixedMLP, float]:
    """Sample *n_trials* random networks and return the best one."""
    rng = rng if rng is not None else np.random.default_rng()
    geos = [get_geodesic(m, geo_cache) if progress == "geodesic" else None for m in maps]
    best_net = None
    best_fitness = -float("inf")

    for _ in range(n_trials):
        net = FixedMLP(layer_sizes, gain=ACTIVATION_GAIN)
        params = rng.standard_normal(net.num_params) * init_std
        net.set_params(params)

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

        mean_fit = float(np.mean(fits))
        if mean_fit > best_fitness:
            best_fitness = mean_fit
            best_net = net

    return best_net, best_fitness
