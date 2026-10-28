"""
NEAT Genome Evaluator — Fitness Computation
=============================================
Runs a genome through one or more maps and computes the 5-component
fitness score.

Reference
---------
- brainstorm/02_neat/fitness_design.md  (Decisions #8, #9)
- overview.md §4.1
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.env.kinematics import MAX_STEPS


# ── Fitness weights (Decision #8) ────────────────────────────────────────
W_PROGRESS: float = 1.0
W_GOAL: float = 10.0
W_TIME: float = 5.0
W_COLLISION: float = 2.0
W_SPIN: float = 1.0
F_OFFSET: float = 3.0       # ensures F > 0 for neat-python


def compute_episode_fitness(
    info: dict[str, Any],
    w1: float = W_PROGRESS,
    w2: float = W_GOAL,
    w3: float = W_TIME,
    w4: float = W_COLLISION,
    w5: float = W_SPIN,
    max_steps: int = MAX_STEPS,
) -> float:
    """Compute fitness for a single episode from the env's ``info`` dict.

    Returns
    -------
    fitness : float ≥ 0.01   (always positive for neat-python)
    """
    d0: float = info["d0"]
    d_min: float = info["min_distance_to_goal"]
    is_goal: float = float(info["is_success"])
    is_col: float = float(info["is_collision"])
    steps: int = info["steps"]
    spin_steps: int = info["spin_steps"]

    # 1. Progress (capped at [0, 1])
    raw_progress = max(0.0, (d0 - d_min) / max(d0, 1e-5))
    progress = 0.5 * raw_progress if is_col else raw_progress

    # 2. Goal + time bonus
    time_bonus = (1.0 - steps / max_steps) * is_goal

    # 3. Spin penalty
    spin_penalty = spin_steps / max_steps

    # 4. Raw fitness
    f_raw = (
        w1 * progress
        + w2 * is_goal
        + w3 * time_bonus
        - w4 * is_col
        - w5 * spin_penalty
    )

    # 5. Offset to guarantee positivity
    return max(f_raw + F_OFFSET, 0.01)


def evaluate_genome_on_maps(
    genome_net,
    maps: list[dict],
    env_cls,
    is_recurrent: bool = False,
) -> tuple[float, list[dict]]:
    """Evaluate a single genome across *K* maps and return mean fitness.

    Parameters
    ----------
    genome_net
        An activated NEAT network (FeedForwardNetwork or RecurrentNetwork).
    maps : list[dict]
        List of map_data dicts loaded from JSON.
    env_cls
        The RobotNavEnv class (passed to avoid circular imports).
    is_recurrent : bool
        If True, call ``genome_net.reset()`` at the start of each episode.

    Returns
    -------
    mean_fitness : float
    episode_infos : list[dict]
    """
    fitnesses: list[float] = []
    infos: list[dict] = []

    for map_data in maps:
        env = env_cls(map_data=map_data)
        obs, _ = env.reset()

        if is_recurrent and hasattr(genome_net, "reset"):
            genome_net.reset()  # Decision #13: clear hidden state

        terminated = truncated = False
        while not (terminated or truncated):
            action = genome_net.activate(obs.tolist())
            action = np.array(action, dtype=np.float32)
            obs, _, terminated, truncated, info = env.step(action)

        fitnesses.append(compute_episode_fitness(info))
        infos.append(info)
        env.close()

    return float(np.mean(fitnesses)), infos
