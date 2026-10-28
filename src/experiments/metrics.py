"""
Evaluation Metrics — SR, SPL, MTT, Collision Rate
===================================================
Computes standardised metrics on a test set of episodes.

Reference
---------
- brainstorm/04_experiments/ablation_design.md (Decision #27)
- overview.md §6.3
"""

from __future__ import annotations

import numpy as np


def success_rate(infos: list[dict]) -> float:
    """Fraction of episodes where the robot reached the goal."""
    if not infos:
        return 0.0
    return sum(1 for i in infos if i["is_success"]) / len(infos)


def collision_rate(infos: list[dict]) -> float:
    """Fraction of episodes that ended in collision."""
    if not infos:
        return 0.0
    return sum(1 for i in infos if i["is_collision"]) / len(infos)


def mean_time_to_traverse(infos: list[dict], dt: float = 0.1) -> float:
    """Average time (seconds) of *successful* episodes only."""
    times = [i["steps"] * dt for i in infos if i["is_success"]]
    return float(np.mean(times)) if times else float("inf")


def compute_spl(
    infos: list[dict],
    optimal_lengths: list[float],
) -> float:
    """Success weighted by Path Length (Anderson et al., 2018).

    SPL = (1/N) Σ S_i · (L_opt / max(L_act, L_opt))

    Parameters
    ----------
    infos : list[dict]
        Episode info dicts (must contain "is_success" and "trajectory").
    optimal_lengths : list[float]
        A* optimal path lengths for each map (same order as infos).
    """
    if not infos:
        return 0.0

    values = []
    for info, l_opt in zip(infos, optimal_lengths):
        if not info["is_success"]:
            values.append(0.0)
            continue
        traj = info["trajectory"]
        l_act = _trajectory_length(traj)
        values.append(l_opt / max(l_act, l_opt))

    return float(np.mean(values))


def _trajectory_length(trajectory: list) -> float:
    """Compute total path length from a list of (x, y) points."""
    total = 0.0
    for i in range(1, len(trajectory)):
        p0 = np.array(trajectory[i - 1])
        p1 = np.array(trajectory[i])
        total += float(np.linalg.norm(p1 - p0))
    return total


def compute_all_metrics(
    infos: list[dict],
    optimal_lengths: list[float] | None = None,
    dt: float = 0.1,
) -> dict[str, float]:
    """Convenience: compute all metrics in one call."""
    result = {
        "SR": success_rate(infos),
        "CR": collision_rate(infos),
        "MTT": mean_time_to_traverse(infos, dt),
    }
    if optimal_lengths is not None:
        result["SPL"] = compute_spl(infos, optimal_lengths)
    return result
