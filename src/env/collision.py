"""
Vectorised Collision Detection (Point-to-Segment)
==================================================
Exact geometric collision check — independent of LiDAR rays, so the robot
can never "phase through" a wall via sensor blind spots.

Reference
---------
- brainstorm/01_env/gymnasium_api.md §3.1 (Decision #6)
"""

from __future__ import annotations

import numpy as np

from src.env.kinematics import ROBOT_RADIUS


def check_collision(
    robot_pos: np.ndarray,
    segments: np.ndarray,
    robot_radius: float = ROBOT_RADIUS,
) -> bool:
    """Return True if the robot body overlaps any segment.

    Parameters
    ----------
    robot_pos : ndarray, shape (2,)
    segments  : ndarray, shape (M, 2, 2)
    robot_radius : float

    Returns
    -------
    collided : bool
    """
    if segments.shape[0] == 0:
        return False
    dists = _point_to_segments_dist(robot_pos, segments)
    return bool(np.any(dists <= robot_radius))


def min_obstacle_distance(
    robot_pos: np.ndarray,
    segments: np.ndarray,
) -> float:
    """Shortest distance from robot centre to any obstacle edge.

    Useful for safety margin metrics (Obstacle Clearance / OC).
    """
    if segments.shape[0] == 0:
        return float("inf")
    return float(np.min(_point_to_segments_dist(robot_pos, segments)))


# ------------------------------------------------------------------ #
#  Internal vectorised distance computation                            #
# ------------------------------------------------------------------ #

def _point_to_segments_dist(
    p: np.ndarray,
    segments: np.ndarray,
) -> np.ndarray:
    """Compute distance from point *p* to each segment.

    Parameters
    ----------
    p : ndarray, shape (2,)
    segments : ndarray, shape (M, 2, 2)

    Returns
    -------
    dists : ndarray, shape (M,)
    """
    A = segments[:, 0, :]                    # (M, 2)
    B = segments[:, 1, :]                    # (M, 2)
    AB = B - A                               # (M, 2)
    AP = p - A                               # (M, 2)

    AB_sq = np.sum(AB ** 2, axis=1)          # (M,)
    AB_sq = np.maximum(AB_sq, 1e-12)         # avoid /0 for degenerate segments

    t = np.sum(AP * AB, axis=1) / AB_sq      # (M,)
    t = np.clip(t, 0.0, 1.0)[:, np.newaxis]  # (M, 1)

    closest = A + t * AB                      # (M, 2)
    dists = np.linalg.norm(p - closest, axis=1)
    return dists
