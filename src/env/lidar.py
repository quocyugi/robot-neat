"""
Vectorised 2-D LiDAR Ray-Casting
=================================
Casts *num_rays* equally-spaced rays from the robot centre and returns
a **proximity index** for each ray (0 = safe, 1 = touching obstacle).

All geometry is reduced to line-segment intersection so that both
rectangular walls and circular obstacles (approximated as regular
polygons) share the same code path — no branching.

Reference
---------
- brainstorm/01_env/ray_casting.md  (Decision #1, #2, #3)

Performance
-----------
> 20 000 FPS on a single CPU core (12 rays × ~200 segments).
"""

from __future__ import annotations

import numpy as np

from src.env.kinematics import LIDAR_NUM_RAYS, LIDAR_MAX_RANGE


class FastLiDAR2D:
    """Vectorised 2-D LiDAR scanner using segment–segment intersection."""

    def __init__(
        self,
        num_rays: int = LIDAR_NUM_RAYS,
        max_range: float = LIDAR_MAX_RANGE,
    ):
        self.num_rays = num_rays
        self.max_range = max_range
        # Pre-compute local ray angles: 0, 30°, 60°, … , 330°
        self.ray_angles: np.ndarray = np.linspace(
            0, 2 * np.pi, num_rays, endpoint=False
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #
    def cast_rays(
        self,
        robot_pos: np.ndarray,
        robot_theta: float,
        segments: np.ndarray,
    ) -> np.ndarray:
        """Cast all rays and return proximity indices.

        Parameters
        ----------
        robot_pos : ndarray, shape (2,)
            Robot centre (x, y).
        robot_theta : float
            Robot heading (radians, CCW from +x).
        segments : ndarray, shape (M, 2, 2)
            Each row is a line segment [[Ax, Ay], [Bx, By]].

        Returns
        -------
        proximity : ndarray, shape (num_rays,)
            Values in [0, 1].  0 → no obstacle within range,
            1 → obstacle at robot centre.
        """
        # Global ray angles
        angles = robot_theta + self.ray_angles                    # (R,)

        # Ray direction vectors scaled to max range
        r = (
            np.stack([np.cos(angles), np.sin(angles)], axis=-1)
            * self.max_range
        )                                                         # (R, 2)

        A = segments[:, 0, :]                                     # (M, 2)
        B = segments[:, 1, :]                                     # (M, 2)
        s = B - A                                                 # (M, 2)

        # Broadcast to (R, M, 2)
        r_exp = r[:, np.newaxis, :]                               # (R, 1, 2)
        s_exp = s[np.newaxis, :, :]                               # (1, M, 2)

        # 2-D cross products  (R, M)
        r_cross_s = (
            r_exp[..., 0] * s_exp[..., 1]
            - r_exp[..., 1] * s_exp[..., 0]
        )

        p_to_A = (A - robot_pos)[np.newaxis, :, :]               # (1, M, 2)

        pA_cross_s = (
            p_to_A[..., 0] * s_exp[..., 1]
            - p_to_A[..., 1] * s_exp[..., 0]
        )
        pA_cross_r = (
            p_to_A[..., 0] * r_exp[..., 1]
            - p_to_A[..., 1] * r_exp[..., 0]
        )

        # Avoid division by zero for parallel lines
        parallel = np.abs(r_cross_s) < 1e-8
        safe_denom = np.where(parallel, 1e-8, r_cross_s)

        t = pA_cross_s / safe_denom                               # (R, M)
        u = pA_cross_r / safe_denom                               # (R, M)

        # Valid intersection: 0 ≤ t ≤ 1  AND  0 ≤ u ≤ 1  AND  not parallel
        valid = (~parallel) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)

        t_valid = np.where(valid, t, 1.0)     # miss → full range
        min_t = np.min(t_valid, axis=1)        # (R,) — closest hit per ray

        # Proximity Index:  0 (safe)  →  1 (touching)
        proximity = 1.0 - min_t
        return proximity.astype(np.float32)


# ------------------------------------------------------------------ #
#  Geometry helpers                                                    #
# ------------------------------------------------------------------ #

def circle_to_segments(
    center: np.ndarray,
    radius: float,
    num_sides: int = 8,
) -> np.ndarray:
    """Approximate a circle as a regular polygon of *num_sides* edges.

    Returns
    -------
    segments : ndarray, shape (num_sides, 2, 2)
    """
    angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    pts = center + radius * np.stack(
        [np.cos(angles), np.sin(angles)], axis=-1
    )
    pts_next = np.roll(pts, -1, axis=0)
    return np.stack([pts, pts_next], axis=1)


def arena_wall_segments(w: float = 10.0, h: float = 10.0) -> np.ndarray:
    """Return the 4 boundary-wall segments of the arena.

    Returns
    -------
    segments : ndarray, shape (4, 2, 2)
    """
    return np.array([
        [[0, 0], [w, 0]],   # bottom
        [[w, 0], [w, h]],   # right
        [[w, h], [0, h]],   # top
        [[0, h], [0, 0]],   # left
    ], dtype=np.float64)
