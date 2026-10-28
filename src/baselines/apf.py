"""
Baseline B5 — Artificial Potential Field (APF)
===============================================
Classic attractive (goal) + repulsive (obstacles) potential field controller.
Known to get stuck in U-traps due to local minima.

Reference: overview.md §6.2 (B5)
"""

from __future__ import annotations

import numpy as np

from src.env.kinematics import LIDAR_MAX_RANGE


class APFController:
    """Artificial Potential Field navigation controller."""

    def __init__(
        self,
        k_att: float = 1.0,
        k_rep: float = 0.5,
        d_safe: float = 1.0,
    ):
        self.k_att = k_att
        self.k_rep = k_rep
        self.d_safe = d_safe

    def activate(self, obs: list[float]) -> list[float]:
        """Compute wheel commands from the 15-D observation vector."""
        proximity = np.array(obs[:12])
        sin_phi = obs[13]
        cos_phi = obs[14]

        # ── Attractive force (toward goal) ────────────────────────────
        f_att = np.array([cos_phi, sin_phi]) * self.k_att

        # ── Repulsive force (away from obstacles) ─────────────────────
        f_rep = np.zeros(2)
        ray_angles_local = np.linspace(0, 2 * np.pi, 12, endpoint=False)

        for i, prox in enumerate(proximity):
            if prox > 0.01:  # obstacle detected
                dist = (1.0 - prox) * LIDAR_MAX_RANGE
                if dist < self.d_safe:
                    # Repulsive force magnitude
                    mag = self.k_rep * (1.0 / max(dist, 0.05) - 1.0 / self.d_safe)
                    # Direction: away from obstacle (opposite of ray)
                    angle = ray_angles_local[i]
                    f_rep[0] -= mag * np.cos(angle)
                    f_rep[1] -= mag * np.sin(angle)

        # ── Combined force → differential drive ──────────────────────
        f_total = f_att + f_rep
        # Project onto forward and lateral
        forward = f_total[0]     # cos component (aligned with heading)
        lateral = f_total[1]     # sin component (perpendicular)

        speed = np.clip(forward * 0.5, -1.0, 1.0)
        turn = np.clip(lateral * 0.5, -1.0, 1.0)

        oL = np.clip(speed - turn, -1.0, 1.0)
        oR = np.clip(speed + turn, -1.0, 1.0)

        return [float(oL), float(oR)]
