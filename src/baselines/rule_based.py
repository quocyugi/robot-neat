"""
Baseline B4 — Heuristic Rule-Based Controller
===============================================
Simple if-then reflexive obstacle avoidance. No learning involved.

Reference: overview.md §6.2 (B4)
"""

from __future__ import annotations

import numpy as np


class RuleBasedController:
    """Reactive controller: steer toward goal, swerve away from obstacles."""

    def __init__(self, danger_threshold: float = 0.5):
        self.danger_threshold = danger_threshold

    def activate(self, obs: list[float]) -> list[float]:
        """
        obs[0:12]  = LiDAR proximity (0=safe, 1=close)
        obs[12]    = normalised distance to goal
        obs[13]    = sin(phi) — heading-to-goal angle
        obs[14]    = cos(phi)
        """
        proximity = np.array(obs[:12])
        sin_phi = obs[13]
        cos_phi = obs[14]

        # Default: steer toward goal
        # phi ≈ atan2(sin_phi, cos_phi)
        turn = np.clip(sin_phi, -1.0, 1.0)
        speed = 0.6

        # Check for danger in front sectors (rays 0, 1, 11 = front arc)
        front_danger = max(proximity[0], proximity[1], proximity[11])

        if front_danger > self.danger_threshold:
            # Obstacle ahead → swerve toward the clearer side.
            # Tia i nằm ở góc i·30° ngược chiều kim đồng hồ: tia 1–5 bên trái, 6–10 bên phải.
            # turn > 0 làm oR > oL nên robot quay trái. Code cũ gán ngược dấu nên rẽ VÀO
            # phía có vật cản (lỗi L12: SR cấp 1 66%, sửa dấu lên 80%).
            left_sum = np.sum(proximity[1:6])
            right_sum = np.sum(proximity[6:11])
            if left_sum < right_sum:
                turn = 0.8   # bên trái thoáng hơn → rẽ trái
            else:
                turn = -0.8  # bên phải thoáng hơn → rẽ phải
            speed = 0.3

        oL = np.clip(speed - turn * 0.5, -1.0, 1.0)
        oR = np.clip(speed + turn * 0.5, -1.0, 1.0)
        return [float(oL), float(oR)]
