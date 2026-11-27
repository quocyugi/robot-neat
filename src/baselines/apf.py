"""
Baseline B5 — Artificial Potential Field (APF)
===============================================
Classic attractive (goal) + repulsive (obstacles) potential field controller.
Known to get stuck in U-traps due to local minima.

Reference: overview.md §6.2 (B5)

Chạy với mọi số hướng cảm biến N (NEAT_LIDAR_N): N = len(obs) - 3, hướng i ở góc 2*pi*i/N
so với hướng robot (với hình quạt là tâm hình quạt). Với N = 12 mọi phép tính trùng từng
bit với bản chỉ có 12 tia (obs[:12], obs[13], obs[14], linspace(0, 2*pi, 12)).

Waffle (NEAT_ROBOT=waffle, đợt 11 bản rút gọn)
-----------------------------------------------
- Tầm nhìn cố định: 12 quạt chính xác đều 30 độ (NEAT_LIDAR_MODE=sector, NEAT_LIDAR_N=12, bố cục
  đều; quạt i có tâm ở 2*pi*i/12), tầm R = 3,5 m (LIDAR_MAX_RANGE của Waffle), bất kể mã hóa
  của NEAT / GA. Gọi với cảm biến khác thì báo lỗi ngay khi dựng (apf_unsupported_reason).
- Lực hút, lực đẩy, k_att, k_rep, d_safe giữ nguyên; speed = clip(0,5 F_x; +-1), turn =
  clip(0,5 F_y; +-1) như bản cũ. Đổi sang (v, w): đầu ra là (o1, o2) = (speed, turn), đi qua
  đúng phép của mạng (kinematics.waffle_step): v_c = 0,26 max(speed; 0) (không lùi),
  w_c = 1,82 turn, rồi kẹp bánh. Robot cũ giữ đầu ra hai bánh (speed - turn, speed + turn).
- Bố cục quạt không đều (e32) không có hướng 2*pi*i/N nên APF báo lỗi với mọi robot.
"""

from __future__ import annotations

import numpy as np

from src.env import kinematics as KM
from src.env.kinematics import LIDAR_MAX_RANGE


def apf_unsupported_reason() -> str | None:
    """Lý do APF không chạy được với robot / cảm biến của tiến trình này (None = chạy được)."""
    if KM.SECTOR_LAYOUT:
        return (f"APF cần quạt đều (hướng 2*pi*i/N); NEAT_SECTOR_LAYOUT={KM.SECTOR_LAYOUT!r} không dùng được. "
                "Chạy APF với NEAT_LIDAR_MODE=sector, NEAT_LIDAR_N=12, NEAT_SECTOR_LAYOUT rỗng")
    if KM.ROBOT == "waffle" and (KM.LIDAR_MODE, KM.LIDAR_NUM_RAYS) != ("sector", 12):
        return (f"APF của Waffle luôn nhìn 12 quạt chính xác: cần NEAT_LIDAR_MODE=sector, NEAT_LIDAR_N=12 "
                f"(đang là {KM.LIDAR_MODE}, {KM.LIDAR_NUM_RAYS})")
    return None


class APFController:
    """Artificial Potential Field navigation controller."""

    def __init__(
        self,
        k_att: float = 1.0,
        k_rep: float = 0.5,
        d_safe: float = 1.0,
    ):
        reason = apf_unsupported_reason()
        if reason:
            raise ValueError(reason)
        self.k_att = k_att
        self.k_rep = k_rep
        self.d_safe = d_safe
        self.vw = KM.ROBOT == "waffle"     # Waffle: đầu ra (speed, turn) = lệnh (tốc độ tiến, tốc độ quay)

    def activate(self, obs: list[float]) -> list[float]:
        """Compute wheel commands from the (N+3)-D observation vector (N = 12: 15-D)."""
        n = len(obs) - 3                 # số hướng cảm biến
        proximity = np.array(obs[:n])
        sin_phi = obs[n + 1]
        cos_phi = obs[n + 2]

        # ── Attractive force (toward goal) ────────────────────────────
        f_att = np.array([cos_phi, sin_phi]) * self.k_att

        # ── Repulsive force (away from obstacles) ─────────────────────
        f_rep = np.zeros(2)
        ray_angles_local = np.linspace(0, 2 * np.pi, n, endpoint=False)

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

        if self.vw:                      # Waffle: (o1, o2) = (speed, turn), rồi waffle_step như đầu ra mạng
            return [float(speed), float(turn)]

        oL = np.clip(speed - turn, -1.0, 1.0)
        oR = np.clip(speed + turn, -1.0, 1.0)

        return [float(oL), float(oR)]
