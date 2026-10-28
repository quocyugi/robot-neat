"""
Differential Drive Robot Kinematics
====================================
Euler-integration model for a 2-wheeled differential-drive robot.

Reference
---------
- overview.md §2.1 (Equations 1–4)
- brainstorm/01_env/gymnasium_api.md

Constants (unified across project)
-----------------------------------
- Robot radius:      r     = 0.25 m
- Wheel base:        L     = 0.40 m
- Max wheel speed:   v_max = 0.50 m/s
- Simulation dt:     dt    = 0.10 s
- Max episode steps: T_max = 500  steps  (50 s)
"""

import numpy as np

# ── Project-Wide Physical Constants ──────────────────────────────────────────
ROBOT_RADIUS: float = 0.25          # m
WHEEL_BASE: float = 0.40            # m  (distance between two wheels)
V_MAX: float = 0.50                 # m/s (max speed per wheel)
DT: float = 0.10                    # s   (simulation timestep)
MAX_STEPS: int = 500                # steps (episode budget)

ARENA_W: float = 10.0               # m (arena width)
ARENA_H: float = 10.0               # m (arena height)
ARENA_DIAGONAL: float = np.sqrt(ARENA_W ** 2 + ARENA_H ** 2)  # ≈ 14.14 m

GOAL_RADIUS: float = 0.30           # m  (distance to count as "reached goal")
LIDAR_MAX_RANGE: float = 3.0        # m
LIDAR_NUM_RAYS: int = 12


def step_kinematics(
    x: float, y: float, theta: float,
    oL: float, oR: float,
    v_max: float = V_MAX,
    L: float = WHEEL_BASE,
    dt: float = DT,
) -> tuple[float, float, float]:
    """Advance the robot state by one timestep using Euler integration.

    Parameters
    ----------
    x, y, theta : float
        Current pose (position + heading in radians).
    oL, oR : float
        Normalised wheel commands in [-1, 1].
    v_max, L, dt : float
        Physical constants (default to project globals).

    Returns
    -------
    (x_new, y_new, theta_new) : tuple[float, float, float]
    """
    vL = v_max * np.clip(oL, -1.0, 1.0)
    vR = v_max * np.clip(oR, -1.0, 1.0)

    omega = (vR - vL) / L
    v = (vR + vL) / 2.0

    theta_new = theta + omega * dt
    x_new = x + v * np.cos(theta_new) * dt
    y_new = y + v * np.sin(theta_new) * dt

    return x_new, y_new, theta_new


def is_spinning(vL: float, vR: float, eps_linear: float = 0.05,
                eps_angular: float = 0.20) -> bool:
    """Detect in-place spinning behaviour (Decision #17 anti-cheat).

    A robot is deemed to be spinning when:
      - net forward speed is near-zero  (|vL + vR| < eps_linear)
      - but differential is significant (|vL - vR| > eps_angular)
    """
    return abs(vL + vR) < eps_linear and abs(vL - vR) > eps_angular
