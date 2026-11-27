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

Cảm biến (đọc từ biến môi trường khi import module này)
--------------------------------------------------------
- NEAT_LIDAR_MODE = "rays" (mặc định: N tia điểm) hoặc "sector" (mỗi đầu vào là khoảng
  cách nhỏ nhất tới vật cản trong cả hình quạt quanh hướng đó, xem src/env/lidar.py
  FastLiDAR2D.cast_sectors).
- NEAT_LIDAR_N = số hướng N (mặc định 12); hướng i ở góc theta + 2*pi*i/N.
  Quan sát có N + 3 chiều (N độ gần, rho, sin phi, cos phi).
Không đặt biến nào thì LIDAR_MODE = "rays", LIDAR_NUM_RAYS = 12: hành vi trùng từng bit
với bản chỉ có 12 tia điểm. Biến phải được đặt TRƯỚC khi import src (worker kiểu spawn
thừa hưởng biến và đọc lại khi import; ParallelEvaluator báo lỗi nếu biến bị đổi sau khi
tiến trình cha đã import). Config NEAT phải có num_inputs = N + 3 (configs/neat_*_n{N}.ini,
sinh bằng scripts/make_sensor_configs.py); summary.json ghi "sensor" khi khác mặc định.
"""

import os

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

# ── Cảm biến: kiểu và số hướng lấy từ biến môi trường (mặc định 12 tia điểm) ──
LIDAR_MODES: tuple[str, ...] = ("rays", "sector")
DEFAULT_LIDAR_MODE: str = "rays"
DEFAULT_LIDAR_NUM_RAYS: int = 12


def lidar_settings_from_env() -> tuple[str, int]:
    """(kiểu, N) từ NEAT_LIDAR_MODE và NEAT_LIDAR_N; không đặt biến nào thì ("rays", 12)."""
    mode = os.environ.get("NEAT_LIDAR_MODE", "").strip().lower() or DEFAULT_LIDAR_MODE
    if mode not in LIDAR_MODES:
        raise ValueError(f"NEAT_LIDAR_MODE={mode!r}: chỉ nhận {LIDAR_MODES}")
    raw = os.environ.get("NEAT_LIDAR_N", "").strip() or str(DEFAULT_LIDAR_NUM_RAYS)
    try:
        n = int(raw)
    except ValueError as e:
        raise ValueError(f"NEAT_LIDAR_N={raw!r} không phải số nguyên") from e
    if n < 1 or (mode == "sector" and n < 2):
        raise ValueError(f"NEAT_LIDAR_N={n} quá nhỏ (tia: N >= 1, hình quạt: N >= 2)")
    return mode, n


LIDAR_MODE, LIDAR_NUM_RAYS = lidar_settings_from_env()   # mặc định "rays", 12
OBS_DIM: int = LIDAR_NUM_RAYS + 3   # N độ gần + rho, sin(phi), cos(phi)


def sensor_summary() -> dict | None:
    """Cảm biến của tiến trình này, để ghi vào summary.json: None khi là mặc định (12 tia
    điểm), nên summary của lượt chạy mặc định giữ nguyên như cũ; ngược lại {"mode", "n"}."""
    if (LIDAR_MODE, LIDAR_NUM_RAYS) == (DEFAULT_LIDAR_MODE, DEFAULT_LIDAR_NUM_RAYS):
        return None
    return {"mode": LIDAR_MODE, "n": LIDAR_NUM_RAYS}


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
