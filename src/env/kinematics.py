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

Robot và bố cục hình quạt (đợt 11, bản rút gọn; đọc cùng lúc, cùng quy tắc)
----------------------------------------------------------------------------
- NEAT_ROBOT = "old" (mặc định, rỗng cũng là "old"): robot đĩa của E4, mọi hằng và mọi đường
  chạy giữ nguyên từng bit. "waffle": TurtleBot3 Waffle Pi (bản rút gọn):
  * lệnh: o1, o2 kẹp về [-1; 1]; v_c = 0,26 max(o1; 0) (không lùi), w_c = 1,82 o2; bánh
    v_L = v_c - 0,1435 w_c, v_R = v_c + 0,1435 w_c, kẹp mỗi bánh về +-0,26669137 m/s
    (337 / 1263,632956882 của firmware OpenCR ROS 2); v = (v_L + v_R)/2, w = (v_R - v_L)/0,287;
    tích phân như E4 (theta' = theta + w dt; x' = x + v cos theta' dt; ...), float32 với hằng
    float32 tường minh (waffle_step, dùng chung cho đường từng lượt và đường theo lô);
  * thân: đĩa bao r = 0,2077 m đặt ở tâm thân, lùi 0,064 m sau tâm quay theo hướng thân;
    kiểm va chạm cuối mỗi bước như E4 (khoảng cách tâm thân tới đoạn <= r);
  * LiDAR: gốc quét ở tâm thân, tầm R = 3,5 m, chỉ kiểu "sector" (quạt chính xác);
  * đích: tâm quay cách đích <= 0,30 m; lượt 1.000 bước (100 s); xoay tại chỗ khi
    |v| < 0,025 m/s và |w| > 0,5 rad/s (trên (v, w) thật sau kẹp bánh).
- NEAT_SECTOR_LAYOUT = "" (mặc định: N quạt đều, quạt 0 giữa hướng trước như E4) hoặc "e32":
  30 quạt 9 độ trên 270 độ phía trước (từ phải sang trái, hướng trước là cạnh giữa quạt 14 và
  15) + quạt sau trái [135; 180] độ + quạt sau phải [-180; -135] độ; cần NEAT_LIDAR_MODE=sector
  và NEAT_LIDAR_N=32 (config neat_recurrent_n32.ini, 35 đầu vào).
- Hằng suy ra theo robot: MAX_STEPS, LIDAR_MAX_RANGE, COLL_RADIUS (bán kính đĩa va chạm),
  BODY_OFFSET (tâm thân lùi sau tâm quay). ROBOT_RADIUS giữ 0,25 cho mọi robot (công cụ bản đồ
  đọc hằng này); robot cũ có COLL_RADIUS = ROBOT_RADIUS, BODY_OFFSET = 0.
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

# ── Robot (NEAT_ROBOT) và bố cục hình quạt (NEAT_SECTOR_LAYOUT) ─────────────
ROBOTS: tuple[str, ...] = ("old", "waffle")
SECTOR_LAYOUTS: tuple[str, ...] = ("", "e32")      # "" = N quạt đều (E4)
E32_NUM_SECTORS: int = 32

# TurtleBot3 Waffle Pi (bản rút gọn của đợt 11; nguồn: ke_hoach_dot11.md mục 2–3)
WAFFLE_V_MAX: float = 0.26            # m/s, tốc độ tiến lớn nhất (thông số hãng)
WAFFLE_W_MAX: float = 1.82            # rad/s, tốc độ quay lớn nhất (thông số hãng)
WAFFLE_HALF_SEP: float = 0.1435       # m, nửa khoảng cách hai bánh
WAFFLE_SEP: float = 0.287             # m, khoảng cách hai bánh
WAFFLE_WHEEL_MAX: float = 0.26669137  # m/s, trần mỗi bánh = 337 / 1263.632956882 (OpenCR ROS 2), 8 chữ số
WAFFLE_MAX_STEPS: int = 1000          # bước (100 s)
WAFFLE_LIDAR_RANGE: float = 3.5       # m, tầm LDS-01
WAFFLE_COLL_RADIUS: float = 0.2077    # m, đĩa bao thân 0,281 x 0,306 m (nửa đường chéo)
WAFFLE_BODY_OFFSET: float = 0.064     # m, tâm thân lùi sau tâm quay theo hướng thân
WAFFLE_SPIN_V: float = 0.025          # m/s: |v| nhỏ hơn ...
WAFFLE_SPIN_W: float = 0.5            # rad/s: ... và |w| lớn hơn thì là xoay tại chỗ


def robot_settings_from_env() -> tuple[str, str, int, str]:
    """(robot, kiểu cảm biến, N, bố cục quạt) từ NEAT_ROBOT, NEAT_LIDAR_MODE, NEAT_LIDAR_N, NEAT_SECTOR_LAYOUT.

    Không đặt biến nào: ("old", "rays", 12, ""). Báo lỗi với tổ hợp không dùng được: "e32" cần kiểu
    "sector" và N = 32; Waffle chỉ dùng kiểu "sector" (quạt chính xác, không bắn tia)."""
    robot = os.environ.get("NEAT_ROBOT", "").strip().lower() or "old"
    if robot not in ROBOTS:
        raise ValueError(f"NEAT_ROBOT={robot!r}: chỉ nhận {ROBOTS} (rỗng = old)")
    layout = os.environ.get("NEAT_SECTOR_LAYOUT", "").strip().lower()
    if layout == "uniform":
        layout = ""
    if layout not in SECTOR_LAYOUTS:
        raise ValueError(f"NEAT_SECTOR_LAYOUT={layout!r}: chỉ nhận '' (quạt đều) hoặc 'e32'")
    mode, n = lidar_settings_from_env()
    if layout == "e32" and (mode != "sector" or n != E32_NUM_SECTORS):
        raise ValueError(f"NEAT_SECTOR_LAYOUT=e32 cần NEAT_LIDAR_MODE=sector và NEAT_LIDAR_N={E32_NUM_SECTORS} "
                         f"(đang là {mode}, {n})")
    if robot == "waffle" and mode != "sector":
        raise ValueError("NEAT_ROBOT=waffle chỉ dùng cảm biến hình quạt chính xác: đặt NEAT_LIDAR_MODE=sector "
                         f"(đang là {mode!r})")
    return robot, mode, n, layout


ROBOT, _MODE_CHECK, _N_CHECK, SECTOR_LAYOUT = robot_settings_from_env()
assert (_MODE_CHECK, _N_CHECK) == (LIDAR_MODE, LIDAR_NUM_RAYS)

if ROBOT == "waffle":
    MAX_STEPS = WAFFLE_MAX_STEPS
    LIDAR_MAX_RANGE = WAFFLE_LIDAR_RANGE
    COLL_RADIUS: float = WAFFLE_COLL_RADIUS
    BODY_OFFSET: float = WAFFLE_BODY_OFFSET
else:                                     # robot cũ: mọi hằng như E4
    COLL_RADIUS = ROBOT_RADIUS
    BODY_OFFSET = 0.0

# Hằng float32 của động học Waffle (ghi một lần, mọi đường đọc cùng giá trị)
_F32 = np.float32
W_VMAX32, W_WMAX32 = _F32(WAFFLE_V_MAX), _F32(WAFFLE_W_MAX)
W_HB32, W_SEP32, W_WM32 = _F32(WAFFLE_HALF_SEP), _F32(WAFFLE_SEP), _F32(WAFFLE_WHEEL_MAX)
W_SPIN_V32, W_SPIN_W32 = _F32(WAFFLE_SPIN_V), _F32(WAFFLE_SPIN_W)
W_DT32, W_ZERO32, W_ONE32, W_TWO32 = _F32(DT), _F32(0.0), _F32(1.0), _F32(2.0)


def waffle_command(xp, a32):
    """(v, w) thật của Waffle (float32, m/s và rad/s) sau kẹp bánh, từ đầu ra a32 (A, 2) float32.

    Thứ tự phép cố định (đường từng lượt và theo lô gọi đúng hàm này):
      a = clip(a32, -1, 1); v_c = 0,26 * max(o1, 0); w_c = 1,82 * o2;
      v_L = clip(v_c - w_c * 0,1435, +-W); v_R = clip(v_c + w_c * 0,1435, +-W);
      v = (v_L + v_R) / 2; w = (v_R - v_L) / 0,287.
    max(o1, 0) viết bằng where(o1 > 0, o1, 0) để kết quả không phụ thuộc cách NumPy xử lý -0.0."""
    a = xp.clip(a32, -W_ONE32, W_ONE32)
    o1, o2 = a[:, 0], a[:, 1]
    vc = W_VMAX32 * xp.where(o1 > W_ZERO32, o1, W_ZERO32)
    wc = W_WMAX32 * o2
    vl = xp.clip(vc - wc * W_HB32, -W_WM32, W_WM32)
    vr = xp.clip(vc + wc * W_HB32, -W_WM32, W_WM32)
    v = (vl + vr) / W_TWO32
    w = (vr - vl) / W_SEP32
    return v, w


def waffle_step(xp, cosf, sinf, x, y, th32, a32):
    """Một bước của A lượt Waffle: trả (x mới, y mới float64; hướng mới float32; cờ xoay tại chỗ bool).

    x, y: float64 (A,) tâm quay; th32: hướng float32 (A,) (bước 1: float32 của hướng xuất phát);
    a32: đầu ra (A, 2) float32; cosf, sinf: cos, sin trên mảng float32 (np.cos / be.cos).
    Tích phân như E4: th = th32 + w dt; x + v cos(th) dt; y + v sin(th) dt (float32 rồi float64 + float32)."""
    v, w = waffle_command(xp, a32)
    spin = (xp.abs(v) < W_SPIN_V32) & (xp.abs(w) > W_SPIN_W32)
    th = th32 + w * W_DT32
    xn = x + v * cosf(th) * W_DT32
    yn = y + v * sinf(th) * W_DT32
    return xn, yn, th, spin


def sensor_summary() -> dict | None:
    """Cảm biến của tiến trình này, để ghi vào summary.json: None khi là mặc định (12 tia
    điểm), nên summary của lượt chạy mặc định giữ nguyên như cũ; ngược lại {"mode", "n"}.
    Bố cục quạt khác đều thêm "layout"; Waffle thêm "layout" (kể cả "uniform"), "robot" và các hằng."""
    if ROBOT == "waffle":
        return {"mode": LIDAR_MODE, "n": LIDAR_NUM_RAYS, "layout": SECTOR_LAYOUT or "uniform", "robot": "waffle",
                "waffle": {"v_max": WAFFLE_V_MAX, "w_max": WAFFLE_W_MAX, "half_sep": WAFFLE_HALF_SEP,
                           "sep": WAFFLE_SEP, "wheel_max": WAFFLE_WHEEL_MAX, "max_steps": WAFFLE_MAX_STEPS,
                           "lidar_range": WAFFLE_LIDAR_RANGE, "coll_radius": WAFFLE_COLL_RADIUS,
                           "body_offset": WAFFLE_BODY_OFFSET, "goal_radius": GOAL_RADIUS, "dt": DT,
                           "spin_v": WAFFLE_SPIN_V, "spin_w": WAFFLE_SPIN_W, "reverse": False}}
    if (LIDAR_MODE, LIDAR_NUM_RAYS, SECTOR_LAYOUT) == (DEFAULT_LIDAR_MODE, DEFAULT_LIDAR_NUM_RAYS, ""):
        return None
    out = {"mode": LIDAR_MODE, "n": LIDAR_NUM_RAYS}
    if SECTOR_LAYOUT:
        out["layout"] = SECTOR_LAYOUT
    return out


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
