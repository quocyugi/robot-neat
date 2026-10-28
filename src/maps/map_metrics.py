"""
Map Metrics — Chỉ số độ khó của một bản đồ
============================================
Tính cùng một bộ chỉ số cho MỌI bản đồ (tự sinh, BARN, mê cung
Lehman–Stanley) để kiểm tra chất lượng dữ liệu, xếp cấp độ và báo cáo.

Các chỉ số (đơn vị mét nếu không ghi khác)
-------------------------------------------
- euclidean_distance        : khoảng cách đường chim bay S → G
- optimal_path_length_grid  : độ dài đường A* 8 hướng trên lưới 5 cm (cách tính gốc)
- optimal_path_length       : độ dài đường A* sau khi làm mượt (dùng làm L_opt cho SPL)
- tortuosity_grid / tortuosity : τ = độ dài đường / khoảng cách chim bay
- bottleneck_clearance      : khe hở hẹp nhất trên lối đi tốt nhất, tức bán kính
                              robot lớn nhất còn đi lọt từ S tới G
- mean_path_clearance       : khoảng cách trung bình tới vật cản dọc đường tối ưu
                              (giống chỉ số "distance to closest obstacle" của BARN)
- start_clearance, goal_clearance : khoảng cách từ S, G tới vật cản gần nhất
- greedy_stuck, greedy_progress   : robot "tham lam" chỉ bước tới ô làm giảm khoảng cách
                              tới đích có bị kẹt không, và kẹt khi đã đi được bao nhiêu %
- trap_depth                : nếu bị kẹt, robot phải lùi xa đích thêm bao nhiêu mét mới
                              thoát được cực tiểu địa phương (bẫy chữ U → giá trị lớn)

Reference
---------
- overview.md §5.1 bước 5 (đo độ khó: τ, khe hở hẹp nhất, cờ ngõ cụt)
- Perille et al. (2020), Benchmarking Metric Ground Navigation (chỉ số BARN)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import ndimage

from src.env.collision import min_obstacle_distance
from src.env.kinematics import GOAL_RADIUS
from src.maps.validator import (
    GRID_RES,
    INFLATION_RADIUS,
    _NEIGHBOURS_8,
    _clamp,
    _distance_field,
    _world_to_grid,
    validate_map,
)


# Dải τ của 4 cấp độ (giống generator._TAU_RANGES, dùng để xếp cấp bản đồ ngoài)
LEVEL_TAU_BOUNDS: list[float] = [1.05, 1.25, 1.60]

# Số lần chia đôi khi dò ngưỡng (25 lần → sai số cỡ 1e-7 m)
_BISECTION_STEPS: int = 25

# Lân cận 8 hướng cho hàm gán nhãn vùng liên thông (khớp với A* 8 hướng)
_STRUCTURE_8 = np.ones((3, 3), dtype=int)


def compute_map_metrics(
    segments,
    start,
    goal,
    validation: Optional[dict] = None,
) -> Optional[dict]:
    """Tính toàn bộ chỉ số cho một bản đồ.

    Parameters
    ----------
    segments : array-like, shape (M, 2, 2)
    start, goal : array-like, shape (2,)
    validation : dict, optional
        Kết quả ``validate_map`` đã chạy trước đó (để khỏi chạy A* lần nữa).

    Returns
    -------
    metrics : dict, hoặc None nếu không có đường đi hợp lệ.
        Thêm khoá "optimal_path" = danh sách điểm (x, y) của đường đã làm mượt.
    """
    segs = np.asarray(segments, dtype=np.float64)
    s = np.asarray(start, dtype=np.float64)
    g = np.asarray(goal, dtype=np.float64)

    # 1. Trường khoảng cách tới vật cản + lưới C-space (vật cản phình 0.30 m)
    dist_field = _distance_field(segs, GRID_RES)
    grid = dist_field <= INFLATION_RADIUS
    rows, cols = grid.shape

    # 2. A* + làm mượt (dùng lại kết quả nếu đã có)
    if validation is None:
        validation = validate_map(segs, s, g, grid=grid)
    if not validation["valid"]:
        return None

    start_cell = _clamp(*_world_to_grid(s, GRID_RES), rows, cols)
    goal_cell = _clamp(*_world_to_grid(g, GRID_RES), rows, cols)

    # 3. Khe hở hẹp nhất và độ thoáng dọc đường tối ưu
    bottleneck = _bottleneck_clearance(dist_field, start_cell, goal_cell)
    path_clear = _path_clearances(dist_field, validation["path_smooth"])

    # 4. Robot tham lam có bị kẹt không, kẹt sâu cỡ nào
    stuck, progress, stuck_cell = _greedy_descent(grid, start_cell, goal_cell)
    if stuck:
        depth = _trap_depth(grid, stuck_cell, goal_cell, g)
    else:
        depth = 0.0

    metrics = {
        "euclidean_distance": round(validation["euclid"], 4),
        "optimal_path_length": round(validation["length_smooth"], 4),
        "tortuosity": round(validation["tortuosity_smooth"], 4),
        "optimal_path_length_grid": round(validation["length"], 4),
        "tortuosity_grid": round(validation["tortuosity"], 4),
        "grid_resolution": GRID_RES,
        "bottleneck_clearance": round(bottleneck, 4),
        "mean_path_clearance": round(float(np.mean(path_clear)), 4),
        "start_clearance": round(min_obstacle_distance(s, segs), 4),
        "goal_clearance": round(min_obstacle_distance(g, segs), 4),
        "greedy_stuck": bool(stuck),
        "greedy_progress": round(progress, 4),
        "trap_depth": round(depth, 4),
    }
    # Đường tối ưu đã làm mượt (để vẽ hình và làm đường tham chiếu cho baseline)
    optimal_path = []
    for p in validation["path_smooth"]:
        optimal_path.append([round(p[0], 4), round(p[1], 4)])
    metrics["optimal_path"] = optimal_path
    return metrics


def assign_level_from_tau(tau: float) -> int:
    """Xếp cấp 1–4 theo τ (dùng cho bản đồ ngoài như BARN, mê cung Lehman)."""
    level = 1
    for bound in LEVEL_TAU_BOUNDS:
        if tau >= bound:
            level += 1
    return level


# ── Khe hở hẹp nhất ─────────────────────────────────────────────────────

def _bottleneck_clearance(
    dist_field: np.ndarray,
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
) -> float:
    """Ngưỡng t lớn nhất sao cho S và G còn thông nhau trong vùng {khoảng cách > t}.

    Dò t bằng chia đôi: với mỗi t, gán nhãn các vùng liên thông của
    các ô có khoảng cách tới vật cản > t rồi xem S, G có cùng nhãn không.
    """
    lo = 0.0
    hi = float(min(dist_field[start_cell], dist_field[goal_cell]))
    for _ in range(_BISECTION_STEPS):
        mid = (lo + hi) / 2.0
        labels, _ = ndimage.label(dist_field > mid, structure=_STRUCTURE_8)
        same_region = labels[start_cell] != 0 and labels[start_cell] == labels[goal_cell]
        if same_region:
            lo = mid
        else:
            hi = mid
    return lo


def _path_clearances(
    dist_field: np.ndarray, points: list[list[float]],
) -> np.ndarray:
    """Khoảng cách tới vật cản tại các điểm lấy mẫu mỗi 5 cm dọc đường đi."""
    rows, cols = dist_field.shape
    values = []
    for k in range(len(points) - 1):
        p0 = np.array(points[k])
        p1 = np.array(points[k + 1])
        n = max(int(np.linalg.norm(p1 - p0) / GRID_RES), 1)
        for i in range(n):
            p = p0 + (i / n) * (p1 - p0)
            r, c = _clamp(*_world_to_grid(p, GRID_RES), rows, cols)
            values.append(dist_field[r, c])
    # Điểm cuối (Goal)
    r, c = _clamp(*_world_to_grid(np.array(points[-1]), GRID_RES), rows, cols)
    values.append(dist_field[r, c])
    return np.array(values)


# ── Cờ ngõ cụt / bẫy đánh lừa ───────────────────────────────────────────

def _greedy_descent(
    grid: np.ndarray,
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
) -> tuple[bool, float, tuple[int, int]]:
    """Mô phỏng robot "tham lam": mỗi bước chỉ đi tới ô lân cận gần đích hơn.

    Đây đúng là hành vi mà hàm fitness thuần khoảng cách khuyến khích.
    Returns (bị kẹt?, tỉ lệ quãng đường đã rút ngắn, ô dừng lại).
    """
    rows, cols = grid.shape
    r, c = start_cell
    gr, gc = goal_cell
    d_start = math.hypot(r - gr, c - gc)
    d_now = d_start
    goal_cells = GOAL_RADIUS / GRID_RES   # bán kính đích tính theo số ô

    for _ in range(rows * cols):
        if d_now <= goal_cells:
            return False, 1.0, (r, c)
        best = None
        best_d = d_now
        for dr, dc in _NEIGHBOURS_8:
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if grid[nr, nc]:
                continue
            nd = math.hypot(nr - gr, nc - gc)
            if nd < best_d - 1e-9:
                best_d = nd
                best = (nr, nc)
        # Không còn ô nào gần đích hơn → kẹt ở cực tiểu địa phương
        if best is None:
            break
        r, c = best
        d_now = best_d

    progress = (d_start - d_now) / max(d_start, 1e-9)
    return True, progress, (r, c)


def _trap_depth(
    grid: np.ndarray,
    stuck_cell: tuple[int, int],
    goal_cell: tuple[int, int],
    goal: np.ndarray,
) -> float:
    """Độ sâu bẫy (m): phải lùi xa đích thêm bao nhiêu mới thoát được chỗ kẹt.

    Tìm ngưỡng T nhỏ nhất sao cho từ ô kẹt tới đích có đường đi chỉ qua các ô
    có khoảng cách tới đích <= T. Độ sâu = T - khoảng cách từ ô kẹt tới đích.
    Vách phẳng chắn ngang → vài chục cm; bẫy chữ U sâu 2 m → cỡ 2 m.
    """
    rows, cols = grid.shape
    ys = np.arange(rows) * GRID_RES + GRID_RES / 2
    xs = np.arange(cols) * GRID_RES + GRID_RES / 2
    xx, yy = np.meshgrid(xs, ys)
    dist_to_goal = np.sqrt((xx - goal[0]) ** 2 + (yy - goal[1]) ** 2)
    free = ~grid

    d_stuck = float(dist_to_goal[stuck_cell])
    lo = d_stuck
    hi = float(dist_to_goal[free].max())
    for _ in range(_BISECTION_STEPS):
        mid = (lo + hi) / 2.0
        labels, _ = ndimage.label(free & (dist_to_goal <= mid), structure=_STRUCTURE_8)
        same_region = labels[stuck_cell] != 0 and labels[stuck_cell] == labels[goal_cell]
        if same_region:
            hi = mid
        else:
            lo = mid
    return hi - d_stuck
