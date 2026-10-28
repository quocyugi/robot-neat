"""
Map Validation — A* Connectivity & Tortuosity
===============================================
Verifies that a generated map is solvable by running A* on a discretised
C-space grid, and computes the tortuosity ratio τ = L_A* / d_Euclid.

Reference
---------
- brainstorm/01_env/map_generator.md §3.1 (Decision #5)
- overview.md §5.1  (Pipeline step 4 — connectivity check)
"""

from __future__ import annotations

import heapq
from typing import Optional

import numpy as np

from src.env.kinematics import ROBOT_RADIUS, ARENA_W, ARENA_H


# ── Constants ────────────────────────────────────────────────────────────
GRID_RES: float = 0.05            # 5 cm per cell
INFLATION_RADIUS: float = ROBOT_RADIUS + 0.05   # 0.30 m  (Decision #5)


def validate_map(
    segments: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    grid_res: float = GRID_RES,
    inflation: float = INFLATION_RADIUS,
    grid: Optional[np.ndarray] = None,
) -> dict:
    """Run A* and return validation results.

    Returns
    -------
    dict with keys:
        "valid"   : bool       — True if a path exists
        "path"    : list       — list of (row, col) on the grid
        "length"  : float      — A* path length (metres)
        "euclid"  : float      — Euclidean distance S→G
        "tortuosity" : float   — τ = length / euclid
        "path_smooth"       : list  — các điểm (x, y) của đường A* đã làm mượt
        "length_smooth"     : float — độ dài đường đã làm mượt (m)
        "tortuosity_smooth" : float — τ tính trên đường đã làm mượt
    """
    # Có thể truyền sẵn lưới chiếm chỗ (map_metrics dùng) để khỏi dựng lại
    if grid is None:
        grid = _build_occupancy_grid(segments, grid_res, inflation)

    sr, sc = _world_to_grid(start, grid_res)
    gr, gc = _world_to_grid(goal,  grid_res)
    rows, cols = grid.shape

    # Clamp to grid bounds
    sr, sc = _clamp(sr, sc, rows, cols)
    gr, gc = _clamp(gr, gc, rows, cols)

    # Quick rejection: start or goal inside an obstacle
    if grid[sr, sc] or grid[gr, gc]:
        return _fail_result(start, goal)

    path = _astar(grid, (sr, sc), (gr, gc))
    if path is None:
        return _fail_result(start, goal)

    path_length = _path_length(path, grid_res)
    euclid = float(np.linalg.norm(goal - start))
    tau = path_length / max(euclid, 1e-6)

    # Đường A* 8 hướng chỉ đi được theo 8 góc nên dài hơn đường thật tới ~8%
    # (ví dụ hướng 22.5°). Làm mượt bằng kiểm tra "nhìn thẳng" để τ sát thực tế.
    smooth_pts = _smooth_path(grid, path, start, goal, grid_res)
    length_smooth = _polyline_length(smooth_pts)
    tau_smooth = length_smooth / max(euclid, 1e-6)

    return {
        "valid": True,
        "path": path,
        "length": path_length,
        "euclid": euclid,
        "tortuosity": tau,
        "path_smooth": smooth_pts,
        "length_smooth": length_smooth,
        "tortuosity_smooth": tau_smooth,
    }


# ── Occupancy grid ───────────────────────────────────────────────────────

def _distance_field(segments: np.ndarray, res: float) -> np.ndarray:
    """Khoảng cách (m) từ tâm mỗi ô lưới tới đoạn vật cản gần nhất."""
    segments = np.asarray(segments, dtype=np.float64)
    rows = int(ARENA_H / res)
    cols = int(ARENA_W / res)

    # For each cell, compute distance to nearest segment
    ys = np.arange(rows) * res + res / 2
    xs = np.arange(cols) * res + res / 2
    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx, yy], axis=-1).reshape(-1, 2)  # (N, 2)
    dist_min = np.full(points.shape[0], np.inf)

    for seg in segments:
        A = seg[0]
        B = seg[1]
        AB = B - A
        AB_sq = np.dot(AB, AB)
        if AB_sq < 1e-12:
            continue
        AP = points - A
        t = np.clip(np.dot(AP, AB) / AB_sq, 0.0, 1.0)[:, np.newaxis]
        closest = A + t * AB
        dists = np.linalg.norm(points - closest, axis=1)
        dist_min = np.minimum(dist_min, dists)

    return dist_min.reshape(rows, cols)


def _build_occupancy_grid(
    segments: np.ndarray,
    res: float,
    inflation: float,
) -> np.ndarray:
    """Create a boolean occupancy grid with inflated obstacles."""
    # Ô bị chiếm nếu tâm ô cách vật cản gần nhất <= bán kính phình (C-space)
    return _distance_field(segments, res) <= inflation


def _world_to_grid(pos: np.ndarray, res: float) -> tuple[int, int]:
    return int(pos[1] / res), int(pos[0] / res)


def _clamp(r: int, c: int, rows: int, cols: int) -> tuple[int, int]:
    return max(0, min(r, rows - 1)), max(0, min(c, cols - 1))


# ── A* search ────────────────────────────────────────────────────────────

_NEIGHBOURS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]
_DIAG_COST = 1.4142135623730951
_STRAIGHT_COST = 1.0


def _astar(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> Optional[list[tuple[int, int]]]:
    """8-connected A* on a boolean occupancy grid."""
    rows, cols = grid.shape
    g_score = np.full((rows, cols), np.inf)
    g_score[start] = 0.0

    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    open_set: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(open_set, (_heuristic(start, goal), start))
    closed = np.zeros((rows, cols), dtype=bool)

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return _reconstruct(came_from, current)
        if closed[current]:
            continue
        closed[current] = True

        cr, cc = current
        for dr, dc in _NEIGHBOURS_8:
            nr, nc = cr + dr, cc + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if grid[nr, nc] or closed[nr, nc]:
                continue
            cost = _DIAG_COST if (dr != 0 and dc != 0) else _STRAIGHT_COST
            tent = g_score[cr, cc] + cost
            if tent < g_score[nr, nc]:
                g_score[nr, nc] = tent
                came_from[(nr, nc)] = current
                f = tent + _heuristic((nr, nc), goal)
                heapq.heappush(open_set, (f, (nr, nc)))

    return None  # no path


def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Octile distance heuristic (admissible for 8-connected grid)."""
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return _STRAIGHT_COST * max(dr, dc) + (_DIAG_COST - _STRAIGHT_COST) * min(dr, dc)


def _reconstruct(
    came_from: dict, current: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return list(reversed(path))


def _path_length(path: list[tuple[int, int]], res: float) -> float:
    total = 0.0
    for i in range(1, len(path)):
        dr = path[i][0] - path[i - 1][0]
        dc = path[i][1] - path[i - 1][1]
        total += np.sqrt(dr * dr + dc * dc) * res
    return total


def _line_of_sight(
    grid: np.ndarray, a: tuple[int, int], b: tuple[int, int],
) -> bool:
    """True nếu đoạn thẳng nối tâm ô a và tâm ô b không đi qua ô bị chiếm."""
    r0, c0 = a
    r1, c1 = b
    # Lấy mẫu 4 điểm cho mỗi ô dọc theo đoạn thẳng rồi làm tròn về chỉ số ô
    n = 4 * max(abs(r1 - r0), abs(c1 - c0)) + 1
    rr = np.floor(np.linspace(r0, r1, n + 1) + 0.5).astype(int)
    cc = np.floor(np.linspace(c0, c1, n + 1) + 0.5).astype(int)
    return not bool(np.any(grid[rr, cc]))


def _smooth_path(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    start: np.ndarray,
    goal: np.ndarray,
    res: float,
) -> list[list[float]]:
    """Làm mượt đường A* (string pulling), trả về các điểm (x, y) tính bằng mét.

    Đi dọc đường A*: từ điểm neo hiện tại, nếu không còn "nhìn thẳng" được
    tới ô tiếp theo thì ô liền trước trở thành một điểm gãy của đường mượt.
    Điểm đầu/cuối dùng đúng toạ độ Start/Goal thay cho tâm ô.
    """
    points = [[float(start[0]), float(start[1])]]
    anchor = path[0]
    for k in range(1, len(path)):
        if not _line_of_sight(grid, anchor, path[k]):
            anchor = path[k - 1]
            # Đổi (row, col) về toạ độ thế giới: tâm của ô
            x = (anchor[1] + 0.5) * res
            y = (anchor[0] + 0.5) * res
            points.append([x, y])
    points.append([float(goal[0]), float(goal[1])])
    return points


def _polyline_length(points: list[list[float]]) -> float:
    """Tổng độ dài các đoạn thẳng nối liên tiếp các điểm."""
    total = 0.0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        total += float(np.sqrt(dx * dx + dy * dy))
    return total


def _fail_result(start: np.ndarray, goal: np.ndarray) -> dict:
    return {
        "valid": False,
        "path": [],
        "length": 0.0,
        "euclid": float(np.linalg.norm(goal - start)),
        "tortuosity": float("inf"),
        "path_smooth": [],
        "length_smooth": 0.0,
        "tortuosity_smooth": float("inf"),
    }
