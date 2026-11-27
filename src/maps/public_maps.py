"""
Public Map Datasets — BARN & Lehman–Stanley Mazes
==================================================
Đọc bản đồ từ 2 bộ dữ liệu công khai và chuyển về đúng schema JSON mà
RobotNavEnv / validator của dự án đang dùng (bounds, start, goal, segments).

1. BARN — Benchmark for Autonomous Robot Navigation (Perille et al., 2020)
   300 môi trường Gazebo cho robot Jackal (0.508 m × 0.430 m). Mỗi vật cản là
   một trụ tròn r = 0.075 m đặt ở tâm một ô lưới 0.15 m. Cả world là lưới
   30 ô (trục x) × 64 ô (trục y):
     - cột i = 0 và i = 29 : tường hai bên hành lang
     - hàng j = 0          : tường phía sau điểm xuất phát
     - hàng j = 34..63     : vùng vật cản sinh bằng cellular automaton (30 × 30 ô)
   Robot xuất phát tại (-2.25, 3.0) quay mặt về +y, đích ở (-2.25, 13.0).

2. Mê cung "medium" và "hard" của Lehman & Stanley (2011), dùng trong bài báo
   Novelty Search gốc. Mỗi tường là một đoạn thẳng (đơn vị pixel), robot gốc
   có bán kính 8 đơn vị.

Reference
---------
- overview.md §5.2 (Tập Test đặc biệt: 2 mê cung Lehman & Stanley + 50 bản đồ BARN)
- overview.md §6.1 Q3 (tổng quát hoá zero-shot sang bản đồ BARN chưa từng thấy)
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from src.env.kinematics import ARENA_W, ARENA_H, ROBOT_RADIUS
from src.env.lidar import arena_wall_segments
from src.maps.validator import INFLATION_RADIUS


# ── Hằng số của BARN ─────────────────────────────────────────────────────
BARN_CELL: float = 0.15            # m, cạnh 1 ô lưới (= đường kính trụ)
BARN_NX: int = 30                  # số ô theo trục x (bề ngang hành lang)
BARN_NY: int = 64                  # số ô theo trục y (hướng robot chạy)
BARN_ORIGIN = (-4.5, 0.0)          # góc dưới-trái của ô (0, 0) trong toạ độ Gazebo
BARN_FIELD_J0: int = 34            # hàng đầu tiên của vùng vật cản
BARN_FIELD_LEN: int = 30           # số hàng của vùng vật cản
BARN_RUN_UP: float = 2.1           # m, từ điểm xuất phát (y = 3.0) tới mép vùng vật cản (y = 5.1)
BARN_START_HEADING: float = math.pi / 2   # robot BARN quay mặt về +y

# ── Hằng số của mê cung Lehman–Stanley ──────────────────────────────────
LEHMAN_ROBOT_RADIUS: float = 8.0   # bán kính robot trong mô phỏng gốc (pixel)


# ======================================================================
#  BARN
# ======================================================================

def read_barn_world(world_path: str | Path) -> np.ndarray:
    """Đọc file .world của BARN, trả về lưới chiếm chỗ (30, 64) kiểu bool.

    grid[i, j] = True nếu ô (i, j) có trụ vật cản; i theo trục x, j theo trục y.
    """
    root = ET.parse(world_path).getroot()
    world = root.find("world")
    grid = np.zeros((BARN_NX, BARN_NY), dtype=bool)

    # Chỉ lấy các model là con trực tiếp của <world> (bỏ qua phần <state>)
    for model in world.findall("model"):
        name = model.get("name", "")
        if not name.startswith("unit_cylinder"):
            continue
        pose = model.find("pose").text.split()
        x = float(pose[0])
        y = float(pose[1])
        # Toạ độ tâm trụ → chỉ số ô
        i = int(round((x - BARN_ORIGIN[0]) / BARN_CELL - 0.5))
        j = int(round((y - BARN_ORIGIN[1]) / BARN_CELL - 0.5))
        if 0 <= i < BARN_NX and 0 <= j < BARN_NY:
            grid[i, j] = True
        else:
            raise ValueError(f"{world_path}: trụ {name} nằm ngoài lưới 30x64")
    return grid


def barn_to_map(
    grid: np.ndarray,
    world_index: int,
    start_y: float = 1.05,
    exit_length: float = 1.5,
) -> dict:
    """Đặt một môi trường BARN vào arena 10 m × 10 m, giữ nguyên tỉ lệ 1:1.

    Giữ tỉ lệ thật vì robot Jackal của BARN cỡ ngang robot của nhóm
    (r = 0.25 m). Phần chạy đà phía trước vùng vật cản được cắt bớt để
    vừa arena:
      - hành lang 4.5 m đặt giữa arena theo x (x từ 2.75 tới 7.25)
      - Start cách mép vùng vật cản 2.1 m như BARN gốc, cách tường dưới ``start_y``
      - Goal ở sau vùng vật cản ``exit_length`` mét (BARN gốc là 3.4 m,
        không đủ chỗ trong arena 10 m)
    """
    cell = BARN_CELL
    # Số hàng giữ lại phía trước vùng vật cản (chạy đà + khoảng tới tường dưới)
    n_before = int(round((start_y + BARN_RUN_UP) / cell))
    j_crop = BARN_FIELD_J0 - n_before
    sub_grid = grid[:, j_crop:]

    # Hàng j_crop đặt sát tường dưới (y = 0), hành lang canh giữa theo x
    origin_x = ARENA_W / 2 - BARN_NX * cell / 2
    origin_y = 0.0
    obstacle_segs = grid_to_segments(sub_grid, cell, origin_x, origin_y)

    field_y0 = n_before * cell
    field_y1 = field_y0 + BARN_FIELD_LEN * cell
    start = [ARENA_W / 2, round(field_y0 - BARN_RUN_UP, 4)]
    goal = [ARENA_W / 2, round(field_y1 + exit_length, 4)]

    segments = arena_wall_segments(ARENA_W, ARENA_H).tolist() + obstacle_segs

    # Mật độ vật cản trong vùng CA (bỏ 2 cột tường hai bên)
    field = grid[1:BARN_NX - 1, BARN_FIELD_J0:BARN_FIELD_J0 + BARN_FIELD_LEN]
    density = float(field.mean())

    return {
        "map_id": f"barn_{world_index:03d}",
        "source": "BARN",
        "source_index": world_index,
        "difficulty_level": None,          # điền sau khi tính τ
        "seed": None,
        "bounds": [ARENA_W, ARENA_H],
        "start": start,
        "goal": goal,
        "start_heading": BARN_START_HEADING,
        "robot_radius": ROBOT_RADIUS,
        "inflation_radius": INFLATION_RADIUS,
        "source_info": {
            "world_file": f"world_{world_index}.world",
            "cell_size": cell,
            "obstacle_density": round(density, 4),
            "placement": "BARN 1:1, cắt bớt đoạn chạy đà, hành lang canh giữa arena",
        },
        "segments": segments,
    }


def grid_to_segments(
    grid: np.ndarray,
    cell: float,
    origin_x: float,
    origin_y: float,
) -> list[list[list[float]]]:
    """Chuyển lưới chiếm chỗ thành các đoạn thẳng bao quanh vùng vật cản.

    Mỗi cạnh nằm giữa một ô vật cản và một ô trống là một phần đường biên.
    Các cạnh liền nhau trên cùng một đường thẳng được gộp thành một đoạn dài
    để LiDAR (12 tia × M đoạn) chạy nhanh. Ngoài lưới coi như ô trống.
    """
    n_i, n_j = grid.shape
    segs = []

    # Cạnh dọc tại x = origin_x + i*cell, nằm giữa ô (i-1, j) và ô (i, j)
    for i in range(n_i + 1):
        j = 0
        while j < n_j:
            if _is_vertical_edge(grid, i, j):
                j_begin = j
                while j < n_j and _is_vertical_edge(grid, i, j):
                    j += 1
                x = origin_x + i * cell
                segs.append([[x, origin_y + j_begin * cell], [x, origin_y + j * cell]])
            else:
                j += 1

    # Cạnh ngang tại y = origin_y + j*cell, nằm giữa ô (i, j-1) và ô (i, j)
    for j in range(n_j + 1):
        i = 0
        while i < n_i:
            if _is_horizontal_edge(grid, i, j):
                i_begin = i
                while i < n_i and _is_horizontal_edge(grid, i, j):
                    i += 1
                y = origin_y + j * cell
                segs.append([[origin_x + i_begin * cell, y], [origin_x + i * cell, y]])
            else:
                i += 1

    # Làm tròn để file JSON gọn
    for seg in segs:
        for p in seg:
            p[0] = round(p[0], 4)
            p[1] = round(p[1], 4)
    return segs


def _cell_value(grid: np.ndarray, i: int, j: int) -> bool:
    """Giá trị ô (i, j); ngoài lưới coi như trống."""
    if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
        return bool(grid[i, j])
    return False


def _is_vertical_edge(grid: np.ndarray, i: int, j: int) -> bool:
    return _cell_value(grid, i - 1, j) != _cell_value(grid, i, j)


def _is_horizontal_edge(grid: np.ndarray, i: int, j: int) -> bool:
    return _cell_value(grid, i, j - 1) != _cell_value(grid, i, j)


# ======================================================================
#  Mê cung Lehman & Stanley
# ======================================================================

def read_lehman_maze(txt_path: str | Path) -> dict:
    """Đọc file mê cung dạng của Lehman & Stanley (medium_maze.txt, hard_maze.txt).

    Định dạng (bỏ dòng trống và dòng bắt đầu bằng #):
        dòng 1 : số tường N
        dòng 2 : toạ độ xuất phát  x y
        dòng 3 : góc hướng ban đầu (độ)
        dòng 4 : toạ độ lối ra (đích)  x y
        N dòng : mỗi tường  x1 y1 x2 y2
    """
    lines = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for raw in f:
            text = raw.strip()
            if text == "" or text.startswith("#"):
                continue
            lines.append(text)

    n_walls = int(lines[0])
    start = [float(v) for v in lines[1].split()]
    heading_deg = float(lines[2])
    goal = [float(v) for v in lines[3].split()]

    walls = []
    for k in range(n_walls):
        values = [float(v) for v in lines[4 + k].split()]
        walls.append(values)
    if len(lines) != 4 + n_walls:
        raise ValueError(f"{txt_path}: số dòng tường không khớp với N = {n_walls}")

    return {"start": start, "heading_deg": heading_deg, "goal": goal, "walls": walls}


def lehman_to_map(maze: dict, name: str) -> dict:
    """Đặt mê cung vào giữa arena với tỉ lệ sao cho robot gốc (8 px) = robot nhóm (0.25 m).

    Tỉ lệ 0.25 / 8 = 0.03125 m/pixel giữ nguyên độ rộng hành lang so với
    kích thước robot như trong bài báo. File gốc dùng toạ độ màn hình (trục y
    hướng xuống) nên lật trục y để hình giống hình trong bài báo; lật gương
    không làm thay đổi độ khó.
    """
    scale = ROBOT_RADIUS / LEHMAN_ROBOT_RADIUS

    # Khung bao của toàn bộ tường
    xs = []
    ys = []
    for w in maze["walls"]:
        xs.append(w[0])
        xs.append(w[2])
        ys.append(w[1])
        ys.append(w[3])
    center = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    width_m = (max(xs) - min(xs)) * scale
    height_m = (max(ys) - min(ys)) * scale
    if width_m > ARENA_W or height_m > ARENA_H:
        raise ValueError(f"{name}: mê cung {width_m:.2f} x {height_m:.2f} m không vừa arena")

    maze_segs = []
    for w in maze["walls"]:
        p1 = _px_to_arena(w[0], w[1], center, scale)
        p2 = _px_to_arena(w[2], w[3], center, scale)
        maze_segs.append([p1, p2])
    segments = arena_wall_segments(ARENA_W, ARENA_H).tolist() + maze_segs

    return {
        "map_id": f"lehman_{name}",
        "source": "Lehman-Stanley",
        "source_index": None,
        "difficulty_level": None,          # điền sau khi tính τ
        "seed": None,
        "bounds": [ARENA_W, ARENA_H],
        "start": _px_to_arena(maze["start"][0], maze["start"][1], center, scale),
        "goal": _px_to_arena(maze["goal"][0], maze["goal"][1], center, scale),
        # lật trục y thì góc hướng cũng đổi dấu
        "start_heading": -math.radians(maze["heading_deg"]) + 0.0,
        "robot_radius": ROBOT_RADIUS,
        "inflation_radius": INFLATION_RADIUS,
        "source_info": {
            "maze_file": f"{name}.txt",
            "scale_m_per_px": scale,
            "maze_size_m": [round(width_m, 3), round(height_m, 3)],
            "placement": "co giãn 0.25/8 m mỗi pixel, lật trục y, đặt giữa arena",
        },
        "segments": segments,
    }


def _px_to_arena(
    px: float, py: float, center: tuple[float, float], scale: float,
) -> list[float]:
    """Đổi 1 điểm (pixel, y hướng xuống) sang toạ độ arena (mét, y hướng lên).

    Tâm mê cung trùng tâm arena.
    """
    x = (px - center[0]) * scale + ARENA_W / 2
    y = (center[1] - py) * scale + ARENA_H / 2
    return [round(x, 4), round(y, 4)]
