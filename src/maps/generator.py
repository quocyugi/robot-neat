"""
Procedural Map Generator — 4 Difficulty Levels
================================================
Generates random 2-D maps with obstacles, validates them via A*,
and exports to JSON for reproducible training.

Reference
---------
- brainstorm/01_env/map_generator.md  (Decisions #4, #5)
- overview.md §5.1–5.2

Level definitions
-----------------
Level 1 — Empty:     walls only, τ ≈ 1.00
Level 2 — Sparse:    5–8 scattered blocks/cylinders, τ ∈ [1.05, 1.25]
Level 3 — Corridors: zig-zag walls with narrow gaps, τ ∈ (1.25, 1.60]
Level 4 — U-Traps:   1–2 oriented U-traps + scattered, τ > 1.60

Sửa đổi khi làm tiền xử lý dữ liệu (xem docs/tien_xu_ly_du_lieu.md)
--------------------------------------------------------------------
1. Hạt giống mỗi lần thử = (level, seed, attempt): hết trùng giữa train/val/test.
2. Cấp 4: chọn S, G trước rồi mới đặt bẫy chữ U chắn trên trục S→G; độ sâu ≥ độ rộng.
3. Cấp 4: thêm 3 VẬT CẢN rải rác (code cũ lấy nhầm 3 cạnh của 1 vật cản).
4. S, G cách vật cản tối thiểu r + 0.2 m như overview §5.1.
5. Lọc dải τ bằng đường A* đã làm mượt; lưu thêm bộ chỉ số độ khó (map_metrics).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from src.env.collision import check_collision
from src.env.kinematics import ARENA_W, ARENA_H, ROBOT_RADIUS
from src.env.lidar import arena_wall_segments, circle_to_segments
from src.maps.map_metrics import compute_map_metrics
from src.maps.utrap import generate_u_trap
from src.maps.validator import INFLATION_RADIUS, validate_map


# ── Tortuosity thresholds per level ──────────────────────────────────────
_TAU_RANGES: dict[int, tuple[float, float]] = {
    1: (0.99, 1.01),     # essentially straight line
    2: (1.05, 1.25),
    3: (1.25, 1.60),
    4: (1.60, 5.00),     # upper bound generous for U-traps
}

_MIN_SG_DIST: float = 5.0   # minimum Start–Goal Euclidean distance
_SG_CLEARANCE: float = ROBOT_RADIUS + 0.2   # S, G cách vật cản tối thiểu r + 0.2 m

# Số lần thử tối đa cho mỗi seed. Cấp 3 cần trung bình ~11 lần thử mới đạt dải τ,
# cấp 4 cần τ > 1.6 nên chỉ ~1.5% lần thử đạt → cho thử nhiều hơn.
_MAX_RETRIES: dict[int, int] = {1: 50, 2: 50, 3: 100, 4: 600}


class MapGenerator:
    """Procedural map factory with seeded RNG for reproducibility."""

    def __init__(self, arena_w: float = ARENA_W, arena_h: float = ARENA_H):
        self.arena_w = arena_w
        self.arena_h = arena_h
        self.wall_segs = arena_wall_segments(arena_w, arena_h)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        level: int,
        seed: int,
        max_retries: Optional[int] = None,
    ) -> Optional[dict]:
        """Generate and validate a single map.

        Returns None if no valid map could be produced within *max_retries*.
        """
        if max_retries is None:
            max_retries = _MAX_RETRIES[level]
        for attempt in range(max_retries):
            # Mỗi bộ (level, seed, attempt) có dòng số ngẫu nhiên riêng.
            # Cách cũ seed + attempt*100_000 làm lần thử thứ 3 của seed train s
            # trùng lần thử đầu của seed test 200_000 + s → bản đồ test lọt vào train.
            rng = np.random.default_rng([level, seed, attempt])
            result = self._try_generate(level, rng, seed)
            if result is not None:
                return result
        return None

    def generate_dataset(
        self,
        level: int,
        seeds: range,
        out_dir: str | Path,
    ) -> list[str]:
        """Batch-generate maps, saving each to ``out_dir/map_{seed}.json``.

        Returns list of file paths written.
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        written = []

        for seed in seeds:
            data = self.generate(level, seed)
            if data is None:
                continue
            fp = out_path / f"map_{seed:06d}.json"
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            written.append(str(fp))

        return written

    # ------------------------------------------------------------------ #
    #  Internal — level-specific generation                                #
    # ------------------------------------------------------------------ #

    def _try_generate(
        self, level: int, rng: np.random.Generator, seed: int,
    ) -> Optional[dict]:
        """Attempt a single generation + validation pass."""

        obstacle_segs: list[list[list[float]]] = []
        start = None
        goal = None

        if level == 1:
            pass  # Empty arena — no internal obstacles
        elif level == 2:
            obstacle_segs = self._gen_sparse(rng)
        elif level == 3:
            obstacle_segs = self._gen_corridors(rng)
        elif level == 4:
            # Chọn S, G trước rồi đặt bẫy chữ U chắn ngay trên trục S→G.
            # Code cũ hướng bẫy theo S/G tạm (1.5,1.5)→(8.5,8.5) rồi mới rải S, G
            # ngẫu nhiên nên bẫy hiếm khi chắn đường (chỉ ~1/3 seed sinh được bản đồ).
            start, goal = self._place_start_goal(rng, self.wall_segs)
            if start is None:
                return None
            obstacle_segs = self._gen_utraps(rng, start, goal)
        else:
            raise ValueError(f"Unknown level: {level}")

        # Merge wall segments + obstacle segments
        obs_arr = (
            np.array(obstacle_segs, dtype=np.float64)
            if obstacle_segs
            else np.empty((0, 2, 2), dtype=np.float64)
        )
        all_segs = np.concatenate([self.wall_segs, obs_arr], axis=0)

        # Generate Start / Goal
        if start is None:
            start, goal = self._place_start_goal(rng, all_segs)
            if start is None:
                return None
        elif not (self._is_clear(start, all_segs) and self._is_clear(goal, all_segs)):
            return None  # bẫy / vật cản vừa đặt đè quá sát S hoặc G

        # Validate with A*
        val = validate_map(all_segs, start, goal)
        if not val["valid"]:
            return None

        # Lọc theo dải τ của cấp: dùng τ trên đường A* đã làm mượt
        tau = val["tortuosity_smooth"]
        lo, hi = _TAU_RANGES[level]
        if not (lo <= tau <= hi):
            return None

        # Đo đủ bộ chỉ số độ khó (dùng lại kết quả A* ở trên)
        metrics = compute_map_metrics(all_segs, start, goal, validation=val)
        optimal_path = metrics.pop("optimal_path")

        # Build output dict
        return {
            "map_id": f"lvl{level}_seed_{seed:06d}",
            "difficulty_level": level,
            "seed": seed,
            "source": "generator",
            "bounds": [self.arena_w, self.arena_h],
            "start": start.tolist(),
            "goal": goal.tolist(),
            "robot_radius": ROBOT_RADIUS,
            "inflation_radius": INFLATION_RADIUS,
            "metrics": metrics,
            "optimal_path": optimal_path,
            "segments": all_segs.tolist(),
        }

    # ── Level 2: Sparse ──────────────────────────────────────────────

    def _gen_sparse(
        self, rng: np.random.Generator, n_obstacles: Optional[int] = None,
    ) -> list:
        segs = []
        if n_obstacles is None:
            n_obstacles = rng.integers(5, 9)
        for _ in range(n_obstacles):
            if rng.random() < 0.5:
                # Rectangle
                cx = rng.uniform(1.5, self.arena_w - 1.5)
                cy = rng.uniform(1.5, self.arena_h - 1.5)
                hw = rng.uniform(0.3, 0.8)
                hh = rng.uniform(0.3, 0.8)
                segs.extend(self._rect_segments(cx, cy, hw, hh))
            else:
                # Circle → octagon
                cx = rng.uniform(1.5, self.arena_w - 1.5)
                cy = rng.uniform(1.5, self.arena_h - 1.5)
                r = rng.uniform(0.3, 0.6)
                oct = circle_to_segments(np.array([cx, cy]), r, num_sides=8)
                segs.extend(oct.tolist())
        return segs

    # ── Level 3: Corridors ───────────────────────────────────────────

    def _gen_corridors(self, rng: np.random.Generator) -> list:
        segs = []
        n_walls = rng.integers(3, 6)
        for i in range(n_walls):
            # Alternating horizontal walls from left/right
            if i % 2 == 0:
                x1 = 0.5
                x2 = rng.uniform(5.0, 8.5)
            else:
                x1 = rng.uniform(1.5, 5.0)
                x2 = self.arena_w - 0.5
            y = rng.uniform(1.5, self.arena_h - 1.5)
            segs.append([[x1, y], [x2, y]])
        return segs

    # ── Level 4: U-Traps ─────────────────────────────────────────────

    def _gen_utraps(
        self, rng: np.random.Generator, start: np.ndarray, goal: np.ndarray,
    ) -> list:
        segs = []
        # Bẫy hướng theo đúng trục Start → Goal của bản đồ này
        n_traps = rng.integers(1, 3)
        for i in range(n_traps):
            ratio = rng.uniform(0.35, 0.65)
            width = rng.uniform(1.5, 2.5)
            # Decision #22: độ sâu >= độ rộng (Depth / Width >= 1.0)
            depth = rng.uniform(width, 2.5)
            trap = generate_u_trap(
                start, goal,
                trap_width=width, trap_depth=depth,
                position_ratio=ratio,
            )
            segs.extend(trap)

        # Thêm 3 vật cản rải rác (code cũ dùng [:3] nên chỉ lấy 3 CẠNH của 1 vật cản)
        segs.extend(self._gen_sparse(rng, n_obstacles=3))
        return segs

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _rect_segments(
        cx: float, cy: float, hw: float, hh: float,
    ) -> list[list[list[float]]]:
        """Axis-aligned rectangle → 4 edge segments."""
        tl = [cx - hw, cy + hh]
        tr = [cx + hw, cy + hh]
        br = [cx + hw, cy - hh]
        bl = [cx - hw, cy - hh]
        return [[tl, tr], [tr, br], [br, bl], [bl, tl]]

    def _place_start_goal(
        self,
        rng: np.random.Generator,
        segments: np.ndarray,
        max_tries: int = 100,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Sample S and G such that d(S,G) ≥ 5 m and both are obstacle-free."""
        margin = ROBOT_RADIUS + 0.2
        for _ in range(max_tries):
            s = rng.uniform(margin, [self.arena_w - margin, self.arena_h - margin])
            g = rng.uniform(margin, [self.arena_w - margin, self.arena_h - margin])
            if np.linalg.norm(s - g) < _MIN_SG_DIST:
                continue
            if not self._is_clear(s, segments):
                continue
            if not self._is_clear(g, segments):
                continue
            return s, g
        return None, None

    @staticmethod
    def _is_clear(p: np.ndarray, segments: np.ndarray) -> bool:
        """True nếu điểm p cách mọi đoạn vật cản hơn r + 0.2 m (overview §5.1)."""
        return not check_collision(p, segments, _SG_CLEARANCE)
