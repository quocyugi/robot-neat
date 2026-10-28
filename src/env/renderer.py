"""
Pygame-based 2-D Renderer
==========================
Supports two Gymnasium render modes:
  - ``"human"``    → interactive window at 30 FPS
  - ``"rgb_array"`` → headless frame buffer for video export

Reference
---------
- brainstorm/01_env/gymnasium_api.md (Decision #7)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.env.kinematics import ARENA_W, ARENA_H, ROBOT_RADIUS, GOAL_RADIUS

# Lazy-import pygame so headless servers don't crash at import time
_pygame = None


def _ensure_pygame():
    global _pygame
    if _pygame is None:
        import pygame
        _pygame = pygame


# ── Colour Palette ───────────────────────────────────────────────────
_BG       = (30,  30,  35)
_WALL     = (180, 180, 190)
_OBSTACLE = (220, 80,  80)
_ROBOT    = (60,  180, 255)
_HEADING  = (255, 255, 255)
_GOAL     = (80,  220, 120)
_LIDAR    = (255, 255, 100, 80)
_TRAIL    = (100, 100, 255, 60)


class Renderer:
    """Lightweight Pygame renderer for :class:`RobotNavEnv`."""

    def __init__(
        self,
        render_mode: Optional[str],
        window_size: int = 600,
    ):
        self.render_mode = render_mode
        self.window_size = window_size
        self.scale = window_size / max(ARENA_W, ARENA_H)

        self._window = None
        self._clock = None
        self._surface = None

    # ------------------------------------------------------------------ #
    #  Coordinate transform                                                #
    # ------------------------------------------------------------------ #
    def _to_px(self, x: float, y: float) -> tuple[int, int]:
        """World (m) → pixel.  y is flipped so +y points up."""
        return int(x * self.scale), int(self.window_size - y * self.scale)

    def _scale_r(self, r: float) -> int:
        return max(int(r * self.scale), 1)

    # ------------------------------------------------------------------ #
    #  Main render call                                                    #
    # ------------------------------------------------------------------ #
    def render(
        self,
        segments: np.ndarray,
        robot_pos: np.ndarray,
        robot_theta: float,
        goal_pos: np.ndarray,
        trajectory: list[np.ndarray] | None = None,
        lidar_hits: np.ndarray | None = None,
        lidar_angles: np.ndarray | None = None,
        lidar_max_range: float = 3.0,
    ) -> np.ndarray | None:
        """Draw one frame.  Returns RGB array if mode is ``"rgb_array"``."""
        _ensure_pygame()
        pg = _pygame

        if self.render_mode is None:
            return None

        # Initialise window / surface on first call
        if self._surface is None:
            if self.render_mode == "human":
                pg.init()
                pg.display.set_caption("NEAT Robot Navigation")
                self._window = pg.display.set_mode(
                    (self.window_size, self.window_size)
                )
                self._clock = pg.time.Clock()
            self._surface = pg.Surface(
                (self.window_size, self.window_size)
            )

        canvas = self._surface
        canvas.fill(_BG)

        # ── Draw segments (walls + obstacles) ─────────────────────────
        for seg in segments:
            p1 = self._to_px(*seg[0])
            p2 = self._to_px(*seg[1])
            pg.draw.line(canvas, _WALL, p1, p2, 2)

        # ── Draw trajectory ───────────────────────────────────────────
        if trajectory and len(trajectory) >= 2:
            pts = [self._to_px(p[0], p[1]) for p in trajectory]
            pg.draw.lines(canvas, _TRAIL, False, pts, 1)

        # ── Draw LiDAR rays ──────────────────────────────────────────
        if lidar_hits is not None and lidar_angles is not None:
            rx, ry = float(robot_pos[0]), float(robot_pos[1])
            for prox, angle in zip(lidar_hits, lidar_angles):
                dist = (1.0 - prox) * lidar_max_range
                ex = rx + dist * np.cos(angle)
                ey = ry + dist * np.sin(angle)
                pg.draw.line(
                    canvas, _LIDAR,
                    self._to_px(rx, ry), self._to_px(ex, ey), 1,
                )

        # ── Draw goal ─────────────────────────────────────────────────
        gx, gy = float(goal_pos[0]), float(goal_pos[1])
        pg.draw.circle(
            canvas, _GOAL,
            self._to_px(gx, gy), self._scale_r(GOAL_RADIUS), 2,
        )

        # ── Draw robot ────────────────────────────────────────────────
        rpx, rpy = self._to_px(float(robot_pos[0]), float(robot_pos[1]))
        pg.draw.circle(canvas, _ROBOT, (rpx, rpy), self._scale_r(ROBOT_RADIUS))

        # Heading indicator
        hx = float(robot_pos[0]) + ROBOT_RADIUS * np.cos(robot_theta)
        hy = float(robot_pos[1]) + ROBOT_RADIUS * np.sin(robot_theta)
        pg.draw.line(canvas, _HEADING, (rpx, rpy), self._to_px(hx, hy), 2)

        # ── Display / return ──────────────────────────────────────────
        if self.render_mode == "human":
            self._window.blit(canvas, (0, 0))
            pg.display.flip()
            pg.event.pump()
            self._clock.tick(30)
            return None
        else:  # rgb_array
            return np.transpose(
                pg.surfarray.array3d(canvas), axes=(1, 0, 2)
            )

    # ------------------------------------------------------------------ #
    #  Cleanup                                                             #
    # ------------------------------------------------------------------ #
    def close(self):
        if self._window is not None:
            # Chỉ cần pygame khi đã mở cửa sổ (render_mode="human"); không vẽ thì
            # không import, để chạy huấn luyện / kiểm tra dữ liệu không cần pygame
            _ensure_pygame()
            _pygame.display.quit()
            _pygame.quit()
            self._window = None
        self._surface = None
