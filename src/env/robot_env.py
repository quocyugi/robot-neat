"""
RobotNavEnv — Gymnasium Environment for 2-D Robot Navigation
==============================================================
A differential-drive robot must navigate from *start* to *goal*
inside a 10 m × 10 m arena cluttered with static obstacles,
using only 12 local LiDAR rays and relative goal bearing.

Observation space (15-D continuous)
-----------------------------------
  0–11 : proximity indices  (0 = safe, 1 = contact)
  12   : normalised distance to goal  ρ ∈ [0, 1]
  13   : sin(φ)  — heading-to-goal angle
  14   : cos(φ)  — heading-to-goal angle

Action space (2-D continuous)
------------------------------
  [oL, oR] ∈ [-1, 1]²  →  left / right wheel commands

Episode termination
-------------------
  terminated = collision  OR  goal reached
  truncated  = step_count ≥ MAX_STEPS

Reference
---------
- brainstorm/01_env/gymnasium_api.md  (Decisions #6, #7)
- brainstorm/01_env/ray_casting.md    (Decisions #1–#3)
- brainstorm/02_neat/fitness_design.md (Decisions #8, #9)
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.env.kinematics import (
    ROBOT_RADIUS, WHEEL_BASE, V_MAX, DT, MAX_STEPS,
    GOAL_RADIUS, LIDAR_MAX_RANGE, LIDAR_NUM_RAYS,
    ARENA_DIAGONAL, step_kinematics, is_spinning,
)
from src.env.lidar import FastLiDAR2D
from src.env.collision import check_collision
from src.env.renderer import Renderer


class RobotNavEnv(gym.Env):
    """2-D differential-drive robot navigation environment."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        map_data: dict,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # ── Map data ──────────────────────────────────────────────────
        self.map_data = map_data
        self.segments: np.ndarray = np.array(
            map_data["segments"], dtype=np.float64
        )
        self.start_pos: np.ndarray = np.array(
            map_data["start"], dtype=np.float64
        )
        self.goal_pos: np.ndarray = np.array(
            map_data["goal"], dtype=np.float64
        )

        # ── Spaces ────────────────────────────────────────────────────
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(15,), dtype=np.float32,
        )

        # ── Internals ────────────────────────────────────────────────
        self._lidar = FastLiDAR2D(LIDAR_NUM_RAYS, LIDAR_MAX_RANGE)
        self._renderer = Renderer(render_mode)
        self.render_mode = render_mode

        # Episode state (initialised in reset)
        self.pos: np.ndarray = np.zeros(2)
        self.theta: float = 0.0
        self.step_count: int = 0
        self.spin_count: int = 0
        self.d0: float = 0.0
        self.d_min: float = 0.0
        self.trajectory: list[np.ndarray] = []
        self._last_proximity: np.ndarray | None = None

    # ================================================================== #
    #  Gymnasium API                                                       #
    # ================================================================== #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.pos = self.start_pos.copy()
        self.theta = 0.0
        self.step_count = 0
        self.spin_count = 0
        self.trajectory = [self.pos.copy()]

        self.d0 = float(np.linalg.norm(self.pos - self.goal_pos))
        self.d_min = self.d0

        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.step_count += 1
        oL, oR = np.clip(action, -1.0, 1.0)

        # ── Kinematics ────────────────────────────────────────────────
        vL = V_MAX * oL
        vR = V_MAX * oR

        if is_spinning(vL, vR):
            self.spin_count += 1

        x_new, y_new, theta_new = step_kinematics(
            self.pos[0], self.pos[1], self.theta, oL, oR,
        )
        self.pos = np.array([x_new, y_new])
        self.theta = theta_new
        self.trajectory.append(self.pos.copy())

        # ── Collision & goal check ────────────────────────────────────
        collided = check_collision(self.pos, self.segments, ROBOT_RADIUS)
        dist_to_goal = float(np.linalg.norm(self.pos - self.goal_pos))
        self.d_min = min(self.d_min, dist_to_goal)
        reached_goal = dist_to_goal <= GOAL_RADIUS

        terminated = collided or reached_goal
        truncated = (self.step_count >= MAX_STEPS) and (not terminated)

        # ── Observation ───────────────────────────────────────────────
        obs = self._get_obs()

        # ── Info dict (consumed by NEAT evaluator & Novelty module) ───
        info: dict[str, Any] = {
            "is_success": reached_goal,
            "is_collision": collided,
            "steps": self.step_count,
            "final_pos": (float(self.pos[0]), float(self.pos[1])),
            "trajectory": self.trajectory,
            "spin_steps": self.spin_count,
            "dist_to_goal": dist_to_goal,
            "min_distance_to_goal": self.d_min,
            "d0": self.d0,
        }

        # Reward is unused — NEAT evaluates fitness *after* the episode
        reward = 0.0
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None

        lidar_angles = self.theta + self._lidar.ray_angles

        return self._renderer.render(
            segments=self.segments,
            robot_pos=self.pos,
            robot_theta=self.theta,
            goal_pos=self.goal_pos,
            trajectory=self.trajectory,
            lidar_hits=self._last_proximity,
            lidar_angles=lidar_angles,
            lidar_max_range=LIDAR_MAX_RANGE,
        )

    def close(self):
        self._renderer.close()

    # ================================================================== #
    #  Observation builder                                                 #
    # ================================================================== #

    def _get_obs(self) -> np.ndarray:
        """Build the 15-D observation vector.

        Layout
        ------
        [0..11]  LiDAR proximity  (0 = safe, 1 = touching)
        [12]     normalised distance to goal  ρ
        [13]     sin(φ)  heading-to-goal angle
        [14]     cos(φ)  heading-to-goal angle
        """
        # LiDAR proximity (12-D)
        proximity = self._lidar.cast_rays(
            self.pos, self.theta, self.segments,
        )
        self._last_proximity = proximity       # cache for render

        # Relative goal bearing (3-D)
        diff = self.goal_pos - self.pos
        dist = float(np.linalg.norm(diff))
        rho = np.clip(dist / ARENA_DIAGONAL, 0.0, 1.0)

        phi = np.arctan2(diff[1], diff[0]) - self.theta
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        obs = np.concatenate([
            proximity,
            np.array([rho, sin_phi, cos_phi], dtype=np.float32),
        ])
        return obs.astype(np.float32)
