"""
U-Trap Obstacle Generator
===========================
Generates oriented U-shaped trap obstacles that block the direct
path from Start to Goal, creating deceptive fitness landscapes.

Reference
---------
- brainstorm/01_env/map_generator.md §3.2 (Decision #22)
- brainstorm/03_novelty/utrap_analysis.md
"""

from __future__ import annotations

import numpy as np


def generate_u_trap(
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    trap_width: float = 2.0,
    trap_depth: float = 2.0,
    position_ratio: float = 0.5,
    wall_thickness: float = 0.0,
) -> list[list[list[float]]]:
    """Create a U-shaped trap oriented perpendicular to the S→G vector.

    The trap's opening faces *toward Start* so that the robot is drawn
    into the dead-end by the deceptive distance gradient.

    Parameters
    ----------
    start_pos, goal_pos : ndarray, shape (2,)
    trap_width : float
        Width of the U-pocket (distance between two wings).
    trap_depth : float
        How far the wings extend back toward Start.
    position_ratio : float
        Where along the S→G line to place the trap centre (0–1).
    wall_thickness : float
        If > 0, each wall becomes a thin rectangle (two parallel segments).

    Returns
    -------
    segments : list of [[Ax, Ay], [Bx, By]]
    """
    vec = goal_pos - start_pos
    dist = np.linalg.norm(vec)
    if dist < 1e-6:
        return []

    u_dir = vec / dist                                    # unit S→G
    ortho = np.array([-u_dir[1], u_dir[0]])               # perpendicular

    trap_center = start_pos + position_ratio * vec

    # Bottom (closed end of the U)
    bl = trap_center - 0.5 * trap_width * ortho
    br = trap_center + 0.5 * trap_width * ortho

    # Wings extend back toward Start
    wl = bl - trap_depth * u_dir
    wr = br - trap_depth * u_dir

    segments = [
        [bl.tolist(), br.tolist()],   # bottom wall
        [bl.tolist(), wl.tolist()],   # left wing
        [br.tolist(), wr.tolist()],   # right wing
    ]

    return segments


def generate_l_trap(
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    arm_length: float = 2.5,
    position_ratio: float = 0.5,
) -> list[list[list[float]]]:
    """Create an L-shaped deflector obstacle.

    Less deceptive than a U-trap but still forces a detour.
    """
    vec = goal_pos - start_pos
    dist = np.linalg.norm(vec)
    if dist < 1e-6:
        return []

    u_dir = vec / dist
    ortho = np.array([-u_dir[1], u_dir[0]])

    corner = start_pos + position_ratio * vec
    end_a = corner + arm_length * ortho
    end_b = corner - arm_length * 0.5 * u_dir

    return [
        [corner.tolist(), end_a.tolist()],
        [corner.tolist(), end_b.tolist()],
    ]
