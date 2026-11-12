"""T10 - Unit-style checks of the simulator (RobotNavEnv, LiDAR, collision, kinematics, fitness).

Usage: python t10_env_checks.py <repo_root> <out_json>
"""
import glob
import json
import math
import os
import random
import sys
import warnings

REPO, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)

import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from src.env.robot_env import RobotNavEnv
from src.env.lidar import FastLiDAR2D, arena_wall_segments
from src.env.collision import check_collision, min_obstacle_distance, _point_to_segments_dist
from src.env.kinematics import (ARENA_DIAGONAL, DT, GOAL_RADIUS, LIDAR_MAX_RANGE, MAX_STEPS,
                                ROBOT_RADIUS, V_MAX, WHEEL_BASE, is_spinning, step_kinematics)
from src.neat_engine.evaluator import compute_episode_fitness

results = {}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def maps_in(rel):
    return [load(p) for p in sorted(glob.glob(os.path.join(REPO, "data", "maps", rel, "*.json")))]


# ---------------------------------------------------------------- Gymnasium API
m0 = load(os.path.join(REPO, "data/maps/train/level_4/map_000000.json"))
env = RobotNavEnv(map_data=m0)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
        check_env(env, skip_render_check=True)
        results["gym_check_env"] = {"ok": True, "warnings": sorted({str(x.message)[:160] for x in w})}
    except Exception as e:
        results["gym_check_env"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

# ---------------------------------------------------------------- Kinematics
x, y, th = step_kinematics(5.0, 5.0, 0.0, 1.0, 1.0)
forward = dict(dx=x - 5.0, dy=y - 5.0, dtheta=th)
x, y, th = step_kinematics(5.0, 5.0, 0.0, -1.0, 1.0)
spin = dict(dx=x - 5.0, dy=y - 5.0, dtheta=th)
results["kinematics"] = {
    "forward_full_speed_step": forward,
    "spin_in_place_step": spin,
    "expected_forward_dx": V_MAX * DT,
    "expected_spin_dtheta": 2 * V_MAX / WHEEL_BASE * DT,
    "max_distance_in_budget_m": V_MAX * DT * MAX_STEPS,
}

# ---------------------------------------------------------------- LiDAR: analytic cases
lidar = FastLiDAR2D()
walls = arena_wall_segments(10, 10)
seg = np.array([[[6.0, 0.0], [6.0, 10.0]]], dtype=float)
p0 = lidar.cast_rays(np.array([5.0, 5.0]), 0.0, seg)
analytic = {
    "ray0_prox_wall_at_1m": float(p0[0]),
    "expected": 1 - 1 / 3,
    "ray6_prox_behind": float(p0[6]),
    "expected_behind": 0.0,
}
p90 = lidar.cast_rays(np.array([5.0, 5.0]), math.pi / 2, seg)
analytic["ray9_after_turning_left_90deg"] = float(p90[9])  # ray pointing to +x after +90 deg heading
results["lidar_analytic"] = analytic


# ---------------------------------------------------------------- LiDAR: independent ray-marching
def march(pos, angle, segs, R=3.0, step=0.002, tol=0.0012):
    d = 0.0
    while d <= R:
        q = np.array([pos[0] + math.cos(angle) * d, pos[1] + math.sin(angle) * d])
        if np.min(_point_to_segments_dist(q, segs)) <= tol:
            return d
        d += step
    return R


random.seed(1)
np.random.seed(1)
samples = []
for lvl in (2, 3, 4):
    samples += random.sample(maps_in(f"train/level_{lvl}"), 3)
errs = []
for mp in samples:
    segs = np.array(mp["segments"], dtype=float)
    for _ in range(25):
        pos = np.random.uniform(0.3, 9.7, 2)
        if check_collision(pos, segs, 0.05):
            continue
        theta = float(np.random.uniform(-math.pi, math.pi))
        prox = lidar.cast_rays(pos, theta, segs)
        for i in range(12):
            ang = theta + 2 * math.pi * i / 12
            d_true = march(pos, ang, segs)
            d_lidar = (1 - float(prox[i])) * 3.0
            errs.append(abs(d_true - d_lidar))
errs = np.array(errs)
results["lidar_vs_raymarch"] = {
    "n_rays": int(len(errs)), "max_err_m": float(errs.max()),
    "p99_err_m": float(np.percentile(errs, 99)), "mean_err_m": float(errs.mean()),
}

# ---------------------------------------------------------------- Collision: analytic
seg = np.array([[[0.0, 0.0], [10.0, 0.0]]], dtype=float)
results["collision_analytic"] = {
    "at_0.24m": bool(check_collision(np.array([5, 0.24]), seg)),
    "at_0.25m": bool(check_collision(np.array([5, 0.25]), seg)),
    "at_0.26m": bool(check_collision(np.array([5, 0.26]), seg)),
    "beyond_segment_end": bool(check_collision(np.array([10.26, 0.0]), seg)),
}

# ---------------------------------------------------------------- Observation layout
env = RobotNavEnv(map_data=m0)
obs, _ = env.reset()
results["observation_sample"] = {
    "shape": list(obs.shape), "dtype": str(obs.dtype),
    "prox_range": [float(obs[:12].min()), float(obs[:12].max())],
    "rho": float(obs[12]), "sin_cos_norm": float(obs[13] ** 2 + obs[14] ** 2),
    "proximity_at_collision_threshold": float(1 - ROBOT_RADIUS / LIDAR_MAX_RANGE),
}

# ---------------------------------------------------------------- Success and collision on the same step
mp = dict(m0)
mp["segments"] = arena_wall_segments(10, 10).tolist()
mp["start"] = [0.26, 4.76]
mp["goal"] = [0.45, 5.0]
env = RobotNavEnv(map_data=mp)
env.reset()
env.pos = np.array([0.26, 4.76])
env.theta = math.pi / 2 + 0.3   # heading slightly towards the wall
_, _, term, trunc, info = env.step(np.array([1.0, 1.0], dtype=np.float32))
results["success_and_collision_same_step"] = {
    "terminated": bool(term),
    "is_success": bool(info["is_success"]),
    "is_collision": bool(info["is_collision"]),
    "fitness": compute_episode_fitness(info),
}

# ---------------------------------------------------------------- Fitness bounds
base = dict(d0=8.0, min_distance_to_goal=8.0, is_success=False, is_collision=False, steps=500, spin_steps=0)
cases = {
    "stand_still_timeout": dict(base),
    "collide_at_once": dict(base, is_collision=True, steps=1),
    "spin_all_episode": dict(base, spin_steps=500),
    "half_way_timeout": dict(base, min_distance_to_goal=4.0),
    "half_way_then_collide": dict(base, min_distance_to_goal=4.0, is_collision=True, steps=250),
    "goal_at_step_100": dict(base, min_distance_to_goal=0.29, is_success=True, steps=100),
    "goal_at_step_300": dict(base, min_distance_to_goal=0.29, is_success=True, steps=300),
    "goal_at_step_499": dict(base, min_distance_to_goal=0.29, is_success=True, steps=499),
}
results["fitness_cases"] = {k: round(compute_episode_fitness(v), 4) for k, v in cases.items()}

# ---------------------------------------------------------------- Spin detector corner cases
results["spin_detector"] = {
    "opposite_full": is_spinning(-0.5, 0.5),
    "opposite_low": is_spinning(-0.09, 0.09),
    "wide_circle(0.5,0.1)": is_spinning(0.1, 0.5),
    "forward_back_jitter(0.5,0.5)": is_spinning(0.5, 0.5),
}

# ---------------------------------------------------------------- start_heading in data
counts = {}
for rel in ["train/level_1", "train/level_2", "train/level_3", "train/level_4",
            "val/level_1", "test/level_1", "test/level_4", "test_barn", "test_lehman"]:
    ms = maps_in(rel)
    counts[rel] = {"n": len(ms), "with_start_heading": sum("start_heading" in x for x in ms)}
results["start_heading_counts"] = counts
lm = maps_in("test_lehman")
envs = {}
for mp in lm:
    e = RobotNavEnv(map_data=mp)
    e.reset()
    envs[mp["map_id"]] = {"reset_theta": float(e.theta), "file_heading": mp.get("start_heading")}
results["heading_in_env"] = envs

results["constants"] = dict(MAX_STEPS=MAX_STEPS, GOAL_RADIUS=GOAL_RADIUS, DT=DT, V_MAX=V_MAX,
                            WHEEL_BASE=WHEEL_BASE, ROBOT_RADIUS=ROBOT_RADIUS,
                            ARENA_DIAGONAL=float(ARENA_DIAGONAL))

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(json.dumps(results, indent=2, ensure_ascii=False))
