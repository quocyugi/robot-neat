"""Draw where the online-mapping planner collides on level-3 maps, and test how LiDAR resolution changes collisions.

Usage: python d02_viz_l3.py <repo_root> <out_png> <out_json>
"""
import json
import math
import os
import sys

REPO, PNG, OUT = sys.argv[1:4]
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.env.robot_env import RobotNavEnv
from src.env.collision import _point_to_segments_dist
import fastsim as fs
import planners as pl

keys = fs.keys_of(REPO, "test/level_3")
maps = [(k, fs.load_json(os.path.join(REPO, "data", "maps", k))) for k in keys]
fails = []
info_rows = []
for k, m in maps:
    env = RobotNavEnv(map_data=m)
    info = pl.run_mapping(env)
    tr = np.array(info["trajectory"])
    if info["is_collision"]:
        p = tr[-1]
        segs = np.array(m["segments"], float)
        dists = _point_to_segments_dist(p, segs)
        j = int(np.argmin(dists))
        a, b = segs[j]
        tip = min(np.linalg.norm(p - a), np.linalg.norm(p - b))
        near = a if np.linalg.norm(p - a) < np.linalg.norm(p - b) else b
        heading_vec = tr[-1] - tr[-4] if len(tr) > 4 else tr[-1] - tr[0]
        wall_dir = (b - a) / np.linalg.norm(b - a)
        ang = math.degrees(math.acos(min(1.0, abs(float(np.dot(heading_vec / (np.linalg.norm(heading_vec) + 1e-9), wall_dir))))))
        info_rows.append({"map": k, "steps": int(info["steps"]), "wall_index": j, "is_arena_wall": j < 4,
                          "distance_to_wall_tip_m": round(float(tip), 3), "angle_to_wall_deg": round(ang, 1),
                          "tip_is_the_end_0p5m_from_arena_wall": bool(min(near[0], 10.0 - near[0]) <= 0.51 or min(near[1], 10.0 - near[1]) <= 0.51),
                          "wall_length_m": round(float(np.linalg.norm(b - a)), 2)})
        fails.append((k, m, tr))

fig, axes = plt.subplots(3, 4, figsize=(11, 8.4))
for ax, (k, m, tr) in zip(axes.ravel(), fails[:12]):
    segs = np.array(m["segments"], float)
    for s in segs:
        ax.plot(s[:, 0], s[:, 1], "k-", lw=1.1)
    p = np.array(m["optimal_path"])
    ax.plot(p[:, 0], p[:, 1], "--", color="#2a78d6", lw=0.9)
    ax.plot(tr[:, 0], tr[:, 1], "-", color="#eb6834", lw=1.2)
    ax.plot(tr[-1, 0], tr[-1, 1], "x", color="#d03b3b", ms=9, mew=2)
    ax.plot(*m["start"], "o", color="#008300", ms=5)
    ax.plot(*m["goal"], "*", color="#eda100", ms=10, mec="k")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(k.split("/")[-1], fontsize=8)
for ax in axes.ravel()[len(fails[:12]):]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(PNG, dpi=110)
res = {"n_collisions": len(info_rows), "rows": info_rows}
tips = np.array([r["distance_to_wall_tip_m"] for r in info_rows]) if info_rows else np.array([])
angs = np.array([r["angle_to_wall_deg"] for r in info_rows]) if info_rows else np.array([])
res["share_within_0.5m_of_tip"] = round(float((tips < 0.5).mean()), 3) if len(tips) else None
res["median_angle_to_wall"] = round(float(np.median(angs)), 1) if len(angs) else None
res["collisions_at_slit_end"] = sum(r["tip_is_the_end_0p5m_from_arena_wall"] for r in info_rows)
res["collisions_at_inner_end"] = len(info_rows) - res["collisions_at_slit_end"]
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
for r in info_rows[:14]:
    print(r)
