"""Validate DUnE's worlds for navigation tests and create fixed test episodes.

For each world: how much of DUnE's spawn/goal box is usable, whether the goal can be reached at all,
how winding the shortest path is, and how often a planar LiDAR hits walkable ground instead of an obstacle.

Usage: python -I d10_audit3d.py <dune_root> <cache_dir> <out_dir> [n_episodes] [world ...]
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sim3d as S
import world3d as W

dune, cache, out = sys.argv[1:4]
n_ep = int(sys.argv[4]) if len(sys.argv) > 4 else 100
names = sys.argv[5:] or ["flat", "maze", "pillars", "construction", "inspect", "inspect_wide", "island", "island_wide", "rubicon"]
os.makedirs(out, exist_ok=True)
result = {}
fig, axes = plt.subplots(3, 3, figsize=(11.5, 10.5))
axes = axes.ravel()
for ax_i, name in enumerate(names):
    t0 = time.time()
    task = W.TASKS[name]
    Vg, Fg, Vl, Fl = W.load_world(name, dune, cache)
    sim = S.Sim3D(Vg, Fg, Vl, Fl, nthreads=2)
    sp, tf = task["spawn"], task["too_far"]
    margin = 8.0
    region = (max(tf[0], sp[0] - margin), min(tf[1], sp[1] + margin), max(tf[2], sp[2] - margin), min(tf[3], sp[3] + margin))
    g = S.Grid(sim, region, res=0.10, z0=task["z0"])
    rng = np.random.default_rng(20261109)
    ep, c = g.sample_episodes(task["spawn"], task["goal"], n_ep, rng, max_geo=task.get("max_geo"))
    e, d = np.array(ep["euclid"]), np.array(ep["geo"])
    tau = d / e
    # how often would a planar LiDAR hit walkable ground? (poses taken from the start of each episode)
    x, y, yaw, z = np.array(ep["x"]), np.array(ep["y"]), np.array(ep["yaw"]), np.array(ep["z"])
    h, miss = sim.contact(x, y, yaw, z)
    X, Y, Z, z1 = sim.frame(yaw, h)
    t, nz = sim.lidar(x, y, z1, X, Y, Z)
    rngs = sim.ranges(t)
    gt = math.cos(math.radians(35.0))
    gnd3 = float(((nz >= gt) & (rngs < 3.0)).mean())
    gnd10 = float(((nz >= gt) & (rngs < 10.0)).mean())
    any3 = float((rngs < 3.0).mean())
    # slopes inside the box where DUnE spawns the robot
    r0, c0 = g.cell(sp[0], sp[2])
    r1, c1 = g.cell(sp[1] - 1e-6, sp[3] - 1e-6)
    box_free = g.free[r0:r1 + 1, c0:c1 + 1]
    dzdy, dzdx = np.gradient(np.nan_to_num(g.h, nan=0.0), 0.10)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))[r0:r1 + 1, c0:c1 + 1]
    # visual mesh (what the LiDAR sees) minus collision mesh (what the wheels touch) at 2000 random points of the box
    px, py = rng.uniform(sp[0], sp[1], 2000), rng.uniform(sp[2], sp[3], 2000)
    top = sim.zmax + 5.0
    rays = np.stack([px, py, np.full(2000, top), np.zeros(2000), np.zeros(2000), -np.ones(2000)], axis=1)
    tg, _ = sim._cast(sim.gs, rays)
    tl, _ = sim._cast(sim.ls, rays)
    dvis = (top - tl) - (top - tg)
    dvis = dvis[np.isfinite(dvis)]
    res = {
        "visual_minus_collision_height_m_p50_p95_max": [round(float(v), 3) for v in (np.percentile(np.abs(dvis), 50), np.percentile(np.abs(dvis), 95), np.abs(dvis).max())] if dvis.size else None,
        "faces_ground": int(len(Fg)), "faces_lidar": int(len(Fl)), "region": region,
        "spawn_box": list(sp), "too_far": list(tf),
        "free_share_of_spawn_box": round(float(box_free.mean()), 4),
        "slope_deg_p50_p90_p99": [round(float(v), 1) for v in np.percentile(slope, [50, 90, 99])],
        "height_range_in_box": [round(float(np.nanmin(g.h[r0:r1 + 1, c0:c1 + 1])), 2), round(float(np.nanmax(g.h[r0:r1 + 1, c0:c1 + 1])), 2)],
        "sampling": c,
        "accepted_share": round(n_ep / c["attempts"], 4),
        "euclid_mean": round(float(e.mean()), 2), "geo_mean": round(float(d.mean()), 2),
        "tau_median": round(float(np.median(tau)), 3), "tau_p90": round(float(np.percentile(tau, 90)), 3),
        "lidar_beams_hitting_walkable_ground_within_3m": round(gnd3, 4),
        "lidar_beams_hitting_anything_within_3m": round(any3, 4),
        "lidar_beams_hitting_walkable_ground_within_10m": round(gnd10, 4),
        "seconds": round(time.time() - t0, 1),
    }
    result[name] = res
    with open(os.path.join(out, f"episodes_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(ep, f)
    print(name, json.dumps(res), flush=True)
    ax = axes[ax_i]
    img = np.where(g.free, np.nan_to_num(g.h, nan=0.0), np.nan)
    ax.imshow(np.nan_to_num(g.h, nan=0.0), origin="lower", extent=region, cmap="terrain", alpha=0.45)
    ax.imshow(img, origin="lower", extent=region, cmap="terrain")
    ax.plot([sp[0], sp[1], sp[1], sp[0], sp[0]], [sp[2], sp[2], sp[3], sp[3], sp[2]], "-", color="#d03b3b", lw=1.2)
    ax.scatter(ep["x"], ep["y"], s=4, color="#008300")
    ax.scatter(ep["gx"], ep["gy"], s=6, color="#eda100", marker="*")
    ax.set_title(name, fontsize=10)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
for ax in axes[len(names):]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(os.path.join(out, "worlds_overview.png"), dpi=120)
with open(os.path.join(out, "audit3d.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, indent=1)
