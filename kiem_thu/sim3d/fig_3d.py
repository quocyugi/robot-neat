"""Report figures for the 3D part: overview of DUnE's worlds, and trajectories of two controllers.

Usage: python -I fig_3d.py <repo> <dune_root> <cache_dir> <results_dir> <out_dir> <label=run_dir> [label=run_dir ...]
"""
import json
import math
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tests2"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np

repo, dune, cache, res_dir, out_dir = sys.argv[1:6]
runs = dict(x.split("=", 1) for x in sys.argv[6:])
sys.path.insert(0, repo)
import fastsim as fs
import sim3d as S
import world3d as W

BLUE = LinearSegmentedColormap.from_list("b", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"])
GRAYS = LinearSegmentedColormap.from_list("g", ["#fcfcfb", "#c3c2b7"])
INK, INK2, MUTED, BLOCK = "#0b0b0b", "#52514e", "#898781", "#898781"
C_OK, C_BAD, C_BOX = "#2a78d6", "#eb6834", "#e34948"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9})
NAMES = {"flat": "Flat", "pillars": "Maze + cột", "maze": "Maze", "construction": "Construction", "inspect": "Inspection (vùng của DUnE)",
         "inspect_wide": "Inspection (toàn khu vực)", "island": "Island (vùng của DUnE)", "island_wide": "Island (toàn khu vực)", "rubicon": "Rubicon"}


def grid_of(w, threads=3):
    task = W.TASKS[w]
    Vg, Fg, Vl, Fl = W.load_world(w, dune, cache)
    sim = S.Sim3D(Vg, Fg, Vl, Fl, nthreads=threads)
    sp, tf = task["spawn"], task["too_far"]
    margin = 8.0
    region = (max(tf[0], sp[0] - margin), min(tf[1], sp[1] + margin), max(tf[2], sp[2] - margin), min(tf[3], sp[3] + margin))
    return sim, task, S.Grid(sim, region, res=0.10, z0=task["z0"]), region


def style(ax, title):
    ax.set_title(title, fontsize=9.5, color=INK)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7, colors=MUTED, length=0)
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------- overview
order = ["flat", "pillars", "maze", "construction", "inspect", "inspect_wide", "island", "island_wide", "rubicon"]
fig, axes = plt.subplots(3, 3, figsize=(9.2, 9.4))
for ax, w in zip(axes.ravel(), order):
    sim, task, g, region = grid_of(w)
    ep = json.load(open(os.path.join(res_dir, f"episodes_{w}.json"), encoding="utf-8"))
    h = np.nan_to_num(g.h, nan=np.nanmin(g.h))
    lo, hi = np.percentile(h[g.free], [1, 99]) if g.free.any() else (0, 1)
    flat = hi - lo < 0.05
    if hi - lo < 0.5:
        lo, hi = lo - 0.25, hi + 0.25
    bg = np.ma.masked_where(g.free, np.ones_like(h))
    ax.imshow(h, origin="lower", extent=region, cmap=BLUE, vmin=lo, vmax=hi)
    ax.imshow(bg, origin="lower", extent=region, cmap=matplotlib.colors.ListedColormap([BLOCK]), vmin=0, vmax=1)
    sp = task["spawn"]
    ax.plot([sp[0], sp[1], sp[1], sp[0], sp[0]], [sp[2], sp[2], sp[3], sp[3], sp[2]], "-", color=C_BOX, lw=1.4)
    ax.scatter(ep["x"], ep["y"], s=9, color=C_BAD, linewidths=0.4, edgecolors="white", zorder=3)
    ax.scatter(ep["gx"], ep["gy"], s=16, color=INK, marker="*", linewidths=0, zorder=3)
    style(ax, f"{NAMES[w]}: mặt phẳng" if flat else f"{NAMES[w]}: cao độ {lo:.1f}–{hi:.1f} m".replace(".", ","))
handles = [Line2D([], [], color=C_BOX, lw=1.4, label="Vùng sinh điểm"), Line2D([], [], marker="o", ls="", color=C_BAD, ms=5, label="Điểm xuất phát"),
           Line2D([], [], marker="*", ls="", color=INK, ms=7, label="Điểm đích"),
           Line2D([], [], marker="s", ls="", color=BLOCK, ms=7, label="Không đi được")]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=INK2)
fig.tight_layout(rect=(0, 0.035, 1, 1))
fig.savefig(os.path.join(out_dir, "fig_worlds3d.png"), dpi=170, facecolor="white")
plt.close(fig)


# ---------------------------------------------------- trajectories
def controller(path):
    s = json.load(open(os.path.join(path, "summary.json"), encoding="utf-8"))
    payload = pickle.load(open(os.path.join(path, "champion.pkl"), "rb"))
    if s["algo"] == "neat":
        import neat
        cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, os.path.join(path, "patched.ini"))
        cls = neat.nn.RecurrentNetwork if s.get("net") == "rnn" else neat.nn.FeedForwardNetwork

        class P:
            def reset(self, n):
                self.p = fs.NetsPolicy([cls.create(payload, cfg) for _ in range(n)])
                self.p.reset(n)

            def act(self, idx, obs):
                return self.p.act(idx, obs)
        return P()
    layers = [s["n_rays"] + 3] + [int(x) for x in str(s["layers"]).split(",")] + [2]
    arch = s.get("arch", "mlp")

    class Q:
        def reset(self, n):
            p = np.repeat(payload[None, :], n, axis=0)
            self.p = fs.ElmanBatch(p, layers) if arch == "elman" else fs.MLPBatch(p, layers)
            self.p.reset(n)

        def act(self, idx, obs):
            return self.p.act(idx, obs)
    return Q()


worlds = ["maze", "inspect_wide", "island_wide", "rubicon"]
fig, axes = plt.subplots(len(runs), len(worlds), figsize=(9.4, 2.75 * len(runs)), squeeze=False)
for r, (label, path) in enumerate(runs.items()):
    for c, w in enumerate(worlds):
        ax = axes[r, c]
        sim, task, g, region = grid_of(w)
        ep = json.load(open(os.path.join(res_dir, f"episodes_{w}.json"), encoding="utf-8"))
        n = 30
        sub = {k: v[:n] for k, v in ep.items()}
        tr = []
        R = S.rollout3d(sim, task, sub, controller(path), ground_filter="none", track=tr)
        h = np.nan_to_num(g.h, nan=np.nanmin(g.h))
        ax.imshow(h, origin="lower", extent=region, cmap=GRAYS)
        ax.imshow(np.ma.masked_where(g.free, np.ones_like(h)), origin="lower", extent=region, cmap=matplotlib.colors.ListedColormap([BLOCK]), vmin=0, vmax=1)
        for i in range(n):
            xs, ys = [sub["x"][i]], [sub["y"][i]]
            for idx, x, y in tr:
                j = np.where(idx == i)[0]
                if j.size:
                    xs.append(float(x[j[0]]))
                    ys.append(float(y[j[0]]))
            ax.plot(xs, ys, "-", color=C_OK if R.reason[i] == 1 else C_BAD, lw=0.9, alpha=0.85)
        srate = int(round(100 * float((R.reason == 1).mean())))
        style(ax, f"{label}, {NAMES[w].split(' (')[0]}: {srate}%")
        ax.set_xlim(region[0], region[1])
        ax.set_ylim(region[2], region[3])
handles = [Line2D([], [], color=C_OK, lw=1.6, label="Tới đích"), Line2D([], [], color=C_BAD, lw=1.6, label="Thất bại"),
           Line2D([], [], marker="s", ls="", color=BLOCK, ms=7, label="Không đi được")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(out_dir, "fig_traj3d.png"), dpi=170, facecolor="white")
print("saved")
