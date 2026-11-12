"""Draw a grid of maps (walls, start, goal, A* path) to a PNG.

Usage: python d32_render_maps.py <out_png> <title> <map.json> [<map.json> ...]
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

out, title, paths = sys.argv[1], sys.argv[2], sys.argv[3:]
cols = min(4, len(paths))
rows = (len(paths) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.3 * rows), squeeze=False)
for ax in axes.ravel():
    ax.axis("off")
for ax, p in zip(axes.ravel(), paths):
    m = json.load(open(p, encoding="utf-8"))
    for (x0, y0), (x1, y1) in m["segments"]:
        ax.plot([x0, x1], [y0, y1], color="#333333", lw=0.8)
    path = m.get("optimal_path") or []
    if path:
        ax.plot([q[0] for q in path], [q[1] for q in path], color="#2a7fd4", lw=1.0)
    ax.plot(*m["start"], "o", color="#1a9850", ms=6)
    ax.plot(*m["goal"], "*", color="#d73027", ms=9)
    mt = m["metrics"]
    ax.set_title(f"{m['map_id']}  τ={mt['tortuosity']:.2f}  L*={mt['optimal_path_length']:.1f} m", fontsize=7)
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(-0.2, 10.2)
    ax.set_aspect("equal")
fig.suptitle(title, fontsize=9)
fig.tight_layout()
fig.savefig(out, dpi=110)
print("saved", out)
