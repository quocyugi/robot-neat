"""Which end of the level-3 walls do the planner's collisions happen at, and how common are impassable 0.5 m slits?

Usage: python d04_l3_tips.py <repo_root> <d02_json> <out_json>
"""
import glob
import json
import os
import sys

import numpy as np

repo, d02, out = sys.argv[1:4]
rows = json.load(open(d02, encoding="utf-8"))["rows"]
root = os.path.join(repo, "data", "maps")
near_boundary, inner = 0, 0
for r in rows:
    m = json.load(open(os.path.join(root, r["map"]), encoding="utf-8"))
    s = np.array(m["segments"][r["wall_index"]], float)
    # the collision point is not stored, so use the end named by the stored distance: both ends are tested against x = 0.5 / 9.5
    ends = [p for p in s if min(p[0], 10 - p[0]) <= 0.51]
    near_boundary += len(ends) > 0
    inner += len(ends) == 0
stats = {"collisions": len(rows), "on_walls_with_a_slit_end": near_boundary, "on_walls_without": inner}

slits = {}
for rel in ("test/level_3", "train/level_3", "val/level_3"):
    n_maps, n_walls, n_slit = 0, 0, 0
    lens = []
    for p in glob.glob(os.path.join(root, rel, "*.json")):
        m = json.load(open(p, encoding="utf-8"))
        segs = np.array(m["segments"], float)[4:]
        n_maps += 1
        for s in segs:
            n_walls += 1
            lens.append(float(np.linalg.norm(s[1] - s[0])))
            n_slit += any(min(q[0], 10 - q[0]) <= 0.51 for q in s)
    slits[rel] = dict(maps=n_maps, walls=n_walls, walls_with_end_0p5m_from_arena=n_slit, mean_walls_per_map=round(n_walls / n_maps, 2),
                      wall_length_median=round(float(np.median(lens)), 2), wall_length_p10_p90=[round(float(v), 2) for v in np.percentile(lens, [10, 90])])
stats["level3"] = slits
print(json.dumps(stats, indent=1))
with open(out, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=1)
