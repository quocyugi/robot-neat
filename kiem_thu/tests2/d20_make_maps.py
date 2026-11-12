"""Build a copy of the team's map set in which level 3 is regenerated with a different wall recipe.

The team's own MapGenerator does the rest (A* validation, tortuosity band, metrics, seeds), and the repo is not modified.
Levels 1, 2, 4, BARN and Lehman are copied unchanged.

Usage: python d20_make_maps.py <repo> <out_root> <variant>
Variants:
  attached    walls start at the arena wall, so no impassable 0.5 m slit at one end; still zero thickness
  thick       attached, 0.20 m thick
  short       attached, 0.20 m thick, length 3-6 m instead of 4.5-8.5 m
"""
import json
import os
import shutil
import sys
import time

REPO, OUT, VARIANT = sys.argv[1:4]
sys.path.insert(0, REPO)

import numpy as np

from src.maps.generator import MapGenerator

THICK = 0.20


class Gen(MapGenerator):
    def _gen_corridors(self, rng):
        segs = []
        n_walls = rng.integers(3, 6)
        for i in range(n_walls):
            if VARIANT == "short":
                a, b = (3.0, 6.0) if i % 2 == 0 else (4.0, 7.0)
            else:
                a, b = (5.0, 8.5) if i % 2 == 0 else (1.5, 5.0)
            if i % 2 == 0:
                x1, x2 = 0.0, rng.uniform(a, b)
            else:
                x1, x2 = rng.uniform(a, b), self.arena_w
            y = rng.uniform(1.5, self.arena_h - 1.5)
            if VARIANT == "attached":
                segs.append([[x1, y], [x2, y]])
            else:
                segs.extend(self._rect_segments((x1 + x2) / 2, y, (x2 - x1) / 2, THICK / 2))
        return segs


SRC = os.path.join(REPO, "data", "maps")
gen = Gen()
stats = {}
t0 = time.time()
for split, n, offset in (("train", 250, 0), ("val", 25, 100_000), ("test", 50, 200_000)):
    out_dir = os.path.join(OUT, split, "level_3")
    os.makedirs(out_dir, exist_ok=True)
    made, seed, skipped = 0, offset, 0
    while made < n:
        m = gen.generate(3, seed)
        if m is None:
            skipped += 1
        else:
            with open(os.path.join(out_dir, f"map_{seed:06d}.json"), "w", encoding="utf-8") as f:
                json.dump(m, f, ensure_ascii=False)
            made += 1
        seed += 1
    stats[split] = dict(maps=made, seeds_tried=seed - offset, seeds_without_valid_map=skipped)
    print(split, stats[split], f"{time.time() - t0:.0f}s", flush=True)
    for lv in (1, 2, 4):
        dst = os.path.join(OUT, split, f"level_{lv}")
        if not os.path.exists(dst):
            shutil.copytree(os.path.join(SRC, split, f"level_{lv}"), dst)
for rel in ("test_barn", "test_lehman"):
    dst = os.path.join(OUT, rel)
    if not os.path.exists(dst):
        shutil.copytree(os.path.join(SRC, rel), dst)
with open(os.path.join(OUT, "generation_stats.json"), "w", encoding="utf-8") as f:
    json.dump(dict(variant=VARIANT, thickness=THICK, splits=stats), f, indent=1)
