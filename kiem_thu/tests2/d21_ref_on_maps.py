"""Difficulty and reference-planner success of a map set (used to calibrate new level definitions).

Usage: python d21_ref_on_maps.py <repo> <map_root> <out_json> [levels_csv]
"""
import glob
import json
import os
import sys

REPO, ROOT, OUT = sys.argv[1:4]
LEVELS = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [3]
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from src.baselines.apf import APFController
from src.env.robot_env import RobotNavEnv
import fastsim as fs
import planners as pl


class Fn:
    def __init__(self, c):
        self.c = c

    def reset(self, n):
        pass

    def act(self, idx, obs):
        return np.array([self.c.activate(obs[0].tolist())], dtype=np.float32)


def summarize(succ, col):
    succ, col = np.array(succ), np.array(col)
    return {"SR": round(float(succ.mean()), 3), "CR": round(float(col.mean()), 3), "timeout": round(float((~succ & ~col).mean()), 3), "n": len(succ)}


res = {}
for lv in LEVELS:
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "test", f"level_{lv}", "*.json"))):
        rows.append(fs.load_json(p))
    tau = [m["metrics"]["tortuosity"] for m in rows]
    trap = [m["metrics"]["trap_depth"] for m in rows]
    bott = [m["metrics"]["bottleneck_clearance"] for m in rows]
    out = {"n": len(rows), "tau_med": round(float(np.median(tau)), 3), "trap_depth_med": round(float(np.median(trap)), 2), "trap_depth_p90": round(float(np.percentile(trap, 90)), 2),
           "share_trap_gt_3m": round(float(np.mean(np.array(trap) > 3.0)), 3), "bottleneck_min": round(float(np.min(bott)), 3)}
    for name, make, steps in (("bug_500", lambda: pl.BugPolicy("open"), 500), ("bug_1000", lambda: pl.BugPolicy("open"), 1000), ("apf_500", lambda: Fn(APFController()), 500)):
        s, c = [], []
        for m in rows:
            r = fs.rollout(fs.FastMap(m, ""), make(), 1, max_steps=steps)
            s.append(bool(r.succ[0]))
            c.append(bool(r.col[0]))
        out[name] = summarize(s, c)
    s, c = [], []
    for m in rows:
        info = pl.run_mapping(RobotNavEnv(map_data=m))
        s.append(bool(info["is_success"]))
        c.append(bool(info["is_collision"]))
    out["mapping_500"] = summarize(s, c)
    res[f"L{lv}"] = out
    print(f"L{lv}", json.dumps(out), flush=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1)
