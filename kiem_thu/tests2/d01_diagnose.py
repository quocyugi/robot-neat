"""Why are levels 3-4 so low? Map audit, hand-written local planners, failure modes of the trained controllers.

Usage: python d01_diagnose.py <repo_root> <old_results_root> <out_json>
"""
import json
import math
import os
import pickle
import sys
import time

REPO, OLD, OUT = sys.argv[1:4]
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import neat
import numpy as np

from src.env.robot_env import RobotNavEnv
import fastsim as fs
import planners as pl


def load(rel):
    keys = fs.keys_of(REPO, rel)
    return [(k, fs.load_json(os.path.join(REPO, "data", "maps", k))) for k in keys]


TESTS = {f"L{k}": load(f"test/level_{k}") for k in (1, 2, 3, 4)}
TESTS["BARN"] = load("test_barn")
TESTS["Lehman"] = load("test_lehman")
res = {}

# ---------------------------------------------------------------------- A. map audit
audit = {}
for name in ("L1", "L2", "L3", "L4", "BARN", "Lehman"):
    rows = TESTS[name]
    tau, trap, bott, lopt, back, wall_len, nseg = [], [], [], [], [], [], []
    for k, m in rows:
        g = np.array(m["goal"])
        met = m["metrics"]
        tau.append(met["tortuosity"])
        trap.append(met["trap_depth"])
        bott.append(met["bottleneck_clearance"])
        lopt.append(met["optimal_path_length"])
        nseg.append(len(m["segments"]))
        # total length of the optimal path along which the straight-line distance to the goal INCREASES
        p = np.array(m["optimal_path"])
        pts = []
        for a, b in zip(p[:-1], p[1:]):
            n = max(int(np.linalg.norm(b - a) / 0.05), 1)
            pts += [a + (b - a) * i / n for i in range(n)]
        pts = np.array(pts + [p[-1]])
        dist = np.linalg.norm(pts - g, axis=1)
        back.append(float(np.clip(np.diff(dist), 0, None).sum()))
        seg = np.array(m["segments"])[4:]
        if len(seg):
            wall_len.append(float(np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1).max()))
    audit[name] = {
        "n": len(rows), "tau_med": round(float(np.median(tau)), 3),
        "trap_depth_med": round(float(np.median(trap)), 2), "trap_depth_p90": round(float(np.percentile(trap, 90)), 2),
        "share_trap_gt_3m": round(float(np.mean(np.array(trap) > 3.0)), 3),
        "backtrack_med_m": round(float(np.median(back)), 2), "backtrack_p90_m": round(float(np.percentile(back, 90)), 2),
        "share_backtrack_gt_3m": round(float(np.mean(np.array(back) > 3.0)), 3),
        "bottleneck_med": round(float(np.median(bott)), 3), "bottleneck_min": round(float(np.min(bott)), 3),
        "share_gap_lt_0.8m": round(float(np.mean(np.array(bott) * 2 < 0.8)), 3),
        "L_opt_med": round(float(np.median(lopt)), 2), "longest_wall_med": round(float(np.median(wall_len)), 2) if wall_len else None,
    }
res["map_audit"] = audit


# ---------------------------------------------------------------------- B. planners
def summarize(succ, col, steps, n):
    succ, col = np.array(succ), np.array(col)
    return {"SR": round(float(succ.mean()), 3), "CR": round(float(col.mean()), 3), "timeout": round(float((~succ & ~col).mean()), 3), "n": n,
            "mean_steps_success": round(float(np.array(steps)[succ].mean()), 1) if succ.any() else None}


def eval_bug(rule, max_steps):
    out = {}
    for name, rows in TESTS.items():
        s, c, st = [], [], []
        for k, m in rows:
            fm = fs.FastMap(m, k)
            r = fs.rollout(fm, pl.BugPolicy(side_rule=rule), 1, max_steps=max_steps)
            s.append(bool(r.succ[0]))
            c.append(bool(r.col[0]))
            st.append(int(r.steps[0]))
        out[name] = summarize(s, c, st, len(rows))
    return out


t0 = time.time()
res["bug"] = {}
for rule in ("open", "goal", "right", "left"):
    res["bug"][rule] = {"steps500": eval_bug(rule, 500)}
res["bug"]["open"]["steps1000"] = eval_bug("open", 1000)
res["bug_seconds"] = round(time.time() - t0, 1)

t0 = time.time()
mp = {}
for name in ("L1", "L2", "L3", "L4", "BARN", "Lehman"):
    s, c, st = [], [], []
    for k, m in TESTS[name]:
        env = RobotNavEnv(map_data=m)
        info = pl.run_mapping(env)
        s.append(bool(info["is_success"]))
        c.append(bool(info["is_collision"]))
        st.append(int(info["steps"]))
    mp[name] = summarize(s, c, st, len(TESTS[name]))
res["mapping_planner"] = mp
res["mapping_seconds"] = round(time.time() - t0, 1)

# ---------------------------------------------------------------------- C. failure modes of the earlier trained controllers
FIX = os.path.join(OLD, "fixed")
ga = pickle.load(open(os.path.join(FIX, "ga_b1_180k_s0", "champion.pkl"), "rb"))
cfgp = os.path.join(FIX, "tuned_fit_k8_s0", "patched.ini")
cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, cfgp)
g = pickle.load(open(os.path.join(FIX, "tuned_fit_k8_s0", "champion.pkl"), "rb"))
modes = {}
for name, make in (("GA_180k", lambda: fs.MLPBatch(np.asarray(ga)[None, :], [15, 8, 2])), ("NEAT_120k", lambda: fs.NetsPolicy([neat.nn.RecurrentNetwork.create(g, cfg)]))):
    modes[name] = {}
    for lv in ("L2", "L3", "L4"):
        succ, col, progress, early = [], [], [], []
        for k, m in TESTS[lv]:
            geo = fs.compute_geodesic(m["segments"], m["goal"])
            fm = fs.FastMap(m, k, geo=geo)
            r = fs.rollout(fm, make(), 1)
            succ.append(bool(r.succ[0]))
            col.append(bool(r.col[0]))
            progress.append(float(np.clip((fm.g0 - r.gmin[0]) / fm.g0, 0, 1)))
            early.append(int(r.steps[0]))
        succ, col = np.array(succ), np.array(col)
        modes[name][lv] = {"SR": round(float(succ.mean()), 3), "CR": round(float(col.mean()), 3),
                           "timeout": round(float((~succ & ~col).mean()), 3),
                           "median_geodesic_progress": round(float(np.median(progress)), 3),
                           "median_steps": int(np.median(early))}
res["failure_modes"] = modes

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
