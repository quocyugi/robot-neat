"""T11 - Re-verify the shipped map dataset independently of the stored manifest.

Checks: counts, schema, duplicates across splits, S-G distance, level tau bands,
A*-following oracle in RobotNavEnv with the current 500-step limit.

Usage: python t11_data_checks.py <repo_root> <out_json>
"""
import collections
import csv
import glob
import hashlib
import json
import os
import sys
import time

REPO, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)

import numpy as np

from src.env.robot_env import RobotNavEnv
from src.env.kinematics import DT, MAX_STEPS, V_MAX, WHEEL_BASE

ROOT = os.path.join(REPO, "data", "maps")
GROUPS = {}
for split in ("train", "val", "test"):
    for lvl in (1, 2, 3, 4):
        GROUPS[f"{split}/level_{lvl}"] = os.path.join(ROOT, split, f"level_{lvl}")
GROUPS["test_barn"] = os.path.join(ROOT, "test_barn")
GROUPS["barn_all"] = os.path.join(ROOT, "barn_all")
GROUPS["test_lehman"] = os.path.join(ROOT, "test_lehman")


def load_group(path):
    return [(p, json.load(open(p, encoding="utf-8"))) for p in sorted(glob.glob(os.path.join(path, "*.json")))]


def chash(m):
    p = {"s": np.round(np.array(m["start"], float), 3).tolist(),
         "g": np.round(np.array(m["goal"], float), 3).tolist(),
         "seg": np.round(np.array(m["segments"], float), 3).tolist()}
    return hashlib.sha1(json.dumps(p, sort_keys=True).encode()).hexdigest()


def oracle(m, max_steps=None):
    env = RobotNavEnv(map_data=m)
    env.reset()
    w = m["optimal_path"][1:]
    k = 0
    turn = 2 * V_MAX / WHEEL_BASE * DT
    info = {"is_success": False, "is_collision": False, "steps": 0}
    term = trunc = False
    while not (term or trunc):
        diff = np.array(w[k]) - env.pos
        dist = float(np.linalg.norm(diff))
        if dist < 0.01 and k < len(w) - 1:
            k += 1
            continue
        err = (np.arctan2(diff[1], diff[0]) - env.theta + np.pi) % (2 * np.pi) - np.pi
        if dist > 0.01 and abs(err) > 0.01:
            u = float(np.clip(err / turn, -1, 1))
            a = [-u, u]
        else:
            u = float(np.clip(dist / (V_MAX * DT), 0, 1))
            a = [u, u]
        _, _, term, trunc, info = env.step(np.array(a, dtype=np.float32))
    env.close()
    return info


out = {"groups": {}}
hash_owner = collections.defaultdict(list)
t0 = time.time()
all_steps = []
for name, path in GROUPS.items():
    items = load_group(path)
    taus, sg, succ, col, steps, heading, lens = [], [], 0, 0, [], 0, []
    for fp, m in items:
        s, g = np.array(m["start"]), np.array(m["goal"])
        sg.append(float(np.linalg.norm(s - g)))
        taus.append(m["metrics"]["tortuosity"])
        lens.append(m["metrics"]["optimal_path_length"])
        heading += int("start_heading" in m)
        if name != "barn_all":
            hash_owner[chash(m)].append(f"{name}/{os.path.basename(fp)}")
        info = oracle(m)
        succ += int(info["is_success"])
        col += int(info["is_collision"])
        steps.append(int(info["steps"]))
    if name != "barn_all":
        all_steps += steps
    out["groups"][name] = {
        "n": len(items),
        "sg_min": round(min(sg), 3), "tau_min": round(min(taus), 3), "tau_med": round(float(np.median(taus)), 3),
        "tau_max": round(max(taus), 3), "L_opt_med": round(float(np.median(lens)), 2), "L_opt_max": round(max(lens), 2),
        "oracle_success": succ, "oracle_collision": col, "oracle_steps_med": int(np.median(steps)),
        "oracle_steps_max": int(max(steps)), "with_start_heading": heading,
    }
dups = {k: v for k, v in hash_owner.items() if len(v) > 1}
out["n_unique_content"] = len(hash_owner)
out["n_duplicates"] = len(dups)
out["duplicates_sample"] = list(dups.values())[:5]
out["total_files_excluding_barn_all"] = int(sum(len(v) for v in hash_owner.values()))
out["seconds"] = round(time.time() - t0, 1)
out["max_steps_env"] = MAX_STEPS

# Fraction of maps where a perfect time still reaches F>=16 (threshold analysis)
frac = {}
rows = list(csv.DictReader(open(os.path.join(ROOT, "manifest.csv"), encoding="utf-8-sig")))
for lvl in "1234":
    st = [int(r["oracle_steps"]) for r in rows if r["level"] == lvl and r["split"] in ("train",)]
    # F = 3 + progress(~0.96) + 10 + 5*(1-t/500) >= 16
    frac[lvl] = {"n": len(st), "frac_F_ge_16_with_oracle_time": round(float(np.mean([3 + 0.96 + 10 + 5 * (1 - t / 500) >= 16 for t in st])), 3),
                 "oracle_steps_max": max(st)}
out["threshold16_vs_oracle_time"] = frac

# Level-4 style statistics
l4 = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(ROOT, "train", "level_4", "*.json")))]
out["level4"] = {
    "n": len(l4),
    "trap_depth_med": round(float(np.median([m["metrics"]["trap_depth"] for m in l4])), 3),
    "n_segments_med": int(np.median([len(m["segments"]) for m in l4])),
    "greedy_stuck_rate": round(float(np.mean([m["metrics"]["greedy_stuck"] for m in l4])), 3),
    "bottleneck_min": round(min(m["metrics"]["bottleneck_clearance"] for m in l4), 3),
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(json.dumps(out, indent=2, ensure_ascii=False))
