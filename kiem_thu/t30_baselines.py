"""T30 - Non-learning baselines (B4 rule-based, B5 APF) on the fixed test sets + a sign check of B4.

Usage: python t30_baselines.py <repo_root> <out_json>
"""
import glob
import json
import math
import os
import sys

REPO, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)

import numpy as np

from src.baselines.apf import APFController
from src.baselines.rule_based import RuleBasedController
from src.env.lidar import arena_wall_segments
from src.env.robot_env import RobotNavEnv
from src.experiments.metrics import compute_all_metrics


def maps_in(rel):
    return [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(REPO, "data", "maps", rel, "*.json")))]


TESTS = {f"test_L{k}": maps_in(f"test/level_{k}") for k in (1, 2, 3, 4)}
TESTS["test_barn"] = maps_in("test_barn")
TESTS["test_lehman"] = maps_in("test_lehman")


class GoToGoal:
    """Only steers to the goal; ignores LiDAR (lower reference)."""
    def activate(self, obs):
        s, c = obs[13], obs[14]
        turn = float(np.clip(s, -1, 1))
        speed = 0.7 if c > 0 else 0.1
        return [float(np.clip(speed - turn * 0.5, -1, 1)), float(np.clip(speed + turn * 0.5, -1, 1))]


class RuleFixed(RuleBasedController):
    """B4 with the swerve sign corrected (positive turn = counter-clockwise = left)."""
    def activate(self, obs):
        p = np.array(obs[:12])
        turn = float(np.clip(obs[13], -1, 1))
        speed = 0.6
        if max(p[0], p[1], p[11]) > self.danger_threshold:
            left = float(np.sum(p[1:6]))
            right = float(np.sum(p[6:11]))
            turn = 0.8 if left < right else -0.8
            speed = 0.3
        return [float(np.clip(speed - turn * 0.5, -1, 1)), float(np.clip(speed + turn * 0.5, -1, 1))]


def rollout_all(ctrl, maps):
    infos = []
    for mp in maps:
        env = RobotNavEnv(map_data=mp)
        obs, _ = env.reset()
        done = False
        while not done:
            act = np.array(ctrl.activate(obs.tolist()), dtype=np.float32)
            obs, _, term, trunc, info = env.step(act)
            done = term or trunc
        infos.append(info)
        env.close()
    return infos


res = {}
for name, ctrl in (("go_to_goal", GoToGoal()), ("B4_rule_as_is", RuleBasedController()),
                   ("B4_rule_sign_fixed", RuleFixed()), ("B5_apf_as_is", APFController())):
    row = {}
    for tname, ms in TESTS.items():
        infos = rollout_all(ctrl, ms)
        lens = [m["metrics"]["optimal_path_length"] for m in ms]
        m = compute_all_metrics(infos, lens)
        timeouts = sum(1 for i in infos if not i["is_success"] and not i["is_collision"])
        row[tname] = {"SR": round(m["SR"], 3), "CR": round(m["CR"], 3), "SPL": round(m["SPL"], 3),
                      "MTT": None if m["MTT"] == float("inf") else round(m["MTT"], 1),
                      "timeout_rate": round(timeouts / len(ms), 3), "n": len(ms)}
    res[name] = row

# ---- direct sign check of B4: wall ahead, RIGHT side crowded, LEFT free, goal straight ahead.
mp = {"segments": arena_wall_segments(10, 10).tolist() + [[[5.7, 3.0], [5.7, 7.0]], [[4.0, 4.6], [5.7, 4.6]]],
      "start": [5.0, 5.0], "goal": [9.0, 5.0], "map_id": "sign_check"}
env = RobotNavEnv(map_data=mp)
obs, _ = env.reset()
ctrl = RuleBasedController()
p = obs[:12]
left, right = float(np.sum(p[1:6])), float(np.sum(p[6:11]))
act = np.array(ctrl.activate(obs.tolist()), dtype=np.float32)
th0 = env.theta
env.step(act)
res["B4_sign_check"] = {
    "front_danger": float(max(p[0], p[1], p[11])),
    "left_sum": left, "right_sum": right,
    "left_is_freer": left < right,
    "action_oL_oR": [float(act[0]), float(act[1])],
    "dtheta_rad": float(env.theta - th0),
    "turned": "LEFT (toward free side)" if env.theta > th0 else "RIGHT (toward crowded side)",
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
