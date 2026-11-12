"""T20 - Throughput of the simulator and of NEAT networks (single process).

Usage: python t20_speed.py <repo_root> <out_json>
"""
import glob
import json
import os
import random
import sys
import time

REPO, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)

import neat
import numpy as np

from src.env.robot_env import RobotNavEnv
from src.env.lidar import FastLiDAR2D
from src.env.collision import check_collision
from src.env.kinematics import MAX_STEPS


def maps_in(rel, n=None):
    fs = sorted(glob.glob(os.path.join(REPO, "data", "maps", rel, "*.json")))
    if n:
        fs = fs[:n]
    return [json.load(open(p, encoding="utf-8")) for p in fs]


def env_steps_per_sec(maps, policy):
    n = 0
    t0 = time.perf_counter()
    for mp in maps:
        env = RobotNavEnv(map_data=mp)
        obs, _ = env.reset()
        done = False
        while not done:
            a = policy(obs)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
            n += 1
        env.close()
    dt = time.perf_counter() - t0
    return n / dt, n, dt


results = {}
idle = lambda obs: np.zeros(2, dtype=np.float32)
groups = {"level_1": "train/level_1", "level_2": "train/level_2", "level_3": "train/level_3",
          "level_4": "train/level_4", "barn": "test_barn"}
for name, rel in groups.items():
    maps = maps_in(rel, 20)
    sps, n, dt = env_steps_per_sec(maps, idle)
    nseg = float(np.mean([len(m["segments"]) for m in maps]))
    results[f"env_only_{name}"] = {"steps_per_s": round(sps), "n_steps": n, "mean_segments": nseg}

# Micro-costs for one step on a level-4 map
mp = maps_in("train/level_4", 1)[0]
segs = np.array(mp["segments"], dtype=float)
pos = np.array(mp["start"])
lidar = FastLiDAR2D()
N = 20000
t0 = time.perf_counter()
for _ in range(N):
    lidar.cast_rays(pos, 0.3, segs)
results["micro_cast_rays_us"] = round((time.perf_counter() - t0) / N * 1e6, 2)
t0 = time.perf_counter()
for _ in range(N):
    check_collision(pos, segs, 0.25)
results["micro_check_collision_us"] = round((time.perf_counter() - t0) / N * 1e6, 2)
env = RobotNavEnv(map_data=mp)
env.reset()
t0 = time.perf_counter()
for _ in range(2000):
    env.step_count = 0
    env.step(np.zeros(2, dtype=np.float32))
results["micro_env_step_us"] = round((time.perf_counter() - t0) / 2000 * 1e6, 2)
t0 = time.perf_counter()
for _ in range(300):
    e = RobotNavEnv(map_data=mp)
    e.reset()
    e.close()
results["micro_env_create_reset_us"] = round((time.perf_counter() - t0) / 300 * 1e6, 2)

# Network costs
cfg_ff = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                     neat.DefaultStagnation, os.path.join(REPO, "configs/neat_feedforward.ini"))
cfg_rc = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                     neat.DefaultStagnation, os.path.join(REPO, "configs/neat_recurrent.ini"))
random.seed(0)
pop_ff = neat.Population(cfg_ff)
g_ff = next(iter(pop_ff.population.values()))
pop_rc = neat.Population(cfg_rc)
g_rc = next(iter(pop_rc.population.values()))
obs = [0.1] * 15
for label, net in (("ff_minimal", neat.nn.FeedForwardNetwork.create(g_ff, cfg_ff)),
                   ("rnn_minimal", neat.nn.RecurrentNetwork.create(g_rc, cfg_rc))):
    t0 = time.perf_counter()
    for _ in range(20000):
        net.activate(obs)
    results[f"micro_{label}_activate_us"] = round((time.perf_counter() - t0) / 20000 * 1e6, 2)
    results[f"{label}_connections"] = len([c for c in g_ff.connections.values() if c.enabled])

# End-to-end with a real minimal genome
maps = maps_in("train/level_4", 8)
net = neat.nn.RecurrentNetwork.create(g_rc, cfg_rc)
def nn_policy(o):
    return np.array(net.activate(o.tolist()), dtype=np.float32)
n = 0
t0 = time.perf_counter()
for mp in maps:
    env = RobotNavEnv(map_data=mp)
    obs_, _ = env.reset()
    net.reset()
    done = False
    while not done:
        obs_, _, term, trunc, _ = env.step(nn_policy(obs_))
        done = term or trunc
        n += 1
    env.close()
results["nn_rollout_steps"] = n
results["nn_rollout_steps_per_s"] = round(n / (time.perf_counter() - t0))

# Worst case budget
worst_steps_per_eval = MAX_STEPS
sps = results["env_only_level_4"]["steps_per_s"]
results["budget_worst_case"] = {
    "assumed_steps_per_s_env_only_level4": sps,
    "60k_evals_hours_single_core": round(60000 * worst_steps_per_eval / sps / 3600, 2),
    "180k_evals_hours_single_core": round(180000 * worst_steps_per_eval / sps / 3600, 2),
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
