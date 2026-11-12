"""Check fastsim against the team's RobotNavEnv, step-exact, on random networks. Also measures throughput.

Usage: python d00_equivalence.py <repo_root> <out_json>
"""
import json
import os
import random
import sys
import time

REPO, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import neat
import numpy as np

from src.env.robot_env import RobotNavEnv
from src.neat_engine.evaluator import compute_episode_fitness
from src.baselines.fixed_ga import FixedMLP
import fastsim as fs

rng = np.random.default_rng(0)
keys = []
for rel in ("train/level_1", "train/level_2", "train/level_3", "train/level_4", "test_barn", "test_lehman"):
    k = fs.keys_of(REPO, rel)
    keys += list(rng.choice(k, min(6, len(k)), replace=False))

maps = {k: fs.load_json(os.path.join(REPO, "data", "maps", k)) for k in keys}
layers = [15, 8, 2]
P = 12
params = rng.normal(0, 1, (P, FixedMLP(layers).num_params))

cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                  os.path.join(REPO, "configs/neat_recurrent.ini"))
pop = neat.Population(cfg, seed=0)
genomes = list(pop.population.values())[:P]

mismatch = 0
checked = 0
worst_f = 0.0
for k, data in maps.items():
    fm = fs.FastMap(data, k)
    # (a) MLP
    res = fs.rollout(fm, fs.MLPBatch(params, layers), P)
    fb = fs.fit_euclid(res, fm)
    for i in range(P):
        net = FixedMLP(layers)
        net.set_params(params[i])
        env = RobotNavEnv(map_data=data)
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(np.array(net.activate(obs.tolist()), dtype=np.float32))
            done = term or trunc
        f = compute_episode_fitness(info)
        checked += 1
        ok = (info["steps"] == res.steps[i] and bool(info["is_success"]) == bool(res.succ[i]) and
              bool(info["is_collision"]) == bool(res.col[i]) and info["spin_steps"] == res.spin[i] and
              abs(info["final_pos"][0] - res.fx[i]) < 1e-6 and abs(info["final_pos"][1] - res.fy[i]) < 1e-6)
        worst_f = max(worst_f, abs(f - fb[i]))
        mismatch += int(not ok)
    # (b) NEAT recurrent networks
    nets = [neat.nn.RecurrentNetwork.create(g, cfg) for g in genomes]
    res = fs.rollout(fm, fs.NetsPolicy(nets), P)
    fb = fs.fit_euclid(res, fm)
    for i, g in enumerate(genomes):
        net = neat.nn.RecurrentNetwork.create(g, cfg)
        env = RobotNavEnv(map_data=data)
        obs, _ = env.reset()
        net.reset()
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(np.array(net.activate(obs.tolist()), dtype=np.float32))
            done = term or trunc
        f = compute_episode_fitness(info)
        checked += 1
        ok = (info["steps"] == res.steps[i] and bool(info["is_success"]) == bool(res.succ[i]) and
              bool(info["is_collision"]) == bool(res.col[i]) and abs(info["final_pos"][0] - res.fx[i]) < 1e-6)
        worst_f = max(worst_f, abs(f - fb[i]))
        mismatch += int(not ok)

# Throughput
out = {"episodes_compared": checked, "mismatches": mismatch, "max_fitness_difference": float(worst_f)}
fmL4 = fs.FastMap(fs.load_json(os.path.join(REPO, "data/maps/train/level_4/map_000000.json")), "l4")


class Idle:
    def reset(self, n):
        pass

    def act(self, idx, obs):
        return np.zeros((len(idx), 2), np.float32)


t0 = time.perf_counter()
for _ in range(2):
    fs.rollout(fmL4, Idle(), 150, max_steps=500)
dt = time.perf_counter() - t0
out["batch_150_envs_steps_per_s_one_process"] = round(2 * 150 * 500 / dt)
big = rng.normal(0, 1, (150, FixedMLP(layers).num_params))
t0 = time.perf_counter()
n = 0
for _ in range(2):
    r = fs.rollout(fmL4, fs.MLPBatch(big, layers), 150, max_steps=500)
    n += int(r.steps.sum())
out["batch_mlp_steps_per_s_one_process"] = round(n / (time.perf_counter() - t0))
print(json.dumps(out, indent=1))
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
