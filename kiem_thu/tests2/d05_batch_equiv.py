"""BatchRecurrentPolicy must give exactly the same episode results as neat-python's RecurrentNetwork, on the team's maps.

Usage: python d05_batch_equiv.py <repo> <neat_ini> <out_json>
"""
import glob
import json
import os
import sys
import time

REPO, INI, OUT = sys.argv[1:4]
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import neat
import numpy as np

import fastsim as fs

cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, INI)
pop = neat.Population(cfg, seed=11)
rng = np.random.default_rng(0)


def fn(genomes, config):
    for _, g in genomes:
        g.fitness = float(rng.random())


pop.run(fn, 40)
gs = list(pop.population.values())
keys = sorted(glob.glob(os.path.join(REPO, "data", "maps", "test", "level_*", "*.json")))[::25]
mism, total, t_lib, t_batch = 0, 0, 0.0, 0.0
for p in keys:
    fm = fs.FastMap(fs.load_json(p), "")
    t0 = time.time()
    r1 = fs.rollout(fm, fs.NetsPolicy([neat.nn.RecurrentNetwork.create(g, cfg) for g in gs]), len(gs))
    t1 = time.time()
    r2 = fs.rollout(fm, fs.BatchRecurrentPolicy(gs, cfg), len(gs))
    t2 = time.time()
    t_lib += t1 - t0
    t_batch += t2 - t1
    total += len(gs)
    mism += int(((r1.succ != r2.succ) | (r1.col != r2.col) | (r1.steps != r2.steps)).sum())
res = dict(genomes=len(gs), maps=len(keys), episodes=total, mismatches=mism, seconds_library=round(t_lib, 1), seconds_batch=round(t_batch, 1), speedup=round(t_lib / max(t_batch, 1e-6), 1))
print(json.dumps(res), flush=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1)
