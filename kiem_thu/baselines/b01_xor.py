"""Baseline 1: XOR with neat-python 2.0.0, the engine check the literature gives numbers for.

Two configurations: the library's documented XOR settings, and the team's own NEAT settings (tanh, no structural boost).
Published reference: NEAT paper, 100 runs, 32 generations, 4,755 evaluations (sd 2,553), no failure.
neat-python documentation: 50-150 generations.

Usage: python b01_xor.py <team_config.ini> <out_json> [seeds] [max_gens]
"""
import json
import multiprocessing as mp
import os
import re
import sys
import tempfile
import time

import numpy as np

import neat

XOR_IN = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
XOR_OUT = [0.0, 1.0, 1.0, 0.0]

DOCS = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 3.9
pop_size              = 150
reset_on_extinction   = False
no_fitness_termination = False

[DefaultGenome]
activation_default      = sigmoid
activation_mutate_rate  = 0.0
activation_options      = sigmoid
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
conn_add_prob           = 0.5
conn_delete_prob        = 0.5
enabled_default         = True
enabled_mutate_rate     = 0.01
feed_forward            = True
initial_connection      = full_direct
node_add_prob           = 0.2
node_delete_prob        = 0.2
num_hidden              = 0
num_inputs              = 2
num_outputs             = 1
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
min_species_size   = 2
"""


def evaluate(genomes, config):
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        f = 4.0
        for x, y in zip(XOR_IN, XOR_OUT):
            f -= (net.activate(x)[0] - y) ** 2
        g.fitness = f


def one(args):
    path, seed, max_gens = args
    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, path)
    pop = neat.Population(cfg, seed=seed)
    evals = {"n": 0}

    def fn(genomes, config):
        evals["n"] += len(genomes)
        evaluate(genomes, config)
    best = pop.run(fn, max_gens)
    solved = best.fitness >= 3.9
    return dict(seed=seed, solved=bool(solved), gens=pop.generation, evals=evals["n"], best=round(float(best.fitness), 4), nodes=len(best.nodes))


def main():
    team_cfg, out = sys.argv[1], sys.argv[2]
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    max_gens = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    tmp = tempfile.mkdtemp()
    docs_path = os.path.join(tmp, "docs.ini")
    open(docs_path, "w", encoding="utf-8").write(DOCS)
    txt = open(team_cfg, encoding="utf-8", errors="ignore").read()
    subs = {"num_inputs": "2", "num_outputs": "1", "fitness_threshold": "3.9", "no_fitness_termination": "False", "feed_forward": "True",
            "activation_default": "sigmoid", "activation_options": "sigmoid"}
    for k, v in subs.items():
        txt = re.sub(rf"(?m)^{k}\s*=.*$", f"{k} = {v}", txt)
    team_path = os.path.join(tmp, "team.ini")
    open(team_path, "w", encoding="utf-8").write(txt)
    res = {}
    for name, path in (("docs_settings", docs_path), ("team_settings_sigmoid", team_path)):
        t0 = time.time()
        with mp.get_context("spawn").Pool(2) as pool:
            rows = pool.map(one, [(path, s, max_gens) for s in range(seeds)], chunksize=2)
        solved = [r for r in rows if r["solved"]]
        gens = np.array([r["gens"] for r in solved]) if solved else np.array([np.nan])
        evs = np.array([r["evals"] for r in solved]) if solved else np.array([np.nan])
        res[name] = dict(runs=seeds, max_gens=max_gens, solved=len(solved), gens_mean=round(float(np.mean(gens)), 1), gens_median=float(np.median(gens)), gens_max=int(np.max(gens)),
                         evals_mean=round(float(np.mean(evs)), 0), evals_sd=round(float(np.std(evs)), 0), seconds=round(time.time() - t0, 1))
        print(name, res[name], flush=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
