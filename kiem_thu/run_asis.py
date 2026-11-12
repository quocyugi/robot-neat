"""Run the team's NEATRunner UNMODIFIED for one ablation config and seed.

Only additions: external seeding (the repo has none), a read-only reporter and a wrapper
that records raw fitness per genome. Evolution logic, fitness, novelty and curriculum are
the repo's own code paths.

Usage: python run_asis.py --repo <root> --model M6 --seed 0 --gens 100 --out <dir>
"""
import argparse
import copy
import glob
import json
import os
import pickle
import random
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--gens", type=int, default=100)
ap.add_argument("--out", required=True)
args = ap.parse_args()

sys.path.insert(0, args.repo)
os.chdir(args.repo)

import neat
import numpy as np

from src.env.robot_env import RobotNavEnv
import src.neat_engine.runner as runner_module
from src.neat_engine.runner import NEATRunner
from src.experiments.run_ablation import ABLATION_CONFIGS
from src.experiments.metrics import compute_all_metrics
from src.neat_engine.evaluator import evaluate_genome_on_maps as _orig_eval

random.seed(args.seed)
np.random.seed(args.seed)

os.makedirs(args.out, exist_ok=True)

RAW, SOLVED = [], []


def wrapped_eval(net, maps, env_cls, is_rec=False):
    mean_fit, infos = _orig_eval(net, maps, env_cls, is_rec)
    RAW.append(mean_fit)
    SOLVED.append(sum(1 for i in infos if i["is_success"]))
    return mean_fit, infos


runner_module.evaluate_genome_on_maps = wrapped_eval


class Tracker(neat.reporting.BaseReporter):
    def __init__(self):
        self.rows = []
        self.extinctions = []
        self.stagnant = 0
        self.solution_generation = None
        self.t0 = time.time()
        self.g = 0
        self.raw_best_last = None
        self.best_key_by_gen = {}

    def start_generation(self, generation):
        self.g = generation
        self.t0 = time.time()
        RAW.clear()
        SOLVED.clear()

    def post_evaluate(self, config, population, species, best_genome):
        idx = int(np.argmax(RAW)) if RAW else 0
        gs = list(population.values())
        raw_best = gs[idx] if RAW else best_genome
        self.raw_best_last = copy.deepcopy(raw_best)
        hist = RUNNER.history[-1] if RUNNER.history else {}
        self.best_key_by_gen[self.g] = (best_genome.key, float(best_genome.fitness))
        self.rows.append({
            "gen": self.g,
            "seconds": round(time.time() - self.t0, 3),
            "n_species": len(species.species),
            "raw_best": float(max(RAW)) if RAW else None,
            "raw_mean": float(np.mean(RAW)) if RAW else None,
            "best_solved_of_K": int(SOLVED[idx]) if SOLVED else 0,
            "any_full_solve": bool(max(SOLVED) >= len(RUNNER_MAPS_K) if SOLVED else False),
            "hybrid_best": float(best_genome.fitness),
            "level": hist.get("level"),
            "lambda": hist.get("lambda"),
            "archive": hist.get("archive_size"),
            "val_sr": hist.get("val_sr"),
            "promoted": hist.get("promoted"),
            "best_nodes": len(raw_best.nodes),
            "best_conns": sum(1 for c in raw_best.connections.values() if c.enabled),
        })

    def complete_extinction(self):
        self.extinctions.append(self.g)

    def species_stagnant(self, sid, species):
        self.stagnant += 1

    def found_solution(self, config, generation, best):
        self.solution_generation = generation


RUNNER_MAPS_K = [0, 0, 0, 0]
cfg = ABLATION_CONFIGS[args.model]
RUNNER = NEATRunner(
    config_path=cfg["config_file"], data_dir="data/maps", output_dir=args.out,
    use_novelty=cfg["use_novelty"], use_curriculum=cfg["use_curriculum"],
    is_recurrent=cfg["is_recurrent"], max_generations=args.gens,
)

t_start = time.time()
population = neat.Population(RUNNER.neat_config)
tracker = Tracker()
population.add_reporter(tracker)
population.add_reporter(neat.StatisticsReporter())
winner = population.run(RUNNER._eval_genomes, n=RUNNER.max_generations)
RUNNER._save_winner(winner)
RUNNER._save_history()
wall = time.time() - t_start


def maps_in(rel):
    return [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join("data", "maps", rel, "*.json")))]


TEST_SETS = {f"test_L{k}": f"test/level_{k}" for k in (1, 2, 3, 4)}
TEST_SETS.update({"test_barn": "test_barn", "test_lehman": "test_lehman"})
loaded = {k: maps_in(v) for k, v in TEST_SETS.items()}


def score(genome):
    net = RUNNER._create_network(genome, RUNNER.neat_config)
    out = {}
    for k, ms in loaded.items():
        _, infos = _orig_eval(net, ms, RobotNavEnv, RUNNER.is_recurrent)
        lens = [m["metrics"]["optimal_path_length"] for m in ms]
        res = compute_all_metrics(infos, lens)
        res["MTT"] = None if res["MTT"] == float("inf") else res["MTT"]
        out[k] = {kk: (None if vv is None else round(float(vv), 4)) for kk, vv in res.items()}
    return out


winner_gen = next((g for g, (key, fit) in tracker.best_key_by_gen.items() if key == winner.key), None)
summary = {
    "model": args.model, "seed": args.seed, "gens_requested": args.gens,
    "gens_run": len(tracker.rows),
    "stopped_early_by_threshold": tracker.solution_generation is not None,
    "solution_generation": tracker.solution_generation,
    "wall_seconds": round(wall, 1),
    "final_level": tracker.rows[-1]["level"],
    "max_level": max((r["level"] or 1) for r in tracker.rows),
    "promotions_at": [r["gen"] for r in tracker.rows if r["promoted"]],
    "extinctions": tracker.extinctions,
    "species_stagnated": tracker.stagnant,
    "species_min": min(r["n_species"] for r in tracker.rows),
    "species_max": max(r["n_species"] for r in tracker.rows),
    "winner_key": winner.key,
    "winner_fitness": float(winner.fitness),
    "winner_generation": winner_gen,
    "winner_level_when_found": next((r["level"] for r in tracker.rows if r["gen"] == winner_gen), None),
    "winner_nodes": len(winner.nodes),
    "winner_conns": sum(1 for c in winner.connections.values() if c.enabled),
    "winner_test": score(winner),
    "last_gen_raw_best_test": score(tracker.raw_best_last),
}
with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
with open(os.path.join(args.out, "gens.json"), "w", encoding="utf-8") as f:
    json.dump(tracker.rows, f, ensure_ascii=False)
with open(os.path.join(args.out, "last_raw_best.pkl"), "wb") as f:
    pickle.dump(tracker.raw_best_last, f)
print(json.dumps({k: v for k, v in summary.items() if not k.endswith("_test")}, indent=2))
