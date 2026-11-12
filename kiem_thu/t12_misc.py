"""T12 - Assorted checks that do not need training.

 (a) generator determinism: regenerate maps and compare with the shipped files
 (b) stats.py: pairwise_tests + plot_boxplot on synthetic data; smallest achievable p-values
 (c) metrics.py on real rollout infos (SPL path)
 (d) novelty: descriptor aliasing and archive across different maps
 (e) AdaptiveHybridManager: lambda after a curriculum drop in fitness
 (f) M4 vs M6 config equality
 (g) GA and random-search smoke tests
 (h) visualize.py headless

Usage: python t12_misc.py <repo_root> <out_json> <python_exe>
"""
import glob
import json
import os
import subprocess
import sys
import time
import warnings

REPO, OUT, PY = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np

res = {}


def maps_in(rel):
    return [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join("data", "maps", rel, "*.json")))]


# (a) generator determinism
from src.maps.generator import MapGenerator

gen = MapGenerator()
det = {}
for lvl, seeds in {1: range(0, 5), 2: range(0, 5), 3: range(0, 5), 4: range(0, 3)}.items():
    same = total = 0
    t0 = time.time()
    for s in seeds:
        m = gen.generate(lvl, s)
        ref = json.load(open(f"data/maps/train/level_{lvl}/map_{s:06d}.json", encoding="utf-8"))
        total += 1
        ok = (m is not None and np.allclose(m["start"], ref["start"]) and np.allclose(m["goal"], ref["goal"])
              and len(m["segments"]) == len(ref["segments"]) and np.allclose(m["segments"], ref["segments"]))
        same += int(ok)
    det[f"level_{lvl}"] = {"identical": same, "checked": total, "seconds": round(time.time() - t0, 1)}
res["generator_determinism"] = det

# (b) stats
from src.experiments import stats as st

rng = np.random.default_rng(0)
methods = {f"m{i}": rng.normal(0.5 + 0.03 * i, 0.05, 10) for i in range(13)}
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    cmp = st.pairwise_tests(methods)
    try:
        st.plot_boxplot(methods, output_path=os.path.join(os.path.dirname(OUT), "t12_boxplot.png"))
        box = "ok"
    except Exception as e:
        box = f"{type(e).__name__}: {e}"
res["stats"] = {"n_pairs": len(cmp), "corrected_alpha": cmp[0]["corrected_alpha"], "boxplot": box,
                "warnings": sorted({str(x.message)[:140] for x in w})}
from scipy import stats as sps

a = np.arange(10, 20, dtype=float)
b = np.arange(0, 10, dtype=float)
p_auto = float(sps.mannwhitneyu(a, b, alternative="two-sided").pvalue)
p_exact = float(sps.mannwhitneyu(a, b, alternative="two-sided", method="exact").pvalue)
# With ties, e.g. run-level SR on 200 maps often ties at 0 or 1
tie_a = np.array([0.6, 0.6, 0.55, 0.65, 0.6, 0.7, 0.6, 0.65, 0.6, 0.55])
tie_b = np.array([0.5, 0.55, 0.5, 0.5, 0.6, 0.5, 0.45, 0.5, 0.55, 0.5])
res["mwu_limits"] = {
    "perfect_separation_10v10_auto_p": p_auto,
    "perfect_separation_10v10_exact_p": p_exact,
    "bonferroni_alpha_78_pairs": 0.05 / 78,
    "holm_first_step_alpha": 0.05 / 78,
    "ties_example_auto_p": float(sps.mannwhitneyu(tie_a, tie_b, alternative="two-sided").pvalue),
}
# Fisher exact on run-level "solved" counts with 10 vs 10 runs
fe = {}
for x, y in [(10, 0), (9, 1), (8, 2), (7, 3), (6, 1), (5, 0), (4, 0)]:
    fe[f"{x}/10 vs {y}/10"] = round(float(sps.fisher_exact([[x, 10 - x], [y, 10 - y]])[1]), 5)
res["fisher_10v10"] = fe

# (c) metrics
from src.experiments.metrics import compute_all_metrics
from src.env.robot_env import RobotNavEnv

m = maps_in("test/level_1")[:2]
infos = []
for mp in m:
    env = RobotNavEnv(map_data=mp)
    obs, _ = env.reset()
    term = trunc = False
    while not (term or trunc):
        diff = env.goal_pos - env.pos
        err = (np.arctan2(diff[1], diff[0]) - env.theta + np.pi) % (2 * np.pi) - np.pi
        u = np.clip(err, -1, 1)
        a = np.array([0.6 - u * 0.5, 0.6 + u * 0.5], dtype=np.float32)
        obs, _, term, trunc, info = env.step(a)
    infos.append(info)
lens = [x["metrics"]["optimal_path_length"] for x in m]
res["metrics_smoke"] = compute_all_metrics(infos, lens)

# (d) novelty
from src.neat_engine.novelty import AdaptiveHybridManager, NoveltyArchive

# Two very different behaviours, identical average descriptor over 4 maps
b1 = np.array([[1.0, 1.0], [9.0, 9.0], [1.0, 9.0], [9.0, 1.0]])
b2 = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
res["novelty_aliasing"] = {"mean_b1": b1.mean(0).tolist(), "mean_b2": b2.mean(0).tolist(),
                           "L2_between_means": float(np.linalg.norm(b1.mean(0) - b2.mean(0))),
                           "L2_between_concatenated": float(np.linalg.norm(b1.ravel() - b2.ravel()))}

# (e) Lambda after a curriculum promotion: fitness drops from ~18 to ~4 then slowly rises
mgr = AdaptiveHybridManager()
trace = []
fit = list(np.linspace(4, 18, 40)) + list(np.linspace(4, 12, 60))
for f in fit:
    trace.append(mgr.update(float(f)))
res["lambda_after_promotion"] = {
    "gens_before_drop_lambda_0.6": int(sum(1 for t in trace[:40] if abs(t - 0.6) < 1e-9)),
    "gens_after_drop": 60,
    "gens_after_drop_at_0.1": int(sum(1 for t in trace[40:] if abs(t - 0.1) < 1e-9)),
    "note": "fitness keeps improving after the drop yet stays below the all-time max",
}

# (f) M4 vs M6
from src.experiments.run_ablation import ABLATION_CONFIGS as AC

m4 = {k: v for k, v in AC["M4"].items() if k != "description"}
m6 = {k: v for k, v in AC["M6"].items() if k != "description"}
res["M4_equals_M6"] = {"identical_arguments": m4 == m6, "M4": m4, "M6": m6}

# (g) GA + random search smoke
from src.baselines.fixed_ga import SimpleGA, FixedMLP
from src.baselines.random_search import random_search

np.random.seed(0)
ms = maps_in("train/level_1")[:2]
t0 = time.time()
ga = SimpleGA([15, 8, 2], pop_size=10)
b0 = ga.evaluate(ms)
ga.evolve()
b1_ = ga.evaluate(ms)
net, fit_rs = random_search([15, 8, 2], ms, n_trials=20)
res["baseline_smoke"] = {"ga_params_B1": ga.n_params, "ga_params_B2": FixedMLP([15, 16, 8, 2]).num_params,
                         "ga_best_gen0": b0, "ga_best_gen1": b1_, "random_search_best": fit_rs,
                         "seconds": round(time.time() - t0, 1)}

# (h) visualize.py headless (dummy video driver) with a random genome
import pickle
import neat
cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                  neat.DefaultStagnation, "configs/neat_recurrent.ini")
p = neat.Population(cfg, seed=0)
g = next(iter(p.population.values()))
pk = os.path.join(os.path.dirname(OUT), "t12_random_genome.pkl")
pickle.dump(g, open(pk, "wb"))
env_vars = dict(os.environ, SDL_VIDEODRIVER="dummy", PYTHONIOENCODING="utf-8")
r = subprocess.run([PY, "scripts/visualize.py", "--genome", pk, "--map", "data/maps/test/level_1/map_200000.json"],
                   capture_output=True, text=True, timeout=120, env=env_vars, encoding="utf-8", errors="replace")
res["visualize_headless"] = {"returncode": r.returncode, "stdout_tail": r.stdout[-200:], "stderr_tail": r.stderr[-300:]}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False, default=str)
print(json.dumps(res, indent=1, ensure_ascii=False, default=str))
