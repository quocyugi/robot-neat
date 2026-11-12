"""The slide's DUnE step in the 3D approximation: take a 2D controller, add pitch/roll inputs, fine-tune 30 generations on 3D worlds.

The new inputs start with zero weights, so generation 0 behaves exactly like the 2D controller.
Train episodes are separate from the 100 test episodes of each world.

Usage: python -I d13_finetune3d.py <repo> <dune_root> <cache_dir> <results_dir> <run_dir> <out_dir> [gens] [pop] [k_per_world] [threads] [sigma0] [seed]
"""
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tests2"))

import numpy as np

repo, dune, cache, res_dir, run_dir, out_dir = sys.argv[1:7]
gens = int(sys.argv[7]) if len(sys.argv) > 7 else 30
pop = int(sys.argv[8]) if len(sys.argv) > 8 else 32
k_w = int(sys.argv[9]) if len(sys.argv) > 9 else 4
threads = int(sys.argv[10]) if len(sys.argv) > 10 else 3
sigma0 = float(sys.argv[11]) if len(sys.argv) > 11 else 0.1
seed = int(sys.argv[12]) if len(sys.argv) > 12 else 0
sys.path.insert(0, repo)

import cma

import fastsim as fs
import sim3d as S
import world3d as W

os.makedirs(out_dir, exist_ok=True)
WORLDS = ["flat", "pillars", "maze", "construction", "inspect", "island", "rubicon"]
s = json.load(open(os.path.join(run_dir, "summary.json"), encoding="utf-8"))
assert s["algo"] in ("cma", "ga") and s.get("arch", "mlp") == "mlp" and s["n_rays"] == 12
base = pickle.load(open(os.path.join(run_dir, "champion.pkl"), "rb"))
h = int(str(s["layers"]).split(",")[0])
L15, L17 = [15, h, 2], [17, h, 2]


def extend(p):
    W1 = p[:15 * h].reshape(15, h)
    return np.concatenate([np.vstack([W1, np.zeros((2, h))]).ravel(), p[15 * h:]])


class Batch:
    """MLP batch where each robot uses params[owner[i]]."""
    def __init__(self, params, owner, layers):
        self.net = fs.MLPBatch(params[owner], layers)

    def reset(self, n):
        pass

    def act(self, idx, obs):
        return self.net.act(idx, obs)


class WD:
    pass


T = {}
rng = np.random.default_rng(1000 + seed)
t0 = time.time()
for w in WORLDS:
    d = WD()
    task = W.TASKS[w]
    Vg, Fg, Vl, Fl = W.load_world(w, dune, cache)
    d.sim = S.Sim3D(Vg, Fg, Vl, Fl, nthreads=threads)
    d.task = task
    sp, tf = task["spawn"], task["too_far"]
    margin = 8.0
    region = (max(tf[0], sp[0] - margin), min(tf[1], sp[1] + margin), max(tf[2], sp[2] - margin), min(tf[3], sp[3] + margin))
    d.grid = S.Grid(d.sim, region, res=0.10, z0=task["z0"])
    d.train, _ = d.grid.sample_episodes(task["spawn"], task["goal"], 40, rng)
    # geodesic field of every training episode
    fields = np.zeros((40, d.grid.ny, d.grid.nx), np.float32)
    for i in range(40):
        f = d.grid.distance_from(d.train["gx"][i], d.train["gy"][i])
        fields[i] = np.where(np.isfinite(f), f, 1e3)
    d.fields = fields
    d.test = json.load(open(os.path.join(res_dir, f"episodes_{w}.json"), encoding="utf-8"))
    T[w] = d
print(f"prepared {len(T)} worlds in {time.time() - t0:.0f}s", flush=True)


def evaluate(xs, epi_idx_per_world):
    """Mean fitness of each parameter vector over the chosen training episodes of every world."""
    P = len(xs)
    fit = np.zeros(P)
    total = 0
    n_succ = np.zeros(P)
    for w, d in T.items():
        idx = epi_idx_per_world[w]
        K = len(idx)
        owner = np.repeat(np.arange(P), K)
        ep_of = np.tile(idx, P)
        ep = {k: np.asarray(v)[ep_of] for k, v in d.train.items()}

        def geo_fn(ids, x, y):
            r, c = d.grid.cell(x, y)
            r = np.clip(r, 0, d.grid.ny - 1)
            c = np.clip(c, 0, d.grid.nx - 1)
            return d.fields[ep_of[ids], r, c]
        R = S.rollout3d(d.sim, d.task, ep, Batch(xs, owner, L17), ground_filter="none", use_imu=True, geo_fn=geo_fn)
        succ = R.reason == 1
        prog = np.clip((ep["geo"] - R.gmin) / np.maximum(ep["geo"], 1e-5), 0, 1)
        f = 10.0 * prog + 10.0 * succ + 5.0 * succ * (1.0 - R.steps / d.sim.cfg.max_ticks) - 2.0 * (R.reason == 2) - 2.0 * (R.reason == 4) - 0.1 * R.blocks
        fit += np.bincount(owner, f, P) / K
        n_succ += np.bincount(owner, succ.astype(float), P) / K
        total += int(R.steps.sum())
    return fit / len(T), n_succ / len(T), total


def test_all(x):
    out = {}
    for w, d in T.items():
        class Q:
            def reset(self, n):
                self.net = fs.MLPBatch(np.repeat(x[None, :], n, axis=0), L17)

            def act(self, idx, obs):
                return self.net.act(idx, obs)
        R = S.rollout3d(d.sim, d.task, d.test, Q(), ground_filter="none", use_imu=True)
        out[w] = S.summarize3d(R, d.test)
    return out


x0 = extend(np.asarray(base, float))
before = test_all(x0)
print("before", {w: before[w]["SR"] for w in before}, flush=True)
es = cma.CMAEvolutionStrategy(x0, sigma0, {"popsize": pop, "seed": seed + 1, "verbose": -9, "CMA_diagonal": 3})
np_rng = np.random.default_rng(seed)
log, best, best_f = [], x0.copy(), -1e9
for g in range(1, gens + 1):
    t1 = time.time()
    xs = np.array(es.ask())
    sel = {w: np_rng.choice(40, k_w, replace=False) for w in T}
    f, sc, steps = evaluate(xs, sel)
    es.tell(list(xs), list(-f))
    i = int(np.argmax(f))
    if f[i] > best_f:          # re-score the challenger on the same fixed 40 episodes? no: keep the mean of CMA below
        best_f, best = float(f[i]), xs[i].copy()
    log.append(dict(gen=g, f_best=round(float(f.max()), 3), f_mean=round(float(f.mean()), 3), sr_mean=round(float(sc.mean()), 3), sec=round(time.time() - t1, 1), sigma=round(float(es.sigma), 4)))
    print(json.dumps(log[-1]), flush=True)
final_mean = np.asarray(es.result.xfavorite)
after_mean = test_all(final_mean)
after_best = test_all(best)
res = {"run_dir": run_dir, "gens": gens, "pop": pop, "k_per_world": k_w, "sigma0": sigma0, "seed": seed, "evals": gens * pop * k_w * len(WORLDS),
       "before": before, "after_cma_mean": after_mean, "after_best_individual": after_best, "log": log, "wall_s": round(time.time() - t0, 1)}
with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f_:
    json.dump(res, f_, indent=1)
with open(os.path.join(out_dir, "tuned_params.pkl"), "wb") as f_:
    pickle.dump({"cma_mean": final_mean, "best_individual": best, "layers": L17}, f_)
print("SR before/after:", {w: (before[w]["SR"], after_mean[w]["SR"], after_best[w]["SR"]) for w in T}, flush=True)
