"""Run 2D-trained and hand-written controllers on DUnE's worlds, without retraining.

Usage: python -I d11_transfer3d.py <repo> <dune_root> <cache_dir> <results_dir> <out_json> <run_dirs_csv> [worlds_csv] [threads]
Each entry of run_dirs_csv is label=path_to_a_train_fast_output_folder
"""
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TESTS2 = os.path.join(os.path.dirname(HERE), "tests2")
sys.path.insert(0, TESTS2)

import numpy as np

repo, dune, cache, res_dir, out_json, runs_csv = sys.argv[1:7]
worlds = sys.argv[7].split(",") if len(sys.argv) > 7 else ["flat", "pillars", "maze", "construction", "inspect", "island", "rubicon"]
threads = int(sys.argv[8]) if len(sys.argv) > 8 else 1
sys.path.insert(0, repo)

import fastsim as fs
import planners as pl
import sim3d as S
import world3d as W


def load_controller(path):
    s = json.load(open(os.path.join(path, "summary.json"), encoding="utf-8"))
    payload = pickle.load(open(os.path.join(path, "champion.pkl"), "rb"))
    if s["algo"] == "neat":
        import neat
        cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, os.path.join(path, "patched.ini"))
        cls = neat.nn.RecurrentNetwork if s.get("net") == "rnn" else neat.nn.FeedForwardNetwork

        class P:
            def reset(self, n):
                self.p = fs.NetsPolicy([cls.create(payload, cfg) for _ in range(n)])
                self.p.reset(n)

            def act(self, idx, obs):
                return self.p.act(idx, obs)
        return P(), s
    layers = [s["n_rays"] + 3] + [int(x) for x in str(s["layers"]).split(",")] + [2]
    arch = s.get("arch", "mlp")

    class Q:
        def reset(self, n):
            p = np.repeat(payload[None, :], n, axis=0)
            self.p = fs.ElmanBatch(p, layers) if arch == "elman" else fs.MLPBatch(p, layers)
            self.p.reset(n)

        def act(self, idx, obs):
            return self.p.act(idx, obs)
    return Q(), s


controllers = {"go_to_goal": lambda: S.GoToGoal(), "apf": lambda: S.APF(), "bug_open": lambda: S.PerRobot(lambda: pl.BugPolicy("open"))}
for item in runs_csv.split(","):
    if "=" in item:
        label, p = item.split("=", 1)
        ctrl, _ = load_controller(p)
        controllers[label] = (lambda c=ctrl: c)

results = {}
for w in worlds:
    t0 = time.time()
    task = W.TASKS[w]
    Vg, Fg, Vl, Fl = W.load_world(w, dune, cache)
    sim = S.Sim3D(Vg, Fg, Vl, Fl, nthreads=threads)
    ep = json.load(open(os.path.join(res_dir, f"episodes_{w}.json"), encoding="utf-8"))
    results[w] = {}
    for name, make in controllers.items():
        for gf in ("none", "normal"):
            if gf == "normal" and name == "go_to_goal":
                continue
            t1 = time.time()
            R = S.rollout3d(sim, task, ep, make(), ground_filter=gf)
            row = S.summarize3d(R, ep)
            row["seconds"] = round(time.time() - t1, 1)
            key = f"{name}|{'raw' if gf == 'none' else 'ideal_ground_filter'}"
            results[w][key] = row
            print(f"{w:13s} {key:34s} SR={row['SR']:.2f} stuck={row['stuck']:.2f} to={row['timeout']:.2f} far={row['too_far']:.2f} fall={row['fall']:.2f} "
                  f"TC={row['TC_blocks_per_ep']:.1f} SPL={row['SPL']:.2f} MTT={row['MTT_s']} gnd3m={row['ground_hit_frac_3m']:.2f} {row['seconds']:.0f}s", flush=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"--- {w}: {time.time() - t0:.0f}s, rays cast so far {sim.rays / 1e6:.1f}M", flush=True)
