"""Training with the minimal bug-fixes applied, to test whether the pipeline is FEASIBLE.

Reuses the team's environment, fitness, NEAT config, GA, novelty and curriculum data.
Changes vs. the repo (each one corresponds to a finding in the report):
  F1  no_fitness_termination = True (curriculum is no longer cut off at level 1)
  F2  parallel evaluation with multiprocessing (spawn-safe) and explicit seeding
  F3  champion picked on the validation set from the top-5 RAW-fitness genomes
  F4  K maps = 2 from the current level + 2 from easier levels (no forgetting)
  F5  novelty = k-NN within the current population on a concatenated 2K-dim descriptor
      (all genomes share the same maps; no archive of positions from other maps)
  F6  hybrid score on stationary scales (raw/19 and novelty/10) instead of per-generation min-max
  F7  on promotion: reset NEAT stagnation clocks and the lambda manager; lambda uses a 5-gen moving mean

Usage:
  python run_fixed.py --repo R --algo neat --net rnn --novelty pop --seed 0 --gens 300 --workers 8 --out DIR
  python run_fixed.py --repo R --algo ga --layers 8 --seed 0 --gens 300 --workers 8 --out DIR
"""
import argparse
import copy
import glob
import json
import multiprocessing as mp
import os
import pickle
import random
import re
import sys
import time

import numpy as np

_W = {}


def _load_all_maps(repo):
    root = os.path.join(repo, "data", "maps")
    out = {}
    for p in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
        rel = os.path.relpath(p, root).replace("\\", "/")
        if rel.startswith("barn_all/"):
            continue
        with open(p, encoding="utf-8") as f:
            out[rel] = json.load(f)
    return out


def init_worker(repo, cfg_path, recurrent, algo, layers):
    sys.path.insert(0, repo)
    import neat
    from src.env.robot_env import RobotNavEnv
    from src.neat_engine.evaluator import compute_episode_fitness
    from src.baselines.fixed_ga import FixedMLP
    _W["neat"] = neat
    _W["Env"] = RobotNavEnv
    _W["fit"] = compute_episode_fitness
    _W["MLP"] = FixedMLP
    _W["recurrent"] = recurrent
    _W["algo"] = algo
    _W["layers"] = layers
    _W["maps"] = _load_all_maps(repo)
    if algo == "neat":
        _W["cfg"] = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, cfg_path)


def _make_net(payload):
    if _W["algo"] == "neat":
        neat = _W["neat"]
        cls = neat.nn.RecurrentNetwork if _W["recurrent"] else neat.nn.FeedForwardNetwork
        return cls.create(payload, _W["cfg"])
    net = _W["MLP"](_W["layers"])
    net.set_params(payload)
    return net


def _rollout(net, key):
    env = _W["Env"](map_data=_W["maps"][key])
    obs, _ = env.reset()
    if hasattr(net, "reset"):
        net.reset()
    term = trunc = False
    info = None
    while not (term or trunc):
        act = np.array(net.activate(obs.tolist()), dtype=np.float32)
        obs, _, term, trunc, info = env.step(act)
    env.close()
    return info


def worker_eval(task):
    payload, keys = task
    net = _make_net(payload)
    out = []
    for key in keys:
        info = _rollout(net, key)
        out.append((float(_W["fit"](info)), bool(info["is_success"]), bool(info["is_collision"]),
                    int(info["steps"]), info["final_pos"]))
    return out


def worker_score(task):
    payload, keys = task
    net = _make_net(payload)
    out = []
    for key in keys:
        info = _rollout(net, key)
        t = np.asarray(info["trajectory"])
        length = float(np.sum(np.linalg.norm(np.diff(t, axis=0), axis=1)))
        lopt = float(_W["maps"][key]["metrics"]["optimal_path_length"])
        out.append((bool(info["is_success"]), bool(info["is_collision"]), int(info["steps"]), length, lopt))
    return out


# ---------------------------------------------------------------------- main-process helpers
def patch_config(src, dst, pop=None, thr=None, survival=None, replace=None, node_add=None, conn_add=None):
    txt = open(src, encoding="utf-8").read()
    if node_add is not None:
        txt = re.sub(r"node_add_prob\s*=\s*[\d.]+", f"node_add_prob           = {node_add}", txt)
    if conn_add is not None:
        txt = re.sub(r"conn_add_prob\s*=\s*[\d.]+", f"conn_add_prob           = {conn_add}", txt)
    txt = re.sub(r"no_fitness_termination\s*=\s*\w+", "no_fitness_termination  = True", txt)
    if pop:
        txt = re.sub(r"pop_size\s*=\s*\d+", f"pop_size                = {pop}", txt)
    if thr is not None:
        txt = re.sub(r"compatibility_threshold\s*=\s*[\d.]+", f"compatibility_threshold = {thr}", txt)
    if survival is not None:
        txt = re.sub(r"survival_threshold\s*=\s*[\d.]+", f"survival_threshold = {survival}", txt)
    if replace is not None:
        txt = re.sub(r"weight_replace_rate\s*=\s*[\d.]+", f"weight_replace_rate     = {replace}", txt)
        txt = re.sub(r"bias_replace_rate\s*=\s*[\d.]+", f"bias_replace_rate       = {replace}", txt)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(txt)
    return dst


def keys_of(repo, rel):
    root = os.path.join(repo, "data", "maps")
    return [os.path.relpath(p, root).replace("\\", "/") for p in sorted(glob.glob(os.path.join(root, rel, "*.json")))]


def summarize(rows):
    n = len(rows)
    succ = [r for r in rows if r[0]]
    spl = sum(min(1.0, r[4] / max(r[3], r[4])) for r in succ) / n if n else 0.0
    return {"SR": round(len(succ) / n, 4), "CR": round(sum(1 for r in rows if r[1]) / n, 4),
            "timeout": round(sum(1 for r in rows if not r[0] and not r[1]) / n, 4),
            "SPL": round(spl, 4),
            "MTT": round(float(np.mean([r[2] * 0.1 for r in succ])), 2) if succ else None, "n": n}


class Trainer:
    def __init__(self, args, pool):
        self.a = args
        self.pool = pool
        self.rng = np.random.default_rng(args.seed)
        self.level = 1
        self.gen = 0
        self.rows = []
        self.promotions = []
        repo = args.repo
        self.train = {L: keys_of(repo, f"train/level_{L}") for L in (1, 2, 3, 4)}
        self.val = {L: keys_of(repo, f"val/level_{L}") for L in (1, 2, 3, 4)}
        self.tests = {f"test_L{L}": keys_of(repo, f"test/level_{L}") for L in (1, 2, 3, 4)}
        self.tests["test_barn"] = keys_of(repo, "test_barn")
        self.tests["test_lehman"] = keys_of(repo, "test_lehman")
        self.best_score = -1.0
        self.best_gen = None
        self.best_payload = None
        self.best_val_detail = None
        self.best_raw_hist = []
        self.mgr = None
        self.species_ref = None

    # -------------------------------------------- maps
    def sample_maps(self):
        L = self.level
        K = self.a.k
        if L == 1:
            return list(self.rng.choice(self.train[1], K, replace=False))
        n_cur = K - K // 2
        cur = list(self.rng.choice(self.train[L], n_cur, replace=False))
        lower = sum((self.train[l] for l in range(1, L)), [])
        return cur + list(self.rng.choice(lower, K // 2, replace=False))

    # -------------------------------------------- scoring
    def eval_batch(self, payloads, keys):
        chunk = max(1, len(payloads) // (self.a.workers * 4))
        return self.pool.map(worker_eval, [(p, keys) for p in payloads], chunksize=chunk)

    def sr_on(self, payload, keys):
        rows = self.pool.apply(worker_score, ((payload, keys),))
        return float(np.mean([r[0] for r in rows]))

    def sr_parallel(self, payload, keys):
        # split keys over workers
        parts = [keys[i::self.a.workers] for i in range(self.a.workers)]
        parts = [p for p in parts if p]
        res = self.pool.map(worker_score, [(payload, p) for p in parts])
        rows = sum(res, [])
        return float(np.mean([r[0] for r in rows]))

    def maybe_validate(self, payloads, raw):
        if self.gen % self.a.val_every != 0:
            return None, None
        top = np.argsort(raw)[-5:][::-1]
        srs = [self.sr_parallel(payloads[i], self.val[self.level]) for i in top]
        best_i = int(np.argmax(srs))
        sr_best = float(srs[best_i])
        cand = payloads[top[best_i]]
        promoted = False
        if sr_best >= 0.8 and self.level < 4:
            self.level += 1
            promoted = True
            self.promotions.append(self.gen)
            self.on_promotion()
        per = {L: self.sr_parallel(cand, self.val[L]) for L in (1, 2, 3, 4)}
        score = float(np.mean(list(per.values())))
        if score >= self.best_score:
            self.best_score = score
            self.best_gen = self.gen
            self.best_payload = copy.deepcopy(cand)
            self.best_val_detail = per
        return sr_best, promoted

    def on_promotion(self):
        self.best_raw_hist = []
        if self.mgr is not None:
            from src.neat_engine.novelty import AdaptiveHybridManager
            self.mgr = AdaptiveHybridManager()
        if self.species_ref is not None:
            for s in self.species_ref.species.values():
                s.last_improved = self.neat_gen()
                s.fitness_history = []

    def neat_gen(self):
        return self.gen

    # -------------------------------------------- final scoring
    def score_test(self, payload):
        out = {}
        for name, keys in self.tests.items():
            parts = [keys[i::self.a.workers] for i in range(self.a.workers)]
            parts = [p for p in parts if p]
            rows = sum(self.pool.map(worker_score, [(payload, p) for p in parts]), [])
            out[name] = summarize(rows)
        return out

    def log_row(self, raw, res, t0, sr, promoted, extra):
        solved = [sum(r[1] for r in rr) for rr in res]
        steps = [np.mean([r[3] for r in rr]) for rr in res]
        idx = int(np.argmax(raw))
        row = {"gen": self.gen, "level": self.level, "seconds": round(time.time() - t0, 2),
               "raw_best": float(raw.max()), "raw_mean": float(raw.mean()),
               "best_solved_of_4": int(solved[idx]), "mean_solved": float(np.mean(solved)),
               "mean_steps": float(np.mean(steps)), "val_sr": sr, "promoted": promoted}
        row.update(extra)
        self.rows.append(row)
        if self.gen % 10 == 0:
            print(json.dumps(row), flush=True)

    # -------------------------------------------- NEAT
    def neat_eval(self, genomes, config):
        t0 = time.time()
        self.gen += 1
        keys = self.sample_maps()
        payloads = [g for _, g in genomes]
        res = self.eval_batch(payloads, keys)
        raw = np.array([np.mean([r[0] for r in rr]) for rr in res])
        extra = {}
        if self.a.novelty == "pop":
            from src.neat_engine.novelty import AdaptiveHybridManager, NoveltyArchive
            if self.mgr is None:
                self.mgr = AdaptiveHybridManager()
            beh = np.array([np.concatenate([np.asarray(r[4], dtype=float) for r in rr]) for rr in res])
            nov = NoveltyArchive(k_neighbours=15).compute_novelty(beh, population_behaviors=beh)
            self.best_raw_hist.append(float(raw.max()))
            lam = self.mgr.update(float(np.mean(self.best_raw_hist[-5:])))
            early = np.array([any(r[2] and r[3] < 30 for r in rr) for rr in res])
            n_scaled = np.minimum(nov / 10.0, 1.0) * np.where(early, 0.2, 1.0)
            final = lam * (raw / 19.0) + (1.0 - lam) * n_scaled
            final = np.maximum(final, 0.01)
            extra = {"lambda": lam, "nov_mean": float(nov.mean())}
        else:
            final = raw
        for (gid, g), f in zip(genomes, final):
            g.fitness = float(f)
        sr, promoted = self.maybe_validate(payloads, raw)
        extra["n_species"] = len(self.species_ref.species) if self.species_ref else None
        best = payloads[int(np.argmax(raw))]
        extra["best_nodes"] = len(best.nodes)
        extra["best_conns"] = sum(1 for c in best.connections.values() if c.enabled)
        self.log_row(raw, res, t0, sr, promoted, extra)

    # -------------------------------------------- GA baseline
    def run_ga(self):
        from src.baselines.fixed_ga import SimpleGA
        layers = [15] + self.a.layer_list + [2]
        ga = SimpleGA(layers, pop_size=self.a.pop)
        for _ in range(self.a.gens):
            t0 = time.time()
            self.gen += 1
            keys = self.sample_maps()
            payloads = [ga.population[i].copy() for i in range(ga.pop_size)]
            res = self.eval_batch(payloads, keys)
            raw = np.array([np.mean([r[0] for r in rr]) for rr in res])
            ga.fitnesses = raw.copy()
            sr, promoted = self.maybe_validate(payloads, raw)
            self.log_row(raw, res, t0, sr, promoted, {"n_params": ga.n_params})
            ga.evolve()
        return ga


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--algo", choices=["neat", "ga"], default="neat")
    ap.add_argument("--net", choices=["rnn", "ff"], default="rnn")
    ap.add_argument("--novelty", choices=["none", "pop"], default="pop")
    ap.add_argument("--layers", default="8", help="hidden sizes for GA, e.g. 8 or 16,8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--pop", type=int, default=150)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--k", type=int, default=4, help="maps per genome per generation")
    ap.add_argument("--thr", type=float, default=None, help="compatibility_threshold override")
    ap.add_argument("--survival", type=float, default=None, help="survival_threshold override")
    ap.add_argument("--replace", type=float, default=None, help="weight/bias replace rate override")
    ap.add_argument("--node-add", type=float, default=None, help="node_add_prob override")
    ap.add_argument("--conn-add", type=float, default=None, help="conn_add_prob override")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.layer_list = [int(x) for x in a.layers.split(",")]

    sys.path.insert(0, a.repo)
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)

    import neat
    recurrent = a.net == "rnn"
    src_cfg = os.path.join(a.repo, "configs", "neat_recurrent.ini" if recurrent else "neat_feedforward.ini")
    cfg_path = patch_config(src_cfg, os.path.join(a.out, "patched.ini"), a.pop, a.thr, a.survival, a.replace, a.node_add, a.conn_add)
    layers = [15] + a.layer_list + [2]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=init_worker,
                                      initargs=(a.repo, cfg_path, recurrent, a.algo, layers)) as pool:
        tr = Trainer(a, pool)
        extinctions = []
        stag = [0]
        neat_winner = None
        if a.algo == "neat":
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                 neat.DefaultStagnation, cfg_path)
            pop = neat.Population(config, seed=a.seed)
            tr.species_ref = pop.species

            class R(neat.reporting.BaseReporter):
                def complete_extinction(self):
                    extinctions.append(tr.gen)

                def species_stagnant(self, sid, species):
                    stag[0] += 1

            pop.add_reporter(R())
            neat_winner = pop.run(tr.neat_eval, n=a.gens)
            final = neat_winner
        else:
            ga = tr.run_ga()
            final = ga.population[int(np.argmax(ga.fitnesses))].copy()
        wall = time.time() - t0
        if tr.best_payload is None:
            tr.best_payload = final
        summary = {
            "algo": a.algo, "net": a.net, "novelty": a.novelty, "layers": a.layers, "seed": a.seed,
            "gens": a.gens, "pop": a.pop, "workers": a.workers, "wall_seconds": round(wall, 1),
            "final_level": tr.level, "promotions_at": tr.promotions,
            "extinctions": extinctions, "species_stagnated": stag[0],
            "champion_gen": tr.best_gen, "champion_val_mean_sr": tr.best_score,
            "champion_val_per_level": tr.best_val_detail,
            "champion_test": tr.score_test(tr.best_payload),
            "neat_best_genome_test": tr.score_test(final),
            "train_evals": a.pop * a.k * a.gens, "k": a.k, "thr": a.thr, "survival": a.survival, "replace": a.replace,
        }
        if a.algo == "neat":
            g = tr.best_payload
            summary["champion_nodes"] = len(g.nodes)
            summary["champion_conns"] = sum(1 for c in g.connections.values() if c.enabled)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(a.out, "gens.json"), "w", encoding="utf-8") as f:
        json.dump(tr.rows, f, ensure_ascii=False)
    with open(os.path.join(a.out, "champion.pkl"), "wb") as f:
        pickle.dump(tr.best_payload, f)
    print(json.dumps({k: v for k, v in summary.items()}, indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
