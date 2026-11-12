"""Fast training driver for the diagnosis experiments (batch simulator, spawn-safe multiprocessing).

Algorithms : ga (the team's SimpleGA on a fixed MLP), cma (CMA-ES on the same MLP), neat.
Fitness    : euclid (the team's 5-term fitness) or geo (same terms, progress along the shortest path).
Maps       : curriculum (K-K//2 current level + K//2 easier levels, promotion at SR>=0.8 on val) or uniform (K/4 per level).
Champion   : every 10 generations the top-5 by raw fitness are scored on the validation maps; the best mean SR over 4 levels is kept.

Example:
 python train_fast.py --repo R --algo ga --fitness geo --schedule uniform --k 8 --gens 250 --workers 3 --out DIR
"""
import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_W = {}


def init_worker(cfg):
    sys.path.insert(0, cfg["repo"])
    sys.path.insert(0, HERE)
    import fastsim as fs
    _W.update(cfg)
    _W["fs"] = fs
    _W["cache"] = {}
    if cfg["algo"] == "neat":
        import neat
        _W["neat"] = neat
        _W["cfg"] = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, cfg["cfg_path"])


def get_fm(key):
    c = _W["cache"]
    if key not in c:
        fs = _W["fs"]
        map_root = _W["map_root"]
        data = fs.load_json(os.path.join(map_root, key))
        geo = None
        gp = os.path.join(_W["geo_dir"], key.replace("/", "__") + ".npy")
        if os.path.exists(gp):
            geo = np.load(gp)
        c[key] = fs.FastMap(data, key, n_rays=_W["n_rays"], geo=geo)
    return c[key]


def make_policy(payload):
    fs = _W["fs"]
    if _W["algo"] == "neat":
        neat = _W["neat"]
        if _W["recurrent"] and _W.get("batch_neat"):
            return fs.BatchRecurrentPolicy(payload, _W["cfg"])
        cls = neat.nn.RecurrentNetwork if _W["recurrent"] else neat.nn.FeedForwardNetwork
        return fs.NetsPolicy([cls.create(g, _W["cfg"]) for g in payload])
    if _W.get("arch") == "elman":
        return fs.ElmanBatch(payload, _W["layers"])
    return fs.MLPBatch(payload, _W["layers"])


def task_eval(args):
    key, payload = args
    fs = _W["fs"]
    fm = get_fm(key)
    n = len(payload)
    r = fs.rollout(fm, make_policy(payload), n, max_steps=_W["max_steps"])
    f = fs.fit_geo(r, fm, _W["max_steps"], _W["w_prog"]) if _W["fitness"] == "geo" else fs.fit_euclid(r, fm, _W["max_steps"])
    return f, r.succ, r.col, r.steps


def task_score(args):
    key, payload = args
    fs = _W["fs"]
    fm = get_fm(key)
    r = fs.rollout(fm, make_policy(payload), len(payload), max_steps=_W["max_steps"], track_length=True)
    return r.succ, r.col, r.steps, r.length, fm.l_opt


def summarize(res):
    succ = np.concatenate([r[0] for r in res])
    col = np.concatenate([r[1] for r in res])
    steps = np.concatenate([r[2] for r in res])
    length = np.concatenate([r[3] for r in res])
    lopt = np.concatenate([np.full(len(r[0]), r[4]) for r in res])
    n = len(succ)
    spl = float(np.sum(succ * lopt / np.maximum(length, lopt)) / n) if n else 0.0
    return {"SR": round(float(succ.mean()), 4), "CR": round(float(col.mean()), 4), "timeout": round(float((~succ & ~col).mean()), 4),
            "SPL": round(spl, 4), "MTT": round(float(steps[succ].mean() * 0.1), 2) if succ.any() else None, "n": n}


class Trainer:
    def __init__(self, a, pool):
        self.a, self.pool = a, pool
        self.rng = np.random.default_rng(a.seed)
        self.level = 1
        self.gen = 0
        self.rows, self.promotions = [], []
        root = a.map_root
        glob = __import__("glob").glob
        ks = lambda rel: sorted(os.path.relpath(p, root).replace("\\", "/") for p in glob(os.path.join(root, rel, "*.json")))
        self.levels = [int(x) for x in a.levels.split(",")]
        self.level = self.levels[0]
        self.train = {L: ks(f"train/level_{L}") for L in self.levels}
        self.val = {L: ks(f"val/level_{L}") for L in self.levels}
        test_levels = sorted(int(os.path.basename(p).split("_")[1]) for p in glob(os.path.join(root, "test", "level_*")))
        self.tests = {f"test_L{L}": ks(f"test/level_{L}") for L in test_levels}
        for rel in ("test_barn", "test_lehman"):
            if ks(rel):
                self.tests[rel] = ks(rel)
        self.best_score, self.best_gen, self.best_payload, self.best_val = -1.0, None, None, None
        self.species_ref = None

    def sample_maps(self):
        a = self.a
        if a.schedule == "uniform":
            per = max(1, a.k // len(self.levels))
            return sum((list(self.rng.choice(self.train[L], per, replace=False)) for L in self.levels), [])
        L = self.level
        if L == 1:
            return list(self.rng.choice(self.train[1], a.k, replace=False))
        n_cur = a.k - a.k // 2
        lower = sum((self.train[l] for l in range(1, L)), [])
        return list(self.rng.choice(self.train[L], n_cur, replace=False)) + list(self.rng.choice(lower, a.k // 2, replace=False))

    def evaluate(self, payload):
        keys = self.sample_maps()
        res = self.pool.map(task_eval, [(k, payload) for k in keys], chunksize=1)
        raw = np.mean([r[0] for r in res], axis=0)
        solved = np.mean([r[1] for r in res], axis=0)
        steps = np.mean([r[3] for r in res], axis=0)
        return raw, solved, steps

    def score(self, one, keys):
        p = self.wrap(one)
        return summarize(self.pool.map(task_score, [(k, p) for k in keys], chunksize=4))

    def wrap(self, one):
        return [one] if self.a.algo == "neat" else one[None, :]

    def validate(self, cands, raw):
        if self.gen % self.a.val_every != 0:
            return None, False
        top = np.argsort(raw)[-5:][::-1]
        L = self.level
        srs = [self.score(cands[i], self.val[L])["SR"] for i in top]
        best = int(np.argmax(srs))
        sr = float(srs[best])
        one = cands[top[best]]
        promoted = False
        if self.a.schedule == "curriculum" and sr >= 0.8 and L < 4:
            self.level += 1
            promoted = True
            self.promotions.append(self.gen)
            if self.species_ref is not None:
                for s in self.species_ref.species.values():
                    s.last_improved = self.gen
                    s.fitness_history = []
        per = {str(l): self.score(one, self.val[l])["SR"] for l in self.levels}
        m = float(np.mean(list(per.values())))
        if m >= self.best_score:
            self.best_score, self.best_gen, self.best_payload, self.best_val = m, self.gen, copy.deepcopy(one), per
        return {"sr_level": sr, "per_level": per}, promoted

    def log(self, raw, solved, steps, t0, val, promoted, extra=None):
        row = {"gen": self.gen, "level": self.level, "sec": round(time.time() - t0, 2), "raw_best": round(float(raw.max()), 3),
               "raw_mean": round(float(raw.mean()), 3), "best_solved": round(float(solved[int(np.argmax(raw))]), 3),
               "mean_solved": round(float(solved.mean()), 3), "mean_steps": round(float(steps.mean()), 1), "val": val, "promoted": promoted}
        if extra:
            row.update(extra)
        self.rows.append(row)
        if self.gen % self.a.val_every == 0:
            print(json.dumps(row), flush=True)

    # ---------------------------------------------------- GA
    def run_ga(self):
        sys.path.insert(0, self.a.repo)
        if self.a.ga_module:
            # the corrected fixed_ga.py delivered to the team, loaded from its own file
            import importlib.util
            spec = importlib.util.spec_from_file_location("fixed_ga_delivered", self.a.ga_module)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            SimpleGA = mod.SimpleGA
        else:
            from src.baselines.fixed_ga import SimpleGA
        ga = SimpleGA(self.a.layer_list, pop_size=self.a.pop)
        for _ in range(self.a.gens):
            t0 = time.time()
            self.gen += 1
            raw, solved, steps = self.evaluate(ga.population.copy())
            ga.fitnesses = raw.copy()
            val, prom = self.validate(list(ga.population), raw)
            self.log(raw, solved, steps, t0, val, prom)
            ga.evolve()

    # ---------------------------------------------------- tuned GA (truncation selection, Gaussian mutation of every weight)
    def run_ga2(self):
        a = self.a
        n = sum(a.layer_list[i] * a.layer_list[i + 1] + a.layer_list[i + 1] for i in range(len(a.layer_list) - 1))
        rng = np.random.default_rng(a.seed + 7)
        pop = rng.normal(0.0, a.ga_init, (a.pop, n))
        n_par = max(2, int(round(a.ga_trunc * a.pop)))
        for _ in range(a.gens):
            t0 = time.time()
            self.gen += 1
            raw, solved, steps = self.evaluate(pop.copy())
            val, prom = self.validate(list(pop), raw)
            self.log(raw, solved, steps, t0, val, prom)
            order = np.argsort(-raw)
            parents = pop[order[:n_par]]
            new = [pop[order[i]].copy() for i in range(a.ga_elite)]
            while len(new) < a.pop:
                child = parents[rng.integers(n_par)].copy()
                if a.ga_xover:
                    other = parents[rng.integers(n_par)]
                    mask = rng.random(n) < 0.5
                    child = np.where(mask, child, other)
                child += rng.normal(0.0, a.ga_sigma, n)
                new.append(child)
            pop = np.array(new)

    # ---------------------------------------------------- CMA-ES
    def run_cma(self):
        import cma
        n = sum(self.a.layer_list[i] * self.a.layer_list[i + 1] + self.a.layer_list[i + 1] for i in range(len(self.a.layer_list) - 1))
        if self.a.arch == "elman":
            ni, nh, no = self.a.layer_list
            n = ni * nh + nh * nh + nh + nh * no + no
        es = cma.CMAEvolutionStrategy(np.zeros(n), self.a.sigma0, {"popsize": self.a.pop, "seed": self.a.seed + 1, "verbose": -9, "CMA_elitist": False})
        for _ in range(self.a.gens):
            t0 = time.time()
            self.gen += 1
            xs = np.array(es.ask())
            raw, solved, steps = self.evaluate(xs)
            es.tell(list(xs), list(-raw))
            val, prom = self.validate(list(xs), raw)
            self.log(raw, solved, steps, t0, val, prom, {"sigma": round(float(es.sigma), 4)})

    # ---------------------------------------------------- NEAT
    def neat_eval(self, genomes, config):
        t0 = time.time()
        self.gen += 1
        gs = [g for _, g in genomes]
        raw, solved, steps = self.evaluate(gs)
        for g, f in zip(gs, raw):
            g.fitness = float(f)
        val, prom = self.validate(gs, raw)
        best = gs[int(np.argmax(raw))]
        n_sp = len(self.species_ref.species)
        extra = {"n_species": n_sp, "best_nodes": len(best.nodes), "thr": round(config.species_set_config.compatibility_threshold, 2)}
        if self.a.target_species and not self.a.fixes:
            # rtNEAT-style control of the number of species (neat-python 2.0.0 has no target_num_species)
            sc = config.species_set_config
            thr = sc.compatibility_threshold
            thr = thr - 0.1 if n_sp < self.a.target_species else (thr + 0.1 if n_sp > self.a.target_species else thr)
            sc.compatibility_threshold = max(0.3, thr)
            extra["thr"] = round(sc.compatibility_threshold, 2)
        self.log(raw, solved, steps, t0, val, prom, extra)


def patch_ini(src, dst, a):
    txt = open(src, encoding="utf-8").read()
    subs = {"no_fitness_termination": "True", "pop_size": str(a.pop), "num_inputs": str(a.n_rays + 3)}
    if a.thr is not None:
        subs["compatibility_threshold"] = str(a.thr)
    if a.survival is not None:
        subs["survival_threshold"] = str(a.survival)
    if a.replace is not None:
        subs["weight_replace_rate"] = str(a.replace)
        subs["bias_replace_rate"] = str(a.replace)
    if a.node_add is not None:
        subs["node_add_prob"] = str(a.node_add)
    if a.conn_add is not None:
        subs["conn_add_prob"] = str(a.conn_add)
    subs["feed_forward"] = "False" if a.net == "rnn" else "True"
    if a.num_hidden is not None:
        subs["num_hidden"] = str(a.num_hidden)
    if a.init_conn is not None:
        subs["initial_connection"] = a.init_conn
    if a.weight_power is not None:
        subs["weight_mutate_power"] = str(a.weight_power)
        subs["bias_mutate_power"] = str(a.weight_power)
    for k, v in subs.items():
        # replace the whole value: some values have two tokens ("partial_direct 0.5")
        txt = re.sub(rf"(?m)^(\s*{k}\s*=).*$", lambda m, v=v: f"{m.group(1)} {v}", txt)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(txt)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--map-root", default=None, help="folder containing train/val/test maps; defaults to repo/data/maps")
    ap.add_argument("--geo-dir", required=True)
    ap.add_argument("--algo", choices=["ga", "ga2", "cma", "neat"], default="ga")
    ap.add_argument("--ga-sigma", type=float, default=0.05)
    ap.add_argument("--ga-trunc", type=float, default=0.2)
    ap.add_argument("--ga-elite", type=int, default=1)
    ap.add_argument("--ga-init", type=float, default=0.5)
    ap.add_argument("--ga-xover", action="store_true")
    ap.add_argument("--target-species", type=int, default=0)
    ap.add_argument("--fixes", default=None, help="path of the delivered neat_fixes.py (hidden-key fix and species control)")
    ap.add_argument("--ga-module", default=None, help="path of a fixed_ga.py to use for --algo ga instead of the repo's")
    ap.add_argument("--fitness", choices=["euclid", "geo"], default="geo")
    ap.add_argument("--schedule", choices=["curriculum", "uniform"], default="curriculum")
    ap.add_argument("--net", choices=["rnn", "ff"], default="rnn")
    ap.add_argument("--arch", choices=["mlp", "elman"], default="mlp", help="fixed-topology network used by ga/cma")
    ap.add_argument("--layers", default="8")
    ap.add_argument("--levels", default="1,2,3,4", help="training levels: folders train/level_N and val/level_N under --map-root")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--pop", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-rays", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--w-prog", type=float, default=10.0)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--sigma0", type=float, default=0.5)
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--survival", type=float, default=None)
    ap.add_argument("--replace", type=float, default=None)
    ap.add_argument("--node-add", type=float, default=None)
    ap.add_argument("--conn-add", type=float, default=None)
    ap.add_argument("--num-hidden", type=int, default=None)
    ap.add_argument("--init-conn", default=None)
    ap.add_argument("--weight-power", type=float, default=None)
    ap.add_argument("--batch-neat", action="store_true", help="evaluate a NEAT population as one vector (same outputs as neat-python, faster)")
    ap.add_argument("--base-ini", default=None, help="NEAT config to start from instead of the team's")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.layer_list = [a.n_rays + 3] + [int(x) for x in a.layers.split(",")] + [2]
    a.map_root = a.map_root or os.path.join(a.repo, "data", "maps")
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    sys.path.insert(0, a.repo)
    recurrent = a.net == "rnn"
    cfg_path = None
    if a.algo == "neat":
        src = a.base_ini or os.path.join(a.repo, "configs", "neat_recurrent.ini" if recurrent else "neat_feedforward.ini")
        cfg_path = patch_ini(src, os.path.join(a.out, "patched.ini"), a)
    wcfg = dict(repo=a.repo, map_root=a.map_root, geo_dir=a.geo_dir, algo=a.algo, fitness=a.fitness, recurrent=recurrent,
                layers=a.layer_list, n_rays=a.n_rays, max_steps=a.max_steps, w_prog=a.w_prog, cfg_path=cfg_path, arch=a.arch, batch_neat=a.batch_neat)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=init_worker, initargs=(wcfg,)) as pool:
        tr = Trainer(a, pool)
        final = None
        if a.algo == "ga":
            tr.run_ga()
        elif a.algo == "ga2":
            tr.run_ga2()
        elif a.algo == "cma":
            tr.run_cma()
        else:
            import neat
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, cfg_path)
            pop = neat.Population(config, seed=a.seed)
            if a.fixes:
                # the neat_fixes.py delivered to the team, loaded from its own file
                import importlib.util
                spec = importlib.util.spec_from_file_location("neat_fixes_delivered", a.fixes)
                nf = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(nf)
                nf.unify_initial_hidden_keys(pop, config)
                if a.target_species:
                    pop.add_reporter(nf.TargetSpeciesReporter(config, target=a.target_species))
            elif a.num_hidden:
                # neat-python gives every initial genome its own hidden-node keys, so all genomes look disjoint and fall into separate species.
                # Give them the same keys, then speciate again.
                from itertools import count
                gc = config.genome_config
                n_out = gc.num_outputs
                for g in pop.population.values():
                    mapping = {old: n_out + i for i, old in enumerate(sorted(k for k in g.nodes if k >= n_out))}
                    nodes, conns = {}, {}
                    for k, ng in g.nodes.items():
                        ng.key = mapping.get(k, k)
                        nodes[ng.key] = ng
                    for (x, y), cg in g.connections.items():
                        cg.key = (mapping.get(x, x), mapping.get(y, y))
                        conns[cg.key] = cg
                    g.nodes, g.connections = nodes, conns
                gc.node_indexer = count(n_out + a.num_hidden)
                pop.species = config.species_set_type(config.species_set_config, pop.reporters)
                pop.species.speciate(config, pop.population, pop.generation)
            tr.species_ref = pop.species
            final = pop.run(tr.neat_eval, n=a.gens)
        wall = time.time() - t0
        test = {k: tr.score(tr.best_payload, v) for k, v in tr.tests.items()}
    total_steps = sum(r["mean_steps"] for r in tr.rows) * a.pop * a.k
    s = {"algo": a.algo, "arch": a.arch, "fitness": a.fitness, "schedule": a.schedule, "levels": a.levels, "net": a.net, "layers": a.layers, "k": a.k, "gens": a.gens, "pop": a.pop,
         "seed": a.seed, "n_rays": a.n_rays, "max_steps": a.max_steps, "train_evals": a.pop * a.k * a.gens, "wall_s": round(wall, 1),
         "approx_steps_M": round(total_steps / 1e6, 1), "final_level": tr.level, "promotions": tr.promotions,
         "champion_gen": tr.best_gen, "champion_val_mean": tr.best_score, "champion_val": tr.best_val, "test": test}
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1)
    with open(os.path.join(a.out, "gens.json"), "w", encoding="utf-8") as f:
        json.dump(tr.rows, f)
    with open(os.path.join(a.out, "champion.pkl"), "wb") as f:
        pickle.dump(tr.best_payload, f)
    print(json.dumps({k: v for k, v in s.items() if k != "champion_val"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
