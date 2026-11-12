"""Baseline 2: independent re-implementation of GayaLahouazi/rl-robot-navigation (MIT licence).

replay : run the repo's stored CMA-ES weights and its potential-field controller on the 200 held-out maps, compare with results.json.
train  : train CMA-ES, the team's GA and NEAT on the same random circle maps with the same budget, test on the same 200 maps.

Usage:
  python -I b02_lahouazi.py replay <lahouazi_repo> <out_json>
  python -I b02_lahouazi.py train <team_repo> <tests2_dir> <out_json> --algo cma|ga|neat [--hidden 32] [--steps 3000000] [--k 4] [--seed 0]
"""
import argparse
import json
import math
import os
import re
import sys
import tempfile
import time

import numpy as np

ARENA, ROBOT_RADIUS, GOAL_RADIUS, N_RAYS, MAX_RANGE, DT, V_MAX, W_MAX = 10.0, 0.2, 0.4, 16, 3.0, 0.1, 1.0, 2.5
ANG = np.linspace(-np.pi, np.pi, N_RAYS, endpoint=False)
EVAL_SEED0 = 10_000_000
SCENARIOS = {"standard (4-8 obstacles)": dict(n_obstacles=(4, 8), sensor_noise=0.0),
             "dense (10-14 obstacles)": dict(n_obstacles=(10, 14), sensor_noise=0.0),
             "noisy lidar (sigma = 10 cm)": dict(n_obstacles=(4, 8), sensor_noise=0.10)}


def gen_rng(seed):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


class Map:
    """Same random draws, in the same order, as NavEnv._random_map."""
    def __init__(self, seed, n_obstacles=(4, 8)):
        self.rng = rng = gen_rng(seed)
        lo, hi = n_obstacles
        k = rng.integers(lo, hi + 1)
        self.c = rng.uniform(1.0, ARENA - 1.0, size=(k, 2))
        self.r = rng.uniform(0.4, 1.0, size=k)
        self.pos0 = self._free()
        for _ in range(100):
            self.goal = self._free()
            if np.linalg.norm(self.goal - self.pos0) > 4.0:
                break
        self.th0 = rng.uniform(-np.pi, np.pi)

    def _free(self):
        for _ in range(1000):
            p = self.rng.uniform(0.7, ARENA - 0.7, size=2)
            if np.all(np.linalg.norm(self.c - p, axis=1) > self.r + 0.6):
                return p
        raise RuntimeError("could not place a free point")

    def lidar(self, xy, th):
        ang = th[:, None] + ANG[None, :]
        dx, dy = np.cos(ang), np.sin(ang)
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(dx > 0, (ARENA - xy[:, 0, None]) / dx, -xy[:, 0, None] / dx)
            ty = np.where(dy > 0, (ARENA - xy[:, 1, None]) / dy, -xy[:, 1, None] / dy)
        dist = np.minimum(np.abs(tx), np.abs(ty))
        ox = xy[:, 0, None] - self.c[None, :, 0]
        oy = xy[:, 1, None] - self.c[None, :, 1]      # (n, K)
        b = dx[:, :, None] * ox[:, None, :] + dy[:, :, None] * oy[:, None, :]   # (n, R, K)
        c = (ox * ox + oy * oy - self.r ** 2)[:, None, :]
        disc = b * b - c
        t = -b - np.sqrt(np.maximum(disc, 0.0))
        t = np.where((disc >= 0) & (t > 0), t, np.inf)
        return np.minimum(np.minimum(dist, t.min(axis=2)), MAX_RANGE)

    def obs(self, xy, th, v, noise=None):
        d = self.lidar(xy, th)
        if noise is not None:
            d = np.clip(d + noise, 0, MAX_RANGE)
        gx, gy = self.goal[0] - xy[:, 0], self.goal[1] - xy[:, 1]
        bearing = np.arctan2(gy, gx) - th
        return np.concatenate([d / MAX_RANGE, np.hypot(gx, gy)[:, None] / (ARENA * math.sqrt(2)), np.cos(bearing)[:, None], np.sin(bearing)[:, None], (v / V_MAX)[:, None]], axis=1).astype(np.float32)

    def clearance(self, xy):
        d_obs = (np.hypot(xy[:, 0, None] - self.c[None, :, 0], xy[:, 1, None] - self.c[None, :, 1]) - self.r[None, :]).min(axis=1)
        d_wall = np.minimum(np.minimum(xy[:, 0], xy[:, 1]), np.minimum(ARENA - xy[:, 0], ARENA - xy[:, 1]))
        return np.minimum(d_obs, d_wall) - ROBOT_RADIUS


def rollout(m, policy, n, max_steps=400, noise_sd=0.0):
    """n robots on one map. Returns (return, success, collision, steps)."""
    xy = np.repeat(m.pos0[None, :], n, axis=0)
    th, v = np.full(n, m.th0), np.zeros(n)
    dist = np.full(n, float(np.linalg.norm(m.goal - m.pos0)))
    ret, succ, col, st = np.zeros(n), np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, np.int32)
    idx = np.arange(n)
    policy.reset(n)

    def noise(k):
        return m.rng.normal(0, noise_sd, size=(k, N_RAYS)) if noise_sd > 0 else None
    obs = m.obs(xy, th, v, noise(n))
    for t in range(1, max_steps + 1):
        a = np.clip(np.asarray(policy.act(idx, obs), dtype=np.float64), -1, 1)
        v = (a[:, 0] + 1) / 2 * V_MAX
        th = (th + a[:, 1] * W_MAX * DT + np.pi) % (2 * np.pi) - np.pi
        xy = xy + (v * DT)[:, None] * np.stack([np.cos(th), np.sin(th)], axis=1)
        nd = np.hypot(m.goal[0] - xy[:, 0], m.goal[1] - xy[:, 1])
        r = 10.0 * (dist - nd) - 0.01
        dist = nd
        cl = m.clearance(xy)
        r -= np.where(cl < 0.3, 0.05 * (0.3 - cl) / 0.3, 0.0)
        hit = cl <= 0
        reach = (nd < GOAL_RADIUS) & ~hit
        ret[idx] += r - 10.0 * hit + 10.0 * reach
        done = hit | reach | (t >= max_steps)
        if done.any():
            d = idx[done]
            succ[d], col[d], st[d] = reach[done], hit[done], t
            keep = ~done
            idx, xy, th, v, dist = idx[keep], xy[keep], th[keep], v[keep], dist[keep]
            if idx.size == 0:
                break
        obs = m.obs(xy, th, v, noise(idx.size))
    return ret, succ, col, st


class One:
    """A single stateless controller."""
    def __init__(self, fn):
        self.fn = fn

    def reset(self, n):
        pass

    def act(self, idx, obs):
        return np.array([self.fn(obs[0])])


def potential_field(obs):
    k_goal, k_obs, safe = 1.5, 0.6, 0.35
    lidar = obs[:N_RAYS]
    bearing = np.arctan2(obs[N_RAYS + 2], obs[N_RAYS + 1])
    front = np.abs(ANG) < np.pi / 2
    push = np.where(front, (1.0 / np.maximum(lidar, 0.05) - 1.0), 0.0)
    turn = k_goal * bearing - k_obs * np.sum(push * np.sin(ANG)) / N_RAYS
    front_clear = lidar[np.abs(ANG) < np.pi / 6].min()
    speed = 1.0 if front_clear > safe else -0.6
    return np.clip(np.array([speed, turn]), -1, 1)


def mlp_act(p, hidden=32, in_dim=20):
    i = 0
    W1 = p[i:i + in_dim * hidden].reshape(in_dim, hidden); i += in_dim * hidden
    b1 = p[i:i + hidden]; i += hidden
    W2 = p[i:i + hidden * 2].reshape(hidden, 2); i += hidden * 2
    b2 = p[i:i + 2]
    return lambda o: np.tanh(np.tanh(o.astype(np.float64) @ W1 + b1) @ W2 + b2)


def test_policy(make_policy, episodes=200):
    out = {}
    for name, kw in SCENARIOS.items():
        s, c = 0, 0
        for i in range(episodes):
            m = Map(EVAL_SEED0 + i, kw["n_obstacles"])
            _, su, co, _ = rollout(m, make_policy(), 1, noise_sd=kw["sensor_noise"])
            s += int(su[0])
            c += int(co[0])
        out[name] = dict(success=s, collision=c, timeout=episodes - s - c, n=episodes)
    return out


def replay(a):
    pub = json.load(open(os.path.join(a.repo, "results", "results.json"), encoding="utf-8"))["test"]
    p = np.load(os.path.join(a.repo, "models", "cmaes.npy"), allow_pickle=False)
    res = {}
    for label, key, fn in (("potential_field", "Potential field (hand-designed)", potential_field), ("cmaes_stored_weights", "CMA-ES neuroevolution", mlp_act(p))):
        t0 = time.time()
        got = test_policy(lambda: One(fn))
        res[label] = {}
        for sc in SCENARIOS:
            res[label][sc] = dict(reproduced=got[sc]["success"], published=round(pub[key][sc]["success"] * 200), difference=got[sc]["success"] - round(pub[key][sc]["success"] * 200), collisions=got[sc]["collision"], published_collisions=round(pub[key][sc]["collision"] * 200))
        print(label, json.dumps(res[label]), f"{time.time() - t0:.0f}s", flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)


def train(a):
    sys.path.insert(0, a.tests2)
    sys.path.insert(0, a.team_repo)
    import fastsim as fs
    in_dim = N_RAYS + 4
    layers = [in_dim, a.hidden, 2]
    rng = np.random.default_rng(a.seed)
    budget = a.steps
    used, gen, t0 = 0, 0, time.time()
    log = []

    def eval_batch(policy, n):
        nonlocal used
        seeds = rng.integers(0, 1_000_000, size=a.k)
        f, sc = np.zeros(n), np.zeros(n)
        for s in seeds:
            ret, su, _, st = rollout(Map(int(s)), policy, n)
            f += ret
            sc += su
            used += int(st.sum())
        return f / a.k, sc / a.k

    if a.algo == "cma":
        import cma
        n = in_dim * a.hidden + a.hidden + a.hidden * 2 + 2
        es = cma.CMAEvolutionStrategy(np.zeros(n), 0.5, {"seed": a.seed + 1, "verbose": -9})
        while used < budget:
            gen += 1
            xs = np.array(es.ask())
            f, sc = eval_batch(fs.MLPBatch(xs, layers), len(xs))
            es.tell(list(xs), list(-f))
            if gen % 20 == 0:
                log.append(dict(gen=gen, steps=used, best=round(float(f.max()), 2), mean_sr=round(float(sc.mean()), 3)))
                print(json.dumps(log[-1]), flush=True)
        best = np.asarray(es.result.xfavorite)

        def make():
            return fs.MLPBatch(best[None, :], layers)
    elif a.algo == "ga":
        from src.baselines.fixed_ga import SimpleGA
        ga = SimpleGA(layers, pop_size=150)
        champion, champ_f = None, -1e9
        while used < budget:
            gen += 1
            xs = ga.population.copy()
            f, sc = eval_batch(fs.MLPBatch(xs, layers), len(xs))
            if f.max() > champ_f or gen % 10 == 0:
                champ_f, champion = float(f.max()), xs[int(np.argmax(f))].copy()
            ga.fitnesses = f.copy()
            ga.evolve()
            if gen % 10 == 0:
                log.append(dict(gen=gen, steps=used, best=round(float(f.max()), 2), mean_sr=round(float(sc.mean()), 3)))
                print(json.dumps(log[-1]), flush=True)

        def make():
            return fs.MLPBatch(champion[None, :], layers)
    else:
        import neat
        if a.neat_ini:
            txt = open(a.neat_ini, encoding="utf-8", errors="ignore").read()
            subs = dict(num_inputs=str(in_dim))
        else:
            txt = open(os.path.join(a.team_repo, "configs", "neat_recurrent.ini"), encoding="utf-8", errors="ignore").read()
            subs = dict(no_fitness_termination="True", pop_size="150", num_inputs=str(in_dim), compatibility_threshold="2.0", survival_threshold="0.5", weight_replace_rate="0.02", bias_replace_rate="0.02")
        for k, v in subs.items():
            txt = re.sub(rf"(?m)^{k}\s*=.*$", f"{k} = {v}", txt)
        p = os.path.join(tempfile.mkdtemp(), "cfg.ini")
        open(p, "w", encoding="utf-8").write(txt)
        cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, p)
        pop = neat.Population(cfg, seed=a.seed)
        if cfg.genome_config.num_hidden:
            from itertools import count as _count
            gc = cfg.genome_config
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
            gc.node_indexer = _count(n_out + gc.num_hidden)
            pop.species = cfg.species_set_type(cfg.species_set_config, pop.reporters)
            pop.species.speciate(cfg, pop.population, pop.generation)
        holder = dict(g=None, f=-1e9)

        def fit(genomes, config):
            nonlocal gen
            gen += 1
            gs = [g for _, g in genomes]
            f, sc = eval_batch(fs.NetsPolicy([neat.nn.RecurrentNetwork.create(g, config) for g in gs]), len(gs))
            for g, x in zip(gs, f):
                g.fitness = float(x)
            i = int(np.argmax(f))
            if f[i] > holder["f"] or gen % 10 == 0:
                import copy
                holder.update(g=copy.deepcopy(gs[i]), f=float(f[i]))
            if gen % 10 == 0:
                log.append(dict(gen=gen, steps=used, best=round(float(f.max()), 2), mean_sr=round(float(sc.mean()), 3), species=len(pop.species.species), nodes=len(gs[i].nodes)))
                print(json.dumps(log[-1]), flush=True)
            if used >= budget:
                raise StopIteration
        try:
            pop.run(fit, 100000)
        except StopIteration:
            pass

        class P:
            def __init__(self):
                self.p = fs.NetsPolicy([neat.nn.RecurrentNetwork.create(holder["g"], cfg)])

            def reset(self, n):
                self.p.reset(n)

            def act(self, idx, obs):
                return self.p.act(idx, obs)
        make = P
    wall = time.time() - t0
    test = test_policy(make)
    out = dict(algo=a.algo, hidden=a.hidden, k=a.k, seed=a.seed, steps=used, generations=gen, wall_s=round(wall, 1), test=test, log=log)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "log"}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("repo")
    r.add_argument("out")
    t = sub.add_parser("train")
    t.add_argument("team_repo")
    t.add_argument("tests2")
    t.add_argument("out")
    t.add_argument("--algo", choices=["cma", "ga", "neat"], required=True)
    t.add_argument("--hidden", type=int, default=32)
    t.add_argument("--steps", type=int, default=3_000_000)
    t.add_argument("--k", type=int, default=4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--neat-ini", default=None, help="NEAT config to use instead of the team's")
    args = ap.parse_args()
    (replay if args.cmd == "replay" else train)(args)
