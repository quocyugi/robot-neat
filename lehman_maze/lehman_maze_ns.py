"""Baseline: the Lehman-Stanley maze, re-implemented from the original C++ (jal278/novelty-search-cpp) and evolved with neat-python 2.0.0.

The simulator follows maze.h line by line (radar overwrites the rear rangefinder input, as in the original; 400 steps; solved when the robot
comes within 5 units; a collision just stops the robot). Evolution is generational neat-python, not rtNEAT.
Published reference: hard maze NS 39/40 (ECJ 2011), 93/100 (Gravina 2017), 10/10 (goNEAT_NS); fitness 3/40, 0/10; medium NS 40/40.

Usage: python -I lehman_maze/lehman_maze_ns.py run --maze-dir D --maze hard --method ns --config ref --seed 0 --out F [--team-cfg C] [--pop 250] [--gens 300]
       python -I lehman_maze/lehman_maze_ns.py check          (compares BatchRecurrent with neat.nn.RecurrentNetwork)
"""
import argparse
import json
import os
import re
import tempfile
import time
from itertools import count

import numpy as np

import neat
from neat.graphs import required_for_output

REF_CFG = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 1e9
pop_size              = {pop}
reset_on_extinction   = True
no_fitness_termination = True

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
compatibility_disjoint_coefficient = 1.1
compatibility_weight_coefficient   = 0.5
conn_add_prob           = 0.5
conn_delete_prob        = 0.1
enabled_default         = True
enabled_mutate_rate     = 0.01
feed_forward            = False
initial_connection      = partial_direct 0.5
node_add_prob           = 0.1
node_delete_prob        = 0.1
num_hidden              = 1
num_inputs              = 10
num_outputs             = 2
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
max_stagnation       = 100
species_elitism      = 1

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.1
min_species_size   = 2
"""


class BatchRecurrent:
    """All networks of a population in one vector; same arithmetic as neat.nn.RecurrentNetwork (sum aggregation, one activation function)."""
    def __init__(self, genomes, config):
        gc = config.genome_config
        keys_in, keys_out = gc.input_keys, gc.output_keys
        n_in, n_out = len(keys_in), len(keys_out)
        self.N, self.n_in = len(genomes), n_in
        src, dst, w, nodes, bias, resp = [], [], [], [], [], []
        in_idx = np.zeros((self.N, n_in), np.int64)
        out_idx = np.zeros((self.N, n_out), np.int64)
        offset = 0
        for gi, g in enumerate(genomes):
            required = required_for_output(keys_in, keys_out, g.connections)
            idx = {k: offset + i for i, k in enumerate(keys_in)}
            nxt = offset + n_in

            def get(k):
                nonlocal nxt
                if k not in idx:
                    idx[k] = nxt
                    nxt += 1
                return idx[k]
            per = {}
            for cg in g.connections.values():
                if not cg.enabled:
                    continue
                i, o = cg.key
                if o not in required and i not in required:
                    continue
                per.setdefault(o, []).append((i, cg.weight))
            for k in keys_out:
                get(k)
            for key, inputs in per.items():
                nd = g.nodes[key]
                d = get(key)
                nodes.append(d)
                bias.append(nd.bias)
                resp.append(nd.response)
                for i, wt in inputs:
                    src.append(get(i))
                    dst.append(d)
                    w.append(wt)
            in_idx[gi] = [idx[k] for k in keys_in]
            out_idx[gi] = [idx[k] for k in keys_out]
            offset = nxt
        self.K = offset
        self.src, self.dst, self.w = np.array(src, np.int64), np.array(dst, np.int64), np.array(w, np.float64)
        self.nodes, self.bias, self.resp = np.array(nodes, np.int64), np.array(bias, np.float64), np.array(resp, np.float64)
        self.in_idx, self.out_idx = in_idx, out_idx
        self.kind = next(iter(next(iter(genomes)).nodes.values())).activation
        self.v = np.zeros(self.K)

    def reset(self):
        self.v[:] = 0.0

    def activate(self, inputs):
        v = self.v
        v[self.in_idx] = inputs
        s = np.bincount(self.dst, weights=self.w * v[self.src], minlength=self.K)[self.nodes] if self.src.size else np.zeros(self.nodes.size)
        z = self.bias + self.resp * s
        if self.kind == "sigmoid":
            v[self.nodes] = 1.0 / (1.0 + np.exp(-np.clip(5.0 * z, -60.0, 60.0)))
        else:
            v[self.nodes] = np.tanh(np.clip(2.5 * z, -60.0, 60.0))
        return v[self.out_idx]


class Maze:
    def __init__(self, path):
        t = [float(x) for x in open(path, encoding="utf-8").read().split()]
        n = int(t[0])
        self.start, self.h0, self.goal = np.array(t[1:3]), t[3], np.array(t[4:6])
        L = np.array(t[6:6 + 4 * n]).reshape(n, 4)
        self.A, self.B = L[:, :2], L[:, 2:]
        self.rf = np.deg2rad([-90.0, -45.0, 0.0, 45.0, 90.0, -180.0])

    def rangefinders(self, x, y, h):
        ang = np.deg2rad(h)[:, None] + self.rf[None, :]
        dx, dy = 100.0 * np.cos(ang), 100.0 * np.sin(ang)        # (N, 6)
        Ax, Ay = self.A[None, None, :, 0], self.A[None, None, :, 1]
        Bx, By = self.B[None, None, :, 0], self.B[None, None, :, 1]
        Cx, Cy = x[:, None, None], y[:, None, None]
        Dx, Dy = dx[:, :, None], dy[:, :, None]          # D - C
        bot = (Bx - Ax) * Dy - (By - Ay) * Dx
        with np.errstate(divide="ignore", invalid="ignore"):
            r = ((Ay - Cy) * Dx - (Ax - Cx) * Dy) / bot
            s = ((Ay - Cy) * (Bx - Ax) - (Ax - Cx) * (By - Ay)) / bot
        ok = (bot != 0) & (r > 0) & (r < 1) & (s > 0) & (s < 1)
        return np.minimum(100.0, np.where(ok, s * 100.0, np.inf).min(axis=2))

    def radar(self, x, y, h):
        dx, dy = self.goal[0] - x, self.goal[1] - y
        a = np.deg2rad(-h)
        rx = np.cos(a) * dx - np.sin(a) * dy
        ry = np.sin(a) * dx + np.cos(a) * dy
        with np.errstate(divide="ignore", invalid="ignore"):
            ang = np.degrees(np.arctan(ry / rx))
        ang = np.where(rx > 0, ang, ang + 180.0)
        ang = np.where(rx == 0, np.where(ry > 0, 90.0, 270.0), ang)
        out = np.zeros((x.size, 4))
        for i, (lo, hi) in enumerate(((315.0, 405.0), (45.0, 135.0), (135.0, 225.0), (225.0, 315.0))):
            out[:, i] = ((ang >= lo) & (ang < hi)) | ((ang + 360.0 >= lo) & (ang + 360.0 < hi))
        return out

    def inputs(self, x, y, h):
        rf = self.rangefinders(x, y, h) / 100.0
        return np.concatenate([np.ones((x.size, 1)), rf[:, :5], self.radar(x, y, h)], axis=1)

    def collide(self, x, y, rad=8.0):
        ax, ay = self.A[None, :, 0], self.A[None, :, 1]
        sx, sy = (self.B - self.A)[None, :, 0], (self.B - self.A)[None, :, 1]
        px, py = x[:, None] - ax, y[:, None] - ay
        u = (px * sx + py * sy) / (sx * sx + sy * sy)
        d = np.where((u < 0) | (u > 1), np.minimum(np.hypot(px, py), np.hypot(x[:, None] - ax - sx, y[:, None] - ay - sy)), np.hypot(px - u * sx, py - u * sy))
        return (d < rad).any(axis=1)

    def simulate(self, net, n, steps=400):
        x, y = np.full(n, self.start[0]), np.full(n, self.start[1])
        h, spd, ang = np.full(n, self.h0), np.zeros(n), np.zeros(n)
        reach = np.zeros(n, bool)
        net.reset()
        inp = self.inputs(x, y, h)
        for _ in range(10):
            net.activate(inp)
        for _ in range(steps):
            o = net.activate(self.inputs(x, y, h))
            ang = np.clip(ang + (o[:, 0] - 0.5), -3.0, 3.0)
            spd = np.clip(spd + (o[:, 1] - 0.5), -3.0, 3.0)
            act = ~reach
            r = np.deg2rad(h)
            nx, ny = x + np.cos(r) * spd, y + np.sin(r) * spd
            hn = h + ang
            hn = np.where(hn > 360, hn - 360, hn)
            hn = np.where(hn < 0, hn + 360, hn)
            h = np.where(act, hn, h)
            move = act & ~self.collide(nx, ny)
            x, y = np.where(move, nx, x), np.where(move, ny, y)
            reach |= np.hypot(x - self.goal[0], y - self.goal[1]) < 5.0
        return x, y, reach, np.hypot(x - self.goal[0], y - self.goal[1])


class Archive:
    def __init__(self):
        self.pts = np.zeros((0, 2))
        self.thr, self.floor, self.timeout = 6.0, 0.25, 0

    def score(self, xy, k=15):
        ref = np.vstack([xy, self.pts])
        d = np.sqrt(((xy[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(d[:, :xy.shape[0]], np.inf)
        k = min(k, ref.shape[0] - 1)
        return np.partition(d, k - 1, axis=1)[:, :k].mean(axis=1)

    def update(self, xy, nov):
        new = xy[nov > self.thr]
        if len(self.pts) == 0 and len(new) == 0:
            new = xy[[int(np.argmax(nov))]]
        self.pts = np.vstack([self.pts, new])
        self.timeout = self.timeout + 1 if len(new) == 0 else 0
        if self.timeout == 10:
            self.thr = max(self.thr * 0.95, self.floor)
            self.timeout = 0
        if len(new) > 4:
            self.thr *= 1.2


class Done(Exception):
    pass


def make_config(a):
    p = os.path.join(tempfile.mkdtemp(), "cfg.ini")
    if a.config == "ref":
        txt = REF_CFG.format(pop=a.pop)
    elif a.config == "file":
        # a config file used as it is; only the task's inputs, outputs, population and stopping rule are set
        txt = open(a.team_cfg, encoding="utf-8", errors="ignore").read()
        subs = dict(num_inputs="10", num_outputs="2", pop_size=str(a.pop), no_fitness_termination="True", fitness_threshold="1000000",
                    activation_default="sigmoid", activation_options="sigmoid")
        for k, v in subs.items():
            txt = re.sub(rf"(?m)^{k}\s*=.*$", f"{k} = {v}", txt)
    else:
        txt = open(a.team_cfg, encoding="utf-8", errors="ignore").read()
        subs = dict(num_inputs="10", num_outputs="2", pop_size=str(a.pop), no_fitness_termination="True", fitness_threshold="1000000", compatibility_threshold="2.0",
                    survival_threshold="0.5", weight_replace_rate="0.02", bias_replace_rate="0.02", activation_default="sigmoid", activation_options="sigmoid")
        for k, v in subs.items():
            txt = re.sub(rf"(?m)^{k}\s*=.*$", f"{k} = {v}", txt)
    open(p, "w", encoding="utf-8").write(txt)
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, p)


def unify_hidden(pop, config, n_hidden):
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
    gc.node_indexer = count(n_out + n_hidden)
    pop.species = config.species_set_type(config.species_set_config, pop.reporters)
    pop.species.speciate(config, pop.population, pop.generation)


def run(a):
    maze = Maze(os.path.join(a.maze_dir, f"{a.maze}_maze.txt"))
    cfg = make_config(a)
    pop = neat.Population(cfg, seed=a.seed)
    if cfg.genome_config.num_hidden:
        unify_hidden(pop, cfg, cfg.genome_config.num_hidden)
    arch = Archive()
    st = dict(evals=0, solved_at=None, best_d=1e9, species=[])

    def fn(genomes, config):
        gs = [g for _, g in genomes]
        net = BatchRecurrent(gs, config)
        x, y, reach, d = maze.simulate(net, len(gs), a.steps)
        st["best_d"] = min(st["best_d"], float(d.min()))
        st["species"].append(len(pop.species.species))
        if reach.any():
            st["solved_at"] = st["evals"] + int(np.argmax(reach)) + 1
            st["evals"] += len(gs)
            raise Done
        st["evals"] += len(gs)
        if a.method == "ns":
            xy = np.stack([x, y], axis=1)
            nov = arch.score(xy)
            for g, v in zip(gs, nov):
                g.fitness = float(v)
            arch.update(xy, nov)
        else:
            for g, v in zip(gs, 300.0 - d):
                g.fitness = max(float(v), 0.1)
    t0 = time.time()
    try:
        pop.run(fn, a.gens)
    except Done:
        pass
    out = dict(maze=a.maze, method=a.method, config=a.config, seed=a.seed, pop=a.pop, gens=a.gens, steps=a.steps, solved=st["solved_at"] is not None, evals_to_solve=st["solved_at"], evals_used=st["evals"],
               best_distance=round(st["best_d"], 2), archive=int(len(arch.pts)), final_threshold=round(arch.thr, 2), species_median=float(np.median(st["species"])), wall_s=round(time.time() - t0, 1))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(json.dumps(out), flush=True)


def check():
    """BatchRecurrent must give the same outputs as the library's network, for random evolved genomes."""
    class A:
        config, pop, team_cfg = "ref", 60, None
    cfg = make_config(A)
    pop = neat.Population(cfg, seed=3)
    rng = np.random.default_rng(0)

    def fn(genomes, config):
        for _, g in genomes:
            g.fitness = float(rng.random())
    pop.run(fn, 25)
    gs = list(pop.population.values())
    nets = [neat.nn.RecurrentNetwork.create(g, cfg) for g in gs]
    b = BatchRecurrent(gs, cfg)
    worst = 0.0
    for _ in range(200):
        inp = rng.random((len(gs), 10))
        o2 = b.activate(inp)
        o1 = np.array([n.activate(list(i)) for n, i in zip(nets, inp)])
        worst = max(worst, float(np.abs(o1 - o2).max()))
    print(f"genomes={len(gs)} max nodes={max(len(g.nodes) for g in gs)} largest output difference over 200 steps: {worst:.3e}")
    # sensor sanity at the start pose
    return worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    r = sub.add_parser("run")
    r.add_argument("--maze-dir", required=True)
    r.add_argument("--maze", choices=["hard", "medium"], required=True)
    r.add_argument("--method", choices=["ns", "fit"], required=True)
    r.add_argument("--config", choices=["ref", "team", "file"], default="ref")
    r.add_argument("--team-cfg", default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--pop", type=int, default=250)
    r.add_argument("--gens", type=int, default=300)
    r.add_argument("--steps", type=int, default=400)
    r.add_argument("--out", required=True)
    args = ap.parse_args()
    check() if args.cmd == "check" else run(args)
