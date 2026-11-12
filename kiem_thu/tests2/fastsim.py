"""Vectorised re-implementation of RobotNavEnv.

One call of rollout() simulates N robots on ONE map at the same time. The physics, sensing,
termination rules and fitness are copied from the team's src/env/* and evaluator.py, with the
same float32/float64 arithmetic, so results match the original simulator (see d00_equivalence.py).
"""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np

ROBOT_RADIUS = 0.25
WHEEL_BASE = 0.40
V_MAX = 0.50
DT = 0.10
GOAL_RADIUS = 0.30
LIDAR_RANGE = 3.0
ARENA_DIAGONAL = float(np.sqrt(10.0 ** 2 + 10.0 ** 2))
GEO_RES = 0.05


class FastMap:
    def __init__(self, data: dict, key: str = "", n_rays: int = 12, geo: np.ndarray | None = None):
        self.key = key
        self.data = data
        segs = np.asarray(data["segments"], dtype=np.float64)
        self.Ax, self.Ay = segs[:, 0, 0], segs[:, 0, 1]
        self.Bx, self.By = segs[:, 1, 0], segs[:, 1, 1]
        self.sx, self.sy = self.Bx - self.Ax, self.By - self.Ay
        self.ss = np.maximum(self.sx ** 2 + self.sy ** 2, 1e-12)
        self.start = np.asarray(data["start"], dtype=np.float64)
        self.goal = np.asarray(data["goal"], dtype=np.float64)
        self.th0 = float(data.get("start_heading", 0.0))
        self.d0 = float(np.linalg.norm(self.start - self.goal))
        self.n_rays = n_rays
        self.ray_angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
        self.geo = geo
        self.g0 = None
        if geo is not None:
            self.g0 = float(self.geo_at(np.array([self.start[0]]), np.array([self.start[1]]))[0])
        m = data.get("metrics") or {}
        self.l_opt = float(m.get("optimal_path_length", self.d0))

    def geo_at(self, x, y):
        n = self.geo.shape[0]
        ix = np.clip((x / GEO_RES).astype(np.int32), 0, n - 1)
        iy = np.clip((y / GEO_RES).astype(np.int32), 0, n - 1)
        return self.geo[iy, ix]

    # ------------------------------------------------------------------ observation
    def proximity(self, x, y, th):
        """Same computation as FastLiDAR2D.cast_rays, for many robots. th may be float32."""
        na = x.shape[0]
        ang = th[:, None].astype(np.float64) + self.ray_angles[None, :]
        rx = np.cos(ang) * LIDAR_RANGE
        ry = np.sin(ang) * LIDAR_RANGE
        pax = self.Ax[None, :] - x[:, None]          # (na, M)
        pay = self.Ay[None, :] - y[:, None]
        sx, sy = self.sx[None, None, :], self.sy[None, None, :]
        rxe, rye = rx[:, :, None], ry[:, :, None]    # (na, R, 1)
        den = rxe * sy - rye * sx                    # (na, R, M)
        pa_s = (pax * self.sy[None, :] - pay * self.sx[None, :])[:, None, :]   # (na, 1, M)
        pa_r = pax[:, None, :] * rye - pay[:, None, :] * rxe                    # (na, R, M)
        par = np.abs(den) < 1e-8
        safe = np.where(par, 1e-8, den)
        t = pa_s / safe
        u = pa_r / safe
        valid = (~par) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
        return 1.0 - np.where(valid, t, 1.0).min(axis=2)

    def obs(self, x, y, th):
        prox = self.proximity(x, y, th).astype(np.float32)
        dx, dy = self.goal[0] - x, self.goal[1] - y
        rho = np.clip(np.sqrt(dx * dx + dy * dy) / ARENA_DIAGONAL, 0.0, 1.0)
        phi = np.arctan2(dy, dx) - th.astype(np.float64)
        return np.concatenate([prox, rho[:, None].astype(np.float32), np.sin(phi)[:, None].astype(np.float32),
                               np.cos(phi)[:, None].astype(np.float32)], axis=1)

    def collided(self, x, y):
        apx, apy = x[:, None] - self.Ax[None, :], y[:, None] - self.Ay[None, :]
        t = np.clip((apx * self.sx[None, :] + apy * self.sy[None, :]) / self.ss[None, :], 0.0, 1.0)
        dx, dy = apx - t * self.sx[None, :], apy - t * self.sy[None, :]
        return np.sqrt(dx * dx + dy * dy).min(axis=1) <= ROBOT_RADIUS


class Result:
    __slots__ = ("succ", "col", "steps", "spin", "dmin", "fx", "fy", "gmin", "length")


def rollout(fm: FastMap, policy, n: int, max_steps: int = 500, stop_on_collision: bool = True, track_length: bool = False) -> Result:
    """Run n robots on one map. policy.reset(n) then policy.act(idx, obs) -> float32 array (len(idx), 2)."""
    x = np.full(n, fm.start[0], dtype=np.float64)
    y = np.full(n, fm.start[1], dtype=np.float64)
    th = np.full(n, fm.th0, dtype=np.float32)
    r = Result()
    r.succ = np.zeros(n, bool)
    r.col = np.zeros(n, bool)
    r.steps = np.zeros(n, np.int32)
    r.spin = np.zeros(n, np.int32)
    r.dmin = np.full(n, fm.d0, np.float64)
    r.fx, r.fy = x.copy(), y.copy()
    r.gmin = np.full(n, fm.g0 if fm.g0 is not None else 0.0, np.float64)
    r.length = np.zeros(n, np.float64)
    idx = np.arange(n)
    policy.reset(n)
    # first observation uses the start heading in float64, like env.reset()
    obs = fm.obs(x, y, np.full(n, fm.th0, np.float64))
    for t in range(1, max_steps + 1):
        act = policy.act(idx, obs)
        o_l = np.clip(act[:, 0], -1.0, 1.0)
        o_r = np.clip(act[:, 1], -1.0, 1.0)
        v_l, v_r = V_MAX * o_l, V_MAX * o_r
        r.spin[idx] += (np.abs(v_l + v_r) < 0.05) & (np.abs(v_l - v_r) > 0.20)
        om = (v_r - v_l) / WHEEL_BASE
        v = (v_r + v_l) / 2.0
        th_new = th + om * DT
        nx = x + v * np.cos(th_new) * DT
        ny = y + v * np.sin(th_new) * DT
        if track_length:
            r.length[idx] += np.hypot(nx - x, ny - y)
        col = fm.collided(nx, ny)
        if not stop_on_collision and col.any():
            # the robot keeps its previous position, heading still changes (as in the original maze code)
            nx = np.where(col, x, nx)
            ny = np.where(col, y, ny)
        x, y, th = nx, ny, th_new
        dg = np.sqrt((x - fm.goal[0]) ** 2 + (y - fm.goal[1]) ** 2)
        r.dmin[idx] = np.minimum(r.dmin[idx], dg)
        if fm.geo is not None:
            r.gmin[idx] = np.minimum(r.gmin[idx], fm.geo_at(x, y))
        reached = dg <= GOAL_RADIUS
        col_flag = col if stop_on_collision else np.zeros_like(col)
        done = reached | col_flag | (t >= max_steps)
        if done.any():
            d = idx[done]
            r.succ[d] = reached[done]
            r.col[d] = col_flag[done]
            r.steps[d] = t
            r.fx[d], r.fy[d] = x[done], y[done]
            keep = ~done
            idx, x, y, th = idx[keep], x[keep], y[keep], th[keep]
            if idx.size == 0:
                break
        obs = fm.obs(x, y, th)
    return r


# ---------------------------------------------------------------------- fitness
def fit_euclid(r: Result, fm: FastMap, max_steps: int = 500) -> np.ndarray:
    """The team's 5-term fitness (evaluator.compute_episode_fitness)."""
    raw = np.maximum(0.0, (fm.d0 - r.dmin) / max(fm.d0, 1e-5))
    prog = np.where(r.col, 0.5 * raw, raw)
    f = prog + 10.0 * r.succ + 5.0 * (1.0 - r.steps / max_steps) * r.succ - 2.0 * r.col - r.spin / max_steps
    return np.maximum(f + 3.0, 0.01)


def fit_geo(r: Result, fm: FastMap, max_steps: int = 500, w_prog: float = 10.0) -> np.ndarray:
    """Same fitness, but progress is measured along the shortest path (geodesic) and weighted more."""
    raw = np.clip((fm.g0 - r.gmin) / max(fm.g0, 1e-5), 0.0, 1.0)
    f = w_prog * raw + 10.0 * r.succ + 5.0 * (1.0 - r.steps / max_steps) * r.succ - 2.0 * r.col - r.spin / max_steps
    return np.maximum(f + 3.0, 0.01)


# ---------------------------------------------------------------------- policies
class MLPBatch:
    """Fixed-topology tanh network (same forward pass as baselines.fixed_ga.FixedMLP) for a batch of genomes."""
    def __init__(self, params: np.ndarray, layers: list[int]):
        self.Ws, self.bs = [], []
        off = 0
        P = params.shape[0]
        for i in range(len(layers) - 1):
            ni, no = layers[i], layers[i + 1]
            self.Ws.append(params[:, off:off + ni * no].reshape(P, ni, no))
            off += ni * no
            self.bs.append(params[:, off:off + no])
            off += no

    def reset(self, n):
        pass

    def act(self, idx, obs):
        x = obs.astype(np.float64)
        for W, b in zip(self.Ws, self.bs):
            x = np.tanh(np.matmul(x[:, None, :], W[idx])[:, 0, :] + b[idx])
        return x.astype(np.float32)


class ElmanBatch:
    """One recurrent hidden layer (Elman): h = tanh(obs*Wi + h*Wh + bh), out = tanh(h*Wo + bo)."""
    @staticmethod
    def size(n_in, n_h, n_out):
        return n_in * n_h + n_h * n_h + n_h + n_h * n_out + n_out

    def __init__(self, params: np.ndarray, layers: list[int]):
        ni, nh, no = layers
        P = params.shape[0]
        off = 0
        self.Wi = params[:, off:off + ni * nh].reshape(P, ni, nh); off += ni * nh
        self.Wh = params[:, off:off + nh * nh].reshape(P, nh, nh); off += nh * nh
        self.bh = params[:, off:off + nh]; off += nh
        self.Wo = params[:, off:off + nh * no].reshape(P, nh, no); off += nh * no
        self.bo = params[:, off:off + no]
        self.nh = nh

    def reset(self, n):
        self.h = np.zeros((n, self.nh), np.float64)

    def act(self, idx, obs):
        x = obs.astype(np.float64)
        h = np.tanh(np.matmul(x[:, None, :], self.Wi[idx])[:, 0, :] + np.matmul(self.h[idx][:, None, :], self.Wh[idx])[:, 0, :] + self.bh[idx])
        self.h[idx] = h
        return np.tanh(np.matmul(h[:, None, :], self.Wo[idx])[:, 0, :] + self.bo[idx]).astype(np.float32)


class BatchRecurrentPolicy:
    """A whole NEAT population in one vector: same outputs as neat.nn.RecurrentNetwork, much faster (tanh or sigmoid, sum aggregation)."""
    def __init__(self, genomes, config):
        from neat.graphs import required_for_output
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
        self.kind = next(iter(genomes[0].nodes.values())).activation
        self.v = np.zeros(self.K)

    def reset(self, n):
        self.v[:] = 0.0

    def act(self, idx, obs):
        v = self.v
        v[self.in_idx[idx]] = obs.astype(np.float64)
        s = np.bincount(self.dst, weights=self.w * v[self.src], minlength=self.K)[self.nodes] if self.src.size else np.zeros(self.nodes.size)
        z = self.bias + self.resp * s
        if self.kind == "sigmoid":
            v[self.nodes] = 1.0 / (1.0 + np.exp(-np.clip(5.0 * z, -60.0, 60.0)))
        else:
            v[self.nodes] = np.tanh(np.clip(2.5 * z, -60.0, 60.0))
        return v[self.out_idx[idx]].astype(np.float32)


class NetsPolicy:
    """A list of neat-python networks, one per robot."""
    def __init__(self, nets):
        self.nets = nets

    def reset(self, n):
        for net in self.nets:
            if hasattr(net, "reset"):
                net.reset()

    def act(self, idx, obs):
        lst = obs.tolist()
        nets = self.nets
        out = [nets[i].activate(o) for i, o in zip(idx.tolist(), lst)]
        return np.array(out, dtype=np.float32)


# ---------------------------------------------------------------------- data access
def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def keys_of(repo, rel):
    root = os.path.join(repo, "data", "maps")
    return [os.path.relpath(p, root).replace("\\", "/") for p in sorted(glob.glob(os.path.join(root, rel, "*.json")))]


def compute_geodesic(segments, goal, res=GEO_RES, inflation=0.30):
    """Distance-to-go field (metres) on a 5 cm grid, obstacles inflated by inflation; unreachable cells filled from the nearest reachable one."""
    from scipy import ndimage
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    n = int(round(10.0 / res))
    ys = (np.arange(n) + 0.5) * res
    xs = (np.arange(n) + 0.5) * res
    xx, yy = np.meshgrid(xs, ys)
    pts_x, pts_y = xx.ravel(), yy.ravel()
    segs = np.asarray(segments, dtype=np.float64)
    occ = np.zeros(pts_x.shape, bool)
    for seg in segs:
        ax, ay, bx, by = seg[0, 0], seg[0, 1], seg[1, 0], seg[1, 1]
        sx, sy = bx - ax, by - ay
        ss = max(sx * sx + sy * sy, 1e-12)
        t = np.clip(((pts_x - ax) * sx + (pts_y - ay) * sy) / ss, 0, 1)
        occ |= (pts_x - (ax + t * sx)) ** 2 + (pts_y - (ay + t * sy)) ** 2 <= inflation ** 2
    occ = occ.reshape(n, n)
    idx = np.arange(n * n).reshape(n, n)
    edges = []
    for dr, dc, w in ((0, 1, res), (1, 0, res), (1, 1, res * math.sqrt(2)), (1, -1, res * math.sqrt(2))):
        r0, c0 = np.arange(max(0, -dr), n - max(0, dr)), None
        a = occ[max(0, -dr):n - max(0, dr), max(0, -dc):n - max(0, dc)]
        b = occ[max(0, dr):n - max(0, -dr), max(0, dc):n - max(0, -dc)]
        ia = idx[max(0, -dr):n - max(0, dr), max(0, -dc):n - max(0, dc)]
        ib = idx[max(0, dr):n - max(0, -dr), max(0, dc):n - max(0, -dc)]
        ok = ~(a | b)
        edges.append((ia[ok], ib[ok], np.full(int(ok.sum()), w)))
    i = np.concatenate([e[0] for e in edges])
    j = np.concatenate([e[1] for e in edges])
    w = np.concatenate([e[2] for e in edges])
    g = csr_matrix((w, (i, j)), shape=(n * n, n * n))
    gc = np.clip((np.asarray(goal) / res).astype(int), 0, n - 1)
    src = idx[gc[1], gc[0]]
    if occ[gc[1], gc[0]]:     # goal cell inside the inflated zone: start from the nearest free cell
        fi = ndimage.distance_transform_edt(occ, return_distances=False, return_indices=True)
        src = idx[fi[0][gc[1], gc[0]], fi[1][gc[1], gc[0]]]
    d = dijkstra(g, directed=False, indices=int(src)).reshape(n, n)
    bad = ~np.isfinite(d)
    if bad.any():
        di, fi = ndimage.distance_transform_edt(bad, return_distances=True, return_indices=True)
        d = d[fi[0], fi[1]] + di * res
    return d.astype(np.float32)
