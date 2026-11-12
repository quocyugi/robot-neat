"""Kinematic rover on DUnE's real meshes: terrain following + ray-cast LiDAR, batched with Open3D/Embree.

This is NOT Gazebo. What it keeps from DUnE: the meshes (collision for wheels, visual for LiDAR), the rover geometry
and LiDAR mount of camera_rover_4wd.sdf, the LiDAR handling and the spawn/goal/termination rules of rover_env.py.
What it leaves out: tyre friction and slip, suspension, mass, motor limits, real tip-over dynamics.
Terrain steeper than slope_limit is treated as an obstacle.
"""
from __future__ import annotations

import math

import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

WHEELS = np.array([[0.1415, 0.18], [0.1415, -0.18], [-0.145, 0.18], [-0.145, -0.18]])   # camera_rover_4wd.sdf
_PINV = np.linalg.pinv(np.c_[WHEELS, np.ones(4)])
LIDAR_X, LIDAR_Z = 0.174, 0.321
V_MAX, WHEEL_BASE, ARENA_DIAGONAL = 0.50, 0.40, float(np.sqrt(200.0))     # the 2D controllers' own scales


class Cfg:
    n_beams = 96             # multiple of 12; Gazebo renders 640 beams, DUnE then keeps 32 min-pooled bins
    lidar_min, lidar_max, no_return = 0.15, 10.0, 12.0   # sdf <min>,<max>; rover_env replaces inf by 12
    range_2d = 3.0           # the 2D controllers saw 3 m
    slope_limit = math.radians(40.0)
    step_limit, drop_limit = 0.13, 0.30    # wheel radius 0.122 m
    dt = 0.10
    max_ticks = 1000         # 2000 steps at 20 Hz in rover_env = 100 s
    stuck_ticks = 300        # 600 steps at 20 Hz
    stuck_dist = 0.001
    goal_dist = 0.5          # rover_env success_distance
    half_len, half_wid = 0.27, 0.23


def make_scene(V, F):
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3d.core.Tensor(np.asarray(V, np.float32)), o3d.core.Tensor(np.asarray(F, np.uint32)))
    return s


class Sim3D:
    def __init__(self, Vg, Fg, Vl, Fl, cfg: Cfg = Cfg(), nthreads: int = 1):
        self.cfg = cfg
        self.gs, self.ls = make_scene(Vg, Fg), make_scene(Vl, Fl)
        self.zmax = float(max(Vg[:, 2].max(), Vl[:, 2].max()))
        self.nt = nthreads
        B = cfg.n_beams
        self.alpha = np.arange(B) * 2 * np.pi / B
        self.sectors = np.array([[(k * B // 12 + j) % B for j in range(-B // 24, B // 24)] for k in range(12)])
        a = np.arange(12) * 2 * np.pi / 12 + 0.13
        self.body_dir = np.stack([np.cos(a), np.sin(a)], axis=1)
        self.body_len = np.minimum(cfg.half_len / np.maximum(np.abs(np.cos(a)), 1e-6), cfg.half_wid / np.maximum(np.abs(np.sin(a)), 1e-6))
        self.body_h = np.array([0.08, 0.28])
        self.rays = 0

    def _cast(self, scene, rays):
        self.rays += len(rays)
        r = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)), nthreads=self.nt)
        return r["t_hit"].numpy().astype(np.float64), r["primitive_normals"].numpy()

    # ------------------------------------------------------------------ terrain contact
    def contact(self, x, y, yaw, zref):
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        wx = x[:, None] + c * WHEELS[:, 0] - s * WHEELS[:, 1]
        wy = y[:, None] + s * WHEELS[:, 0] + c * WHEELS[:, 1]
        oz = np.broadcast_to((zref + 0.45)[:, None], wx.shape)
        rays = np.stack([wx, wy, oz, np.zeros_like(wx), np.zeros_like(wx), -np.ones_like(wx)], axis=2).reshape(-1, 6)
        t, _ = self._cast(self.gs, rays)
        t = t.reshape(-1, 4)
        miss = ~np.isfinite(t)
        h = oz - np.where(miss, 0.45, t)
        return h, miss

    @staticmethod
    def frame(yaw, h):
        sol = np.einsum("ij,nj->ni", _PINV, h)    # p, q, c
        p, q, c0 = sol[:, 0], sol[:, 1], sol[:, 2]
        co, si = np.cos(yaw), np.sin(yaw)
        X = np.stack([co, si, p], axis=1)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        Y0 = np.stack([-si, co, q], axis=1)
        Z = np.cross(X, Y0)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        Y = np.cross(Z, X)
        return X, Y, Z, c0

    def body_hit(self, x, y, z, X, Y, Z):
        n, nd, nh = len(x), len(self.body_dir), len(self.body_h)
        base = np.stack([x, y, z], axis=1)
        dirs = self.body_dir[None, :, 0, None] * X[:, None, :] + self.body_dir[None, :, 1, None] * Y[:, None, :]    # n,nd,3
        o = base[:, None, None, :] + self.body_h[None, None, :, None] * Z[:, None, None, :]                         # n,1,nh,3
        o = np.broadcast_to(o, (n, nd, nh, 3))
        d = np.broadcast_to(dirs[:, :, None, :], (n, nd, nh, 3))
        rays = np.concatenate([o, d], axis=3).reshape(-1, 6)
        t, nor = self._cast(self.gs, rays)
        L = np.broadcast_to(self.body_len[None, :, None], (n, nd, nh)).ravel()
        wall = np.abs(nor[:, 2]) < math.cos(self.cfg.slope_limit)
        return ((t < L) & wall).reshape(n, -1).any(axis=1)

    # ------------------------------------------------------------------ sensing
    def lidar(self, x, y, z, X, Y, Z):
        B = self.cfg.n_beams
        base = np.stack([x, y, z], axis=1)
        O = base + LIDAR_X * X + LIDAR_Z * Z
        ca, sa = np.cos(self.alpha)[None, :, None], np.sin(self.alpha)[None, :, None]
        D = ca * X[:, None, :] + sa * Y[:, None, :]
        rays = np.concatenate([np.repeat(O[:, None, :], B, axis=1), D], axis=2).reshape(-1, 6)
        t, nor = self._cast(self.ls, rays)
        return t.reshape(-1, B), np.abs(nor[:, 2]).reshape(-1, B)

    def ranges(self, t):
        c = self.cfg
        return np.where((t < c.lidar_min) | (t > c.lidar_max) | ~np.isfinite(t), c.no_return, t)

    def obs2d(self, rng, x, y, yaw, gx, gy, sector=True, imu=None):
        """The 15-D observation of the 2D controllers (optionally followed by pitch and roll)."""
        if sector:
            d12 = rng[np.arange(len(rng))[:, None, None], self.sectors[None, :, :]].min(axis=2)
        else:
            d12 = rng[:, self.sectors[:, self.cfg.n_beams // 24]]
        prox = 1.0 - np.minimum(d12, self.cfg.range_2d) / self.cfg.range_2d
        dx, dy = gx - x, gy - y
        rho = np.clip(np.hypot(dx, dy) / ARENA_DIAGONAL, 0.0, 1.0)
        phi = np.arctan2(dy, dx) - yaw
        o = [prox, rho[:, None], np.sin(phi)[:, None], np.cos(phi)[:, None]]
        if imu is not None:
            o.append(imu)
        return np.concatenate(o, axis=1).astype(np.float32)


class EP:
    __slots__ = ("reason", "steps", "path", "blocks", "block_steps", "close_steps", "ok_ticks", "min_lidar", "pitch", "roll", "ground_frac", "gmin")


def rollout3d(sim: Sim3D, task: dict, ep: dict, policy, ground_filter: str = "none", sector: bool = True, use_imu: bool = False, geo_fn=None, max_ticks: int | None = None, track: list | None = None):
    """ep: x,y,yaw,gx,gy arrays. Reason codes: 1 success, 2 fall, 3 too far, 4 stuck, 5 timeout."""
    c = sim.cfg
    max_ticks = max_ticks or c.max_ticks
    n = len(ep["x"])
    x, y, yaw = np.array(ep["x"], float), np.array(ep["y"], float), np.array(ep["yaw"], float)
    gx, gy = np.array(ep["gx"], float), np.array(ep["gy"], float)
    z0 = np.array(ep["z"], float)
    h, _ = sim.contact(x, y, yaw, z0)
    X, Y, Z, z = sim.frame(yaw, h)
    R = EP()
    R.reason = np.zeros(n, np.int32)
    R.steps = np.zeros(n, np.int32)
    for k in ("path", "block_steps", "close_steps", "min_lidar", "pitch", "roll", "ground_frac", "blocks"):
        setattr(R, k, np.zeros(n, np.float64))
    R.ok_ticks = np.zeros(n, np.int32)
    R.gmin = None
    if geo_fn is not None:
        R.gmin = np.array(ep["geo"], float).copy()
    idx = np.arange(n)
    was_blocked = np.zeros(n, bool)
    ring_x, ring_y = np.repeat(x[:, None], c.stuck_ticks, 1), np.repeat(y[:, None], c.stuck_ticks, 1)
    tf = task["too_far"]
    gt = math.cos(math.radians(35.0))
    policy.reset(n)
    for tick in range(1, max_ticks + 1):
        t_hit, nz = sim.lidar(x, y, z, X, Y, Z)
        rng = sim.ranges(t_hit)
        groundish = (nz >= gt) & (rng < 3.0)
        R.ground_frac[idx] += groundish.mean(axis=1)
        R.close_steps[idx] += (rng.min(axis=1) < 0.3)
        R.min_lidar[idx] += rng.min(axis=1)
        R.ok_ticks[idx] += 1
        if ground_filter == "normal":
            rng = np.where(nz >= gt, c.no_return, rng)
        pitch = np.arcsin(np.clip(X[:, 2], -1, 1))
        roll = np.arcsin(np.clip(Y[:, 2], -1, 1))
        R.pitch[idx] += np.abs(pitch)
        R.roll[idx] += np.abs(roll)
        imu = np.stack([pitch, roll], axis=1) if use_imu else None
        obs = sim.obs2d(rng, x, y, yaw, gx[idx], gy[idx], sector=sector, imu=imu)
        act = policy.act(idx, obs)
        o_l, o_r = np.clip(act[:, 0], -1, 1).astype(np.float64), np.clip(act[:, 1], -1, 1).astype(np.float64)
        v_l, v_r = V_MAX * o_l, V_MAX * o_r
        om, v = (v_r - v_l) / WHEEL_BASE, (v_r + v_l) / 2.0
        yaw_new = yaw + om * c.dt
        step = v * c.dt * np.sqrt(np.maximum(X[:, 0] ** 2 + X[:, 1] ** 2, 0.0))
        xn, yn = x + step * np.cos(yaw_new), y + step * np.sin(yaw_new)
        hn, miss = sim.contact(xn, yn, yaw_new, z)
        Xn, Yn, Zn, zn = sim.frame(yaw_new, hn)
        dz = hn - h
        slope = np.arccos(np.clip(Zn[:, 2], -1, 1))
        blocked = miss.any(axis=1) | (dz > c.step_limit).any(axis=1) | (slope > c.slope_limit)
        free = np.where(~blocked)[0]
        if free.size:
            bh = sim.body_hit(xn[free], yn[free], zn[free], Xn[free], Yn[free], Zn[free])
            blocked[free[bh]] = True
        fall = ((-dz) > c.drop_limit).any(axis=1) & ~blocked
        moved = ~blocked
        R.path[idx] += np.where(moved, np.hypot(xn - x, yn - y), 0.0)
        R.blocks[idx] += blocked & ~was_blocked[idx]
        R.block_steps[idx] += blocked
        was_blocked[idx] = blocked
        x = np.where(moved, xn, x)
        y = np.where(moved, yn, y)
        yaw = yaw_new
        if blocked.any():        # stay in place, heading still changes
            b = np.where(blocked)[0]
            hb, _ = sim.contact(x[b], y[b], yaw[b], z[b])
            hn[b] = np.where(np.abs(hb - h[b]) < 0.4, hb, h[b])
            Xb, Yb, Zb, zb = sim.frame(yaw[b], hn[b])
            Xn[b], Yn[b], Zn[b], zn[b] = Xb, Yb, Zb, zb
        h, X, Y, Z, z = hn, Xn, Yn, Zn, zn
        if track is not None:
            track.append((idx.copy(), x.copy(), y.copy()))
        if R.gmin is not None:
            R.gmin[idx] = np.minimum(R.gmin[idx], geo_fn(idx, x, y))
        ring_x[idx, tick % c.stuck_ticks], ring_y[idx, tick % c.stuck_ticks] = x, y
        dg = np.hypot(x - gx[idx], y - gy[idx])
        reason = np.zeros_like(idx)
        reason[(x < tf[0]) | (x > tf[1]) | (y < tf[2]) | (y > tf[3])] = 3
        if tick >= c.stuck_ticks:
            prev = (tick + 1) % c.stuck_ticks
            still = np.hypot(x - ring_x[idx, prev], y - ring_y[idx, prev]) < c.stuck_dist
            reason[still] = 4
        reason[fall] = 2
        reason[dg < c.goal_dist] = 1
        if tick == max_ticks:
            reason[reason == 0] = 5
        done = reason > 0
        if done.any():
            d = idx[done]
            R.reason[d], R.steps[d] = reason[done], tick
            keep = ~done
            idx, x, y, yaw, h, X, Y, Z, z = idx[keep], x[keep], y[keep], yaw[keep], h[keep], X[keep], Y[keep], Z[keep], z[keep]
            if idx.size == 0:
                break
    return R


def summarize3d(R: EP, ep: dict, dt: float = 0.10):
    succ = R.reason == 1
    n = len(succ)
    ticks = np.maximum(R.ok_ticks, 1)
    s = {"n": n, "SR": round(float(succ.mean()), 4)}
    for name, code in (("fall", 2), ("too_far", 3), ("stuck", 4), ("timeout", 5)):
        s[name] = round(float((R.reason == code).mean()), 4)
    s["TC_blocks_per_ep"] = round(float(R.blocks.mean()), 2)
    s["block_steps_frac"] = round(float((R.block_steps / ticks).mean()), 4)
    s["close_lt_0.3_frac"] = round(float((R.close_steps / ticks).mean()), 4)
    s["mean_min_lidar"] = round(float((R.min_lidar / ticks).mean()), 3)
    s["MTT_s"] = round(float(R.steps[succ].mean() * dt), 2) if succ.any() else None
    geo = np.asarray(ep["geo"], float)
    s["SPL"] = round(float(np.sum(succ * geo / np.maximum(R.path, geo)) / n), 4)
    s["mean_speed"] = round(float((R.path / (R.steps * dt + 1e-9))[R.steps > 0].mean()), 3)
    s["mean_abs_pitch_deg"] = round(float(np.degrees((R.pitch / ticks).mean())), 2)
    s["mean_abs_roll_deg"] = round(float(np.degrees((R.roll / ticks).mean())), 2)
    s["ground_hit_frac_3m"] = round(float((R.ground_frac / ticks).mean()), 4)
    return s


# ---------------------------------------------------------------------------------------- oracle on a 2.5D grid
class Grid:
    """Which cells a rover footprint can stand on, and shortest paths on them (upper bound used to validate episodes)."""

    def __init__(self, sim: Sim3D, region, res=0.10, z0=None, inflate=0.30):
        self.sim, self.res = sim, res
        x0, x1, y0, y1 = region
        self.x0, self.y0 = x0, y0
        self.nx, self.ny = int((x1 - x0) / res), int((y1 - y0) / res)
        xs = x0 + (np.arange(self.nx) + 0.5) * res
        ys = y0 + (np.arange(self.ny) + 0.5) * res
        xx, yy = np.meshgrid(xs, ys)
        ztop = (z0 + 1.0) if z0 is not None else sim.zmax + 5.0
        rays = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, ztop), np.zeros(xx.size), np.zeros(xx.size), -np.ones(xx.size)], axis=1)
        t, nor = sim._cast(sim.gs, rays)
        h = ztop - t.reshape(self.ny, self.nx)
        nz = np.abs(nor[:, 2]).reshape(self.ny, self.nx)
        hole = ~np.isfinite(h)
        lim = math.cos(sim.cfg.slope_limit)
        steep = nz < lim
        h = np.where(hole, np.nan, h)
        wall = np.zeros_like(hole)
        # wall probes at 0.2 m above the local ground
        ok = np.where(~hole.ravel())[0]
        ang = np.arange(8) * np.pi / 4
        ox = np.repeat(xx.ravel()[ok], 8)
        oy = np.repeat(yy.ravel()[ok], 8)
        oz = np.repeat(h.ravel()[ok] + 0.2, 8)
        dx, dy = np.tile(np.cos(ang), ok.size), np.tile(np.sin(ang), ok.size)
        t2, nor2 = sim._cast(sim.gs, np.stack([ox, oy, oz, dx, dy, np.zeros_like(dx)], axis=1))
        hit = ((t2 < 0.25) & (np.abs(nor2[:, 2]) < lim)).reshape(ok.size, 8).any(axis=1)
        wall.ravel()[ok] = hit
        # height jumps between neighbours
        filled = np.nan_to_num(h, nan=-1e3)
        jump = np.zeros_like(hole)
        for ax in (0, 1):
            d = np.abs(np.diff(filled, axis=ax))
            pad = [(0, 0), (0, 0)]
            pad[ax] = (1, 0)
            a = np.pad(d, pad, mode="constant")
            pad[ax] = (0, 1)
            b = np.pad(d, pad, mode="constant")
            jump |= (a > sim.cfg.step_limit) | (b > sim.cfg.step_limit)
        self.h, self.steep, self.wall, self.hole, self.jump = h, steep, wall, hole, jump
        bad = hole | steep | wall | jump
        r = int(round(inflate / res))
        yy_d, xx_d = np.ogrid[-r:r + 1, -r:r + 1]
        disk = xx_d * xx_d + yy_d * yy_d <= r * r
        self.free = ~ndimage.binary_dilation(bad, structure=disk)
        self._graph()

    def _graph(self):
        n = self.nx * self.ny
        idx = np.arange(n).reshape(self.ny, self.nx)
        I, J, W = [], [], []
        for dr, dc, w in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2))):
            r0 = (0, self.ny - dr)
            c0 = (max(0, -dc), self.nx - max(0, dc))
            a = idx[r0[0]:r0[1], c0[0]:c0[1]]
            b = a + dr * self.nx + dc
            ok = self.free.ravel()[a.ravel()] & self.free.ravel()[b.ravel()]
            I.append(a.ravel()[ok]), J.append(b.ravel()[ok]), W.append(np.full(int(ok.sum()), w * self.res))
        self.graph = csr_matrix((np.concatenate(W), (np.concatenate(I), np.concatenate(J))), shape=(n, n))

    def cell(self, x, y):
        return np.floor((np.asarray(y) - self.y0) / self.res).astype(int), np.floor((np.asarray(x) - self.x0) / self.res).astype(int)

    def distance_from(self, x, y):
        r, c = self.cell(x, y)
        d = dijkstra(self.graph, directed=False, indices=int(r * self.nx + c))
        return d.reshape(self.ny, self.nx)

    def sample_episodes(self, spawn, goal, n, rng, min_dist=1.0, max_geo=None):
        out = {k: [] for k in ("x", "y", "yaw", "z", "gx", "gy", "euclid", "geo")}
        c = dict(attempts=0, goal_not_free=0, start_not_free=0, too_close=0, unreachable=0, too_long=0)
        while len(out["x"]) < n:
            c["attempts"] += 1
            gx, gy = rng.uniform(goal[0], goal[1]), rng.uniform(goal[2], goal[3])
            sx, sy = rng.uniform(spawn[0], spawn[1]), rng.uniform(spawn[2], spawn[3])
            gr, gc = self.cell(gx, gy)
            sr, sc = self.cell(sx, sy)
            if not self.free[gr, gc]:
                c["goal_not_free"] += 1
                continue
            if not self.free[sr, sc]:
                c["start_not_free"] += 1
                continue
            e = math.hypot(sx - gx, sy - gy)
            if e < min_dist:
                c["too_close"] += 1
                continue
            d = self.distance_from(gx, gy)
            if not np.isfinite(d[sr, sc]):
                c["unreachable"] += 1
                continue
            if max_geo is not None and d[sr, sc] > max_geo:
                c["too_long"] += 1
                continue
            for k, v in zip(("x", "y", "yaw", "z", "gx", "gy", "euclid", "geo"),
                            (sx, sy, rng.uniform(-np.pi, np.pi), float(self.h[sr, sc]), gx, gy, e, float(d[sr, sc]))):
                out[k].append(v)
        return out, c


# ---------------------------------------------------------------------------------------- simple policies
class GoToGoal:
    """No obstacle avoidance at all: lower bound."""
    def reset(self, n):
        pass

    def act(self, idx, obs):
        phi = np.arctan2(obs[:, 13], obs[:, 14])
        speed = (np.abs(phi) <= 0.30).astype(np.float32)
        turn = np.where(np.abs(phi) > 0.30, np.sign(phi) * 0.8, np.clip(phi * 2.0, -0.6, 0.6))
        return np.stack([np.clip(speed - turn, -1, 1), np.clip(speed + turn, -1, 1)], axis=1).astype(np.float32)


class PerRobot:
    """Wrap a controller that handles one robot and keeps per-robot state."""
    def __init__(self, factory):
        self.factory = factory

    def reset(self, n):
        self.items = [self.factory() for _ in range(n)]
        for p in self.items:
            p.reset(1)

    def act(self, idx, obs):
        out = [self.items[i].act(None, obs[j:j + 1])[0] for j, i in enumerate(idx.tolist())]
        return np.array(out, dtype=np.float32)


class APF:
    """Same arithmetic as the team's src/baselines/apf.py, on batches."""
    def __init__(self, k_att=1.0, k_rep=0.5, d_safe=1.0):
        self.k_att, self.k_rep, self.d_safe = k_att, k_rep, d_safe
        a = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        self.ca, self.sa = np.cos(a), np.sin(a)

    def reset(self, n):
        pass

    def act(self, idx, obs):
        o = obs.astype(np.float64)
        prox = o[:, :12]
        dist = (1.0 - prox) * 3.0
        mag = self.k_rep * (1.0 / np.maximum(dist, 0.05) - 1.0 / self.d_safe)
        use = (prox > 0.01) & (dist < self.d_safe)
        fx = o[:, 14] * self.k_att - (np.where(use, mag, 0.0) * self.ca).sum(axis=1)
        fy = o[:, 13] * self.k_att - (np.where(use, mag, 0.0) * self.sa).sum(axis=1)
        speed, turn = np.clip(fx * 0.5, -1, 1), np.clip(fy * 0.5, -1, 1)
        return np.stack([np.clip(speed - turn, -1, 1), np.clip(speed + turn, -1, 1)], axis=1).astype(np.float32)
