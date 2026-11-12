"""Exercise neat_dune_bridge.py without ROS: a fake RoverEnv with DUnE's observation/action format on top of Sim3D.

Checks (1) the 32-bin -> 12-sector mapping, (2) the [speed, heading] action mapping, (3) that the bridge reaches the same
success rate as the batched simulator on the same episodes.

Usage: python -I d12_mock_bridge.py <repo> <dune_root> <cache_dir> <results_dir> <bridge_dir> <run_dir> <out_json> [world ...]
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tests2"))

import numpy as np

repo, dune, cache, res_dir, bridge_dir, run_dir, out = sys.argv[1:8]
worlds = sys.argv[8:] or ["pillars", "inspect"]
sys.path.insert(0, repo)
sys.path.insert(0, bridge_dir)

import neat_dune_bridge as B
import sim3d as S
import world3d as W


class FakeRoverEnv:
    """Same interface as DUnE's RoverEnv for the parts the bridge uses. dt = 0.05 s, 2000 steps, heading P controller with Kp = 2."""
    def __init__(self, sim, task, ep):
        self.sim, self.task, self.ep, self.i = sim, task, ep, -1
        self.cfg = sim.cfg

    def _sense(self):
        x, y, yaw = np.array([self.x]), np.array([self.y]), np.array([self.yaw])
        h, _ = self.sim.contact(x, y, yaw, np.array([self.z]))
        X, Y, Z, z = self.sim.frame(yaw, h)
        self.h, self.X, self.Y, self.Z, self.z = h, X, Y, Z, float(z[0])
        t, _ = self.sim.lidar(x, y, z, X, Y, Z)
        rng = self.sim.ranges(t)[0]
        B_ = self.cfg.n_beams
        bins = np.array([rng[(i * B_ // 32):((i + 1) * B_ // 32)].min() for i in range(32)])
        dx, dy = self.gx - self.x, self.gy - self.y
        az = math.atan2(math.sin(math.atan2(dy, dx) - self.yaw), math.cos(math.atan2(dy, dx) - self.yaw))
        return {"lidar": bins, "target": np.array([math.hypot(dx, dy), az])}

    def reset(self):
        self.i += 1
        e = self.ep
        self.x, self.y, self.yaw, self.z = e["x"][self.i], e["y"][self.i], e["yaw"][self.i], e["z"][self.i]
        self.gx, self.gy = e["gx"][self.i], e["gy"][self.i]
        self.t = 0
        return self._sense(), {}

    def step(self, a):
        speed, rel = float(a[0]), float(a[1])
        om = np.clip(2.0 * rel, -7.0, 7.0)
        dt = 0.05
        yaw_new = self.yaw + om * dt
        step = speed * dt * math.hypot(self.X[0, 0], self.X[0, 1])
        xn, yn = self.x + step * math.cos(yaw_new), self.y + step * math.sin(yaw_new)
        xa, ya, ta = np.array([xn]), np.array([yn]), np.array([yaw_new])
        hn, miss = self.sim.contact(xa, ya, ta, np.array([self.z]))
        Xn, Yn, Zn, zn = self.sim.frame(ta, hn)
        dz = hn - self.h
        blocked = bool(miss.any() or (dz > self.cfg.step_limit).any() or math.acos(min(1.0, Zn[0, 2])) > self.cfg.slope_limit)
        if not blocked:
            blocked = bool(self.sim.body_hit(xa, ya, zn, Xn, Yn, Zn)[0])
        if not blocked:
            self.x, self.y = xn, yn
        self.yaw = yaw_new
        self.t += 1
        obs = self._sense()
        tf = self.task["too_far"]
        far = not (tf[0] <= self.x <= tf[1] and tf[2] <= self.y <= tf[3])
        reward = 100.0 if math.hypot(self.x - self.gx, self.y - self.gy) < 0.5 else 0.0
        return obs, reward, far, self.t >= 2000, {}


# ---------------------------------------------------- unit checks of the pure conversions
checks = {}
ad = B.Adapter()
lidar = np.full(32, 12.0)
lidar[0] = 1.5                  # first bin: just left of straight ahead
o = ad.obs(lidar, [7.0710678, math.pi / 2])
checks["ray0_sees_bin0"] = bool(abs(o[0] - 0.5) < 1e-6)
checks["rho_scale_ok"] = bool(abs(o[12] - 0.5) < 1e-6)
checks["sin_phi_ok"] = bool(abs(o[13] - 1.0) < 1e-6 and abs(o[14]) < 1e-6)
checks["each_bin_used"] = sorted(set(sum(ad.sectors, []))) == list(range(32))
checks["bins_per_sector"] = sorted(set(len(s) for s in ad.sectors))
a = ad.action([1.0, 1.0])
checks["straight_full_speed"] = bool(abs(a[0] - 0.5) < 1e-6 and abs(a[1]) < 1e-6)
a = ad.action([-1.0, 1.0])
checks["spin_left"] = bool(abs(a[0]) < 1e-6 and abs(a[1] - 1.25) < 1e-6)

results = {"unit_checks": checks}
for w in worlds:
    task = W.TASKS[w]
    Vg, Fg, Vl, Fl = W.load_world(w, dune, cache)
    sim = S.Sim3D(Vg, Fg, Vl, Fl, nthreads=1)
    ep = json.load(open(os.path.join(res_dir, f"episodes_{w}.json"), encoding="utf-8"))
    n = 40
    sub = {k: v[:n] for k, v in ep.items()}
    env = FakeRoverEnv(sim, task, sub)
    pol = B.Policy(run_dir)
    r = B.evaluate(env, pol, B.Adapter(), n, log=lambda s: None)

    class Q:
        def reset(self, n_):
            self.items = [B.Policy(run_dir) for _ in range(n_)]
            for p in self.items:
                p.reset()

        def act(self, idx, obs):
            return np.array([self.items[i](obs[j]) for j, i in enumerate(idx.tolist())], dtype=np.float32)
    R = S.rollout3d(sim, task, sub, Q(), ground_filter="none")
    results[w] = {"bridge_SR": r["SR"], "bridge_MTT_s": r["MTT_s"], "batch_sim_SR": round(float((R.reason == 1).mean()), 4),
                  "batch_sim_MTT_s": round(float(R.steps[R.reason == 1].mean() * 0.1), 2) if (R.reason == 1).any() else None, "n": n}
    print(w, results[w], flush=True)
print(json.dumps(checks, indent=1))
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1)
