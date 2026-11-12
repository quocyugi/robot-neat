"""Hand-written reference controllers that use the same sensing as the evolved networks.

BugPolicy      : only the 15-D observation plus a small internal state (mode, hit distance, side).
MappingPlanner : 12-ray LiDAR + true pose, builds an occupancy grid online and replans (upper bound for local sensing).
"""
from __future__ import annotations

import heapq
import math

import numpy as np

DIAG = math.sqrt(200.0)
ANG = np.deg2rad(np.arange(12) * 30.0)
ANG = np.where(ANG > np.pi, ANG - 2 * np.pi, ANG)


def blocked(d, direction, width=0.34, look=0.70):
    """True if a disc moving toward `direction` (radians, robot frame) would hit something a ray sees."""
    dl = (ANG - direction + np.pi) % (2 * np.pi) - np.pi
    fw = d * np.cos(dl)
    lat = np.abs(d * np.sin(dl))
    return bool(np.any((fw > 0.0) & (fw < look) & (lat < width)))


class BugPolicy:
    """Bug-style go-to-goal / boundary-follow behaviour."""

    def __init__(self, side_rule="open", d_wall=0.60, flip_after=260):
        self.side_rule, self.d_wall, self.flip_after = side_rule, d_wall, flip_after

    def reset(self, n):
        self.mode = 0
        self.hit = 1e9
        self.side = 1
        self.cnt = 0
        self.flips = 0

    def _choose_side(self, d, phi):
        left = float(d[1] + d[2] + d[3])
        right = float(d[11] + d[10] + d[9])
        if self.side_rule == "right":
            return 1
        if self.side_rule == "left":
            return -1
        if self.side_rule == "goal":
            return 1 if phi >= 0 else -1
        return 1 if left >= right else -1   # "open": keep the wall on the side that is more blocked

    def act(self, idx, obs):
        o = obs[0]
        d = (1.0 - o[:12].astype(np.float64)) * 3.0
        rho = float(o[12]) * DIAG
        phi = math.atan2(float(o[13]), float(o[14]))
        if self.mode == 0:
            if blocked(d, phi):
                self.mode = 1
                self.hit = rho
                self.side = self._choose_side(d, phi)
                self.cnt = 0
            else:
                if abs(phi) > 0.30:
                    turn, speed = (1.0 if phi > 0 else -1.0) * 0.8, 0.0
                else:
                    turn, speed = float(np.clip(phi * 2.0, -0.6, 0.6)), 1.0
                return self._cmd(speed, turn)
        # boundary following; side=+1 means the wall is on the right, so the robot turns left to avoid it
        self.cnt += 1
        s = self.side
        if self.cnt > self.flip_after:
            self.side = -s
            s = self.side
            self.cnt = 0
            self.hit = min(self.hit, rho) + 0.5
        if rho < self.hit - 0.05 and not blocked(d, phi, look=1.0) and abs(phi) < 2.6:
            self.mode = 0
            if abs(phi) > 0.30:
                return self._cmd(0.0, (1.0 if phi > 0 else -1.0) * 0.8)
            return self._cmd(1.0, float(np.clip(phi * 2.0, -0.6, 0.6)))
        i90, i60, i30, i120 = (9, 10, 11, 8) if s > 0 else (3, 2, 1, 4)
        if blocked(d, 0.0, look=0.62):
            return self._cmd(0.0, 0.8 * s)
        w = min(d[i90], d[i60] * 0.866, d[i120] * 0.866, d[i30] * 0.5)
        if w > 1.15:
            return self._cmd(0.55, -0.55 * s)
        return self._cmd(0.9, -s * float(np.clip(2.5 * (w - self.d_wall), -0.6, 0.6)))

    @staticmethod
    def _cmd(speed, turn):
        o_l = float(np.clip(speed - turn, -1.0, 1.0))
        o_r = float(np.clip(speed + turn, -1.0, 1.0))
        return np.array([[o_l, o_r]], dtype=np.float32)


# ---------------------------------------------------------------------- online mapping
RES = 0.10
N = 100
NB = [(-1, -1, 1.4142), (-1, 0, 1.0), (-1, 1, 1.4142), (0, -1, 1.0), (0, 1, 1.0), (1, -1, 1.4142), (1, 0, 1.0), (1, 1, 1.4142)]
_D = int(math.ceil(0.32 / RES))
DISK = [(r, c) for r in range(-_D, _D + 1) for c in range(-_D, _D + 1) if r * r + c * c <= (0.32 / RES) ** 2]


def _cell(x, y):
    return max(0, min(N - 1, int(y / RES))), max(0, min(N - 1, int(x / RES)))


def _astar(blk, s, g):
    openh = [(0.0, s)]
    gs = {s: 0.0}
    came = {}
    seen = set()
    while openh:
        _, cur = heapq.heappop(openh)
        if cur == g:
            p = [cur]
            while cur in came:
                cur = came[cur]
                p.append(cur)
            return p[::-1]
        if cur in seen:
            continue
        seen.add(cur)
        for dr, dc, w in NB:
            nb = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nb[0] < N and 0 <= nb[1] < N) or nb in seen or blk[nb]:
                continue
            t = gs[cur] + w
            if t < gs.get(nb, 1e18):
                gs[nb] = t
                came[nb] = cur
                h = math.hypot(nb[0] - g[0], nb[1] - g[1])
                heapq.heappush(openh, (t + h, nb))
    return None


def run_mapping(env, replan=4):
    """One episode on a RobotNavEnv. The planner only learns the map through its own LiDAR hits."""
    obs, _ = env.reset()
    occ = np.zeros((N, N), bool)
    blk = np.zeros((N, N), bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = False
    path, k = None, 0
    term = trunc = False
    info = None
    t = 0

    def mark(px, py):
        r, c = _cell(px, py)
        if not occ[r, c]:
            occ[r, c] = True
            for dr, dc in DISK:
                rr, cc = r + dr, c + dc
                if 0 <= rr < N and 0 <= cc < N:
                    blk[rr, cc] = True

    while not (term or trunc):
        x, y, th = float(env.pos[0]), float(env.pos[1]), float(env.theta)
        d = (1.0 - obs[:12].astype(np.float64)) * 3.0
        for i in range(12):
            if d[i] < 2.99:
                a = th + i * math.pi / 6
                mark(x + d[i] * math.cos(a), y + d[i] * math.sin(a))
        if t % replan == 0 or path is None or k >= len(path):
            s = _cell(x, y)
            g = _cell(float(env.goal_pos[0]), float(env.goal_pos[1]))
            sblk = blk.copy()
            # keep the cells around the robot and the goal usable, otherwise A* has no start or end
            for cell in (s, g):
                r0, c0 = cell
                sblk[max(0, r0 - 2):min(N, r0 + 3), max(0, c0 - 2):min(N, c0 + 3)] = False
            path = _astar(sblk, s, g)
            k = 0
        t += 1
        if path is None:
            action = np.array([0.0, 0.0], dtype=np.float32)
        else:
            while k < len(path) - 1 and math.hypot((path[k][1] + 0.5) * RES - x, (path[k][0] + 0.5) * RES - y) < 0.40:
                k += 1
            wx, wy = (path[k][1] + 0.5) * RES, (path[k][0] + 0.5) * RES
            if k >= len(path) - 1:
                wx, wy = float(env.goal_pos[0]), float(env.goal_pos[1])
            err = (math.atan2(wy - y, wx - x) - th + math.pi) % (2 * math.pi) - math.pi
            if abs(err) > 0.28:
                u = float(np.clip(err, -0.8, 0.8))
                action = np.array([-u, u], dtype=np.float32)
            elif blocked(d, 0.0, width=0.30, look=0.55):
                # something the sparse rays only just revealed: mark it and rotate towards the path
                mark(x + 0.6 * math.cos(th), y + 0.6 * math.sin(th))
                path = None
                u = 0.8 if err >= 0 else -0.8
                action = np.array([-u, u], dtype=np.float32)
            else:
                u = float(np.clip(err * 2.0, -0.5, 0.5))
                action = np.array([1.0 - u, 1.0 + u], dtype=np.float32)
        obs, _, term, trunc, info = env.step(action)
    env.close()
    return info
