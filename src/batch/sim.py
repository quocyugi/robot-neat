"""
Mô phỏng nhiều lượt cùng lúc (lockstep), trùng từng bit với RobotNavEnv + run_episode
======================================================================================
Mỗi bước, mọi lượt còn chạy cùng đi một bước bằng vài phép trên mảng (A lượt): bộ điều khiển
(src/batch/policies.py), động học vi sai, va chạm, tiến độ, quan sát (LiDAR + hướng tới đích).
Lượt kết thúc được ghi kết quả rồi bị bỏ khỏi mọi mảng (nén), nên lượt va chạm sớm không tốn thêm
phép tính. Mỗi lượt độc lập (mạng hồi quy reset đầu lượt, các bộ điều khiển khác không có trạng
thái), nên kết quả một lượt không phụ thuộc thành phần của lô.

Thứ tự và kiểu của từng phép tính chép đúng đường từng lượt (src/env/robot_env.py, kinematics.py,
lidar.py, collision.py, geodesic.py, src/neat_engine/parallel.py):
- vị trí float64; góc hướng là float Python lúc reset (quan sát reset dùng float64 h0), từ bước 1 là
  float32 = float32(h0) + omega * dt (quy tắc NEP 50: số Python "yếu" bị ép về float32 trước khi cộng);
- động học float32 (hằng float32 tường minh), so sánh "quay tại chỗ" ở float32;
- LiDAR tia / hình quạt và va chạm từng phần tử theo đúng chuỗi phép của bản gốc, đoạn đệm của bucket
  luôn bị che bằng mặt nạ (đoạn suy biến được bản gốc tính là một điểm ở hình quạt và va chạm);
- np.linalg.norm của vectơ 2 phần tử = sqrt(x0*x0 + x1*x1) (được kiểm lúc khởi tạo, src/batch/numerics.py);
- min(d_min, d) của Python giữ giá trị cũ trừ khi giá trị mới nhỏ hơn hẳn;
- độ dài quãng đường: np.sum trên đúng dãy T độ dài từng bước (cùng thứ tự cộng cặp đôi của NumPy);
- trường geodesic tra trên lưới đổi sang float64 (GeodesicField.at trả float() của giá trị đang lưu: lưới float32
  đổi sang float64 không làm tròn, lưới float64 giữ nguyên);
- fitness: gọi lại chính compute_episode_fitness với dict info dựng từ số cuối lượt, cùng các khóa của
  RobotNavEnv.step trừ "trajectory" (không giữ quỹ đạo; hàm nào đọc khóa này thì bước tự kiểm trọn lượt ở
  src/batch/numerics.py báo lệch và chương trình quay về đường từng lượt).

Hình học (LiDAR + va chạm) là một đối tượng thay được: HostGeometry (xp, theo bucket số đoạn) hoặc
GpuGeometry (src/batch/gpu.py, kernel CUDA gộp). Cả hai có setup / collide / scan / keep.

Waffle (NEAT_ROBOT=waffle): động học là kinematics.waffle_step (cùng hàm với RobotNavEnv); hình học
điểm (HostGeometry / GpuGeometry, bán kính đĩa KM.COLL_RADIUS) được bọc bởi BodyCenterGeometry, đổi
tâm quay sang tâm thân (x - BODY_OFFSET cos th, y - BODY_OFFSET sin th, cos / sin float64 ở host) rồi
mới quét LiDAR và kiểm va chạm, đúng như RobotNavEnv.body_center. Kernel GPU không đổi. robot_geometry()
chọn đúng cách bọc; robot cũ nhận nguyên hình học điểm (không bọc).
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from src.env import kinematics as KM
from src.neat_engine import parallel as PAR
from src.neat_engine.evaluator import compute_episode_fitness

FAR = 1.0e6                       # tọa độ đoạn đệm (luôn bị che bằng mặt nạ)
LIDAR_ELEMS = 1 << 15             # số phần tử tối đa của một mảng tạm LiDAR (lượt x (N+1) x đoạn), ~256 KB
# (trước 27/11/2026 là 1 << 19, ~4 MB). Khúc nhỏ nằm gọn trong cache nên LiDAR của hàng batch nhanh hơn; kết quả
# không đổi vì khúc chỉ chia theo lượt, mỗi lượt vẫn tính đúng các phép như cũ.
# Giá trị 1 << 15 chọn cho NumPy (CPU). Đường gpu exact mặc định (lai) dùng kernel CUDA, không dùng hằng này;
# HostGeometry chạy trên CuPy (chế độ gpu-device không exact, biến thể gpu-full) cũng chia khúc theo hằng này
# nên có thể chậm hơn so với 1 << 19 (kết quả không đổi).
F32 = np.float32
VMAX32, L32, DT32, TWO32 = F32(KM.V_MAX), F32(KM.WHEEL_BASE), F32(KM.DT), F32(2.0)
_SPIN = inspect.signature(KM.is_spinning).parameters
EPS_LIN32 = F32(_SPIN["eps_linear"].default)       # is_spinning so sánh ở float32
EPS_ANG32 = F32(_SPIN["eps_angular"].default)
ONE32 = F32(1.0)


@dataclass(frozen=True)
class Episode:
    """Một lượt cần chạy: bản đồ, cách đo tiến độ, thư mục đệm geodesic, chỉ số bộ điều khiển trong nhóm."""
    map_path: str
    progress: str
    geo_cache: Optional[str]
    task: int


@dataclass
class MapInfo:
    """Hằng của một bản đồ, tính đúng như RobotNavEnv.__init__ / reset và run_episode."""
    path: str
    data: dict
    segs: np.ndarray            # (M, 2, 2) float64
    start: np.ndarray
    goal: np.ndarray
    h0: float                   # hướng xuất phát (float Python)
    d0: float
    geo: Any                    # GeodesicField hoặc None
    g0: Optional[float]
    l_opt: float


def map_info(path: str, progress: str, geo_cache: Optional[str]) -> MapInfo:
    m = PAR.load_map(path)
    segs = np.array(m["segments"], dtype=np.float64)
    start = np.array(m["start"], dtype=np.float64)
    goal = np.array(m["goal"], dtype=np.float64)
    h0 = float(m.get("start_heading", 0.0))
    d0 = float(np.linalg.norm(start.copy() - goal))
    geo = PAR._geodesic(path, geo_cache) if progress == "geodesic" else None
    g0 = geo.at(start[0], start[1]) if geo is not None else None
    l_opt = float((m.get("metrics") or {}).get("optimal_path_length", d0))
    return MapInfo(path, m, segs, start, goal, h0, d0, geo, g0, l_opt)


def segments_ok(segs: np.ndarray) -> bool:
    """Bản đồ chạy được theo lô: có ít nhất một đoạn, dạng (M, 2, 2) (M = 0 thì cast_rays của bản gốc báo lỗi)."""
    return segs.ndim == 3 and segs.shape[0] > 0 and segs.shape[1:] == (2, 2)


def bucketize(ms, tol: float = 1.3, slack: int = 4) -> dict:
    """Số đoạn M -> mã bucket (= M nhỏ nhất của bucket); bucket mới khi M > ceil(1,3 M_đầu) + 4."""
    out, cur = {}, None
    for m in sorted(set(ms)):
        if cur is None or m > math.ceil(tol * cur) + slack:
            cur = m
        out[m] = cur
    return out


# ── Động học, quan sát, geodesic theo lô ──────────────────────────────────
def kinematics_step(be, x, y, th32, act32):
    """RobotNavEnv.step (phần động học) cho A lượt: trả (x mới, y mới, góc float32 mới, cờ quay tại chỗ)."""
    xp = be.xp
    if KM.ROBOT == "waffle":                           # cùng hàm với RobotNavEnv._move_waffle
        return KM.waffle_step(xp, be.cos, be.sin, x, y, th32, act32)
    a = xp.clip(act32, -ONE32, ONE32)                  # oL, oR = np.clip(action, -1.0, 1.0)
    oL, oR = a[:, 0], a[:, 1]
    vL, vR = VMAX32 * oL, VMAX32 * oR
    spin = (xp.abs(vL + vR) < EPS_LIN32) & (xp.abs(vL - vR) > EPS_ANG32)
    vL2 = VMAX32 * xp.clip(oL, -ONE32, ONE32)          # step_kinematics kẹp lại (không đổi giá trị)
    vR2 = VMAX32 * xp.clip(oR, -ONE32, ONE32)
    omega = (vR2 - vL2) / L32
    v = (vR2 + vL2) / TWO32
    th = th32 + omega * DT32
    xn = x + v * be.cos(th) * DT32                     # float32 rồi float64 + float32
    yn = y + v * be.sin(th) * DT32
    return xn, yn, th, spin


def obs_tail(be, x, y, gx, gy, th):
    """rho, sin(phi), cos(phi) của RobotNavEnv._get_obs, float32 (A, 3); th là góc float64."""
    xp = be.xp
    dx = gx - x                                         # diff = goal - pos
    dy = gy - y
    dist = xp.sqrt(dx * dx + dy * dy)
    rho = xp.clip(dist / KM.ARENA_DIAGONAL, 0.0, 1.0)
    phi = be.arctan2(dy, dx) - th
    return xp.stack([rho, be.sin(phi), be.cos(phi)], axis=1).astype(xp.float32)


def geo_lookup(xp, gflat, x, y, gbase, gnx, gny, gres):
    """GeodesicField.at cho A lượt: int() cắt về 0 rồi kẹp vào lưới, trả float64."""
    ix = xp.clip(xp.trunc(x / gres).astype(xp.int64), 0, gnx - 1)
    iy = xp.clip(xp.trunc(y / gres).astype(xp.int64), 0, gny - 1)
    return gflat[gbase + iy * gnx + ix].astype(xp.float64)


# ── LiDAR + va chạm bằng xp (CPU, hoặc CuPy ở biến thể "full") ─────────────
class HostGeometry:
    """LiDAR (tia / hình quạt) và va chạm theo lô bằng xp. Lượt được xếp theo bucket số đoạn (mỗi
    bucket là một lát liền); đoạn của mỗi lượt đệm tới M lớn nhất của bucket, đoạn đệm bị che bằng
    mặt nạ. min theo trục đoạn không phụ thuộc thứ tự, nên chia lô con (lidar_chunk) không đổi bit nào."""

    def __init__(self, be, lidar, lidar_chunk: Optional[int] = None, radius: Optional[float] = None):
        self.be, self.xp = be, be.xp
        # bán kính đĩa va chạm: mặc định KM.COLL_RADIUS (robot cũ: = ROBOT_RADIUS 0,25; Waffle: 0,2077)
        self.radius = float(KM.COLL_RADIUS if radius is None else radius)
        self.mode, self.N, self.R = lidar.mode, int(lidar.num_rays), lidar.max_range
        self.ray_angles = be.dev(np.asarray(lidar.ray_angles, dtype=np.float64))
        self.sector_edges = be.dev(np.asarray(lidar.sector_edges, dtype=np.float64))
        self.lidar_chunk = int(lidar_chunk) if lidar_chunk else None
        self.buckets: list[dict] = []
        self.counts: list[int] = []

    def setup(self, infos, bucket_ids):
        """infos: MapInfo của từng lượt theo thứ tự mô phỏng; bucket_ids: mã bucket không giảm."""
        self.buckets, self.counts = [], []
        E, i = len(infos), 0
        while i < E:
            j = i
            while j < E and bucket_ids[j] == bucket_ids[i]:
                j += 1
            group = infos[i:j]
            n, Mb = j - i, max(mi.segs.shape[0] for mi in group)
            ax = np.full((n, Mb), FAR)
            ay = np.full((n, Mb), FAR)
            sx = np.zeros((n, Mb))
            sy = np.zeros((n, Mb))
            mk = np.zeros((n, Mb), dtype=bool)
            for r, mi in enumerate(group):
                s = mi.segs
                M = s.shape[0]
                ax[r, :M], ay[r, :M] = s[:, 0, 0], s[:, 0, 1]
                sx[r, :M], sy[r, :M] = s[:, 1, 0] - s[:, 0, 0], s[:, 1, 1] - s[:, 0, 1]      # s = B - A
                mk[r, :M] = True
            self.buckets.append({k: self.be.dev(v) for k, v in (("ax", ax), ("ay", ay), ("sx", sx), ("sy", sy), ("mk", mk))})
            self.counts.append(n)
            i = j

    def keep(self, keep):
        kh = self.be.host(keep)
        lo, counts = 0, []
        for b, n_b in enumerate(self.counts):
            sl = slice(lo, lo + n_b)
            lo += n_b
            kb = keep[sl]
            self.buckets[b] = {k: v[kb] for k, v in self.buckets[b].items()}
            counts.append(int(kh[sl].sum()))
        self.counts = counts

    def collide(self, x, y, thobs=None):
        """check_collision(pos, segments, self.radius) cho mọi lượt (bool (A,))."""
        xp = self.xp
        coll = xp.zeros(x.shape[0], dtype=bool)
        lo = 0
        for b, n_b in enumerate(self.counts):
            if n_b == 0:
                continue
            sl = slice(lo, lo + n_b)
            lo += n_b
            S = self.buckets[b]
            px, py = x[sl][:, None], y[sl][:, None]
            apx, apy = px - S["ax"], py - S["ay"]                           # AP = p - A
            ab_sq = xp.maximum(S["sx"] * S["sx"] + S["sy"] * S["sy"], 1e-12)
            t = xp.clip((apx * S["sx"] + apy * S["sy"]) / ab_sq, 0.0, 1.0)
            cx, cy = S["ax"] + t * S["sx"], S["ay"] + t * S["sy"]           # closest = A + t AB
            qx, qy = px - cx, py - cy
            d = xp.sqrt(qx * qx + qy * qy)
            coll[sl] = xp.any((d <= self.radius) & S["mk"], axis=1)
        return coll

    def scan(self, x, y, thobs):
        """FastLiDAR2D.scan cho mọi lượt: độ gần float32 (A, N); thobs là góc hướng float64."""
        be, xp = self.be, self.xp
        A, N = x.shape[0], self.N
        prox = xp.empty((A, N), dtype=xp.float32)
        if A == 0:
            return prox
        if self.mode == "rays":
            ang = thobs[:, None] + self.ray_angles[None, :]
            rxa, rya = be.cos(ang) * self.R, be.sin(ang) * self.R
        else:
            edges = thobs[:, None] + self.sector_edges[None, :]
            c, s = be.cos(edges), be.sin(edges)
            exa = xp.concatenate((c, c[:, :1]), axis=1)       # cạnh N là cạnh 0 (khép vòng)
            eya = xp.concatenate((s, s[:, :1]), axis=1)
        lo = 0
        for b, n_b in enumerate(self.counts):
            if n_b == 0:
                continue
            S = self.buckets[b]
            Mb = S["ax"].shape[1]
            step = self.lidar_chunk or max(1, LIDAR_ELEMS // ((N + 1) * max(Mb, 1)))
            for c0 in range(0, n_b, step):
                c1 = min(n_b, c0 + step)
                sl, bs = slice(lo + c0, lo + c1), slice(c0, c1)
                px, py = x[sl], y[sl]
                ax_, ay_, sx_, sy_, mk = S["ax"][bs], S["ay"][bs], S["sx"][bs], S["sy"][bs], S["mk"][bs]
                if self.mode == "rays":
                    prox[sl] = self._rays(rxa[sl], rya[sl], px, py, ax_, ay_, sx_, sy_, mk)
                else:
                    prox[sl] = self._sectors(exa[sl], eya[sl], px, py, ax_, ay_, sx_, sy_, mk)
            lo += n_b
        return prox

    def _rays(self, rxa, rya, px, py, ax_, ay_, sx_, sy_, mk):
        """FastLiDAR2D.cast_rays: (n, N, 1) tia x (n, 1, M) đoạn."""
        xp = self.xp
        rx, ry = rxa[:, :, None], rya[:, :, None]
        sx, sy = sx_[:, None, :], sy_[:, None, :]
        r_cross_s = rx * sy - ry * sx
        pax = (ax_ - px[:, None])[:, None, :]                     # p_to_A = A - p
        pay = (ay_ - py[:, None])[:, None, :]
        pA_cross_s = pax * sy - pay * sx
        pA_cross_r = pax * ry - pay * rx
        parallel = xp.abs(r_cross_s) < 1e-8
        safe = xp.where(parallel, 1e-8, r_cross_s)
        t = pA_cross_s / safe
        u = pA_cross_r / safe
        valid = (~parallel) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0) & mk[:, None, :]
        min_t = xp.min(xp.where(valid, t, 1.0), axis=2)
        return (1.0 - min_t).astype(xp.float32)

    def _sectors(self, exa, eya, px, py, ax_, ay_, sx, sy, mk):
        """FastLiDAR2D.cast_sectors / sector_distances: (n, N+1, 1) cạnh x (n, 1, M) đoạn."""
        xp = self.xp
        ex, ey = exa[:, :, None], eya[:, :, None]
        ax = ax_ - px[:, None]                                    # a = A - p
        ay = ay_ - py[:, None]
        ca = ex * ay[:, None, :] - ey * ax[:, None, :]
        cs = ex * sy[:, None, :] - ey * sx[:, None, :]
        pos, neg, flat = cs > 0.0, cs < 0.0, cs == 0.0
        r = -ca / xp.where(flat, 1.0, cs)
        t_lo = xp.maximum(xp.maximum(xp.where(pos[:, :-1], r[:, :-1], -xp.inf),
                                     xp.where(neg[:, 1:], r[:, 1:], -xp.inf)), 0.0)
        t_hi = xp.minimum(xp.minimum(xp.where(neg[:, :-1], r[:, :-1], xp.inf),
                                     xp.where(pos[:, 1:], r[:, 1:], xp.inf)), 1.0)
        inside = t_lo <= t_hi
        inside &= ~((flat[:, :-1] & (ca[:, :-1] < 0.0)) | (flat[:, 1:] & (ca[:, 1:] > 0.0)))
        ss = sx * sx + sy * sy
        nz = ss > 0.0
        ss1 = xp.where(nz, ss, 1.0)
        t_star = xp.where(nz, -(ax * sx + ay * sy) / ss1, 0.0)
        cr = ax * sy - ay * sx
        h2 = xp.where(nz, cr * cr / ss1, ax * ax + ay * ay)
        dt = xp.minimum(xp.maximum(t_star[:, None, :], t_lo), t_hi) - t_star[:, None, :]
        d2 = xp.where(inside & mk[:, None, :], h2[:, None, :] + ss[:, None, :] * (dt * dt), xp.inf)
        d = xp.sqrt(xp.min(d2, axis=2))
        return (1.0 - xp.minimum(d, self.R) / self.R).astype(xp.float32)


# ── Waffle: LiDAR và đĩa va chạm ở tâm thân ───────────────────────────────
class BodyCenterGeometry:
    """Bọc một hình học điểm (HostGeometry / GpuGeometry): mọi lời gọi collide / scan nhận tâm quay (x, y) và
    góc thobs (float64), đổi sang tâm thân bx = x - off cos(thobs), by = y - off sin(thobs) (cos, sin float64
    bằng be.cos / be.sin, cùng phép như RobotNavEnv.body_center) rồi gọi hình học bên trong. Cùng tư thế cho
    cùng (bx, by) tới từng bit, nên bộ đệm LiDAR của GpuGeometry (collide rồi scan cùng tư thế) vẫn trúng."""

    def __init__(self, be, inner, offset: float):
        self.be, self.inner, self.off = be, inner, float(offset)

    def center(self, x, y, thobs):
        c, s = self.be.cos(thobs), self.be.sin(thobs)
        return x - self.off * c, y - self.off * s

    def setup(self, infos, bucket_ids):
        self.inner.setup(infos, bucket_ids)

    def keep(self, keep):
        self.inner.keep(keep)

    def collide(self, x, y, thobs):
        bx, by = self.center(x, y, thobs)
        return self.inner.collide(bx, by, thobs)

    def scan(self, x, y, thobs):
        bx, by = self.center(x, y, thobs)
        return self.inner.scan(bx, by, thobs)


def robot_geometry(be, inner):
    """Hình học theo robot đang chạy: Waffle bọc BodyCenterGeometry; robot cũ trả nguyên `inner`."""
    if KM.ROBOT == "waffle":
        return BodyCenterGeometry(be, inner, KM.BODY_OFFSET)
    return inner


def make_host_geometry(be, lidar, lidar_chunk: Optional[int] = None):
    """HostGeometry với bán kính va chạm của robot đang chạy, bọc theo robot_geometry."""
    return robot_geometry(be, HostGeometry(be, lidar, lidar_chunk, radius=KM.COLL_RADIUS))


# ── Vòng lockstep ──────────────────────────────────────────────────────────
def simulate(be, episodes: list[Episode], policy_factory: Callable, geom, maps: Optional[dict] = None) -> list[dict]:
    """Chạy mọi lượt tới khi kết thúc; trả dict kết quả của từng lượt theo đúng thứ tự `episodes`,
    cùng khóa, cùng giá trị và cùng kiểu Python với run_episode.
    policy_factory(ep_task) dựng bộ điều khiển theo lô cho các lượt (theo thứ tự mô phỏng)."""
    xp = be.xp
    E = len(episodes)
    if E == 0:
        return []
    maps = {} if maps is None else maps
    keys = [(ep.map_path, ep.progress, ep.geo_cache) for ep in episodes]
    for k in keys:
        if k not in maps:
            maps[k] = map_info(*k)
    Ms = [maps[k].segs.shape[0] for k in keys]
    bmap = bucketize(Ms)
    order = sorted(range(E), key=lambda e: (bmap[Ms[e]], e))      # mỗi bucket là một lát liền
    infos = [maps[keys[e]] for e in order]
    geom.setup(infos, [bmap[Ms[e]] for e in order])

    # trường geodesic: mọi lưới nối vào một mảng phẳng float64 (lưới float32 đổi sang float64 không làm tròn;
    # lưới float64, ví dụ file .npy do mã khác ghi, giữ nguyên giá trị như GeodesicField.at)
    grids, gbase, gshape, off = [], {}, {}, 0
    for k in dict.fromkeys(keys):
        mi = maps[k]
        if mi.geo is not None:
            g = np.ascontiguousarray(mi.geo.grid, dtype=np.float64)
            gbase[k], gshape[k] = off, g.shape
            grids.append(g.ravel())
            off += g.size
    gflat = be.dev(np.concatenate(grids) if grids else np.zeros(1))
    any_geo = bool(grids)
    okeys = [keys[e] for e in order]

    st = {
        "x": np.array([mi.start[0] for mi in infos]),
        "y": np.array([mi.start[1] for mi in infos]),
        "gx": np.array([mi.goal[0] for mi in infos]),
        "gy": np.array([mi.goal[1] for mi in infos]),
        "th32": np.array([mi.h0 for mi in infos]).astype(np.float32),   # bước 1 làm tròn h0 về float32 trước khi cộng
        "thobs": np.array([mi.h0 for mi in infos], dtype=np.float64),   # quan sát lúc reset dùng h0 float64
        "steps": np.zeros(E, np.int64),
        "spin": np.zeros(E, np.int64),
        "dmin": np.array([mi.d0 for mi in infos], dtype=np.float64),
        "gmin": np.array([mi.g0 if mi.geo is not None else 0.0 for mi in infos], dtype=np.float64),
        "hasgeo": np.array([mi.geo is not None for mi in infos], dtype=bool),
        "gbase": np.array([gbase.get(k, 0) for k in okeys], np.int64),
        "gny": np.array([gshape[k][0] if k in gshape else 1 for k in okeys], np.int64),
        "gnx": np.array([gshape[k][1] if k in gshape else 1 for k in okeys], np.int64),
        "gres": np.array([float(maps[k].geo.res) if k in gshape else 1.0 for k in okeys], dtype=np.float64),
        "id": np.array(order, np.int64),
    }
    st = {k: be.dev(v) for k, v in st.items()}
    policy = policy_factory([episodes[e].task for e in order])
    seglen = xp.zeros((E, KM.MAX_STEPS))
    out = {k: np.zeros(E, t) for k, t in (("succ", bool), ("coll", bool), ("steps", np.int64), ("spin", np.int64),
                                           ("x", np.float64), ("y", np.float64), ("dist", np.float64),
                                           ("dmin", np.float64), ("gmin", np.float64))}

    def get_obs():
        prox = geom.scan(st["x"], st["y"], st["thobs"])
        tail = obs_tail(be, st["x"], st["y"], st["gx"], st["gy"], st["thobs"])
        return xp.concatenate([prox, tail], axis=1)

    obs = get_obs()
    for _ in range(KM.MAX_STEPS):
        act = policy.step(obs).astype(xp.float32)                 # np.asarray(policy.activate(...), float32)
        st["steps"] = st["steps"] + 1
        xn, yn, th, spin = kinematics_step(be, st["x"], st["y"], st["th32"], act)
        st["spin"] = st["spin"] + spin
        ddx, ddy = xn - st["x"], yn - st["y"]
        seglen[st["id"], st["steps"] - 1] = xp.sqrt(ddx * ddx + ddy * ddy)   # norm(diff(traj)) từng bước
        st["x"], st["y"], st["th32"] = xn, yn, th
        st["thobs"] = th.astype(xp.float64)
        coll = geom.collide(xn, yn, st["thobs"])
        gdx, gdy = xn - st["gx"], yn - st["gy"]                     # pos - goal
        dist = xp.sqrt(gdx * gdx + gdy * gdy)
        st["dmin"] = xp.where(dist < st["dmin"], dist, st["dmin"])  # min(d_min, dist) của Python
        reached = dist <= KM.GOAL_RADIUS
        if any_geo:
            g = geo_lookup(xp, gflat, xn, yn, st["gbase"], st["gnx"], st["gny"], st["gres"])
            st["gmin"] = xp.where(st["hasgeo"] & (g < st["gmin"]), g, st["gmin"])
        done = coll | reached | (st["steps"] >= KM.MAX_STEPS)
        if be.any(done):
            ids = be.host(st["id"][done])
            out["succ"][ids] = be.host(reached[done])
            out["coll"][ids] = be.host((coll & ~reached)[done])     # tới đích cùng bước va chạm = thành công
            for k in ("steps", "spin", "x", "y", "dmin", "gmin"):
                out[k][ids] = be.host(st[k][done])
            out["dist"][ids] = be.host(dist[done])                  # dist_to_goal của bước cuối
            keep = ~done
            st = {k: v[keep] for k, v in st.items()}
            policy.keep(keep)
            geom.keep(keep)
            if st["x"].shape[0] == 0:
                break
        obs = get_obs()

    seglen_h = be.host(seglen)
    results = []
    for e, ep in enumerate(episodes):
        mi = maps[keys[e]]
        T = int(out["steps"][e])
        # info của bước cuối như RobotNavEnv.step (cùng thứ tự khóa), trừ "trajectory"
        info: dict[str, Any] = {"is_success": bool(out["succ"][e]), "is_collision": bool(out["coll"][e]),
                                "steps": T, "final_pos": (float(out["x"][e]), float(out["y"][e])),
                                "spin_steps": int(out["spin"][e]), "dist_to_goal": float(out["dist"][e]),
                                "min_distance_to_goal": float(out["dmin"][e]), "d0": mi.d0}
        if mi.geo is not None:
            info["g0"], info["g_min"] = mi.g0, float(out["gmin"][e])
        length = float(np.sum(seglen_h[e, :T])) if T >= 1 else 0.0
        results.append({                                          # như run_episode
            "fitness": compute_episode_fitness(info, progress=ep.progress),
            "success": bool(info["is_success"]),
            "collision": bool(info["is_collision"]),
            "steps": int(info["steps"]),
            "final_pos": (float(info["final_pos"][0]), float(info["final_pos"][1])),
            "length": length,
            "l_opt": mi.l_opt,
        })
    return results
