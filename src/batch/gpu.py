"""
LiDAR và va chạm trên GPU (biến thể lai, trùng từng bit)
=========================================================
Mọi trạng thái, bộ điều khiển, động học, tiến độ và phần đuôi của quan sát vẫn ở host bằng đúng mã
NumPy của src/batch/sim.py; chỉ phần nặng O(lượt x hướng x đoạn) chạy trên GPU, gộp trong hai kernel
CUDA (cupy.RawModule):
- "rays" (một luồng cho mỗi (lượt, tia)) hoặc "sector" (một luồng cho mỗi (lượt, hình quạt i), cạnh i
  và (i + 1) mod N): lặp qua các đoạn của bản đồ của lượt đó, tính đúng chuỗi phép của
  FastLiDAR2D.cast_rays / sector_distances (từng tích làm tròn riêng), lấy min dồn (chính xác, không
  phụ thuộc thứ tự);
- "coll" (một luồng cho mỗi lượt): đúng chuỗi phép của collision._point_to_segments_dist.
Biên dịch với --fmad=false (không gộp nhân-cộng thành FMA), nên mọi phép + - x / sqrt float64 làm
tròn đúng như NumPy. Hướng tia / cạnh (np.cos, np.sin float64) tính ở host như bản gốc rồi gói cùng
x, y vào một mảng tải lên; mỗi bước 1 lần tải lên, 2 lần tải về. Đoạn của mỗi bản đồ để dạng CSR
(không nhân bản theo lượt, không đệm). Kernel không #include gì.

startup_check() so kernel với FastLiDAR2D.scan và check_collision trên các tư thế ngẫu nhiên (bản đồ
tổng hợp có đoạn suy biến); lệch một bit là từ chối GPU (BatchEvaluator quay về đường mặc định).

Waffle (NEAT_ROBOT=waffle): kernel không đổi. Kernel nhận đúng điểm cần quét / kiểm (X[e], X[n + e]),
nên BodyCenterGeometry (src/batch/sim.py) truyền tâm thân tính ở host; bán kính đĩa va chạm là tham số
robot_radius (mặc định KM.COLL_RADIUS: 0,25 với robot cũ, 0,2077 với Waffle); cạnh quạt đọc từ
lidar.sector_edges (bố cục e32 không đều dùng được vì kernel lấy cạnh i và (i + 1) mod N từ mảng).
startup_check kiểm cả bố cục quạt và bán kính của robot đang chạy.
Lỗi CUDA lúc chạy (hết bộ nhớ GPU khi nhiều job cùng dùng, lỗi chạy kernel) được báo thành GpuUnavailable
(job dừng có lý do, không ra số sai).
"""

from __future__ import annotations

import numpy as np

from src.batch.backend import GpuUnavailable, import_cupy

_SRC = r'''
#define NP_MAX(a, b) ((a) >= (b) ? (a) : (b))
#define NP_MIN(a, b) ((a) <= (b) ? (a) : (b))
extern "C" __global__
void sector(const double* __restrict__ X, const int n, const int N,
            const int* __restrict__ s0, const int* __restrict__ sn,
            const double* __restrict__ ax_, const double* __restrict__ ay_,
            const double* __restrict__ sx_, const double* __restrict__ sy_,
            const double R, float* __restrict__ prox)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n * N) return;
    int e = tid / N, i = tid - e * N;
    const int j = (i + 1 == N) ? 0 : i + 1;
    const double pxe = X[e], pye = X[n + e];
    const double ex0 = X[2 * n + e * N + i], ey0 = X[2 * n + n * N + e * N + i];
    const double ex1 = X[2 * n + e * N + j], ey1 = X[2 * n + n * N + e * N + j];
    const double INF = __longlong_as_double(0x7ff0000000000000LL);
    double best = INF;
    const int a = s0[e], b = a + sn[e];
    for (int k = a; k < b; ++k) {
        const double ax = ax_[k] - pxe, ay = ay_[k] - pye;
        const double sx = sx_[k], sy = sy_[k];
        const double ca0 = ex0 * ay - ey0 * ax, cs0 = ex0 * sy - ey0 * sx;
        const double ca1 = ex1 * ay - ey1 * ax, cs1 = ex1 * sy - ey1 * sx;
        const bool pos0 = cs0 > 0.0, neg0 = cs0 < 0.0, flat0 = cs0 == 0.0;
        const bool pos1 = cs1 > 0.0, neg1 = cs1 < 0.0, flat1 = cs1 == 0.0;
        const double r0 = -ca0 / (flat0 ? 1.0 : cs0);
        const double r1 = -ca1 / (flat1 ? 1.0 : cs1);
        const double t_lo = NP_MAX(NP_MAX(pos0 ? r0 : -INF, neg1 ? r1 : -INF), 0.0);
        const double t_hi = NP_MIN(NP_MIN(neg0 ? r0 : INF, pos1 ? r1 : INF), 1.0);
        bool inside = t_lo <= t_hi;
        inside = inside && !((flat0 && (ca0 < 0.0)) || (flat1 && (ca1 > 0.0)));
        const double ss = sx * sx + sy * sy;
        const bool nz = ss > 0.0;
        const double ss1 = nz ? ss : 1.0;
        const double t_star = nz ? -(ax * sx + ay * sy) / ss1 : 0.0;
        const double cr = ax * sy - ay * sx;
        const double h2 = nz ? cr * cr / ss1 : ax * ax + ay * ay;
        const double dt = NP_MIN(NP_MAX(t_star, t_lo), t_hi) - t_star;
        const double d2 = inside ? h2 + ss * (dt * dt) : INF;
        best = d2 < best ? d2 : best;
    }
    const double d = sqrt(best);
    prox[tid] = (float)(1.0 - NP_MIN(d, R) / R);
}
extern "C" __global__
void rays(const double* __restrict__ X, const int n, const int N,
          const int* __restrict__ s0, const int* __restrict__ sn,
          const double* __restrict__ ax, const double* __restrict__ ay,
          const double* __restrict__ sx, const double* __restrict__ sy,
          float* __restrict__ prox)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n * N) return;
    int e = tid / N;
    const double pxe = X[e], pye = X[n + e];
    const double rxx = X[2 * n + tid], ryy = X[2 * n + n * N + tid];
    double best = 1.0;
    const int a = s0[e], b = a + sn[e];
    for (int k = a; k < b; ++k) {
        const double sxx = sx[k], syy = sy[k];
        const double rcs = rxx * syy - ryy * sxx;
        const double pax = ax[k] - pxe, pay = ay[k] - pye;
        const double pcs = pax * syy - pay * sxx;
        const double pcr = pax * ryy - pay * rxx;
        const bool par = fabs(rcs) < 1e-8;
        const double den = par ? 1e-8 : rcs;
        const double t = pcs / den, u = pcr / den;
        const bool ok = (!par) && (t >= 0.0) && (t <= 1.0) && (u >= 0.0) && (u <= 1.0);
        const double tv = ok ? t : 1.0;
        best = tv < best ? tv : best;
    }
    prox[tid] = (float)(1.0 - best);
}
extern "C" __global__
void coll(const double* __restrict__ X, const int n,
          const int* __restrict__ s0, const int* __restrict__ sn,
          const double* __restrict__ ax, const double* __restrict__ ay,
          const double* __restrict__ sx, const double* __restrict__ sy,
          const double radius, bool* __restrict__ out)
{
    int e = blockDim.x * blockIdx.x + threadIdx.x;
    if (e >= n) return;
    const double pxe = X[e], pye = X[n + e];
    bool hit = false;
    const int a = s0[e], b = a + sn[e];
    for (int k = a; k < b; ++k) {
        const double sxx = sx[k], syy = sy[k];
        const double apx = pxe - ax[k], apy = pye - ay[k];
        double ab = sxx * sxx + syy * syy;
        ab = ab >= 1e-12 ? ab : 1e-12;
        double t = (apx * sxx + apy * syy) / ab;
        t = t < 0.0 ? 0.0 : t;
        t = t > 1.0 ? 1.0 : t;
        const double cx = ax[k] + t * sxx, cy = ay[k] + t * syy;
        const double qx = pxe - cx, qy = pye - cy;
        hit = hit || (sqrt(qx * qx + qy * qy) <= radius);
    }
    out[e] = hit;
}
'''

_MODULE = None
_THREADS = 128


def _module():
    """RawModule biên dịch một lần mỗi tiến trình (--fmad=false: không gộp FMA)."""
    global _MODULE
    if _MODULE is None:
        cp = import_cupy()
        try:
            mod = cp.RawModule(code=_SRC, options=("--fmad=false",))
            fns = {name: mod.get_function(name) for name in ("rays", "sector", "coll")}
        except Exception as e:  # noqa: BLE001  (CompileException, thiếu NVRTC, ...)
            raise GpuUnavailable(f"không biên dịch được kernel CUDA: {e!r}") from e
        _MODULE = (cp, fns)
    return _MODULE


class GpuGeometry:
    """Cùng giao diện với sim.HostGeometry (setup / collide / scan / keep), mảng vào ra là NumPy ở host."""

    def __init__(self, lidar, robot_radius: float | None = None):
        from src.env.kinematics import COLL_RADIUS
        self.cp, fns = _module()
        self.k_rays, self.k_sector, self.k_coll = fns["rays"], fns["sector"], fns["coll"]
        self.mode, self.N, self.R = lidar.mode, int(lidar.num_rays), float(lidar.max_range)
        self.ang = np.asarray(lidar.ray_angles if self.mode == "rays" else lidar.sector_edges, dtype=np.float64)
        self.radius = float(COLL_RADIUS if robot_radius is None else robot_radius)
        self._prox = None
        self._pose = None                                   # (x, y, thobs) của LiDAR giữ trong _prox

    def setup(self, infos, bucket_ids=None):
        """infos: MapInfo của từng lượt theo thứ tự mô phỏng (đoạn của mỗi bản đồ chỉ tải lên một lần)."""
        cp = self.cp
        start: dict = {}
        cols: tuple[list, list, list, list] = ([], [], [], [])
        off = 0
        for mi in infos:
            if id(mi) in start:
                continue
            s = mi.segs
            start[id(mi)] = (off, s.shape[0])
            off += s.shape[0]
            cols[0].append(s[:, 0, 0])
            cols[1].append(s[:, 0, 1])
            cols[2].append(s[:, 1, 0] - s[:, 0, 0])          # s = B - A
            cols[3].append(s[:, 1, 1] - s[:, 0, 1])
        self.ax, self.ay, self.sx, self.sy = (cp.asarray(np.ascontiguousarray(np.concatenate(c) if c else np.zeros(1)))
                                              for c in cols)
        self.s0 = np.array([start[id(mi)][0] for mi in infos], np.int32)
        self.sn = np.array([start[id(mi)][1] for mi in infos], np.int32)
        self._upload_idx()
        self._prox = self._pose = None

    def _upload_idx(self):
        self.s0d, self.snd = self.cp.asarray(self.s0), self.cp.asarray(self.sn)

    def keep(self, keep):
        keep = np.asarray(keep)
        self.s0, self.sn = self.s0[keep], self.sn[keep]
        self._upload_idx()
        if self._prox is not None:
            self._prox = self._prox[keep]
            self._pose = tuple(a[keep] for a in self._pose)

    def _directions(self, thobs):
        ang = thobs[:, None] + self.ang[None, :]
        if self.mode == "rays":
            return np.cos(ang) * self.R, np.sin(ang) * self.R      # như cast_rays
        return np.cos(ang), np.sin(ang)                            # cạnh hình quạt như sector_distances

    def _run(self, x, y, thobs, want_coll: bool):
        try:
            return self._launch(x, y, thobs, want_coll)
        except GpuUnavailable:
            raise
        except Exception as e:  # noqa: BLE001  (OutOfMemoryError, CUDARuntimeError, CUDADriverError, ...)
            mod = type(e).__module__ or ""
            if not (mod.startswith("cupy") or mod.startswith("cupy_backends")):
                raise
            raise GpuUnavailable(f"lỗi CUDA khi chạy kernel LiDAR / va chạm: {e!r}. Nếu nhiều job cùng dùng GPU, "
                                 "chỉ để một job dùng --backend gpu, các job khác dùng --backend batch") from e

    def _launch(self, x, y, thobs, want_coll: bool):
        cp = self.cp
        n, N = int(x.shape[0]), self.N
        if n == 0:
            return np.zeros((0, N), np.float32), np.zeros(0, bool)
        rx, ry = self._directions(np.asarray(thobs, dtype=np.float64))
        X = cp.asarray(np.concatenate([np.asarray(x, np.float64), np.asarray(y, np.float64), rx.ravel(), ry.ravel()]))
        prox = cp.empty(n * N, dtype=cp.float32)
        grid = ((n * N + _THREADS - 1) // _THREADS,)
        if self.mode == "rays":
            self.k_rays(grid, (_THREADS,), (X, np.int32(n), np.int32(N), self.s0d, self.snd,
                                            self.ax, self.ay, self.sx, self.sy, prox))
        else:
            self.k_sector(grid, (_THREADS,), (X, np.int32(n), np.int32(N), self.s0d, self.snd,
                                              self.ax, self.ay, self.sx, self.sy, np.float64(self.R), prox))
        coll = None
        if want_coll:
            cd = cp.empty(n, dtype=cp.bool_)
            self.k_coll(((n + _THREADS - 1) // _THREADS,), (_THREADS,),
                        (X, np.int32(n), self.s0d, self.snd, self.ax, self.ay, self.sx, self.sy,
                         np.float64(self.radius), cd))
            coll = cd.get()
        return prox.get().reshape(n, N), coll

    def collide(self, x, y, thobs):
        """Va chạm tại tư thế mới; LiDAR ở cùng tư thế được tính luôn và giữ cho scan() kế tiếp."""
        pose = tuple(np.array(a, dtype=np.float64, copy=True) for a in (x, y, thobs))
        prox, coll = self._run(pose[0], pose[1], pose[2], True)
        self._prox, self._pose = prox, pose
        return coll

    @staticmethod
    def _same_pose(a, b) -> bool:
        """Cùng tư thế tới từng bit (so dạng uint64, nên -0.0 khác 0.0)."""
        a = np.asarray(a)
        return (a.dtype == np.float64 and a.shape == b.shape
                and np.array_equal(np.ascontiguousarray(a).view(np.uint64), b.view(np.uint64)))

    def scan(self, x, y, thobs):
        """LiDAR tại (x, y, thobs): dùng kết quả của collide() trước đó chỉ khi đúng cùng tư thế (sau keep)."""
        prox, pose = self._prox, self._pose
        self._prox = self._pose = None
        if prox is not None and all(self._same_pose(a, b) for a, b in zip((x, y, thobs), pose)):
            return prox
        return self._run(x, y, thobs, False)[0]


def startup_check(lidar, n_poses: int = 96, seed: int = 2024) -> list[str]:
    """So kernel với FastLiDAR2D.scan / check_collision (cả hai kiểu cảm biến) trên bản đồ tổng hợp;
    trả danh sách chỗ lệch (rỗng = trùng từng bit)."""
    from src.batch.sim import MapInfo
    from src.env.collision import check_collision
    from src.env.kinematics import COLL_RADIUS
    from src.env.lidar import FastLiDAR2D
    from src.batch.numerics import synthetic_maps, random_poses
    problems = []
    rng = np.random.default_rng(seed)
    layout = getattr(lidar, "layout", "") if lidar.mode == "sector" else ""   # bố cục quạt của robot đang chạy
    modes = [("rays", lidar.num_rays, "")] + ([("sector", lidar.num_rays, layout)] if lidar.num_rays >= 2 else [])
    for mode, n, lay in modes:
        lid = FastLiDAR2D(n, lidar.max_range, mode, lay)
        geom = GpuGeometry(lid, COLL_RADIUS)
        for segs in synthetic_maps(rng):
            mi = MapInfo("check", {}, segs, np.zeros(2), np.zeros(2), 0.0, 0.0, None, None, 0.0)
            x, y, th_py, th32 = random_poses(rng, n_poses, segs)
            geom.setup([mi] * n_poses)
            for th_list, thobs in ((th_py, np.array(th_py, dtype=np.float64)),
                                   (list(th32), th32.astype(np.float64))):
                coll = geom.collide(x, y, thobs)
                prox = geom.scan(x, y, thobs)
                ref_p = np.array([lid.scan(np.array([x[i], y[i]]), th_list[i], segs) for i in range(n_poses)])
                ref_c = np.array([check_collision(np.array([x[i], y[i]]), segs, COLL_RADIUS) for i in range(n_poses)])
                if not np.array_equal(prox, ref_p) or prox.dtype != ref_p.dtype:
                    problems.append(f"GPU {mode}:{n} lệch FastLiDAR2D.scan ở {int((prox != ref_p).any(axis=1).sum())} tư thế")
                if not np.array_equal(coll, ref_c):
                    problems.append(f"GPU va chạm lệch check_collision ở {int((coll != ref_c).sum())} tư thế")
    return problems
