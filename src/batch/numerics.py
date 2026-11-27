"""
Tự kiểm của chế độ exact (chạy lúc dựng BatchEvaluator, khoảng 1 giây)
======================================================================
Chạy theo lô chỉ trùng từng bit với đường từng lượt khi vài tính chất của Python / NumPy / BLAS trên
máy đang chạy đúng như đã kiểm khi thiết kế:
- sum() của Python là tổng bù Neumaier (Python >= 3.12) và bản sao neumaier_sum cho cùng kết quả;
- np.linalg.norm của vectơ 2 phần tử (BLAS) = sqrt(x0*x0 + x1*x1);
- np.cos / np.sin / np.arctan2 (float64) và np.cos / np.sin (float32) trên mảng cho cùng kết quả với
  lời gọi vô hướng (góc trong [-140; 140], phủ mọi góc LiDAR của một lượt 500 bước, thêm các điểm sát k pi/2);
- quy tắc ép kiểu NEP 50 (số Python "yếu") trong step_kinematics và is_spinning;
- np.matmul xếp chồng + np.tanh cho cùng kết quả với FixedMLP.activate (x @ W qua BLAS gemv);
- np.sum trên lát 1-D (độ dài quãng đường) như trên mảng mới.
Ngoài ra so trực tiếp các hàm theo lô (LiDAR tia và hình quạt, va chạm, quan sát, tra geodesic với lưới
float32 và float64, động học) với hàm gốc (FastLiDAR2D.scan, check_collision, RobotNavEnv._get_obs,
GeodesicField.at, step_kinematics) trên bản đồ tổng hợp, và so TRỌN LƯỢT (episode_check): PAR.run_episode
với simulate theo lô cho FixedMLP, APF và mạng NEAT hồi quy dựng tay, tiến độ geodesic và euclid, có lượt tới
đích, va chạm và hết giờ. Phép so trọn lượt bắt được việc sửa RobotNavEnv.step, compute_episode_fitness hay
run_episode (đọc thêm khóa info, đổi công thức, thêm khóa kết quả) mà bản theo lô chưa chép theo.
self_check() trả danh sách chỗ lệch; rỗng là đạt.

Waffle (NEAT_ROBOT=waffle): khoảng góc kiểm mở tới [-200; 200] (quay tối đa 1,82 rad/s x 100 s), động
học so kinematics_step theo lô với waffle_step từng lượt (lô 1), hình học dùng bán kính KM.COLL_RADIUS và
bố cục quạt đang chạy, quan sát và trọn lượt đi qua BodyCenterGeometry (tâm thân); APF chỉ được so khi
apf_unsupported_reason() là None (Waffle: quạt đều 12). Robot cũ giữ nguyên mọi phép kiểm.
"""

from __future__ import annotations

import math
import struct

import numpy as np

_CACHE: dict = {}


def synthetic_maps(rng, n_random: int = 24):
    """Ba bản đồ tổng hợp: tường biên + đoạn ngẫu nhiên + đoạn suy biến (A = B) + đoạn ngang / dọc; hai bản đồ
    đầu (31 và 37 đoạn) chung một bucket nên có đoạn đệm, bản đồ thứ ba (71 đoạn) ở bucket khác."""
    walls = [[[0.0, 0.0], [10.0, 0.0]], [[10.0, 0.0], [10.0, 10.0]], [[10.0, 10.0], [0.0, 10.0]], [[0.0, 10.0], [0.0, 0.0]]]
    out = []
    for extra in (0, 6, 40):
        segs = list(walls)
        for _ in range(n_random + extra):
            a = rng.uniform(0.5, 9.5, 2)
            b = a + rng.normal(0.0, 1.2, 2)
            segs.append([a.tolist(), b.tolist()])
        p = rng.uniform(1.0, 9.0, 2).tolist()
        segs.append([p, p])                                    # đoạn suy biến
        segs.append([[2.0, 5.0], [8.0, 5.0]])                  # ngang
        segs.append([[5.0, 1.5], [5.0, 3.5]])                  # dọc
        out.append(np.array(segs, dtype=np.float64))
    return out


def random_poses(rng, n: int, segs):
    """Tư thế ngẫu nhiên trong sân (một phần sát một đoạn để có ca va chạm): x, y float64, góc dạng
    float Python (lúc reset) và float32 (sau bước 1)."""
    x = rng.uniform(0.3, 9.7, n)
    y = rng.uniform(0.3, 9.7, n)
    for i in range(0, n, 4):                                   # sát một đoạn ngẫu nhiên
        s = segs[rng.integers(len(segs))]
        t = rng.uniform(0.0, 1.0)
        q = s[0] + t * (s[1] - s[0]) + rng.normal(0.0, 0.2, 2)
        x[i], y[i] = q
    th_py = [float(v) for v in rng.uniform(-7.0, 7.0, n)]
    th_py[0], th_py[1] = 0.0, 1.5707963267948966              # hướng xuất phát thật (bản đồ tự sinh, BARN)
    th32 = rng.uniform(-7.0, 7.0, n).astype(np.float32)
    return x, y, th_py, th32


def _neumaier_check(rng, problems):
    from src.batch.policies import neumaier_sum
    rows, K = 400, 24
    P = rng.uniform(-3.0, 3.0, (rows, K)) * rng.choice([1e-3, 1.0, 1e3, 1e16], size=(rows, K))
    P[::7, 0] = -0.0
    P[::11, :] = 0.0
    P[5::11, 1] = -0.0
    fanin = rng.integers(0, K + 1, rows)
    got = neumaier_sum(np, P, fanin)
    ref = np.array([float(sum(P[r, :fanin[r]].tolist())) for r in range(rows)])
    same = (got == ref) & (np.signbit(got) == np.signbit(ref))
    if not same.all():
        problems.append(f"sum() khác bản sao Neumaier ở {int((~same).sum())}/{rows} hàng")


def _scalar_vs_array(rng, problems):
    v = rng.normal(size=(2000, 2)) * rng.choice([1e-3, 1.0, 7.0], size=(2000, 1))
    ref = np.array([np.linalg.norm(r) for r in v])
    if not np.array_equal(ref, np.sqrt(v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1])):
        problems.append("np.linalg.norm(vectơ 2 phần tử) khác sqrt(x0*x0 + x1*x1)")
    if not np.array_equal(np.linalg.norm(v, axis=1), np.sqrt(v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1])):
        problems.append("np.linalg.norm(axis=1) khác sqrt(x0*x0 + x1*x1)")
    # góc LiDAR thật = thobs + ray_angles / sector_edges: |thobs - h0| <= 125 (|omega| <= 2,5 rad/s, 500 bước của
    # 0,1 s), h0 <= pi/2, góc tia < 2 pi, tức trong khoảng [-126; 133]; kiểm [-140; 140] và các điểm sát k pi/2.
    # Waffle: |omega| <= 1,82 rad/s, 1.000 bước, nên |thobs - h0| <= 182 và góc cạnh quạt trong [-186; 191];
    # kiểm [-200; 200] và k pi/2 với |k| <= 127
    from src.env import kinematics as KM
    lim, kmax = (200.0, 128) if KM.ROBOT == "waffle" else (140.0, 90)
    kq = np.arange(1 - kmax, kmax) * (np.pi / 2)
    t = rng.permutation(np.concatenate([rng.uniform(-lim, lim, 3000), kq, np.nextafter(kq, np.inf),
                                        np.nextafter(kq, -np.inf)]))
    m = t[:2400].reshape(200, 12)
    for name, fn in (("cos", np.cos), ("sin", np.sin)):
        if not np.array_equal(np.array([fn(np.float64(a)) for a in t]), fn(t)):
            problems.append(f"np.{name} float64: mảng khác vô hướng")
        if not np.array_equal(fn(m), np.array([fn(r) for r in m])):
            problems.append(f"np.{name} float64: mảng 2 chiều khác từng hàng")
        t32 = t.astype(np.float32)
        if not np.array_equal(np.array([fn(a) for a in t32], dtype=np.float32), fn(t32)):
            problems.append(f"np.{name} float32: mảng khác vô hướng")
    a, b = rng.normal(size=2000), rng.normal(size=2000)
    if not np.array_equal(np.array([np.arctan2(p, q) for p, q in zip(a, b)]), np.arctan2(a, b)):
        problems.append("np.arctan2: mảng khác vô hướng")
    s = rng.uniform(0.0, 0.06, (40, 500))
    if not all(float(np.sum(s[i, :T])) == float(np.sum(np.array(s[i, :T].tolist()))) for i, T in enumerate(range(1, 500, 13))):
        problems.append("np.sum trên lát 1-D khác trên mảng mới")
    h = rng.uniform(-4.0, 4.0, 500)
    inc = rng.uniform(-0.25, 0.25, 500).astype(np.float32)
    a32 = np.array([float(p) + q for p, q in zip(h, inc)])
    if a32.dtype != np.float32 or not np.array_equal(a32, h.astype(np.float32) + inc):
        problems.append("float Python + np.float32 không cho float32(float32(h) + inc) (quy tắc NEP 50)")


def _kinematics_check(rng, problems):
    from src.batch.backend import Backend
    from src.batch.sim import kinematics_step
    from src.env import kinematics as KM
    be = Backend("numpy")
    n = 300
    act = rng.uniform(-1.3, 1.3, (n, 2)).astype(np.float32)
    act[:6] = np.array([[0.4, 0.0], [0.0, 0.4], [0.05, -0.05], [0.2, -0.2], [0.21, -0.2], [1.0, 1.0]], dtype=np.float32)
    x = rng.uniform(0.0, 10.0, n)
    y = rng.uniform(0.0, 10.0, n)
    th_py = [float(v) for v in rng.uniform(-7.0, 7.0, n)]
    th32 = rng.uniform(-7.0, 7.0, n).astype(np.float32)
    if KM.ROBOT == "waffle":
        # theo lô (A lượt) so với waffle_step của RobotNavEnv (lô 1 lượt): (v, w) có o1 < 0, |o| > 1, bánh bão hòa,
        # 0 và -0.0, ngưỡng xoay tại chỗ
        act[6:16] = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.1], [-0.5, 0.3], [0.0, 0.0], [-0.0, -0.0],
                              [1.3, -1.3], [0.0961538, 0.0], [0.05, -0.3], [0.5, 0.95]], dtype=np.float32)
        for th_arr in (np.array(th_py).astype(np.float32), th32):
            xn, yn, thn, spin = kinematics_step(be, x, y, th_arr, act)
            for i in range(n):
                xs, ys, ts, ss = KM.waffle_step(np, np.cos, np.sin, x[i:i + 1], y[i:i + 1], th_arr[i:i + 1], act[i:i + 1])
                if (not same_bits(float(xs[0]), float(xn[i])) or not same_bits(float(ys[0]), float(yn[i]))
                        or ts[0].tobytes() != thn[i].tobytes() or type(ts[0]) is not type(thn[i])
                        or thn.dtype != np.float32 or bool(ss[0]) != bool(spin[i])):
                    problems.append(f"động học Waffle theo lô khác waffle_step từng lượt (hành động {act[i].tolist()})")
                    return
        return
    for th_list, th_arr in ((th_py, np.array(th_py).astype(np.float32)), (list(th32), th32)):
        xn, yn, thn, spin = kinematics_step(be, x, y, th_arr, act)
        for i in range(n):
            oL, oR = np.clip(act[i], -1.0, 1.0)
            vL, vR = KM.V_MAX * oL, KM.V_MAX * oR
            xs, ys, ts = KM.step_kinematics(x[i], y[i], th_list[i], oL, oR)
            if (xs != xn[i] or ys != yn[i] or ts != thn[i] or type(ts) is not type(thn[i])
                    or bool(KM.is_spinning(vL, vR)) != bool(spin[i])):
                problems.append(f"động học theo lô khác step_kinematics / is_spinning (hành động {act[i].tolist()})")
                return


def _geometry_check(rng, problems):
    from src.batch.backend import Backend
    from src.batch.sim import HostGeometry, MapInfo, bucketize
    from src.env import kinematics as KM
    from src.env.collision import check_collision
    from src.env.lidar import FastLiDAR2D
    be = Backend("numpy")
    maps = synthetic_maps(rng)
    infos = [MapInfo("check", {}, s, np.zeros(2), np.zeros(2), 0.0, 0.0, None, None, 0.0) for s in maps]
    n = 48
    poses = [random_poses(rng, n, s) for s in maps]
    xs = np.concatenate([p[0] for p in poses])
    ys = np.concatenate([p[1] for p in poses])
    Ms = [mi.segs.shape[0] for mi in infos for _ in range(n)]
    bm = bucketize(Ms)
    ep_infos = [mi for mi in infos for _ in range(n)]
    modes = [("rays", KM.LIDAR_NUM_RAYS, ""), ("rays", 12, "")] \
        + ([("sector", KM.LIDAR_NUM_RAYS, "")] if KM.LIDAR_NUM_RAYS >= 2 else []) + [("sector", 12, "")]
    if KM.SECTOR_LAYOUT:                                       # bố cục quạt không đều đang chạy (e32)
        modes.append(("sector", KM.LIDAR_NUM_RAYS, KM.SECTOR_LAYOUT))
    # bán kính đĩa va chạm của robot đang chạy (robot cũ: ROBOT_RADIUS = mặc định của check_collision)
    ref_c = np.array([check_collision(np.array([xs[i], ys[i]]), ep_infos[i].segs, KM.COLL_RADIUS)
                      for i in range(len(xs))])
    for mode, N, layout in dict.fromkeys(modes):
        lid = FastLiDAR2D(N, KM.LIDAR_MAX_RANGE, mode, layout)
        geoms = []
        for chunk in (None, 5):                                # chia lô con không đổi kết quả
            geom = HostGeometry(be, lid, lidar_chunk=chunk, radius=KM.COLL_RADIUS)
            geom.setup(ep_infos, [bm[M] for M in Ms])
            geoms.append(geom)
        for which in (2, 3):                                   # góc float Python, góc float32
            th_list = [t for p in poses for t in (p[2] if which == 2 else list(p[3]))]
            thobs = np.array([float(t) for t in th_list], dtype=np.float64)
            ref = np.array([lid.scan(np.array([xs[i], ys[i]]), th_list[i], ep_infos[i].segs) for i in range(len(xs))])
            for geom in geoms:
                prox = geom.scan(xs, ys, thobs)
                if prox.dtype != np.float32 or not np.array_equal(prox, ref):
                    problems.append(f"LiDAR {mode}:{N}{':' + layout if layout else ''} theo lô khác FastLiDAR2D.scan")
                    return
        for geom in geoms:
            if not np.array_equal(geom.collide(xs, ys), ref_c):
                problems.append("va chạm theo lô khác check_collision")
                return
    # nén theo lượt còn chạy (bucket giữ liền) cho cùng kết quả
    lid = FastLiDAR2D(KM.LIDAR_NUM_RAYS, KM.LIDAR_MAX_RANGE, KM.LIDAR_MODE, KM.SECTOR_LAYOUT)
    geom = HostGeometry(be, lid, radius=KM.COLL_RADIUS)
    geom.setup(ep_infos, [bm[M] for M in Ms])
    keep = rng.uniform(size=len(xs)) < 0.6
    geom.keep(keep)
    th = np.array([float(t) for p in poses for t in p[2]])[keep]
    prox = geom.scan(xs[keep], ys[keep], th)
    sel = np.flatnonzero(keep)
    ref = np.array([lid.scan(np.array([xs[i], ys[i]]), float(th[j]), ep_infos[i].segs) for j, i in enumerate(sel)])
    if not np.array_equal(prox, ref):
        problems.append("LiDAR theo lô sau khi nén khác FastLiDAR2D.scan")


def _obs_and_geo_check(rng, problems):
    from src.batch.backend import Backend
    from src.batch.sim import HostGeometry, MapInfo, geo_lookup, obs_tail, robot_geometry
    from src.env.geodesic import GeodesicField
    from src.env.robot_env import RobotNavEnv
    be = Backend("numpy")
    segs = synthetic_maps(rng)[0]
    m = {"segments": segs.tolist(), "start": [1.0, 1.0], "goal": [8.5, 7.25], "start_heading": 0.5}
    env = RobotNavEnv(map_data=m)
    env.reset()
    n = 40
    x, y, th_py, th32 = random_poses(rng, n, segs)
    mi = MapInfo("check", m, env.segments, env.start_pos, env.goal_pos, 0.5, 0.0, None, None, 0.0)
    geom = robot_geometry(be, HostGeometry(be, env._lidar))      # Waffle: quét từ tâm thân như env._get_obs
    geom.setup([mi] * n, [0] * n)
    gx, gy = np.full(n, env.goal_pos[0]), np.full(n, env.goal_pos[1])
    for th_list in (th_py, list(th32)):
        thobs = np.array([float(t) for t in th_list], dtype=np.float64)
        got = np.concatenate([geom.scan(x, y, thobs), obs_tail(be, x, y, gx, gy, thobs)], axis=1)
        ref = []
        for i in range(n):
            env.pos = np.array([x[i], y[i]])
            env.theta = th_list[i]
            ref.append(env._get_obs())
        if not np.array_equal(got, np.array(ref)):
            problems.append("quan sát theo lô khác RobotNavEnv._get_obs")
            break
    env.close()
    for grid in (rng.uniform(0.0, 14.0, (200, 200)).astype(np.float32), rng.uniform(0.0, 14.0, (200, 200))):
        field = GeodesicField(grid)                            # lưới float32 (compute_geodesic) và float64
        px = rng.uniform(-0.7, 10.7, 500)
        py = rng.uniform(-0.7, 10.7, 500)
        ones = np.ones(500, np.int64)
        flat = np.ascontiguousarray(grid, dtype=np.float64).ravel()      # như simulate
        got = geo_lookup(np, flat, px, py, 0 * ones, 200 * ones, 200 * ones, np.full(500, float(field.res)))
        ref = np.array([field.at(px[i], py[i]) for i in range(500)])
        if not np.array_equal(got, ref):
            problems.append(f"tra geodesic theo lô khác GeodesicField.at (lưới {grid.dtype})")


def _mlp_check(rng, problems):
    from src.baselines.fixed_ga import FixedMLP
    from src.batch.backend import Backend
    from src.batch.policies import MLPPolicy
    from src.env.kinematics import OBS_DIM
    be = Backend("numpy")
    for layers in ([OBS_DIM, 8, 2], [OBS_DIM, 16, 8, 2]):
        nets = []
        for _ in range(3):
            net = FixedMLP(layers)
            net.set_params(rng.normal(0.0, 0.3, net.num_params))
            nets.append(net)
        n = 90
        ep = np.arange(n) % 3
        obs = rng.uniform(-1.0, 1.0, (n, OBS_DIM)).astype(np.float32)
        got = MLPPolicy(be, nets, ep).step(obs)
        ref = np.array([nets[ep[i]].activate(obs[i].tolist()) for i in range(n)])
        if not np.array_equal(got, ref):
            problems.append(f"FixedMLP {layers} theo lô (np.matmul xếp chồng) khác FixedMLP.activate")


# ── So trọn một lượt: run_episode với simulate ────────────────────────────
_EP_PREFIX = "<tự kiểm trọn lượt>"            # khóa giả trong bộ đệm bản đồ của parallel (không phải file)


def same_bits(a, b) -> bool:
    """Cùng kiểu và cùng giá trị tới từng bit (float so bằng struct.pack, nên -0.0 khác 0.0); dict so cả thứ tự khóa."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return struct.pack("<d", a) == struct.pack("<d", b)
    if isinstance(a, dict):
        return list(a) == list(b) and all(same_bits(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same_bits(x, y) for x, y in zip(a, b))
    return bool(a == b)


def _episode_maps(rng) -> dict:
    """Hai bản đồ tổng hợp {khóa giả: (dict bản đồ, GeodesicField)}: "thoáng" (tường biên, hai đoạn, một đoạn suy
    biến; đi thẳng từ (2, 2) theo hướng 0 là tới đích (5, 2); có metrics.optimal_path_length, lưới geodesic
    float32) và "rối" (37 đoạn ngẫu nhiên, hướng xuất phát 0,3, không có metrics, lưới float64 có giá trị không
    biểu diễn được bằng float32)."""
    from src.env.collision import check_collision
    from src.env.geodesic import GEO_RES, GeodesicField
    walls = [[[0.0, 0.0], [10.0, 0.0]], [[10.0, 0.0], [10.0, 10.0]], [[10.0, 10.0], [0.0, 10.0]], [[0.0, 10.0], [0.0, 0.0]]]
    open_map = {"map_id": "tu_kiem_thoang",
                "segments": walls + [[[6.5, 6.0], [8.5, 6.0]], [[3.0, 7.0], [3.0, 8.5]], [[7.0, 8.0], [7.0, 8.0]]],
                "start": [2.0, 2.0], "goal": [5.0, 2.0], "start_heading": 0.0, "metrics": {"optimal_path_length": 3.0}}
    segs = synthetic_maps(rng)[1]
    pts: list = []
    while len(pts) < 2:                                        # xuất phát và đích cách mọi đoạn hơn 0,6 m
        p = rng.uniform(0.8, 9.2, 2)
        if not check_collision(p, segs, 0.6):
            pts.append([float(p[0]), float(p[1])])
    clutter = {"map_id": "tu_kiem_roi", "segments": segs.tolist(), "start": pts[0], "goal": pts[1], "start_heading": 0.3}
    n = int(round(10.0 / GEO_RES))
    c = (np.arange(n) + 0.5) * GEO_RES
    xx, yy = np.meshgrid(c, c)
    out = {}
    for name, m, dtype in (("thoang", open_map, np.float32), ("roi", clutter, np.float64)):
        grid = np.hypot(xx - m["goal"][0], yy - m["goal"][1])
        if dtype is np.float64:
            grid = grid * 1.07 + rng.uniform(0.0, 1e-3, grid.shape)
        out[f"{_EP_PREFIX}/{name}.json"] = (m, GeodesicField(grid.astype(dtype)))
    return out


def _episode_groups(rng, p_open: str, p_clut: str) -> list:
    """[(tên, be -> hàm dựng bộ điều khiển theo lô, [(bộ điều khiển từng lượt, chỉ số trong nhóm, bản đồ, progress)])]."""
    import neat.activations as A
    import neat.aggregations as G
    from neat.nn import RecurrentNetwork
    from src.baselines.apf import APFController, apf_unsupported_reason
    from src.baselines.fixed_ga import FixedMLP
    from src.batch.policies import APFPolicy, MLPPolicy, NeatPolicy
    from src.env import kinematics as KM
    N = KM.LIDAR_NUM_RAYS
    # FixedMLP [N + 3, 4, 2]: đi thẳng (tới đích; trên bản đồ rối thì va chạm), quay tại chỗ (hết giờ, đếm bước
    # quay), trọng số ngẫu nhiên. Waffle ra lệnh (tốc độ tiến, tốc độ quay): đi thẳng (0,6; 0), quay tại chỗ (0; 0,6)
    biases = ((0.6, 0.0), (0.0, 0.6), None) if KM.ROBOT == "waffle" else ((0.6, 0.6), (0.6, -0.6), None)
    mlps = []
    for out_bias in biases:
        net = FixedMLP([KM.OBS_DIM, 4, 2])
        flat = rng.normal(0.0, 0.5, net.num_params) if out_bias is None else np.zeros(net.num_params)
        if out_bias is not None:
            flat[-2:] = out_bias
        net.set_params(flat)
        mlps.append(net)
    # mạng NEAT hồi quy dựng tay: tự nối, chu trình ẩn <-> ẩn, output -> ẩn, output tự nối, nguồn 99 không bao giờ
    # được tính (luôn 0.0), response khác 1; trọng số float của Python như genome của neat-python
    ins = [-(i + 1) for i in range(KM.OBS_DIM)]

    def w() -> float:
        return float(rng.normal(0.0, 0.8))

    nets = []
    for k in range(2):
        evals = [(2, A.tanh_activation, G.sum_aggregation, float(rng.normal(0.0, 0.5)), 1.0,
                  [(ins[0], w()), (ins[N + 1], w()), (3, w()), (2, w())]),
                 (3, A.tanh_activation, G.sum_aggregation, float(rng.normal(0.0, 0.5)), 0.8 if k else 1.0,
                  [(ins[N], w()), (2, w()), (0, w()), (99, w())]),
                 (0, A.tanh_activation, G.sum_aggregation, 0.5, 1.0, [(2, w()), (ins[N + 2], w()), (ins[1], w())]),
                 (1, A.tanh_activation, G.sum_aggregation, 0.5, 1.0, [(3, w()), (ins[N + 1], w()), (1, w())])]
        nets.append(RecurrentNetwork(ins, [0, 1], evals))
    geo, euc = "geodesic", "euclid"
    groups = [("FixedMLP", lambda be: (lambda t: MLPPolicy(be, mlps, t)),
               [(mlps[0], 0, p_open, geo), (mlps[1], 1, p_open, euc), (mlps[0], 0, p_clut, euc), (mlps[2], 2, p_clut, geo)])]
    if apf_unsupported_reason() is None:          # APF chỉ chạy với quạt đều (Waffle: quạt 12); e32 thì bỏ nhóm APF
        apf = APFController()
        groups.append(("APF", lambda be: (lambda t: APFPolicy(be, N, apf)),
                       [(apf, 0, p_open, euc), (apf, 0, p_clut, geo)]))
    groups.append(("NEAT hồi quy", lambda be: (lambda t: NeatPolicy(be, nets, True, t)),
                   [(nets[0], 0, p_clut, geo), (nets[1], 1, p_clut, euc), (nets[0], 0, p_open, euc), (nets[1], 1, p_open, geo)]))
    return groups


def episode_check(be=None, geom_factory=None, seed: int = 20261127) -> list[str]:
    """So trọn lượt PAR.run_episode với sim.simulate (cùng dict kết quả tới từng bit, cùng thứ tự khóa, cùng kiểu).
    be / geom_factory: mặc định NumPy + HostGeometry; GPU truyền Backend của nó và hàm dựng GpuGeometry.
    Bản đồ tổng hợp được đặt tạm vào PAR._MAPS / PAR._GEO dưới khóa giả (load_map và _geodesic đọc bộ đệm trước),
    xong thì gỡ ra."""
    from src.batch.backend import Backend
    from src.batch.sim import Episode, make_host_geometry, simulate
    from src.env import kinematics as KM
    from src.env.lidar import FastLiDAR2D
    from src.neat_engine import parallel as PAR
    rng = np.random.default_rng(seed)
    be = be if be is not None else Backend("numpy")
    if geom_factory is None:
        lidar = FastLiDAR2D(KM.LIDAR_NUM_RAYS, KM.LIDAR_MAX_RANGE, KM.LIDAR_MODE, KM.SECTOR_LAYOUT)
        geom_factory = lambda: make_host_geometry(be, lidar)  # noqa: E731
    maps = _episode_maps(rng)
    problems: list[str] = []
    for path, (m, field) in maps.items():
        PAR._MAPS[path], PAR._GEO[path] = m, field
    try:
        p_open, p_clut = list(maps)
        for name, make, items in _episode_groups(rng, p_open, p_clut):
            ref = [PAR.run_episode(pol, path, {"progress": prog, "geo_cache": None}) for pol, _, path, prog in items]
            got = simulate(be, [Episode(path, prog, None, t) for _, t, path, prog in items], make(be), geom_factory())
            bad = [i for i in range(len(ref)) if i >= len(got) or not same_bits(ref[i], got[i])]
            if bad:
                i = bad[0]
                problems.append(f"trọn lượt theo lô khác run_episode ({name}: {len(bad)}/{len(ref)} lượt; từng lượt "
                                f"{ref[i]} / theo lô {got[i] if i < len(got) else None})")
    finally:
        for path in maps:
            PAR._MAPS.pop(path, None)
            PAR._GEO.pop(path, None)
    return problems


def _episode_check(rng, problems):
    problems.extend(episode_check())


def self_check(force: bool = False) -> list[str]:
    """Danh sách chỗ lệch (rỗng = mọi giả định của chế độ exact đúng trên máy này); nhớ kết quả trong tiến trình."""
    if "result" in _CACHE and not force:
        return list(_CACHE["result"])
    problems: list[str] = []
    if sum([1e16, 1.0, -1e16]) != 1.0:
        problems.append("sum() của Python không phải tổng bù (cần Python >= 3.12)")
    rng = np.random.default_rng(20261126)
    for fn in (_neumaier_check, _scalar_vs_array, _kinematics_check, _geometry_check, _obs_and_geo_check, _mlp_check,
               _episode_check):
        try:
            fn(rng, problems)
        except Exception as e:  # noqa: BLE001  (một phép kiểm hỏng cũng là "không bảo đảm trùng bit")
            problems.append(f"{fn.__name__}: {e!r}")
    if not math.isfinite(math.tanh(0.5)):
        problems.append("math.tanh hỏng")
    _CACHE["result"] = tuple(problems)
    return problems
