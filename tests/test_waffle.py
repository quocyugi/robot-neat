"""
Kiểm thử robot Waffle và cảm biến (c) sector:12, (e) sector:e32 (bản rút gọn)
==============================================================================
Chạy:  python -B -X utf8 tests/test_waffle.py            (dưới 1 phút; chạy bằng python như trên, không qua pytest)

Robot và cảm biến đọc từ biến môi trường lúc import src (src/env/kinematics.py), nên mỗi cấu hình chạy
trong một tiến trình con riêng (python tests/test_waffle.py --inner <cấu hình>):
  waffle_c   NEAT_ROBOT=waffle NEAT_LIDAR_MODE=sector NEAT_LIDAR_N=12
  waffle_e   NEAT_ROBOT=waffle NEAT_LIDAR_MODE=sector NEAT_LIDAR_N=32 NEAT_SECTOR_LAYOUT=e32
  old        NEAT_ROBOT=old (robot cũ, cảm biến mặc định 12 tia)
  old_sec    NEAT_ROBOT=old NEAT_LIDAR_MODE=sector NEAT_LIDAR_N=12 (robot cũ, 12 quạt đều)
Mỗi hàm test_* ghi các cấu hình nó áp dụng trong thuộc tính CONFIGS (mặc định: mọi cấu hình).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

CONFIG_ENV = {
    "waffle_c": {"NEAT_ROBOT": "waffle", "NEAT_LIDAR_MODE": "sector", "NEAT_LIDAR_N": "12", "NEAT_SECTOR_LAYOUT": ""},
    "waffle_e": {"NEAT_ROBOT": "waffle", "NEAT_LIDAR_MODE": "sector", "NEAT_LIDAR_N": "32", "NEAT_SECTOR_LAYOUT": "e32"},
    "old": {"NEAT_ROBOT": "old", "NEAT_LIDAR_MODE": "", "NEAT_LIDAR_N": "", "NEAT_SECTOR_LAYOUT": ""},
    "old_sec": {"NEAT_ROBOT": "old", "NEAT_LIDAR_MODE": "sector", "NEAT_LIDAR_N": "12", "NEAT_SECTOR_LAYOUT": ""},
}
WAFFLE = ("waffle_c", "waffle_e")
W = 0.26669137                      # trần mỗi bánh (m/s)
F32 = np.float32


def configs(*names):
    def deco(fn):
        fn.CONFIGS = names
        return fn
    return deco


# ── công cụ ──────────────────────────────────────────────────────────────────
def ref_cmd_exact(o1, o2):
    """(v, w) bằng số hữu tỉ (Fraction) theo đúng công thức, từ o1, o2 float: tham chiếu độc lập (không float32)."""
    fr = Fraction
    o1 = min(max(fr(float(o1)), fr(-1)), fr(1))
    o2 = min(max(fr(float(o2)), fr(-1)), fr(1))
    vc = fr("0.26") * max(o1, fr(0))
    wc = fr("1.82") * o2
    wm = fr("0.26669137")
    vl = min(max(vc - fr("0.1435") * wc, -wm), wm)
    vr = min(max(vc + fr("0.1435") * wc, -wm), wm)
    return (vl + vr) / 2, (vr - vl) / fr("0.287"), vl, vr


def ref_cmd_f32_scalar(o1, o2):
    """(v, w) float32 bằng phép vô hướng np.float32 (cách viết khác của cùng thứ tự phép trong waffle_command)."""
    o1 = min(max(F32(o1), F32(-1.0)), F32(1.0))
    o2 = min(max(F32(o2), F32(-1.0)), F32(1.0))
    vc = F32(0.26) * (o1 if o1 > F32(0.0) else F32(0.0))
    wc = F32(1.82) * o2
    wm = F32(W)
    vl = min(max(vc - wc * F32(0.1435), -wm), wm)
    vr = min(max(vc + wc * F32(0.1435), -wm), wm)
    return (vl + vr) / F32(2.0), (vr - vl) / F32(0.287)


def cmd(o1, o2):
    from src.env.kinematics import waffle_command
    v, w = waffle_command(np, np.array([[o1, o2]], dtype=np.float32))
    return v[0], w[0]


def env_for(segments, start, heading, goal=(1.0, 9.0)):
    from src.env.robot_env import RobotNavEnv
    m = {"segments": [list(map(list, s)) for s in segments], "start": list(start), "goal": list(goal),
         "start_heading": float(heading)}
    return RobotNavEnv(map_data=m), m


def const_mlp(o1_bias, o2_bias):
    """FixedMLP [OBS_DIM, 8, 2] mọi trọng số 0, chỉ bias đầu ra: ra hằng (tanh(2,5 b1), tanh(2,5 b2))."""
    from src.baselines.fixed_ga import FixedMLP
    from src.env.kinematics import OBS_DIM
    net = FixedMLP([OBS_DIM, 8, 2])
    flat = np.zeros(net.num_params)
    flat[-2:] = (o1_bias, o2_bias)
    net.set_params(flat)
    return net


def per_episode_and_batch(policy, m, progress="euclid", key="<test_waffle>/ban_do.json"):
    """(kết quả run_episode, kết quả simulate theo lô) của một bộ điều khiển FixedMLP trên bản đồ m."""
    from src.batch.backend import Backend
    from src.batch.policies import MLPPolicy
    from src.batch.sim import Episode, make_host_geometry, simulate
    from src.env import kinematics as KM
    from src.env.lidar import FastLiDAR2D
    from src.neat_engine import parallel as PAR
    PAR._MAPS[key] = m
    try:
        ref = PAR.run_episode(policy, key, {"progress": progress, "geo_cache": None})
        be = Backend("numpy")
        lid = FastLiDAR2D(KM.LIDAR_NUM_RAYS, KM.LIDAR_MAX_RANGE, KM.LIDAR_MODE, KM.SECTOR_LAYOUT)
        got = simulate(be, [Episode(key, progress, None, 0)], lambda t: MLPPolicy(be, [policy], t),
                       make_host_geometry(be, lid))[0]
    finally:
        PAR._MAPS.pop(key, None)
    return ref, got


# ── hằng, số đầu vào ─────────────────────────────────────────────────────────
@configs(*WAFFLE)
def test_waffle_constants_and_input_counts():
    import re
    from src.env import kinematics as KM
    from src.env.robot_env import RobotNavEnv  # noqa: F401
    assert KM.ROBOT == "waffle"
    assert (KM.MAX_STEPS, KM.LIDAR_MAX_RANGE, KM.COLL_RADIUS, KM.BODY_OFFSET) == (1000, 3.5, 0.2077, 0.064)
    assert (KM.ROBOT_RADIUS, KM.GOAL_RADIUS, KM.DT) == (0.25, 0.30, 0.10), "hằng chung không đổi"
    n_bins = 12 if not KM.SECTOR_LAYOUT else 32
    assert KM.LIDAR_NUM_RAYS == n_bins and KM.OBS_DIM == n_bins + 3 == (15 if n_bins == 12 else 35)
    cfg = ROOT / "configs" / ("neat_recurrent.ini" if n_bins == 12 else "neat_recurrent_n32.ini")
    n_in = int(re.search(r"(?m)^num_inputs\s*=\s*(\d+)", cfg.read_text(encoding="utf-8")).group(1))
    assert n_in == KM.OBS_DIM, (cfg.name, n_in)
    # GA B1 = [số đầu vào, 8, 2]
    import importlib.util
    spec = importlib.util.spec_from_file_location("rb", ROOT / "scripts" / "run_baselines.py")
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    assert rb.LAYERS["ga_b1"] == [KM.OBS_DIM, 8, 2]
    env, _ = env_for([[[3.0, 3.0], [3.0, 3.0]]], (5.0, 5.0), 0.0)
    obs, _ = env.reset()
    assert obs.shape == (KM.OBS_DIM,) and env.observation_space.shape == (KM.OBS_DIM,) and obs.dtype == np.float32
    s = KM.sensor_summary()
    assert s["robot"] == "waffle" and s["n"] == n_bins and s["mode"] == "sector"
    assert s["layout"] == (KM.SECTOR_LAYOUT or "uniform")


@configs(*WAFFLE)
def test_neat_runner_checks_num_inputs():
    """NEATRunner nhận config đúng số đầu vào của mã hóa và báo lỗi với config sai."""
    from src.env import kinematics as KM
    from src.neat_engine.runner import NEATRunner
    good, bad = ("neat_recurrent.ini", "neat_recurrent_n32.ini")[:: 1 if KM.LIDAR_NUM_RAYS == 12 else -1]
    with tempfile.TemporaryDirectory() as tmp:
        NEATRunner(ROOT / "configs" / good, ROOT / "data" / "maps_barn", Path(tmp) / "a", use_novelty=False,
                   use_curriculum=False, max_generations=1, levels=(1,), verbose=False)
        try:
            NEATRunner(ROOT / "configs" / bad, ROOT / "data" / "maps_barn", Path(tmp) / "b", use_novelty=False,
                       use_curriculum=False, max_generations=1, levels=(1,), verbose=False)
        except ValueError as e:
            assert "num_inputs" in str(e)
        else:
            raise AssertionError("config sai số đầu vào phải báo lỗi")


# ── lệnh (v, w), kẹp bánh, không lùi ─────────────────────────────────────────
@configs(*WAFFLE)
def test_wheel_clamp_examples():
    # (1; 0): v = 0,26, w = 0 (bánh chưa bão hòa)
    v, w = cmd(1.0, 0.0)
    assert v == F32(0.26) and w == F32(0.0), (v, w)
    # (0; 1): đứng yên quay 1,82 rad/s; bánh +-0,26117 < 0,26669 chưa bão hòa
    v, w = cmd(0.0, 1.0)
    assert v == F32(0.0) and abs(float(w) - 1.82) < 2e-7, (v, w)
    # (1; 1): xin v 0,26, w 1,82 -> bánh phải ở trần 0,26669137, bánh trái -0,00117
    v, w = cmd(1.0, 1.0)
    ve, we, vl, vr = ref_cmd_exact(1.0, 1.0)
    assert vr == Fraction("0.26669137") and vl == Fraction("0.26") - Fraction("0.1435") * Fraction("1.82")
    assert abs(float(v) - 0.13276067) < 1e-7 and abs(float(w) - 0.93331486) < 1e-7, (v, w)
    assert abs(float(v) - float(ve)) < 1e-7 and abs(float(w) - float(we)) < 1e-6
    # (0,13335; 0,9293) là kết quả khi coi bánh trái = 0; thực tế bánh trái = -0,00117 m/s
    assert abs(float(v) - 0.13335) > 5e-4 and abs(float(w) - 0.9293) > 3e-3
    # (1; 0,1): bánh phải bão hòa
    v, w = cmd(1.0, 0.1)
    assert abs(float(v) - 0.25028718) < 1e-7 and abs(float(w) - 0.11431485) < 1e-7, (v, w)
    # (1,3; -1,3) kẹp về (1; -1): đối xứng của (1; 1)
    v2, w2 = cmd(1.3, -1.3)
    v1, w1 = cmd(1.0, 1.0)
    assert v2 == v1 and w2 == -w1


@configs(*WAFFLE)
def test_wheel_clamp_random_vs_references():
    from src.env.kinematics import waffle_command
    rng = np.random.default_rng(7)
    a = rng.uniform(-1.3, 1.3, (10000, 2)).astype(np.float32)
    a[:6] = [[0.0, 0.0], [-0.0, -0.0], [1.0, 1.0], [-1.0, -1.0], [0.5, -0.999], [1e-9, 0.3]]
    v, w = waffle_command(np, a)
    assert v.dtype == np.float32 and w.dtype == np.float32
    for i in range(a.shape[0]):
        vs, ws = ref_cmd_f32_scalar(a[i, 0], a[i, 1])
        assert vs == v[i] and ws == w[i], (a[i], v[i], vs, w[i], ws)
    for i in range(0, a.shape[0], 25):                      # so số hữu tỉ (sai số làm tròn float32)
        ve, we, vl, vr = ref_cmd_exact(a[i, 0], a[i, 1])
        assert abs(float(v[i]) - float(ve)) < 1e-7 and abs(float(w[i]) - float(we)) < 1e-6, (a[i], v[i], ve)
    # giới hạn: không lùi, v <= 0,26, |w| <= 1,82, mỗi bánh trong +-W
    assert (v >= 0.0).all() and (v <= F32(0.26)).all() and (np.abs(w) <= F32(1.82) * F32(1.0000001)).all()
    vl = v - w * F32(0.1435)
    vr = v + w * F32(0.1435)
    assert (np.abs(vl) <= W + 1e-6).all() and (np.abs(vr) <= W + 1e-6).all()


@configs(*WAFFLE)
def test_no_reverse():
    from src.env.kinematics import waffle_step
    for o1 in (-1.0, -0.5, -1e-8, -0.0, 0.0, -1.3):
        v, w = cmd(o1, 0.0)
        assert v == 0.0 and w == 0.0, (o1, v, w)
        x, y = np.array([5.0]), np.array([4.0])
        xn, yn, th, spin = waffle_step(np, np.cos, np.sin, x, y, np.array([0.3], np.float32),
                                       np.array([[o1, 0.0]], np.float32))
        assert xn[0] == 5.0 and yn[0] == 4.0 and th[0] == F32(0.3) and not spin[0]
    v, w = cmd(-1.0, 0.7)                                    # o1 < 0: chỉ quay tại chỗ
    assert v == 0.0 and w > 0.0


# ── thân, va chạm, gốc LiDAR ─────────────────────────────────────────────────
@configs(*WAFFLE)
def test_body_center_and_collision_radius():
    from src.env.robot_env import RobotNavEnv  # noqa: F401
    env, _ = env_for([[[3.0, 3.0], [3.0, 3.0]]], (5.0, 5.0), 0.7)
    env.reset()
    bc = env.body_center()
    assert np.allclose(bc, [5.0 - 0.064 * np.cos(0.7), 5.0 - 0.064 * np.sin(0.7)], atol=0, rtol=1e-15)
    # tường sau / trước, hướng 0, lệnh (0; 0): đứng yên, va chạm xét ở tâm thân (lùi 0,064 m), r = 0,2077
    # sau: chạm khi khoảng cách tới tâm quay <= 0,064 + 0,2077 = 0,2717; trước: <= 0,2077 - 0,064 = 0,1437
    cases = [(-0.2716, True), (-0.2718, False), (0.1436, True), (0.1438, False)]
    for d, hit in cases:
        xw = 5.0 + d
        seg = [[[xw, 4.0], [xw, 6.0]]]
        env, m = env_for(seg, (5.0, 5.0), 0.0)
        env.reset()
        _, _, term, trunc, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert info["is_collision"] is hit and term is hit, (d, info["is_collision"])
        assert tuple(env.pos) == (5.0, 5.0)
        ref, got = per_episode_and_batch(const_mlp(0.0, 0.0), m)
        from src.batch.numerics import same_bits
        assert same_bits(ref, got), (ref, got)
        assert ref["collision"] is hit and ref["steps"] == (1 if hit else 1000), ref


@configs(*WAFFLE)
def test_lidar_origin_is_body_center():
    """Vật cản điểm đặt theo tâm thân: quạt đúng hướng đọc đúng 1 - D/3,5, mọi quạt khác 0 (ở cả hai đường)."""
    from src.batch.backend import Backend
    from src.batch.sim import HostGeometry, MapInfo, robot_geometry
    from src.env import kinematics as KM
    th0 = 0.7
    rc = np.array([5.0, 5.0])
    bc = rc - 0.064 * np.array([np.cos(th0), np.sin(th0)])
    lid = KM.LIDAR_NUM_RAYS
    if KM.SECTOR_LAYOUT == "e32":
        mids = [-130.5 + 9.0 * j for j in range(30)] + [157.5, -157.5]
    else:
        mids = [30.0 * i for i in range(lid)]
    be = Backend("numpy")
    for b, beta in enumerate(mids):
        for D in (1.2, 3.4):
            q = bc + D * np.array([np.cos(th0 + np.radians(beta)), np.sin(th0 + np.radians(beta))])
            env, m = env_for([[q.tolist(), q.tolist()]], rc, th0)
            obs, _ = env.reset()
            want = np.zeros(lid, np.float32)
            want[b] = F32(1.0 - D / 3.5)
            assert np.allclose(obs[:lid], want, atol=2e-7, rtol=0), (beta, D, obs[:lid])
            assert np.flatnonzero(obs[:lid]).tolist() == [b], (beta, obs[:lid])
            # đường theo lô: BodyCenterGeometry + HostGeometry
            mi = MapInfo("t", m, env.segments, env.start_pos, env.goal_pos, th0, 0.0, None, None, 0.0)
            geom = robot_geometry(be, HostGeometry(be, env._lidar, radius=KM.COLL_RADIUS))
            geom.setup([mi], [0])
            prox = geom.scan(np.array([5.0]), np.array([5.0]), np.array([th0]))
            assert np.array_equal(prox[0], obs[:lid])
    # ngoài tầm 3,5 m: không thấy
    q = bc + 3.6 * np.array([np.cos(th0), np.sin(th0)])
    env, _ = env_for([[q.tolist(), q.tolist()]], rc, th0)
    obs, _ = env.reset()
    assert not obs[:lid].any()


# ── bố cục e32 ────────────────────────────────────────────────────────────────
def test_e32_edges():
    """32 quạt: 30 x 9 độ trên [-135; 135] từ phải sang trái, [135; 180], [180; 225]; khép vòng, mỗi quạt <= 180."""
    from src.env.lidar import FastLiDAR2D, check_sector_edges, sector_layout
    edges, mids = sector_layout(32, "e32")
    assert edges.shape == (32,) and mids.shape == (32,)
    deg = np.degrees(edges)
    want = [-135.0 + 9.0 * j for j in range(31)] + [180.0]
    assert np.allclose(deg, want, atol=1e-12, rtol=0)
    assert edges[15] == 0.0, "hướng trước phải đúng là cạnh giữa quạt 14 và 15"
    width = np.diff(np.concatenate((deg, deg[:1] + 360.0)))
    assert np.allclose(width[:30], 9.0) and np.allclose(width[30:], 45.0)
    assert abs(width.sum() - 360.0) < 1e-9 and (width <= 180.0).all()
    check_sector_edges(edges)
    for bad in (np.radians([0.0, 90.0, 300.0]),        # quạt 90 -> 300 mở 210 độ
                np.radians([0.0, 120.0, 100.0]),       # không tăng dần
                np.radians([10.0, 200.0])):            # 190 độ
        try:
            check_sector_edges(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"phải báo lỗi: {np.degrees(bad)}")
    # quạt thấy đúng vật: điểm ở hướng giữa quạt j chỉ thuộc quạt j; điểm ở hướng 0 thuộc cả quạt 14 và 15
    lid = FastLiDAR2D(32, 3.5, "sector", "e32")
    p = np.array([5.0, 5.0])
    for th in (0.0, 1.1, -2.5):
        for j, beta in enumerate(mids):
            q = p + 2.0 * np.array([np.cos(th + beta), np.sin(th + beta)])
            prox = lid.scan(p, th, np.array([[q, q]]))
            assert np.flatnonzero(prox).tolist() == [j], (th, j, prox)
            assert abs(float(prox[j]) - (1.0 - 2.0 / 3.5)) < 2e-7
    q = p + np.array([2.0, 0.0])
    prox = lid.scan(p, 0.0, np.array([[q, q]]))
    assert np.flatnonzero(prox).tolist() == [14, 15], prox
    # vật ở hướng trước hơi trái (+4,5 độ) là quạt 15, hơi phải là quạt 14, sau trái 170 độ là 30, sau phải -170 là 31
    for beta, j in ((4.5, 15), (-4.5, 14), (170.0, 30), (-170.0, 31), (-134.0, 0), (134.0, 29)):
        q = p + 1.0 * np.array([np.cos(np.radians(beta)), np.sin(np.radians(beta))])
        prox = lid.scan(p, 0.0, np.array([[q, q]]))
        assert np.flatnonzero(prox).tolist() == [j], (beta, prox)
    # e32 chỉ với sector, 32 quạt
    for args in ((32, 3.5, "rays", "e32"), (12, 3.5, "sector", "e32")):
        try:
            FastLiDAR2D(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"phải báo lỗi: {args}")


def test_uniform_12_layout_unchanged():
    """Quạt đều 12 (mã (c)): cạnh = linspace(0, 2 pi, 12) - pi/12 như robot cũ, quạt 0 giữa hướng trước."""
    from src.env.lidar import FastLiDAR2D
    lid = FastLiDAR2D(12, 3.5, "sector")
    assert np.array_equal(lid.sector_edges, np.linspace(0, 2 * np.pi, 12, endpoint=False) - np.pi / 12)
    assert lid.layout == ""
    p = np.array([5.0, 5.0])
    for i in range(12):
        q = p + 2.0 * np.array([np.cos(np.radians(30 * i)), np.sin(np.radians(30 * i))])
        assert np.flatnonzero(lid.scan(p, 0.0, np.array([[q, q]]))).tolist() == [i]


# ── lượt 1.000 bước, xoay tại chỗ, fitness ───────────────────────────────────
@configs(*WAFFLE)
def test_spin_threshold():
    from src.env.kinematics import waffle_step
    cases = [((0.0, 1.0), True), ((0.0, 0.27), False), ((0.0, 0.28), True), ((0.0961, 1.0), True),
             ((0.2, 1.0), False), ((1.0, 0.0), False), ((0.0, 0.0), False), ((0.0, -1.0), True)]
    for (o1, o2), want in cases:
        v, w = cmd(o1, o2)
        ref = abs(float(v)) < 0.025 and abs(float(w)) > 0.5
        _, _, _, spin = waffle_step(np, np.cos, np.sin, np.array([5.0]), np.array([5.0]), np.array([0.0], np.float32),
                                    np.array([[o1, o2]], np.float32))
        assert bool(spin[0]) is want is ref, ((o1, o2), float(v), float(w), bool(spin[0]))


@configs(*WAFFLE)
def test_episode_1000_steps_spin_penalty_and_time_bonus():
    from src.batch.numerics import same_bits
    from src.neat_engine.evaluator import compute_episode_fitness
    walls = [[[0.0, 0.0], [10.0, 0.0]], [[10.0, 0.0], [10.0, 10.0]], [[10.0, 10.0], [0.0, 10.0]], [[0.0, 10.0], [0.0, 0.0]]]
    m = {"segments": walls, "start": [2.0, 2.0], "goal": [5.0, 2.0], "start_heading": 0.0}
    # quay tại chỗ (0; 1): hết giờ đúng 1.000 bước, 1.000 bước xoay, fitness = 3 + 0 - 1 * 1000/1000 = 2
    ref, got = per_episode_and_batch(const_mlp(0.0, 10.0), m)
    assert same_bits(ref, got), (ref, got)
    assert ref["steps"] == 1000 and not ref["success"] and not ref["collision"], ref
    assert ref["fitness"] == 2.0, ref
    # đi thẳng (1; 0): tới đích; thưởng thời gian 5 (1 - bước / 1000)
    env, _ = env_for(walls, (2.0, 2.0), 0.0, goal=(5.0, 2.0))
    env.reset()
    done = False
    while not done:
        _, _, term, trunc, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        done = term or trunc
    assert info["is_success"] and info["spin_steps"] == 0
    steps = info["steps"]
    assert steps == int(np.ceil((3.0 - 0.30) / (0.26 * 0.1) - 1e-9)), steps      # 104 bước
    f = compute_episode_fitness(info, progress="euclid")
    prog = (info["d0"] - info["min_distance_to_goal"]) / info["d0"]
    assert abs(f - (prog + 10.0 + 5.0 * (1.0 - steps / 1000.0) + 3.0)) < 1e-12, f
    assert f == compute_episode_fitness(info, progress="euclid", max_steps=1000)
    ref, got = per_episode_and_batch(const_mlp(10.0, 0.0), m)
    assert same_bits(ref, got) and ref["success"] and ref["steps"] == steps, (ref, got)


# ── APF ──────────────────────────────────────────────────────────────────────
@configs("waffle_c")
def test_apf_waffle_commands_within_limits():
    from src.baselines.apf import APFController
    from src.batch.backend import Backend
    from src.batch.policies import APFPolicy
    from src.env.kinematics import waffle_command
    apf = APFController()
    assert apf.vw
    rng = np.random.default_rng(3)
    obs = rng.uniform(0.0, 1.0, (4000, 15)).astype(np.float32)
    obs[:, :12] *= (rng.uniform(size=(4000, 12)) < 0.4)           # phần lớn quạt trống
    ph = rng.uniform(-np.pi, np.pi, 4000)
    obs[:, 12] = rng.uniform(0, 1, 4000)
    obs[:, 13], obs[:, 14] = np.sin(ph), np.cos(ph)
    out = np.array([apf.activate(o.tolist()) for o in obs])
    assert (np.abs(out) <= 1.0).all()
    got = APFPolicy(Backend("numpy"), 12, apf).step(obs)
    assert np.array_equal(got, out), "APF theo lô khác APFController.activate"
    v, w = waffle_command(np, out.astype(np.float32))
    assert (v >= 0).all() and (v <= F32(0.26)).all() and (np.abs(w) <= F32(1.82) * F32(1.0000001)).all()
    assert (np.abs(v - w * F32(0.1435)) <= W + 1e-6).all() and (np.abs(v + w * F32(0.1435)) <= W + 1e-6).all()
    # đích thẳng trước, không vật cản: speed 0,5 (v = 0,13), không quay; đích sau lưng: v = 0, quay tại chỗ
    o = [0.0] * 12 + [0.5, 0.0, 1.0]
    assert apf.activate(o) == [0.5, 0.0]
    o = [0.0] * 12 + [0.5, 0.3, -0.95]
    sp, tu = apf.activate(o)
    v, w = cmd(sp, tu)
    assert sp < 0 and v == 0.0 and w > 0.0
    # vật cản gần ngay trước (quạt 0 ở 0,3 m): lực đẩy làm speed âm -> đứng, không lao vào
    o = [1.0 - 0.3 / 3.5] + [0.0] * 11 + [0.5, 0.0, 1.0]
    sp, tu = apf.activate(o)
    assert sp < 0.0 and cmd(sp, tu)[0] == 0.0


@configs("waffle_e", "old")
def test_apf_rejects_other_sensors_for_waffle_e32():
    from src.env import kinematics as KM
    from src.baselines.apf import APFController, apf_unsupported_reason
    if KM.ROBOT == "old" and not KM.SECTOR_LAYOUT:
        assert apf_unsupported_reason() is None and APFController().vw is False   # robot cũ: đầu ra hai bánh
        return
    try:
        APFController()
    except ValueError as e:
        assert "12" in str(e)
    else:
        raise AssertionError("APF với e32 phải báo lỗi")


# ── robot cũ không đổi ───────────────────────────────────────────────────────
@configs("old", "old_sec")
def test_old_robot_constants_unchanged():
    from src.env import kinematics as KM
    assert KM.ROBOT == "old" and KM.SECTOR_LAYOUT == ""
    assert (KM.MAX_STEPS, KM.LIDAR_MAX_RANGE, KM.COLL_RADIUS, KM.BODY_OFFSET, KM.ROBOT_RADIUS) == (500, 3.0, 0.25, 0.0, 0.25)
    assert KM.OBS_DIM == 15
    want = None if KM.LIDAR_MODE == "rays" else {"mode": "sector", "n": 12}
    assert KM.sensor_summary() == want
    from src.batch.sim import HostGeometry, robot_geometry
    from src.batch.backend import Backend
    from src.env.lidar import FastLiDAR2D
    g = HostGeometry(Backend("numpy"), FastLiDAR2D(12, 3.0, KM.LIDAR_MODE))
    assert robot_geometry(None, g) is g and g.radius == 0.25
    from src.baselines.apf import APFController
    apf = APFController()
    assert apf.activate([0.0] * 12 + [0.5, 0.0, 1.0]) == [0.5, 0.5]     # hai bánh như cũ


# ── tự kiểm của chế độ exact (trọn lượt theo lô = từng lượt) ─────────────────
def test_numerics_self_check():
    from src.batch import numerics
    problems = numerics.self_check(force=True)
    assert problems == [], problems


def test_episode_check_outcomes():
    """episode_check (từng lượt = theo lô) có đủ kết cục tới đích, va chạm, hết giờ với robot đang chạy."""
    import src.batch.numerics as NUM
    from src.neat_engine import parallel as PAR
    outcomes = []
    real = PAR.run_episode

    def spy(policy, path, opts):
        r = real(policy, path, opts)
        outcomes.append("goal" if r["success"] else ("coll" if r["collision"] else "timeout"))
        return r
    PAR.run_episode = spy
    try:
        assert NUM.episode_check() == []
    finally:
        PAR.run_episode = real
    assert {"goal", "coll", "timeout"} <= set(outcomes), outcomes


# ── cấu hình sai báo lỗi ngay khi import ─────────────────────────────────────
@configs("old")
def test_bad_settings_fail_at_import():
    bad = [{"NEAT_ROBOT": "waffle"},                                                   # Waffle với 12 tia điểm
           {"NEAT_ROBOT": "waffle", "NEAT_LIDAR_MODE": "rays", "NEAT_LIDAR_N": "12"},
           {"NEAT_ROBOT": "waffle", "NEAT_LIDAR_MODE": "sector", "NEAT_LIDAR_N": "12", "NEAT_SECTOR_LAYOUT": "e32"},
           {"NEAT_ROBOT": "waffle", "NEAT_LIDAR_MODE": "sector", "NEAT_SECTOR_LAYOUT": "e32"},  # thiếu N = 32
           {"NEAT_ROBOT": "jackal", "NEAT_LIDAR_MODE": "sector"},
           {"NEAT_SECTOR_LAYOUT": "e30", "NEAT_LIDAR_MODE": "sector", "NEAT_LIDAR_N": "32"}]
    for extra in bad:
        env = {k: v for k, v in os.environ.items() if not k.startswith("NEAT_")}
        env.update(extra)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        r = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", "import src.env.kinematics"], cwd=str(ROOT),
                           env=env, capture_output=True, text=True, encoding="utf-8")
        assert r.returncode != 0 and "ValueError" in r.stderr, (extra, r.returncode, r.stderr[-300:])


# ── chạy ─────────────────────────────────────────────────────────────────────
def _tests_for(cfg):
    return [(k, v) for k, v in sorted(globals().items())
            if k.startswith("test_") and callable(v) and cfg in getattr(v, "CONFIGS", tuple(CONFIG_ENV))]


def inner(cfg):
    from src.console import utf8_stdout
    utf8_stdout()
    rows = []
    for name, fn in _tests_for(cfg):
        try:
            fn()
            rows.append({"cfg": cfg, "test": name, "ok": True})
        except Exception as e:  # noqa: BLE001
            import traceback
            rows.append({"cfg": cfg, "test": name, "ok": False, "error": repr(e)[:1500],
                         "trace": traceback.format_exc()[-1500:]})
    print("@@RESULT@@" + json.dumps(rows, ensure_ascii=False), flush=True)


def main():
    from src.console import utf8_stdout
    utf8_stdout()
    total, failed, report = 0, 0, []
    for cfg, extra in CONFIG_ENV.items():
        env = {k: v for k, v in os.environ.items() if not k.startswith("NEAT_")}
        env.update({k: v for k, v in extra.items() if v != ""})
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        r = subprocess.run([sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()), "--inner", cfg],
                           cwd=str(ROOT), env=env, capture_output=True, text=True, encoding="utf-8")
        line = [x for x in r.stdout.splitlines() if x.startswith("@@RESULT@@")]
        if r.returncode != 0 or not line:
            print(f"  FAIL  [{cfg}] tiến trình con lỗi (mã {r.returncode}): {r.stderr[-1500:]}")
            failed += 1
            total += 1
            continue
        for row in json.loads(line[0][len("@@RESULT@@"):]):
            total += 1
            report.append(row)
            if row["ok"]:
                print(f"  OK    [{cfg}] {row['test']}")
            else:
                failed += 1
                print(f"  FAIL  [{cfg}] {row['test']}: {row['error']}\n{row.get('trace', '')}")
    print(f"\n{total - failed}/{total} passed")
    if "--json" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--json") + 1])
        out.write_text(json.dumps({"passed": total - failed, "total": total, "rows": report}, ensure_ascii=False,
                                  indent=1), encoding="utf-8", newline="\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    if "--inner" in sys.argv:
        inner(sys.argv[sys.argv.index("--inner") + 1])
    else:
        main()
