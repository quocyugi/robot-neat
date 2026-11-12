"""
Kiểm thử nhanh cho các lỗi đã sửa (mã lỗi theo báo cáo, mục 9)
================================================================
Chạy:  python tests/test_fixes.py        (khoảng 10–30 giây, không cần pytest)
Hoặc:  python -m pytest tests/test_fixes.py

Mỗi hàm test_* kiểm một lỗi; tên hàm ghi mã lỗi. Phép thử tích hợp (cùng seed
cho cùng log, chạy thật NEATRunner) nằm ở tests/test_reproducible.py.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

DATA = ROOT / "data" / "maps"


def _map(rel: str) -> dict:
    return json.load(open(DATA / rel, encoding="utf-8"))


def test_L12_rule_based_turns_away_from_obstacle():
    from src.baselines.rule_based import RuleBasedController
    ctl = RuleBasedController()
    obs = [0.0] * 15
    obs[13], obs[14] = 0.0, 1.0          # đích ngay phía trước
    obs[0] = 0.9                          # vật cản phía trước
    obs[7] = obs[8] = obs[9] = 0.8        # bên phải (tia 6–10) bị chắn
    oL, oR = ctl.activate(obs)
    assert oR > oL, "bên phải bị chắn thì phải rẽ trái (oR > oL)"
    obs[7] = obs[8] = obs[9] = 0.0
    obs[2] = obs[3] = obs[4] = 0.8        # bên trái (tia 1–5) bị chắn
    oL, oR = ctl.activate(obs)
    assert oL > oR, "bên trái bị chắn thì phải rẽ phải (oL > oR)"


def test_L20_goal_and_collision_same_step_counts_as_success():
    from src.env.robot_env import RobotNavEnv
    m = {"segments": [[[0, 0], [10, 0]], [[10, 0], [10, 10]], [[10, 10], [0, 10]], [[0, 10], [0, 0]],
                      [[4.0, 4.94], [6.0, 4.94]]],
         "start": [5.0, 4.0], "goal": [5.0, 4.97], "start_heading": float(np.pi / 2)}
    env = RobotNavEnv(map_data=m)
    env.reset()
    terminated = False
    while not terminated:
        _, _, terminated, truncated, info = env.step(np.array([1.0, 1.0], dtype=np.float32))
        assert not truncated
    assert info["is_success"] and not info["is_collision"], info


def test_start_heading_is_used():
    from src.env.robot_env import RobotNavEnv
    m = _map("test_barn/" + sorted(p.name for p in (DATA / "test_barn").glob("*.json"))[0])
    env = RobotNavEnv(map_data=m)
    env.reset()
    assert abs(env.theta - m.get("start_heading", 0.0)) < 1e-9


def test_L26_geodesic_field_and_fitness():
    from src.env.geodesic import compute_geodesic, GeodesicField
    from src.env.robot_env import RobotNavEnv
    from src.neat_engine.evaluator import compute_episode_fitness
    m = _map("test/level_3/" + sorted(p.name for p in (DATA / "test" / "level_3").glob("*.json"))[0])
    geo = GeodesicField(compute_geodesic(m["segments"], m["goal"]))
    g0 = geo.at(*m["start"])
    l_opt = m["metrics"]["optimal_path_length"]
    assert geo.at(*m["goal"]) < 0.1
    assert 0.9 * l_opt <= g0 <= 1.25 * l_opt, (g0, l_opt)
    # lượt đứng yên: tiến độ 0, fitness = offset trừ phạt quay tại chỗ (bằng 0)
    env = RobotNavEnv(map_data=m, geodesic=geo)
    env.reset()
    for _ in range(3):
        _, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert info["g0"] == g0 and abs(info["g_min"] - g0) < 1e-9
    f = compute_episode_fitness(dict(info, steps=500), progress="geodesic")
    assert abs(f - 3.0) < 1e-9, f


def test_L24_gain_reparametrisation_is_exact():
    from src.baselines.fixed_ga import FixedMLP
    rng = np.random.default_rng(0)
    a, b = FixedMLP([15, 8, 2], gain=2.5), FixedMLP([15, 8, 2], gain=1.0)
    theta = rng.normal(0, 0.2, a.num_params)
    a.set_params(theta)
    b.set_params(2.5 * theta)
    for _ in range(20):
        x = rng.uniform(-1, 1, 15).tolist()
        assert np.allclose(a.activate(x), b.activate(x), atol=1e-12)


def test_L06_descriptor_keeps_each_map():
    from src.neat_engine.novelty import behavior_descriptor
    d1 = behavior_descriptor([(1.0, 1.0), (9.0, 9.0)])
    d2 = behavior_descriptor([(5.0, 5.0), (5.0, 5.0)])   # cùng trung bình với d1
    assert d1.shape == (4,) and not np.allclose(d1, d2)


def test_L08_hybrid_uses_fixed_scale():
    from src.neat_engine.novelty import AdaptiveHybridManager
    h = AdaptiveHybridManager(fixed_lambda=1.0, fitness_scale=28.0)
    raw = np.array([7.0, 14.0])
    s1 = h.combine_scores(raw, np.zeros(2), np.full(2, 100), np.zeros(2, bool))
    s2 = h.combine_scores(np.array([7.0, 28.0]), np.zeros(2), np.full(2, 100), np.zeros(2, bool))
    assert np.isclose(s1[0], 0.25) and np.isclose(s2[0], 0.25), "cùng fitness thô phải cùng điểm, không phụ thuộc thế hệ"


def test_L09_lambda_not_stuck_and_resets():
    from src.neat_engine.novelty import AdaptiveHybridManager
    h = AdaptiveHybridManager()
    rng = np.random.default_rng(1)
    lams = [h.update(10 + 0.05 * g + rng.normal(0, 0.5)) for g in range(60)]   # tăng chậm, có nhiễu
    assert np.mean(np.array(lams) <= h.exploration_lambda) < 0.5, "fitness vẫn tăng thì không được kẹt ở chế độ khám phá"
    assert min(lams) >= h.exploration_lambda
    h.reset()
    assert h.current_lambda == h.start_lambda


def test_L03_M4_is_pure_novelty():
    from src.experiments.run_ablation import ABLATION_CONFIGS
    assert ABLATION_CONFIGS["M4"].get("fixed_lambda") == 0.0
    assert ABLATION_CONFIGS["M6"].get("fixed_lambda") is None


def test_L05_L34_schedules():
    from src.neat_engine.curriculum import CurriculumManager, UniformSchedule
    u = UniformSchedule(DATA, rng=np.random.default_rng(0))
    maps = u.get_train_maps(8)
    levels = sorted(int(Path(p).parent.name.split("_")[1]) for p in maps)
    assert levels == [1, 1, 2, 2, 3, 3, 4, 4], levels
    c = CurriculumManager(DATA, rng=np.random.default_rng(0), max_gens_per_level=3)
    for _ in range(3):
        c.tick()
    assert c.check_promotion(0.1, generation=3) == "max_gens" and c.current_level == 2
    lv = [int(Path(p).parent.name.split("_")[1]) for p in c.get_train_maps(8)]
    assert lv.count(2) == 4 and lv.count(1) == 4, "từ cấp 2: nửa bản đồ cấp hiện tại, nửa cấp thấp hơn (L10)"
    assert c.check_promotion(0.9, generation=4) == "val_sr" and c.current_level == 3


def test_L23_missing_maps_raise():
    from src.neat_engine.curriculum import UniformSchedule
    try:
        UniformSchedule(ROOT / "khong_ton_tai")
    except FileNotFoundError:
        return
    raise AssertionError("thiếu bản đồ phải báo lỗi")


def test_L27_level3_walls_are_thick_and_attached():
    m = _map("train/level_3/" + sorted(p.name for p in (DATA / "train" / "level_3").glob("*.json"))[0])
    segs = np.array(m["segments"][4:])                 # bỏ 4 tường biên
    ys = np.round(np.concatenate([segs[:, 0, 1], segs[:, 1, 1]]), 6)
    xs = np.concatenate([segs[:, 0, 0], segs[:, 1, 0]])
    assert len(segs) % 4 == 0, "mỗi tường là hình chữ nhật 4 cạnh"
    gaps = sorted(set(ys))
    assert any(abs((b - a) - 0.20) < 1e-6 for a, b in zip(gaps, gaps[1:])), "tường dày 0,20 m"
    assert xs.min() <= 1e-9 or xs.max() >= 10.0 - 1e-9, "tường nối liền tường biên"


def test_L21_recurrent_inferred_from_config():
    import neat
    for name, rec in (("neat_feedforward.ini", False), ("neat_recurrent.ini", True)):
        cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                          neat.DefaultStagnation, str(ROOT / "configs" / name))
        assert (not cfg.genome_config.feed_forward) == rec


def test_config_files_load_without_utf8_mode():
    raw = [(ROOT / "configs" / n).read_bytes() for n in ("neat_recurrent.ini", "neat_feedforward.ini")]
    for r in raw:
        r.decode("cp1252")
        r.decode("cp1258")


def main():
    from src.console import utf8_stdout
    utf8_stdout()
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  OK    {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
