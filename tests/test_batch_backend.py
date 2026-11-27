"""
Kiểm thử nhanh đánh giá theo lô (src/batch): trùng từng bit với đường từng lượt
================================================================================
Chạy:  python tests/test_batch_backend.py           (khoảng 20–60 giây, không cần pytest, không dùng GPU)
       python tests/test_batch_backend.py --pool    (thêm: BatchEvaluator chia lô cho 2 worker)
       python tests/test_batch_backend.py --gpu     (thêm: backend gpu, cần CuPy và GPU CUDA)

Mỗi hàm test_* so một phần của bản chạy theo lô với hàm gốc: tự kiểm số học (cả phép so trọn lượt và việc nó
bắt được một thay đổi fitness mà bản theo lô chưa chép theo), tổng Neumaier, bảng mạng NEAT (hồi quy, truyền
thẳng, genome có chu trình) so với net.activate, rồi BatchEvaluator.map so với [eval_task(t) for t in tasks]
(cùng dict, cùng thứ tự khóa, cùng kiểu, float trùng từng bit kể cả dấu của số 0) trên vài bản đồ mỗi cấp 1–4,
BARN, mê cung Lehman, tiến độ geodesic và euclid, với champion demo, genome hồi quy / truyền thẳng, genome đột
biến có chu trình, GA B1/B2, APF; genome có trọng số kiểu np.float64 phải chạy bằng eval_task.
Cảm biến theo NEAT_LIDAR_MODE / NEAT_LIDAR_N như mọi script (mặc định 12 tia).
NEAT_GEO_CACHE (nếu đặt) là thư mục đệm geodesic; không đặt thì trường geodesic được tính trong bộ nhớ.
Kiểm đầy đủ (nhiều genome, mọi tập, cả hai kiểu cảm biến, NumPy và GPU): tests/verify_batch_full.py.
"""

import copy
import json
import os
import pickle
import random
import struct
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.env import kinematics as KM  # noqa: E402
from src.neat_engine import parallel as PAR  # noqa: E402

DATA = ROOT / "data" / "maps"
GEO = os.environ.get("NEAT_GEO_CACHE") or None
SUFFIX = "" if KM.LIDAR_NUM_RAYS == 12 else f"_n{KM.LIDAR_NUM_RAYS}"
REC_CFG = ROOT / "configs" / f"neat_recurrent{SUFFIX}.ini"
FF_CFG = ROOT / "configs" / f"neat_feedforward{SUFFIX}.ini"
GEO_OPTS = {"progress": "geodesic", "geo_cache": GEO}
EUC_OPTS = {"progress": "euclid"}
OPTIONS = {"pool": "--pool" in sys.argv, "gpu": "--gpu" in sys.argv}


# ── dữ liệu thử ────────────────────────────────────────────────────────────
def sample_maps() -> list[str]:
    """Một bản đồ test mỗi cấp 1–4, một bản đồ BARN, một mê cung Lehman."""
    out = [DATA / "test" / f"level_{k}" / "map_200000.json" for k in (1, 2, 3, 4)]
    out.append(sorted((DATA / "test_barn").glob("*.json"))[0])
    out.append(DATA / "test_lehman" / "lehman_medium_maze.json")
    return [str(p) for p in out]


def neat_config(path):
    import neat
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                       neat.DefaultStagnation, str(path))


def initial_genomes(cfg_path, n: int, seed: int) -> list:
    """Genome của quần thể ban đầu (nối thẳng đủ đầu vào -> đầu ra), trọng số ngẫu nhiên theo config."""
    import neat
    pop = neat.Population(neat_config(cfg_path), seed=seed)
    return [pop.population[k] for k in sorted(pop.population)[:n]]


def noisy_copies(g0, n: int, seed: int) -> list:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        g = copy.deepcopy(g0)
        for c in g.connections.values():
            c.weight += rng.gauss(0.0, 0.15)
        for nd in g.nodes.values():
            nd.bias += rng.gauss(0.0, 0.15)
        out.append(g)
    return out


def mutants(genomes, cfg_path, seed: int) -> list:
    """Bản sao có cấu trúc hồi quy tường minh: tự nối, nối ngược output -> ẩn và ẩn -> ẩn, một nút ẩn không
    có đầu vào nối vào output 0; genome thứ 0, 3, ... còn bị tắt mọi kết nối vào output 1."""
    gc = neat_config(cfg_path).genome_config
    rng = random.Random(seed)
    out, inno = [], 10_000_000
    for gi, g0 in enumerate(genomes):
        g = copy.deepcopy(g0)
        hidden = [k for k in g.nodes if k not in gc.output_keys]
        new = max(g.nodes) + 1
        ng = gc.node_gene_type(new)
        ng.init_attributes(gc)
        g.nodes[new] = ng
        hidden.append(new)
        edges = [(new, 0)] + [(o, o) for o in gc.output_keys] + [(h, h) for h in hidden[:2]]
        edges += [(o, h) for o in gc.output_keys for h in hidden[:2]]
        edges += [(hidden[i], hidden[i - 1]) for i in range(1, len(hidden))]
        for a, b in edges:
            if (a, b) not in g.connections:
                inno += 1
                g.add_connection(gc, a, b, rng.gauss(0.0, 1.5), True, innovation=inno)
        if gi % 3 == 0:
            for k, c in g.connections.items():
                if k[1] == 1:
                    c.enabled = False
        out.append(g)
    return out


def deep_ff(g0, cfg_path, seed: int):
    """Genome truyền thẳng nhiều lớp: đầu vào -> h1 -> h2 -> output, thêm nút "bias neuron" h3 (không có
    đầu vào) nối vào output 1."""
    gc = neat_config(cfg_path).genome_config
    rng = random.Random(seed)
    g = copy.deepcopy(g0)
    h1, h2, h3 = (max(g.nodes) + i for i in (1, 2, 3))
    for h in (h1, h2, h3):
        ng = gc.node_gene_type(h)
        ng.init_attributes(gc)
        ng.bias = rng.gauss(0.0, 1.0)
        g.nodes[h] = ng
    inno = 20_000_000
    for a, b in ((-1, h1), (-2, h1), (h1, h2), (-3, h2), (h2, 0), (h1, 1), (h3, 1), (h2, 1)):
        if (a, b) not in g.connections:
            inno += 1
            g.add_connection(gc, a, b, rng.gauss(0.0, 1.5), True, innovation=inno)
    return g


def demo_spec():
    champ = pickle.load(open(ROOT / "demo" / "champion_genome.pkl", "rb"))
    return PAR.neat_spec(champ, str(ROOT / "demo" / "neat_config.ini"), True), champ


def bits(v):
    """Khóa so tới từng bit: float -> 8 byte (nên -0.0 khác 0.0), tuple / list -> từng phần tử; kèm tên kiểu."""
    if type(v) is float:
        return ("float", struct.pack("<d", v))
    if isinstance(v, (tuple, list)):
        return (type(v).__name__, tuple(bits(x) for x in v))
    return (type(v).__name__, v)


def same_episode(r: dict, g: dict) -> bool:
    """Cùng khóa, cùng thứ tự khóa, cùng kiểu Python và cùng giá trị tới từng bit."""
    return list(r) == list(g) and all(bits(r[k]) == bits(g[k]) for k in r)


def same_results(ref, got) -> tuple[bool, str]:
    """Cùng dict (khóa, thứ tự khóa, giá trị tới từng bit) và cùng kiểu Python của mọi giá trị."""
    if len(ref) != len(got):
        return False, f"số tác vụ {len(ref)} != {len(got)}"
    bad = 0
    first = ""
    for rr, gg in zip(ref, got):
        if len(rr) != len(gg):
            return False, "số lượt của một tác vụ khác nhau"
        for r, g in zip(rr, gg):
            ok = same_episode(r, g) and all(type(v) is float for v in g["final_pos"])
            if not ok:
                bad += 1
                first = first or f"đường từng lượt {r} / theo lô {g}"
    return bad == 0, f"{bad} lượt lệch; {first}"


def scenarios() -> list[tuple[str, list]]:
    M = sample_maps()
    spec, champ = demo_spec()
    rec_cfg, ff_cfg = str(REC_CFG), str(FF_CFG)
    rec0 = initial_genomes(REC_CFG, 2, seed=1)
    ff0 = initial_genomes(FF_CFG, 2, seed=2)
    out = []
    if KM.LIDAR_NUM_RAYS == 12:
        out.append(("champion demo, geodesic", [(spec, M, GEO_OPTS)]))
        out.append(("champion demo, kiểu chấm test (euclid)", [(spec, [p], EUC_OPTS) for p in M]))
        out.append(("champion + nhiễu, hồi quy", [(PAR.neat_spec(g, str(ROOT / "demo" / "neat_config.ini"), True), M, GEO_OPTS)
                                                   for g in noisy_copies(champ, 2, seed=3)]))
        out.append(("champion chạy như mạng truyền thẳng", [(PAR.neat_spec(champ, str(ROOT / "demo" / "neat_config.ini"), False),
                                                              M[:3], EUC_OPTS)]))
        mut_src = [champ] + rec0
        mut_cfg = str(ROOT / "demo" / "neat_config.ini")
    else:
        mut_src, mut_cfg = rec0, rec_cfg
    out.append(("genome ban đầu, hồi quy", [(PAR.neat_spec(g, rec_cfg, True), M, GEO_OPTS) for g in rec0]))
    mut = mutants(mut_src, mut_cfg, seed=9)
    out.append(("đột biến có chu trình, hồi quy", [(PAR.neat_spec(g, mut_cfg, True), M, GEO_OPTS) for g in mut]))
    out.append(("đột biến có chu trình, truyền thẳng", [(PAR.neat_spec(g, mut_cfg, False), M[:3], EUC_OPTS) for g in mut[:2]]))
    ffs = ff0 + [deep_ff(ff0[0], ff_cfg, seed=4)]
    out.append(("truyền thẳng (có nút bias neuron, 3 lớp)", [(PAR.neat_spec(g, ff_cfg, False), M, GEO_OPTS) for g in ffs]))
    from src.baselines.fixed_ga import FixedMLP
    rng = np.random.default_rng(5)
    for name, layers in (("GA B1", [KM.OBS_DIM, 8, 2]), ("GA B2", [KM.OBS_DIM, 16, 8, 2])):
        npar = FixedMLP(layers).num_params
        out.append((name, [(PAR.mlp_spec(rng.normal(0.0, 0.3, npar), layers), M, GEO_OPTS) for _ in range(2)]))
    out.append(("APF, kiểu chấm test", [({"kind": "apf"}, [p], EUC_OPTS) for p in M]))
    return out


_SCEN: dict = {}


def cached_scenarios():
    """Kịch bản + kết quả đường từng lượt, tính một lần cho mọi phép thử."""
    if "list" not in _SCEN:
        sc = scenarios()
        t0 = time.perf_counter()
        _SCEN["list"] = [(name, tasks, [PAR.eval_task(t) for t in tasks]) for name, tasks in sc]
        _SCEN["seconds"] = time.perf_counter() - t0
    return _SCEN["list"]


# ── phép thử ───────────────────────────────────────────────────────────────
def test_self_check_passes():
    from src.batch import numerics
    problems = numerics.self_check(force=True)
    assert not problems, problems


def test_episode_check_detects_unmirrored_change():
    """Phép so trọn lượt của bước tự kiểm phải báo lệch khi fitness đọc khóa info mà bản theo lô không có
    ("trajectory"), dù đọc bằng .get có mặc định (lệch âm thầm) hay bằng [] (bản theo lô báo KeyError)."""
    from src.batch import numerics
    from src.batch import sim
    from src.neat_engine import evaluator as EV
    assert numerics.episode_check() == []
    base = EV.compute_episode_fitness
    variants = (lambda info, **kw: base(info, **kw) + 1e-3 * len(info.get("trajectory", ())),
                lambda info, **kw: base(info, **kw) + 1e-3 * len(info["trajectory"]))
    for fn in variants:
        PAR.compute_episode_fitness = sim.compute_episode_fitness = fn
        try:
            try:
                problems = numerics.episode_check()
            except KeyError as e:
                problems = [repr(e)]
        finally:
            PAR.compute_episode_fitness = sim.compute_episode_fitness = base
        assert problems, "phép so trọn lượt không thấy fitness đã đổi"
    assert numerics.episode_check() == []


def test_float64_weights_fall_back_to_eval_task():
    """Trọng số kiểu np.float64 (sum() khi đó không cộng bù): neat_slots báo Unsupported, tác vụ chạy bằng eval_task."""
    from src.batch.evaluator import BatchEvaluator
    from src.batch.policies import Unsupported, neat_slots
    cfg = str(ROOT / "demo" / "neat_config.ini") if KM.LIDAR_NUM_RAYS == 12 else str(REC_CFG)
    g = demo_spec()[1] if KM.LIDAR_NUM_RAYS == 12 else initial_genomes(REC_CFG, 1, seed=1)[0]
    g = copy.deepcopy(g)
    for c in g.connections.values():
        c.weight = np.float64(c.weight) + np.float64(1e-3)
    spec = PAR.neat_spec(g, cfg, True)
    try:
        neat_slots(PAR.make_policy(spec), True)
        raise AssertionError("neat_slots phải báo Unsupported")
    except Unsupported:
        pass
    tasks = [(spec, sample_maps()[:2], GEO_OPTS)]
    ev = BatchEvaluator()
    ok, msg = same_results([PAR.eval_task(t) for t in tasks], ev.map(tasks))
    assert ok, msg
    assert ev.stats["fallback_tasks"] == 1, ev.stats


def test_neumaier_matches_builtin_sum():
    from src.batch.policies import neumaier_sum
    rng = random.Random(7)
    rows, K = 3000, 30
    P = np.zeros((rows, K))
    fanin = np.zeros(rows, np.int64)
    for r in range(rows):
        k = rng.randint(0, K)
        vals = [rng.uniform(-3, 3) * rng.choice([1e-8, 1e-3, 1.0, 1e3, 1e16]) for _ in range(k)]
        if k and rng.random() < 0.1:
            vals[0] = -0.0
        P[r, :k] = vals
        fanin[r] = k
    got = neumaier_sum(np, P, fanin)
    ref = np.array([float(sum(P[r, :fanin[r]].tolist())) for r in range(rows)])
    assert np.array_equal(got, ref) and np.array_equal(np.signbit(got), np.signbit(ref))


def test_neat_tables_match_activate():
    """Bảng theo lô so với net.activate từng bước (60 bước đầu vào ngẫu nhiên) cho mạng hồi quy, truyền
    thẳng và mạng dựng từ genome có chu trình."""
    from src.batch.backend import Backend
    from src.batch.policies import NeatPolicy
    be = Backend("numpy")
    cfg = str(ROOT / "demo" / "neat_config.ini") if KM.LIDAR_NUM_RAYS == 12 else str(REC_CFG)
    base = [demo_spec()[1]] if KM.LIDAR_NUM_RAYS == 12 else []
    base += initial_genomes(REC_CFG, 2, seed=1)
    genomes = base + mutants(base, cfg, seed=9)
    ff0 = initial_genomes(FF_CFG, 1, seed=2)
    cases = [(cfg, True, genomes), (cfg, False, genomes), (str(FF_CFG), False, ff0 + [deep_ff(ff0[0], str(FF_CFG), 4)])]
    rng = np.random.default_rng(11)
    for cfg_path, recurrent, gs in cases:
        nets = [PAR.make_policy(PAR.neat_spec(g, cfg_path, recurrent)) for g in gs]
        pol = NeatPolicy(be, nets, recurrent, list(range(len(nets))))
        for n in nets:
            if hasattr(n, "reset"):
                n.reset()
        for _ in range(60):
            obs = rng.uniform(-1.0, 1.0, (len(nets), KM.OBS_DIM)).astype(np.float32)
            got = pol.step(obs)
            ref = np.array([n.activate(obs[i].tolist()) for i, n in enumerate(nets)])
            assert np.array_equal(got, ref), (cfg_path, recurrent, np.abs(got - ref).max())


def test_episodes_match_eval_task():
    from src.batch.evaluator import BatchEvaluator
    ev = BatchEvaluator()
    lines = []
    for name, tasks, ref in cached_scenarios():
        got = ev.map(tasks)
        ok, msg = same_results(ref, got)
        assert ok, f"{name}: {msg}"
        lines.append(f"{name}: {sum(len(r) for r in ref)} lượt")
    assert ev.stats["fallback_tasks"] == 0, ev.stats
    print("        " + "; ".join(lines) + f" (đường từng lượt {_SCEN['seconds']:.1f} s)")


def test_chunks_and_order_do_not_change_results():
    from src.batch.evaluator import BatchEvaluator
    sc = cached_scenarios()
    tasks = [t for _, ts, _ in sc for t in ts]
    ref = [r for _, _, rs in sc for r in rs]
    ev = BatchEvaluator(max_episodes=7, lidar_chunk=3)
    rev = ev.map(tasks[::-1])[::-1]
    ok, msg = same_results(ref, rev)
    assert ok, msg


def test_unsupported_tasks_fall_back_to_eval_task():
    import neat  # noqa: F401
    from src.batch.evaluator import BatchEvaluator
    M = sample_maps()[:2]
    spec, champ = demo_spec() if KM.LIDAR_NUM_RAYS == 12 else (None, None)
    if champ is None:
        return
    g = copy.deepcopy(champ)
    for nd in g.nodes.values():
        nd.activation = "sigmoid"                 # hàm kích hoạt khác tanh: chạy bằng eval_task
    tasks = [(PAR.neat_spec(g, str(ROOT / "demo" / "neat_config.ini"), True), M, GEO_OPTS)]
    ev = BatchEvaluator()
    got = ev.map(tasks)
    ok, msg = same_results([PAR.eval_task(t) for t in tasks], got)
    assert ok, msg
    assert ev.stats["fallback_tasks"] == 1, ev.stats
    bad = [(PAR.mlp_spec(np.zeros(10), [KM.OBS_DIM + 1, 2]), M[:1], EUC_OPTS)]   # sai lớp vào: cùng lỗi như cũ
    errs = []
    for fn in (lambda: PAR.eval_task(bad[0]), lambda: BatchEvaluator().map(bad)):
        try:
            fn()
            errs.append(None)
        except Exception as e:  # noqa: BLE001
            errs.append(type(e))
    assert errs[0] is not None and errs[0] == errs[1], errs


def test_make_evaluator_selection_and_interface():
    from src.batch.evaluator import BatchEvaluator
    saved = os.environ.pop("NEAT_EVAL_BACKEND", None)
    try:
        ev = PAR.make_evaluator(1)
        assert type(ev) is PAR.ParallelEvaluator and PAR.backend_label(ev) == "cpu"
        assert type(PAR.make_evaluator(1, "process")) is PAR.ParallelEvaluator
        with PAR.make_evaluator(1, "batch") as ev:
            assert isinstance(ev, BatchEvaluator) and PAR.backend_label(ev) == "batch"
            assert ev.map([]) == []
            ev.close()
            ev.close()                            # gọi lại vô hại
        assert ev.__exit__(None, None, None) is False
        os.environ["NEAT_EVAL_BACKEND"] = "batch"
        assert PAR.resolve_backend(None) == "batch" and PAR.resolve_backend("cpu") == "cpu"
        assert isinstance(PAR.make_evaluator(1), BatchEvaluator)
        for bad_call in (lambda: PAR.resolve_backend("tpu"), lambda: PAR.make_evaluator(1, "batch", exact=False),
                         lambda: PAR.make_evaluator(1, "cpu", exact=False)):
            try:
                bad_call()
                raise AssertionError("phải báo ValueError")
            except ValueError:
                pass
    finally:
        os.environ.pop("NEAT_EVAL_BACKEND", None)
        if saved is not None:
            os.environ["NEAT_EVAL_BACKEND"] = saved


def test_numerics_mismatch_falls_back_to_default():
    """Tự kiểm số học không đạt -> make_evaluator cảnh báo và dùng ParallelEvaluator (nhãn "batch->cpu");
    với NEAT_EVAL_STRICT=1 hoặc strict=True thì báo lỗi NumericsMismatch."""
    import warnings
    from src.batch import numerics
    from src.batch.evaluator import NumericsMismatch
    saved = numerics.self_check
    saved_strict = os.environ.pop("NEAT_EVAL_STRICT", None)
    numerics.self_check = lambda force=False: ["giả lập lệch"]
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ev = PAR.make_evaluator(1, "batch")
        assert type(ev) is PAR.ParallelEvaluator and PAR.backend_label(ev) == "batch->cpu"
        assert any("giả lập lệch" in str(x.message) for x in w)
        os.environ["NEAT_EVAL_STRICT"] = "1"
        for call in (lambda: PAR.make_evaluator(1, "batch"), lambda: PAR.make_evaluator(1, "batch", strict=True)):
            try:
                call()
                raise AssertionError("strict phải báo NumericsMismatch")
            except NumericsMismatch:
                pass
        assert type(PAR.make_evaluator(1, "cpu")) is PAR.ParallelEvaluator   # đường mặc định không bị ảnh hưởng
    finally:
        numerics.self_check = saved
        os.environ.pop("NEAT_EVAL_STRICT", None)
        if saved_strict is not None:
            os.environ["NEAT_EVAL_STRICT"] = saved_strict


def test_evaluate_sets_backend_batch_equals_default():
    from src.experiments.evaluation import evaluate_sets
    M = sample_maps()
    sets = {"test/level_1": M[:2], "test/level_3": M[2:4], "test_barn": M[4:5]}
    a = evaluate_sets({"kind": "apf"}, sets, evaluator=PAR.ParallelEvaluator(1))
    b = evaluate_sets({"kind": "apf"}, sets, workers=1, backend="batch")
    assert a == b, (a, b)


def test_runner_records_backend_only_when_not_default():
    from src.neat_engine.runner import NEATRunner
    with tempfile.TemporaryDirectory() as tmp:
        keys = {}
        for label in ("cpu", "batch"):
            r = NEATRunner(REC_CFG, DATA, Path(tmp) / label, use_novelty=False, use_curriculum=False,
                           max_generations=1, maps_per_eval=4, verbose=False, backend=label)
            r.backend_used = label
            r._save(None, None)
            keys[label] = json.load(open(Path(tmp) / label / "summary.json", encoding="utf-8"))
        assert "backend" not in keys["cpu"] and keys["batch"]["backend"] == "batch"
        assert [k for k in keys["batch"] if k != "backend"] == list(keys["cpu"])


def test_runner_runs_with_batch_backend():
    """Một thế hệ NEATRunner với backend batch chạy hết vòng (map, close, __exit__) và ghi "backend"."""
    from src.neat_engine.runner import NEATRunner
    with tempfile.TemporaryDirectory() as tmp:
        r = NEATRunner(REC_CFG, DATA, Path(tmp), use_novelty=False, use_curriculum=False, max_generations=1,
                       maps_per_eval=4, val_every=1, verbose=False, backend="batch", progress="euclid")
        r.run()
        s = json.load(open(Path(tmp) / "summary.json", encoding="utf-8"))
        h = json.load(open(Path(tmp) / "training_history.json", encoding="utf-8"))
    assert s["backend"] == "batch" and len(h) == 1 and h[0]["champion_check"] is not None, (s, h)


def test_cupy_short_path_fix_without_gpu():
    """Đường dẫn có dấu: prefer_ascii_cupy_path chèn tên ngắn 8.3 của site-packages để CuPy được nạp qua
    đường dẫn ASCII (chỉ kiểm việc tìm module, không khởi động CUDA)."""
    import importlib.util
    from src.batch import backend as B
    if "cupy" in sys.modules:
        print("        (bỏ qua: CuPy đã được import)")
        return
    spec = importlib.util.find_spec("cupy")
    if spec is None or spec.origin.isascii():
        print("        (bỏ qua: " + ("không có CuPy" if spec is None else "CuPy đã nằm dưới đường dẫn ASCII; muốn kiểm "
                                    "phần sửa này thì chạy python của .venv bằng đường dẫn thật, có dấu") + ")")
        return
    saved = list(sys.path)
    try:
        short = B.prefer_ascii_cupy_path()
        assert short and short.isascii(), short
        assert importlib.util.find_spec("cupy").origin.isascii()
        site = os.path.normcase(os.path.dirname(os.path.dirname(os.path.abspath(spec.origin))))
        norm = [os.path.normcase(os.path.abspath(p)) if p else p for p in sys.path]
        assert norm.index(os.path.normcase(os.path.abspath(short))) < norm.index(site), "phải đứng ngay trước site-packages"
        assert norm.index(os.path.normcase(str(ROOT))) < norm.index(os.path.normcase(os.path.abspath(short))), \
            "thư mục repo phải đứng trước"
    finally:
        sys.path[:] = saved
        importlib.invalidate_caches()


def test_pool_matches_in_process():
    if not OPTIONS["pool"]:
        print("        (bỏ qua: thêm --pool để chạy)")
        return
    from src.batch.evaluator import BatchEvaluator
    sc = cached_scenarios()
    tasks = [t for _, ts, _ in sc for t in ts]
    ref = [r for _, _, rs in sc for r in rs]
    with BatchEvaluator(workers=2) as ev:
        got = ev.map(tasks)
        ev.close()
    ok, msg = same_results(ref, got)
    assert ok, msg


def test_gpu_backend():
    if not OPTIONS["gpu"]:
        print("        (bỏ qua: thêm --gpu để chạy)")
        return
    from src.batch.evaluator import BatchEvaluator
    from src.batch.gpu import GpuGeometry
    from src.batch.numerics import random_poses, synthetic_maps
    from src.batch.sim import MapInfo
    from src.env.lidar import FastLiDAR2D
    # LiDAR giữ lại sau collide() chỉ được dùng khi scan() ở đúng cùng tư thế
    lid = FastLiDAR2D(KM.LIDAR_NUM_RAYS, KM.LIDAR_MAX_RANGE, KM.LIDAR_MODE)
    rng = np.random.default_rng(3)
    segs = synthetic_maps(rng)[0]
    x, y, th, _ = random_poses(rng, 32, segs)
    geom = GpuGeometry(lid)
    geom.setup([MapInfo("check", {}, segs, np.zeros(2), np.zeros(2), 0.0, 0.0, None, None, 0.0)] * 32)
    th = np.array(th)
    geom.collide(x, y, th)
    got = geom.scan(x + 0.1, y, th + 0.2)
    ref = np.array([lid.scan(np.array([x[i] + 0.1, y[i]]), float(th[i] + 0.2), segs) for i in range(32)])
    assert np.array_equal(got, ref), "scan() ở tư thế mới trả LiDAR cũ"
    sc = cached_scenarios()
    with BatchEvaluator(device="gpu") as ev:
        for name, tasks, ref in sc:
            ok, msg = same_results(ref, ev.map(tasks))
            assert ok, f"GPU {name}: {msg}"
    with BatchEvaluator(device="gpu", exact=False) as ev:   # chế độ device: chỉ cần chạy được
        n = sum(len(r) for r in ev.map([t for _, ts, _ in sc for t in ts]))
    print(f"        GPU exact trùng bit; chế độ device chạy {n} lượt (không so)")


def main():
    from src.console import utf8_stdout
    utf8_stdout()
    print(f"cảm biến {KM.LIDAR_MODE}:{KM.LIDAR_NUM_RAYS}, numpy {np.__version__}, đệm geodesic {GEO or '(trong bộ nhớ)'}")
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    t_all = time.perf_counter()
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            print(f"  OK    {name} ({time.perf_counter() - t0:.1f} s)")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed ({time.perf_counter() - t_all:.1f} s)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
