"""
Kiểm đầy đủ: chạy theo lô (NumPy, GPU) trùng từng bit với đường từng lượt
==========================================================================
Chạy lâu (khoảng 5–30 phút tùy tùy chọn): nên chạy nền ở mức ưu tiên thấp.

    python -B tests/verify_batch_full.py                          # NumPy, cảm biến rays:12, sector:12, sector:24
    python -B tests/verify_batch_full.py --device cpu,gpu         # thêm GPU (biến thể lai, exact)
    python -B tests/verify_batch_full.py --device cpu,gpu --workers 4 --timing --priority idle

Tùy chọn:
  --sensors rays:12,sector:12,sector:24   cảm biến cần kiểm; "current" = robot và cảm biến của bốn biến môi
                                          trường hiện tại (NEAT_ROBOT, NEAT_LIDAR_MODE, NEAT_LIDAR_N,
                                          NEAT_SECTOR_LAYOUT), cách kiểm Waffle và bố cục e32; đứng một mình
                                          thì chạy ngay trong tiến trình này, đứng chung danh sách thì chạy
                                          trong tiến trình con với bốn biến giữ nguyên
  --device cpu[,gpu]                      gpu: BatchEvaluator(device="gpu") exact phải 0 lệch; chế độ device
                                          (không exact) chỉ in thống kê
  --gpu-full                              thêm biến thể "full" exact (mọi mảng trên CuPy, chậm; bản tham chiếu)
  --workers W                             thêm phép thử Pool W worker (W > 1)
  --timing                                đo một thế hệ kiểu huấn luyện (150 genome x 8 bản đồ, cấp 1–4 và BARN)
  --quick / --smoke                       ít genome / bản đồ hơn; --smoke rất nhỏ, chỉ để thử script
  --priority idle|below|normal            mức ưu tiên của tiến trình (Windows; tiến trình con thừa hưởng)
Mỗi cảm biến dạng kiểu:N chạy trong một tiến trình con (NEAT_LIDAR_MODE / NEAT_LIDAR_N được đọc lúc import
src). Tiến trình con này chạy robot cũ với quạt đều: NEAT_ROBOT và NEAT_SECTOR_LAYOUT bị bỏ khỏi môi trường của
nó, nên rays:12 vẫn chạy được khi đang đặt NEAT_ROBOT=waffle. "current" giữ nguyên cả bốn biến.
N khác 12 dùng configs/neat_*_n{N}.ini. NEAT_GEO_CACHE (nếu đặt) là thư mục đệm geodesic.
Kịch bản APF được bỏ qua khi APF không chạy được với robot / cảm biến đó (apf_unsupported_reason, ví dụ e32).
Mỗi kịch bản in: số lượt, số dict lệch (so tới từng bit: khóa, thứ tự khóa, kiểu, float kể cả dấu của số 0),
số lượt đổi kết cục (tới đích / va chạm), max |Δfitness|, max |Δfinal_pos|, max |Δsteps|, số giá trị lệch kiểu, thời
gian. Exact phải 0 lệch; mã thoát 1 nếu có lệch. GPU không dựng được (thiếu CuPy / driver, tự kiểm lệch) thì in lý do,
bỏ phần GPU, vẫn chạy phần CPU và thoát với mã 1.
Nếu repo nằm dưới thư mục có dấu, chạy GPU bằng python qua đường dẫn ngắn 8.3 cho chắc
(src/batch/backend.py tự chèn đường dẫn ngắn của CuPy, đã thử được với python qua đường dẫn thật).
"""

import argparse
import copy
import math
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

SENSORS_DEFAULT = "rays:12,sector:12,sector:24"


def set_priority(level: str) -> None:
    if os.name != "nt" or level == "normal":
        return
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    k32.SetPriorityClass.restype = wintypes.BOOL
    k32.SetPriorityClass(k32.GetCurrentProcess(), {"idle": 0x40, "below": 0x4000}[level])


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Kiểm đầy đủ chạy theo lô so với đường từng lượt")
    ap.add_argument("--sensors", default=SENSORS_DEFAULT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gpu-full", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--priority", choices=["idle", "below", "normal"], default="normal")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return ap.parse_args(argv)


# ── phần chạy trong tiến trình con (một cảm biến) ─────────────────────────
def run_suite(args) -> int:
    import numpy as np
    import neat
    from src.env import kinematics as KM
    from src.neat_engine import parallel as PAR
    from src.batch.evaluator import BatchEvaluator
    from src.baselines.fixed_ga import FixedMLP
    import test_batch_backend as T

    N = KM.LIDAR_NUM_RAYS
    suf = "" if N == 12 else f"_n{N}"
    data = ROOT / "data" / "maps"
    geo = os.environ.get("NEAT_GEO_CACHE") or None
    GEO = {"progress": "geodesic", "geo_cache": geo}
    EUC = {"progress": "euclid"}
    devices = [d.strip() for d in args.device.split(",") if d.strip()]
    size = "smoke" if args.smoke else ("quick" if args.quick else "full")
    per_level = {"smoke": 1, "quick": 2, "full": 3}[size]
    n_gen = {"smoke": 2, "quick": 8, "full": 16}[size]
    n_mlp = {"smoke": 1, "quick": 6, "full": 10}[size]
    tag = (f"{KM.LIDAR_MODE}:{N}" + (f" {KM.SECTOR_LAYOUT}" if KM.SECTOR_LAYOUT else "")
           + (f", robot {KM.ROBOT}" if KM.ROBOT != "old" else ""))
    print(f"\n=== cảm biến {tag} | numpy {np.__version__} | {size} | thiết bị {devices} ===", flush=True)

    maps = []
    for lvl in (1, 2, 3, 4):
        maps += sorted((data / "test" / f"level_{lvl}").glob("*.json"))[:per_level]
    barn = sorted((data / "test_barn").glob("*.json"))
    maps += [barn[0]] if size == "smoke" else [barn[0], barn[len(barn) // 2], barn[-1]]
    if size != "smoke":
        import json
        allb = sorted((data / "barn_all").glob("*.json"))
        maps.append(max(allb, key=lambda p: len(json.load(open(p, encoding="utf-8"))["segments"])))   # bản đồ nhiều đoạn nhất
    maps += sorted((data / "test_lehman").glob("*.json"))
    M = [str(p) for p in maps]
    for p in M:                                   # nạp sẵn ngoài phép đo thời gian
        PAR.load_map(p)
        PAR._geodesic(p, geo)

    def evolve(cfg_path, n, gens, seed):
        cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                          neat.DefaultStagnation, str(cfg_path))
        pop = neat.Population(cfg, seed=seed)
        rng = random.Random(seed)

        def fn(genomes, config):
            for _, g in genomes:
                g.fitness = rng.random()
        pop.run(fn, gens)
        gs = sorted(pop.population.values(), key=lambda g: (-len(g.nodes), -len(g.connections), g.key))
        return gs[:n]

    rec_cfg = ROOT / "configs" / f"neat_recurrent{suf}.ini"
    ff_cfg = ROOT / "configs" / f"neat_feedforward{suf}.ini"
    gens = 5 if size == "smoke" else 40
    rec = evolve(rec_cfg, n_gen, gens, seed=1)
    ff = evolve(ff_cfg, n_gen, gens, seed=2)
    mut = T.mutants(rec[:6] + ff[:3], str(rec_cfg), seed=9)
    scen = []
    if N == 12:
        spec, champ = T.demo_spec()
        scen.append(("champion demo, kiểu chấm test", [(spec, [p], EUC) for p in M]))
        scen.append(("champion demo, geodesic", [(spec, M, GEO)]))
    scen += [
        ("NEAT hồi quy, geodesic", [(PAR.neat_spec(g, str(rec_cfg), True), M, GEO) for g in rec]),
        ("NEAT truyền thẳng, geodesic", [(PAR.neat_spec(g, str(ff_cfg), False), M, GEO) for g in ff]),
        ("NEAT truyền thẳng nhiều lớp + bias neuron", [(PAR.neat_spec(T.deep_ff(g, str(ff_cfg), 4 + i), str(ff_cfg), False), M, GEO)
                                                       for i, g in enumerate(ff[:3])]),
        ("genome hồi quy chạy như mạng truyền thẳng, euclid", [(PAR.neat_spec(g, str(rec_cfg), False), M, EUC) for g in rec[:6]]),
        ("đột biến có chu trình, hồi quy", [(PAR.neat_spec(g, str(rec_cfg), True), M, GEO) for g in mut]),
        ("đột biến có chu trình, truyền thẳng", [(PAR.neat_spec(g, str(rec_cfg), False), M, EUC) for g in mut]),
    ]
    rng = np.random.default_rng(5)
    for name, layers in (("GA B1", [KM.OBS_DIM, 8, 2]), ("GA B2", [KM.OBS_DIM, 16, 8, 2])):
        npar = FixedMLP(layers).num_params
        scen.append((name, [(PAR.mlp_spec(rng.normal(0.0, 0.3, npar), layers), M, GEO) for _ in range(n_mlp)]))
    from src.baselines.apf import apf_unsupported_reason
    apf_reason = apf_unsupported_reason()
    if apf_reason is None:
        scen.append(("APF, kiểu chấm test", [({"kind": "apf"}, [p], EUC) for p in M]))
    else:
        print(f"  (bỏ qua kịch bản APF: {apf_reason})", flush=True)

    def stats(name, ref, got, t_ref, t_got, count=True):
        n = mism = outc = dst = badtype = 0
        dfit = dpos = 0.0
        for rr, gg in zip(ref, got):
            for r, g in zip(rr, gg):
                n += 1
                mism += not T.same_episode(r, g)          # khóa, thứ tự khóa, kiểu, float tới từng bit (cả -0.0)
                badtype += sum(type(r[k]) is not type(g.get(k)) for k in r)
                outc += (r["success"], r["collision"]) != (g["success"], g["collision"])
                dfit = max(dfit, abs(r["fitness"] - g["fitness"]))
                dpos = max(dpos, math.hypot(r["final_pos"][0] - g["final_pos"][0], r["final_pos"][1] - g["final_pos"][1]))
                dst = max(dst, abs(r["steps"] - g["steps"]))
        if len(ref) != len(got) or any(len(a) != len(b) for a, b in zip(ref, got)):
            mism += 1
        print(f"  {name:58s} lượt={n:5d} dict!={mism:4d} kết cục!={outc:3d} max|dfit|={dfit:.1e} max|dpos|={dpos:.1e} "
              f"max|dsteps|={dst:3d} kiểu!={badtype:3d} | từng lượt {t_ref:6.1f}s theo lô {t_got:6.1f}s", flush=True)
        return (mism + badtype) if count else 0

    fails = 0
    evs = {"cpu": BatchEvaluator()}
    if "gpu" in devices:
        from src.batch import GpuUnavailable, NumericsMismatch
        for label, kw in (("gpu", {}), ("gpu-full", {"gpu_variant": "full"})):
            if label == "gpu-full" and (not args.gpu_full or "gpu" not in evs):
                continue
            try:
                evs[label] = BatchEvaluator(device="gpu", **kw)
            except (GpuUnavailable, NumericsMismatch) as e:      # vẫn chạy phần CPU, nhưng mã thoát khác 0
                fails += 1
                print(f"  [{label}] KHÔNG DỰNG ĐƯỢC ({type(e).__name__}): {e}; bỏ phần {label}, vẫn chạy phần CPU",
                      flush=True)
        if "gpu" not in evs:
            devices = [d for d in devices if d != "gpu"]
    all_tasks, all_ref = [], []
    for name, tasks in scen:
        t0 = time.perf_counter()
        ref = [PAR.eval_task(t) for t in tasks]
        t_ref = time.perf_counter() - t0
        all_tasks += tasks
        all_ref += ref
        for label, ev in evs.items():
            t0 = time.perf_counter()
            got = ev.map(tasks)
            fails += stats(f"[{label}] {name}", ref, got, t_ref, time.perf_counter() - t0)
    for label, ev in evs.items():
        if ev.stats["fallback_tasks"]:
            print(f"  [{label}] cảnh báo: {ev.stats['fallback_tasks']} tác vụ chạy bằng eval_task (không theo lô)")
    # cùng tác vụ, khúc / thứ tự / số worker khác -> kết quả y hệt
    order = list(range(len(all_tasks)))
    random.Random(3).shuffle(order)
    t0 = time.perf_counter()
    shuf = BatchEvaluator(max_episodes=50, lidar_chunk=7).map([all_tasks[i] for i in order])
    back = [None] * len(order)
    for j, i in enumerate(order):
        back[i] = shuf[j]
    fails += stats("[cpu] mọi kịch bản, thứ tự xáo, khúc 50 lượt, lidar_chunk 7", all_ref, back, 0.0, time.perf_counter() - t0)
    if args.workers > 1:
        t0 = time.perf_counter()
        with BatchEvaluator(workers=args.workers) as ev:
            got = ev.map(all_tasks)
            ev.close()
        fails += stats(f"[cpu] mọi kịch bản, Pool {args.workers} worker", all_ref, got, 0.0, time.perf_counter() - t0)
    if "gpu" in devices:
        with BatchEvaluator(device="gpu", exact=False) as ev:
            t0 = time.perf_counter()
            got = ev.map(all_tasks)
            stats("[gpu-device, KHÔNG exact, chỉ thống kê] mọi kịch bản", all_ref, got, 0.0, time.perf_counter() - t0,
                  count=False)
    if args.timing:
        timing(args, devices, KM, PAR, BatchEvaluator, np, champ_spec=(T.demo_spec() if N == 12 else None), rec=rec,
               rec_cfg=rec_cfg, geo=geo)
    for ev in evs.values():
        ev.close()
    print(f"=== cảm biến {tag}: TỔNG LỆCH (exact) = {fails} ===", flush=True)
    return 1 if fails else 0


def timing(args, devices, KM, PAR, BatchEvaluator, np, champ_spec, rec, rec_cfg, geo):
    """Một thế hệ kiểu huấn luyện: 150 genome (champion hoặc genome tiến hóa ngẫu nhiên, cộng nhiễu) x 8 bản đồ."""
    rng = random.Random(0)
    if champ_spec is not None:
        base, cfg = champ_spec[1], str(ROOT / "demo" / "neat_config.ini")
    else:
        base, cfg = rec[0], str(rec_cfg)
    pop = []
    for _ in range(150):
        g = copy.deepcopy(base)
        for c in g.connections.values():
            c.weight += rng.gauss(0.0, 0.1)
        for nd in g.nodes.values():
            nd.bias += rng.gauss(0.0, 0.1)
        pop.append(g)
    data = ROOT / "data" / "maps"
    for kind in ("cấp 1–4", "BARN"):
        if kind == "BARN":
            ps = sorted((data / "barn_all").glob("*.json"))
            maps = [str(ps[i]) for i in range(0, len(ps), len(ps) // 8)][:8]
        else:
            maps = []
            for lvl in (1, 2, 3, 4):
                ps = sorted((data / "train" / f"level_{lvl}").glob("*.json"))
                maps += [str(ps[7]), str(ps[101])]
        for p in maps:
            PAR.load_map(p)
            PAR._geodesic(p, geo)
        tasks = [(PAR.neat_spec(g, cfg, True), maps, {"progress": "geodesic", "geo_cache": geo}) for g in pop]
        sub = 12 if args.smoke else 24
        t0 = time.perf_counter()
        ref = [PAR.eval_task(t) for t in tasks[:sub]]
        t_ref = (time.perf_counter() - t0) * len(tasks) / sub
        row = [f"từng lượt 1 tiến trình ~{t_ref:5.1f}s (ước từ {sub} genome)"]
        t0 = time.perf_counter()
        got = BatchEvaluator().map(tasks)
        row.append(f"theo lô 1 tiến trình {time.perf_counter() - t0:5.1f}s (trùng: {got[:sub] == ref})")
        if args.workers > 1:
            with BatchEvaluator(workers=args.workers) as ev:
                ev.map(tasks[:args.workers])              # khởi động worker ngoài phép đo
                t0 = time.perf_counter()
                gp = ev.map(tasks)
                row.append(f"theo lô Pool {args.workers} {time.perf_counter() - t0:5.1f}s (trùng: {gp == got})")
                ev.close()
        if "gpu" in devices:
            with BatchEvaluator(device="gpu") as ev:
                ev.map(tasks[:2])
                t0 = time.perf_counter()
                gg = ev.map(tasks)
                row.append(f"GPU lai {time.perf_counter() - t0:5.1f}s (trùng: {gg == got})")
        mean_steps = np.mean([r["steps"] for rr in got for r in rr])
        print(f"  thời gian một thế hệ 150 x 8, {kind} (TB {mean_steps:.0f} bước): " + " | ".join(row), flush=True)


# ── tiến trình cha: mỗi cảm biến một tiến trình con ───────────────────────
def main():
    args = parse_args()
    set_priority(args.priority)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    if args.child or args.sensors.strip() == "current":
        sys.exit(run_suite(args))
    sensors = [s.strip() for s in args.sensors.split(",") if s.strip()]
    dropped = [f"{k}={os.environ[k]}" for k in ("NEAT_ROBOT", "NEAT_SECTOR_LAYOUT") if os.environ.get(k, "").strip()]
    if dropped and any(s != "current" for s in sensors):
        print(f"Lưu ý: cảm biến dạng kiểu:N chạy robot cũ với quạt đều (bỏ {', '.join(dropped)}); muốn kiểm robot / "
              "bố cục đang đặt thì dùng --sensors current.", flush=True)
    fails = 0
    t0 = time.time()
    for sensor in sensors:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        if sensor != "current":                   # kiểu:N: đúng cảm biến ghi trong --sensors, robot cũ, quạt đều
            mode, n = sensor.split(":")
            env.update(NEAT_LIDAR_MODE=mode, NEAT_LIDAR_N=str(int(n)))
            env.pop("NEAT_ROBOT", None)
            env.pop("NEAT_SECTOR_LAYOUT", None)
        env.pop("NEAT_EVAL_BACKEND", None)
        cmd = [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "--child", "--device", args.device,
               "--workers", str(args.workers), "--priority", "normal"]
        cmd += [f for f, on in (("--gpu-full", args.gpu_full), ("--timing", args.timing), ("--quick", args.quick),
                                ("--smoke", args.smoke)) if on]
        rc = subprocess.call(cmd, env=env)
        print(f"--> {sensor}: {'ĐẠT' if rc == 0 else 'CÓ LỆCH / LỖI (mã ' + str(rc) + ')'}", flush=True)
        fails += rc != 0
    print(f"\nXong sau {time.time() - t0:.0f} s: {'mọi cảm biến ĐẠT (0 lệch)' if fails == 0 else f'{fails} cảm biến có lệch / lỗi'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
