"""
Parallel Episode Evaluation (sửa lỗi L13)
==========================================
Chạy các lượt (episode) trên nhiều tiến trình bằng multiprocessing.Pool kiểu
"spawn" (chạy được trên Windows). Dùng chung cho NEATRunner, GA/baseline
(scripts/run_baselines.py) và scripts/evaluate.py.

Thiết kế (theo bảng "Biến cố có thể gặp" của báo cáo):
- neat.ParallelEvaluator chỉ trả một số fitness cho mỗi genome, không trả được
  vị trí cuối, số bước, va chạm cần cho novelty, nên dùng Pool riêng.
- Tiến trình cha chỉ gửi genome (hoặc vector trọng số) và ĐƯỜNG DẪN bản đồ.
  Mỗi worker tự nạp config NEAT, bản đồ và trường geodesic rồi giữ trong bộ nhớ,
  không pickle cả config (pickle config làm lệch bộ đếm khóa nút).
- Lượt chạy tất định (không có nhiễu), nên kết quả không phụ thuộc số worker.
- Worker đọc lại cảm biến (NEAT_LIDAR_MODE / NEAT_LIDAR_N) khi import src, nên Pool chỉ
  được dựng khi biến môi trường còn khớp cảm biến mà tiến trình cha đã import.
- make_evaluator(workers, backend) chọn cách đánh giá: "cpu" (mặc định) là ParallelEvaluator
  này, không đổi gì; "batch" (NumPy) và "gpu" (CuPy) chạy nhiều lượt cùng lúc theo lô
  (src/batch), kết quả trùng từng bit với đường từng lượt.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from typing import Any, Iterable, Optional

import numpy as np

from src.env.geodesic import DEFAULT_CACHE_DIR, ensure_cached, get_geodesic
from src.env.kinematics import LIDAR_MODE, LIDAR_NUM_RAYS, lidar_settings_from_env
from src.env.robot_env import RobotNavEnv
from src.neat_engine.evaluator import compute_episode_fitness

# ── Bộ nhớ đệm trong từng tiến trình ─────────────────────────────────────
_MAPS: dict[str, dict] = {}
_GEO: dict[str, Any] = {}
_NEAT_CFG: dict[str, Any] = {}


def load_map(path: str) -> dict:
    m = _MAPS.get(path)
    if m is None:
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        _MAPS[path] = m
    return m


def _geodesic(path: str, cache_dir: Optional[str]):
    g = _GEO.get(path)
    if g is None:
        g = get_geodesic(load_map(path), cache_dir)
        _GEO[path] = g
    return g


def _neat_config(path: str):
    cfg = _NEAT_CFG.get(path)
    if cfg is None:
        import neat
        cfg = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation, path,
        )
        _NEAT_CFG[path] = cfg
    return cfg


# ── Chính sách điều khiển ─────────────────────────────────────────────────
def neat_spec(genome, config_path: str, recurrent: bool) -> dict:
    return {"kind": "neat", "genome": genome, "config": str(config_path), "recurrent": bool(recurrent)}


def mlp_spec(params: np.ndarray, layers: list[int]) -> dict:
    return {"kind": "mlp", "params": np.asarray(params, dtype=np.float64), "layers": list(layers)}


def make_policy(spec: dict):
    kind = spec["kind"]
    if kind == "neat":
        import neat
        cfg = _neat_config(spec["config"])
        cls = neat.nn.RecurrentNetwork if spec["recurrent"] else neat.nn.FeedForwardNetwork
        return cls.create(spec["genome"], cfg)
    if kind == "mlp":
        from src.baselines.fixed_ga import FixedMLP
        net = FixedMLP(spec["layers"])
        net.set_params(spec["params"])
        return net
    if kind == "apf":
        from src.baselines.apf import APFController
        return APFController()
    raise ValueError(f"unknown policy kind: {kind}")


# ── Một lượt chạy ──────────────────────────────────────────────────────────
def run_episode(policy, map_path: str, opts: dict) -> dict:
    """Chạy một lượt, trả các số cần cho fitness, novelty và chỉ số đánh giá."""
    m = load_map(map_path)
    progress = opts.get("progress", "geodesic")
    geo = _geodesic(map_path, opts.get("geo_cache", DEFAULT_CACHE_DIR)) if progress == "geodesic" else None
    env = RobotNavEnv(map_data=m, geodesic=geo)
    obs, _ = env.reset()
    if hasattr(policy, "reset"):
        policy.reset()      # mạng hồi quy: xóa trạng thái mỗi lượt (Decision #13)
    terminated = truncated = False
    info: dict = {}
    while not (terminated or truncated):
        action = np.asarray(policy.activate(obs.tolist()), dtype=np.float32)
        obs, _, terminated, truncated, info = env.step(action)
    traj = np.asarray(env.trajectory)
    length = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))) if len(traj) > 1 else 0.0
    env.close()
    l_opt = float((m.get("metrics") or {}).get("optimal_path_length", info.get("d0", 0.0)))
    return {
        "fitness": compute_episode_fitness(info, progress=progress),
        "success": bool(info["is_success"]),
        "collision": bool(info["is_collision"]),
        "steps": int(info["steps"]),
        "final_pos": (float(info["final_pos"][0]), float(info["final_pos"][1])),
        "length": length,
        "l_opt": l_opt,
    }


_PREFETCHED: set[str] = set()


def prefetch_geodesic(map_paths: Iterable[str], opts: dict) -> None:
    """Tính sẵn (ở tiến trình cha) trường geodesic của các bản đồ sắp dùng.

    Mọi genome trong một thế hệ chạy cùng bộ bản đồ, nên nếu để worker tự tính thì
    mọi worker tính trùng cùng một bản đồ và tranh nhau ghi cùng file đệm.
    """
    if opts.get("progress", "geodesic") != "geodesic":
        return
    cache = opts.get("geo_cache", DEFAULT_CACHE_DIR)
    for p in map_paths:
        if p not in _PREFETCHED:
            ensure_cached(load_map(p), cache)
            _PREFETCHED.add(p)


def eval_task(task: tuple) -> list[dict]:
    """task = (policy_spec, [đường dẫn bản đồ], opts) -> kết quả từng lượt."""
    spec, map_paths, opts = task
    policy = make_policy(spec)
    return [run_episode(policy, p, opts) for p in map_paths]


# ── Pool ───────────────────────────────────────────────────────────────────
class ParallelEvaluator:
    """Bọc multiprocessing.Pool; workers=1 chạy ngay trong tiến trình hiện tại."""

    def __init__(self, workers: int = 1):
        self.workers = max(1, int(workers))
        self.pool = None
        if self.workers > 1:
            # Worker kiểu spawn đọc biến cảm biến lúc import: biến bị đổi sau khi tiến trình này đã
            # import src thì worker chạy cảm biến khác tiến trình cha (kết quả phụ thuộc số worker).
            env_sensor = lidar_settings_from_env()
            if env_sensor != (LIDAR_MODE, LIDAR_NUM_RAYS):
                raise RuntimeError(
                    f"cảm biến đã import là {(LIDAR_MODE, LIDAR_NUM_RAYS)}, nhưng NEAT_LIDAR_MODE / NEAT_LIDAR_N "
                    f"nay cho {env_sensor}: đặt hai biến TRƯỚC khi import src"
                )
            # Mỗi worker một luồng BLAS, tránh 4 worker x 16 luồng tranh nhau CPU
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(var, "1")
            self.pool = mp.get_context("spawn").Pool(self.workers)

    def map(self, tasks: Iterable[tuple]) -> list[list[dict]]:
        tasks = list(tasks)
        if self.pool is not None:
            seen: set[str] = set()
            for _, map_paths, opts in tasks:
                new = [p for p in map_paths if p not in seen]
                seen.update(new)
                prefetch_geodesic(new, opts)
        if self.pool is None:
            return [eval_task(t) for t in tasks]
        chunk = max(1, len(tasks) // (self.workers * 4))
        return self.pool.map(eval_task, tasks, chunksize=chunk)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None
        return False


# ── Chọn cách đánh giá (tùy chọn chạy theo lô; mặc định không đổi) ─────────
EVAL_BACKENDS = ("cpu", "batch", "gpu")
_BACKEND_ALIASES = {"process": "cpu", "pool": "cpu", "numpy": "batch", "cuda": "gpu"}


def resolve_backend(backend: Optional[str] = None) -> str:
    """Tên backend: tham số tường minh, nếu không có thì biến môi trường NEAT_EVAL_BACKEND, mặc định "cpu".

    "cpu"   ParallelEvaluator: từng lượt, đường mặc định, không đổi gì;
    "batch" chạy theo lô bằng NumPy (src/batch), workers > 1 thì chia lô cho Pool;
    "gpu"   chạy theo lô, LiDAR + va chạm trên GPU (CuPy), phần còn lại ở host.
    "batch" và "gpu" (chế độ exact, mặc định) cho kết quả trùng từng bit với "cpu".
    """
    raw = backend if backend is not None else os.environ.get("NEAT_EVAL_BACKEND", "")
    name = str(raw).strip().lower() or "cpu"
    name = _BACKEND_ALIASES.get(name, name)
    if name not in EVAL_BACKENDS:
        raise ValueError(f"backend={raw!r}: chỉ nhận {EVAL_BACKENDS} (tham số hoặc NEAT_EVAL_BACKEND)")
    return name


def check_backend_options(name: str, options: Optional[dict]) -> dict:
    """Tùy chọn của BatchEvaluator; exact=False (chế độ device, không trùng bit) chỉ nhận với "gpu"."""
    options = dict(options or {})
    if not options.get("exact", True) and name != "gpu":
        raise ValueError("chế độ không exact (--inexact) chỉ dùng với backend gpu")
    return options


def make_evaluator(workers: int = 1, backend: Optional[str] = None, **options):
    """Dựng bộ đánh giá có giao diện của ParallelEvaluator (map, close, with).

    backend xem resolve_backend; "cpu" trả đúng ParallelEvaluator(workers) và không nạp gì mới.
    options chuyển cho src.batch.BatchEvaluator (exact, max_episodes, lidar_chunk, gpu_variant); riêng
    strict (mặc định theo biến môi trường NEAT_EVAL_STRICT=1) quyết định khi tự kiểm không đạt.
    Tự kiểm của chạy theo lô không đạt -> in cảnh báo ra stdout (và warnings), dùng ParallelEvaluator cho lần
    chạy này (nhãn "<backend>->cpu"); với strict thì báo lỗi NumericsMismatch (job benchmark nên dừng hơn là
    chạy chậm đi hàng chục lần mà không ai thấy).
    GPU không dùng được (thiếu CuPy, driver, NVRTC, lỗi CUDA) -> báo lỗi GpuUnavailable.
    """
    name = resolve_backend(backend)
    strict = options.pop("strict", None)
    options = check_backend_options(name, options)
    if name == "cpu":
        return ParallelEvaluator(workers)
    if strict is None:
        strict = os.environ.get("NEAT_EVAL_STRICT", "").strip().lower() in ("1", "true", "yes", "on")
    from src.batch.evaluator import BatchEvaluator, NumericsMismatch
    try:
        return BatchEvaluator(workers=workers, device="cpu" if name == "batch" else "gpu", **options)
    except NumericsMismatch as e:
        if strict:
            raise
        import warnings
        msg = (f"backend {name}: tự kiểm không đạt ({e}); lần chạy này dùng ParallelEvaluator (từng lượt, "
               f"{max(1, int(workers))} worker), chậm hơn nhiều")
        print(f"CẢNH BÁO: {msg}", flush=True)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        ev = ParallelEvaluator(workers)
        ev.backend = f"{name}->cpu"
        return ev


def backend_label(evaluator) -> str:
    """Nhãn backend thực dùng ("cpu" với ParallelEvaluator); summary.json chỉ ghi khi khác "cpu"."""
    return getattr(evaluator, "backend", "cpu")


# ── Tổng hợp chỉ số ────────────────────────────────────────────────────────
def summarize(results: list[dict], dt: float = 0.1) -> dict:
    """SR, CR, TC, tỉ lệ hết giờ, SPL, MTT (giây, chỉ lượt thành công), fitness trung bình.

    - SR (Success Rate): tỉ lệ lượt tới đích mà không va chạm.
    - TC (Total Collisions): tổng số lần va chạm trên cả tập. Lượt chạy dừng ngay ở lần va chạm đầu
      tiên, nên mỗi lượt có nhiều nhất một va chạm và TC = số lượt kết thúc vì va chạm (= CR × n).
    - MTT (Mean Time to Traverse): thời gian trung bình (giây) đi từ S tới G, tính trên các lượt tới đích.
    """
    n = len(results)
    if n == 0:
        return {"n": 0}
    succ = np.array([r["success"] for r in results])
    col = np.array([r["collision"] for r in results])
    steps = np.array([r["steps"] for r in results], dtype=float)
    spl = [r["l_opt"] / max(r["length"], r["l_opt"]) if r["success"] and r["l_opt"] > 0 else 0.0 for r in results]
    return {
        "n": n,
        "SR": round(float(succ.mean()), 4),
        "CR": round(float(col.mean()), 4),
        "TC": int(col.sum()),
        "timeout": round(float((~succ & ~col).mean()), 4),
        "SPL": round(float(np.mean(spl)), 4),
        "MTT": round(float(steps[succ].mean() * dt), 2) if succ.any() else None,
        "mean_fitness": round(float(np.mean([r["fitness"] for r in results])), 4),
    }
