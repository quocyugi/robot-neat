"""
BatchEvaluator: cùng giao diện với ParallelEvaluator, chạy các lượt theo lô
============================================================================
map(tasks) nhận đúng các tác vụ (spec, [đường dẫn bản đồ], opts) của parallel.eval_task và trả
list (theo tác vụ) của list (theo bản đồ) các dict kết quả, cùng khóa, cùng giá trị, cùng kiểu Python
với đường từng lượt; close() gọi lại vô hại; dùng được với "with" (__exit__ dọn pool và bộ nhớ GPU).

Cách chạy (device):
- "cpu": NumPy trong tiến trình hiện tại; workers > 1 thì chia danh sách tác vụ thành `workers` khúc
  liền (cân theo số lượt) cho một Pool kiểu spawn, ghép lại đúng thứ tự. Mỗi worker tự kiểm số học
  lúc khởi động; worker nào không đạt thì chấm khúc của nó bằng eval_task (vẫn đúng kết quả).
- "gpu" (exact, mặc định khi chọn GPU): biến thể lai, LiDAR + va chạm trên GPU (src/batch/gpu.py), mọi
  thứ khác ở host; chạy trong tiến trình chính (không nhân context CUDA); workers bị bỏ qua.
- "gpu" với exact=False (chế độ device, phải bật tường minh): mọi mảng trên CuPy, hàm siêu việt tính
  trên GPU; KHÔNG trùng bit (lịch sử tiến hóa rẽ nhánh), chỉ để thử.
- gpu_variant="full" với exact=True: mọi mảng trên CuPy, hàm siêu việt ở host (bản tham chiếu để gỡ
  lỗi; chậm hơn biến thể lai).

Tác vụ không chạy được theo lô (hàm kích hoạt khác tanh, trọng số không phải float, số đầu vào khác N + 3,
bản đồ không có đoạn nào, loại bộ điều khiển lạ, ...) được chạy bằng parallel.eval_task: cùng kết quả hoặc cùng
lỗi như đường cũ, nhưng chậm hơn; lần đầu có tác vụ như vậy, map() in một dòng lưu ý (kèm lý do đầu tiên).
Tác vụ không bị cắt ngang: mọi bản đồ của một tác vụ chạy trong cùng một lô.
Lúc dựng: tự kiểm số học và so trọn lượt (src/batch/numerics.py). Với GPU (exact) thêm gpu.startup_check, rồi
sau khi đã nạp CuPy / DLL CUDA thì kiểm lại số học của host (nạp DLL có thể đổi số học của cả tiến trình, ví
dụ cờ FTZ / DAZ) và so trọn lượt với chính hình học GPU. Lệch thì báo NumericsMismatch; parallel.make_evaluator
khi đó cảnh báo và dùng ParallelEvaluator cho lần chạy đó (hoặc báo lỗi nếu đặt NEAT_EVAL_STRICT=1).
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Iterable, Optional

import numpy as np

from src.batch import numerics
from src.batch.backend import Backend, GpuUnavailable
from src.batch.policies import APFPolicy, MLPPolicy, NeatPolicy, Unsupported, neat_slots
from src.batch.sim import Episode, make_host_geometry, robot_geometry, segments_ok, simulate
from src.env import kinematics as KM
from src.env.geodesic import DEFAULT_CACHE_DIR
from src.env.lidar import FastLiDAR2D
from src.neat_engine import parallel as PAR
from src.neat_engine.evaluator import PROGRESS_MODES

DEVICES = ("cpu", "gpu")
GPU_VARIANTS = ("hybrid", "full")
DEFAULT_MAX_EPISODES = {"cpu": 4096, "gpu": 16384}

__all__ = ["BatchEvaluator", "NumericsMismatch", "GpuUnavailable", "DEVICES", "GPU_VARIANTS"]


class NumericsMismatch(RuntimeError):
    """Tự kiểm số học không đạt: chạy theo lô không còn bảo đảm trùng từng bit trên máy này."""


def backend_name(device: str, exact: bool, gpu_variant: str) -> str:
    """Nhãn ghi vào summary.json: "batch", "gpu", "gpu-full" (gỡ lỗi) hoặc "gpu-device" (không exact)."""
    if device == "cpu":
        return "batch"
    if not exact:
        return "gpu-device"
    return "gpu" if gpu_variant == "hybrid" else "gpu-full"


def _recheck_after_gpu(be, geom_factory) -> list[str]:
    """Sau khi nạp CuPy: kiểm lại số học của host và so trọn lượt với hình học của thiết bị. GpuUnavailable (lỗi
    CUDA) được báo nguyên; lỗi khác của phép kiểm tính là lệch."""
    problems = numerics.self_check(force=True)
    try:
        problems += numerics.episode_check(be, geom_factory)
    except GpuUnavailable:
        raise
    except Exception as e:  # noqa: BLE001
        problems.append(f"episode_check trên GPU: {e!r}")
    return problems


class _Engine:
    """Phần chạy theo lô trong một tiến trình: tách nhóm theo loại bộ điều khiển, chia khúc, mô phỏng,
    quay về eval_task cho tác vụ không hỗ trợ."""

    def __init__(self, device: str = "cpu", exact: bool = True, gpu_variant: str = "hybrid",
                 max_episodes: Optional[int] = None, lidar_chunk: Optional[int] = None):
        self.lidar = FastLiDAR2D(KM.LIDAR_NUM_RAYS, KM.LIDAR_MAX_RANGE, KM.LIDAR_MODE, KM.SECTOR_LAYOUT)
        self.max_episodes = int(max_episodes or DEFAULT_MAX_EPISODES[device])
        self.lidar_chunk = lidar_chunk
        self.stats = {"tasks": 0, "episodes": 0, "fallback_tasks": 0}
        self.fallback_reason: Optional[str] = None          # lý do của tác vụ đầu tiên phải chạy bằng eval_task
        if device == "cpu":
            self.be = Backend("numpy", True)
            self._geom = self._host_geometry
        elif exact and gpu_variant == "hybrid":
            from src.batch import gpu
            self.be = Backend("numpy", True)
            gpu.GpuGeometry(self.lidar, KM.COLL_RADIUS)      # biên dịch kernel ngay, lỗi báo sớm
            # Waffle: kernel không đổi; tâm thân (gốc LiDAR, tâm đĩa va chạm) tính ở host (BodyCenterGeometry)
            gpu_geom = lambda: robot_geometry(self.be, gpu.GpuGeometry(self.lidar, KM.COLL_RADIUS))  # noqa: E731
            problems = gpu.startup_check(self.lidar)
            problems += _recheck_after_gpu(self.be, gpu_geom)
            if problems:
                raise NumericsMismatch("; ".join(problems))
            self._geom = gpu_geom
        else:
            self.be = Backend("cupy", exact)
            self._geom = self._host_geometry
            if exact:                                        # biến thể "full" exact (bản tham chiếu để gỡ lỗi)
                problems = _recheck_after_gpu(self.be, self._host_geometry)
                if problems:
                    raise NumericsMismatch("; ".join(problems))

    def _host_geometry(self):
        return make_host_geometry(self.be, self.lidar, self.lidar_chunk)

    # ── tách nhóm ──
    def _classify(self, spec, paths, opts, pol_cache: dict, map_ok: dict):
        """(khóa nhóm, đối tượng bộ điều khiển) hoặc Unsupported."""
        if opts.get("progress", "geodesic") not in PROGRESS_MODES or not paths:
            raise Unsupported("progress lạ hoặc không có bản đồ")
        for p in paths:
            ok = map_ok.get(p)
            if ok is None:
                ok = map_ok[p] = segments_ok(np.array(PAR.load_map(p)["segments"], dtype=np.float64))
            if not ok:
                raise Unsupported("bản đồ không có đoạn nào")
        kind = spec.get("kind")
        if kind not in ("neat", "mlp", "apf"):
            raise Unsupported(f"loại bộ điều khiển {kind!r}")
        pol = pol_cache.get(id(spec))
        if pol is None:
            pol = PAR.make_policy(spec)
            pol_cache[id(spec)] = pol
        if kind == "neat":
            recurrent = bool(spec["recurrent"])
            if len(pol.input_nodes) != KM.OBS_DIM or len(pol.output_nodes) != 2:
                raise Unsupported("số đầu vào / đầu ra của mạng khác N + 3 / 2")
            neat_slots(pol, recurrent)
            return ("neat", recurrent), pol
        if kind == "mlp":
            layers = tuple(int(v) for v in spec["layers"])
            if layers[0] != KM.OBS_DIM or layers[-1] != 2:
                raise Unsupported("lớp vào / ra của MLP khác N + 3 / 2")
            return ("mlp", layers, float(pol.gain)), pol
        return ("apf", pol.k_att, pol.k_rep, pol.d_safe), pol

    def evaluate(self, tasks: list) -> list:
        out: list = [None] * len(tasks)
        groups: dict = {}
        pol_cache: dict = {}
        map_ok: dict = {}
        for ti, (spec, paths, opts) in enumerate(tasks):
            try:
                key, pol = self._classify(spec, paths, opts, pol_cache, map_ok)
            except Exception as e:  # noqa: BLE001  (đường cũ cho cùng kết quả, hoặc báo đúng lỗi của nó)
                if self.fallback_reason is None:
                    self.fallback_reason = f"{type(e).__name__}: {e}"
                out[ti] = PAR.eval_task((spec, paths, opts))
                self.stats["fallback_tasks"] += 1
                continue
            groups.setdefault(key, []).append((ti, pol))
        for key, items in groups.items():
            chunk, n_ep = [], 0
            for it in items:
                n = len(tasks[it[0]][1])
                if chunk and n_ep + n > self.max_episodes:
                    self._run(key, chunk, tasks, out)
                    chunk, n_ep = [], 0
                chunk.append(it)
                n_ep += n
            if chunk:
                self._run(key, chunk, tasks, out)
        self.stats["tasks"] += len(tasks)
        return out

    def _run(self, key, items, tasks, out):
        pols = [pol for _, pol in items]
        episodes = []
        for local, (ti, _) in enumerate(items):
            _, paths, opts = tasks[ti]
            progress = opts.get("progress", "geodesic")
            cache = opts.get("geo_cache", DEFAULT_CACHE_DIR)
            episodes.extend(Episode(p, progress, cache, local) for p in paths)
        be = self.be
        if key[0] == "neat":
            factory = lambda ep_task: NeatPolicy(be, pols, key[1], ep_task)  # noqa: E731
        elif key[0] == "mlp":
            factory = lambda ep_task: MLPPolicy(be, pols, ep_task)  # noqa: E731
        else:
            factory = lambda ep_task: APFPolicy(be, KM.LIDAR_NUM_RAYS, pols[0])  # noqa: E731
        res = simulate(be, episodes, factory, self._geom())
        self.stats["episodes"] += len(episodes)
        i = 0
        for ti, _ in items:
            n = len(tasks[ti][1])
            out[ti] = res[i:i + n]
            i += n


# ── Pool (chỉ CPU) ─────────────────────────────────────────────────────────
_WORKER: dict = {}


def _worker_init(engine_kwargs: dict) -> None:
    problems = numerics.self_check()
    _WORKER["problems"] = problems
    _WORKER["engine"] = None if problems else _Engine(**engine_kwargs)
    if problems:
        import warnings
        warnings.warn("worker chạy theo lô: tự kiểm số học không đạt (" + "; ".join(problems)
                      + "), khúc của worker này chạy bằng eval_task", RuntimeWarning)


def _worker_evaluate(tasks: list):
    eng = _WORKER.get("engine")
    if eng is None:
        return [PAR.eval_task(t) for t in tasks], {"tasks": len(tasks), "episodes": 0, "fallback_tasks": len(tasks)}
    before = dict(eng.stats)
    res = eng.evaluate(tasks)
    return res, {k: eng.stats[k] - before[k] for k in before}


def split_balanced(tasks: list, n_chunks: int) -> list:
    """Chia danh sách tác vụ thành tối đa n_chunks khúc liền, cân theo số lượt (số bản đồ)."""
    n_chunks = max(1, min(int(n_chunks), len(tasks)))
    sizes = [max(1, len(t[1])) for t in tasks]
    total = sum(sizes)
    chunks, start, acc = [], 0, 0
    for i, s in enumerate(sizes):
        acc += s
        if len(chunks) < n_chunks - 1 and acc * n_chunks >= total * (len(chunks) + 1):
            chunks.append(tasks[start:i + 1])
            start = i + 1
    chunks.append(tasks[start:])
    return [c for c in chunks if c]


# ── BatchEvaluator ─────────────────────────────────────────────────────────
class BatchEvaluator:
    """Bản chạy theo lô của ParallelEvaluator (cùng map / close / with)."""

    def __init__(self, workers: int = 1, device: str = "cpu", exact: bool = True, gpu_variant: str = "hybrid",
                 max_episodes: Optional[int] = None, lidar_chunk: Optional[int] = None, self_check: bool = True):
        device = str(device).strip().lower()
        if device not in DEVICES:
            raise ValueError(f"device={device!r}: chỉ nhận {DEVICES}")
        if gpu_variant not in GPU_VARIANTS:
            raise ValueError(f"gpu_variant={gpu_variant!r}: chỉ nhận {GPU_VARIANTS}")
        exact = bool(exact)
        if device == "cpu" and not exact:
            raise ValueError("chạy theo lô trên CPU luôn exact; chế độ không exact (device) chỉ có với GPU")
        if not exact:
            gpu_variant = "full"                       # chế độ device: mọi thứ trên CuPy
        self.workers = max(1, int(workers))
        self.device, self.exact, self.gpu_variant = device, exact, gpu_variant
        self.backend = backend_name(device, exact, gpu_variant)
        self.stats = {"tasks": 0, "episodes": 0, "fallback_tasks": 0}
        self._fallback_noted = False
        self.pool = None
        if self_check:
            problems = numerics.self_check()
            if problems:
                raise NumericsMismatch("; ".join(problems))
        engine_kwargs = dict(device=device, exact=exact, gpu_variant=gpu_variant,
                             max_episodes=max_episodes, lidar_chunk=lidar_chunk)
        self._engine = _Engine(**engine_kwargs)
        if device == "cpu" and self.workers > 1:
            # như ParallelEvaluator: worker spawn đọc lại robot và cảm biến lúc import, mỗi worker một luồng BLAS
            env_sensor = KM.robot_settings_from_env()
            have = (KM.ROBOT, KM.LIDAR_MODE, KM.LIDAR_NUM_RAYS, KM.SECTOR_LAYOUT)
            if env_sensor != have:
                raise RuntimeError(
                    f"robot / cảm biến đã import là {have}, nhưng NEAT_ROBOT / NEAT_LIDAR_MODE / NEAT_LIDAR_N / "
                    f"NEAT_SECTOR_LAYOUT nay cho {env_sensor}: đặt các biến TRƯỚC khi import src")
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(var, "1")
            self.pool = mp.get_context("spawn").Pool(self.workers, initializer=_worker_init,
                                                     initargs=(engine_kwargs,))

    def map(self, tasks: Iterable[tuple]) -> list[list[dict]]:
        tasks = [(spec, list(paths), opts) for spec, paths, opts in tasks]
        if not tasks:
            return []
        if self.pool is None:
            before = dict(self._engine.stats)
            res = self._engine.evaluate(tasks)
            for k in self.stats:
                self.stats[k] += self._engine.stats[k] - before[k]
            self._note_fallback()
            return res
        seen: set[str] = set()
        for _, map_paths, opts in tasks:                # trường geodesic tính sẵn ở tiến trình cha
            new = [p for p in map_paths if p not in seen]
            seen.update(new)
            PAR.prefetch_geodesic(new, opts)
        parts = self.pool.map(_worker_evaluate, split_balanced(tasks, self.workers), chunksize=1)
        out = []
        for res, st in parts:
            out.extend(res)
            for k in self.stats:
                self.stats[k] += st.get(k, 0)
        self._note_fallback()
        return out

    def _note_fallback(self):
        """Lần đầu có tác vụ phải chạy bằng eval_task: in một dòng (đúng số nhưng chậm, nên phải thấy được)."""
        if self.stats["fallback_tasks"] and not self._fallback_noted:
            self._fallback_noted = True
            reason = self._engine.fallback_reason
            print(f"Lưu ý (backend {self.backend}): {self.stats['fallback_tasks']} tác vụ không chạy được theo lô, đã chạy "
                  f"bằng đường từng lượt (cùng kết quả, chậm hơn)" + (f"; lý do đầu tiên: {reason}" if reason else ""),
                  flush=True)

    def _free_gpu(self):
        if self.device == "gpu":
            try:
                import cupy
                cupy.get_default_memory_pool().free_all_blocks()
                cupy.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:  # noqa: BLE001
                pass

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
        self._free_gpu()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None
        self._free_gpu()
        return False
