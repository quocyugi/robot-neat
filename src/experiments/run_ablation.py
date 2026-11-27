"""
Ablation Study Runner — M1 to M6 Experiment Configurations
=============================================================
Automates running all 6 ablation variants and collecting metrics.

Reference
---------
- brainstorm/04_experiments/ablation_design.md (Decision #26)

Các lỗi đã sửa (báo cáo mục 9)
-------------------------------
L03  M4 ("novelty thuần") trước đây có tham số y hệt M6, hai lịch sử 100 thế hệ
     cùng seed giống hệt nhau. Nay M4 đặt fixed_lambda = 0.0 (λ nhân với fitness,
     λ = 0 là chỉ còn novelty).
L14  Biến seed trước chỉ dùng đặt tên thư mục. Nay seed được truyền vào NEATRunner
     (random, numpy, neat.Population, bốc bản đồ), nên 10 lần chạy tái lập được.
L22  Không còn nuốt lỗi rồi chạy tiếp lặng lẽ: lỗi được ghi kèm traceback, cuối
     chương trình in danh sách lượt chạy hỏng và trả mã lỗi 1.
L04  Chạy được bằng "python -m src.experiments.run_ablation" hoặc
     "python src/experiments/run_ablation.py"; có tham số dòng lệnh.

Sau khi chạy: scripts/evaluate.py chấm champion trên tập test,
scripts/summarize_results.py gom các seed và làm kiểm định.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

# Cho phép chạy trực tiếp file này từ thư mục gốc dự án
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.neat_engine.parallel import EVAL_BACKENDS, resolve_backend  # noqa: E402
from src.neat_engine.runner import NEATRunner  # noqa: E402

# ── Ablation matrix definition ───────────────────────────────────────────
ABLATION_CONFIGS: dict[str, dict] = {
    "M1": {
        "description": "FF + Distance Fitness + No Curriculum (baseline floor)",
        "config_file": "configs/neat_feedforward.ini",
        "is_recurrent": False,
        "use_novelty": False,
        "use_curriculum": False,
    },
    "M2": {
        "description": "Recurrent + Distance Fitness + Curriculum",
        "config_file": "configs/neat_recurrent.ini",
        "is_recurrent": True,
        "use_novelty": False,
        "use_curriculum": True,
    },
    "M3": {
        "description": "FF + Hybrid Novelty + Curriculum",
        "config_file": "configs/neat_feedforward.ini",
        "is_recurrent": False,
        "use_novelty": True,
        "use_curriculum": True,
    },
    "M4": {
        "description": "Recurrent + Pure Novelty + Curriculum",
        "config_file": "configs/neat_recurrent.ini",
        "is_recurrent": True,
        "use_novelty": True,
        "use_curriculum": True,
        "fixed_lambda": 0.0,   # L03: novelty thuần, λ cố định = 0
    },
    "M5": {
        "description": "Recurrent + Hybrid Novelty + No Curriculum",
        "config_file": "configs/neat_recurrent.ini",
        "is_recurrent": True,
        "use_novelty": True,
        "use_curriculum": False,
    },
    "M6": {
        "description": "FULL MODEL: Recurrent + Hybrid Novelty + Curriculum",
        "config_file": "configs/neat_recurrent.ini",
        "is_recurrent": True,
        "use_novelty": True,
        "use_curriculum": True,
    },
}


# Biến thể NEAT ngoài ma trận M1–M6, chạy khi gọi đích danh (models=["LNS"] hoặc ["LNS_ref"]).
#   LNS:     novelty search kiểu Lehman với cấu hình NEAT của nhóm (configs/neat_recurrent.ini, như R0, M4, M6)
#   LNS_ref: cùng thuật toán với cấu hình NEAT tham chiếu của thí nghiệm mê cung (configs/neat_lehman_ref.ini)
EXTRA_CONFIGS = {
    "LNS": {
        "description": "Recurrent + Lehman-Stanley novelty search (novelty thuần, luật archive của bài báo) + No Curriculum",
        "config_file": "configs/neat_recurrent.ini",
        "is_recurrent": True,
        "use_novelty": True,
        "use_curriculum": False,
        "novelty_rule": "lehman",
    },
    "LNS_ref": {
        "description": "Recurrent + Lehman-Stanley novelty search + cấu hình NEAT tham chiếu của thí nghiệm mê cung + No Curriculum",
        "config_file": "configs/neat_lehman_ref.ini",
        "is_recurrent": True,
        "use_novelty": True,
        "use_curriculum": False,
        "novelty_rule": "lehman",
    },
}
ALL_CONFIGS = {**ABLATION_CONFIGS, **EXTRA_CONFIGS}


def _inexact_done(summary_path: Path) -> bool:
    """summary.json ghi "backend": "gpu-device" (chế độ không exact)? File hỏng thì coi như không (giữ cách cũ)."""
    try:
        with open(summary_path, encoding="utf-8") as f:
            return json.load(f).get("backend") == "gpu-device"
    except (OSError, ValueError, AttributeError):
        return False


def run_ablation(
    data_dir: str = "data/maps",
    output_root: str = "output/ablation",
    max_generations: int = 300,
    n_seeds: int = 10,
    models: list[str] | None = None,
    seeds: list[int] | None = None,
    workers: int = 4,
    maps_per_eval: int = 8,
    skip_done: bool = True,
    backend: str | None = None,
    backend_options: dict | None = None,
) -> list[dict]:
    """Run all ablation experiments across multiple seeds. Trả danh sách lượt chạy hỏng.

    backend / backend_options: cách đánh giá của NEATRunner (mặc định "cpu", xem parallel.make_evaluator).
    skip_done bỏ qua lượt đã có summary.json, trừ lượt cũ chạy ở chế độ không exact ("backend": "gpu-device")
    khi lần này chạy exact: lượt đó được chạy lại (số không trùng bit không được lẫn vào kết quả).
    """
    output_root = Path(output_root)
    models = models or list(ABLATION_CONFIGS)
    seeds = seeds if seeds is not None else list(range(n_seeds))
    failures: list[dict] = []

    for model_id in models:
        cfg = ALL_CONFIGS[model_id]
        for seed in seeds:
            run_dir = output_root / model_id / f"seed_{seed}"
            if skip_done and (run_dir / "summary.json").exists():
                if _inexact_done(run_dir / "summary.json") and (backend_options or {}).get("exact", True):
                    print(f"  {model_id} seed {seed}: summary.json cũ chạy ở chế độ không exact (gpu-device), "
                          f"chạy lại")
                else:
                    print(f"  {model_id} seed {seed}: đã có summary.json, bỏ qua")
                    continue
            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"  {model_id} - Seed {seed} - {cfg['description']}")
            print(f"{'='*60}\n", flush=True)

            try:
                runner = NEATRunner(
                    config_path=str(_ROOT / cfg["config_file"]),
                    data_dir=data_dir,
                    output_dir=str(run_dir),
                    use_novelty=cfg["use_novelty"],
                    use_curriculum=cfg["use_curriculum"],
                    is_recurrent=cfg["is_recurrent"],
                    max_generations=max_generations,
                    maps_per_eval=maps_per_eval,
                    seed=seed,
                    workers=workers,
                    fixed_lambda=cfg.get("fixed_lambda"),
                    novelty_rule=cfg.get("novelty_rule", "project"),
                    backend=backend,
                    backend_options=backend_options,
                )
                runner.run()
                (run_dir / "error.txt").unlink(missing_ok=True)
            except Exception as e:  # ghi đủ traceback, báo lại ở cuối (L22)
                print(f"  FAILED: {e!r}", flush=True)
                (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                failures.append({"model": model_id, "seed": seed, "error": repr(e)})

    if failures:
        print("\nCác lượt chạy hỏng:")
        for f in failures:
            print(f"  {f['model']} seed {f['seed']}: {f['error']}")
    return failures


def main():
    from src.console import utf8_stdout
    utf8_stdout()
    ap = argparse.ArgumentParser(description="Chạy ma trận ablation M1–M6 (LNS: novelty search kiểu Lehman, khi gọi đích danh)")
    ap.add_argument("--data-dir", default="data/maps")
    ap.add_argument("--output-root", default="output/ablation")
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=10, help="số seed (0..n-1)")
    ap.add_argument("--seed-list", type=str, default=None, help="danh sách seed, ví dụ 0,1,2")
    ap.add_argument("--models", type=str, default=",".join(ABLATION_CONFIGS), help="ví dụ M2,M6; thêm LNS để chạy novelty search kiểu Lehman")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--maps-per-eval", type=int, default=8)
    ap.add_argument("--rerun", action="store_true", help="chạy lại cả lượt đã có summary.json")
    ap.add_argument("--backend", choices=list(EVAL_BACKENDS), default=None,
                    help="cách đánh giá: cpu (mặc định, hoặc theo NEAT_EVAL_BACKEND), batch (NumPy theo lô), "
                         "gpu (theo lô, LiDAR + va chạm trên GPU); batch và gpu cho cùng kết quả từng bit")
    ap.add_argument("--inexact", action="store_true",
                    help="chỉ với --backend gpu: hàm siêu việt trên GPU, KHÔNG trùng bit; không dùng cho benchmark")
    a = ap.parse_args()
    if a.inexact and resolve_backend(a.backend) != "gpu":
        ap.error("--inexact chỉ dùng với --backend gpu")
    seeds = [int(s) for s in a.seed_list.split(",")] if a.seed_list else list(range(a.seeds))
    failures = run_ablation(a.data_dir, a.output_root, a.generations, models=a.models.split(","),
                            seeds=seeds, workers=a.workers, maps_per_eval=a.maps_per_eval, skip_done=not a.rerun,
                            backend=a.backend, backend_options={"exact": False} if a.inexact else None)
    with open(Path(a.output_root) / "failures.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
