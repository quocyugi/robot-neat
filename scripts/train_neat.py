"""
Train NEAT — Main Entry Point
===============================
Launches the full NEAT evolution pipeline with configurable options.

Usage
-----
    python scripts/train_neat.py --config configs/neat_recurrent.ini --novelty --curriculum --seed 0 --workers 4
    python scripts/train_neat.py --config configs/neat_feedforward.ini --no-novelty --no-curriculum

Kết quả trong --output-dir: champion_genome.pkl (genome tốt nhất trên val),
summary.json, training_history.json, neat_config.ini. Log huấn luyện dạng hình ghi thêm
vào cùng thư mục: training_history_bang.png (ảnh bảng), training_history_bieu_do.png
(ảnh biểu đồ fitness, tỉ lệ tới đích, SR val, số loài theo thế hệ) và training_history.html
(trang mở bằng trình duyệt); --no-visual để bỏ. Chấm trên tập test bằng
    python scripts/evaluate.py --run-dir output/train
"""

import argparse
import sys
from pathlib import Path

# Cho phép chạy "python scripts/train_neat.py" trực tiếp từ thư mục gốc
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.neat_engine.runner import NEATRunner  # noqa: E402


def main():
    from src.console import utf8_stdout
    utf8_stdout()
    parser = argparse.ArgumentParser(description="Train NEAT Robot Navigation")
    parser.add_argument(
        "--config", type=str, default="configs/neat_recurrent.ini",
        help="Path to NEAT config file",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/maps",
        help="Path to pre-generated map dataset",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/train",
        help="Output directory for results",
    )
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--maps-per-eval", type=int, default=8,
                        help="số bản đồ mỗi thế hệ, dùng chung cho cả quần thể")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4, help="số tiến trình đánh giá song song")
    parser.add_argument(
        "--novelty", dest="novelty", action="store_true", default=True,
    )
    parser.add_argument("--no-novelty", dest="novelty", action="store_false")
    parser.add_argument(
        "--curriculum", dest="curriculum", action="store_true", default=True,
    )
    parser.add_argument("--no-curriculum", dest="curriculum", action="store_false",
                        help="học đều trên cả 4 cấp (lịch học chung để so với GA)")
    parser.add_argument("--recurrent", dest="recurrent", action="store_true", default=None,
                        help="mặc định suy ra từ feed_forward của config")
    parser.add_argument("--no-recurrent", dest="recurrent", action="store_false")
    parser.add_argument("--fixed-lambda", type=float, default=None,
                        help="giữ λ cố định (0 = novelty thuần, như M4)")
    parser.add_argument("--novelty-space", choices=["probe", "population"], default="probe")
    parser.add_argument("--progress", choices=["geodesic", "euclid"], default="geodesic",
                        help="đo tiến độ theo đường đi thật (mặc định) hay đường chim bay")
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--max-gens-per-level", type=int, default=None,
                        help="curriculum: số thế hệ tối đa mỗi cấp (mặc định generations/5)")
    parser.add_argument("--no-visual", action="store_true",
                        help="không ghi ảnh bảng, ảnh biểu đồ và trang HTML của log huấn luyện")

    args = parser.parse_args()

    runner = NEATRunner(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_novelty=args.novelty,
        use_curriculum=args.curriculum,
        is_recurrent=args.recurrent,
        maps_per_eval=args.maps_per_eval,
        max_generations=args.generations,
        seed=args.seed,
        workers=args.workers,
        fixed_lambda=args.fixed_lambda,
        novelty_space=args.novelty_space,
        progress=args.progress,
        val_every=args.val_every,
        max_gens_per_level=args.max_gens_per_level if args.max_gens_per_level else "auto",
    )

    print("=" * 60)
    print("  NEAT Robot Navigation Training")
    print(f"  Config:     {args.config}")
    print(f"  Novelty:    {args.novelty}")
    print(f"  Curriculum: {args.curriculum}")
    print(f"  Recurrent:  {runner.is_recurrent}")
    print(f"  Max Gen:    {args.generations}")
    print(f"  Seed:       {args.seed}   Workers: {args.workers}")
    print("=" * 60, flush=True)

    runner.run()
    print(f"\nTraining complete. Champion saved to {args.output_dir}/champion_genome.pkl")
    if not args.no_visual:
        from src.visual_report import print_written, training_report, warn_failed
        try:
            print_written(training_report(args.output_dir))
        except Exception as e:  # noqa: BLE001  (ảnh là phần thêm: champion và log JSON đã ghi xong)
            warn_failed("ảnh và trang HTML của log huấn luyện", e)


if __name__ == "__main__":
    main()
