"""
Evaluate — chấm một bộ điều khiển trên các tập test (sửa lỗi L04)
====================================================================
Trước đây repo không có script kiểm thử: compute_all_metrics, SimpleGA,
random_search, APFController, RuleBasedController không được gọi ở đâu.

Usage
-----
    # champion NEAT của một lượt huấn luyện (đọc champion_genome.pkl, neat_config.ini, summary.json)
    python scripts/evaluate.py --run-dir output/train

    # genome bất kỳ
    python scripts/evaluate.py --genome path/to/genome.pkl --config configs/neat_recurrent.ini

    # champion GA của scripts/run_baselines.py
    python scripts/evaluate.py --mlp-params output/baselines/ga_b1/seed_0/champion_params.npz

    # bộ điều khiển không học
    python scripts/evaluate.py --controller apf
    python scripts/evaluate.py --controller rule

Kết quả in ra màn hình và ghi JSON (mặc định <run-dir>/test_results.json).
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import neat  # noqa: E402
import numpy as np  # noqa: E402

from src.experiments.evaluation import DEFAULT_SETS, evaluate_sets, find_sets, format_report  # noqa: E402
from src.neat_engine.parallel import mlp_spec, neat_spec  # noqa: E402


def build_spec(args) -> tuple[dict, str]:
    if args.run_dir:
        run = Path(args.run_dir)
        cfg = run / "neat_config.ini"
        if not cfg.exists():
            cfg = Path(json.load(open(run / "summary.json", encoding="utf-8"))["config"])
        with open(run / "champion_genome.pkl", "rb") as f:
            genome = pickle.load(f)
        return _neat(genome, cfg, args.recurrent), f"NEAT champion {run}"
    if args.genome:
        with open(args.genome, "rb") as f:
            genome = pickle.load(f)
        return _neat(genome, Path(args.config), args.recurrent), f"NEAT genome {args.genome}"
    if args.mlp_params:
        z = np.load(args.mlp_params)
        return mlp_spec(z["params"], [int(x) for x in z["layers"]]), f"MLP {args.mlp_params}"
    if args.controller:
        return {"kind": args.controller}, f"controller {args.controller}"
    raise SystemExit("Cần một trong: --run-dir, --genome, --mlp-params, --controller")


def _neat(genome, cfg_path: Path, recurrent):
    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                      neat.DefaultStagnation, str(cfg_path))
    if recurrent is None:
        recurrent = not cfg.genome_config.feed_forward
    return neat_spec(genome, str(cfg_path.resolve()), recurrent)


def main():
    from src.console import utf8_stdout
    utf8_stdout()
    ap = argparse.ArgumentParser(description="Chấm bộ điều khiển trên các tập test")
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--genome", type=str, default=None)
    ap.add_argument("--config", type=str, default="configs/neat_recurrent.ini")
    ap.add_argument("--mlp-params", type=str, default=None)
    ap.add_argument("--controller", choices=["apf", "rule"], default=None)
    ap.add_argument("--recurrent", dest="recurrent", action="store_true", default=None)
    ap.add_argument("--no-recurrent", dest="recurrent", action="store_false")
    ap.add_argument("--data-dir", type=str, default="data/maps")
    ap.add_argument("--sets", type=str, default=",".join(DEFAULT_SETS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    spec, label = build_spec(args)
    sets = find_sets(args.data_dir, args.sets.split(","))
    report = evaluate_sets(spec, sets, workers=args.workers)
    print(label)
    print(format_report(report))

    out = args.out or (str(Path(args.run_dir) / "test_results.json") if args.run_dir else None)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"policy": label, "results": report}, f, indent=2, ensure_ascii=False)
        print(f"\nĐã ghi {out}")


if __name__ == "__main__":
    main()
