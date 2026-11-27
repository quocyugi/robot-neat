"""
Evaluate — chấm một bộ điều khiển trên các tập test (sửa lỗi L04)
====================================================================
Trước đây repo không có script kiểm thử: SimpleGA và APFController không được gọi ở đâu.

Usage
-----
    # champion NEAT của một lượt huấn luyện (đọc champion_genome.pkl, neat_config.ini, summary.json)
    python scripts/evaluate.py --run-dir output/train

    # genome bất kỳ
    python scripts/evaluate.py --genome path/to/genome.pkl --config configs/neat_recurrent.ini

    # champion GA của scripts/run_baselines.py
    python scripts/evaluate.py --mlp-params output/baselines/ga_b1/seed_0/champion_params.npz

    # bộ điều khiển không học (B5)
    python scripts/evaluate.py --controller apf

Kết quả in ra màn hình và ghi JSON (mặc định <run-dir>/test_results.json). Có file JSON
thì ghi thêm, cạnh nó: <tên>_bang.png (ảnh bảng), <tên>_bieu_do.png (ảnh biểu đồ TỚI ĐÍCH /
VA CHẠM / HẾT GIỜ từng tập) và <tên>.html (một trang mở bằng trình duyệt, ảnh nhúng sẵn);
--no-visual để bỏ. Đường dẫn các file in ở cuối terminal.

Lượt chạy với robot hoặc cảm biến khác mặc định (NEAT_ROBOT, NEAT_LIDAR_MODE, NEAT_LIDAR_N,
NEAT_SECTOR_LAYOUT; src/env/kinematics.py) ghi "sensor" trong summary.json; chấm lại phải đặt đúng
bốn biến đó như lúc huấn luyện trước khi chạy script, ví dụ PowerShell: $env:NEAT_LIDAR_MODE = "sector";
$env:NEAT_LIDAR_N = "24". Với --run-dir và --mlp-params, script dừng nếu robot / cảm biến hiện tại
khác "sensor" đã ghi, và dừng nếu lượt chạy không ghi "sensor" (robot cũ mặc định) mà tiến trình này
là Waffle; lượt không ghi "sensor" chấm bằng robot cũ với cảm biến khác mặc định thì chỉ cảnh báo.
Số bước tối đa ghi trên ảnh và trang HTML lấy từ robot của tiến trình này (500 hoặc 1.000 bước).

--backend batch | gpu chạy các lượt theo lô (src/batch); kết quả trùng từng bit với mặc định cpu (chế độ exact).
File kết quả ghi thêm "backend" khi khác mặc định (ví dụ "gpu-device" với --inexact: số KHÔNG trùng bit, không
dùng cho benchmark; "batch->cpu" khi tự kiểm không đạt và đã chạy bằng đường từng lượt).
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import neat  # noqa: E402
import numpy as np  # noqa: E402

from src.env.kinematics import DT, MAX_STEPS, sensor_summary  # noqa: E402
from src.experiments.evaluation import DEFAULT_SETS, evaluate_sets, find_sets, format_report  # noqa: E402
from src.neat_engine.parallel import (EVAL_BACKENDS, backend_label, make_evaluator, mlp_spec, neat_spec,  # noqa: E402
                                      resolve_backend)


def check_sensor(summary_path: Path) -> None:
    """Dừng nếu summary.json ghi "sensor" (lượt chạy với robot hoặc cảm biến khác mặc định) mà robot / cảm
    biến hiện tại khác. Nếu không ghi "sensor" (lượt chạy robot cũ mặc định, hoặc code cũ chưa ghi): dừng khi
    tiến trình này là Waffle (lượt Waffle luôn ghi "sensor"), chỉ cảnh báo khi tiến trình này là robot cũ với
    cảm biến khác mặc định. Không có summary.json thì bỏ qua."""
    if not summary_path.exists():
        return
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    have = sensor_summary()

    def name(s):
        if s is None:
            return "robot old, 12 tia điểm (mặc định)"
        return (f"robot {s.get('robot', 'old')}, {s['mode']}, N = {s['n']}"
                + (f", bố cục {s['layout']}" if s.get("layout") else ""))

    if "sensor" in summary and summary["sensor"] != have:
        run, now = name(summary["sensor"]), name(have)
        extra = " (khác ở các hằng của robot: lượt chạy từ bản mã khác)" if run == now else ""
        raise SystemExit(f"{summary_path}: lượt chạy dùng {run}, tiến trình này dùng {now}{extra}. Đặt NEAT_ROBOT, "
                         f"NEAT_LIDAR_MODE, NEAT_LIDAR_N, NEAT_SECTOR_LAYOUT theo \"sensor\" rồi chạy lại.")
    if "sensor" not in summary and have is not None:
        if have.get("robot") == "waffle":
            raise SystemExit(f"{summary_path} không ghi \"sensor\": lượt chạy dùng robot cũ, 12 tia điểm (mặc định), "
                             f"tiến trình này dùng {name(have)}. Bỏ cả bốn biến NEAT_ROBOT, NEAT_LIDAR_MODE, "
                             f"NEAT_LIDAR_N, NEAT_SECTOR_LAYOUT rồi chạy lại.")
        print(f"Cảnh báo: {summary_path} không ghi \"sensor\" (lượt chạy 12 tia điểm mặc định?), "
              f"tiến trình này dùng {name(have)}.")


def build_spec(args) -> tuple[dict, str]:
    if args.run_dir:
        run = Path(args.run_dir)
        check_sensor(run / "summary.json")
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
        check_sensor(Path(args.mlp_params).parent / "summary.json")
        z = np.load(args.mlp_params)
        return mlp_spec(z["params"], [int(x) for x in z["layers"]]), f"MLP {args.mlp_params}"
    if args.controller:
        return {"kind": args.controller}, f"controller {args.controller}"
    raise SystemExit("Cần một trong: --run-dir, --genome, --mlp-params, --controller")


def display_name(args, spec: dict) -> str:
    """Tên bộ điều khiển ghi trên ảnh và trang HTML (NEAT; GA và APF ghi nhãn baseline)."""
    from src.visual_report import short_path
    if spec["kind"] == "neat":
        kind = "hồi quy" if spec.get("recurrent") else "truyền thẳng"
        src = args.run_dir if args.run_dir else args.genome
        return f"NEAT {kind} · {short_path(src)}"
    if spec["kind"] == "mlp":
        layers = "-".join(str(x) for x in spec.get("layers") or [])
        return f"GA, mạng cố định {layers} (baseline) · {short_path(args.mlp_params)}"
    if spec["kind"] == "apf":
        return "APF, trường thế nhân tạo (baseline B5, không học)"
    return spec["kind"]


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
    ap.add_argument("--controller", choices=["apf"], default=None)
    ap.add_argument("--recurrent", dest="recurrent", action="store_true", default=None)
    ap.add_argument("--no-recurrent", dest="recurrent", action="store_false")
    ap.add_argument("--data-dir", type=str, default="data/maps")
    ap.add_argument("--sets", type=str, default=",".join(DEFAULT_SETS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--no-visual", action="store_true",
                    help="không ghi ảnh bảng, ảnh biểu đồ và trang HTML cạnh file JSON")
    ap.add_argument("--backend", choices=list(EVAL_BACKENDS), default=None,
                    help="cách đánh giá: cpu (mặc định, hoặc theo NEAT_EVAL_BACKEND), batch (NumPy theo lô), "
                         "gpu (theo lô, LiDAR + va chạm trên GPU); batch và gpu cho cùng kết quả từng bit")
    ap.add_argument("--inexact", action="store_true",
                    help="chỉ với --backend gpu: hàm siêu việt trên GPU, KHÔNG trùng bit; không dùng cho benchmark")
    args = ap.parse_args()
    if args.inexact and resolve_backend(args.backend) != "gpu":
        ap.error("--inexact chỉ dùng với --backend gpu")

    spec, label = build_spec(args)
    sets = find_sets(args.data_dir, args.sets.split(","))
    with make_evaluator(args.workers, args.backend, **({"exact": False} if args.inexact else {})) as ev:
        backend = backend_label(ev)
        if backend != "cpu":               # đường mặc định không in thêm gì
            print(f"Backend đánh giá thực dùng: {backend}", flush=True)
        report = evaluate_sets(spec, sets, evaluator=ev)
        ev.close()
    print(label)
    print(format_report(report))

    out = args.out or (str(Path(args.run_dir) / "test_results.json") if args.run_dir else None)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        result = {"policy": label, "results": report}
        if backend != "cpu":               # chỉ ghi khi khác mặc định (file của đường mặc định giữ nguyên)
            result["backend"] = backend
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nĐã ghi {out}")
        if not args.no_visual:
            from src.visual_report import eval_report, print_written, short_path, warn_failed
            try:
                print_written(eval_report(report, display_name(args, spec), Path(out).with_suffix(""),
                                          source=f"{short_path(out)} (scripts/evaluate.py)",
                                          max_steps=MAX_STEPS, dt=DT))
            except Exception as e:  # noqa: BLE001  (ảnh là phần thêm: lỗi vẽ không làm mất file JSON)
                warn_failed("ảnh và trang HTML", e)


if __name__ == "__main__":
    main()
