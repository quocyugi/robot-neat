"""
Summarize Results — gom kết quả test nhiều seed và so các phương pháp (sửa lỗi L04)
====================================================================================
Đọc <thư mục phương pháp>/seed_*/test_results.json (do scripts/evaluate.py hoặc
scripts/run_baselines.py ghi), tính trung bình, trung vị, IQR theo seed, rồi chạy
pairwise_tests và vẽ boxplot của src/experiments/stats.py (trước đây không được gọi).

Usage
-----
    python scripts/summarize_results.py \
        --runs M2=output/ablation/M2 M6=output/ablation/M6 GA=output/baselines/ga_b1 \
        --set mean_levels_1_4 --out output/summary_SR

--set: mean_levels_1_4 (mặc định), test/level_3, test/level_4, test_barn, ...
Seed là đơn vị phân tích: cần ít nhất khoảng 10 seed mỗi phương pháp thì kiểm
định mới có ý nghĩa (báo cáo mục 6.7, 11.2).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402


def collect(root: Path, set_name: str, metric: str) -> dict[str, float]:
    out = {}
    for f in sorted(root.glob("seed_*/test_results.json")):
        rep = json.load(open(f, encoding="utf-8"))["results"]
        if set_name in rep and metric in rep[set_name]:
            out[f.parent.name] = float(rep[set_name][metric])
    return out


def main():
    from src.console import utf8_stdout
    utf8_stdout()
    ap = argparse.ArgumentParser(description="Gom kết quả theo seed và kiểm định")
    ap.add_argument("--runs", nargs="+", required=True, help="TÊN=thư_mục, ví dụ M6=output/ablation/M6")
    ap.add_argument("--set", default="mean_levels_1_4")
    ap.add_argument("--metric", default="SR")
    ap.add_argument("--out", default="output/summary")
    a = ap.parse_args()

    data: dict[str, dict[str, float]] = {}
    for item in a.runs:
        name, path = item.split("=", 1)
        data[name] = collect(Path(path), a.set, a.metric)

    table = {}
    print(f"{a.metric} trên {a.set}")
    print(f"{'phương pháp':<14}{'seed':>6}{'TB':>8}{'trung vị':>10}{'IQR':>14}{'min':>7}{'max':>7}")
    for name, per_seed in data.items():
        v = np.array(list(per_seed.values()))
        if v.size == 0:
            print(f"{name:<14}{0:>6}  (không có test_results.json)")
            continue
        q1, q3 = np.percentile(v, [25, 75])
        table[name] = {"n_seeds": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
                       "q1": float(q1), "q3": float(q3), "min": float(v.min()), "max": float(v.max()), "per_seed": per_seed}
        print(f"{name:<14}{v.size:>6}{v.mean():>8.3f}{np.median(v):>10.3f}{f'{q1:.3f}-{q3:.3f}':>14}{v.min():>7.3f}{v.max():>7.3f}")

    result = {"set": a.set, "metric": a.metric, "methods": table}
    usable = {k: np.array(list(data[k].values())) for k in table if table[k]["n_seeds"] >= 2}
    if len(usable) >= 2:
        from src.experiments.stats import pairwise_tests, plot_boxplot
        if min(len(v) for v in usable.values()) < 5:
            print("\nLưu ý: dưới 5 seed mỗi phương pháp, kiểm định gần như không thể có ý nghĩa.")
        tests = pairwise_tests(usable)
        for t in tests:
            print(f"  {t['comparison']}: p = {t['p_value']:.4f} (ngưỡng đã hiệu chỉnh {t['corrected_alpha']:.4f}), r = {t['effect_size_r']:.2f}")
        result["pairwise_tests"] = tests
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        plot_boxplot(usable, metric_name=f"{a.metric} ({a.set})", output_path=a.out + ".png")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=float)
    print(f"\nĐã ghi {a.out}.json")


if __name__ == "__main__":
    main()
