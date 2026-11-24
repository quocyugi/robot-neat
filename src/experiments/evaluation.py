"""
Test-set Evaluation Helpers (sửa lỗi L04)
==========================================
Chấm một bộ điều khiển (champion NEAT hoặc vector trọng số GA) trên các
tập test và tính SR, CR, tỉ lệ hết giờ, SPL, MTT cho từng tập. Dùng chung cho
scripts/evaluate.py và scripts/run_baselines.py.

Tập test mặc định: test/level_1 … test/level_4 và test_barn (tập nào có trong
data_dir thì chấm tập đó). Mọi bộ điều khiển chạy cùng một env, cùng bản đồ, nên
số liệu so được trực tiếp.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from src.neat_engine.parallel import ParallelEvaluator, summarize

DEFAULT_SETS = ("test/level_1", "test/level_2", "test/level_3", "test/level_4", "test_barn")


def find_sets(data_dir: str | Path, names=DEFAULT_SETS) -> dict[str, list[str]]:
    """Tên tập -> danh sách đường dẫn bản đồ (bỏ qua tập không có)."""
    out = {}
    for name in names:
        paths = [str(p.resolve()) for p in sorted((Path(data_dir) / name).glob("*.json"))]
        if paths:
            out[name] = paths
    if not out:
        raise FileNotFoundError(f"Không thấy tập test nào trong {Path(data_dir).resolve()}")
    return out


def evaluate_sets(
    spec: dict,
    sets: dict[str, list[str]],
    evaluator: Optional[ParallelEvaluator] = None,
    workers: int = 4,
) -> dict:
    """Chấm `spec` trên mọi tập. Mỗi bản đồ là một tác vụ để chia đều cho các worker.

    Tiến độ đo theo đường chim bay ở đây: chỉ số đánh giá (SR, CR, SPL) không cần
    trường geodesic, nên không phải tính trường cho bản đồ test.
    """
    own = evaluator is None
    ev = evaluator or ParallelEvaluator(workers)
    opts = {"progress": "euclid"}
    try:
        tasks, owners = [], []
        for name, paths in sets.items():
            for p in paths:
                tasks.append((spec, [p], opts))
                owners.append(name)
        flat = [res[0] for res in ev.map(tasks)]
    finally:
        if own:
            ev.close()
    report: dict = {}
    for name in sets:
        rows = [r for r, o in zip(flat, owners) if o == name]
        s = summarize(rows)
        s.pop("mean_fitness", None)
        report[name] = s
    levels = [k for k in report if k.startswith("test/level_")]
    if levels:
        report["mean_levels_1_4"] = {"SR": round(float(np.mean([report[k]["SR"] for k in levels])), 4),
                                     "CR": round(float(np.mean([report[k]["CR"] for k in levels])), 4)}
    return report


def format_report(report: dict) -> str:
    lines = [f"{'tập':<16}{'n':>5}{'SR':>8}{'CR':>8}{'hết giờ':>9}{'SPL':>8}{'MTT (s)':>9}"]
    for name, s in report.items():
        if "n" not in s:
            lines.append(f"{name:<16}{'':>5}{s['SR']:>8.2f}{s['CR']:>8.2f}")
            continue
        mtt = f"{s['MTT']:.1f}" if s.get("MTT") is not None else "-"
        lines.append(f"{name:<16}{s['n']:>5}{s['SR']:>8.2f}{s['CR']:>8.2f}{s['timeout']:>9.2f}{s['SPL']:>8.2f}{mtt:>9}")
    return "\n".join(lines)
