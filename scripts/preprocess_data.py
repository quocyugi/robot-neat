"""
Preprocess Map Data — Tiền xử lý dữ liệu bản đồ
=================================================
Gom toàn bộ dữ liệu của đề tài về cùng một định dạng JSON và kiểm tra chất lượng:

  [1] BARN: 300 world Gazebo → lưới 30×64 → đoạn thẳng → JSON đúng schema dự án;
      chạy A* (robot r = 0.25 m, vật cản phình 0.30 m) + đo chỉ số độ khó;
      chọn 50 bản đồ trải đều từ dễ tới khó làm tập test zero-shot (câu hỏi Q3).
        → data/maps/barn_all/   (mọi bản đồ BARN robot đi lọt)
        → data/maps/test_barn/  (50 bản đồ test)
  [2] Mê cung Lehman & Stanley (medium, hard) → data/maps/test_lehman/
  [3] Kiểm tra bộ bản đồ tự sinh data/maps/{train,val,test}/level_k/:
      đủ trường, nằm trong arena, |SG| ≥ 5 m, S/G cách vật cản ≥ r + 0.2 m,
      A* có đường, τ đúng dải của cấp, không trùng lặp giữa các tập (rò rỉ)
  [3b] Kiểm tra bằng mô phỏng cho MỌI bản đồ: robot bám theo đường A* trong
      chính RobotNavEnv phải tới đích, không va chạm, trong 500 bước
  [4] Ghi data/maps/manifest.csv (1 dòng / bản đồ), reports/data/summary.json
      và các hình minh hoạ trong reports/data/figures/

Reference
---------
- overview.md §5 (Dữ liệu: bộ bản đồ & curriculum), §6.1 Q3
- docs/tien_xu_ly_du_lieu.md (giải thích từng bước)

Usage
-----
    python scripts/download_public_data.py     # 1. tải dữ liệu gốc (1 lần)
    python scripts/pregenerate_maps.py         # 2. sinh 1,300 bản đồ tự sinh
    python scripts/preprocess_data.py          # 3. bước này
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # chỉ vẽ ra file PNG, không mở cửa sổ
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.stats import rankdata

# Cho phép chạy "python scripts/preprocess_data.py" từ thư mục gốc dự án
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env.kinematics import ARENA_W, ARENA_H, DT, ROBOT_RADIUS, V_MAX, WHEEL_BASE
from src.env.robot_env import RobotNavEnv
from src.maps.generator import _MIN_SG_DIST, _SG_CLEARANCE, _TAU_RANGES
from src.maps.map_metrics import assign_level_from_tau, compute_map_metrics
from src.maps.public_maps import (
    barn_to_map,
    lehman_to_map,
    read_barn_world,
    read_lehman_maze,
)
from src.maps.validator import INFLATION_RADIUS


# ── Hằng số ──────────────────────────────────────────────────────────────
N_BARN_WORLDS = 300
N_BARN_TEST = 50                      # overview §5.2: "50 bản đồ trích từ BARN"
LEHMAN_MAZES = ["medium_maze", "hard_maze"]
SPLITS = ["train", "val", "test"]
LEVELS = [1, 2, 3, 4]
EXPECTED_PER_LEVEL = {"train": 250, "val": 25, "test": 50}
REQUIRED_KEYS = ["map_id", "difficulty_level", "bounds", "start", "goal", "segments"]
TAU_TOLERANCE = 0.01                  # lệch τ cho phép giữa giá trị đã lưu và tính lại

MANIFEST_COLUMNS = [
    "map_id", "source", "split", "level", "file", "seed", "n_segments",
    "euclidean_distance", "optimal_path_length", "tortuosity",
    "optimal_path_length_grid", "tortuosity_grid",
    "bottleneck_clearance", "mean_path_clearance", "start_clearance", "goal_clearance",
    "greedy_stuck", "greedy_progress", "trap_depth",
    "oracle_success", "oracle_steps",
    "barn_difficulty_rank", "barn_difficulty_group", "content_hash", "flags",
]
METRIC_COLUMNS = [
    "euclidean_distance", "optimal_path_length", "tortuosity",
    "optimal_path_length_grid", "tortuosity_grid",
    "bottleneck_clearance", "mean_path_clearance", "start_clearance", "goal_clearance",
    "greedy_stuck", "greedy_progress", "trap_depth",
]

# Màu cho hình (nền sáng, chữ trung tính, 1 màu chính cho dữ liệu)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
BLUE_LIGHT = "#cde2fb"
GREEN = "#008300"
YELLOW = "#eda100"


def main():
    parser = argparse.ArgumentParser(description="Tiền xử lý dữ liệu bản đồ")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--data-dir", type=str, default="data/maps")
    parser.add_argument("--report-dir", type=str, default="reports/data")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    data_dir = Path(args.data_dir)
    report_dir = Path(args.report_dir)
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # [1] + [2] Dữ liệu công khai
    barn_maps, barn_dropped = process_barn(raw_dir, data_dir)
    lehman_maps = process_lehman(raw_dir, data_dir)

    # [3] Kiểm tra bộ bản đồ tự sinh
    gen_maps, gen_rows = audit_generated_maps(data_dir)

    # Gom tất cả thành các dòng của manifest
    rows = list(gen_rows)
    for m in barn_maps:
        # 50 bản đồ test ghi đường dẫn trong test_barn/, 250 bản đồ còn lại trong barn_all/
        if m["source_info"]["in_test_barn"]:
            split = "test_barn"
            folder = "test_barn"
        else:
            split = "barn_extra"
            folder = "barn_all"
        rows.append(make_row(m, split, data_dir / folder / f"{m['map_id']}.json", []))
    for m in lehman_maps:
        rows.append(make_row(m, "test_lehman", data_dir / "test_lehman" / f"{m['map_id']}.json", []))

    # Trùng lặp nội dung giữa các tập (rò rỉ dữ liệu)
    dup_counts = mark_duplicates(rows)

    # [3b] Mô phỏng robot bám đường A* (thứ tự all_maps khớp với rows)
    all_maps = list(gen_maps) + list(barn_maps) + list(lehman_maps)
    oracle_fail = check_with_simulation(all_maps, rows)

    # [4] Ghi kết quả
    manifest_path = data_dir / "manifest.csv"
    write_manifest(rows, manifest_path)
    summary = build_summary(rows, barn_maps, barn_dropped, lehman_maps, dup_counts)
    summary["mo_phong_bam_duong_Astar"] = {
        "so_ban_do": len(all_maps),
        "khong_toi_dich": oracle_fail,
    }
    summary_path = report_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[4] Vẽ hình minh hoạ")
    figures = make_figures(gen_maps, barn_maps, lehman_maps, rows, fig_dir)

    print_summary(summary)
    print(f"\nĐã ghi: {manifest_path}")
    print(f"        {summary_path}")
    for fp in figures:
        print(f"        {fp}")
    print(f"Thời gian chạy: {time.time() - t0:.0f} s")


# ======================================================================
#  [1] BARN
# ======================================================================

def process_barn(raw_dir: Path, data_dir: Path) -> tuple[list[dict], list[int]]:
    print("[1] BARN: chuyển 300 world về schema của dự án")
    barn_dir = raw_dir / "barn"
    if not (barn_dir / "world_0.world").exists():
        print("  Không thấy data/raw/barn — hãy chạy scripts/download_public_data.py trước")
        return [], []

    out_all = data_dir / "barn_all"
    out_test = data_dir / "test_barn"
    clear_json_files(out_all)
    clear_json_files(out_test)

    maps = []
    dropped = []
    for idx in range(N_BARN_WORLDS):
        grid = read_barn_world(barn_dir / f"world_{idx}.world")
        map_data = barn_to_map(grid, idx)
        metrics = compute_map_metrics(map_data["segments"], map_data["start"], map_data["goal"])
        if metrics is None:
            # Robot r = 0.25 m (phình 0.30 m) không đi lọt → bỏ
            dropped.append(idx)
            continue
        map_data["optimal_path"] = metrics.pop("optimal_path")
        map_data["metrics"] = metrics
        map_data["difficulty_level"] = assign_level_from_tau(metrics["tortuosity"])
        maps.append(map_data)
        if (idx + 1) % 50 == 0:
            print(f"  đã xử lý {idx + 1}/{N_BARN_WORLDS}")

    print(f"  robot đi lọt {len(maps)}/{N_BARN_WORLDS} bản đồ; bỏ {len(dropped)}")
    if not maps:
        return maps, dropped

    add_barn_difficulty(maps)
    selected = select_evenly(maps, N_BARN_TEST)

    out_all.mkdir(parents=True, exist_ok=True)
    out_test.mkdir(parents=True, exist_ok=True)
    for m in maps:
        m["source_info"]["in_test_barn"] = m["map_id"] in selected
        save_json(m, out_all / f"{m['map_id']}.json")
        if m["source_info"]["in_test_barn"]:
            save_json(m, out_test / f"{m['map_id']}.json")
    print(f"  chọn {len(selected)} bản đồ trải đều độ khó → {out_test}")
    return maps, dropped


def add_barn_difficulty(maps: list[dict]):
    """Điểm khó = trung bình thứ hạng của 3 chỉ số (thang 0 → 1, càng lớn càng khó).

    - mean_path_clearance nhỏ  → đường đi sát vật cản → khó
    - bottleneck_clearance nhỏ → khe hẹp nhất hẹp     → khó
    - tortuosity lớn           → phải đi vòng nhiều   → khó
    """
    clearance = []
    bottleneck = []
    tau = []
    for m in maps:
        clearance.append(m["metrics"]["mean_path_clearance"])
        bottleneck.append(m["metrics"]["bottleneck_clearance"])
        tau.append(m["metrics"]["tortuosity"])

    # rankdata: hạng 1 = giá trị nhỏ nhất → đổi dấu để "hạng cao = khó hơn"
    rank_clear = rankdata(-np.array(clearance))
    rank_bottle = rankdata(-np.array(bottleneck))
    rank_tau = rankdata(np.array(tau))
    n = len(maps)
    score = (rank_clear + rank_bottle + rank_tau) / 3.0
    score = (score - 1.0) / max(n - 1, 1)

    # Thứ hạng cuối cùng: 1 = dễ nhất, n = khó nhất
    order = np.argsort(score, kind="stable")
    rank_of = np.zeros(n, dtype=int)
    for position in range(n):
        rank_of[order[position]] = position + 1

    for k in range(n):
        info = maps[k]["source_info"]
        info["difficulty_score"] = round(float(score[k]), 4)
        info["difficulty_rank"] = int(rank_of[k])
        if rank_of[k] <= n / 3:
            info["difficulty_group"] = "de"
        elif rank_of[k] <= 2 * n / 3:
            info["difficulty_group"] = "trung_binh"
        else:
            info["difficulty_group"] = "kho"


def select_evenly(maps: list[dict], n_pick: int) -> list[str]:
    """Xếp bản đồ từ dễ tới khó rồi lấy n_pick vị trí cách đều nhau."""
    n = len(maps)
    ids_by_rank = [""] * n
    for m in maps:
        ids_by_rank[m["source_info"]["difficulty_rank"] - 1] = m["map_id"]

    selected = []
    for pos in np.linspace(0, n - 1, min(n_pick, n)):
        map_id = ids_by_rank[int(round(pos))]
        if map_id not in selected:
            selected.append(map_id)
    return selected


# ======================================================================
#  [2] Mê cung Lehman & Stanley
# ======================================================================

def process_lehman(raw_dir: Path, data_dir: Path) -> list[dict]:
    print("[2] Mê cung Lehman & Stanley")
    out_dir = data_dir / "test_lehman"
    clear_json_files(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    maps = []
    for name in LEHMAN_MAZES:
        txt_path = raw_dir / "lehman_stanley" / f"{name}.txt"
        if not txt_path.exists():
            print(f"  Không thấy {txt_path} — hãy chạy scripts/download_public_data.py trước")
            continue
        map_data = lehman_to_map(read_lehman_maze(txt_path), name)
        metrics = compute_map_metrics(map_data["segments"], map_data["start"], map_data["goal"])
        if metrics is None:
            print(f"  {name}: robot r = 0.25 m KHÔNG đi lọt — bỏ qua")
            continue
        map_data["optimal_path"] = metrics.pop("optimal_path")
        map_data["metrics"] = metrics
        map_data["difficulty_level"] = assign_level_from_tau(metrics["tortuosity"])
        save_json(map_data, out_dir / f"{map_data['map_id']}.json")
        maps.append(map_data)
        print(f"  {name}: τ = {metrics['tortuosity']:.2f}, đường tối ưu "
              f"{metrics['optimal_path_length']:.2f} m, khe hẹp nhất "
              f"{metrics['bottleneck_clearance']:.2f} m")
    return maps


# ======================================================================
#  [3] Kiểm tra bộ bản đồ tự sinh
# ======================================================================

def audit_generated_maps(data_dir: Path) -> tuple[list[dict], list[dict]]:
    print("[3] Kiểm tra bộ bản đồ tự sinh (tính lại A* và chỉ số cho từng bản đồ)")
    maps = []
    rows = []
    for split in SPLITS:
        for level in LEVELS:
            folder = data_dir / split / f"level_{level}"
            if not folder.exists():
                print(f"  thiếu thư mục {folder}")
                continue
            files = sorted(folder.glob("*.json"))
            for fp in files:
                with open(fp, "r", encoding="utf-8") as f:
                    map_data = json.load(f)
                metrics, flags = check_generated_map(map_data, level)
                # Dùng chỉ số vừa tính lại cho manifest (độc lập với giá trị đã lưu)
                map_view = dict(map_data)
                if metrics is not None:
                    map_view["optimal_path"] = metrics.pop("optimal_path")
                    map_view["metrics"] = metrics
                map_view["split"] = split
                maps.append(map_view)
                rows.append(make_row(map_view, split, fp, flags))
            print(f"  {split}/level_{level}: {len(files)} bản đồ")
    return maps, rows


def check_generated_map(map_data: dict, folder_level: int) -> tuple[dict | None, list[str]]:
    """Kiểm tra 1 bản đồ tự sinh, trả về (chỉ số tính lại, danh sách cờ lỗi)."""
    flags = []
    for key in REQUIRED_KEYS:
        if key not in map_data:
            flags.append(f"thieu_truong_{key}")
    if flags:
        return None, flags

    if map_data["difficulty_level"] != folder_level:
        flags.append("sai_thu_muc_cap")

    start = np.array(map_data["start"], dtype=float)
    goal = np.array(map_data["goal"], dtype=float)
    if not inside_arena(start):
        flags.append("start_ngoai_arena")
    if not inside_arena(goal):
        flags.append("goal_ngoai_arena")
    if np.linalg.norm(goal - start) < _MIN_SG_DIST - 1e-9:
        flags.append("SG_duoi_5m")

    metrics = compute_map_metrics(map_data["segments"], start, goal)
    if metrics is None:
        flags.append("khong_co_duong_Astar")
        return None, flags

    if metrics["start_clearance"] < _SG_CLEARANCE - 1e-6:
        flags.append("start_sat_vat_can")
    if metrics["goal_clearance"] < _SG_CLEARANCE - 1e-6:
        flags.append("goal_sat_vat_can")
    lo, hi = _TAU_RANGES[folder_level]
    if not (lo <= metrics["tortuosity"] <= hi):
        flags.append("tau_ngoai_dai_cap")
    stored = map_data.get("metrics", {})
    if "tortuosity" in stored and abs(stored["tortuosity"] - metrics["tortuosity"]) > TAU_TOLERANCE:
        flags.append("tau_da_luu_lech")
    return metrics, flags


def inside_arena(p: np.ndarray) -> bool:
    """Tâm robot phải nằm trong arena và cách tường biên ít nhất 1 bán kính."""
    ok_x = ROBOT_RADIUS <= p[0] <= ARENA_W - ROBOT_RADIUS
    ok_y = ROBOT_RADIUS <= p[1] <= ARENA_H - ROBOT_RADIUS
    return bool(ok_x and ok_y)


# ======================================================================
#  [3b] Kiểm tra bằng mô phỏng
# ======================================================================

def check_with_simulation(all_maps: list[dict], rows: list[dict]) -> list[str]:
    """Chạy oracle_rollout cho từng bản đồ, ghi kết quả vào dòng manifest tương ứng."""
    print(f"[3b] Mô phỏng: robot bám đường A* trong RobotNavEnv ({len(all_maps)} bản đồ)")
    failed = []
    for k in range(len(all_maps)):
        m = all_maps[k]
        row = rows[k]
        if "optimal_path" not in m:
            row["oracle_success"] = ""
            row["oracle_steps"] = ""
            continue
        info = oracle_rollout(m)
        row["oracle_success"] = bool(info["is_success"])
        row["oracle_steps"] = int(info["steps"])
        if not info["is_success"]:
            if info["is_collision"]:
                row["flags"].append("oracle_va_cham")
            else:
                row["flags"].append("oracle_het_500_buoc")
            failed.append(f"{row['split']}/{row['map_id']}")
    print(f"  tới đích {len(all_maps) - len(failed)}/{len(all_maps)}; không tới: {failed}")
    return failed


def oracle_rollout(map_data: dict) -> dict:
    """Robot "biết trước bản đồ" bám theo đường A* đã làm mượt trong RobotNavEnv.

    Điều khiển đơn giản: lệch hướng thì quay tại chỗ (bánh trái lùi, bánh phải
    tiến) cho tới khi nhắm đúng điểm kế tiếp, rồi đi thẳng tới điểm đó.
    Dùng để chắc chắn bản đồ đi được thật với động học + va chạm + giới hạn
    500 bước của môi trường (không phải để so sánh với NEAT).
    """
    env = RobotNavEnv(map_data=map_data)
    env.reset()
    waypoints = map_data["optimal_path"][1:]
    k = 0
    turn_per_step = 2 * V_MAX / WHEEL_BASE * DT     # góc quay mỗi bước khi quay hết tốc (rad)
    info = {"is_success": False, "is_collision": False, "steps": 0}
    terminated = False
    truncated = False

    while not (terminated or truncated):
        diff = np.array(waypoints[k]) - env.pos
        dist = float(np.linalg.norm(diff))
        # Tới điểm gãy thì chuyển sang điểm kế tiếp
        if dist < 0.01 and k < len(waypoints) - 1:
            k += 1
            continue
        heading_error = np.arctan2(diff[1], diff[0]) - env.theta
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi   # đưa về [-π, π]
        if dist > 0.01 and abs(heading_error) > 0.01:
            u = float(np.clip(heading_error / turn_per_step, -1.0, 1.0))
            action = [-u, u]                     # quay tại chỗ
        else:
            u = float(np.clip(dist / (V_MAX * DT), 0.0, 1.0))
            action = [u, u]                      # đi thẳng, chậm lại khi sắp tới điểm
        _, _, terminated, truncated, info = env.step(np.array(action, dtype=np.float32))
    env.close()
    return info


# ======================================================================
#  Manifest, trùng lặp, tóm tắt
# ======================================================================

def make_row(map_data: dict, split: str, file_path: Path, flags: list[str]) -> dict:
    """1 dòng manifest cho 1 bản đồ."""
    metrics = map_data.get("metrics", {})
    info = map_data.get("source_info", {})
    row = {
        "map_id": map_data.get("map_id", ""),
        "source": map_data.get("source", "generator"),
        "split": split,
        "level": map_data.get("difficulty_level", ""),
        "file": file_path.as_posix(),
        "seed": map_data.get("seed", ""),
        "n_segments": len(map_data.get("segments", [])),
        "barn_difficulty_rank": info.get("difficulty_rank", ""),
        "barn_difficulty_group": info.get("difficulty_group", ""),
        "content_hash": content_hash(map_data),
        "flags": list(flags),
    }
    for col in METRIC_COLUMNS:
        row[col] = metrics.get(col, "")
    if row["seed"] is None:
        row["seed"] = ""
    return row


def content_hash(map_data: dict) -> str:
    """Mã băm nội dung (start, goal, segments làm tròn tới mm) để tìm bản đồ trùng."""
    if "segments" not in map_data:
        return ""
    payload = {
        # cộng 0.0 để -0.0 thành 0.0 (tránh băm khác nhau cho cùng một hình)
        "start": (np.round(np.asarray(map_data["start"], dtype=float), 3) + 0.0).tolist(),
        "goal": (np.round(np.asarray(map_data["goal"], dtype=float), 3) + 0.0).tolist(),
        "segments": (np.round(np.asarray(map_data["segments"], dtype=float), 3) + 0.0).tolist(),
    }
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def mark_duplicates(rows: list[dict]) -> dict:
    """Gắn cờ các bản đồ có nội dung trùng nhau; đếm trùng giữa các tập / trong một tập."""
    groups = {}
    for row in rows:
        if row["content_hash"] == "":
            continue
        groups.setdefault(row["content_hash"], []).append(row)

    n_cross = 0
    n_within = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        splits = set()
        for row in members:
            splits.add(row["split"])
        for row in members:
            others = []
            for other in members:
                if other is not row:
                    others.append(f"{other['split']}/{other['map_id']}")
            if len(splits) > 1:
                row["flags"].append("trung_giua_tap:" + "|".join(others))
                n_cross += 1
            else:
                row["flags"].append("trung_trong_tap:" + "|".join(others))
                n_within += 1
    return {"so_ban_do_trung_giua_cac_tap": n_cross, "so_ban_do_trung_trong_cung_tap": n_within}


def write_manifest(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig để mở bằng Excel không lỗi tiếng Việt
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["flags"] = ";".join(row["flags"])
            writer.writerow(out)


def build_summary(rows, barn_maps, barn_dropped, lehman_maps, dup_counts) -> dict:
    # Số bản đồ tự sinh theo tập / cấp
    counts = {}
    for split in SPLITS:
        counts[split] = {}
        for level in LEVELS:
            counts[split][f"level_{level}"] = 0
    flag_counts = {}
    for row in rows:
        if row["split"] in SPLITS:
            counts[row["split"]][f"level_{row['level']}"] += 1
        for flag in row["flags"]:
            name = flag.split(":")[0]
            flag_counts[name] = flag_counts.get(name, 0) + 1

    # Thống kê chỉ số theo nhóm
    stats = {}
    for group_name, group_rows in group_rows_for_report(rows).items():
        stats[group_name] = {"n": len(group_rows)}
        for col in ["tortuosity", "optimal_path_length", "bottleneck_clearance",
                    "mean_path_clearance", "trap_depth", "n_segments"]:
            values = numeric_values(group_rows, col)
            if values:
                stats[group_name][col] = {
                    "min": round(float(np.min(values)), 3),
                    "median": round(float(np.median(values)), 3),
                    "max": round(float(np.max(values)), 3),
                }
        stuck = numeric_values(group_rows, "greedy_stuck")
        if stuck:
            stats[group_name]["ti_le_robot_tham_lam_bi_ket"] = round(float(np.mean(stuck)), 3)

    selected = []
    for m in barn_maps:
        if m["source_info"]["in_test_barn"]:
            selected.append(m["map_id"])

    lehman_info = {}
    for m in lehman_maps:
        lehman_info[m["map_id"]] = {
            "maze_size_m": m["source_info"]["maze_size_m"],
            "tortuosity": m["metrics"]["tortuosity"],
            "optimal_path_length": m["metrics"]["optimal_path_length"],
            "bottleneck_clearance": m["metrics"]["bottleneck_clearance"],
            "trap_depth": m["metrics"]["trap_depth"],
        }

    return {
        "thoi_diem_chay": datetime.now().isoformat(timespec="seconds"),
        "robot_radius": ROBOT_RADIUS,
        "inflation_radius": INFLATION_RADIUS,
        "ban_do_tu_sinh": {
            "so_luong": counts,
            "ky_vong_moi_cap": EXPECTED_PER_LEVEL,
        },
        "co_loi": flag_counts,
        "trung_lap": dup_counts,
        "barn": {
            "so_world": N_BARN_WORLDS,
            "robot_di_lot": len(barn_maps),
            "bi_bo": barn_dropped,
            "so_ban_do_test": len(selected),
            "test_barn": selected,
        },
        "lehman_stanley": lehman_info,
        "chi_so_theo_nhom": stats,
    }


def group_rows_for_report(rows: list[dict]) -> dict:
    """Chia các dòng manifest thành nhóm để thống kê / vẽ: Cấp 1..4, BARN, Lehman."""
    groups = {}
    for level in LEVELS:
        groups[f"Cấp {level}"] = []
    groups["BARN (50 test)"] = []
    groups["BARN (cả 300)"] = []
    groups["Lehman"] = []
    for row in rows:
        if row["split"] in SPLITS:
            groups[f"Cấp {row['level']}"].append(row)
        elif row["split"] == "test_barn":
            groups["BARN (50 test)"].append(row)
            groups["BARN (cả 300)"].append(row)
        elif row["split"] == "barn_extra":
            groups["BARN (cả 300)"].append(row)
        elif row["split"] == "test_lehman":
            groups["Lehman"].append(row)
    return groups


def numeric_values(rows: list[dict], col: str) -> list[float]:
    values = []
    for row in rows:
        v = row.get(col, "")
        if v == "" or v is None:
            continue
        values.append(float(v))
    return values


def print_summary(summary: dict):
    print("\n" + "=" * 64)
    print("  TÓM TẮT TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 64)
    counts = summary["ban_do_tu_sinh"]["so_luong"]
    for split in SPLITS:
        parts = []
        total = 0
        for level in LEVELS:
            n = counts[split][f"level_{level}"]
            total += n
            parts.append(f"L{level}={n}")
        expected = EXPECTED_PER_LEVEL[split] * len(LEVELS)
        print(f"  {split:5s}: {total:4d}/{expected} bản đồ  ({', '.join(parts)})")
    if summary["co_loi"]:
        print("  Cờ lỗi:", summary["co_loi"])
    else:
        print("  Cờ lỗi: không có")
    print("  Trùng lặp:", summary["trung_lap"])
    barn = summary["barn"]
    print(f"  BARN: robot đi lọt {barn['robot_di_lot']}/{barn['so_world']}, "
          f"chọn {barn['so_ban_do_test']} bản đồ test")
    for name, info in summary["lehman_stanley"].items():
        print(f"  {name}: τ = {info['tortuosity']}, L* = {info['optimal_path_length']} m, "
              f"khe hẹp = {info['bottleneck_clearance']} m")


# ======================================================================
#  Hình minh hoạ
# ======================================================================

def make_figures(gen_maps, barn_maps, lehman_maps, rows, fig_dir: Path) -> list[Path]:
    setup_plot_style()
    written = []

    # 1. Ví dụ bản đồ tự sinh theo cấp (4 bản đồ đầu tiên của tập train mỗi cấp)
    examples = {}
    for level in LEVELS:
        examples[level] = []
    for m in gen_maps:
        if m["split"] == "train" and len(examples[m["difficulty_level"]]) < 4:
            examples[m["difficulty_level"]].append(m)
    fig, axes = plt.subplots(4, 4, figsize=(9, 9.6))
    for r, level in enumerate(LEVELS):
        for c in range(4):
            ax = axes[r][c]
            if c < len(examples[level]):
                m = examples[level][c]
                title = f"Cấp {level} · {m['map_id']}\nτ = {m['metrics']['tortuosity']:.2f}"
                draw_map(ax, m, title)
            else:
                ax.axis("off")
    add_map_legend(fig)
    fig.suptitle("Bản đồ tự sinh theo 4 cấp độ (tập train)", color=INK, fontsize=11)
    written.append(save_figure(fig, fig_dir / "01_ban_do_tu_sinh_theo_cap.png"))

    # 2. Ví dụ BARN: 8 bản đồ trong tập test, trải từ dễ tới khó
    test_barn = []
    for m in barn_maps:
        if m["source_info"]["in_test_barn"]:
            test_barn.append(m)
    if test_barn:
        ranked = sort_by_barn_rank(test_barn)
        picks = []
        for pos in np.linspace(0, len(ranked) - 1, min(8, len(ranked))):
            picks.append(ranked[int(round(pos))])
        fig, axes = plt.subplots(2, 4, figsize=(9, 5.4))
        for k in range(8):
            ax = axes[k // 4][k % 4]
            if k < len(picks):
                m = picks[k]
                info = m["source_info"]
                title = (f"{m['map_id']} · hạng {info['difficulty_rank']}/{len(barn_maps)}\n"
                         f"khe hẹp {m['metrics']['bottleneck_clearance']:.2f} m")
                draw_map(ax, m, title)
            else:
                ax.axis("off")
        add_map_legend(fig)
        fig.suptitle("BARN đã chuyển về arena 10 m × 10 m (tỉ lệ 1:1), từ dễ tới khó",
                     color=INK, fontsize=11)
        written.append(save_figure(fig, fig_dir / "02_barn_vi_du.png"))

    # 3. Hai mê cung Lehman & Stanley
    if lehman_maps:
        fig, axes = plt.subplots(1, len(lehman_maps), figsize=(4.6 * len(lehman_maps), 5))
        if len(lehman_maps) == 1:
            axes = [axes]
        for k, m in enumerate(lehman_maps):
            title = (f"{m['map_id']}\nτ = {m['metrics']['tortuosity']:.2f}, "
                     f"độ sâu bẫy {m['metrics']['trap_depth']:.2f} m")
            draw_map(axes[k], m, title)
        add_map_legend(fig)
        fig.suptitle("Mê cung Lehman & Stanley (robot gốc 8 px = 0.25 m)", color=INK, fontsize=11)
        written.append(save_figure(fig, fig_dir / "03_me_cung_lehman.png"))

    # 4. Phân bố chỉ số theo nhóm
    groups = group_rows_for_report(rows)
    order = ["Cấp 1", "Cấp 2", "Cấp 3", "Cấp 4", "BARN (50 test)", "Lehman"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    # Mỗi đường tham chiếu: (giá trị, nhãn, nhãn nằm "tren" hay "duoi" đường)
    plot_metric_panel(axes[0], groups, order, "tortuosity", "Độ quanh co τ",
                      [(1.05, "ranh giới cấp 2", "tren"), (1.25, "ranh giới cấp 3", "tren"),
                       (1.60, "ranh giới cấp 4", "tren")])
    plot_metric_panel(axes[1], groups, order, "bottleneck_clearance", "Khe hẹp nhất (m)",
                      [(INFLATION_RADIUS, "biên A* 0.30 m", "tren"),
                       (ROBOT_RADIUS, "bán kính robot 0.25 m", "duoi")])
    axes[1].set_ylim(bottom=0.1)
    plot_metric_panel(axes[2], groups, order, "trap_depth", "Độ sâu bẫy (m)", [])
    axes[0].set_title("Phải đi vòng bao nhiêu", color=INK, fontsize=10)
    axes[1].set_title("Lối đi hẹp nhất rộng bao nhiêu", color=INK, fontsize=10)
    axes[2].set_title("Robot tham lam phải lùi bao xa để thoát", color=INK, fontsize=10)
    fig.tight_layout()
    written.append(save_figure(fig, fig_dir / "04_phan_bo_chi_so.png"))

    # 5. Cách chọn 50 bản đồ BARN
    if barn_maps:
        ranked = sort_by_barn_rank(barn_maps)
        x_other, y_other, x_sel, y_sel = [], [], [], []
        for m in ranked:
            info = m["source_info"]
            if info["in_test_barn"]:
                x_sel.append(info["difficulty_rank"])
                y_sel.append(info["difficulty_score"])
            else:
                x_other.append(info["difficulty_rank"])
                y_other.append(info["difficulty_score"])
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        ax.scatter(x_other, y_other, s=10, color=MUTED, alpha=0.5, linewidths=0,
                   label=f"BARN còn lại ({len(x_other)}, lưu ở barn_all)")
        ax.scatter(x_sel, y_sel, s=22, color=BLUE, linewidths=0,
                   label=f"Chọn vào test_barn ({len(x_sel)})")
        ax.set_xlabel("Hạng độ khó (1 = dễ nhất)")
        ax.set_ylabel("Điểm độ khó (0 → 1)")
        ax.set_title("50 bản đồ BARN được lấy cách đều theo hạng độ khó", color=INK, fontsize=10)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        written.append(save_figure(fig, fig_dir / "05_barn_chon_50.png"))

    return written


def setup_plot_style():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 9,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "legend.labelcolor": INK_2,
    })


def draw_map(ax, map_data: dict, title: str):
    """Vẽ 1 bản đồ: vật cản, đường A* đã làm mượt, Start (đúng bán kính robot), Goal."""
    segs = np.asarray(map_data["segments"], dtype=float)
    ax.add_collection(LineCollection(segs, colors=INK, linewidths=0.9))
    path = map_data.get("optimal_path")
    if path:
        p = np.array(path)
        ax.plot(p[:, 0], p[:, 1], color=BLUE, linewidth=1.3, linestyle="--")
    s = map_data["start"]
    g = map_data["goal"]
    ax.add_patch(plt.Circle((s[0], s[1]), ROBOT_RADIUS, color=GREEN, zorder=3))
    ax.plot(g[0], g[1], marker="*", markersize=11, color=YELLOW,
            markeredgecolor=INK, markeredgewidth=0.6, zorder=3)
    ax.set_xlim(-0.1, ARENA_W + 0.1)
    ax.set_ylim(-0.1, ARENA_H + 0.1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=8, color=INK)


def add_map_legend(fig):
    handles = [
        Line2D([0], [0], color=INK, linewidth=1, label="Vật cản / tường"),
        Line2D([0], [0], color=BLUE, linewidth=1.3, linestyle="--", label="Đường A* đã làm mượt"),
        Line2D([0], [0], marker="o", color=GREEN, linestyle="", markersize=8,
               label="Start (đúng cỡ robot r = 0.25 m)"),
        Line2D([0], [0], marker="*", color=YELLOW, markeredgecolor=INK, linestyle="",
               markersize=11, label="Goal"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.07, top=0.9, hspace=0.35, wspace=0.08)


def plot_metric_panel(ax, groups: dict, order: list[str], col: str, ylabel: str, ref_lines: list):
    """Hộp (trung vị, tứ phân vị) + các điểm của từng nhóm cho 1 chỉ số."""
    rng = np.random.default_rng(0)   # cố định độ rung ngang của điểm để hình tái lập được
    labels = []
    for k, name in enumerate(order):
        values = numeric_values(groups.get(name, []), col)
        position = k + 1
        n = len(values)
        labels.append(f"{name}\n(n={n})")
        if n == 0:
            continue
        if n >= 5:
            ax.boxplot([values], positions=[position], widths=0.55, showfliers=False,
                       patch_artist=True,
                       boxprops={"facecolor": BLUE_LIGHT, "edgecolor": BLUE, "linewidth": 1},
                       medianprops={"color": INK, "linewidth": 1.4},
                       whiskerprops={"color": BLUE, "linewidth": 1},
                       capprops={"color": BLUE, "linewidth": 1})
            jitter = rng.uniform(-0.17, 0.17, size=n)
            ax.scatter(position + jitter, values, s=5, color=BLUE, alpha=0.3, linewidths=0, zorder=3)
        else:
            # Quá ít điểm (mê cung Lehman) → chỉ vẽ điểm và ghi tên;
            # điểm cao hơn ghi nhãn phía trên, điểm thấp hơn ghi phía dưới cho khỏi đè nhau
            ax.scatter([position] * n, values, s=28, color=BLUE, zorder=3)
            names = []
            for row in groups[name]:
                if row.get(col, "") != "":
                    names.append(row["map_id"].replace("lehman_", "").replace("_maze", ""))
            for v, label in zip(values, names):
                if v == max(values):
                    offset, va = (5, 3), "bottom"
                else:
                    offset, va = (5, -3), "top"
                ax.annotate(label, (position, v), xytext=offset, textcoords="offset points",
                            fontsize=7, color=INK_2, va=va)
    # Nhãn đường tham chiếu đặt sát mép trái, phía trên hoặc dưới đường
    for y, text, side in ref_lines:
        ax.axhline(y, color=MUTED, linewidth=0.8, linestyle=":")
        if side == "tren":
            offset, va = (0, 2), "bottom"
        else:
            offset, va = (0, -2), "top"
        ax.annotate(text, (0.5, y), xytext=offset, textcoords="offset points",
                    fontsize=7, color=MUTED, va=va, ha="left")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_xlim(0.4, len(order) + 0.6)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def sort_by_barn_rank(maps: list[dict]) -> list[dict]:
    """Sắp xếp bản đồ BARN theo hạng độ khó (không dùng lambda)."""
    pairs = []
    for m in maps:
        pairs.append((m["source_info"]["difficulty_rank"], m["map_id"], m))
    pairs.sort(key=_first_two)
    result = []
    for p in pairs:
        result.append(p[2])
    return result


def _first_two(item):
    return item[0], item[1]


def save_figure(fig, path: Path) -> Path:
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ======================================================================
#  Tiện ích
# ======================================================================

def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def clear_json_files(folder: Path):
    """Xoá file .json cũ trong thư mục đầu ra để không lẫn kết quả lần chạy trước."""
    if not folder.exists():
        return
    for fp in folder.glob("*.json"):
        fp.unlink()


if __name__ == "__main__":
    main()
