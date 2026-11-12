"""
Pre-Generate Map Dataset
=========================
Creates 1,300 maps across 4 difficulty levels:
  - Train: 1,000 maps (250/level, seeds từ 0)
  - Val:     100 maps (25/level,  seeds từ 100,000)
  - Test:    200 maps (50/level,  seeds từ 200,000)

Mỗi tập luôn đủ số bản đồ: seed nào không sinh được bản đồ hợp lệ thì thử seed
kế tiếp (trước đây dùng đúng n seed nên tập có thể thiếu bản đồ).

Usage
-----
    python scripts/pregenerate_maps.py [--data-dir data/maps]
    python scripts/pregenerate_maps.py --levels 3          # chỉ sinh lại cấp 3 (tường dày, L27)
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

# Cho phép chạy "python scripts/pregenerate_maps.py" từ thư mục gốc dự án
# (nếu không, Python không tìm thấy package src)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.maps.generator import MapGenerator


def main():
    from src.console import utf8_stdout
    utf8_stdout()
    parser = argparse.ArgumentParser(description="Generate map dataset")
    parser.add_argument(
        "--data-dir", type=str, default="data/maps",
        help="Output root directory",
    )
    parser.add_argument("--levels", type=str, default="1,2,3,4", help="ví dụ 3 hoặc 1,2,3,4")
    parser.add_argument("--keep-old", action="store_true",
                        help="không xóa bản đồ cũ của cấp được sinh lại")
    args = parser.parse_args()

    gen = MapGenerator()
    data_dir = Path(args.data_dir)
    levels = [int(x) for x in args.levels.split(",")]

    splits = {
        "train": {"maps_per_level": 250, "seed_offset": 0},
        "val":   {"maps_per_level": 25,  "seed_offset": 100_000},
        "test":  {"maps_per_level": 50,  "seed_offset": 200_000},
    }

    total_written = 0
    for split_name, split_cfg in splits.items():
        for level in levels:
            n = split_cfg["maps_per_level"]
            offset = split_cfg["seed_offset"]
            out_dir = data_dir / split_name / f"level_{level}"
            if out_dir.exists() and not args.keep_old:
                shutil.rmtree(out_dir)   # tránh trộn bản đồ cũ và mới khi seed hỏng thay đổi

            print(f"Generating {split_name}/level_{level}: {n} maps (seeds from {offset})...")

            t0 = time.time()
            written = gen.generate_n(level, n, offset, out_dir)
            elapsed = time.time() - t0

            total_written += len(written)
            last_seed = int(Path(written[-1]).stem.split("_")[1]) if written else offset
            print(f"  -> {len(written)}/{n} valid maps, seeds {offset}-{last_seed}, {elapsed:.1f}s")

    print(f"\nDone! Total maps written: {total_written}")


if __name__ == "__main__":
    main()
