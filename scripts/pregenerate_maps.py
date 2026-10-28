"""
Pre-Generate Map Dataset
=========================
Creates 1,300 maps across 4 difficulty levels:
  - Train: 1,000 maps (250/level, seeds 0–99,999)
  - Val:     100 maps (25/level,  seeds 100,000–100,099)
  - Test:    200 maps (50/level,  seeds 200,000–200,199)

Usage
-----
    python scripts/pregenerate_maps.py [--data-dir data/maps]
"""

import argparse
import sys
import time
from pathlib import Path

# Cho phép chạy "python scripts/pregenerate_maps.py" từ thư mục gốc dự án
# (nếu không, Python không tìm thấy package src)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.maps.generator import MapGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate map dataset")
    parser.add_argument(
        "--data-dir", type=str, default="data/maps",
        help="Output root directory",
    )
    args = parser.parse_args()

    gen = MapGenerator()
    data_dir = Path(args.data_dir)

    splits = {
        "train": {"maps_per_level": 250, "seed_offset": 0},
        "val":   {"maps_per_level": 25,  "seed_offset": 100_000},
        "test":  {"maps_per_level": 50,  "seed_offset": 200_000},
    }

    total_written = 0
    for split_name, split_cfg in splits.items():
        for level in range(1, 5):
            n = split_cfg["maps_per_level"]
            offset = split_cfg["seed_offset"]
            seeds = range(offset, offset + n)
            out_dir = data_dir / split_name / f"level_{level}"

            print(f"Generating {split_name}/level_{level}: "
                  f"{n} maps (seeds {offset}–{offset+n-1})...")

            t0 = time.time()
            written = gen.generate_dataset(level, seeds, out_dir)
            elapsed = time.time() - t0

            total_written += len(written)
            print(f"  → {len(written)}/{n} valid maps in {elapsed:.1f}s")

    print(f"\nDone! Total maps written: {total_written}")


if __name__ == "__main__":
    main()
