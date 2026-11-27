"""
Download Public Map Datasets
=============================
Tải 2 bộ dữ liệu bản đồ công khai dùng cho tập test đặc biệt (overview.md §5.2):

1. BARN — 300 world Gazebo + 300 path file, từ repo The BARN Challenge
   https://github.com/Daffan/the-barn-challenge  (MIT License)
   Bài báo: Perille et al. (2020), "Benchmarking Metric Ground Navigation", IEEE SSRR.
2. Mê cung medium / hard của Lehman & Stanley (2011), từ repo goNEAT_NS
   https://github.com/yaricom/goNEAT_NS  (MIT License)

Hai nguồn được cố định theo mã commit nên lần nào tải cũng ra đúng một phiên bản.
Dữ liệu gốc nằm ở data/raw/ (không cần đưa lên git, đã có trong .gitignore).

Usage
-----
    python scripts/download_public_data.py [--raw-dir data/raw]
"""

import argparse
import sys
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

# ── Nguồn dữ liệu (cố định commit) ───────────────────────────────────────
BARN_REPO = "Daffan/the-barn-challenge"
BARN_COMMIT = "bf5a226f6088ec96bf0d2dbee3253a8ea6119b83"
BARN_ZIP_URL = f"https://codeload.github.com/{BARN_REPO}/zip/{BARN_COMMIT}"
N_BARN_WORLDS = 300

LEHMAN_REPO = "yaricom/goNEAT_NS"
LEHMAN_COMMIT = "2ad1fce70afdfaaeb12b4991572841e588394e81"
LEHMAN_RAW_URL = f"https://raw.githubusercontent.com/{LEHMAN_REPO}/{LEHMAN_COMMIT}/"
# Đường dẫn trong repo gốc → tên file lưu trong data/raw/lehman_stanley/
LEHMAN_FILES = {
    "data/medium_maze.txt": "medium_maze.txt",
    "data/hard_maze.txt": "hard_maze.txt",
    "LICENSE": "LICENSE_goNEAT_NS.txt",
}


def main():
    parser = argparse.ArgumentParser(description="Tải dữ liệu bản đồ công khai")
    parser.add_argument("--raw-dir", type=str, default="data/raw",
                        help="Thư mục lưu dữ liệu gốc")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    download_barn(raw_dir)
    download_lehman(raw_dir)
    write_sources_file(raw_dir)
    print("\nXong. Dữ liệu gốc nằm trong", raw_dir.resolve())


def download_file(url: str, dest: Path):
    """Tải 1 file về dest (bỏ qua nếu đã có)."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  đã có {dest.name}, bỏ qua")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  tải {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def download_barn(raw_dir: Path):
    print("[1] BARN (the-barn-challenge)")
    barn_dir = raw_dir / "barn"
    path_dir = barn_dir / "path_files"

    # Đã giải nén đủ 300 world + 300 path file thì thôi
    if _count(barn_dir, "world_*.world") == N_BARN_WORLDS and \
            _count(path_dir, "path_*.npy") == N_BARN_WORLDS:
        print("  đã đủ 300 world và 300 path file, bỏ qua")
        return

    zip_path = raw_dir / "_downloads" / f"the-barn-challenge-{BARN_COMMIT[:7]}.zip"
    download_file(BARN_ZIP_URL, zip_path)

    # Chỉ lấy những file cần, đặt tên theo tên gốc (không dùng đường dẫn trong zip)
    path_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            parts = member.split("/")
            inner = "/".join(parts[1:])     # bỏ thư mục gốc "the-barn-challenge-<sha>/"
            name = parts[-1]
            dest = None
            if inner.startswith("jackal_helper/worlds/BARN/path_files/") and name.endswith(".npy"):
                dest = path_dir / name
            elif inner.startswith("jackal_helper/worlds/BARN/") and name.endswith(".world") \
                    and inner.count("/") == 3:
                dest = barn_dir / name
            elif inner == "LICENSE":
                dest = barn_dir / "LICENSE_the-barn-challenge.txt"
            elif inner == "README.md":
                dest = barn_dir / "README_the-barn-challenge.md"
            elif inner == "run.py":
                # Giữ để tham khảo toạ độ start/goal gốc, đổi đuôi cho khỏi nhầm là code dự án
                dest = barn_dir / "run_tham_khao.py.txt"
            if dest is None:
                continue
            with zf.open(member) as src, open(dest, "wb") as out:
                out.write(src.read())
            n_written += 1

    n_world = _count(barn_dir, "world_*.world")
    n_path = _count(path_dir, "path_*.npy")
    print(f"  giải nén {n_written} file: {n_world} world, {n_path} path file")
    if n_world != N_BARN_WORLDS or n_path != N_BARN_WORLDS:
        sys.exit("  LỖI: thiếu file BARN, hãy xoá data/raw/barn và chạy lại")


def download_lehman(raw_dir: Path):
    print("[2] Mê cung Lehman & Stanley (goNEAT_NS)")
    out_dir = raw_dir / "lehman_stanley"
    for repo_path, local_name in LEHMAN_FILES.items():
        download_file(LEHMAN_RAW_URL + repo_path, out_dir / local_name)


def write_sources_file(raw_dir: Path):
    """Ghi lại nguồn, commit, giấy phép để trích dẫn trong báo cáo."""
    text = (
        "NGUỒN DỮ LIỆU GỐC (tải bằng scripts/download_public_data.py)\n"
        f"Ngày tải: {date.today().isoformat()}\n\n"
        "1. BARN - Benchmark for Autonomous Robot Navigation\n"
        f"   Repo: https://github.com/{BARN_REPO} (commit {BARN_COMMIT})\n"
        "   Giấy phép: MIT (barn/LICENSE_the-barn-challenge.txt)\n"
        "   Trích dẫn: D. Perille, A. Truong, X. Xiao, P. Stone. Benchmarking Metric Ground\n"
        "   Navigation. IEEE SSRR 2020. arXiv:2008.13315\n\n"
        "2. Mê cung medium / hard của Lehman & Stanley\n"
        f"   Repo: https://github.com/{LEHMAN_REPO} (commit {LEHMAN_COMMIT})\n"
        "   Giấy phép: MIT (lehman_stanley/LICENSE_goNEAT_NS.txt)\n"
        "   Trích dẫn: J. Lehman, K. O. Stanley. Abandoning Objectives: Evolution through the\n"
        "   Search for Novelty Alone. Evolutionary Computation 19(2), 2011.\n"
    )
    (raw_dir / "SOURCES.txt").write_text(text, encoding="utf-8")


def _count(folder: Path, pattern: str) -> int:
    if not folder.exists():
        return 0
    return len(list(folder.glob(pattern)))


if __name__ == "__main__":
    main()
