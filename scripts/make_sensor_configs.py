"""
Sinh config NEAT cho cảm biến N hướng
======================================
configs/<tên>_n{N}.ini cho mỗi config gốc trong BASES: bản sao từng byte của file gốc,
chỉ đổi dòng num_inputs thành N + 3 (N độ gần + rho, sin phi, cos phi). Chú thích trong
file giữ nguyên như file gốc (có chỗ vẫn tả 12 tia, 15 đầu vào); giá trị dùng thật là
dòng num_inputs. Với N = 12 dùng luôn file gốc (num_inputs = 15), không sinh file.
Sửa file gốc thì chạy lại script này (tests/test_fixes.py kiểm các file _n{N} khớp file gốc).

Cảm biến chọn bằng NEAT_LIDAR_MODE / NEAT_LIDAR_N (src/env/kinematics.py); NEATRunner báo
lỗi ngay nếu num_inputs của config khác N + 3.

Usage:  python scripts/make_sensor_configs.py 24 36
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASES = ("neat_recurrent", "neat_feedforward", "neat_lehman_ref")
NUM_INPUTS = re.compile(rb"(?m)^(num_inputs[ \t]*=[ \t]*)(\d+)")


def sensor_config_bytes(data: bytes, n: int) -> bytes:
    """Nội dung config cho N hướng: data với dòng num_inputs duy nhất đổi thành N + 3."""
    hits = NUM_INPUTS.findall(data)
    if len(hits) != 1:
        raise RuntimeError(f"cần đúng một dòng num_inputs, thấy {len(hits)}")
    return NUM_INPUTS.sub(lambda m: m.group(1) + str(n + 3).encode(), data, count=1)


def make(n: int) -> list[Path]:
    if n < 1:
        raise ValueError(f"N = {n}: cần N >= 1")
    if n == 12:
        return []                       # dùng luôn file gốc
    out = []
    for base in BASES:
        src = ROOT / "configs" / f"{base}.ini"
        if not src.exists():
            continue
        dst = ROOT / "configs" / f"{base}_n{n}.ini"
        dst.write_bytes(sensor_config_bytes(src.read_bytes(), n))
        out.append(dst)
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    from src.console import utf8_stdout
    utf8_stdout()
    for arg in sys.argv[1:] or ["24", "36"]:
        for p in make(int(arg)):
            print("wrote", p.relative_to(ROOT).as_posix())
