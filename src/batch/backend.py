"""
Lớp mảng xp: NumPy (CPU) hoặc CuPy (GPU)
=========================================
Cùng một mã mô phỏng theo lô (src/batch/sim.py) chạy với xp = numpy hoặc xp = cupy.
Các hàm siêu việt (cos, sin, arctan2, tanh) gom vào lớp Backend vì chúng quyết định việc
kết quả có trùng từng bit với đường chạy từng lượt (src/neat_engine/parallel.py) hay không:

- exact (mặc định): tính trên host bằng ĐÚNG hàm mà đường từng lượt dùng: np.cos / np.sin /
  np.arctan2 trên mảng NumPy (trên máy này cho cùng kết quả với lời gọi vô hướng; được kiểm
  lại lúc khởi tạo, xem src/batch/numerics.py), math.tanh từng phần tử cho mạng NEAT
  (np.tanh lệch math.tanh ở bit cuối với khoảng 1/5 giá trị), np.tanh cho FixedMLP (đúng hàm
  FixedMLP dùng). Với xp = cupy, mỗi lần gọi là một lần chép về host rồi chép lên.
- device (chỉ GPU, phải bật tường minh): cupy.cos, cupy.tanh, ... ngay trên GPU. Lệch bit cuối
  nên kết cục một số lượt đổi và lịch sử tiến hóa rẽ nhánh: chỉ để thử, không dùng cho benchmark.

CuPy chỉ được import trong import_cupy() (khi chọn GPU), nên đường mặc định không cần CuPy.
Đường dẫn có dấu: NVRTC (trình biên dịch kernel của CuPy) không mở được header của CuPy khi
thư mục cài CuPy có ký tự ngoài ASCII (ví dụ repo nằm dưới "...\\môn tính toán tiến hóa\\...").
import_cupy() khi đó chèn tên ngắn 8.3 (GetShortPathNameW) của thư mục site-packages vào sys.path, ngay trước
mục site-packages gốc (thư mục của repo vẫn đứng trước), rồi mới import, nên CuPy được nạp qua đường dẫn ASCII.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
import sys

import numpy as np


class GpuUnavailable(RuntimeError):
    """Không dùng được GPU: thiếu CuPy, driver, NVRTC hoặc biên dịch kernel lỗi."""


# ── CuPy: import trễ, sửa đường dẫn có dấu ───────────────────────────────────
def _short_path(path: str) -> str | None:
    """Tên ngắn 8.3 của một thư mục (Windows); None nếu không lấy được."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    fn = k32.GetShortPathNameW
    fn.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD)
    fn.restype = wintypes.DWORD
    n = fn(path, None, 0)
    if not n:
        return None
    buf = ctypes.create_unicode_buffer(n)
    if not fn(path, buf, n):
        return None
    return buf.value


def prefer_ascii_cupy_path() -> str | None:
    """Nếu CuPy (chưa import) nằm dưới đường dẫn có ký tự ngoài ASCII, chèn tên ngắn 8.3 của thư mục
    site-packages chứa nó vào sys.path ngay trước mục site-packages gốc (không có mục đó thì chèn ở đầu), để
    gói cấp cao của repo vẫn thắng gói trùng tên trong site-packages; trả thư mục đã chèn, hoặc None nếu không
    cần / không làm được."""
    if "cupy" in sys.modules:
        return None
    try:
        spec = importlib.util.find_spec("cupy")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin or spec.origin.isascii():
        return None
    site = os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))
    short = _short_path(site)
    if not short or not short.isascii():
        return None
    if short not in sys.path:
        key = os.path.normcase(os.path.abspath(site))
        pos = next((i for i, p in enumerate(sys.path)
                    if p and os.path.normcase(os.path.abspath(p)) == key), 0)
        sys.path.insert(pos, short)
        importlib.invalidate_caches()
    return short


def import_cupy():
    """Import CuPy (sửa đường dẫn có dấu trước), báo GpuUnavailable kèm lý do nếu không được."""
    if "cupy" not in sys.modules:
        prefer_ascii_cupy_path()
    try:
        import cupy
    except Exception as e:  # noqa: BLE001  (ImportError, lỗi nạp DLL CUDA, ...)
        raise GpuUnavailable(f"không import được CuPy: {e!r}") from e
    origin = getattr(cupy, "__file__", "") or ""
    if not origin.isascii():
        raise GpuUnavailable(
            f"CuPy nằm dưới đường dẫn có ký tự ngoài ASCII ({origin}): NVRTC sẽ không mở được header. "
            "Chạy python của .venv bằng đường dẫn ngắn 8.3 (xem tên ngắn bằng lệnh dir /x)")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            raise GpuUnavailable("không thấy GPU CUDA nào")
    except GpuUnavailable:
        raise
    except Exception as e:  # noqa: BLE001  (CUDARuntimeError: thiếu driver, ...)
        raise GpuUnavailable(f"CUDA không chạy được: {e!r}") from e
    return cupy


# ── Backend ────────────────────────────────────────────────────────────────
def tanh_math_host(z: np.ndarray) -> np.ndarray:
    """math.tanh từng phần tử (hàm tanh_activation của neat-python dùng math.tanh)."""
    flat = z.ravel().tolist()
    return np.fromiter(map(math.tanh, flat), dtype=np.float64, count=len(flat)).reshape(z.shape)


class Backend:
    """Mảng trên xp = numpy ("numpy") hoặc cupy ("cupy"), hàm siêu việt theo chế độ exact / device."""

    def __init__(self, xp_name: str = "numpy", exact: bool = True):
        if xp_name not in ("numpy", "cupy"):
            raise ValueError(f"xp_name={xp_name!r}: chỉ nhận 'numpy' hoặc 'cupy'")
        if xp_name == "numpy" and not exact:
            raise ValueError("NumPy luôn chạy chế độ exact (chế độ device chỉ có trên GPU)")
        self.exact = bool(exact)
        if xp_name == "numpy":
            self.xp, self.gpu = np, False
        else:
            self.xp, self.gpu = import_cupy(), True

    # chép giữa host và thiết bị (NumPy: không làm gì)
    def host(self, a):
        return a.get() if self.gpu else a

    def dev(self, a):
        return self.xp.asarray(a) if self.gpu else a

    def any(self, a) -> bool:
        return bool(self.xp.any(a))

    # hàm siêu việt: exact -> hàm NumPy trên host (đúng hàm của đường từng lượt)
    def _host1(self, fn, a):
        if self.gpu and self.exact:
            return self.dev(fn(self.host(a)))
        return getattr(self.xp, fn.__name__)(a)

    def cos(self, a):
        return self._host1(np.cos, a)

    def sin(self, a):
        return self._host1(np.sin, a)

    def arctan2(self, y, x):
        if self.gpu and self.exact:
            return self.dev(np.arctan2(self.host(y), self.host(x)))
        return self.xp.arctan2(y, x)

    def tanh_neat(self, z):
        """tanh_activation của neat-python sau bước kẹp: math.tanh từng phần tử (exact)."""
        if not self.exact:
            return self.xp.tanh(z)
        return self.dev(tanh_math_host(self.host(z)))

    def tanh_np(self, z):
        """np.tanh như FixedMLP.activate."""
        return self._host1(np.tanh, z)
