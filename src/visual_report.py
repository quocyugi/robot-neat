"""
Visual Report — ảnh bảng, ảnh biểu đồ và trang HTML cho các kết quả trình bày
==============================================================================
Mỗi kết quả mà nhóm đưa ra khi demo (chấm trên tập test, bảng kết quả của
scripts/demo.py, log huấn luyện, bảng benchmark) ghi thêm, cạnh file JSON/MD/text
sẵn có (các file cũ giữ nguyên):

  <tên>_bang.png      ảnh bảng số liệu, rộng 1600 px, mở được ngay trong VS Code
  <tên>_bieu_do.png   ảnh biểu đồ, rộng 1600 px
  <tên>.html          một trang HTML tự chứa (CSS trong file, ảnh nhúng base64), mở bằng trình duyệt

Quy ước trình bày: chữ tiếng Việt có dấu, dấu phẩy thập phân (56,5%), dấu chấm
phân cách hàng nghìn (6.000); NEAT trước, baseline ghi nhãn (GA B1, GA B2); màu kết quả giống cửa sổ demo:
TỚI ĐÍCH xanh lá, VA CHẠM đỏ, HẾT GIỜ cam (thứ tự này giữ cặp xanh lá–đỏ đứng
cạnh nhau, dễ phân biệt hơn với người mù màu so với cặp xanh lá–cam). Màu luôn đi
kèm chữ (chú giải, nhãn trên thanh, cột số trong bảng), không dùng màu đơn độc.

Chỉ dùng matplotlib (đã có trong requirements.txt), vẽ bằng Figure + FigureCanvasAgg,
không qua pyplot. matplotlib chỉ được import bên trong hàm, nên các tiến trình con
của multiprocessing (spawn) không phải nạp nó.

Hàm chính
---------
    eval_report(results, policy, base)       scripts/evaluate.py (cấu hình 4)
    demo_report(rows, name, desc, base, ...) scripts/demo.py (cấu hình 2, 3, 7, 8, 9)
    training_report(run_dir)                 scripts/train_neat.py (cấu hình 5)
    heat_table_png(...), html_page(...)      scripts/plot_benchmark.py (cấu hình 6, docs/benchmark/)
    print_written(paths)                     in "Đã ghi ảnh: ..." / "Đã ghi trang HTML: ..."
"""

from __future__ import annotations

import base64
import datetime as _dt
import html as _html
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

# ── Kích thước ───────────────────────────────────────────────────────────
DPI = 160
WIDTH = 1600                 # px, mọi ảnh cùng bề rộng
MARGIN = 56                  # px, lề trái và phải
AVAIL = WIDTH - 2 * MARGIN   # bề rộng dùng được
TOP, BOTTOM = 44, 40         # px, lề trên và dưới
S = 1.5                      # px ảnh cho 1 px CSS (ảnh 1600 px thường hiện ở ~1070 px CSS)

TITLE_PT, SUB_PT = 19, 12
BODY_PT, HEAD_PT, SMALL_PT = 12.5, 11.5, 10.5
NOTE_PT, TICK_PT = 10.5, 11

# ── Màu (nền sáng) ───────────────────────────────────────────────────────
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BAND = "#f3f2ee"             # nền dòng nhóm, dải cấp curriculum
WHITE = "#ffffff"

# Kết quả một lượt: chữ và màu giống ô trạng thái của cửa sổ demo (scripts/demo.py)
OUTCOMES = {
    "goal": ("TỚI ĐÍCH", "#46c864"),
    "collision": ("VA CHẠM", "#e64646"),
    "timeout": ("HẾT GIỜ", "#f59b2d"),
}
OUTCOME_ORDER = ("goal", "collision", "timeout")
ON_OUTCOME = "#121216"       # chữ trên nền màu kết quả (như chữ trong ô trạng thái của demo)

# 8 màu phân loại theo thứ tự cố định (như scripts/plot_benchmark.py)
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
# Thang xanh một màu, nhạt -> đậm (bảng nhiệt: giá trị càng lớn ô càng đậm)
BLUE_RAMP = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b")

MEAN_SET = "mean_levels_1_4"
SET_ORDER = ("test/level_1", "test/level_2", "test/level_3", "test/level_4", "test_barn", "test_barn2", "test_lehman")


# ════════════════════════════════════════════════════════════════════════
#  Định dạng số kiểu Việt Nam
# ════════════════════════════════════════════════════════════════════════
def vn_num(x: float, nd: int = 1) -> str:
    """1234.5 -> '1.234,5' (dấu chấm hàng nghìn, dấu phẩy thập phân)."""
    s = f"{x:,.{nd}f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def vn_int(n: float) -> str:
    return vn_num(round(n), 0)


def vn_pct(rate: Optional[float], sign: bool = True, fixed: bool = False) -> str:
    """Tỉ lệ 0–1 -> phần trăm: số nguyên thì không có phần thập phân (0.88 -> '88%'),
    còn lại một chữ số (0.565 -> '56,5%'); fixed=True luôn một chữ số (cột số trong bảng:
    '88,0'). Không bao giờ ghi '100' cho giá trị chưa đủ 100."""
    if rate is None or (isinstance(rate, float) and not math.isfinite(rate)):
        return "—"
    v = 100.0 * float(rate)
    r1 = round(v, 1)
    if r1 >= 100 and v < 100 - 1e-9:
        r1 = 99.9
    if fixed:
        txt = vn_num(r1, 1)
    else:
        txt = str(int(round(r1))) if abs(r1 - round(r1)) < 1e-9 else vn_num(r1, 1)
    return txt + ("%" if sign else "")


def vn_tick(v: float) -> str:
    """Nhãn trên trục: số nguyên giữ nguyên, số lẻ một hoặc hai chữ số, dấu phẩy thập phân."""
    if abs(v - round(v)) < 1e-9:
        return vn_int(v)
    return vn_num(v, 1) if abs(v * 10 - round(v * 10)) < 1e-6 else vn_num(v, 2)


def set_label(name: str) -> str:
    """'test/level_3' -> 'Cấp 3', 'test_barn' -> 'BARN', 'mean_levels_1_4' -> 'TB cấp 1–4'."""
    m = re.fullmatch(r"(train|val|test)/level_(\d+)", name)
    if m:
        return f"Cấp {m.group(2)}" if m.group(1) == "test" else f"Cấp {m.group(2)} ({m.group(1)})"
    return {"test_barn": "BARN", "test_barn2": "BARN (tập 2)", "test_lehman": "Lehman",
            MEAN_SET: "TB cấp 1–4"}.get(name, name)


def display_path(p: Path | str) -> str:
    """Đường dẫn để in ra terminal: tương đối theo thư mục hiện tại nếu được."""
    p = Path(p)
    if not p.is_absolute():
        return p.as_posix()
    try:
        return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except (OSError, ValueError):
        return str(p)


_HOME_RE = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s]+", re.IGNORECASE)


def scrub(text: str) -> str:
    """Thay thư mục người dùng (C:\\Users\\<tên>) trong chữ ghi lên ảnh/HTML bằng '~'."""
    return _HOME_RE.sub("~", str(text))


def short_path(p: Path | str) -> str:
    """Đường dẫn ghi lên ảnh và trang HTML: tương đối nếu nằm trong thư mục hiện tại, không thì chỉ
    giữ 2 cấp cuối ('.../bench_out/M6'), để ảnh gọn và không mang tên người dùng (C:\\Users\\<tên>)."""
    q = Path(p)
    s = display_path(q)
    if q.is_absolute() and s == str(q):          # nằm ngoài thư mục hiện tại
        tail = [part for part in q.parts if part != q.anchor][-2:]
        return ".../" + "/".join(tail)
    return s


def outputs_for(base: Path | str) -> tuple[Path, Path, Path]:
    """base 'output/demo/test_results' -> (…_bang.png, …_bieu_do.png, ….html)."""
    base = Path(base)
    return (base.with_name(base.name + "_bang.png"), base.with_name(base.name + "_bieu_do.png"),
            base.with_name(base.name + ".html"))


def print_written(paths: Sequence[Path], stream=None) -> None:
    """In đường dẫn các file vừa ghi (Ctrl+bấm trong terminal của VS Code để mở)."""
    out = stream or sys.stdout
    sys.stdout.flush()          # in sau mọi dòng đã in trước đó, kể cả khi stdout đi qua pipe
    for p in paths:
        kind = "trang HTML" if Path(p).suffix.lower() == ".html" else "ảnh"
        print(f"Đã ghi {kind}: {display_path(p)}", file=out, flush=True)


def warn_failed(what: str, err: BaseException, stream=None) -> None:
    print(f"Cảnh báo: không ghi được {what} ({type(err).__name__}: {err}). "
          "Các file kết quả khác vẫn ghi bình thường.", file=stream or sys.stderr, flush=True)


# ════════════════════════════════════════════════════════════════════════
#  Phông chữ, đo chữ, khung vẽ theo pixel
# ════════════════════════════════════════════════════════════════════════
_FONT: dict = {}


def _family() -> dict:
    if not _FONT:
        from matplotlib import font_manager
        installed = {f.name for f in font_manager.fontManager.ttflist}
        fam = [f for f in ("Segoe UI", "Noto Sans", "DejaVu Sans") if f in installed] or ["DejaVu Sans"]
        _FONT["family"] = fam
        _FONT["semibold"] = "Segoe UI Semibold" if "Segoe UI Semibold" in installed else None
        _FONT["props"] = {}
    return _FONT


def _fp(size: float, weight: str = "regular"):
    """FontProperties theo cỡ (pt) và độ đậm: regular, semibold, bold."""
    from matplotlib.font_manager import FontProperties
    f = _family()
    key = (size, weight)
    if key not in f["props"]:
        if weight == "semibold" and f["semibold"]:
            f["props"][key] = FontProperties(family=[f["semibold"]] + f["family"], size=size)
        else:
            f["props"][key] = FontProperties(family=f["family"], size=size,
                                             weight="bold" if weight in ("bold", "semibold") else "normal")
    return f["props"][key]


def px(pt: float) -> float:
    return pt * DPI / 72.0


def _lw(px_width: float) -> float:
    """Độ dày nét theo px ảnh -> point (đơn vị linewidth của matplotlib)."""
    return px_width * 72.0 / DPI


class _Measure:
    """Đo bề rộng chữ (px) bằng chính bộ vẽ Agg, cùng DPI với ảnh."""

    def __init__(self):
        from matplotlib.backends.backend_agg import RendererAgg
        self.r = RendererAgg(8, 8, DPI)
        self.cache: dict = {}

    def w(self, s: str, size: float, weight: str = "regular") -> float:
        if not s:
            return 0.0
        key = (s, size, weight)
        if key not in self.cache:
            self.cache[key] = self.r.get_text_width_height_descent(s, _fp(size, weight), ismath=False)[0]
        return self.cache[key]

    def cap(self, size: float, weight: str = "regular") -> float:
        """Chiều cao chữ hoa (px): dùng để canh dòng chữ vào giữa theo đường chân chữ."""
        key = ("\x00cap", size, weight)
        if key not in self.cache:
            _, h, d = self.r.get_text_width_height_descent("H", _fp(size, weight), ismath=False)
            self.cache[key] = h - d
        return self.cache[key]

    def wrap(self, text: str, size: float, max_w: float, weight: str = "regular") -> list[str]:
        out: list[str] = []
        for para in str(text).split("\n"):
            cur = ""
            for word in para.split(" "):
                # một "từ" dài hơn cả dòng (đường dẫn dài): cắt theo ký tự
                while self.w(word, size, weight) > max_w and len(word) > 1:
                    if cur:
                        out.append(cur)
                        cur = ""
                    k = len(word)
                    while k > 1 and self.w(word[:k], size, weight) > max_w:
                        k -= 1
                    out.append(word[:k])
                    word = word[k:]
                trial = word if not cur else cur + " " + word
                if cur and self.w(trial, size, weight) > max_w:
                    out.append(cur)
                    cur = word
                else:
                    cur = trial
            out.append(cur)
        return out

    def shrink(self, text: str, size: float, max_w: float, weight: str = "regular", min_size: float = 12.0) -> float:
        """Cỡ chữ lớn nhất (không quá size, không dưới min_size) để text vừa max_w."""
        while size > min_size and self.w(text, size, weight) > max_w:
            size -= 0.5
        return size

    def fit(self, text: str, size: float, max_w: float, weight: str = "regular") -> str:
        """Cắt chữ (thêm …) cho vừa max_w."""
        if self.w(text, size, weight) <= max_w:
            return text
        while text and self.w(text + "…", size, weight) > max_w:
            text = text[:-1]
        return text + "…"


_M: Optional[_Measure] = None


def _m() -> _Measure:
    global _M
    if _M is None:
        _M = _Measure()
    return _M


def _rgb(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _luminance(hex_color: str) -> float:
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(c) for c in _rgb(hex_color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a: str, b: str) -> float:
    la, lb = _luminance(a), _luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def text_on(fill: str) -> str:
    """Màu chữ đặt trên nền `fill`: trắng hoặc gần đen, chọn bên có độ tương phản cao hơn."""
    return WHITE if _contrast(WHITE, fill) > _contrast(INK, fill) else INK


def _darker(hex_color: str, k: float = 0.38) -> tuple:
    r, g, b = _rgb(hex_color)
    return (r * (1 - k), g * (1 - k), b * (1 - k))


def ramp_color(v: float) -> str:
    """Giá trị 0–1 -> một bậc của thang xanh (0 nhạt nhất, 1 đậm nhất)."""
    v = min(1.0, max(0.0, float(v)))
    return BLUE_RAMP[int(round(v * (len(BLUE_RAMP) - 1)))]


def _round_rect_path(x, y, w, h, rtl=0.0, rtr=0.0, rbr=0.0, rbl=0.0):
    """Hình chữ nhật (y hướng xuống) bo từng góc riêng: tl, tr, br, bl."""
    from matplotlib.path import Path as MPath
    lim = max(0.0, min(w, h) / 2)
    rtl, rtr, rbr, rbl = (min(max(r, 0.0), lim) for r in (rtl, rtr, rbr, rbl))
    k = 0.5523
    x0, x1, y0, y1 = x, x + w, y, y + h
    v: list = [(x0 + rtl, y0)]
    c: list = [MPath.MOVETO]

    def line(p):
        v.append(p)
        c.append(MPath.LINETO)

    def curve(c1, c2, p):
        v.extend([c1, c2, p])
        c.extend([MPath.CURVE4] * 3)

    line((x1 - rtr, y0))
    if rtr:
        curve((x1 - rtr + k * rtr, y0), (x1, y0 + rtr - k * rtr), (x1, y0 + rtr))
    line((x1, y1 - rbr))
    if rbr:
        curve((x1, y1 - rbr + k * rbr), (x1 - rbr + k * rbr, y1), (x1 - rbr, y1))
    line((x0 + rbl, y1))
    if rbl:
        curve((x0 + rbl - k * rbl, y1), (x0, y1 - rbl + k * rbl), (x0, y1 - rbl))
    line((x0, y0 + rtl))
    if rtl:
        curve((x0, y0 + rtl - k * rtl), (x0 + rtl - k * rtl, y0), (x0 + rtl, y0))
    v.append((x0 + rtl, y0))
    c.append(MPath.CLOSEPOLY)
    return MPath(v, c)


class _Canvas:
    """Một ảnh vẽ theo pixel: gốc ở góc trên trái, trục y hướng xuống."""

    def __init__(self, height: float, width: int = WIDTH):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        self.w, self.h = int(width), int(math.ceil(height))
        self.fig = Figure(figsize=(self.w / DPI, self.h / DPI), dpi=DPI, facecolor=SURFACE)
        FigureCanvasAgg(self.fig)
        ax = self.fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, self.w)
        ax.set_ylim(self.h, 0)
        ax.set_axis_off()
        self.ax = ax

    def text(self, x, y, s, size=BODY_PT, weight="regular", color=INK, ha="left", va="baseline", z=5):
        self.ax.text(x, y, s, fontproperties=_fp(size, weight), color=color, ha=ha, va=va, zorder=z,
                     parse_math=False)

    def rect(self, x, y, w, h, color, r=0.0, z=1, hatch=None, corners=None):
        """corners = (tl, tr, br, bl) nếu chỉ bo một số góc; r = bo đều 4 góc."""
        from matplotlib.patches import PathPatch
        if w <= 0 or h <= 0:
            return
        radii = corners if corners is not None else (r, r, r, r)
        path = _round_rect_path(x, y, w, h, *radii)
        self.ax.add_patch(PathPatch(path, facecolor=color, linewidth=0, zorder=z, hatch=hatch,
                                    edgecolor=_darker(color) if hatch else "none"))

    def line(self, x0, y0, x1, y1, color=GRID, width=1.5, z=2):
        self.ax.plot([x0, x1], [y0, y1], color=color, linewidth=_lw(width), solid_capstyle="butt", zorder=z)

    def axes(self, x, y, w, h):
        """Một Axes thật của matplotlib đặt đúng khung pixel (x, y, w, h)."""
        a = self.fig.add_axes((x / self.w, 1 - (y + h) / self.h, w / self.w, h / self.h))
        a.set_facecolor("none")
        return a

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=DPI, facecolor=SURFACE)
        return path


def _rc():
    """rcParams cho phần do matplotlib tự vẽ (nhãn trục)."""
    import matplotlib
    f = _family()
    return matplotlib.rc_context({"font.family": f["family"], "font.size": TICK_PT,
                                  "axes.unicode_minus": False, "hatch.linewidth": 0.9,
                                  "svg.fonttype": "none"})


class _Part:
    """Một khối của ảnh (tiêu đề, bảng, biểu đồ, ghi chú): biết chiều cao và tự vẽ."""

    gap_after = 30.0

    def height(self) -> float:
        raise NotImplementedError

    def draw(self, cv: _Canvas, y: float) -> None:
        raise NotImplementedError


def _compose(path: Path, parts: Sequence[_Part]) -> Path:
    parts = [p for p in parts if p is not None]
    total = TOP + BOTTOM + sum(p.height() for p in parts) + sum(p.gap_after for p in parts[:-1])
    with _rc():
        cv = _Canvas(total)
        y = TOP
        for p in parts:
            p.draw(cv, y)
            y += p.height() + p.gap_after
        return cv.save(path)


# ════════════════════════════════════════════════════════════════════════
#  Khối tiêu đề: tiêu đề, phụ đề, chú giải, thang màu, ô số liệu
# ════════════════════════════════════════════════════════════════════════
@dataclass
class Tile:
    label: str
    value: str
    note: str = ""


class Header(_Part):
    def __init__(self, title: str, subtitle: str = "", legend: Sequence[tuple[str, str]] = (),
                 ramp: Optional[tuple[str, str]] = None, tiles: Sequence[Tile] = (), width: float = AVAIL):
        m = _m()
        self.width = width
        self.title = m.wrap(scrub(title), TITLE_PT, width, "semibold")
        self.sub = m.wrap(scrub(subtitle), SUB_PT, width) if subtitle else []
        self.legend = list(legend)
        self.ramp = ramp
        self.tiles = list(tiles)
        self.lh_title = px(TITLE_PT) * 1.3
        self.lh_sub = px(SUB_PT) * 1.45
        self.legend_rows = self._legend_rows() if self.legend else []
        self.tile_h = px(SMALL_PT) * 1.5 + px(21) * 1.25 + 30 if self.tiles else 0
        self.gap_after = 34.0

    def _legend_rows(self) -> list[list[tuple[str, str, float]]]:
        m, sw, gap_txt, gap_item = _m(), 22.0, 12.0, 40.0
        rows, cur, cur_w = [], [], 0.0
        for label, color in self.legend:
            w = sw + gap_txt + m.w(label, SUB_PT)
            if cur and cur_w + gap_item + w > self.width:
                rows.append(cur)
                cur, cur_w = [], 0.0
            cur.append((label, color, w))
            cur_w += (gap_item if len(cur) > 1 else 0) + w
        if cur:
            rows.append(cur)
        return rows

    def height(self) -> float:
        h = len(self.title) * self.lh_title
        if self.sub:
            h += 6 + len(self.sub) * self.lh_sub
        if self.legend_rows:
            h += 16 + len(self.legend_rows) * px(SUB_PT) * 1.7
        if self.ramp:
            h += 14 + px(SUB_PT) * 1.7
        if self.tiles:
            h += 22 + self.tile_h
        return h

    def draw(self, cv: _Canvas, y: float) -> None:
        m, x0 = _m(), MARGIN
        for line in self.title:
            cv.text(x0, y + px(TITLE_PT) * 1.02, line, TITLE_PT, "semibold", INK)
            y += self.lh_title
        if self.sub:
            y += 6
            for line in self.sub:
                cv.text(x0, y + px(SUB_PT) * 1.08, line, SUB_PT, "regular", INK_2)
                y += self.lh_sub
        if self.legend_rows:
            y += 16
            row_h = px(SUB_PT) * 1.7
            for row in self.legend_rows:
                x = x0
                base = y + row_h / 2 + m.cap(SUB_PT) / 2
                for label, color, w in row:
                    cv.rect(x, y + row_h / 2 - 11, 22, 22, color, r=5, z=3)
                    cv.text(x + 22 + 12, base, label, SUB_PT, "regular", INK)
                    x += w + 40
                y += row_h
        if self.ramp:
            y += 14
            row_h = px(SUB_PT) * 1.7
            lo, hi = self.ramp
            base = y + row_h / 2 + m.cap(SUB_PT) / 2
            cv.text(x0, base, "Thang màu ô:", SUB_PT, "regular", INK_2)
            x = x0 + m.w("Thang màu ô:", SUB_PT) + 14
            cv.text(x, base, lo, SUB_PT, "regular", INK_2)
            x += m.w(lo, SUB_PT) + 10
            for c in BLUE_RAMP:
                cv.rect(x, y + row_h / 2 - 12, 26, 24, c, r=0, z=3)
                x += 28
            cv.text(x + 8, base, hi, SUB_PT, "regular", INK_2)
            y += row_h
        if self.tiles:
            y += 22
            n = len(self.tiles)
            gap = 18.0
            tw = min((self.width - gap * (n - 1)) / n, 380.0)
            for i, t in enumerate(self.tiles):
                tx = x0 + i * (tw + gap)
                inner = tw - 40
                cv.rect(tx, y, tw, self.tile_h, "#f6f6f3", r=10, z=1)
                ls = m.shrink(t.label, SMALL_PT, inner, min_size=9.0)
                cv.text(tx + 20, y + 16 + px(SMALL_PT) * 1.05, m.fit(t.label, ls, inner), ls, "regular", INK_2)
                vy = y + 16 + px(SMALL_PT) * 1.5 + px(21) * 1.02
                note_w = (m.w(t.note, SMALL_PT) + 10) if t.note else 0.0
                vs = m.shrink(t.value, 21, inner - note_w, "semibold", min_size=13.0)
                val = m.fit(t.value, vs, inner - note_w, "semibold")
                cv.text(tx + 20, vy, val, vs, "semibold", INK)
                if t.note:
                    cv.text(tx + 20 + m.w(val, vs, "semibold") + 10, vy, t.note, SMALL_PT, "regular", MUTED)


class Notes(_Part):
    def __init__(self, lines: Sequence[str], width: float = AVAIL, size: float = NOTE_PT):
        m = _m()
        self.size = size
        self.lines: list[str] = []
        self.starts: list[bool] = []
        for ln in lines:
            if not ln:
                continue
            wrapped = m.wrap(scrub(ln), size, width)
            self.lines += wrapped
            self.starts += [True] + [False] * (len(wrapped) - 1)
        self.lh = px(size) * 1.5
        self.gap_after = 20.0

    def height(self) -> float:
        return len(self.lines) * self.lh + 4 * sum(self.starts)

    def draw(self, cv: _Canvas, y: float) -> None:
        for line, start in zip(self.lines, self.starts):
            if start:
                y += 4
            cv.text(MARGIN, y + px(self.size) * 1.08, line, self.size, "regular", INK_2)
            y += self.lh


# ════════════════════════════════════════════════════════════════════════
#  Bảng
# ════════════════════════════════════════════════════════════════════════
@dataclass
class Col:
    title: str
    align: str = "left"            # left | right | center
    sub: str = ""                  # dòng thứ hai của tiêu đề cột (đơn vị, số bản đồ)
    min_w: float = 0.0             # bề rộng tối thiểu (px)
    grow: bool = False             # cột được nhận phần bề rộng còn thừa (thanh tỉ lệ)
    max_w: float = 0.0             # 0 = không giới hạn khi nhận thêm


@dataclass
class Cell:
    text: str = ""
    sub: str = ""
    weight: str = "regular"
    color: str = INK
    fill: Optional[str] = None         # nền cả ô (bảng nhiệt)
    badge: Optional[str] = None        # nhãn bo tròn có nền màu (kết quả)
    bar: Optional[list] = None         # [(tỉ lệ, màu), ...]: thanh chia phần mảnh, không chữ
    swatch: Optional[str] = None       # ô màu nhỏ trước chữ (màu của phương pháp trên biểu đồ)
    swatch_hatch: Optional[str] = None


@dataclass
class Row:
    cells: list = field(default_factory=list)
    group: str = ""                    # dòng tiêu đề nhóm, kéo hết bề rộng bảng
    rule_above: bool = False           # kẻ đậm phía trên (dòng tổng, trung bình)
    bold: bool = False


class Table(_Part):
    PAD = 18.0

    def __init__(self, columns: Sequence[Col], rows: Sequence[Row], heading: str = "", width: float = AVAIL,
                 uniform: bool = False):
        self.cols = list(columns)
        self.rows = list(rows)
        self.heading = heading
        self.width = width
        self.uniform = uniform      # mọi dòng số liệu cùng chiều cao (bảng nhiệt)
        self.gap_after = 34.0
        self.body = BODY_PT
        self._layout()

    # cỡ chữ theo body
    def _sizes(self):
        k = self.body / BODY_PT
        return {"body": self.body, "head": HEAD_PT * k, "small": SMALL_PT * k, "badge": 10.5 * k}

    def _natural(self) -> list[float]:
        m, sz, pad = _m(), self._sizes(), self.PAD
        widths = []
        for j, col in enumerate(self.cols):
            w = max(m.w(col.title, sz["head"], "semibold"), m.w(col.sub, sz["small"]), col.min_w)
            for row in self.rows:
                if row.group or j >= len(row.cells):
                    continue
                c = row.cells[j]
                if c is None:
                    continue
                weight = "semibold" if row.bold else c.weight
                if c.badge:
                    cw = m.w(c.text, sz["badge"], "semibold") + 2 * 16
                elif c.bar is not None:
                    cw = max(col.min_w, 160.0)
                else:
                    cw = max(m.w(c.text, sz["body"], weight), m.w(c.sub, sz["small"]))
                    if c.swatch:
                        cw += 20 + 12
                w = max(w, cw)
            widths.append(w + 2 * pad)
        return widths

    def _layout(self) -> None:
        widths = self._natural()
        while sum(widths) > self.width and self.body > 9.0:
            self.body -= 0.5
            widths = self._natural()
        extra = self.width - sum(widths)
        if extra > 0:
            grow = [j for j, c in enumerate(self.cols) if c.grow]
            for j in grow:
                add = extra / len(grow)
                if self.cols[j].max_w:
                    add = max(0.0, min(add, self.cols[j].max_w - widths[j]))
                widths[j] += add
            extra = self.width - sum(widths)
            # phần còn thừa chia đều cho mọi cột, nhưng mỗi cột rộng thêm không quá 35 %
            if extra > 0:
                room = [w * 0.35 for w in widths]
                total_room = sum(room)
                for j in range(len(widths)):
                    widths[j] += min(room[j], extra * room[j] / total_room) if total_room else 0
        self.widths = widths
        self.table_w = sum(widths)
        sz = self._sizes()
        self.h_one = px(sz["body"]) * 2.1
        self.h_two = px(sz["body"]) * 1.35 + px(sz["small"]) * 1.35 + 22
        self.h_group = px(sz["head"]) * 2.0
        self.h_head = (px(sz["head"]) * 1.3 + px(sz["small"]) * 1.3 + 24) if any(c.sub for c in self.cols) \
            else px(sz["head"]) * 2.2
        self.h_heading = px(14) * 1.9 if self.heading else 0.0

    def _row_h(self, row: Row) -> float:
        if row.group:
            return self.h_group
        if self.uniform and any(c is not None and c.sub for r in self.rows for c in r.cells):
            return self.h_two
        if any(c is not None and c.sub for c in row.cells):
            return self.h_two
        return self.h_one

    def height(self) -> float:
        return self.h_heading + self.h_head + sum(self._row_h(r) for r in self.rows)

    def draw(self, cv: _Canvas, y: float) -> None:
        m, sz, pad = _m(), self._sizes(), self.PAD
        x0 = MARGIN
        if self.heading:
            cv.text(x0, y + px(14) * 1.1, self.heading, 14, "semibold", INK)
            y += self.h_heading
        # tiêu đề cột
        xs = [x0 + sum(self.widths[:j]) for j in range(len(self.cols))]
        two = any(c.sub for c in self.cols)
        for j, col in enumerate(self.cols):
            x, w = xs[j], self.widths[j]
            tx, ha = self._anchor(x, w, col.align)
            if two:
                b1 = y + 12 + px(sz["head"]) * 1.0
                cv.text(tx, b1, col.title, sz["head"], "semibold", INK_2, ha=ha)
                if col.sub:
                    cv.text(tx, b1 + px(sz["small"]) * 1.3, col.sub, sz["small"], "regular", MUTED, ha=ha)
            else:
                cv.text(tx, y + self.h_head / 2 + m.cap(sz["head"], "semibold") / 2, col.title, sz["head"],
                        "semibold", INK_2, ha=ha)
        y += self.h_head
        cv.line(x0, y, x0 + self.table_w, y, BASELINE, 2.0, z=3)
        # các dòng
        for row in self.rows:
            h = self._row_h(row)
            if row.group:
                cv.rect(x0, y + 1, self.table_w, h - 2, BAND, r=0, z=0.5)
                cv.text(x0 + pad, y + h / 2 + m.cap(sz["head"], "semibold") / 2, row.group, sz["head"],
                        "semibold", INK_2)
                y += h
                continue
            if row.rule_above:
                cv.line(x0, y, x0 + self.table_w, y, BASELINE, 2.0, z=3)
            for j, col in enumerate(self.cols):
                c = row.cells[j] if j < len(row.cells) else None
                if c is not None:
                    self._cell(cv, c, xs[j], y, self.widths[j], h, col, row, sz)
            y += h
            cv.line(x0, y, x0 + self.table_w, y, GRID, 1.5, z=0.8)

    @classmethod
    def _anchor(cls, x, w, align):
        if align == "right":
            return x + w - cls.PAD, "right"
        if align == "center":
            return x + w / 2, "center"
        return x + cls.PAD, "left"

    def _cell(self, cv, c: Cell, x, y, w, h, col: Col, row: Row, sz) -> None:
        m, pad = _m(), self.PAD
        weight = "semibold" if row.bold else c.weight
        color = c.color
        if c.fill:
            cv.rect(x + 1, y + 1, w - 2, h - 2, c.fill, r=4, z=0.6)
            color = text_on(c.fill)
        if c.badge:
            bw = m.w(c.text, sz["badge"], "semibold") + 2 * 16
            bh = px(sz["badge"]) * 1.75
            bx = {"right": x + w - pad - bw, "center": x + (w - bw) / 2}.get(col.align, x + pad)
            cv.rect(bx, y + (h - bh) / 2, bw, bh, c.badge, r=bh / 2, z=2)
            cv.text(bx + bw / 2, y + h / 2 + m.cap(sz["badge"], "semibold") / 2, c.text, sz["badge"], "semibold",
                    text_on(c.badge) if c.badge not in [v[1] for v in OUTCOMES.values()] else ON_OUTCOME,
                    ha="center")
            return
        if c.bar is not None:
            _thin_bar(cv, x + pad, y + h / 2, w - 2 * pad, c.bar)
            return
        tx, ha = self._anchor(x, w, col.align)
        if c.swatch:
            cv.rect(x + pad, y + h / 2 - 10, 20, 20, c.swatch, r=4, z=2, hatch=c.swatch_hatch)
            if ha == "left":
                tx += 20 + 12
        if c.sub:
            b1 = y + h * 0.40 + m.cap(sz["body"], weight) / 2
            cv.text(tx, b1, c.text, sz["body"], weight, color, ha=ha)
            sub_color = color if c.fill else MUTED
            cv.text(tx, y + h * 0.74 + m.cap(sz["small"]) / 2, c.sub, sz["small"], "regular", sub_color, ha=ha)
        else:
            cv.text(tx, y + h / 2 + m.cap(sz["body"], weight) / 2, c.text, sz["body"], weight, color, ha=ha)


def _thin_bar(cv: _Canvas, x: float, cy: float, w: float, parts: list, height: float = 14.0) -> None:
    """Thanh chia phần mảnh (một tổng thể chia thành các phần), không chữ, khe 2 px giữa các phần."""
    parts = [(float(v), c) for v, c in parts if v and v > 1e-9]
    total = sum(v for v, _ in parts)
    if total <= 0:
        cv.rect(x, cy - height / 2, w, height, GRID, r=height / 2, z=2)
        return
    gap = 2 * S
    avail = w - gap * (len(parts) - 1)
    xx = x
    for v, color in parts:
        seg = avail * v / total
        cv.rect(xx, cy - height / 2, seg, height, color, r=min(height / 2, seg / 2), z=2)
        xx += seg + gap


def table_png(path: Path, header: Header, tables: Sequence[Table], notes: Sequence[str] = ()) -> Path:
    parts: list[_Part] = [header] + list(tables)
    if notes:
        parts.append(Notes(notes))
    return _compose(path, parts)


# ════════════════════════════════════════════════════════════════════════
#  Biểu đồ thanh ngang (vẽ trực tiếp theo pixel)
# ════════════════════════════════════════════════════════════════════════
@dataclass
class HBar:
    label: str                         # nhãn bên trái (dòng 1)
    sub: str = ""                      # dòng 2 của nhãn (nhỏ, mờ)
    parts: list = field(default_factory=list)   # [(giá trị, màu, chữ trong đoạn hoặc "")]
    tip: str = ""                      # chữ sau đầu thanh


class HBarChart(_Part):
    """Thanh ngang, mỗi thanh một dòng; nhiều đoạn thì là thanh xếp chồng.

    Thanh mảnh (<= 24 px CSS), đầu thanh bo 4 px CSS, gốc vuông; khe 2 px giữa các đoạn;
    chữ trong đoạn chỉ ghi khi vừa, không thì để bảng mang số liệu.
    """

    def __init__(self, bars: Sequence[HBar], xmax: float, ticks: Sequence[float], xlabel: str,
                 tick_fmt: Callable[[float], str] = vn_tick, ref: Optional[tuple[float, str]] = None,
                 bar_h: float = 24 * S, pitch: float = 0.0, groups_gap: Sequence[int] = ()):
        m = _m()
        self.bars = list(bars)
        self.xmax = float(xmax)
        self.ticks = list(ticks)
        self.xlabel = xlabel
        self.fmt = tick_fmt
        self.ref = ref
        self.bar_h = bar_h
        self.pitch = pitch or max(bar_h + 30, px(BODY_PT) * 1.25 + px(SMALL_PT) * 1.3 + 16)
        self.groups_gap = set(groups_gap)   # chỉ số thanh có khoảng trống phía trên
        self.label_w = max([m.w(b.label, BODY_PT) for b in self.bars] +
                           [m.w(b.sub, SMALL_PT) for b in self.bars] + [60]) + 28
        self.tip_w = max([m.w(b.tip, TICK_PT) for b in self.bars] + [0]) + (16 if any(b.tip for b in self.bars) else 0)
        self.axis_h = px(TICK_PT) * 1.4 + 14 + px(SUB_PT) * 1.6
        self.ref_h = px(SMALL_PT) * 1.6 if ref else 0.0
        self.gap_after = 30.0

    def height(self) -> float:
        return self.ref_h + len(self.bars) * self.pitch + 18 * len(self.groups_gap) + self.axis_h

    def draw(self, cv: _Canvas, y: float) -> None:
        m = _m()
        x_plot = MARGIN + self.label_w
        plot_w = AVAIL - self.label_w - self.tip_w
        sx = plot_w / self.xmax
        top = y + self.ref_h
        body_h = len(self.bars) * self.pitch + 18 * len(self.groups_gap)
        bottom = top + body_h
        # lưới dọc mảnh và nhãn trục
        for t in self.ticks:
            xx = x_plot + t * sx
            cv.line(xx, top, xx, bottom, GRID if t else BASELINE, 1.5 if t else 2.0, z=0.5)
            cv.text(xx, bottom + 10 + px(TICK_PT), self.fmt(t), TICK_PT, "regular", INK_2, ha="center")
        cv.text(x_plot + plot_w / 2, bottom + 14 + px(TICK_PT) * 1.4 + px(SUB_PT) * 1.1, self.xlabel, SUB_PT,
                "regular", INK_2, ha="center")
        if self.ref:
            rv, rlabel = self.ref
            xx = x_plot + rv * sx
            cv.line(xx, top - 4, xx, bottom, INK_2, 2.0, z=4)
            cv.text(xx, y + px(SMALL_PT) * 1.05, rlabel, SMALL_PT, "regular", INK_2, ha="center")
        yy = top
        for i, b in enumerate(self.bars):
            if i in self.groups_gap:
                yy += 18
            cy = yy + self.pitch / 2
            if b.sub:
                cv.text(MARGIN, cy - 4, b.label, BODY_PT, "regular", INK)
                cv.text(MARGIN, cy - 4 + px(SMALL_PT) * 1.35, b.sub, SMALL_PT, "regular", MUTED)
            else:
                cv.text(MARGIN, cy + m.cap(BODY_PT) / 2, b.label, BODY_PT, "regular", INK)
            parts = [(float(v), c, t) for v, c, t in b.parts if v and v > 1e-12]
            xx = x_plot
            gap = 2 * S
            end = x_plot
            for k, (v, color, label) in enumerate(parts):
                wseg = v * sx - (gap if k < len(parts) - 1 else 0)
                last = k == len(parts) - 1
                r = 4 * S
                cv.rect(xx, cy - self.bar_h / 2, max(wseg, 1.0), self.bar_h, color, z=2,
                        corners=(0, r, r, 0) if last else (0, 0, 0, 0))
                if label and m.w(label, SMALL_PT, "semibold") + 2 * 10 <= wseg:
                    cv.text(xx + 10, cy + m.cap(SMALL_PT, "semibold") / 2, label, SMALL_PT, "semibold",
                            ON_OUTCOME if color in [o[1] for o in OUTCOMES.values()] else text_on(color), z=6)
                xx += v * sx
                end = xx
            if not parts:   # giá trị 0: một vạch 2 px để vẫn thấy dòng có số liệu
                cv.rect(x_plot, cy - self.bar_h / 2, 2 * S, self.bar_h, BASELINE, z=2)
            if b.tip:
                tw = m.w(b.tip, TICK_PT)
                base = cy + m.cap(TICK_PT) / 2
                x_ref = x_plot + self.ref[0] * sx if self.ref else None
                crosses = x_ref is not None and end < x_ref - 1 and end + 12 + tw > x_ref - 8
                if crosses and parts and tw + 24 <= end - x_plot:
                    # chữ ngoài đầu thanh sẽ cắt vạch giới hạn: ghi vào trong thanh, sát đầu thanh
                    fill = parts[-1][1]
                    ink = ON_OUTCOME if fill in [o[1] for o in OUTCOMES.values()] else text_on(fill)
                    cv.text(end - 12, base, b.tip, TICK_PT, "regular", ink, ha="right", z=6)
                else:
                    cv.text(end + 12, base, b.tip, TICK_PT, "regular", INK_2)
            yy += self.pitch


# ════════════════════════════════════════════════════════════════════════
#  Trang HTML tự chứa
# ════════════════════════════════════════════════════════════════════════
_CSS = """
:root { color-scheme: light; --page: #f9f9f7; --card: #ffffff; --ink: #0b0b0b; --ink2: #52514e;
        --muted: #898781; --line: #e1e0d9; --band: #f3f2ee; }
@media (prefers-color-scheme: dark) {
  :root { color-scheme: dark; --page: #0d0d0d; --card: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7;
          --muted: #898781; --line: #2c2c2a; --band: #232322; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
       font: 16px/1.55 "Segoe UI", system-ui, -apple-system, "Noto Sans", sans-serif; }
main { max-width: 1680px; margin: 0 auto; padding: 28px 16px 48px; }
h1 { font-size: 28px; line-height: 1.25; font-weight: 600; margin: 0 0 6px; }
p.sub { color: var(--ink2); margin: 0 0 22px; font-size: 17px; }
figure { margin: 0 0 26px; background: var(--card); border: 1px solid var(--line); border-radius: 12px;
         padding: 14px; }
figcaption { font-weight: 600; color: var(--ink2); margin: 0 4px 10px; }
figure img { display: block; width: 100%; height: auto; border-radius: 6px; background: #fff; }
section.notes { color: var(--ink2); font-size: 15px; }
section.notes p { margin: 4px 0; }
details { margin: 0 0 26px; background: var(--card); border: 1px solid var(--line); border-radius: 12px;
          padding: 12px 14px; }
summary { cursor: pointer; font-weight: 600; color: var(--ink2); }
.scroll { overflow-x: auto; margin-top: 12px; }
table.data { border-collapse: collapse; font-size: 15px; font-variant-numeric: tabular-nums; min-width: 100%; }
table.data th { text-align: left; color: var(--ink2); font-weight: 600; border-bottom: 2px solid var(--line);
                padding: 6px 12px; white-space: nowrap; }
table.data td { border-bottom: 1px solid var(--line); padding: 5px 12px; white-space: nowrap; }
table.data td.num, table.data th.num { text-align: right; }
table.data tr.strong td { font-weight: 600; }
ul.files { color: var(--ink2); font-size: 14px; padding-left: 20px; }
footer { color: var(--muted); font-size: 13px; margin-top: 28px; }
"""


def esc(s) -> str:
    return _html.escape(str(s), quote=True)


def html_table(headers: Sequence[str], rows: Sequence[Sequence], num_cols: Sequence[int] = (),
               strong_rows: Sequence[int] = ()) -> str:
    """Bảng HTML thường (chữ chọn và sao chép được), dùng cho bảng dài như log từng thế hệ."""
    num = set(num_cols)
    strong = set(strong_rows)
    head = "".join(f'<th class="num">{esc(h)}</th>' if j in num else f"<th>{esc(h)}</th>"
                   for j, h in enumerate(headers))
    body = []
    for i, r in enumerate(rows):
        cells = "".join(f'<td class="num">{esc(v)}</td>' if j in num else f"<td>{esc(v)}</td>"
                        for j, v in enumerate(r))
        body.append(f'<tr class="strong">{cells}</tr>' if i in strong else f"<tr>{cells}</tr>")
    return f'<div class="scroll"><table class="data"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def html_page(path: Path, title: str, subtitle: str, images: Sequence[tuple[str, Path]],
              notes: Sequence[str] = (), extra_html: str = "", source: str = "",
              files: Sequence[Path] = ()) -> Path:
    """Một trang HTML tự chứa: CSS trong file, các ảnh PNG nhúng base64 (không cần file nào khác)."""
    path = Path(path)
    title, subtitle, source = scrub(title), scrub(subtitle), scrub(source)
    notes = [scrub(n) for n in notes]
    figs = []
    for caption, png in images:
        data = base64.b64encode(Path(png).read_bytes()).decode("ascii")
        figs.append(f'<figure><figcaption>{esc(caption)}</figcaption>'
                    f'<img alt="{esc(caption)}" src="data:image/png;base64,{data}"></figure>')
    note_html = "".join(f"<p>{esc(n)}</p>" for n in notes if n)
    file_html = "".join(f"<li><code>{esc(short_path(f))}</code></li>" for f in files)
    now = _dt.datetime.now().strftime("%d/%m/%Y %H:%M")
    doc = (
        "<!doctype html>\n<html lang=\"vi\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{esc(title)}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n<main>\n"
        f"<h1>{esc(title)}</h1>\n" + (f"<p class=\"sub\">{esc(subtitle)}</p>\n" if subtitle else "")
        + "\n".join(figs) + "\n" + extra_html
        + (f"\n<section class=\"notes\">{note_html}</section>" if note_html else "")
        + (f"\n<section class=\"notes\"><p>Các file của kết quả này:</p><ul class=\"files\">{file_html}</ul></section>"
           if file_html else "")
        + f"\n<footer>Tạo lúc {esc(now)}" + (f" · {esc(source)}" if source else "") + "</footer>\n"
        "</main>\n</body>\n</html>\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path


# ════════════════════════════════════════════════════════════════════════
#  1. Chấm trên tập test (scripts/evaluate.py, cấu hình 4)
# ════════════════════════════════════════════════════════════════════════
METRIC_DEF = ("SPL (0–1): vừa tới đích vừa đi ngắn, bằng độ dài đường tối ưu chia độ dài đường đi thật, lượt "
              "không tới đích tính 0. MTT: thời gian trung bình của các lượt tới đích.")


def outcome_def(max_steps: Optional[int] = None, dt: Optional[float] = None) -> str:
    """Định nghĩa TỚI ĐÍCH / VA CHẠM / HẾT GIỜ. Số bước tối đa và dt không truyền vào thì lấy từ
    src.env.kinematics của tiến trình này (robot cũ: 500 bước, 50 giây; NEAT_ROBOT=waffle: 1.000 bước, 100 giây)."""
    if max_steps is None or dt is None:
        from src.env import kinematics as KM
        max_steps = KM.MAX_STEPS if max_steps is None else max_steps
        dt = KM.DT if dt is None else dt
    return ("TỚI ĐÍCH: tới vùng đích (SR); VA CHẠM: chạm tường hoặc vật cản (CR); HẾT GIỜ: dùng hết "
            f"{vn_int(max_steps)} bước ({vn_tick(max_steps * dt)} giây) mà chưa tới đích.")


def _ordered_sets(results: dict) -> list[str]:
    sets = [s for s in results if s != MEAN_SET and isinstance(results[s], dict) and results[s].get("n")]
    known = [s for s in SET_ORDER if s in sets]
    return known + [s for s in sets if s not in known]


def eval_report(results: dict, policy: str, base: Path | str, source: str = "",
                max_steps: Optional[int] = None, dt: Optional[float] = None) -> list[Path]:
    """Ảnh bảng, ảnh biểu đồ và trang HTML cho kết quả của scripts/evaluate.py.

    results: mục "results" của test_results.json (tập -> {n, SR, CR, timeout, SPL, MTT}).
    max_steps, dt: giới hạn bước và bước thời gian ghi trong định nghĩa HẾT GIỜ (mặc định: của tiến trình này).
    """
    odef = outcome_def(max_steps, dt)
    table_path, chart_path, html_path = outputs_for(base)
    sets = _ordered_sets(results)
    if not sets:
        raise ValueError("không có tập nào có số liệu")
    total = sum(int(results[s]["n"]) for s in sets)
    levels = [s for s in sets if s.startswith("test/level_")]
    mean = results.get(MEAN_SET) if isinstance(results.get(MEAN_SET), dict) else None
    mean_txt = f" · SR trung bình cấp 1–4: {vn_pct(mean.get('SR'))}" if mean and mean.get("SR") is not None else ""
    subtitle = f"{policy} · {vn_int(total)} bản đồ test (không dùng khi huấn luyện){mean_txt}"
    legend = [(f"{OUTCOMES[k][0]}", OUTCOMES[k][1]) for k in OUTCOME_ORDER]

    def _tc(r: dict):
        """TC (tổng số va chạm); kết quả cũ chưa có TC thì suy từ CR × n (mỗi lượt nhiều nhất 1 va chạm)."""
        if r.get("TC") is not None:
            return int(r["TC"])
        if r.get("CR") is not None and r.get("n") is not None:
            return int(round(r["CR"] * r["n"]))
        return None

    # Bảng
    cols = [Col("Tập test"), Col("Số bản đồ", "right"), Col("Tới đích", "right", "SR"),
            Col("Va chạm", "right", "CR"), Col("Tổng va chạm", "right", "TC"), Col("Hết giờ", "right"),
            Col("Tỉ lệ kết quả", "left", "xanh: tới đích · đỏ: va chạm · cam: hết giờ", min_w=260, grow=True,
                max_w=520),
            Col("SPL", "right", "0–1"), Col("MTT", "right", "giây")]
    rows = []
    for s in sets:
        r = results[s]
        timeout = r.get("timeout")
        if timeout is None and r.get("SR") is not None and r.get("CR") is not None:
            timeout = max(0.0, 1 - r["SR"] - r["CR"])
        bar = [(r.get("SR") or 0, OUTCOMES["goal"][1]), (r.get("CR") or 0, OUTCOMES["collision"][1]),
               (timeout or 0, OUTCOMES["timeout"][1])]
        tc = _tc(r)
        rows.append(Row([Cell(set_label(s)), Cell(vn_int(r["n"])), Cell(vn_pct(r.get("SR")), weight="semibold"),
                         Cell(vn_pct(r.get("CR"))), Cell(vn_int(tc) if tc is not None else "—"),
                         Cell(vn_pct(timeout)), Cell(bar=bar),
                         Cell(vn_num(r["SPL"], 2) if r.get("SPL") is not None else "—"),
                         Cell(vn_num(r["MTT"], 1) if r.get("MTT") is not None else "—")]))
    if mean and len(levels) >= 2:
        to = [results[s].get("timeout") for s in levels]
        spl = [results[s].get("SPL") for s in levels]
        t_mean = sum(to) / len(to) if all(v is not None for v in to) else None
        spl_mean = sum(spl) / len(spl) if all(v is not None for v in spl) else None
        bar = [(mean.get("SR") or 0, OUTCOMES["goal"][1]), (mean.get("CR") or 0, OUTCOMES["collision"][1]),
               (t_mean or 0, OUTCOMES["timeout"][1])]
        n_lv = sum(int(results[s]["n"]) for s in levels)
        tc_lv = [_tc(results[s]) for s in levels]
        tc_mean = mean.get("TC") if mean.get("TC") is not None else (
            sum(tc_lv) if all(v is not None for v in tc_lv) else None)
        rows.append(Row([Cell(set_label(MEAN_SET)), Cell(vn_int(n_lv)), Cell(vn_pct(mean.get("SR"))),
                         Cell(vn_pct(mean.get("CR"))), Cell(vn_int(tc_mean) if tc_mean is not None else "—"),
                         Cell(vn_pct(t_mean)), Cell(bar=bar),
                         Cell(vn_num(spl_mean, 2) if spl_mean is not None else "—"),
                         Cell(vn_num(mean["MTT"], 1) if mean.get("MTT") is not None else "—")],
                        rule_above=True, bold=True))
    notes = [odef + " " + METRIC_DEF]
    if mean:
        notes.append("TB cấp 1–4: SR, CR, hết giờ, SPL là trung bình cộng của 4 cấp; TC là tổng của 4 cấp; "
                     "MTT tính trên mọi lượt tới đích của 4 cấp; không gồm BARN"
                     + (" và Lehman." if "test_lehman" in sets else "."))
    notes.append("TC: tổng số lần va chạm. Lượt dừng ngay ở lần va chạm đầu tiên nên TC = số lượt va chạm.")
    if source:
        notes.append(f"Số liệu: {source}.")
    table_png(table_path, Header("Kết quả chấm trên tập test", subtitle), [Table(cols, rows)], notes)

    # Biểu đồ: mỗi tập một thanh 100 %, chia theo kết quả
    bars = []
    for s in sets:
        r = results[s]
        timeout = r.get("timeout")
        if timeout is None:
            timeout = max(0.0, 1 - (r.get("SR") or 0) - (r.get("CR") or 0))
        parts = [(100 * (r.get("SR") or 0), OUTCOMES["goal"][1], vn_pct(r.get("SR"))),
                 (100 * (r.get("CR") or 0), OUTCOMES["collision"][1], vn_pct(r.get("CR"))),
                 (100 * timeout, OUTCOMES["timeout"][1], vn_pct(timeout))]
        bars.append(HBar(set_label(s), f"{vn_int(r['n'])} bản đồ", parts))
    chart = HBarChart(bars, 100, [0, 20, 40, 60, 80, 100], "Tỉ lệ số lượt (%)")
    hdr = Header("Tỉ lệ TỚI ĐÍCH, VA CHẠM, HẾT GIỜ trên từng tập test", subtitle, legend=legend)
    _compose(chart_path, [hdr, chart, Notes(["Số trong thanh là tỉ lệ số lượt; đoạn quá hẹp không ghi số "
                                             "(xem ảnh bảng). " + odef])])

    html_page(html_path, "Kết quả chấm trên tập test", subtitle,
              [("Biểu đồ: kết quả từng tập", chart_path), ("Bảng số liệu", table_path)],
              source=source, files=[table_path, chart_path])
    return [table_path, chart_path, html_path]


# ════════════════════════════════════════════════════════════════════════
#  2. Bảng kết quả của scripts/demo.py (cấu hình 2, 3, 7, 8, 9)
# ════════════════════════════════════════════════════════════════════════
def demo_report(rows: Sequence[dict], name: str, desc: str, base: Path | str, where: str = "",
                max_steps: int = 500, dt: float = 0.1, source: str = "scripts/demo.py") -> list[Path]:
    """rows: mỗi bản đồ một dict {idx, label, map_id, outcome, steps, dist, cut}."""
    table_path, chart_path, html_path = outputs_for(base)
    rows = list(rows)
    if not rows:
        raise ValueError("chưa chạy xong bản đồ nào")
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["label"], []).append(r)
    n_ok = sum(r["outcome"] == "goal" for r in rows)
    counts = ", ".join(f"{g}: {len(v)}" for g, v in groups.items())
    subtitle = (f"{len(rows)} bản đồ ({counts}{'; ' + where if where else ''}) · tới đích {n_ok}/{len(rows)} "
                f"({vn_pct(n_ok / len(rows))})")
    title = f"Kết quả demo: {name}"
    cut = any(r.get("cut") for r in rows)

    # Bảng 1: từng bản đồ
    cols = [Col("#", "right"), Col("Tập"), Col("Bản đồ"), Col("Kết quả"), Col("Số bước", "right", f"tối đa {max_steps}"),
            Col("Thời gian", "right", "giây"), Col("Cách đích", "right", "mét, cuối lượt")]
    trs = []
    for r in rows:
        label, color = OUTCOMES[r["outcome"]]
        star = " *" if r.get("cut") else ""
        trs.append(Row([Cell(str(r["idx"]), color=INK_2), Cell(r["label"]), Cell(r["map_id"]),
                        Cell(label, badge=color), Cell(f"{vn_int(r['steps'])}{star}"),
                        Cell(vn_num(r["steps"] * dt, 1)), Cell(vn_num(r["dist"], 2))]))
    t1 = Table(cols, trs, heading="Từng bản đồ")

    # Bảng 2: SR theo nhóm
    gcols = [Col("Nhóm"), Col("Số bản đồ", "right"), Col("Tới đích", "right"), Col("Va chạm", "right"),
             Col("Hết giờ", "right"), Col("SR", "right", "tỉ lệ tới đích"),
             Col("Tỉ lệ kết quả", min_w=260, grow=True, max_w=560)]
    grows = []
    items = list(groups.items())
    if len(items) > 1:
        items.append(("Tất cả", rows))
    for i, (g, rs) in enumerate(items):
        k = {o: sum(r["outcome"] == o for r in rs) for o in OUTCOME_ORDER}
        bar = [(k[o], OUTCOMES[o][1]) for o in OUTCOME_ORDER]
        grows.append(Row([Cell(g), Cell(str(len(rs))), Cell(str(k["goal"])), Cell(str(k["collision"])),
                          Cell(str(k["timeout"])), Cell(f"{k['goal']}/{len(rs)} · {vn_pct(k['goal'] / len(rs))}"),
                          Cell(bar=bar)],
                         rule_above=(g == "Tất cả"), bold=(g == "Tất cả")))
    t2 = Table(gcols, grows, heading="Tỉ lệ tới đích (SR) theo nhóm, chỉ tính các bản đồ vừa chạy")
    notes = [f"Bộ điều khiển: {desc}" if desc else "",
             "* Đã bấm N/Esc giữa lượt: phần còn lại của lượt chạy nhanh, không hiển thị." if cut else "",
             "Chỉ là vài bản đồ minh họa; SR, CR, SPL trên toàn bộ tập test: scripts/evaluate.py (cấu hình 4).",
             f"Số liệu: {source}." if source else ""]
    table_png(table_path, Header(title, subtitle), [t1, t2], notes)

    # Biểu đồ: số bước mỗi bản đồ, màu theo kết quả
    bars = []
    gap_before = []
    prev = None
    for i, r in enumerate(rows):
        label, color = OUTCOMES[r["outcome"]]
        if prev is not None and r["label"] != prev:
            gap_before.append(i)
        prev = r["label"]
        tip = f"{label} · {vn_int(r['steps'])} bước ({vn_num(r['steps'] * dt, 1)} s)" + (" *" if r.get("cut") else "")
        bars.append(HBar(f"{r['idx']}. {r['label']}", r["map_id"], [(r["steps"], color, "")], tip))
    ticks = list(range(0, max_steps + 1, 100 if max_steps >= 300 else 50))
    chart = HBarChart(bars, max_steps, ticks, f"Số bước mô phỏng (1 bước = {vn_num(dt, 1)} giây)",
                      ref=(max_steps, f"giới hạn {vn_int(max_steps)} bước"), bar_h=20 * S,
                      groups_gap=gap_before)
    tiles = [Tile(g, f"{sum(r['outcome'] == 'goal' for r in rs)}/{len(rs)}", "tới đích") for g, rs in groups.items()]
    if len(groups) > 1:
        tiles.append(Tile("Tất cả", f"{n_ok}/{len(rows)}", vn_pct(n_ok / len(rows))))
    if len(tiles) > 6:
        tiles = tiles[-1:]
    legend = [(OUTCOMES[k][0], OUTCOMES[k][1]) for k in OUTCOME_ORDER]
    hdr = Header(title, "Mỗi thanh là một bản đồ: dài tới bước kết thúc lượt, màu theo kết quả. " + subtitle,
                 legend=legend, tiles=tiles)
    _compose(chart_path, [hdr, chart, Notes(["Lượt TỚI ĐÍCH dừng khi robot vào vùng đích, VA CHẠM dừng ngay khi "
                                             "chạm vật cản, HẾT GIỜ chạy đủ giới hạn số bước."
                                             + (" * Bấm N/Esc giữa lượt." if cut else "")])])
    html_page(html_path, title, subtitle, [("Biểu đồ: số bước và kết quả từng bản đồ", chart_path),
                                           ("Bảng kết quả", table_path)],
              notes=[n for n in notes if n], source=source, files=[table_path, chart_path])
    return [table_path, chart_path, html_path]


# ════════════════════════════════════════════════════════════════════════
#  3. Log huấn luyện (scripts/train_neat.py, cấu hình 5)
# ════════════════════════════════════════════════════════════════════════
def _duration(seconds: float) -> str:
    """99.4 -> '99,4 giây'; 4593 -> '4.593 giây (khoảng 77 phút)'."""
    s = float(seconds)
    if s < 100:
        return f"{vn_num(s, 1)} giây"
    if s < 120:
        return f"{vn_int(s)} giây"
    return f"{vn_int(s)} giây (khoảng {vn_int(s / 60)} phút)"


def _trained_levels(hist: list[dict]) -> list[int]:
    """Cấp mà thế hệ đó học (log ghi cấp SAU khi xét lên cấp ở cuối thế hệ)."""
    return [int(h.get("level") or 0) - (1 if h.get("promotion") else 0) for h in hist]


def _novelty_text(s: dict) -> str:
    if not s.get("novelty"):
        return "không novelty (chỉ fitness)"
    fl = s.get("fixed_lambda")
    if fl is None:
        return "novelty lai (λ thích nghi)"
    if float(fl) == 0.0:
        return "novelty thuần (λ = 0)"
    return f"novelty, λ cố định = {vn_tick(float(fl))}"


def _schedule_text(s: dict) -> str:
    return {"curriculum": "curriculum (học từ cấp 1 lên cấp 4)", "uniform": "học đều cả 4 cấp"}.get(
        s.get("schedule", ""), str(s.get("schedule") or "?"))


def training_report(run_dir: Path | str) -> list[Path]:
    """Ảnh bảng, ảnh biểu đồ và trang HTML từ training_history.json (+ summary.json) của một lượt chạy."""
    run_dir = Path(run_dir)
    hist = json.loads((run_dir / "training_history.json").read_text(encoding="utf-8"))
    if not hist:
        raise ValueError("training_history.json rỗng")
    sp = run_dir / "summary.json"
    summ = json.loads(sp.read_text(encoding="utf-8")) if sp.is_file() else {}
    table_path, chart_path, html_path = outputs_for(run_dir / "training_history")
    G = len(hist)
    gens = [int(h["generation"]) for h in hist]
    lv = _trained_levels(hist)
    uniform = summ.get("schedule") == "uniform" or (not summ and len(set(lv)) == 1 and lv[0] == 4)
    champ_gen = summ.get("champion_generation")
    rec = summ.get("recurrent")
    kind = "mạng hồi quy" if rec else ("mạng truyền thẳng" if rec is not None else "NEAT")
    eps = summ.get("train_episodes")
    wall = summ.get("wall_seconds")
    sub_parts = [f"NEAT, {kind}", _novelty_text(summ) if summ else "", _schedule_text(summ) if summ else "",
                 f"{G} thế hệ", f"quần thể {summ['pop_size']}" if summ.get("pop_size") else "",
                 f"{vn_int(eps)} lượt huấn luyện" if eps else "", _duration(wall) if wall else ""]
    subtitle = " · ".join(p for p in sub_parts if p)
    title = f"Huấn luyện NEAT: {short_path(run_dir)}"

    # ô số liệu chính
    tiles = []
    if champ_gen is not None:
        tiles.append(Tile("Champion (theo SR val)", f"thế hệ {champ_gen}"))
    if summ.get("champion_val_mean") is not None:
        tiles.append(Tile("SR val trung bình 4 cấp", vn_pct(summ["champion_val_mean"])))
    cv = summ.get("champion_val") or {}
    if cv:
        keys = sorted(cv, key=lambda k: int(k))
        tiles.append(Tile("SR val cấp " + " · ".join(keys), " · ".join(vn_pct(cv[k], sign=False) for k in keys),
                          "%"))
    if summ:
        tiles.append(Tile("Cấp cao nhất đã học", "học đều 4 cấp" if uniform else f"cấp {summ.get('highest_level_trained')}"))
    if summ.get("champion_nodes") is not None:
        tiles.append(Tile("Mạng champion", f"{summ['champion_nodes']} nút · {summ.get('champion_connections')} kết nối"))

    # ── Biểu đồ: 3 khung chung trục thế hệ ─────────────────────────────
    best_raw = [float(h["best_raw_fitness"]) for h in hist]
    mean_raw = [float(h["mean_raw_fitness"]) for h in hist]
    best_sol = [100 * float(h["best_solved"]) for h in hist]
    mean_sol = [100 * float(h["mean_solved"]) for h in hist]
    species = [h.get("n_species") for h in hist]
    val_pts = [(int(h["generation"]), 100 * float(h["champion_check"]["best_val_sr"]))
               for h in hist if h.get("champion_check")]
    hdr = Header(title, subtitle, tiles=tiles[:5])
    chart = _TrainingPanels(gens, lv, uniform, champ_gen, best_raw, mean_raw, best_sol, mean_sol, species, val_pts)
    _compose(chart_path, [hdr, chart, Notes([
        f"Fitness thô: điểm trung bình trên {summ.get('maps_per_eval') or 8} bản đồ huấn luyện của thế hệ "
        "(không gồm novelty). Tới đích: tỉ lệ bản đồ huấn luyện tới đích của cá thể tốt nhất và trung bình cả "
        "quần thể. SR val: 5 cá thể fitness cao nhất chấm trên 100 bản đồ val của cả 4 cấp, ghi cá thể tốt nhất; "
        "champion là cá thể có SR val cao nhất.",
        "Chữ Cấp k phía trên và nền xám nhạt xen kẽ: cấp curriculum mà quần thể đang học." if not uniform else ""])])

    # ── Bảng ────────────────────────────────────────────────────────────
    kv = []
    if summ:
        kv += [("Kiểu mạng", kind), ("Novelty", _novelty_text(summ)), ("Lịch học", _schedule_text(summ)),
               ("Số thế hệ", f"{summ.get('generations', G)} (quần thể {summ.get('pop_size', '?')}, "
                             f"{summ.get('maps_per_eval', '?')} bản đồ mỗi thế hệ)"),
               ("Lượt huấn luyện", vn_int(eps) if eps else "—"),
               ("Thời gian", f"{_duration(wall)} với {summ.get('workers', '?')} worker" if wall else "—"),
               ("Champion", f"thế hệ {champ_gen}, SR val trung bình {vn_pct(summ.get('champion_val_mean'))}"
                if champ_gen is not None else "—")]
        if cv:
            kv.append(("SR val theo cấp", " · ".join(f"cấp {k}: {vn_pct(cv[k])}" for k in sorted(cv, key=int))))
        promos = summ.get("promotions") or []
        if not uniform:
            # SR val của lần xét lên cấp là SR trên val của riêng cấp đang học, không phải 4 cấp
            texts = [f"thế hệ {p.get('generation')}: cấp {p.get('from')} → {p.get('to')}"
                     + (f", SR val cấp {p.get('from')}: {vn_pct(p['val_sr'])}" if p.get("val_sr") is not None else "")
                     + (" (lên vì hết số thế hệ tối đa của cấp)" if p.get("reason") == "max_gens" else "")
                     for p in promos] or ["chưa lên cấp lần nào"]
            kv += [("Lên cấp" if i == 0 else "", t) for i, t in enumerate(texts)]
        if summ.get("champion_nodes") is not None:
            kv.append(("Mạng champion", f"{summ['champion_nodes']} nút (đầu ra và ẩn), "
                                        f"{summ.get('champion_connections')} kết nối đang bật"))
    t_sum = Table([Col("Mục"), Col("Giá trị", grow=True)], [Row([Cell(k, color=INK_2), Cell(v)]) for k, v in kv],
                  heading="Tóm tắt (summary.json)") if kv else None

    show = list(range(G))
    sampled = G > 30
    if sampled:
        keep = {0, G - 1}
        for i, h in enumerate(hist):
            g = int(h["generation"])
            if g % 10 == 0 or h.get("champion_check") or h.get("promotion") or g == champ_gen:
                keep.add(i)
        show = sorted(keep)
    has_lambda = any(abs(float(h.get("lambda", 1.0)) - 1.0) > 1e-12 for h in hist)
    gcols = [Col("Thế hệ", "right"), Col("Cấp", "center")]
    gcols += [Col("Fitness thô", "right", "tốt nhất"), Col("Fitness thô", "right", "trung bình"),
              Col("Tới đích", "right", "tốt nhất"), Col("Tới đích", "right", "trung bình"),
              Col("Số loài", "right")]
    if has_lambda:
        gcols.append(Col("λ", "right", "tỉ trọng fitness"))
    gcols += [Col("SR val", "right", "top 5, cả 4 cấp"), Col("Thời gian", "right", "giây")]
    grows = []
    for i in show:
        h = hist[i]
        g = int(h["generation"])
        lvl = "1–4" if uniform else (f"{lv[i]} → {h['level']}" if h.get("promotion") else str(lv[i]))
        cells = [Cell(str(g)), Cell(lvl), Cell(vn_num(h["best_raw_fitness"], 2)), Cell(vn_num(h["mean_raw_fitness"], 2)),
                 Cell(vn_pct(h["best_solved"], fixed=True)), Cell(vn_pct(h["mean_solved"], fixed=True)),
                 Cell(str(h.get("n_species") or "—"))]
        if has_lambda:
            cells.append(Cell(vn_num(float(h.get("lambda", 1.0)), 2)))
        cc = h.get("champion_check")
        cells += [Cell(vn_pct(cc["best_val_sr"]) if cc else "", weight="semibold" if cc else "regular"),
                  Cell(vn_num(float(h.get("seconds") or 0), 1))]
        grows.append(Row(cells, bold=(g == champ_gen)))
    heading = "Từng thế hệ" + (f" (trích {len(show)}/{G} dòng)" if sampled else "")
    t_gen = Table(gcols, grows, heading=heading)
    notes = ["Dòng in đậm: thế hệ của champion." if champ_gen is not None else "",
             (f"Bảng chỉ ghi thế hệ đầu, cuối, các thế hệ chia hết cho 10, có chấm val hoặc lên cấp; đủ {G} dòng "
              "trong training_history.html và training_history.json.") if sampled else "",
             "Cấp a → b: thế hệ đó học cấp a và cuối thế hệ được lên cấp b." if not uniform else "",
             "Số liệu: training_history.json, summary.json (scripts/train_neat.py)."]
    table_png(table_path, Header(title, subtitle), [t for t in (t_sum, t_gen) if t], notes)

    # ── HTML: hai ảnh + bảng đủ mọi thế hệ ─────────────────────────────
    head = ["Thế hệ", "Cấp", "Fitness thô tốt nhất", "Fitness thô TB", "Tới đích tốt nhất", "Tới đích TB",
            "Số loài"] + (["λ"] if has_lambda else []) + ["SR val", "Giây"]
    body = []
    for i, h in enumerate(hist):
        cc = h.get("champion_check")
        lvl = "1–4" if uniform else (f"{lv[i]} → {h['level']}" if h.get("promotion") else str(lv[i]))
        r = [h["generation"], lvl, vn_num(h["best_raw_fitness"], 2), vn_num(h["mean_raw_fitness"], 2),
             vn_pct(h["best_solved"], fixed=True), vn_pct(h["mean_solved"], fixed=True), h.get("n_species") or "—"]
        if has_lambda:
            r.append(vn_num(float(h.get("lambda", 1.0)), 2))
        r += [vn_pct(cc["best_val_sr"]) if cc else "", vn_num(float(h.get("seconds") or 0), 1)]
        body.append(r)
    strong = [i for i, h in enumerate(hist) if int(h["generation"]) == champ_gen]
    nums = [0] + list(range(2, len(head)))
    extra = (f"<details{' open' if G <= 30 else ''}><summary>Bảng đủ {G} thế hệ (chữ chọn và sao chép được)</summary>"
             + html_table(head, body, num_cols=nums, strong_rows=strong) + "</details>")
    html_page(html_path, title, subtitle, [("Biểu đồ quá trình huấn luyện", chart_path), ("Bảng log", table_path)],
              notes=[n for n in notes if n], extra_html=extra, source="scripts/train_neat.py",
              files=[table_path, chart_path, run_dir / "training_history.json"])
    return [table_path, chart_path, html_path]


class _TrainingPanels(_Part):
    """Ba khung xếp dọc, chung trục thế hệ: fitness thô, tỉ lệ tới đích (+ SR val), số loài."""

    def __init__(self, gens, levels, uniform, champ_gen, best_raw, mean_raw, best_sol, mean_sol, species, val_pts):
        self.gens, self.levels, self.uniform, self.champ = gens, levels, uniform, champ_gen
        self.best_raw, self.mean_raw = best_raw, mean_raw
        self.best_sol, self.mean_sol = best_sol, mean_sol
        self.species = species
        self.val_pts = val_pts
        self.panel_hs = (300.0, 300.0, 190.0)     # fitness, tới đích, số loài
        self.title_h = px(HEAD_PT) * 2.3
        self.axis_h = px(TICK_PT) * 1.6 + 16
        self.gap = 26.0
        self.xlabel_h = px(SUB_PT) * 1.8
        # làn chữ "Cấp k" phía trên khung đầu (chỉ khi có curriculum)
        self.lane_h = 0.0 if uniform else px(SMALL_PT) * 1.6
        self.gap_after = 26.0

    def height(self) -> float:
        return (3 * (self.title_h + self.axis_h) + sum(self.panel_hs) + 2 * self.gap + self.lane_h
                + self.xlabel_h)

    def draw(self, cv: _Canvas, y: float) -> None:
        from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
        m = _m()
        left = MARGIN + 96
        right_pad = 120
        w = AVAIL - 96 - right_pad
        G = len(self.gens)
        x_lo, x_hi = self.gens[0] - 0.5, self.gens[-1] + 0.5
        lw = _lw(2 * S)
        ms = 8 * S * 72 / DPI           # đường kính điểm 8 px CSS (point)
        mew = _lw(2 * S)                 # viền màu nền 2 px CSS quanh điểm
        blue, orange, aqua = SERIES[0], SERIES[1], SERIES[2]

        champ_key = [(f"champion (thế hệ {self.champ})", None)] if self.champ is not None else []
        val_key = []
        if self.val_pts:
            best = max(v for _, v in self.val_pts)
            val_key = [(f"SR val của 5 cá thể tốt nhất (cao nhất {vn_pct(best / 100)})", aqua)]
        panels = [
            ("Fitness thô trên bản đồ huấn luyện", [("tốt nhất", blue), ("trung bình quần thể", orange)] + champ_key),
            ("Tỉ lệ tới đích (%)", [("tốt nhất (huấn luyện)", blue), ("trung bình quần thể", orange)] + val_key),
            ("Số loài (NEAT chia loài để bảo vệ cấu trúc mới)", []),
        ]
        # dải cấp curriculum
        spans = []
        if not self.uniform:
            start = 0
            for i in range(1, G + 1):
                if i == G or self.levels[i] != self.levels[start]:
                    spans.append((self.gens[start] - 0.5, self.gens[i - 1] + 0.5, self.levels[start]))
                    start = i
        yy = y
        axes = []
        for k, (ptitle, legend) in enumerate(panels):
            base = yy + self.title_h * 0.62
            cv.text(left, base, ptitle, HEAD_PT, "semibold", INK)
            lx = left + m.w(ptitle, HEAD_PT, "semibold") + 36
            for lab, color in legend:
                mid = base - m.cap(SMALL_PT) / 2
                if color is None:            # khóa của đường dọc champion
                    cv.line(lx + 8, mid - 13, lx + 8, mid + 13, INK_2, 2.0, z=6)
                    tx = lx + 22
                elif lab.startswith("SR val"):
                    cv.ax.plot([lx + 12], [mid], marker="o", markersize=ms * 1.1,
                               color=color, markeredgecolor=SURFACE, markeredgewidth=mew, zorder=6)
                    tx = lx + 30
                else:
                    cv.line(lx, mid, lx + 26, mid, color, 3 * S, z=6)
                    tx = lx + 36
                cv.text(tx, base, lab, SMALL_PT, "regular", INK_2)
                lx = tx + m.w(lab, SMALL_PT) + 30
            yy += self.title_h
            if k == 0 and self.lane_h:
                for a, b, lvl in spans:
                    xa = left + (a - x_lo) / (x_hi - x_lo) * w
                    xb = left + (b - x_lo) / (x_hi - x_lo) * w
                    lab = f"Cấp {lvl}" if m.w(f"Cấp {lvl}", SMALL_PT) + 8 <= xb - xa else str(lvl)
                    cv.text((xa + xb) / 2, yy + px(SMALL_PT) * 1.15, lab, SMALL_PT, "regular", INK_2, ha="center")
                yy += self.lane_h
            ax = cv.axes(left, yy, w, self.panel_hs[k])
            axes.append(ax)
            ax.set_xlim(x_lo, x_hi)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color(BASELINE)
            ax.spines["bottom"].set_linewidth(_lw(2))
            ax.yaxis.grid(True, color=GRID, linewidth=_lw(1.5))
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", length=0, pad=8, labelsize=TICK_PT, labelcolor=INK_2)
            for i, (a, b, lvl) in enumerate(spans):
                if i % 2 == 1:          # cấp 2, 4 nền xám nhạt; cấp 1, 3 nền trắng
                    ax.axvspan(a, b, color=BAND, zorder=0, linewidth=0)
            if self.champ is not None:
                ax.axvline(self.champ, color=INK_2, linewidth=_lw(2), zorder=1.5)
            yy += self.panel_hs[k] + self.axis_h + (self.gap if k < 2 else 0)

        def mark(ax, xs, ys, color):
            ax.plot(xs, ys, color=color, linewidth=lw, solid_joinstyle="round", solid_capstyle="round", zorder=3)
            if G <= 25:
                ax.plot(xs, ys, linestyle="none", marker="o", markersize=ms, color=color,
                        markeredgecolor=SURFACE, markeredgewidth=mew, zorder=4)

        def end_labels(ax, series, fmt, top):
            """Số ở cuối đường; hai số quá gần nhau thì bỏ (chú giải và bảng vẫn có số)."""
            ends = [(s[-1], c) for s, c in series]
            ppu = self.panel_hs[0] / top
            if len(ends) == 2 and abs(ends[0][0] - ends[1][0]) * ppu < px(SMALL_PT) * 1.2:
                return
            for v, c in ends:
                ax.annotate(fmt(v), (self.gens[-1], v), xytext=(14, 0), textcoords="offset points", ha="left",
                            va="center", fontproperties=_fp(SMALL_PT), color=INK_2, annotation_clip=False)

        # khung 1: fitness thô
        ax = axes[0]
        top = max(max(self.best_raw), 1e-9) * 1.15
        ax.set_ylim(0, top)
        mark(ax, self.gens, self.best_raw, blue)
        mark(ax, self.gens, self.mean_raw, orange)
        end_labels(ax, [(self.best_raw, blue), (self.mean_raw, orange)], lambda v: vn_num(v, 1), top)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        # khung 2: tỉ lệ tới đích
        ax = axes[1]
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
        mark(ax, self.gens, self.best_sol, blue)
        mark(ax, self.gens, self.mean_sol, orange)
        if self.val_pts:
            vx, vy = zip(*self.val_pts)
            ax.plot(vx, vy, linestyle="none", marker="o", markersize=ms * 1.25, color=aqua,
                    markeredgecolor=SURFACE, markeredgewidth=mew, zorder=5)
        end_labels(ax, [(self.best_sol, blue), (self.mean_sol, orange)], lambda v: vn_pct(v / 100), 105)
        # khung 3: số loài
        ax = axes[2]
        sp_vals = [s if s is not None else float("nan") for s in self.species]
        finite = [s for s in sp_vals if s == s]
        ax.set_ylim(0, (max(finite) if finite else 1) * 1.25 + 0.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        mark(ax, self.gens, sp_vals, blue)
        for a in axes:
            a.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: vn_tick(v)))
            if G <= 12:
                a.xaxis.set_major_locator(FixedLocator(self.gens))
            else:
                a.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
            a.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: vn_int(v)))
        cv.text(left + w / 2, yy + px(SUB_PT) * 1.1, "Thế hệ", SUB_PT, "regular", INK_2, ha="center")


# ════════════════════════════════════════════════════════════════════════
#  4. Bảng nhiệt cho benchmark (scripts/plot_benchmark.py, cấu hình 6, docs/benchmark/)
# ════════════════════════════════════════════════════════════════════════
@dataclass
class MethodRow:
    name: str
    color: str
    hatch: Optional[str]
    baseline: bool
    seeds: int
    values: dict          # tập -> {"mean", "min", "max", "n_seeds"} (tỉ lệ 0–1)


_BASELINE_KEYS = ("ga", "apf")


def is_baseline(name: str) -> bool:
    """Baseline = không phải NEAT: GA B1, GA B2, APF (B5)."""
    import unicodedata
    key = unicodedata.normalize("NFC", name).casefold()
    if re.search(r"\bb[125]\b", key):
        return True
    return any(re.search(rf"(^|[^a-zà-ỹ]){re.escape(k)}($|[^a-zà-ỹ])", key) for k in _BASELINE_KEYS)


def heat_table_png(path: Path, title: str, subtitle: str, sets: Sequence[str], set_subs: dict,
                   methods: Sequence[MethodRow], notes: Sequence[str] = ()) -> Path:
    """Bảng phương pháp × tập: mỗi ô ghi trung bình theo seed (min–max), nền xanh càng đậm giá trị càng lớn."""
    cols = [Col("Phương pháp"), Col("Seed", "right")]
    for s in sets:
        cols.append(Col(set_label(s), "center", set_subs.get(s, "")))
    rows: list[Row] = []
    neat = [mm for mm in methods if not mm.baseline]
    base = [mm for mm in methods if mm.baseline]
    grouped = bool(neat) and bool(base)
    for title_g, group in (("NEAT (thuật toán chính)", neat), ("Baseline (để so sánh)", base)):
        if not group:
            continue
        if grouped:
            rows.append(Row(group=title_g))
        for mm in group:
            cells = [Cell(mm.name, swatch=mm.color, swatch_hatch=mm.hatch), Cell(str(mm.seeds), color=INK_2)]
            for s in sets:
                st = mm.values.get(s)
                if st is None:
                    cells.append(Cell("—", color=MUTED))
                    continue
                sub = ""
                if st.get("n_seeds", 1) >= 2:
                    sub = f"{vn_pct(st['min'], sign=False)}–{vn_pct(st['max'], sign=False)}"
                cells.append(Cell(vn_pct(st["mean"], sign=False, fixed=True), sub=sub, weight="semibold",
                                  fill=ramp_color(st["mean"])))
            rows.append(Row(cells))
    hdr = Header(title, subtitle, ramp=("0%", "100%"))
    return table_png(path, hdr, [Table(cols, rows, uniform=True)], notes)
