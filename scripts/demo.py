"""
Demo — xem bộ điều khiển chạy trên nhiều bản đồ (dùng khi trình bày với giáo viên)
===================================================================================
Mỗi bản đồ chạy một lượt trong cửa sổ pygame riêng của demo: sân 10 m x 10 m
(600 x 600 px) và một dải thông tin (HUD) phía trên ghi bộ điều khiển, cấp, mã
bản đồ, số bước, khoảng cách tới đích. Cuối lượt, kết quả hiện bằng màu:
TỚI ĐÍCH (xanh lá), VA CHẠM (đỏ), HẾT GIỜ (cam). Chạy xong, script in bảng kết
quả từng bản đồ và SR theo cấp, rồi ghi bảng đó dạng hình: <--report>_bang.png
(ảnh bảng), <--report>_bieu_do.png (ảnh biểu đồ số bước và kết quả từng bản đồ)
và <--report>.html (trang mở bằng trình duyệt); mặc định --report là
output/demo/ket_qua_demo, --no-visual để bỏ.

Usage
-----
    python scripts/demo.py                                      # champion trong demo/, 1 bản đồ test mỗi cấp
    python scripts/demo.py --levels 1,2,3,4 --maps-per-level 2
    python scripts/demo.py --run-dir output/demo/train_m6 --levels 1,2
    python scripts/demo.py --genome path/champion_genome.pkl --config path/neat_config.ini
    python scripts/demo.py --maps-per-level 0 --barn 4
    python scripts/demo.py --no-window --save-gif output/demo/demo.gif   # không mở cửa sổ, chỉ xuất GIF
    python scripts/demo.py --report output/demo/ket_qua_cau_hinh_2       # tên các file ảnh/HTML kết quả

Phím trong cửa sổ demo: Space tạm dừng / chạy tiếp; N hoặc mũi tên phải sang bản
đồ tiếp; + / - nhanh hơn / chậm hơn; Esc hoặc Q thoát (đóng cửa sổ cũng thoát).
Bấm N hay thoát giữa lượt thì phần còn lại của lượt vẫn được chạy (không hiển
thị, dưới 0,1 s) để bảng kết quả cuối cùng đúng; các dòng đó có dấu *.

Không có tham số bộ điều khiển thì dùng demo/champion_genome.pkl và
demo/neat_config.ini. Kiểu mạng (hồi quy hay truyền thẳng) suy ra từ
feed_forward của config (như scripts/visualize.py); --recurrent / --no-recurrent để ép.

GIF: Pillow giữ mọi khung trong RAM tới khi ghi file (khoảng 0,9 MB mỗi khung), nên
GIF quá 1.000 khung thì script tự lấy thưa khung hơn và in ra số bước mỗi khung.
Cửa sổ đóng trước khi ghi GIF, nên không bị treo trong lúc ghi.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
# Cho phép chạy "python scripts/demo.py" từ thư mục gốc dự án
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")   # không in dòng chào của pygame

import numpy as np  # noqa: E402

from src.env.kinematics import ARENA_H, ARENA_W, DT, MAX_STEPS  # noqa: E402
from src.env.renderer import _BG as ARENA_BG  # noqa: E402  (màu nền sân do renderer vẽ)
from src.env.robot_env import RobotNavEnv  # noqa: E402
from src.neat_engine.parallel import make_policy, neat_spec  # noqa: E402

DEMO_DIR = ROOT / "demo"
DEFAULT_REPORT = ROOT / "output" / "demo" / "ket_qua_demo"   # tiền tố tên file ảnh/HTML kết quả
ARENA = 600                 # cạnh sân tính bằng px (= Renderer.window_size)
PX_PER_M = ARENA / max(ARENA_W, ARENA_H)    # như Renderer.scale
HUD_H = 72                  # chiều cao dải thông tin phía trên sân
FPS_LADDER = (2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240)
GIF_STEP = 5                # GIF: tối đa một khung hình mỗi 5 bước mô phỏng (0,5 s)
GIF_FIRST_MS = 500          # GIF: giữ khung đầu của mỗi bản đồ 0,5 s
# Pillow chỉ ghi GIF khi đóng file nên mọi khung nằm trong RAM tới lúc đó: đo được khoảng
# 0,9 MB mỗi khung lúc ghi. Quá GIF_MAX_FRAMES khung thì lấy thưa khung hơn (plan_gif).
GIF_MAX_FRAMES = 1000
GIF_MB_PER_FRAME = 0.9
# Renderer vẽ quỹ đạo nét 1 px, khó thấy trên máy chiếu: demo vẽ thêm nét đậm cùng màu
# NẰM DƯỚI tường, tia LiDAR, đích và robot (xem Display.render_env).
TRAIL = (100, 100, 255)     # = _TRAIL của src/env/renderer.py (bỏ kênh alpha)
TRAIL_W = 3

OUTCOMES = {                # mã kết quả -> (chữ hiện ra, màu)
    "goal": ("TỚI ĐÍCH", (70, 200, 100)),
    "collision": ("VA CHẠM", (230, 70, 70)),
    "timeout": ("HẾT GIỜ", (245, 155, 45)),
}
HUD_BG = (16, 16, 20)
HUD_LINE = (70, 70, 85)
TEXT = (235, 235, 240)
MUTED = (150, 150, 165)
BADGE_TEXT = (18, 18, 22)
RUNNING = (55, 65, 85)
PAUSED = (235, 200, 70)

# Phông chữ: tên SysFont -> file kiểu chữ thường / đậm trong %WINDIR%\Fonts.
# Trên Windows, pygame.font.SysFont("segoeui") trả về Segoe UI *Light* (nét mảnh,
# khó đọc trên máy chiếu) và SysFont("arial") trả về Arial *Narrow* (không có chữ
# Việt có dấu), nên thử đúng file trước rồi mới dùng SysFont.
FONT_CANDIDATES = ("segoeui", "arial", "tahoma", "notosans", "dejavusans")
_WIN_FONT_FILES = {
    "segoeui": ("segoeui.ttf", "segoeuib.ttf"),
    "arial": ("arial.ttf", "arialbd.ttf"),
    "tahoma": ("tahoma.ttf", "tahomabd.ttf"),
}
FONT_SPECS = {"title": (21, True), "text": (17, False), "small": (16, False), "badge": (20, True)}
VN_SAMPLE = ("ạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹđơư"
             "ẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸĐƠƯ")


# ════════════════════════════════════════════════════════════════════════
#  Tiện ích
# ════════════════════════════════════════════════════════════════════════
def fail(msg: str) -> None:
    print(f"Lỗi: {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


def warn(msg: str) -> None:
    print(f"Cảnh báo: {msg}", flush=True)


def _resolve(p: str | Path) -> Path:
    """Đường dẫn người dùng nhập: theo thư mục hiện tại, không có thì thử từ gốc repo."""
    q = Path(p).expanduser()
    if q.is_absolute() or q.exists():
        return q
    alt = ROOT / q
    return alt if alt.exists() else q


def _rel(p: Path) -> str:
    try:
        return Path(os.path.abspath(p)).relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def _num(x: float, nd: int = 2) -> str:
    """Số thập phân kiểu Việt Nam (dấu phẩy)."""
    return f"{x:.{nd}f}".replace(".", ",")


def _plain(text: str) -> str:
    """Bỏ dấu tiếng Việt (chỉ dùng khi không có phông nào vẽ được chữ có dấu)."""
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def _outcome(info: dict) -> str:
    if info.get("is_success"):
        return "goal"
    if info.get("is_collision"):
        return "collision"
    return "timeout"


# ════════════════════════════════════════════════════════════════════════
#  Bộ điều khiển
# ════════════════════════════════════════════════════════════════════════
def _neat_controller(genome_path: Path, cfg_path: Path, recurrent: Optional[bool], source: str):
    import neat

    missing = [str(p) for p in (genome_path, cfg_path) if not p.is_file()]
    if missing:
        fail("không thấy file " + ", ".join(missing))
    try:
        with open(genome_path, "rb") as f:
            genome = pickle.load(f)
    except Exception as e:  # noqa: BLE001
        fail(f"không đọc được genome {genome_path}: {e!r}")
    try:
        cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                          neat.DefaultStagnation, str(cfg_path))
    except Exception as e:  # noqa: BLE001
        fail(f"không nạp được config {cfg_path}: {e!r}\n"
             "  Gợi ý: đặt biến môi trường PYTHONUTF8=1 (các cấu hình trong .vscode/launch.json đã đặt sẵn).")
    if recurrent is None:
        recurrent = not cfg.genome_config.feed_forward
    policy = make_policy(neat_spec(genome, str(cfg_path.resolve()), recurrent))
    kind = "hồi quy" if recurrent else "truyền thẳng"
    hidden = len(genome.nodes) - cfg.genome_config.num_outputs
    enabled = sum(1 for c in genome.connections.values() if c.enabled)
    desc = (f"NEAT {kind}: genome {_rel(genome_path)}, config {_rel(cfg_path)} "
            f"({hidden} nút ẩn, {enabled} kết nối đang bật)")
    return policy, f"NEAT {kind} · {source}", desc


def build_controller(args):
    """Trả (policy, tên hiện trên HUD, mô tả in ra terminal)."""

    if args.run_dir:
        run = _resolve(args.run_dir)
        if not run.is_dir():
            fail(f"không thấy thư mục {args.run_dir}")
        cfg = run / "neat_config.ini"
        if not cfg.is_file() and (run / "summary.json").is_file():
            # lượt chạy cũ không chép config: lấy đường dẫn ghi trong summary.json (như evaluate.py)
            try:
                alt = Path(json.load(open(run / "summary.json", encoding="utf-8"))["config"])
                if alt.is_file():
                    cfg = alt
            except (OSError, KeyError, ValueError):
                pass
        return _neat_controller(run / "champion_genome.pkl", cfg, args.recurrent, f"champion {run.name}")

    if args.genome:
        g = _resolve(args.genome)
        if args.config:
            cfg = _resolve(args.config)
        else:
            cfg = g.with_name("neat_config.ini")
            if not cfg.is_file():
                cfg = ROOT / "configs" / "neat_recurrent.ini"
                warn(f"không có --config và không có neat_config.ini cạnh genome, dùng {_rel(cfg)}")
        return _neat_controller(g, cfg, args.recurrent, f"{g.parent.name}/{g.name}")

    if args.config:
        fail("--config phải đi cùng --genome")
    g, cfg = DEMO_DIR / "champion_genome.pkl", DEMO_DIR / "neat_config.ini"
    if not (g.is_file() and cfg.is_file()):
        fail("chưa có champion mặc định của demo (cần demo/champion_genome.pkl và demo/neat_config.ini).\n"
             "  Chép champion_genome.pkl và neat_config.ini từ một thư mục kết quả huấn luyện "
             "(ví dụ output/demo/train_m6) vào thư mục demo/,\n"
             "  hoặc chỉ định bộ điều khiển: --run-dir THƯ_MỤC | --genome FILE --config FILE")
    return _neat_controller(g, cfg, args.recurrent, "champion trong demo/")


# ════════════════════════════════════════════════════════════════════════
#  Chọn bản đồ
# ════════════════════════════════════════════════════════════════════════
@dataclass
class MapItem:
    path: Path
    label: str          # "Cấp 1".."Cấp 4", "BARN"
    map_id: str
    data: dict


def _load_item(path: Path) -> MapItem:
    try:
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError) as e:
        fail(f"không đọc được bản đồ {path}: {e!r}")
    source = str(m.get("source") or "")
    folder = path.parent.name
    if folder == "test_barn" or source.upper() == "BARN":
        label = "BARN"
    elif m.get("difficulty_level") is not None:
        label = f"Cấp {m['difficulty_level']}"
    else:
        label = folder
    return MapItem(path, label, str(m.get("map_id") or path.stem), m)


def select_maps(args) -> list[MapItem]:
    data_dir = _resolve(args.data_dir)
    items: list[MapItem] = []
    if args.maps:
        for s in (x.strip() for x in args.maps.split(",")):
            if not s:
                continue
            p = _resolve(s)
            if not p.is_file():
                fail(f"không thấy file bản đồ {s}")
            items.append(_load_item(p))
    else:
        for lvl in args.levels:
            folder = data_dir / args.split / f"level_{lvl}"
            files = sorted(folder.glob("*.json"))
            if not files:
                warn(f"không có bản đồ nào trong {_rel(folder)}")
                continue
            chosen = files[args.start_index: args.start_index + args.maps_per_level]
            if args.maps_per_level > 0 and not chosen:
                warn(f"--start-index {args.start_index} vượt quá số bản đồ ({len(files)}) trong {_rel(folder)}")
            items.extend(_load_item(p) for p in chosen)
    if args.barn > 0:
        files = sorted((data_dir / "test_barn").glob("*.json"))
        if not files:
            warn(f"không có bản đồ BARN trong {_rel(data_dir / 'test_barn')}")
        items.extend(_load_item(p) for p in files[: args.barn])
    return items


# ════════════════════════════════════════════════════════════════════════
#  Hiển thị: cửa sổ pygame, HUD, GIF
# ════════════════════════════════════════════════════════════════════════
def _load_font(pg, name: str, size: int, bold: bool):
    windir = os.environ.get("WINDIR")
    if windir and name in _WIN_FONT_FILES:
        f = Path(windir) / "Fonts" / _WIN_FONT_FILES[name][int(bold)]
        if f.is_file():
            return pg.font.Font(str(f), size)
    if not pg.font.match_font(name, bold=bold):
        return None          # SysFont sẽ âm thầm trả phông mặc định, nên bỏ qua tên này
    return pg.font.SysFont(name, size, bold=bold)


def _has_vietnamese(pg, font) -> bool:
    """So từng chữ có dấu với ô vuông "thiếu chữ" (.notdef) của chính phông đó."""
    tobytes = getattr(pg.image, "tobytes", None) or pg.image.tostring

    def sig(ch: str):
        s = font.render(ch, False, (255, 255, 255), (0, 0, 0))
        return s.get_size(), tobytes(s, "RGB")

    tofu = sig("\u0378")     # U+0378 chưa được gán trong Unicode: không phông nào có chữ này
    return all(sig(ch) != tofu for ch in VN_SAMPLE)


def pick_fonts(pg):
    """Trả (dict phông, tên phông, plain). plain=True: không phông nào có chữ Việt, phải bỏ dấu."""
    for name in FONT_CANDIDATES:
        try:
            fonts = {k: _load_font(pg, name, size, bold) for k, (size, bold) in FONT_SPECS.items()}
            if all(f is not None and _has_vietnamese(pg, f) for f in fonts.values()):
                return fonts, name, False
        except Exception:  # noqa: BLE001  (file phông hỏng, ...)
            continue
    fonts = {}
    for k, (size, bold) in FONT_SPECS.items():
        f = pg.font.Font(None, round(size * 1.3))
        try:
            f.bold = bold
        except AttributeError:
            f.set_bold(bold)
        fonts[k] = f
    plain = not all(_has_vietnamese(pg, f) for f in fonts.values())
    return fonts, "phông mặc định của pygame", plain


@dataclass
class Hud:
    title: str
    counter: str
    line: str
    outcome: Optional[str] = None


class GifRecorder:
    """Ghi GIF động bằng imageio (plugin pillow), từng khung một."""

    def __init__(self, path: Path, fps: float):
        import imageio.v3 as iio

        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.frame_ms = max(20, int(round(1000.0 / fps / 10)) * 10)   # GIF đo thời gian theo 1/100 s
        self.durations: list[int] = []
        self._w = iio.imopen(str(path), "w", plugin="pillow", extension=".gif")

    def add(self, rgb: np.ndarray, ms: Optional[float] = None) -> None:
        self.durations.append(max(20, int(round((ms if ms is not None else self.frame_ms) / 10)) * 10))
        # bits=8: lượng tử hóa ngay về 256 màu (ảnh P, 1/3 bộ nhớ của ảnh RGB). Pillow chỉ
        # ghi file khi đóng và đọc duration lúc đó, nên truyền lại cả danh sách mỗi lần.
        self._w.write(rgb, bits=8, duration=list(self.durations), loop=0)

    def close(self) -> tuple[int, float, int]:
        """Ghi file; trả (số khung, thời lượng giây, kích thước byte)."""
        self._w.close()
        n = len(self.durations)
        size = self.path.stat().st_size if (n and self.path.exists()) else 0
        return n, sum(self.durations) / 1000.0, size


class Display:
    """Ghép khung hình sân + HUD; mở cửa sổ pygame và/hoặc ghi GIF."""

    def __init__(self, window: bool, gif_path: Optional[Path], fps: float, scale: float, gif_fps: float):
        import pygame

        self.pg = pygame
        pygame.font.init()
        self.fonts, self.font_name, self.plain = pick_fonts(pygame)
        self.canvas = pygame.Surface((ARENA, ARENA + HUD_H))
        self.arena = None
        self._trail_layer = None
        self.window = None
        self.clock = None
        self.scale = 1.0
        self.fps = float(fps)
        self.paused = False
        self._caption = ""
        if window:
            self._open_window(scale)
        self.gif = GifRecorder(gif_path, gif_fps) if gif_path else None

    # ── cửa sổ ───────────────────────────────────────────────────────────
    def _open_window(self, scale: float) -> None:
        pg = self.pg
        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
        try:
            pg.display.init()
            w, h = ARENA, ARENA + HUD_H
            try:   # thu nhỏ nếu cửa sổ cao hơn màn hình (laptop đặt tỉ lệ hiển thị 150 %)
                dw, dh = pg.display.get_desktop_sizes()[0]
                if dw > 0 and dh > 0:
                    fit = min((dh - 90) / h, (dw - 40) / w)
                    if scale > fit:
                        print(f"Cửa sổ thu lại tỉ lệ {fit:.2f} cho vừa màn hình {dw}x{dh}.")
                        scale = fit
            except (AttributeError, IndexError, pg.error):
                pass
            self.scale = max(0.3, float(scale))
            size = (round(w * self.scale), round(h * self.scale))
            self.window = pg.display.set_mode(size)
        except pg.error as e:
            fail(f"không mở được cửa sổ pygame ({e}).\n"
                 "  Dùng --no-window --save-gif output/demo/demo.gif để xuất GIF thay cho cửa sổ.")
        pg.display.set_caption("Demo robot NEAT")
        try:
            pg.key.stop_text_input()     # không để bộ gõ tiếng Việt (IME) nuốt phím N, Q, ...
        except (AttributeError, pg.error):
            pass
        self.clock = pg.time.Clock()

    def poll(self) -> Optional[str]:
        """Xử lý sự kiện cửa sổ; trả "quit", "next" hoặc None. Space và +/- xử lý tại chỗ."""
        if self.window is None:
            return None
        pg = self.pg
        cmd = None
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                return "quit"
            if ev.type != pg.KEYDOWN:
                continue
            key = ev.key
            if key in (pg.K_ESCAPE, pg.K_q):
                return "quit"
            if key in (pg.K_n, pg.K_RIGHT):
                cmd = "next"
            elif key == pg.K_SPACE:
                self.paused = not self.paused
            elif key in (pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS):
                self.fps = next((f for f in FPS_LADDER if f > self.fps), FPS_LADDER[-1])
            elif key in (pg.K_MINUS, pg.K_KP_MINUS, pg.K_UNDERSCORE):
                self.fps = next((f for f in reversed(FPS_LADDER) if f < self.fps), FPS_LADDER[0])
        return cmd

    def present(self, idle: bool = False) -> None:
        """Đưa khung hình lên cửa sổ và giữ nhịp fps (không có cửa sổ thì bỏ qua)."""
        if self.window is None:
            return
        pg = self.pg
        if self.scale != 1.0:
            self.window.blit(pg.transform.smoothscale(self.canvas, self.window.get_size()), (0, 0))
        else:
            self.window.blit(self.canvas, (0, 0))
        pg.display.flip()
        self.clock.tick(min(self.fps, 30.0) if idle else self.fps)

    def close_window(self) -> None:
        if self.window is not None:
            self.pg.display.quit()
            self.window = None

    def close(self) -> Optional[tuple[int, float, int]]:
        """Đóng cửa sổ TRƯỚC rồi mới ghi GIF: Pillow mã hóa mọi khung lúc này (vài giây với
        vài trăm khung), cửa sổ còn mở thì bị treo và Windows báo "Not Responding"."""
        self.close_window()
        gif_info = None
        if self.gif is not None:
            n = len(self.gif.durations)
            if n:
                print(f"\nĐang ghi GIF ({n} khung hình)...", flush=True)
            gif_info = self.gif.close()
            self.gif = None
        self.pg.quit()
        return gif_info

    # ── vẽ ───────────────────────────────────────────────────────────────
    def _t(self, text: str) -> str:
        return _plain(text) if self.plain else text

    @staticmethod
    def _fit(font, text: str, max_w: int) -> str:
        if font.size(text)[0] <= max_w:
            return text
        while text and font.size(text + "...")[0] > max_w:
            text = text[:-1]
        return text + "..."

    def render_env(self, env: RobotNavEnv) -> None:
        """Khung hình sân của env, thêm quỹ đạo nét đậm nằm dưới mọi thứ khác.

        Vẽ quỹ đạo TRAIL_W px lên một lớp nền riêng, rồi dán khung của env lên trên với
        màu nền sân là trong suốt (colorkey): tường, tia LiDAR, đích, robot vẫn nằm trên.
        """
        pg = self.pg
        frame = pg.surfarray.make_surface(env.render().swapaxes(0, 1))   # (600, 600) RGB
        traj = env.trajectory
        if len(traj) < 2:
            self.arena = frame
            return
        if self._trail_layer is None:
            self._trail_layer = pg.Surface((ARENA, ARENA))
        layer = self._trail_layer
        layer.fill(ARENA_BG)
        # cùng phép đổi tọa độ với Renderer._to_px (trục y hướng lên)
        pts = [(int(p[0] * PX_PER_M), int(ARENA - p[1] * PX_PER_M)) for p in traj]
        pg.draw.lines(layer, TRAIL, False, pts, TRAIL_W)
        frame.set_colorkey(ARENA_BG)
        layer.blit(frame, (0, 0))
        self.arena = layer

    def draw(self, hud: Hud) -> None:
        pg, c, fonts = self.pg, self.canvas, self.fonts
        c.fill(HUD_BG)
        if self.arena is not None:
            c.blit(self.arena, (0, HUD_H))
        pg.draw.line(c, HUD_LINE, (0, HUD_H - 1), (ARENA, HUD_H - 1))

        # Ô trạng thái bên phải: kết quả (màu) / tạm dừng / đang chạy
        if hud.outcome:
            label, color = OUTCOMES[hud.outcome]
            fg = BADGE_TEXT
            pg.draw.rect(c, color, (0, HUD_H, ARENA, ARENA), width=4)    # viền sân cùng màu kết quả
        elif self.paused:
            label, color, fg = "TẠM DỪNG", PAUSED, BADGE_TEXT
        else:
            label, color, fg = "ĐANG CHẠY", RUNNING, TEXT
        badge = fonts["badge"].render(self._t(label), True, fg)
        rect = pg.Rect(0, 6, max(150, badge.get_width() + 28), 34)
        rect.right = ARENA - 10
        pg.draw.rect(c, color, rect, border_radius=7)
        c.blit(badge, badge.get_rect(center=rect.center))

        # Dòng 1: tên bộ điều khiển, rồi "Bản đồ k/n" sát ô trạng thái nếu còn chỗ
        title = self._fit(fonts["title"], self._t(hud.title), rect.left - 24)
        ts = fonts["title"].render(title, True, TEXT)
        c.blit(ts, (12, rect.centery - ts.get_height() // 2))
        cnt = fonts["small"].render(self._t(hud.counter), True, MUTED)
        if 12 + ts.get_width() + 16 + cnt.get_width() + 12 <= rect.left:
            c.blit(cnt, (rect.left - 12 - cnt.get_width(), rect.centery - cnt.get_height() // 2))

        # Dòng 2: cấp · mã bản đồ · bước · khoảng cách, rồi fps (chỉ khi có cửa sổ) nếu còn chỗ
        ls = fonts["text"].render(self._fit(fonts["text"], self._t(hud.line), ARENA - 24), True, TEXT)
        c.blit(ls, (12, 45))
        if self.window is not None:
            right = "TẠM DỪNG" if (self.paused and hud.outcome) else f"{self.fps:g} fps"
            rs = fonts["small"].render(self._t(right), True, PAUSED if self.paused else MUTED)
            if 12 + ls.get_width() + 16 + rs.get_width() + 12 <= ARENA:
                c.blit(rs, (ARENA - 12 - rs.get_width(), 47))
            # thanh tiêu đề cửa sổ luôn ghi số bản đồ và tốc độ (HUD có thể không đủ chỗ)
            caption = (f"Demo robot NEAT — {hud.counter} — {self.fps:g} fps"
                       f"{' — TẠM DỪNG' if self.paused else ''}   |   Space: tạm dừng · N/→: bản đồ tiếp · "
                       "+/-: nhanh/chậm · Esc/Q: thoát")
            if caption != self._caption:
                pg.display.set_caption(caption)
                self._caption = caption

    def snapshot(self) -> np.ndarray:
        tobytes = getattr(self.pg.image, "tobytes", None) or self.pg.image.tostring
        return np.frombuffer(tobytes(self.canvas, "RGB"), dtype=np.uint8).reshape(ARENA + HUD_H, ARENA, 3)


# ════════════════════════════════════════════════════════════════════════
#  Chạy demo
# ════════════════════════════════════════════════════════════════════════
@dataclass
class Result:
    idx: int
    label: str
    map_id: str
    outcome: str          # goal / collision / timeout
    steps: int
    dist: float           # khoảng cách tới đích cuối lượt (m)
    cut: bool             # bấm N/Esc giữa lượt: phần còn lại chạy không hiển thị


class DemoRunner:
    def __init__(self, policy, name: str, items: list[MapItem], disp: Optional[Display],
                 frame_skip: int, gif_step: int, pause: float):
        self.policy = policy
        self.name = name
        self.items = items
        self.disp = disp
        self.frame_skip = frame_skip
        self.gif_step = gif_step
        self.pause = pause
        self.quit = False

    def _hud(self, k: int, item: MapItem, env: RobotNavEnv, outcome: Optional[str] = None) -> Hud:
        dist = float(np.linalg.norm(env.pos - env.goal_pos))
        line = f"{item.label} · {item.map_id} · bước {env.step_count}/{MAX_STEPS} · cách đích {_num(dist)} m"
        return Hud(self.name, f"Bản đồ {k}/{len(self.items)}", line, outcome)

    def play(self, k: int, item: MapItem) -> Result:
        d = self.disp
        env = RobotNavEnv(map_data=item.data, render_mode="rgb_array" if d else None)
        obs, _ = env.reset()
        if hasattr(self.policy, "reset"):
            self.policy.reset()       # mạng hồi quy: xóa trạng thái đầu mỗi lượt
        state = {"obs": obs, "done": False, "info": {}}

        def advance(n: int) -> None:
            for _ in range(n):
                action = np.asarray(self.policy.activate(state["obs"].tolist()), dtype=np.float32)
                state["obs"], _, terminated, truncated, state["info"] = env.step(action)
                if terminated or truncated:
                    state["done"] = True
                    return

        cut = False
        if d is None:
            advance(MAX_STEPS)        # MAX_STEPS bước luôn đủ để lượt kết thúc
        else:
            d.paused = False
            d.render_env(env)
            d.draw(self._hud(k, item, env))
            if d.gif:
                d.gif.add(d.snapshot(), ms=max(GIF_FIRST_MS, d.gif.frame_ms))
            d.present()
            while not state["done"]:
                cmd = d.poll()
                if cmd:
                    cut = True
                    self.quit = cmd == "quit"
                    break
                if d.paused:
                    d.draw(self._hud(k, item, env))
                    d.present(idle=True)
                    continue
                advance(self.frame_skip)
                d.render_env(env)
                d.draw(self._hud(k, item, env))
                if d.gif and not state["done"] and env.step_count % self.gif_step == 0:
                    d.gif.add(d.snapshot())
                d.present()
            if cut:
                advance(MAX_STEPS)    # chạy nốt lượt, không hiển thị
                d.render_env(env)
            outcome = _outcome(state["info"])
            d.draw(self._hud(k, item, env, outcome))
            if d.gif:
                d.gif.add(d.snapshot(), ms=max(self.pause * 1000.0, d.gif.frame_ms))
            if d.window is not None and not cut:
                self._hold(d, k, item, env, outcome)

        env.close()
        info = state["info"]
        return Result(k, item.label, item.map_id, _outcome(info), int(info["steps"]),
                      float(info["dist_to_goal"]), cut)

    def _hold(self, d: Display, k: int, item: MapItem, env: RobotNavEnv, outcome: str) -> None:
        """Giữ khung hình kết quả --pause giây; vẫn xử lý phím (Space giữ lại, N bỏ qua)."""
        remaining = self.pause
        last = time.monotonic()
        while True:
            cmd = d.poll()
            if cmd == "quit":
                self.quit = True
                return
            if cmd == "next":
                return
            now = time.monotonic()
            if not d.paused:
                remaining -= now - last
            last = now
            if remaining <= 0:
                return
            d.draw(self._hud(k, item, env, outcome))
            d.present(idle=True)

    def run(self) -> list[Result]:
        results: list[Result] = []
        n = len(self.items)
        try:
            for k, item in enumerate(self.items, 1):
                r = self.play(k, item)
                results.append(r)
                text = OUTCOMES[r.outcome][0]
                star = "  (*)" if r.cut else ""
                print(f"[{k}/{n}] {r.label:<7} {r.map_id:<22} -> {text} sau {r.steps} bước "
                      f"({_num(r.steps * DT, 1)} s){star}", flush=True)
                if self.quit:
                    if k < n:
                        print(f"Đã thoát, bỏ qua {n - k} bản đồ còn lại.")
                    break
        except KeyboardInterrupt:
            print("\nĐã dừng bằng Ctrl+C.")
        return results


def print_summary(results: list[Result], name: str) -> None:
    print()
    if not results:
        print("Chưa chạy xong bản đồ nào.")
        return
    wl = max(4, max(len(r.label) for r in results))
    wm = max(6, max(len(r.map_id) for r in results))
    print(f"Kết quả demo: {name}")
    head = f"{'#':>3}  {'Tập':<{wl}}  {'Bản đồ':<{wm}}  {'Kết quả':<9}  {'Số bước':>7}  {'Cách đích (m)':>13}"
    print(head)
    print("-" * len(head))
    for r in results:
        star = " *" if r.cut else ""
        print(f"{r.idx:>3}  {r.label:<{wl}}  {r.map_id:<{wm}}  {OUTCOMES[r.outcome][0]:<9}  "
              f"{r.steps:>7}  {_num(r.dist):>13}{star}")
    if any(r.cut for r in results):
        print("  * đã bấm N/Esc giữa lượt: phần còn lại của lượt chạy nhanh, không hiển thị")

    print()
    kind = "theo cấp" if all(r.label.startswith("Cấp") for r in results) else "theo nhóm bản đồ"
    print(f"Tỉ lệ tới đích (SR) {kind}, chỉ tính các bản đồ vừa chạy:")
    groups: dict[str, list[Result]] = {}
    for r in results:
        groups.setdefault(r.label, []).append(r)
    rows = list(groups.items())
    if len(rows) > 1:
        rows.append(("Tất cả", results))
    wg = max(len(g) for g, _ in rows)
    for g, rs in rows:
        ok = sum(r.outcome == "goal" for r in rs)
        print(f"  {g:<{wg}}  {ok:>2}/{len(rs):<2}  SR = {_num(100.0 * ok / len(rs), 1):>5} %")
    print("(Chỉ là vài bản đồ minh họa. SR, CR, SPL trên toàn bộ tập test: scripts/evaluate.py)")


# ════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════
def _levels(text: str) -> list[int]:
    try:
        lv = [int(x) for x in text.replace(" ", "").split(",") if x]
    except ValueError:
        raise argparse.ArgumentTypeError("dạng 1,2,3,4") from None
    if not lv or any(x not in (1, 2, 3, 4) for x in lv):
        raise argparse.ArgumentTypeError("chỉ gồm các cấp 1..4, ví dụ 1,2,3,4")
    return lv


def gif_frames_max(n_maps: int, gif_step: int) -> int:
    """Số khung GIF tối đa: mỗi bản đồ có khung đầu, một khung mỗi gif_step bước trước bước
    MAX_STEPS (lượt kết thúc ở bước cuối thì không ghi khung đó) và khung kết quả."""
    return n_maps * (math.ceil(MAX_STEPS / gif_step) + 1)


def plan_gif(n_maps: int, frame_skip: int, gif_step: int, window: bool) -> tuple[int, int]:
    """Nới gif_step tới khi GIF không quá GIF_MAX_FRAMES khung; trả (frame_skip, gif_step).

    Có cửa sổ: gif_step giữ là bội của frame_skip (khung GIF rơi đúng khung đã vẽ), cửa sổ
    vẫn chạy như cũ. Không cửa sổ: frame_skip = gif_step (chỉ vẽ những khung được ghi).
    """
    unit = frame_skip if window else 1
    step = gif_step
    while gif_frames_max(n_maps, step) > GIF_MAX_FRAMES and step < MAX_STEPS:
        step += unit
    if gif_frames_max(n_maps, step) > GIF_MAX_FRAMES:
        fail(f"{n_maps} bản đồ là quá nhiều cho một GIF (tối đa {GIF_MAX_FRAMES // 2} bản đồ): "
             "giảm số bản đồ hoặc bỏ --save-gif")
    if step != gif_step:
        before, after = gif_frames_max(n_maps, gif_step), gif_frames_max(n_maps, step)
        print(f"GIF: {n_maps} bản đồ, lấy một khung mỗi {gif_step} bước thì tới {before} khung "
              f"(khoảng {_num(before * GIF_MB_PER_FRAME / 1000, 1)} GB RAM lúc ghi file). "
              f"Lấy thưa hơn, một khung mỗi {step} bước: tối đa {after} khung.")
    return (frame_skip if window else step), step


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Demo trực quan: bộ điều khiển chạy trên nhiều bản đồ, cửa sổ pygame có HUD tiếng Việt.")
    g = ap.add_argument_group("bộ điều khiển (mặc định: demo/champion_genome.pkl + demo/neat_config.ini)")
    src = g.add_mutually_exclusive_group()
    src.add_argument("--genome", type=str, default=None, help="file genome NEAT (.pkl)")
    src.add_argument("--run-dir", type=str, default=None,
                     help="thư mục kết quả huấn luyện (champion_genome.pkl + neat_config.ini)")
    g.add_argument("--config", type=str, default=None,
                   help="config NEAT đi cùng --genome (mặc định: neat_config.ini cạnh genome)")
    g.add_argument("--recurrent", dest="recurrent", action="store_true", default=None,
                   help="ép mạng hồi quy (mặc định suy ra từ feed_forward của config)")
    g.add_argument("--no-recurrent", dest="recurrent", action="store_false", help="ép mạng truyền thẳng")
    g.add_argument("--label", type=str, default=None, help="tên bộ điều khiển hiện trên HUD")

    m = ap.add_argument_group("bản đồ")
    m.add_argument("--data-dir", type=str, default=str(ROOT / "data" / "maps"))
    m.add_argument("--split", choices=["test", "val", "train"], default="test")
    m.add_argument("--levels", type=_levels, default=[1, 2, 3, 4], help="ví dụ 1,2,3,4 (mặc định cả 4 cấp)")
    m.add_argument("--maps-per-level", type=int, default=1, help="số bản đồ mỗi cấp (0 = không lấy bản đồ cấp)")
    m.add_argument("--start-index", type=int, default=0, help="vị trí bắt đầu trong danh sách file đã sắp xếp")
    m.add_argument("--barn", type=int, default=0, help="thêm N bản đồ đầu của data/maps/test_barn")
    m.add_argument("--maps", type=str, default=None,
                   help="danh sách file bản đồ, cách nhau dấu phẩy (thay cho --split/--levels)")

    v = ap.add_argument_group("hiển thị")
    v.add_argument("--fps", type=float, default=30.0, help="khung hình mỗi giây của cửa sổ (mặc định 30)")
    v.add_argument("--frame-skip", type=int, default=None,
                   help="số bước mô phỏng mỗi khung hình (mặc định 1 khi có cửa sổ, 5 khi --no-window)")
    v.add_argument("--pause", type=float, default=1.5, help="giây giữ khung kết quả cuối mỗi bản đồ")
    v.add_argument("--scale", type=float, default=1.0, help="phóng to cửa sổ, ví dụ 1.3 khi chiếu màn lớn")
    v.add_argument("--save-gif", type=str, default=None,
                   help=f"ghi GIF động (có HUD) vào đường dẫn này; tối đa {GIF_MAX_FRAMES} khung, "
                        "dài hơn thì tự lấy thưa khung")
    v.add_argument("--gif-fps", type=float, default=10.0, help="khung hình mỗi giây của GIF (mặc định 10)")
    v.add_argument("--no-window", action="store_true", help="không mở cửa sổ (chạy ngầm, dùng với --save-gif)")

    r = ap.add_argument_group("bảng kết quả dạng hình")
    r.add_argument("--report", type=str, default=None,
                   help="tiền tố tên file: ghi <tiền tố>_bang.png, <tiền tố>_bieu_do.png và <tiền tố>.html "
                        f"(mặc định {_rel(DEFAULT_REPORT)})")
    r.add_argument("--no-visual", action="store_true", help="không ghi ảnh bảng, ảnh biểu đồ và trang HTML")
    args = ap.parse_args(argv)

    for name in ("maps_per_level", "start_index", "barn"):
        if getattr(args, name) < 0:
            ap.error(f"--{name.replace('_', '-')} phải >= 0")
    if args.frame_skip is not None and args.frame_skip < 1:
        ap.error("--frame-skip phải >= 1")
    if args.fps <= 0 or args.gif_fps <= 0 or args.scale <= 0 or args.pause < 0:
        ap.error("--fps, --gif-fps, --scale phải > 0 và --pause >= 0")
    return args


def main():
    from src.console import utf8_stdout
    utf8_stdout()
    args = parse_args()
    if args.no_window:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # không cần màn hình

    policy, name, desc = build_controller(args)
    if args.label:
        name = args.label
    items = select_maps(args)
    if not items:
        fail("không có bản đồ nào để chạy (xem --levels, --maps-per-level, --barn, --maps)")

    window = not args.no_window
    frame_skip = args.frame_skip or (1 if window else 5)
    # GIF lấy một khung mỗi gif_step bước; gif_step là bội của frame_skip để rơi đúng khung đã vẽ
    gif_step = frame_skip * math.ceil(GIF_STEP / frame_skip) if window else frame_skip

    counts: dict[str, int] = {}
    for it in items:
        counts[it.label] = counts.get(it.label, 0) + 1
    where = "file chỉ định" if args.maps else f"tập {args.split}"
    print(f"Bộ điều khiển: {name}")
    print(f"  {desc}")
    print(f"Bản đồ: {len(items)} ({', '.join(f'{k}: {v}' for k, v in counts.items())}; {where})")
    if args.save_gif:
        frame_skip, gif_step = plan_gif(len(items), frame_skip, gif_step, window)

    disp = None
    if window or args.save_gif:
        disp = Display(window, Path(args.save_gif) if args.save_gif else None,
                       args.fps, args.scale, args.gif_fps)
        if disp.plain:
            warn("không tìm thấy phông chữ có tiếng Việt, chữ trong cửa sổ sẽ bỏ dấu")
        if window:
            print("Cửa sổ demo: Space tạm dừng/chạy tiếp · N hoặc → sang bản đồ tiếp · "
                  "+ / - nhanh/chậm · Esc hoặc Q thoát")
        if args.save_gif:
            print(f"Đang ghi GIF: một khung mỗi {gif_step} bước, {args.gif_fps:g} khung/giây")
    print(flush=True)

    runner = DemoRunner(policy, name, items, disp, frame_skip, gif_step, args.pause)
    try:
        results = runner.run()
    finally:
        if disp:
            disp.close_window()     # cửa sổ đóng ngay khi chạy xong, trước khi in bảng và ghi GIF
    print_summary(results, name)
    gif_info = disp.close() if disp else None
    if gif_info:
        n, secs, size = gif_info
        print(f"Đã ghi GIF {args.save_gif}: {n} khung hình, {_num(secs, 1)} s, {_num(size / 1e6, 1)} MB")
    if results and not args.no_visual:
        write_visuals(results, name, desc, args, where)


def write_visuals(results: list[Result], name: str, desc: str, args, where: str) -> None:
    """Bảng kết quả dạng hình: ảnh bảng, ảnh biểu đồ, trang HTML (src/visual_report.py)."""
    from src.visual_report import demo_report, print_written, warn_failed
    base = Path(args.report) if args.report else DEFAULT_REPORT
    try:
        paths = demo_report([asdict(r) for r in results], name, desc, base, where=where,
                            max_steps=MAX_STEPS, dt=DT)
    except Exception as e:  # noqa: BLE001  (ảnh là phần thêm: bảng ở terminal đã in xong)
        warn_failed("ảnh và trang HTML của bảng kết quả", e)
        return
    print()
    print_written(paths)


if __name__ == "__main__":
    main()
