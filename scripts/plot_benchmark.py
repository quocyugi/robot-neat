"""
Plot Benchmark — biểu đồ cột nhóm so sánh các phương pháp trên các tập test
==============================================================================
Đọc <thư mục phương pháp>/seed_*/test_results.json (do scripts/evaluate.py hoặc
scripts/run_baselines.py ghi) và gom một chỉ số theo seed: trung bình, min, max.
Kết quả:
  - biểu đồ cột nhóm (PNG, rộng 1600 px, file --out): mỗi nhóm là một tập test, mỗi cột
    là một phương pháp theo đúng thứ tự trong --runs; số trên cột là trung bình theo seed,
    thanh lỗi là min–max theo seed, mỗi chấm là một seed;
  - ảnh bảng (PNG, cạnh --out, tên <tên>_bang.png): phương pháp × tập test, mỗi ô ghi trung
    bình theo seed và min–max, nền xanh càng đậm giá trị càng lớn; NEAT trước, baseline sau;
  - trang HTML (cạnh --out, tên <tên>.html): biểu đồ và ảnh bảng nhúng sẵn, mở bằng trình
    duyệt. --no-visual để bỏ ảnh bảng và trang HTML (biểu đồ vẫn ghi);
  - bảng markdown (phương pháp × tập test) in ra màn hình, ghi ra --md nếu có;
  - file JSON đã gom (--json) để vẽ lại bằng --from-json khi không còn thư mục kết quả;
    file chỉ giữ số liệu của các tập đã chọn (--sets).

Usage
-----
    python scripts/plot_benchmark.py \
        --runs "M6=output/ablation/M6" "GA B1=output/baselines/ga_b1" \
        --metric SR --out output/benchmark_SR.png \
        --json output/benchmark_SR.json --md output/benchmark_SR.md

    # vẽ lại từ file JSON đã gom
    python scripts/plot_benchmark.py --from-json output/benchmark_SR.json --out output/benchmark_SR.png

--runs    TÊN=THƯ_MỤC; THƯ_MỤC chứa seed_*/test_results.json. Không có seed_* mà
          có THƯ_MỤC/test_results.json thì coi là một seed.
--sets    tên tập, cách nhau dấu phẩy. Mặc định test/level_1..test/level_4,test_barn;
          thêm mean_levels_1_4 (trung bình 4 cấp) nếu cần.
--metric  SR (mặc định), CR, SPL hoặc timeout; biểu đồ và bảng ghi theo % (giá trị × 100) với
          dấu phẩy thập phân (96,5); file --json vẫn ghi số (tỉ lệ 0–1).
--json    file JSON để commit (docs/benchmark/): mục "dir" của mỗi phương pháp chỉ để tham
          khảo; đường dẫn tuyệt đối nằm ngoài thư mục hiện tại chỉ giữ 2 cấp cuối
          (".../bench_out/M6"), để file không mang tên người dùng (C:\\Users\\<tên>\\...).

Mỗi phương pháp lấy màu theo vị trí trong --runs (phương pháp chưa
có kết quả vẫn giữ chỗ màu của nó), nên muốn màu giống nhau giữa các biểu đồ thì giữ
nguyên thứ tự --runs. Từ phương pháp học thứ 9 trở đi màu lặp lại kèm nền gạch chéo.
Tập hoặc phương pháp thiếu số liệu được bỏ qua kèm cảnh báo; không còn gì để vẽ
thì dừng với mã lỗi 1.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import textwrap
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_SETS = ("test/level_1", "test/level_2", "test/level_3", "test/level_4", "test_barn")
LEVELS = ("test/level_1", "test/level_2", "test/level_3", "test/level_4")
MEAN_SET = "mean_levels_1_4"

METRICS = {
    "SR": {"name": "Tỉ lệ thành công (SR)", "axis": "Tỉ lệ thành công (%)", "better": "Cao hơn là tốt hơn"},
    "CR": {"name": "Tỉ lệ va chạm (CR)", "axis": "Tỉ lệ va chạm (%)", "better": "Thấp hơn là tốt hơn"},
    "SPL": {"name": "SPL (thành công có trọng số theo độ dài đường đi)", "axis": "SPL (%)",
            "better": "Cao hơn là tốt hơn"},
    "timeout": {"name": "Tỉ lệ hết giờ", "axis": "Tỉ lệ hết giờ (%)", "better": "Thấp hơn là tốt hơn"},
}
SET_LABELS = {"test_barn": "BARN", MEAN_SET: "TB cấp 1–4"}

# Light chart palette. The 8 categorical slots keep a fixed order: adjacent pairs stay
# apart under protan/deutan simulation (worst OKLab dE 9.1) and for normal vision
# (worst 19.6). Yellow, aqua and magenta sit under 3:1 against white, which is why every
# bar carries its value.
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
# A 9th learning method never gets a made-up hue: the colours repeat with a 45° texture
# (then 135°, then both), so identity is hue x texture and the legend shows both.
HATCHES = (None, "///", "\\\\\\", "xxx")


def warn(msg: str) -> None:
    print(f"Cảnh báo: {msg}", file=sys.stderr)


def set_label(name: str) -> str:
    if name in SET_LABELS:
        return SET_LABELS[name]
    m = re.fullmatch(r"(train|val|test)/level_(\d+)", name)
    if m:
        return f"Cấp {m.group(2)}" if m.group(1) == "test" else f"Cấp {m.group(2)} ({m.group(1)})"
    return name


def scope(sets: list[str]) -> str:
    """'test' when every plotted set comes from the test maps, else a neutral word."""
    return "test" if all(s.startswith("test") or s == MEAN_SET for s in sets) else "bản đồ"


def natural_key(s: str) -> list:
    return [int(t) if t.isdigit() else t.casefold() for t in re.split(r"(\d+)", s)]


def decimal_comma(s: str) -> str:
    """A formatted number as the project's Vietnamese documents write it: '96.5' -> '96,5'.
    Only for text people read (table, bar labels); the --json file keeps plain numbers."""
    return s.replace(".", ",")


def pct(v: float) -> str:
    """Rate 0–1 as a percentage with 1 decimal and a decimal comma (0.965 -> '96,5')."""
    return decimal_comma(f"{100 * v:.1f}")


def short_num(v: float) -> str:
    """Label on a bar (value already in %): whole numbers from 10 up (43.6 -> '44'), one
    decimal below 10 (1.2 -> '1,2', 4.0 -> '4'), so labels stay short; the table keeps 1 decimal.
    Decimal comma, like pct()."""
    if 99.5 <= v < 100 - 1e-9:      # never print '100' for a run that was not perfect
        return decimal_comma(f"{math.floor(v * 10) / 10:.1f}")
    r = math.floor(v * 10 + 0.5) / 10
    if r >= 10:
        return str(math.floor(v + 0.5))
    return str(int(r)) if r == int(r) else decimal_comma(f"{r:.1f}")


def with_seeds(name: str, k: int) -> str:
    """Legend label with the seed count. A name that already ends in a bracket takes the count
    inside it, so 'GA (B1)' reads 'GA (B1, 1 seed)' rather than 'GA (B1) (1 seed)'."""
    if name.endswith(")") and "(" in name:
        return f"{name[:-1]}, {k} seed)"
    return f"{name} ({k} seed)"


# ── Reading test_results.json ─────────────────────────────────────────────

def parse_runs(items: list[str]) -> list[tuple[str, Path]]:
    runs, seen = [], set()
    for item in items:
        name, sep, path = item.partition("=")
        if not sep:  # bare folder: the folder name is the method name
            name, path = Path(item.rstrip("/\\")).name, item
        name, path = unicodedata.normalize("NFC", name.strip()), path.strip()
        if not name or not path:
            sys.exit(f"Lỗi: --runs cần dạng TÊN=THƯ_MỤC, nhận được '{item}'.")
        if name in seen:
            sys.exit(f"Lỗi: tên phương pháp '{name}' xuất hiện hai lần trong --runs.")
        seen.add(name)
        runs.append((name, Path(path)))
    return runs


def discover(root: Path) -> tuple[list[tuple[str, Path]], list[str]]:
    """(seed name, file) for every seed_*/test_results.json, plus seed folders without one."""
    if root.is_file():
        return [(root.parent.name, root)], []
    seed_dirs = sorted((d for d in root.glob("seed_*") if d.is_dir()), key=lambda d: natural_key(d.name))
    files = [(d.name, d / "test_results.json") for d in seed_dirs if (d / "test_results.json").is_file()]
    pending = [d.name for d in seed_dirs if not (d / "test_results.json").is_file()]
    if not files and (root / "test_results.json").is_file():
        files = [(root.name, root / "test_results.json")]
    return files, pending


def metric_value(results: dict, set_name: str, metric: str) -> float | None:
    entry = results.get(set_name)
    v = entry.get(metric) if isinstance(entry, dict) else None
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
        return float(v)
    if set_name == MEAN_SET:  # evaluation.py stores only SR and CR here: rebuild from the 4 levels
        vals = [metric_value(results, s, metric) for s in LEVELS]
        if all(x is not None for x in vals):
            return sum(vals) / len(vals)
    return None


def summarize(per_seed: dict[str, float], n_maps: set[int]) -> dict:
    vals = list(per_seed.values())
    out = {"mean": round(statistics.fmean(vals), 6), "min": min(vals), "max": max(vals),
           "n_seeds": len(vals), "per_seed": per_seed}
    if n_maps:
        out["n_maps"] = min(n_maps) if len(n_maps) == 1 else [min(n_maps), max(n_maps)]
    return out


def collect_method(name: str, root: Path, metric: str) -> dict | None:
    if not root.exists():
        warn(f"bỏ qua {name}: không có thư mục {root}")
        return None
    files, pending = discover(root)
    if not files:
        if pending:
            warn(f"bỏ qua {name}: {', '.join(pending)} chưa có test_results.json "
                 f"(có thể còn đang chạy) trong {root}")
        else:
            warn(f"bỏ qua {name}: không thấy seed_*/test_results.json trong {root}")
        return None
    if pending:
        warn(f"{name}: {len(pending)} thư mục seed chưa có test_results.json, bỏ qua: {', '.join(pending)}")
    seeds: dict[str, dict] = {}
    for seed, f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
        except ValueError as e:
            warn(f"{name}: bỏ qua {f} (JSON hỏng, có thể đang được ghi dở: {e})")
            continue
        except OSError as e:
            warn(f"{name}: bỏ qua {f} (không mở được: {e})")
            continue
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, dict):
            warn(f"{name}: bỏ qua {f} (không có mục 'results')")
            continue
        seeds[seed] = results
    if not seeds:
        warn(f"bỏ qua {name}: không đọc được file kết quả nào trong {root}")
        return None
    set_names = list(dict.fromkeys(k for res in seeds.values() for k in res))
    if MEAN_SET not in set_names:
        set_names.append(MEAN_SET)
    sets = {}
    for s in set_names:
        per_seed, n_maps = {}, set()
        for seed, res in seeds.items():
            v = metric_value(res, s, metric)
            if v is None:
                continue
            per_seed[seed] = round(v, 6)
            n = res[s].get("n") if isinstance(res.get(s), dict) else None
            if isinstance(n, int):
                n_maps.add(n)
        if per_seed:
            sets[s] = summarize(per_seed, n_maps)
    return {"dir": str(root), "seeds": list(seeds), "sets": sets}


def portable_dir(d: str) -> str:
    """Folder as written to --json. The JSON is meant to be committed (docs/benchmark/), so it
    must not carry the user's home folder (C:\\Users\\<name>\\...): a relative path stays as
    given; an absolute one becomes relative to the current folder when it lies inside it,
    otherwise only its last two parts are kept (".../bench_out/M6"). The field is informational."""
    if not d:
        return d
    p = Path(d)
    if not p.is_absolute():
        return p.as_posix()
    try:
        return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except (OSError, ValueError):
        tail = [part for part in p.parts if part != p.anchor][-2:]
        return ".../" + "/".join(tail)


def aggregate(runs: list[tuple[str, Path]], metric: str) -> dict:
    methods = {}
    for name, root in runs:
        m = collect_method(name, root, metric)
        if m is not None:
            methods[name] = m
    return {"metric": metric, "methods": methods}


def load_aggregate(path: Path) -> dict:
    """Read back the file written by --json."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as e:
        sys.exit(f"Lỗi: không đọc được {path}: {e}")
    if not isinstance(data, dict) or data.get("metric") not in METRICS or not isinstance(data.get("methods"), dict):
        sys.exit(f"Lỗi: {path} không phải file do plot_benchmark.py --json ghi (thiếu 'metric' hoặc 'methods').")
    order = [n for n in data.get("method_order") or [] if n in data["methods"]]
    order += [n for n in data["methods"] if n not in order]
    methods = {}
    for name in order:
        m = data["methods"][name]
        if not isinstance(m, dict) or not isinstance(m.get("sets"), dict):
            warn(f"bỏ qua {name}: thiếu số liệu trong {path}")
            continue
        style = None
        if re.fullmatch(r"#[0-9a-fA-F]{6}", str(m.get("color", ""))) and m.get("hatch") in HATCHES:
            style = {"color": m["color"], "hatch": m.get("hatch")}
        sets = {}
        for s, st in m["sets"].items():
            try:
                per_seed = {str(k): float(v) for k, v in (st.get("per_seed") or {}).items()}
                entry = {"mean": float(st["mean"]), "min": float(st["min"]), "max": float(st["max"]),
                         "n_seeds": int(st.get("n_seeds") or len(per_seed) or 1), "per_seed": per_seed}
            except (AttributeError, KeyError, TypeError, ValueError):
                warn(f"{name}: bỏ qua tập {s} (số liệu hỏng trong {path})")
                continue
            n_maps = st.get("n_maps")
            if isinstance(n_maps, int) or (isinstance(n_maps, list) and n_maps
                                           and all(isinstance(x, int) for x in n_maps)):
                entry["n_maps"] = n_maps
            sets[s] = entry
        methods[name] = {"dir": m.get("dir", ""), "seeds": list(m.get("seeds") or []), "sets": sets,
                         "style": style}
    return {"metric": data["metric"], "methods": methods, "sets": data.get("sets"), "title": data.get("title")}


def select(methods: dict, wanted: list[str]) -> tuple[list[str], list[str]]:
    """Sets with data for at least one method, and methods with data in at least one of them."""
    sets = []
    for s in wanted:
        have = [n for n, m in methods.items() if s in m["sets"]]
        if not have:
            warn(f"không phương pháp nào có số liệu tập {s}, bỏ qua tập này")
            continue
        sets.append(s)
        for n in have:
            got, total = methods[n]["sets"][s]["n_seeds"], len(methods[n]["seeds"])
            if total and got < total:
                warn(f"{n}: tập {s} chỉ có ở {got}/{total} seed")
    names = []
    for n, m in methods.items():
        missing = [s for s in sets if s not in m["sets"]]
        if sets and len(missing) == len(sets):
            warn(f"bỏ qua {n}: không có số liệu ở tập nào được chọn")
            continue
        if missing:
            warn(f"{n}: thiếu số liệu tập {', '.join(missing)} (ô để trống)")
        names.append(n)
    return sets, names


def assign_styles(names: list[str]) -> dict[str, dict]:
    """Colour follows the method's position in --runs."""
    styles = {}
    for k, n in enumerate(names):
        styles[n] = {"color": SERIES[k % len(SERIES)], "hatch": HATCHES[(k // len(SERIES)) % len(HATCHES)]}
    if len(names) > len(SERIES) * len(HATCHES):
        warn(f"{len(names)} phương pháp: màu bắt đầu lặp lại, nên tách thành nhiều biểu đồ")
    return styles


def maps_count(methods: dict, names: list[str], s: str) -> str:
    ns = set()
    for n in names:
        v = methods[n]["sets"].get(s, {}).get("n_maps")
        if v is not None:
            ns.update(v if isinstance(v, list) else [v])
    if not ns:
        return ""
    lo, hi = min(ns), max(ns)
    return f"{lo} bản đồ" if lo == hi else f"{lo}–{hi} bản đồ"


def seeds_of(methods: dict, n: str, sets: list[str]) -> int:
    return max((methods[n]["sets"][s]["n_seeds"] for s in sets if s in methods[n]["sets"]), default=0)


# ── Markdown table ────────────────────────────────────────────────────────

def markdown_table(metric: str, methods: dict, names: list[str], sets: list[str]) -> str:
    groups: dict[str, list[str]] = {}
    for s in sets:
        cnt = maps_count(methods, names, s)
        if cnt:
            groups.setdefault(cnt, []).append(set_label(s))
    maps = "; ".join(f"{', '.join(lbls)}: {cnt}" for cnt, lbls in groups.items())
    info = METRICS[metric]
    lines = [f"**{info['name']}**, %: trung bình theo seed (min–max), số seed. {info['better']}."
             + (f" Số bản đồ: {maps}." if maps else ""), ""]
    lines.append("| Phương pháp | " + " | ".join(set_label(s) for s in sets) + " |")
    lines.append("|:--|" + "--:|" * len(sets))
    for n in names:
        cells = []
        for s in sets:
            st = methods[n]["sets"].get(s)
            cells.append("—" if st is None else
                         f"{pct(st['mean'])}% ({pct(st['min'])}–{pct(st['max'])}), {st['n_seeds']} seed")
        name_cell = n.replace("|", "\\|")
        lines.append(f"| {name_cell} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ── Chart ────────────────────────────────────────────────────────────────

def swarm(ys: list[float], d: float, limit: float) -> list[float]:
    """Horizontal offsets (px) for dots of diameter d at heights ys (px): a dot stays on the
    whisker unless it would cover an earlier one, then moves sideways, at most `limit` px."""
    step = min(0.8 * d, limit) if limit > 0 else 0.0
    placed: list[tuple[float, float]] = []
    offs = []
    for y in ys:
        best, best_gap = 0.0, -1.0
        for k in range(41 if step else 1):
            x = ((k + 1) // 2) * step * (1 if k % 2 else -1)
            if abs(x) > limit + 1e-9:
                break
            gap = min((math.hypot(x - px, y - py) for px, py in placed), default=math.inf)
            if gap >= 0.8 * d:
                best = x
                break
            if gap > best_gap:
                best, best_gap = x, gap
        placed.append((best, y))
        offs.append(best)
    # Centre each cluster of dots that touch vertically on the whisker (ties sit symmetric).
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    clusters, cur = [], [order[0]] if order else []
    for i in order[1:]:
        if ys[i] - ys[cur[-1]] < 0.8 * d:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
    if cur:
        clusters.append(cur)
    for cl in clusters:
        mid = (min(offs[i] for i in cl) + max(offs[i] for i in cl)) / 2
        for i in cl:
            offs[i] -= mid
    return offs


def plot_chart(metric: str, methods: dict, names: list[str], sets: list[str], styles: dict,
               out: Path, title: str | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import FancyBboxPatch, PathPatch
    from matplotlib.path import Path as MPath

    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = [f for f in ("Segoe UI", "DejaVu Sans") if f in installed] or ["DejaVu Sans"]
    if "Segoe UI Semibold" in installed:
        title_font = {"fontfamily": ["Segoe UI Semibold"] + family}
    else:
        title_font = {"fontfamily": family, "fontweight": "bold"}

    W, H, DPI = 10.0, 5.5, 160          # 1600 × 880 px
    PX = DPI / 96                      # device pixels per CSS pixel (mark sizes below are CSS px)
    TEXT_X, AX_LEFT, AX_RIGHT, AX_BOTTOM = 0.25, 0.85, 0.3, 0.8   # inches
    G, M = len(sets), len(names)
    info = METRICS[metric]
    seed_counts = {n: seeds_of(methods, n, sets) for n in names}
    multi = any(k >= 2 for k in seed_counts.values())
    same_k = len(set(seed_counts.values())) == 1

    if multi:
        how = (f"Cột: trung bình {seed_counts[names[0]]} seed" if same_k else "Cột: trung bình các seed")
        how += " · thanh lỗi: min–max · chấm: từng seed"
    else:
        how = "Mỗi phương pháp 1 seed"
    subtitle = f"{info['better']}. {how}."
    if M == 1:
        subtitle = f"Phương pháp {names[0]}. " + subtitle

    def darker(hex_color: str, k: float = 0.38) -> tuple:
        r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
        return (r * (1 - k), g * (1 - k), b * (1 - k))

    def fit_width(t, renderer, avail_px: float, min_fs: float) -> None:
        fs = t.get_fontsize()
        while t.get_window_extent(renderer).width > avail_px and fs > min_fs:
            fs -= 0.5
            t.set_fontsize(fs)
        w = t.get_window_extent(renderer).width
        if w > avail_px:
            s = t.get_text()
            t.set_text(textwrap.fill(s, max(20, int(len(s) * avail_px / w) - 2)))

    rc = {"font.family": family, "font.size": 9, "axes.unicode_minus": False, "hatch.linewidth": 0.9}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURFACE)
        rend = fig.canvas.get_renderer()
        avail = (W - TEXT_X - AX_RIGHT) * DPI

        # Header: title, subtitle, legend stacked from the top; the plot gets what is left.
        y = H - 0.2
        if title:
            t = fig.text(TEXT_X / W, y / H, title, ha="left", va="top", fontsize=13, color=INK, **title_font)
            fit_width(t, rend, avail, 10)
            y -= t.get_window_extent(rend).height / DPI + 0.07
        t = fig.text(TEXT_X / W, y / H, subtitle, ha="left", va="top", fontsize=9, color=INK_2)
        fit_width(t, rend, avail, 8)
        y -= t.get_window_extent(rend).height / DPI + 0.14

        if M >= 2:
            # Legend drawn as a grid by hand: rows read left to right in bar order and every
            # row shares one baseline (matplotlib's legend lays out each column on its own,
            # so a name with stacked diacritics such as "Ngẫu" shifts its row).
            fs_leg = 9
            em = fs_leg / 72                                      # inches
            sw, tpad, cpad, row_h = 0.85 * em, 0.5 * em, 1.6 * em, 1.8 * em
            labels = [n if same_k or not multi else with_seeds(n, seed_counts[n]) for n in names]
            item_w = []
            for lab in labels:
                t = fig.text(0, 0, lab, fontsize=fs_leg)
                item_w.append(sw + tpad + t.get_window_extent(rend).width / DPI)
                t.remove()

            def col_widths(ncol: int) -> list[float]:
                return [max(item_w[i] for i in range(c, M, ncol)) for c in range(ncol)]

            ncol = M
            while ncol > 1 and sum(col_widths(ncol)) + cpad * (ncol - 1) > avail / DPI:
                ncol -= 1
            ncol = math.ceil(M / math.ceil(M / ncol))                # same rows, balanced
            cw = col_widths(ncol)
            for i, (n, lab) in enumerate(zip(names, labels)):
                r, c = divmod(i, ncol)
                x = TEXT_X + sum(cw[:c]) + cpad * c
                base = y - r * row_h - 0.92 * em
                sty = styles[n]
                fig.add_artist(FancyBboxPatch(
                    (x, base - 0.08 * em), sw, sw, boxstyle="round,pad=0,rounding_size=0.012",
                    transform=fig.dpi_scale_trans, facecolor=sty["color"], hatch=sty["hatch"], linewidth=0,
                    edgecolor=darker(sty["color"]) if sty["hatch"] else "none"))
                fig.text(x + sw + tpad, base, lab, transform=fig.dpi_scale_trans, ha="left", va="baseline",
                         fontsize=fs_leg, color=INK)
            y -= (math.ceil(M / ncol) - 1) * row_h + 1.2 * em + 0.16

        ax = fig.add_axes((AX_LEFT / W, AX_BOTTOM / H, (W - AX_LEFT - AX_RIGHT) / W, (y - AX_BOTTOM) / H))
        ax.set_facecolor(SURFACE)
        ax.set_xlim(-0.5, G - 0.5)
        box = ax.get_window_extent(rend)
        ppx = box.width / G                               # device px per x unit

        # Bar geometry: <= 24 CSS px wide, 2 CSS px surface gap between neighbours.
        gap = 2 * PX / ppx
        slot = min(0.84 / M, (24 + 2) * PX / ppx)
        bw = slot - gap
        bw_px, slot_px = bw * ppx, slot * ppx

        # Value labels: horizontal when they fit the slot, otherwise vertical.
        fs_val = 8 if slot_px >= 22 else 7.5
        texts = [short_num(100 * methods[n]["sets"][s]["mean"]) for n in names for s in sets
                 if s in methods[n]["sets"]]
        probe = fig.text(0, 0, max(texts, key=len), fontsize=fs_val)
        tw, th = probe.get_window_extent(rend).width, probe.get_window_extent(rend).height
        probe.remove()
        rotate = tw + 2 > slot_px
        show_values = th <= slot_px + 2
        if not show_values:
            warn("cột quá hẹp, không ghi số trên cột (xem bảng markdown)")

        dot_px = min(max(0.4 * bw_px, 7.0), 10.0)       # small: 10 seeds must not hide the bar
        dot_pt = dot_px * 72 / DPI
        pad_pt = 2.5

        # Headroom above 100 % so a label on a full bar stays inside the plot.
        label_px = (tw if rotate else th) if show_values else 0
        ext_px = (dot_px / 2 if multi else 0) + pad_pt * DPI / 72 + label_px + 3
        ytop = 100 * box.height / (box.height - ext_px)
        ax.set_ylim(0, ytop)
        ppy = box.height / ytop                           # device px per percentage point

        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.spines["bottom"].set_linewidth(0.9)
        ax.spines["bottom"].set_zorder(1.5)          # under the bars, over the grid
        ax.set_yticks(range(0, 101, 20))
        ax.set_yticklabels([str(v) for v in range(0, 101, 20)])
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, pad=6, labelsize=8.5, labelcolor=INK_2)
        ax.tick_params(axis="x", length=0, pad=6, labelsize=10, labelcolor=INK)
        ax.set_ylabel(info["axis"], color=INK_2, fontsize=9, labelpad=8)
        ax.set_xticks(range(G))
        ax.set_xticklabels([set_label(s) for s in sets])
        for gi, s in enumerate(sets):
            cnt = maps_count(methods, names, s)
            if cnt:
                ax.annotate(cnt, xy=(gi, 0), xycoords=("data", "axes fraction"), xytext=(0, -21),
                            textcoords="offset points", ha="center", va="top", fontsize=8, color=MUTED)
        ax.annotate(f"Tập {scope(sets)}", xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -37),
                    textcoords="offset points", ha="center", va="top", fontsize=9, color=INK_2)

        rx, ry = 4 * PX / ppx, 4 * PX / ppy               # 4 CSS px rounded data end
        kappa = 0.5523

        def bar_path(x0: float, h: float) -> MPath:
            cx, cy = min(rx, bw / 2), min(ry, h)
            verts = [(x0, 0), (x0, h - cy),
                     (x0, h - cy + kappa * cy), (x0 + cx - kappa * cx, h), (x0 + cx, h),
                     (x0 + bw - cx, h),
                     (x0 + bw - cx + kappa * cx, h), (x0 + bw, h - cy + kappa * cy), (x0 + bw, h - cy),
                     (x0 + bw, 0), (x0, 0)]
            codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO,
                     MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CLOSEPOLY]
            return MPath(verts, codes)

        for gi, s in enumerate(sets):
            left = gi - (M * slot - gap) / 2
            for mi, n in enumerate(names):
                x0 = left + mi * slot
                xc = x0 + bw / 2
                st = methods[n]["sets"].get(s)
                if st is None:   # no data: a dash on the baseline, not a zero bar
                    ax.annotate("–", (xc, 0), xytext=(0, 2), textcoords="offset points", ha="center",
                                va="bottom", fontsize=8, color=MUTED, annotation_clip=False)
                    continue
                sty = styles[n]
                mean, lo, hi = 100 * st["mean"], 100 * st["min"], 100 * st["max"]
                h = max(mean, 2 * PX / ppy)    # a 0 still shows a 2 px stub in the method's colour
                ax.add_patch(PathPatch(bar_path(x0, h), facecolor=sty["color"], linewidth=0, zorder=2,
                                       hatch=sty["hatch"],
                                       edgecolor=darker(sty["color"]) if sty["hatch"] else "none"))
                top = mean
                vals = [100 * v for v in st.get("per_seed", {}).values()]
                if st["n_seeds"] >= 2 and hi > lo:
                    ax.vlines(xc, lo, hi, color=INK, linewidth=0.9, zorder=3)
                    if len(vals) < 2:   # no per-seed values: caps mark the ends instead of dots
                        cap = min(0.5 * bw, 10 * PX / ppx) / 2
                        ax.hlines([lo, hi], xc - cap, xc + cap, color=INK, linewidth=0.9, zorder=3)
                    top = max(top, hi)
                if len(vals) >= 2:
                    offs = swarm([v * ppy for v in vals], dot_px, max(0.0, (slot_px - dot_px) / 2))
                    ax.scatter([xc + o / ppx for o in offs], vals, s=dot_pt ** 2, c=INK, edgecolors=SURFACE,
                               linewidths=0.9, zorder=4, clip_on=False)
                    top = max(top, max(vals))
                if show_values:
                    lift = pad_pt + (dot_pt / 2 if len(vals) >= 2 else 0)
                    ax.annotate(short_num(mean), (xc, top), xytext=(0, lift), textcoords="offset points",
                                ha="center", va="bottom", rotation=90 if rotate else 0, fontsize=fs_val,
                                color=INK_2, zorder=5, annotation_clip=False)

        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=DPI, facecolor=SURFACE)
        plt.close(fig)


# ── Ảnh bảng và trang HTML ───────────────────────────────────────────────

def write_table_and_html(metric: str, methods: dict, names: list[str], sets: list[str], styles: dict,
                         chart: Path, title: str, source: str) -> list[Path]:
    """Ảnh bảng (<tên>_bang.png) và trang HTML (<tên>.html) cạnh biểu đồ `chart`."""
    from src.visual_report import MethodRow, heat_table_png, html_page, is_baseline, outputs_for
    info = METRICS[metric]
    table_path, _, html_path = outputs_for(chart.with_suffix(""))
    subs = {}
    for s in sets:
        subs[s] = "trung bình 4 cấp" if s == MEAN_SET else maps_count(methods, names, s)
    rows = []
    for n in names:
        m = methods[n]
        rows.append(MethodRow(n, styles[n]["color"], styles[n]["hatch"], is_baseline(n), seeds_of(methods, n, sets),
                              {s: m["sets"][s] for s in sets if s in m["sets"]}))
    subtitle = (f"{info['better']}. Mỗi ô: trung bình theo seed (%); dòng nhỏ: seed thấp nhất–cao nhất "
                "(chỉ khi có từ 2 seed).")
    notes = ["Ô càng đậm giá trị càng lớn. NEAT trước, baseline sau, cùng thứ tự với các cột của biểu đồ; "
             "ô màu cạnh tên là màu cột của phương pháp đó trên biểu đồ (gạch chéo: màu lặp lại).",
             f"Số liệu: {source}." if source else ""]
    heat_table_png(table_path, f"{info['name']} trên các tập {scope(sets)}", subtitle, sets, subs, rows,
                   [n for n in notes if n])
    html_page(html_path, title, subtitle, [("Biểu đồ", chart), ("Bảng số liệu", table_path)],
              notes=[n for n in notes if n], source="scripts/plot_benchmark.py", files=[chart, table_path])
    return [table_path, html_path]


# ── Main ─────────────────────────────────────────────────────────────────

def split_sets(text: str) -> list[str]:
    return list(dict.fromkeys(s.strip().strip("/") for s in text.split(",") if s.strip()))


def main():
    from src.console import utf8_stdout
    utf8_stdout()
    ap = argparse.ArgumentParser(description="Vẽ biểu đồ cột nhóm so sánh các phương pháp trên các tập test")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--runs", nargs="+", metavar="TÊN=THƯ_MỤC",
                     help="mỗi phương pháp một mục; THƯ_MỤC chứa seed_*/test_results.json")
    src.add_argument("--from-json", metavar="FILE", help="vẽ lại từ file JSON do --json ghi")
    ap.add_argument("--sets", default=None,
                    help="tập cần vẽ, cách nhau dấu phẩy (mặc định: " + ",".join(DEFAULT_SETS) + ")")
    ap.add_argument("--metric", choices=list(METRICS), default=None, help="SR (mặc định), CR, SPL, timeout")
    ap.add_argument("--out", default=None, help="file PNG (mặc định output/benchmark_<chỉ số>.png)")
    ap.add_argument("--json", default=None, help="ghi số liệu đã gom ra file JSON")
    ap.add_argument("--md", default=None, help="ghi bảng markdown ra file")
    ap.add_argument("--title", default=None, help="tiêu đề biểu đồ (mặc định theo chỉ số)")
    ap.add_argument("--no-visual", action="store_true",
                    help="không ghi ảnh bảng (<tên>_bang.png) và trang HTML (<tên>.html) cạnh biểu đồ")
    a = ap.parse_args()

    if a.from_json:
        agg = load_aggregate(Path(a.from_json))
        metric = agg["metric"]
        if a.metric and a.metric != metric:
            sys.exit(f"Lỗi: {a.from_json} chỉ có số liệu {metric}, không có {a.metric}. "
                     f"Muốn vẽ {a.metric} hãy chạy lại với --runs.")
        wanted = split_sets(a.sets) if a.sets else (agg.get("sets") or list(DEFAULT_SETS))
        title_arg = a.title if a.title is not None else agg.get("title")
    else:
        metric = a.metric or "SR"
        runs = parse_runs(a.runs)
        agg = aggregate(runs, metric)
        wanted = split_sets(a.sets) if a.sets else list(DEFAULT_SETS)
        title_arg = a.title
    methods = agg["methods"]
    if not methods:
        sys.exit("Lỗi: không tìm thấy test_results.json nào. Kiểm tra lại đường dẫn trong --runs "
                 "(mỗi thư mục cần seed_*/test_results.json, do scripts/evaluate.py ghi).")
    sets, names = select(methods, wanted)
    if not sets or not names:
        have = sorted({s for m in methods.values() for s in m["sets"]}, key=natural_key)
        sys.exit(f"Lỗi: không có số liệu {metric} cho tập nào trong {', '.join(wanted)}. "
                 f"Các tập có trong kết quả: {', '.join(have) or '(không có)'}.")

    # Colour follows the method, not what happens to be loaded: its position in --runs
    # (a method still training keeps its slot), or the colour stored in the JSON.
    if a.from_json:
        fallback = assign_styles(list(methods))
        styles = {n: methods[n].get("style") or fallback[n] for n in methods}
    else:
        styles = assign_styles([n for n, _ in runs])
    table = markdown_table(metric, methods, names, sets)
    print(table)
    if a.md:
        Path(a.md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.md).write_text(table + "\n", encoding="utf-8")
        print(f"Đã ghi {a.md}", file=sys.stderr)

    if a.json:
        doc = {"metric": metric, "metric_name": METRICS[metric]["name"],
               "unit": "tỉ lệ 0–1 như test_results.json; biểu đồ và bảng nhân 100 thành %",
               "sets": sets, "set_labels": {s: set_label(s) for s in sets},
               "method_order": list(methods), "methods": {}}
        if title_arg is not None:
            doc["title"] = title_arg
        for n, m in methods.items():
            # chỉ giữ các tập đã chọn (--sets), theo đúng thứ tự đó
            kept = {s: m["sets"][s] for s in sets if s in m["sets"]}
            doc["methods"][n] = {"dir": portable_dir(m["dir"]), "seeds": m["seeds"],
                                 "color": styles[n]["color"], "hatch": styles[n]["hatch"], "sets": kept}
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print(f"Đã ghi {a.json}", file=sys.stderr)

    title = title_arg if title_arg is not None else f"{METRICS[metric]['name']} trên các tập {scope(sets)}"
    out = Path(a.out or f"output/benchmark_{metric}.png")
    plot_chart(metric, methods, names, sets, styles, out, title)
    if a.no_visual:
        print(f"Đã ghi {out}", file=sys.stderr)
        return

    from src.visual_report import print_written, warn_failed
    source = Path(a.from_json).as_posix() if a.from_json else runs_source(runs, methods)
    written = [out]
    try:
        written += write_table_and_html(metric, methods, names, sets, styles, out, title, source)
    except Exception as e:  # noqa: BLE001  (ảnh bảng là phần thêm: biểu đồ đã ghi xong)
        warn_failed("ảnh bảng và trang HTML", e)
    print_written(written, stream=sys.stderr)


def runs_source(runs: list[tuple[str, Path]], methods: dict) -> str:
    """Nguồn số liệu ghi dưới ảnh bảng khi gom bằng --runs (đường dẫn rút gọn như file --json)."""
    dirs = [Path(p) for n, p in runs if n in methods]
    parents = {d.resolve().parent for d in dirs}
    if len(parents) == 1:
        parent = parents.pop()
        try:
            where = parent.relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            where = ".../" + parent.name          # như mục "dir" của file --json: không mang tên người dùng
        return f"seed_*/test_results.json trong {where}/<phương pháp>/"
    return f"seed_*/test_results.json của {len(dirs)} thư mục trong --runs"


if __name__ == "__main__":
    main()
