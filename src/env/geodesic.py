"""
Geodesic distance-to-go field (sửa lỗi L26)
============================================
Khoảng cách theo đường đi thật từ mọi ô lưới 5 cm tới đích, vật cản phình thêm
0,30 m (bằng INFLATION_RADIUS của bộ kiểm tra A*). Đo tiến độ trên trường này
thì việc lao vào một ngõ cụt "hướng về đích" không còn được cộng điểm, còn đo
theo đường chim bay (code cũ) thì có.

Trường được tính một lần cho mỗi bản đồ (Dijkstra trên lưới 200 x 200, khoảng
0,1–0,5 s) và lưu đệm trên đĩa, mặc định ở ``output/geo_cache/`` (đổi bằng biến
môi trường ``NEAT_GEO_CACHE``), để mọi tiến trình và mọi lần chạy dùng lại.
Thuật toán giống hệt bản đã dùng cho các kết quả trong báo cáo (mục 8.10, 8.13).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np

GEO_RES: float = 0.05          # m, cạnh ô lưới
GEO_INFLATION: float = 0.30    # m, = src.maps.validator.INFLATION_RADIUS
DEFAULT_CACHE_DIR: str = os.environ.get("NEAT_GEO_CACHE", os.path.join("output", "geo_cache"))


def compute_geodesic(
    segments,
    goal,
    bounds=(10.0, 10.0),
    res: float = GEO_RES,
    inflation: float = GEO_INFLATION,
) -> np.ndarray:
    """Trường khoảng cách tới đích (mét), mảng float32 shape (ny, nx), hàng = trục y.

    Ô nằm trong vùng vật cản phình không đi được; ô không tới được đích lấy giá trị
    của ô đi được gần nhất cộng khoảng cách tới ô đó.
    """
    from scipy import ndimage
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    w, h = float(bounds[0]), float(bounds[1])
    nx, ny = int(round(w / res)), int(round(h / res))
    xs = (np.arange(nx) + 0.5) * res
    ys = (np.arange(ny) + 0.5) * res
    xx, yy = np.meshgrid(xs, ys)
    px, py = xx.ravel(), yy.ravel()

    occ = np.zeros(px.shape, dtype=bool)
    for seg in np.asarray(segments, dtype=np.float64):
        ax, ay, bx, by = seg[0, 0], seg[0, 1], seg[1, 0], seg[1, 1]
        sx, sy = bx - ax, by - ay
        ss = max(sx * sx + sy * sy, 1e-12)
        t = np.clip(((px - ax) * sx + (py - ay) * sy) / ss, 0.0, 1.0)
        occ |= (px - (ax + t * sx)) ** 2 + (py - (ay + t * sy)) ** 2 <= inflation ** 2
    occ = occ.reshape(ny, nx)

    idx = np.arange(ny * nx).reshape(ny, nx)
    edges = []
    for dr, dc, wgt in ((0, 1, res), (1, 0, res), (1, 1, res * math.sqrt(2)), (1, -1, res * math.sqrt(2))):
        ra, rb = slice(max(0, -dr), ny - max(0, dr)), slice(max(0, dr), ny - max(0, -dr))
        ca, cb = slice(max(0, -dc), nx - max(0, dc)), slice(max(0, dc), nx - max(0, -dc))
        ok = ~(occ[ra, ca] | occ[rb, cb])
        edges.append((idx[ra, ca][ok], idx[rb, cb][ok], np.full(int(ok.sum()), wgt)))
    i = np.concatenate([e[0] for e in edges])
    j = np.concatenate([e[1] for e in edges])
    wts = np.concatenate([e[2] for e in edges])
    graph = csr_matrix((wts, (i, j)), shape=(ny * nx, ny * nx))

    gx = min(max(int(goal[0] / res), 0), nx - 1)
    gy = min(max(int(goal[1] / res), 0), ny - 1)
    src = idx[gy, gx]
    if occ[gy, gx]:   # đích nằm trong vùng phình: xuất phát từ ô trống gần nhất
        fi = ndimage.distance_transform_edt(occ, return_distances=False, return_indices=True)
        src = idx[fi[0][gy, gx], fi[1][gy, gx]]
    d = dijkstra(graph, directed=False, indices=int(src)).reshape(ny, nx)
    bad = ~np.isfinite(d)
    if bad.any():
        di, fi = ndimage.distance_transform_edt(bad, return_distances=True, return_indices=True)
        d = d[fi[0], fi[1]] + di * res
    return d.astype(np.float32)


class GeodesicField:
    """Tra cứu khoảng cách còn lại tới đích tại một điểm (x, y)."""

    __slots__ = ("grid", "res")

    def __init__(self, grid: np.ndarray, res: float = GEO_RES):
        self.grid = grid
        self.res = res

    def at(self, x: float, y: float) -> float:
        ny, nx = self.grid.shape
        ix = min(max(int(x / self.res), 0), nx - 1)
        iy = min(max(int(y / self.res), 0), ny - 1)
        return float(self.grid[iy, ix])


def map_cache_key(map_data: dict) -> str:
    """Khóa đệm: map_id + băm của hình học, nên sửa bản đồ thì đệm cũ không bị dùng nhầm."""
    payload = json.dumps([map_data["segments"], map_data["goal"], map_data.get("bounds", [10.0, 10.0])])
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{map_data.get('map_id', 'map')}_{digest}"


def _cache_path(map_data: dict, cache_dir: Optional[str]) -> Optional[Path]:
    return Path(cache_dir) / f"{map_cache_key(map_data)}.npy" if cache_dir else None


def _save(grid: np.ndarray, path: Path) -> None:
    """Ghi nguyên tử (file tạm rồi đổi tên). Trên Windows, đổi tên thất bại nếu tiến trình
    khác đang ghi hoặc đọc đúng file đó; khi ấy file đã có người ghi nên chỉ cần bỏ file tạm."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        np.save(f, grid)
    try:
        os.replace(tmp, path)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass


def ensure_cached(map_data: dict, cache_dir: Optional[str] = DEFAULT_CACHE_DIR) -> None:
    """Tính và lưu đệm trường của bản đồ nếu chưa có (gọi ở tiến trình cha trước khi chia việc)."""
    path = _cache_path(map_data, cache_dir)
    if path is not None and not path.exists():
        _save(compute_geodesic(map_data["segments"], map_data["goal"], map_data.get("bounds", (10.0, 10.0))), path)


def get_geodesic(map_data: dict, cache_dir: Optional[str] = DEFAULT_CACHE_DIR) -> GeodesicField:
    """Trả trường geodesic của bản đồ; đọc từ đệm trên đĩa nếu có, nếu không thì tính rồi lưu."""
    grid = None
    path = _cache_path(map_data, cache_dir)
    if path is not None and path.exists():
        try:
            grid = np.load(path)
        except (OSError, ValueError):
            grid = None
    if grid is None:
        grid = compute_geodesic(map_data["segments"], map_data["goal"], map_data.get("bounds", (10.0, 10.0)))
        if path is not None:
            _save(grid, path)
    return GeodesicField(grid)
