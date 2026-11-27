"""
Vectorised 2-D LiDAR Ray-Casting
=================================
Casts *num_rays* equally-spaced rays from the robot centre and returns
a **proximity index** for each ray (0 = safe, 1 = touching obstacle).

All geometry is reduced to line-segment intersection so that both
rectangular walls and circular obstacles (approximated as regular
polygons) share the same code path — no branching.

Reference
---------
- brainstorm/01_env/ray_casting.md  (Decision #1, #2, #3)

Performance
-----------
> 20 000 FPS on a single CPU core (12 rays × ~200 segments).

Hai kiểu cảm biến (chọn bằng NEAT_LIDAR_MODE, xem src/env/kinematics.py)
-------------------------------------------------------------------------
- "rays" (mặc định): cast_rays, N tia điểm.
- "sector": cast_sectors. Đầu vào i = 1 - d_i / R (R = 3 m, cắt về [0, 1], float32),
  d_i = khoảng cách Euclid nhỏ nhất từ tâm robot tới MỌI điểm vật cản nằm trong hình
  quạt ĐÓNG [theta_i - pi/N, theta_i + pi/N] (cả hai cạnh thuộc hình quạt; điểm nằm
  đúng trên cạnh chung của hai hình quạt kề nhau được tính cho cả hai). Bằng giới hạn
  của vô số tia trong hình quạt (điểm gần nhất của một hình quạt không bao giờ bị che),
  nên vật cản mảnh lọt giữa hai tia điểm vẫn được thấy. Tính giải tích, vector hóa trên
  (N, số đoạn): không bắn tia dày trong vòng lặp lượt chạy.
scan() gọi đúng hàm theo self.mode; RobotNavEnv gọi scan().

Bố cục hình quạt không đều (layout, NEAT_SECTOR_LAYOUT ở src/env/kinematics.py)
-------------------------------------------------------------------------------
Mọi đường chạy (sector_distances ở đây, HostGeometry._sectors ở src/batch/sim.py, kernel "sector"
ở src/batch/gpu.py) đọc cạnh quạt từ cùng mảng sector_edges: quạt i nằm giữa cạnh i và cạnh
(i + 1) mod N. Bố cục "e32" chỉ thay mảng này (cạnh dưới, độ): quạt j = 0..29 là
[-135 + 9j; -135 + 9(j + 1)] (270 độ phía trước, từ phải sang trái; hướng trước 0 độ là cạnh giữa
quạt 14 và 15), quạt 30 là [135; 180], quạt 31 là [180; 225] = [-180; -135]. Phép thử "trong quạt"
của sector_distances đúng khi mỗi quạt mở không quá pi: check_sector_edges kiểm điều đó và việc
các quạt khép đúng một vòng. ray_angles của bố cục này là hướng giữa từng quạt (chỉ để vẽ).
"""

from __future__ import annotations

import numpy as np

from src.env.kinematics import LIDAR_MODE, LIDAR_MODES, LIDAR_NUM_RAYS, LIDAR_MAX_RANGE, SECTOR_LAYOUTS


def sector_layout(num_rays: int, layout: str) -> tuple[np.ndarray, np.ndarray]:
    """(cạnh dưới của từng quạt, hướng giữa từng quạt) theo bố cục không đều; rad, float64."""
    if layout == "e32":
        if num_rays != 32:
            raise ValueError(f"bố cục e32 cần 32 quạt (num_rays = {num_rays})")
        lower = [-135.0 + 9.0 * j for j in range(31)] + [180.0]        # 30 quạt 9 độ, quạt [135; 180], [180; 225]
        mid = [-130.5 + 9.0 * j for j in range(30)] + [157.5, -157.5]
        return np.radians(np.array(lower)), np.radians(np.array(mid))
    raise ValueError(f"layout={layout!r}: chỉ nhận {SECTOR_LAYOUTS}")


def check_sector_edges(edges: np.ndarray) -> None:
    """Báo lỗi nếu cạnh quạt không tăng dần, có quạt mở quá pi, hoặc các quạt không khép đúng một vòng."""
    e = np.asarray(edges, dtype=np.float64)
    if e.ndim != 1 or e.size < 2:
        raise ValueError("cần ít nhất 2 cạnh quạt")
    width = np.diff(np.concatenate((e, e[:1] + 2.0 * np.pi)))   # quạt cuối quấn về cạnh 0
    if not np.all(width > 0.0):
        raise ValueError("cạnh quạt phải tăng dần và quạt cuối kết thúc ở cạnh 0 + 2 pi")
    if not np.all(width <= np.pi):
        raise ValueError(f"có quạt mở quá pi ({float(np.degrees(width.max())):.3f} độ)")
    if abs(float(width.sum()) - 2.0 * np.pi) > 1e-9:
        raise ValueError("các quạt không khép đúng một vòng")


class FastLiDAR2D:
    """Vectorised 2-D LiDAR scanner using segment–segment intersection."""

    def __init__(
        self,
        num_rays: int = LIDAR_NUM_RAYS,
        max_range: float = LIDAR_MAX_RANGE,
        mode: str = LIDAR_MODE,
        layout: str = "",
    ):
        self.num_rays = num_rays
        self.max_range = max_range
        # Pre-compute local ray angles: 0, 30°, 60°, … , 330°
        self.ray_angles: np.ndarray = np.linspace(
            0, 2 * np.pi, num_rays, endpoint=False
        )
        if mode not in LIDAR_MODES:
            raise ValueError(f"mode={mode!r}: chỉ nhận {LIDAR_MODES}")
        if mode == "sector" and num_rays < 2:
            raise ValueError("kiểu 'sector' cần num_rays >= 2")
        if layout not in SECTOR_LAYOUTS:
            raise ValueError(f"layout={layout!r}: chỉ nhận {SECTOR_LAYOUTS}")
        if layout and mode != "sector":
            raise ValueError(f"bố cục quạt {layout!r} chỉ dùng với mode='sector'")
        self.mode = mode
        self.layout = layout
        # Cạnh dưới của hình quạt i ở góc 2*pi*i/N - pi/N (so với hướng robot);
        # cạnh trên của hình quạt i là cạnh dưới của hình quạt i + 1.
        # Chỉ tính khi num_rays > 0: kiểu "rays" dựng được với mọi num_rays, kể cả 0
        # (không chia cho 0; sector_edges khi đó là mảng rỗng như ray_angles).
        if layout:
            self.sector_edges, self.ray_angles = sector_layout(num_rays, layout)
            check_sector_edges(self.sector_edges)
        elif num_rays > 0:
            self.sector_edges: np.ndarray = self.ray_angles - np.pi / num_rays
        else:
            self.sector_edges = np.empty(0)

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #
    def scan(
        self,
        robot_pos: np.ndarray,
        robot_theta: float,
        segments: np.ndarray,
    ) -> np.ndarray:
        """Độ gần (num_rays,) theo kiểu đã chọn: tia điểm (mặc định) hoặc hình quạt."""
        if self.mode == "sector":
            return self.cast_sectors(robot_pos, robot_theta, segments)
        return self.cast_rays(robot_pos, robot_theta, segments)

    def cast_rays(
        self,
        robot_pos: np.ndarray,
        robot_theta: float,
        segments: np.ndarray,
    ) -> np.ndarray:
        """Cast all rays and return proximity indices.

        Parameters
        ----------
        robot_pos : ndarray, shape (2,)
            Robot centre (x, y).
        robot_theta : float
            Robot heading (radians, CCW from +x).
        segments : ndarray, shape (M, 2, 2)
            Each row is a line segment [[Ax, Ay], [Bx, By]].

        Returns
        -------
        proximity : ndarray, shape (num_rays,)
            Values in [0, 1].  0 → no obstacle within range,
            1 → obstacle at robot centre.
        """
        # Global ray angles
        angles = robot_theta + self.ray_angles                    # (R,)

        # Ray direction vectors scaled to max range
        r = (
            np.stack([np.cos(angles), np.sin(angles)], axis=-1)
            * self.max_range
        )                                                         # (R, 2)

        A = segments[:, 0, :]                                     # (M, 2)
        B = segments[:, 1, :]                                     # (M, 2)
        s = B - A                                                 # (M, 2)

        # Broadcast to (R, M, 2)
        r_exp = r[:, np.newaxis, :]                               # (R, 1, 2)
        s_exp = s[np.newaxis, :, :]                               # (1, M, 2)

        # 2-D cross products  (R, M)
        r_cross_s = (
            r_exp[..., 0] * s_exp[..., 1]
            - r_exp[..., 1] * s_exp[..., 0]
        )

        p_to_A = (A - robot_pos)[np.newaxis, :, :]               # (1, M, 2)

        pA_cross_s = (
            p_to_A[..., 0] * s_exp[..., 1]
            - p_to_A[..., 1] * s_exp[..., 0]
        )
        pA_cross_r = (
            p_to_A[..., 0] * r_exp[..., 1]
            - p_to_A[..., 1] * r_exp[..., 0]
        )

        # Avoid division by zero for parallel lines
        parallel = np.abs(r_cross_s) < 1e-8
        safe_denom = np.where(parallel, 1e-8, r_cross_s)

        t = pA_cross_s / safe_denom                               # (R, M)
        u = pA_cross_r / safe_denom                               # (R, M)

        # Valid intersection: 0 ≤ t ≤ 1  AND  0 ≤ u ≤ 1  AND  not parallel
        valid = (~parallel) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)

        t_valid = np.where(valid, t, 1.0)     # miss → full range
        min_t = np.min(t_valid, axis=1)        # (R,) — closest hit per ray

        # Proximity Index:  0 (safe)  →  1 (touching)
        proximity = 1.0 - min_t
        return proximity.astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Cảm biến hình quạt (NEAT_LIDAR_MODE = "sector")                     #
    # ------------------------------------------------------------------ #
    def cast_sectors(
        self,
        robot_pos: np.ndarray,
        robot_theta: float,
        segments: np.ndarray,
    ) -> np.ndarray:
        """Độ gần theo hình quạt: 1 - min(d_i, R) / R, float32, shape (num_rays,).

        d_i xem sector_distances(); không có điểm vật cản nào trong tầm R thì 0.
        """
        d = self.sector_distances(robot_pos, robot_theta, segments)
        proximity = 1.0 - np.minimum(d, self.max_range) / self.max_range
        return proximity.astype(np.float32)

    def sector_distances(
        self,
        robot_pos: np.ndarray,
        robot_theta: float,
        segments: np.ndarray,
    ) -> np.ndarray:
        """d_i (m, float64): khoảng cách nhỏ nhất từ tâm robot p tới điểm vật cản trong
        hình quạt ĐÓNG i = [theta_i - pi/N, theta_i + pi/N]; inf nếu không có điểm nào.

        Với đoạn A→B, q(t) = A - p + t (B - A), t ∈ [0, 1]. Điểm q ở trong hình quạt
        (góc mở 2*pi/N <= pi) khi cross(u_lo, q) >= 0 và cross(q, u_hi) >= 0 (bất đẳng
        thức không chặt: hình quạt đóng); cả hai tuyến tính theo t nên cho một khoảng
        đóng [t_lo, t_hi] ⊂ [0, 1]. |q(t)|^2 lồi theo t: kẹp tham số điểm gần nhất (không
        ràng buộc) vào khoảng đó rồi lấy khoảng cách. Khoảng rỗng = đoạn không có điểm
        nào trong hình quạt. Cạnh chung của hai hình quạt kề nhau dùng cùng một bất
        phương trình đổi dấu (cùng nghiệm r_k = -ca/cs, cùng phép so sánh không chặt),
        nên mọi điểm thuộc ít nhất một hình quạt, và điểm nằm đúng trên cạnh chung được
        tính cho CẢ HAI hình quạt. "Đúng trên cạnh" hiểu theo chính các phép tính dấu
        phẩy động ở đây: đoạn chỉ nằm dọc theo một cạnh trong phạm vi sai số làm tròn
        thì làm tròn quyết định phần nào của đoạn thuộc hình quạt nào (một hình quạt có
        thể đọc một điểm xa hơn của cùng đoạn), còn giá trị nhỏ nhất của hai hình quạt
        kề nhau vẫn đúng.
        """
        n = self.num_rays
        if segments.shape[0] == 0:
            return np.full(n, np.inf)
        edges = robot_theta + self.sector_edges                   # (N,)
        c, s = np.cos(edges), np.sin(edges)
        # N + 1 cạnh: cạnh N chính là cạnh 0 (cùng giá trị) để hình quạt N - 1 khép vòng
        ex = np.concatenate((c, c[:1]))[:, np.newaxis]            # (N+1, 1)
        ey = np.concatenate((s, s[:1]))[:, np.newaxis]

        A = segments[:, 0, :]                                     # (M, 2)
        ax = A[:, 0] - robot_pos[0]                               # (M,)  a = A - p
        ay = A[:, 1] - robot_pos[1]
        sx = segments[:, 1, 0] - A[:, 0]                          # (M,)  s = B - A
        sy = segments[:, 1, 1] - A[:, 1]

        # f_k(t) = cross(e_k, q(t)) = ca[k] + t cs[k]                   (N+1, M)
        # Hình quạt i: f_i(t) >= 0 (cạnh dưới e_i) và cross(q, e_{i+1}) = -f_{i+1}(t) >= 0
        ca = ex * ay - ey * ax
        cs = ex * sy - ey * sx
        pos = cs > 0.0
        neg = cs < 0.0
        flat = cs == 0.0                                          # đoạn song song đúng với cạnh: f_k không đổi
        any_flat = bool(flat.any())
        r = -ca / (np.where(flat, 1.0, cs) if any_flat else cs)   # nghiệm của f_k(t) = 0
        # f_i >= 0: t >= r_i nếu cs_i > 0, t <= r_i nếu cs_i < 0, cần ca_i >= 0 nếu cs_i = 0
        # f_{i+1} <= 0: t <= r_{i+1} nếu cs > 0, t >= r_{i+1} nếu cs < 0, cần ca_{i+1} <= 0 nếu cs = 0
        t_lo = np.maximum(np.maximum(np.where(pos[:-1], r[:-1], -np.inf), np.where(neg[1:], r[1:], -np.inf)), 0.0)
        t_hi = np.minimum(np.minimum(np.where(neg[:-1], r[:-1], np.inf), np.where(pos[1:], r[1:], np.inf)), 1.0)
        inside = t_lo <= t_hi                                     # (N, M)
        if any_flat:
            inside &= ~((flat[:-1] & (ca[:-1] < 0.0)) | (flat[1:] & (ca[1:] > 0.0)))

        # |q(t)|^2 = h2 + ss (t - t*)^2: h2 = khoảng cách^2 tới đường thẳng AB, t* = chân
        # đường vuông góc; kẹp t* vào [t_lo, t_hi]. Đoạn suy biến (A = B): t* = 0, h2 = |a|^2.
        ss = sx * sx + sy * sy                                    # (M,)
        nz = ss > 0.0
        ss1 = np.where(nz, ss, 1.0)
        t_star = np.where(nz, -(ax * sx + ay * sy) / ss1, 0.0)
        cr = ax * sy - ay * sx
        h2 = np.where(nz, cr * cr / ss1, ax * ax + ay * ay)
        dt = np.minimum(np.maximum(t_star, t_lo), t_hi) - t_star  # (N, M)
        d2 = np.where(inside, h2 + ss * (dt * dt), np.inf)
        return np.sqrt(np.min(d2, axis=1))


# ------------------------------------------------------------------ #
#  Geometry helpers                                                    #
# ------------------------------------------------------------------ #

def circle_to_segments(
    center: np.ndarray,
    radius: float,
    num_sides: int = 8,
) -> np.ndarray:
    """Approximate a circle as a regular polygon of *num_sides* edges.

    Returns
    -------
    segments : ndarray, shape (num_sides, 2, 2)
    """
    angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    pts = center + radius * np.stack(
        [np.cos(angles), np.sin(angles)], axis=-1
    )
    pts_next = np.roll(pts, -1, axis=0)
    return np.stack([pts, pts_next], axis=1)


def arena_wall_segments(w: float = 10.0, h: float = 10.0) -> np.ndarray:
    """Return the 4 boundary-wall segments of the arena.

    Returns
    -------
    segments : ndarray, shape (4, 2, 2)
    """
    return np.array([
        [[0, 0], [w, 0]],   # bottom
        [[w, 0], [w, h]],   # right
        [[w, h], [0, h]],   # top
        [[0, h], [0, 0]],   # left
    ], dtype=np.float64)
