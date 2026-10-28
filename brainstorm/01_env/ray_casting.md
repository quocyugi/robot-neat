# THIẾT KẾ RAY-CASTING 12 TIA LIDAR & QUAN SÁT ROBOT (15D)
> **Ngày**: 2026-10-02  
> **Trạng thái**: [DECIDED]  
> **Liên quan**: overview.md phần 2.1 & 2.2  

---

## 1. Câu hỏi gốc tôi đã hỏi LLM
> "Tối ưu thuật toán Ray-casting 12 tia LiDAR bằng NumPy vectorization & kiểm tra tính hợp lệ của 15D input."

---

## 2. Tóm tắt kết luận rút ra (Key Takeaways)
- **Chuẩn hóa khoảng cách LiDAR**: Chuyển từ $d/R_{\max}$ sang **Proximity Index: $s_i = 1.0 - (d_i / R_{\max})$**. Robot nhận tín hiệu 0.0 khi an toàn và tiệm cận 1.0 khi cận kề nguy hiểm, phù hợp phản xạ nơ-ron sinh học.
- **Bố trí 12 tia**: Giữ nguyên $360^\circ$ phân bố đều (cách nhau $30^\circ$) để robot có tầm nhìn phía sau khi lùi và quay trong góc hẹp.
- **Ray-casting Vectorized**: Dùng thuật toán tích có hướng 2D (Determinant / 2D Cross Product) với cơ chế broadcasting của NumPy trên ma trận `(12, M)`. Đạt tốc độ $> 20,000$ FPS trên CPU, loại bỏ hoàn toàn vòng lặp Python.
- **Xử lý vật cản tròn**: Xấp xỉ hình tròn thành đa giác đều (bát giác 8 cạnh) để đồng nhất toàn bộ môi trường thành mảng đoạn thẳng `(M, 2, 2)`, tránh rẽ nhánh if/else tính toán.

---

## 3. Chi tiết Kỹ thuật & Code hoàn chỉnh

### 3.1. Thuật toán Giao điểm Đoạn thẳng - Đoạn thẳng (Segment - Segment)
Mỗi tia $i$ xuất phát từ $\mathbf{p} = (x_r, y_r)$ theo vector $\mathbf{r}_i = R_{\max} (\cos\alpha_i, \sin\alpha_i)$.  
Mỗi đoạn vật cản $j$ nối từ $\mathbf{A}_j$ đến $\mathbf{B}_j$ với vector chỉ phương $\mathbf{s}_j = \mathbf{B}_j - \mathbf{A}_j$.

Tham số điểm giao:
$$t_{i, j} = \frac{(\mathbf{A}_j - \mathbf{p}) \times \mathbf{s}_j}{\mathbf{r}_i \times \mathbf{s}_j}, \quad u_{i, j} = \frac{(\mathbf{A}_j - \mathbf{p}) \times \mathbf{r}_i}{\mathbf{r}_i \times \mathbf{s}_j}$$

Điều kiện giao nhau hợp lệ: $0 \le t \le 1$, $0 \le u \le 1$, $|\mathbf{r}_i \times \mathbf{s}_j| > 10^{-8}$.

### 3.2. Code Triển khai Chuẩn (Ready for `src/`)

```python
import numpy as np

class FastLiDAR2D:
    def __init__(self, num_rays: int = 12, max_range: float = 3.0):
        self.num_rays = num_rays
        self.max_range = max_range
        # 12 góc cách đều: 0, 30, 60, ..., 330 độ
        self.ray_angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

    def cast_rays(self, robot_pos: np.ndarray, robot_theta: float, segments: np.ndarray) -> np.ndarray:
        """
        robot_pos: array shape (2,) -> tọa độ (x, y) robot
        robot_theta: float -> hướng xe (radian)
        segments: array shape (M, 2, 2) -> danh sách M đoạn thẳng vật cản
        Trả về: proximity array shape (12,), giá trị [0, 1] (gần = 1.0, xa = 0.0)
        """
        angles = robot_theta + self.ray_angles
        r = np.stack([np.cos(angles), np.sin(angles)], axis=-1) * self.max_range  # (12, 2)

        A = segments[:, 0, :]  # (M, 2)
        B = segments[:, 1, :]  # (M, 2)
        s = B - A              # (M, 2)

        # Broadcasting (12, M)
        r_exp = r[:, np.newaxis, :]  # (12, 1, 2)
        s_exp = s[np.newaxis, :, :]  # (1, M, 2)

        r_cross_s = r_exp[..., 0] * s_exp[..., 1] - r_exp[..., 1] * s_exp[..., 0]  # (12, M)
        p_to_A = (A - robot_pos)[np.newaxis, :, :]                                # (1, M, 2)

        pA_cross_s = p_to_A[..., 0] * s_exp[..., 1] - p_to_A[..., 1] * s_exp[..., 0]
        pA_cross_r = p_to_A[..., 0] * r_exp[..., 1] - p_to_A[..., 1] * r_exp[..., 0]

        parallel_mask = np.abs(r_cross_s) < 1e-8
        safe_div = np.where(parallel_mask, 1e-8, r_cross_s)

        t = pA_cross_s / safe_div
        u = pA_cross_r / safe_div

        valid = (~parallel_mask) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
        t_valid = np.where(valid, t, 1.0)

        min_t = np.min(t_valid, axis=1)  # shape (12,)
        
        # PROXIMITY INDEX: 0 (an toàn) -> 1 (va chạm)
        proximity = 1.0 - min_t
        return proximity

def circle_to_segments(center: np.ndarray, radius: float, num_sides: int = 8) -> np.ndarray:
    """Xấp xỉ hình tròn thành đa giác đều (bát giác 8 cạnh)"""
    angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    pts = center + radius * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    pts_next = np.roll(pts, -1, axis=0)
    return np.stack([pts, pts_next], axis=1)  # shape (num_sides, 2, 2)
```

---

## 4. Quyết định của tôi
- [x] **Chọn Phương án B cho LiDAR**: Dùng $s_i = 1.0 - (d_i / R_{\max})$. Lý do: Giúp bias nơ-ron không bị kích thích ảo khi không gian trống, mô hình dễ hội tụ hơn ở những thế hệ đầu.
- [x] **Chọn Phương án A cho vật cản tròn**: Xấp xỉ hình tròn bằng đa giác đều 8 cạnh. Lý do: Đưa tất cả về một mảng `segments` duy nhất `(M, 2, 2)`, giải bằng đúng 1 câu lệnh NumPy broadcasting, không phân nhánh điều kiện, tốc độ tối đa.

---

## 5. TODO tiếp theo
- [ ] Tích hợp class `FastLiDAR2D` vào `RobotNavEnv` trong `src/env/`.
- [ ] Brainstorm tiếp về Pipeline sinh bản đồ tự động & kiểm tra liên thông $A^*$ (`01_env/map_generator.md`).
