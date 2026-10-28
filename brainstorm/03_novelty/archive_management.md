# QUẢN LÝ KHO LƯU TRỮ ARCHIVE, CẤU TRÚC CKDTREE & NGƯỠNG NẠP THÍCH NGHI
> **Ngày**: 2026-10-02  
> **Trạng thái**: [DECIDED]  
> **Liên quan**: overview.md phần 4.3  

---

## 1. Câu hỏi gốc tôi đã hỏi LLM
> "Thiết kế cơ chế quản lý kho lưu trữ Archive trong Novelty Search: chính sách nạp cá thể, ngưỡng thích nghi rho_min (Adaptive Novelty Threshold), tối ưu truy vấn bằng cKDTree và giới hạn dung lượng Archive."

---

## 2. Tóm tắt kết luận rút ra (Key Takeaways)
- **Archive là trí nhớ dài hạn**: Lưu giữ các hành vi độc đáo trong lịch sử tiến hóa, ngăn quần thể xoay vòng lặp lại các hành vi cũ (cycling behaviors).
- **Ngưỡng nạp thích nghi ($\rho_{\min}$)**: Khởi tạo $\rho_{\min} = 1.0\text{m}$. Tự động tăng $5\%$ khi có $> 4$ cá thể được nạp/thế hệ (siết chặt tiêu chuẩn), và giảm $5\%$ khi sau $15$ thế hệ không có cá thể nào được nạp (nới lỏng tiêu chuẩn để chống đóng băng).
- **Thăm dò ngẫu nhiên (1% Random Injection)**: Dù $\rho(x) \le \rho_{\min}$, vẫn có xác suất $1\%$ nạp ngẫu nhiên cá thể vào Archive để duy trì tính khám phá không thiên lệch.
- **Truy vấn siêu tốc với `cKDTree`**: Dùng cấu trúc KD-Tree từ thư viện C++ Scipy, thời gian truy vấn $k_{\text{nn}} = 15$ cho toàn bộ 150 cá thể chỉ mất $< 0.5\text{ms}$.
- **Trần dung lượng an toàn $M_{\max} = 2,000$**: Áp dụng cơ chế FIFO khi số lượng điểm vượt quá 2,000, đảm bảo tiêu thụ RAM cực thấp và thời gian rebuild KD-Tree luôn tức thời.

---

## 3. Mã Nguồn Hoàn Chỉnh Lớp `NoveltyArchive` (Ready for `src/`)

```python
import numpy as np
from scipy.spatial import cKDTree

class NoveltyArchive:
    def __init__(self, k_nn: int = 15, initial_threshold: float = 1.0, max_capacity: int = 2000):
        self.k_nn = k_nn
        self.threshold = initial_threshold
        self.max_capacity = max_capacity
        self.archive = []  # Danh sách các vector hành vi (x_final, y_final)
        self.stagnation_counter = 0

    def evaluate_population(self, population_behaviors: np.ndarray) -> np.ndarray:
        """
        Tính điểm novelty cho toàn bộ quần thể (N, 2)
        """
        N = len(population_behaviors)
        if len(self.archive) > 0:
            all_pts = np.vstack([population_behaviors, np.array(self.archive)])
        else:
            all_pts = population_behaviors

        tree = cKDTree(all_pts)
        actual_k = min(self.k_nn + 1, len(all_pts))
        distances, _ = tree.query(population_behaviors, k=actual_k)

        if actual_k > 1:
            novelty_scores = np.mean(distances[:, 1:], axis=1)
        else:
            novelty_scores = np.zeros(N)

        # Cập nhật Archive sau khi tính điểm
        self._update_archive(population_behaviors, novelty_scores)
        return novelty_scores

    def _update_archive(self, population_behaviors: np.ndarray, novelty_scores: np.ndarray):
        added_count = 0
        for b, score in zip(population_behaviors, novelty_scores):
            # Điều kiện nạp: vượt ngưỡng HOẶC xác suất ngẫu nhiên 1%
            if score > self.threshold or np.random.rand() < 0.01:
                self.archive.append(b.copy())
                added_count += 1
                if len(self.archive) > self.max_capacity:
                    self.archive.pop(0)  # FIFO

        # Điều chỉnh ngưỡng thích nghi
        if added_count > 4:
            self.threshold *= 1.05  # Tăng 5%
            self.stagnation_counter = 0
        elif added_count == 0:
            self.stagnation_counter += 1
            if self.stagnation_counter >= 15:
                self.threshold = max(0.1, self.threshold * 0.95)  # Giảm 5%
                self.stagnation_counter = 0
```

---

## 4. Quyết định của tôi
- [x] **Chọn Phương án A cho Ngưỡng thích nghi**: Tự động tinh chỉnh $\rho_{\min}$ tăng/giảm theo nhịp độ nạp thực tế. Đảm bảo chất lượng điểm nạp luôn được chắt lọc khắt khe khi không gian hành vi dần được lấp đầy.
- [x] **Chọn Phương án A cho Giới hạn dung lượng**: Giới hạn trần $2,000$ điểm kèm `cKDTree` và cơ chế FIFO. Giúp mô hình chạy hàng trăm thế hệ mà không lo tràn bộ nhớ hay nghẽn CPU.

---

## 5. TODO tiếp theo
- [ ] Tích hợp `NoveltyArchive` vào module `src/novelty/archive.py`.
- [ ] Brainstorm tiếp chủ đề `03_novelty/adaptive_lambda.md`: Cơ chế lai Hybrid Fitness-Novelty $\lambda$ và Minimal Criteria.
