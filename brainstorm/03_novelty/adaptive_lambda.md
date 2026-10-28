# CƠ CHẾ LAI GHÉP HYBRID FITNESS-NOVELTY (ADAPTIVE LAMBDA) & MINIMAL CRITERIA
> **Ngày**: 2026-10-02  
> **Trạng thái**: [DECIDED]  
> **Liên quan**: overview.md phần 4.3  

---

## 1. Câu hỏi gốc tôi đã hỏi LLM
> "Thiết kế cơ chế lai ghép Hybrid Fitness-Novelty: chuẩn hóa thang đo Min-Max, chính sách trọng số lambda thích nghi theo Stagnation (Adaptive Lambda), và tiêu chuẩn sinh tồn tối thiểu (Minimal Criteria Novelty Search - MCNS)."

---

## 2. Tóm tắt kết luận rút ra (Key Takeaways)
- **Chuẩn hóa Min-Max bắt buộc**: Điểm thích nghi thô ($F \in [0.01, 18]$) và điểm Novelty ($\rho \in [0, 5\text{m}]$) phải được chuẩn hóa về $[0, 1]$ trước khi kết hợp tuyến tính, tránh hiện tượng một bên lấn át hoàn toàn bên kia.
- **Adaptive Lambda ($\lambda$) thông minh**: 
  * Trạng thái bình thường: $\lambda = 0.6$ (ưu tiên tiến độ về đích).
  * Trạng thái tắc nghẽn (Stagnation $\ge 5$ thế hệ không cải thiện Fitness): Tự động hạ $\lambda = 0.1$, kích hoạt chế độ "Phá bẫy cực đoan" thuần Novelty Search.
  * Khi phá bẫy thành công và tìm thấy đường đi mới: Tăng dần $\lambda$ trở lại $0.6$ để dẫn đường robot cán đích.
- **Tiêu chuẩn sinh tồn tối thiểu (MCNS)**: Yêu cầu robot phải di chuyển an toàn tối thiểu 30 bước đầu tiên ($t \ge 30$ steps) mà không đâm va mới được nhận điểm Novelty đầy đủ. Nếu đâm ngay vạch xuất phát, điểm Novelty bị giảm phạt $80\%$, triệt tiêu việc "khen thưởng" hành vi tự sát sớm.

---

## 3. Công thức Toán học & Triển khai Code Hoàn Chỉnh

### 3.1. Chuẩn hóa & Công thức Kết hợp
Cho quần thể $N$ cá thể tại thế hệ $g$:
$$\tilde{F}_i = \frac{F_i - \min_j F_j}{\max(\max_j F_j - \min_j F_j, \; 10^{-6})}$$
$$\tilde{\rho}_i = \frac{\rho_i - \min_j \rho_j}{\max(\max_j \rho_j - \min_j \rho_j, \; 10^{-6})}$$

Hệ số kiểm tra Minimal Criteria:
$$c_{\text{mc}, i} = \begin{cases} 1.0 & \text{nếu an toàn qua 30 bước đầu} \\ 0.2 & \text{nếu va chạm trước bước 30} \end{cases}$$

Điểm Hybrid cuối cùng gán cho genome trong `neat-python`:
$$F_{\text{hybrid}, i} = \lambda \cdot \tilde{F}_i + (1 - \lambda) \cdot (c_{\text{mc}, i} \cdot \tilde{\rho}_i)$$

### 3.2. Code Module Điều khiển Hybrid Thích Nghi
```python
import numpy as np

class AdaptiveHybridManager:
    def __init__(self, base_lambda: float = 0.6, exploration_lambda: float = 0.1, stagnation_limit: int = 5):
        self.base_lambda = base_lambda
        self.exploration_lambda = exploration_lambda
        self.stagnation_limit = stagnation_limit
        self.current_lambda = base_lambda
        self.best_fitness_history = []
        self.stagnant_gens = 0

    def update_lambda(self, current_max_fitness: float) -> float:
        """Cập nhật trọng số lambda dựa trên độ trì trệ của quần thể"""
        if len(self.best_fitness_history) > 0 and current_max_fitness <= max(self.best_fitness_history):
            self.stagnant_gens += 1
        else:
            self.stagnant_gens = 0  # Có tiến bộ, reset đếm

        self.best_fitness_history.append(current_max_fitness)

        # Nếu bị tắc nghẽn >= 5 thế hệ -> Kích hoạt chế độ phá bẫy
        if self.stagnant_gens >= self.stagnation_limit:
            self.current_lambda = self.exploration_lambda
        else:
            # Phục hồi dần về base_lambda
            self.current_lambda = min(self.base_lambda, self.current_lambda + 0.1)

        return self.current_lambda

    def combine_scores(self, raw_fitness: np.ndarray, raw_novelty: np.ndarray, steps: np.ndarray, collisions: np.ndarray) -> np.ndarray:
        """Chuẩn hóa Min-Max và kết hợp theo trọng số hiện tại"""
        # 1. Min-Max Scaling
        f_norm = (raw_fitness - np.min(raw_fitness)) / max(np.ptp(raw_fitness), 1e-6)
        nov_norm = (raw_novelty - np.min(raw_novelty)) / max(np.ptp(raw_novelty), 1e-6)

        # 2. Minimal Criteria Mask (va chạm trước bước 30 bị phạt)
        mc_factor = np.where((collisions) & (steps < 30), 0.2, 1.0)
        adjusted_novelty = nov_norm * mc_factor

        # 3. Kết hợp lai
        hybrid_scores = self.current_lambda * f_norm + (1.0 - self.current_lambda) * adjusted_novelty
        
        # Đảm bảo điểm luôn dương cho neat-python
        return np.maximum(hybrid_scores, 0.01)
```

---

## 4. Quyết định của tôi
- [x] **Chọn Phương án A cho Trọng số $\lambda$**: Dùng Adaptive $\lambda$ ($\lambda = 0.6$ thông thường, tự động hạ $\lambda = 0.1$ khi bị kẹt bẫy 5 thế hệ). Tạo nên một cơ chế linh hoạt, giải quyết triệt để sự giằng co giữa Exploration và Exploitation.
- [x] **Chọn Phương án A cho Tiêu chuẩn tối thiểu (MCNS)**: Ép điều kiện sinh tồn tối thiểu 30 bước đầu, loại bỏ các hành vi "đột tử" vô nghĩa nhưng lại có tính mới giả tạo.

---

## 5. TODO tiếp theo
- [ ] Tích hợp `AdaptiveHybridManager` vào quy trình đánh giá thế hệ trong `src/novelty/hybrid.py`.
- [ ] Brainstorm chủ đề cuối cùng của nhóm 3: `03_novelty/utrap_analysis.md` (Giải tích hình học bẫy chữ U và minh chứng khoa học vì sao Fitness khoảng cách thất bại).
