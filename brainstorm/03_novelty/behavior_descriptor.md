# THIẾT KẾ BỘ MÔ TẢ HÀNH VI b(x) VÀ KHÔNG GIAN HÀNH VI (BEHAVIOR SPACE)
> **Ngày**: 2026-10-02  
> **Trạng thái**: [DECIDED]  
> **Liên quan**: overview.md phần 4.3  

---

## 1. Câu hỏi gốc tôi đã hỏi LLM
> "Thiết kế bộ mô tả hành vi b(x) (Behavior Characterization) trong Novelty Search: so sánh giữa Final Position (tọa độ điểm cuối 2D) và Trajectory Sampling (lấy mẫu quỹ đạo 10D/20D), số lượng láng giềng k-NN và cơ chế tính độ thưa thớt (Sparsity)."

---

## 2. Tóm tắt kết luận rút ra (Key Takeaways)
- **Chuẩn hóa bộ mô tả hành vi $b(x) \in \mathbb{R}^2$**: Sử dụng tọa độ dừng chân cuối cùng $b(x) = (x_{\text{final}}, y_{\text{final}})$ theo đúng paper kinh điển của Lehman & Stanley (2011). Tránh hoàn toàn Lời nguyền số chiều (Curse of Dimensionality) và ngăn ngừa hành vi robot "lách luật" uốn lượn vô ích trên đường đi.
- **Tự động trừng phạt đâm va tập thể**: Khi các robot cùng lao đầu vào đáy bẫy chữ U và va chạm, vị trí dừng cuối của chúng sẽ co cụm lại $\implies$ Mật độ láng giềng tại đáy bẫy tăng vọt $\implies$ Điểm Novelty giảm về $0$, buộc quần thể phải khám phá sang các hướng rẽ hai bên.
- **Tham số $k_{\text{nn}} = 15$**: Đo lường độ mới lạ thông qua khoảng cách Euclid trung bình tới 15 láng giềng gần nhất (gồm cả quần thể hiện tại và kho lưu trữ Archive), bảo đảm làm mượt nhiễu cục bộ.
- **Đánh giá đa bản đồ**: Điểm Novelty của cá thể là trung bình cộng điểm Novelty trên $K = 4$ bản đồ ngẫu nhiên trong mỗi thế hệ.

---

## 3. Công thức Toán học & Triển khai Kỹ thuật

### 3.1. Độ mới lạ (Sparsity / Novelty Score)
Cho cá thể $x$ có bộ mô tả hành vi $\mathbf{b}(x) = (x_{\text{final}}, y_{\text{final}})$. Tập ứng viên láng giềng bao gồm:
$$\mathcal{S} = \mathcal{P}_{\text{current}} \cup \mathcal{A}_{\text{archive}}$$

Khoảng cách Euclid tới cá thể $y$:
$$\text{dist}(\mathbf{b}(x), \mathbf{b}(y)) = \|\mathbf{b}(x) - \mathbf{b}(y)\|_2 = \sqrt{(x_{\text{final}} - x'_y)^2 + (y_{\text{final}} - y'_y)^2}$$

Điểm Novelty $\rho(x)$ là trung bình khoảng cách tới $k_{\text{nn}} = 15$ láng giềng gần nhất trong $\mathcal{S}$:
$$\rho(x) = \frac{1}{k_{\text{nn}}} \sum_{i=1}^{k_{\text{nn}}} \text{dist}(\mathbf{b}(x), \boldsymbol{\mu}_i)$$
*(với $\boldsymbol{\mu}_i$ là láng giềng thứ $i$ gần nhất của $\mathbf{b}(x)$)*.

### 3.2. Module Tính Điểm Novelty
```python
import numpy as np
from scipy.spatial import cKDTree

def compute_novelty_scores(population_behaviors: np.ndarray, archive_behaviors: np.ndarray, k_nn: int = 15) -> np.ndarray:
    """
    population_behaviors: array shape (N, 2)
    archive_behaviors: array shape (M, 2) (có thể rỗng)
    k_nn: số láng giềng gần nhất (15)
    Trả về: novelty_scores shape (N,)
    """
    N = len(population_behaviors)
    
    # Tập hợp không gian hành vi: Quần thể hiện tại + Archive
    if len(archive_behaviors) > 0:
        all_behaviors = np.vstack([population_behaviors, archive_behaviors])
    else:
        all_behaviors = population_behaviors

    # Xây dựng KD-Tree để truy vấn láng giềng trong O(log N)
    tree = cKDTree(all_behaviors)

    # Truy vấn k_nn + 1 láng giềng (vì láng giềng gần nhất là chính nó với khoảng cách = 0)
    actual_k = min(k_nn + 1, len(all_behaviors))
    distances, _ = tree.query(population_behaviors, k=actual_k)

    # Bỏ cột đầu tiên (khoảng cách tới chính nó) và tính trung bình
    if actual_k > 1:
        novelty_scores = np.mean(distances[:, 1:], axis=1)
    else:
        novelty_scores = np.zeros(N)

    return novelty_scores
```

---

## 4. Quyết định của tôi
- [x] **Chọn Phương án A cho $b(x)$**: Dùng $b(x) = (x_{\text{final}}, y_{\text{final}}) \in \mathbb{R}^2$. Vừa tối ưu tốc độ tính toán, vừa phù hợp hoàn hảo với bài toán điều hướng bẫy chữ U 2D.
- [x] **Chọn Phương án A cho Láng giềng**: Cố định $k_{\text{nn}} = 15$. Phản ánh độ thưa thớt (sparsity) tin cậy của không gian hành vi.

---

## 5. TODO tiếp theo
- [ ] Brainstorm tiếp chủ đề `03_novelty/archive_management.md`: Cơ chế quản lý kho lưu trữ Archive (KD-Tree, ngưỡng nạp động $\rho_{\min}$, giới hạn kích thước archive).
