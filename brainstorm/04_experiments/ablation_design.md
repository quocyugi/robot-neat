# THIẾT KẾ MA TRẬN ABLATION STUDY VÀ CHỈ SỐ ĐÁNH GIÁ CHUẨN QUỐC TẾ (SPL)
> **Ngày**: 2026-10-02  
> **Trạng thái**: [DECIDED]  
> **Liên quan**: overview.md phần 6.1 & 6.3  

---

## 1. Câu hỏi gốc tôi đã hỏi LLM
> "Thiết kế ma trận Ablation Study bóc tách 3 trục thành phần (Kiến trúc Recurrent, Cơ chế Novelty Search, và Curriculum Learning), và phương pháp tính toán chỉ số chất lượng đường đi chuẩn quốc tế SPL (Success weighted by Path Length)."

---

## 2. Tóm tắt kết luận rút ra (Key Takeaways)
- **Ma trận 6 Cấu hình (M1 – M6)**: Bóc tách độc lập và chính xác vai trò của từng thành phần kỹ thuật, trả lời dứt khoát câu hỏi "hiệu năng cao là nhờ module nào".
- **Thước đo SPL (Success weighted by Path Length)**: Bắt buộc sử dụng bên cạnh Success Rate (SR). SPL phạt nặng các hành vi đi đường vòng vo vô nghĩa, chỉ chấm điểm cao cho robot vừa về đích an toàn vừa bám sát đường đi tối ưu $L_{A^*}$.
- **Đánh giá độ phức tạp mạng (Parsimony / Network Complexity)**: Theo dõi số lượng nút ẩn và số liên kết của genome chiến thắng để chứng minh NEAT sinh ra các cấu trúc nơ-ron nhỏ gọn, tiết kiệm năng lượng tính toán hơn hẳn mạng Deep MLP cố định (B1, B2).
- **Quy chuẩn Tập Test Độc lập**: Toàn bộ các cấu hình M1 – M6 sau khi huấn luyện xong đều được kiểm thử trên cùng **200 bản đồ Test cố định** (seed `200,000 – 200,199`).

---

## 3. Ma Trận Ablation Study & Mã Nguồn Tính SPL

### 3.1. Bảng Ma Trận Thực Nghiệm 6 Cấu Hình
| Mã | Cấu trúc Nơ-ron | Cơ chế Tìm kiếm | Giáo trình (Curriculum) | Mục tiêu Khoa học |
| :---: | :---: | :---: | :---: | :--- |
| **M1** | Feedforward | Distance Fitness | Không | **Sàn đối chứng**: Cấu hình cơ bản nhất (dự kiến thất bại ở Cấp 4) |
| **M2** | Recurrent | Distance Fitness | Có | **Bóc tách vai trò của Recurrent**: Đo hiệu quả bộ nhớ khi chưa có Novelty |
| **M3** | Feedforward | Hybrid Novelty | Có | **Bóc tách vai trò của Novelty**: Đo hiệu quả Novelty khi mạng không có bộ nhớ |
| **M4** | Recurrent | Pure Novelty | Có | **Bóc tách vai trò của Hybrid**: Đánh giá Novelty thuần không có hướng đích |
| **M5** | Recurrent | Hybrid Novelty | **Không** | **Bóc tách vai trò của Curriculum**: Đo tác động khi nhảy cóc vào bẫy khó |
| **M6** | **Recurrent** | **Hybrid Novelty** | **Có** | **MÔ HÌNH ĐỀ XUẤT TOÀN DIỆN (Full Proposed Model)** |

### 3.2. Công thức & Hàm Tính SPL trong Python
$$\text{SPL} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} S_i \cdot \frac{L_{\text{optimal}, i}}{\max(L_{\text{actual}, i}, L_{\text{optimal}, i})}$$

```python
import numpy as np

def compute_spl(is_success_list: list, actual_path_lengths: list, optimal_path_lengths: list) -> float:
    """
    is_success_list: list các giá trị bool [True, False, ...]
    actual_path_lengths: list độ dài đường đi thực tế robot đã chạy (mét)
    optimal_path_lengths: list độ dài đường đi tối ưu tính bằng A* (mét)
    """
    spl_values = []
    for s, l_act, l_opt in zip(is_success_list, actual_path_lengths, optimal_path_lengths):
        if not s:
            spl_values.append(0.0)
        else:
            spl_values.append(l_opt / max(l_act, l_opt))
    return float(np.mean(spl_values))
```

---

## 4. Quyết định của tôi
- [x] **Chọn Phương án A cho Ma trận Ablation**: Khóa danh mục 6 mô hình M1 – M6. Đủ chiều sâu học thuật để viết báo cáo khoa học xuất sắc, thể hiện tư duy nghiên cứu thực nghiệm chặt chẽ.
- [x] **Chọn Phương án A cho Thước đo SPL**: Bắt buộc tính toán và trình bày SPL trong mọi bảng kết quả. Khẳng định robot không chỉ biết "sống sót mò mẫm" mà còn điều hướng thông minh tiệm cận đường đi tối ưu.

---

## 5. TODO tiếp theo
- [ ] Brainstorm tiếp chủ đề `04_experiments/statistical_methods.md`: Quy chuẩn kiểm định ý nghĩa thống kê (Mann-Whitney U Test, Bonferroni correction, 10 seeds độc lập).
