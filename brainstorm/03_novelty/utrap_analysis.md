# PHÂN TÍCH HÌNH HỌC BẪY CHỮ U, HIỆN TƯỢNG DECEPTIVE GRADIENT & MINH CHỨNG KHOA HỌC
> **Ngày**: 2026-10-02  
> **Trạng thái**: [DECIDED]  
> **Liên quan**: overview.md phần 4.2 & 4.3  

---

## 1. Câu hỏi gốc tôi đã hỏi LLM
> "Phân tích hình học bẫy chữ U, bản chất toán học của hiện tượng Deceptive Gradient, cơ chế khuếch tán hành vi (Behavioral Diffusion) của Novelty Search, và cách trực quan hóa so sánh trực diện trên Slide báo cáo."

---

## 2. Tóm tắt kết luận rút ra (Key Takeaways)
- **Bản chất Deceptive Gradient**: Đáy bẫy chữ U là cực tiểu cục bộ (local minimum) có khoảng cách ngắn nhất tới Goal theo đường chim bay. Mọi cá thể tiến hóa theo Fitness khoảng cách đều bị dồn ứ tại đây. Đột biến quay đầu đi lùi bị coi là "thụt lùi thích nghi" và bị chọn lọc tự nhiên đào thải ngay lập tức.
- **Khuếch tán hành vi (Behavioral Diffusion)**: Novelty Search không quan tâm tới vị trí đích. Khi đáy bẫy bị lấp đầy, độ mới lạ tại đó giảm về 0, biến hành vi "quay đầu đi lùi" thành hành vi độc nhất vô nhị $\implies$ Quần thể tràn ra khỏi miệng bẫy chữ U như chất lỏng.
- **Tiêu chuẩn thiết kế bẫy chữ U Cấp 4**: Đặt tỉ lệ $Depth / Width \ge 1.0$ (sâu $2.0\text{m}$, rộng $2.0\text{m}$) chắn trực diện trên trục $S-G$. Độ sâu này đủ lớn để loại trừ khả năng robot "vô tình lách qua", buộc phải dựa vào cơ chế khám phá tính mới.
- **Trực quan hóa đối chứng trực diện**: Dùng biểu đồ phân bố 2D Scatter Plot / Heatmap của các vị trí kết thúc $(x_{\text{final}}, y_{\text{final}})$. Thể hiện sự tương phản tuyệt đối: Fitness NEAT co cụm đặc quánh ở đáy bẫy vs Novelty NEAT tỏa đều khắp không gian và luồn lách qua mép tường.

---

## 3. Mã Nguồn Tạo Biểu Đồ So Sánh Trực Quan Hóa (Matplotlib)

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_deceptive_comparison(fitness_final_pts: np.ndarray, novelty_final_pts: np.ndarray, map_data: dict, save_path: str = "docs/trap_comparison.png"):
    """
    Vẽ 2 subplot so sánh phân bố dừng chân: Fitness vs Novelty Search
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, pts, title, color in zip([ax1, ax2], 
                                     [fitness_final_pts, novelty_final_pts], 
                                     ["(a) Distance Fitness (Mắc kẹt cực trị)", "(b) Novelty Search (Khuếch tán thoát bẫy)"],
                                     ['red', 'blue']):
        # Vẽ các đoạn tường vật cản
        for seg in map_data["segments"]:
            ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], 'k-', lw=2)

        # Vẽ Start và Goal
        S = map_data["start"]
        G = map_data["goal"]
        ax.scatter(S[0], S[1], c='green', s=150, marker='o', label='Start')
        ax.scatter(G[0], G[1], c='gold', s=200, marker='*', label='Goal')

        # Vẽ các điểm dừng chân cuối cùng của quần thể
        ax.scatter(pts[:, 0], pts[:, 1], c=color, alpha=0.4, s=30, label='Robot final positions')

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
```

---

## 4. Quyết định của tôi
- [x] **Chọn Phương án A cho Hình học U-Trap**: Đặt tỉ lệ $Depth / Width \ge 1.0$. Đảm bảo bẫy chữ U đủ sâu để tạo ra thử thách thực thụ cho giải thuật, chứng minh giá trị của Novelty Search.
- [x] **Chọn Phương án A cho Trực quan hóa**: Dùng biểu đồ Scatter Plot 2 cột đối chứng. Đây là hình ảnh trực quan nhất, thuyết phục 100% hội đồng chấm điểm mà không cần giải thích dài dòng.

---

## 5. TODO tiếp theo
- [ ] Hoàn thành 100% nhóm `03_novelty`.
- [ ] Chuyển sang Nhóm 4: **`04_experiments/baseline_fairness.md`** (Thiết kế đối chuẩn công bằng với 7 baselines B1 - B7 và ngân sách tính toán).
