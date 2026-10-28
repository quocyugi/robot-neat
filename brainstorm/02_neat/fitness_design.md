# THIẾT KẾ HÀM FITNESS 5 THÀNH PHẦN & CHỐNG GIAN LẬN (REWARD HACKING)
> **Ngày**: 2026-10-02  
> **Trạng thái**: [DECIDED]  
> **Liên quan**: overview.md phần 4.1  

---

## 1. Câu hỏi gốc tôi đã hỏi LLM
> "Phân tích các trường hợp biên của hàm Fitness 5 thành phần, cách khắc phục hành vi gian lận (đứng im, quay vòng), và kỹ thuật xử lý giá trị Fitness không âm trong neat-python."

---

## 2. Tóm tắt kết luận rút ra (Key Takeaways)
- **Tiến độ dựa trên $d_{\min}$**: Dùng $\frac{d_0 - d_{\min}}{d_0}$ thay vì $\frac{d_0 - d_T}{d_0}$ để ghi nhận công sức của robot khi đã tiến rất gần đích trước khi phải lùi hoặc chuyển hướng tránh chướng ngại vật.
- **Bảo đảm Fitness luôn dương ($F > 0$)**: Thư viện `neat-python` yêu cầu fitness không âm để tính toán chia sẻ độ thích nghi (Fitness Sharing). Thiết lập công thức: $F_{\text{final}} = \max(F_{\text{raw}} + F_{\text{offset}}, \; 0.01)$ với $F_{\text{offset}} = 3.0$.
- **Triệt tiêu hành vi đứng im (Inactivity)**: Robot đứng im tại chỗ không có điểm tiến độ ($d_{\min} = d_0$), chịu phạt thời gian trôi qua, đảm bảo bị đào thải tự nhiên qua các thế hệ.
- **Chống Spinning Hack toàn diện**: Kiểm tra cả vận tốc bánh vi sai ($\|v_L + v_R\| < 0.05$) lẫn tích lũy độ biến thiên góc hướng $\sum |\Delta \theta| > 4\pi$ khi khoảng cách tới đích không cải thiện.

---

## 3. Công thức Toán học & Code Triển khai Chuẩn

### 3.1. Công thức Chi tiết
$$F_{\text{raw}} = w_1 \cdot \text{Progress} + w_2 \cdot I_{\text{goal}} + w_3 \cdot \left(1 - \frac{t_{\text{goal}}}{T_{\max}}\right) \cdot I_{\text{goal}} - w_4 \cdot I_{\text{collision}} - w_5 \cdot \frac{n_{\text{spin}}}{T_{\max}}$$

Trong đó:
$$\text{Progress} = \begin{cases} \frac{d_0 - d_{\min}}{d_0} & \text{nếu an toàn (không va chạm)} \\ 0.5 \cdot \frac{d_0 - d_{\min}}{d_0} & \text{nếu kết thúc bằng va chạm} \end{cases}$$

$$F_{\text{final}} = \max(F_{\text{raw}} + 3.0, \; 0.01)$$

### 3.2. Bảng Trọng số & Phân tích Ý nghĩa
| Thành phần | Ký hiệu | Trọng số | Giá trị đóng góp | Ý nghĩa kỹ thuật |
| :--- | :---: | :---: | :---: | :--- |
| Tiến độ khoảng cách | $w_1$ | $1.0$ | $[0.0, 1.0]$ | Khuyến khích tiếp cận đích |
| Thưởng chạm đích | $w_2$ | $10.0$ | $\{0.0, 10.0\}$ | Động lực tối thượng để hoàn thành nhiệm vụ |
| Thưởng thời gian tối ưu | $w_3$ | $5.0$ | $[0.0, 5.0]$ | Khuyến khích chạy nhanh, đường đi ngắn |
| Phạt va chạm | $w_4$ | $2.0$ | $\{0.0, 2.0\}$ | Trừng phạt đâm va, kết thúc sớm |
| Phạt quay vòng tại chỗ | $w_5$ | $1.0$ | $[0.0, 1.0]$ | Loại bỏ hành vi lách luật, xoay tròn |
| **Bù dịch chuyển (Base Offset)**| $F_{\text{offset}}$ | **$+3.0$** | Hằng số | Bảo đảm $F_{\text{final}} \ge 0.01$ cho neat-python |

### 3.3. Hàm Tính Fitness trong Python
```python
def compute_episode_fitness(info: dict, d0: float, w1=1.0, w2=10.0, w3=5.0, w4=2.0, w5=1.0, max_steps=500) -> float:
    """
    Tính fitness cho 1 episode chạy đơn lẻ của robot
    """
    d_min = info["min_distance_to_goal"]
    is_goal = float(info["is_success"])
    is_col = float(info["is_collision"])
    steps = info["steps"]
    spin_steps = info["spin_steps"]

    # 1. Điểm tiến độ
    raw_progress = max(0.0, (d0 - d_min) / max(d0, 1e-5))
    if is_col:
        progress_score = 0.5 * raw_progress
    else:
        progress_score = raw_progress

    # 2. Điểm thưởng đích & thời gian
    time_bonus = (1.0 - (steps / max_steps)) * is_goal

    # 3. Phạt quay tròn
    spin_penalty = spin_steps / max_steps

    # 4. Fitness tổng thô
    f_raw = (w1 * progress_score +
             w2 * is_goal +
             w3 * time_bonus -
             w4 * is_col -
             w5 * spin_penalty)

    # 5. Dịch chuyển offset chống số âm
    f_final = max(f_raw + 3.0, 0.01)
    return f_final
```

---

## 4. Quyết định của tôi
- [x] **Chọn Phương án A cho Tiến độ**: Dùng $d_{\min}$ thay vì $d_T$. Giúp bảo tồn các đột biến hữu ích đã tìm được đường tiếp cận gần đích, tránh việc chỉ vì né vật cản ở giây cuối mà mất hết điểm.
- [x] **Chọn Phương án A cho Đảm bảo Fitness dương**: Cộng bù $F_{\text{offset}} = 3.0$ và sàn $0.01$. Tương thích hoàn hảo với cơ chế phân loài (Speciation) và chia sẻ fitness của `neat-python`.

---

## 5. TODO tiếp theo
- [ ] Tích hợp hàm `compute_episode_fitness` vào pipeline đánh giá cá thể trong `src/neat/evaluator.py`.
- [ ] Brainstorm tiếp chủ đề `02_neat/config_tuning.md` (Tinh chỉnh tham số config neat-python: population size, mutation rates, compatibility threshold).
