# PHÂN TÍCH THẤT BẠI (FAILURE ANALYSIS) & ĐÁNH GIÁ ĐỘ BỀN VỮNG (ROBUSTNESS TEST)
> **Ngày**: 2026-10-02  
> **Trạng thái**: [DECIDED]  
> **Liên quan**: overview.md phần 6.3 & 7.1  

---

## 1. Câu hỏi gốc tôi đã hỏi LLM
> "Phân tích các trường hợp thất bại (Failure Mode Analysis), giới hạn cơ học của robot trong các góc hẹp cực đoan, cách trình bày khoa học và trung thực trên Slide báo cáo, và thiết kế bài kiểm thử độ bền (Robustness Test) với nhiễu cảm biến Sim-to-Real."

---

## 2. Tóm tắt kết luận rút ra (Key Takeaways)
- **Minh bạch khoa học là sức mạnh**: Không che giấu các trường hợp thất bại. Việc dành riêng 1 slide và 1 mục báo cáo để mổ xẻ nguyên nhân gốc rễ (Root Causes) khẳng định sự thấu hiểu sâu sắc bản chất hệ thống của người nghiên cứu.
- **Nhận diện 3 Dạng Thất bại Điển hình**:
  1. *Khe hẹp cực hạn (Narrow Passage)*: Khe hẹp $< 0.55\text{m}$ tiệm cận đường kính thân xe ($0.50\text{m}$).
  2. *Bẫy chữ U kép (Nested Double U-Trap)*: Vượt qua bẫy thứ nhất nhưng bộ nhớ ngắn hạn bị bão hòa ở bẫy thứ hai.
  3. *Khám phá quá đà (Exploration Timeout)*: Mải mê thám hiểm các vùng thưa thớt dẫn đến hết hạn $T_{\max} = 500$ steps.
- **Thử nghiệm Độ bền vững (Robustness Test)**: Thêm $5\%$ Gaussian Noise vào khoảng cách 12 tia LiDAR và $2\%$ Slip Noise vào vận tốc bánh vi sai. Đánh giá mức độ suy giảm hiệu năng để chứng minh mô hình không bị quá khớp (overfit) vào môi trường mô phỏng lý tưởng.

---

## 3. Thiết Kế Thử Nghiệm Nhiễu Sim-to-Real (Sensor & Actuator Noise)

```python
import numpy as np

def apply_sim2real_noise(obs: np.ndarray, action: np.ndarray, lidar_noise_std: float = 0.05, wheel_slip_std: float = 0.02):
    """
    obs: 15D observation (12 tia LiDAR proximity, 1 khoảng cách, 2 góc)
    action: 2D continuous action (vL, vR)
    """
    # 1. Thêm nhiễu Gauss 5% vào 12 tia LiDAR
    noisy_obs = obs.copy()
    lidar_noise = np.random.normal(0, lidar_noise_std, size=12)
    noisy_obs[:12] = np.clip(noisy_obs[:12] + lidar_noise, 0.0, 1.0)

    # 2. Thêm nhiễu trượt bánh xe (Actuator Slip) 2%
    noisy_action = action.copy()
    slip_noise = np.random.normal(0, wheel_slip_std, size=2)
    noisy_action = np.clip(noisy_action + slip_noise, -1.0, 1.0)

    return noisy_obs, noisy_action
```

---

## 4. Quyết định của tôi
- [x] **Chọn Phương án A cho Trình bày Thất bại**: Dành riêng 1 Slide phân tích thất bại trong bài thuyết trình. Minh họa bằng hình vẽ quỹ đạo 3 ca điển hình, biến điểm yếu thành cơ hội thể hiện tư duy phản biện và định hướng phát triển (Future Work).
- [x] **Chọn Phương án A cho Đánh giá Độ bền (Robustness)**: Kiểm thử thêm kịch bản có nhiễu cảm biến $5\%$ và trượt bánh $2\%$. Khẳng định tiềm năng ứng dụng thực tế và chuyển giao Sim-to-Real.

---

## 5. TODO tiếp theo
- [ ] Hoàn thành 100% nhóm 04_experiments.
- [ ] Chuyển sang Nhóm 5: **`05_defense/scope_negotiation.md`** & **`05_defense/qa_preparation.md`** (Chiến lược phản biện, bảo vệ phạm vi 2D vs 3D, và chuẩn bị bộ câu hỏi hóc búa của Hội đồng).
