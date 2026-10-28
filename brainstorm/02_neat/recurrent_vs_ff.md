# SO SÁNH FEEDFORWARD NEAT VS RECURRENT NEAT (BỘ NHỚ TRONG & BẪY CHỮ U)
> **Ngày**: 2026-10-02  
> **Trạng thái**: [DECIDED]  
> **Liên quan**: overview.md phần 4.3 & 6.1  

---

## 1. Câu hỏi gốc tôi đã hỏi LLM
> "So sánh Feedforward NEAT và Recurrent NEAT: cơ chế bộ nhớ trong (internal memory), khả năng thoát bẫy chữ U, thiết kế ma trận thực nghiệm đối chuẩn (Ablation Study) và cách quản lý trạng thái kích hoạt trong neat-python."

---

## 2. Tóm tắt kết luận rút ra (Key Takeaways)
- **Hạn chế của Feedforward**: Chỉ phản xạ tức thời $y_t = f(x_t)$, không có ký ức về quá khứ. Khi lọt vào bẫy chữ U, robot dễ bị dao động tiến/lùi (oscillation) vì điểm đích luôn kéo nó về đáy bẫy.
- **Sức mạnh của Recurrent**: Các kết nối ngược tạo nên bộ nhớ ngắn hạn $h_t = \sigma(W_{xh} x_t + W_{hh} h_{t-1} + b)$. Robot có thể "nhận thức" được mình đang bị mắc kẹt kéo dài, từ đó duy trì hành vi quay đầu và đi ra khỏi bẫy dù khoảng cách tới đích tạm thời tăng lên.
- **Hỗ trợ song song cả 2 kiến trúc**: Xây dựng codebase linh hoạt cho phép chuyển đổi giữa `FeedForwardNetwork` và `RecurrentNetwork` thông qua file cấu hình. Tạo thành trục so sánh thực nghiệm trọng tâm (Ablation Study) trong báo cáo và slide bảo vệ.
- **Quy tắc Reset bộ nhớ nghiêm ngặt**: Mỗi khi chuyển sang bản đồ mới (`env.reset()`), bắt buộc gọi `net.reset()` để xóa sạch điện thế kích hoạt cũ, bảo đảm tính độc lập và công bằng giữa các lần đánh giá.

---

## 3. Thiết kế Thực nghiệm Đối chuẩn (Ablation Matrix) & Triển khai Code

### 3.1. Ma trận 4 Biến thể Đối chuẩn
| Biến thể | Cấu trúc Mạng | Cơ chế Tìm kiếm | Mục tiêu kiểm chứng | Dự đoán Thành công Cấp 4 (U-Traps) |
| :---: | :---: | :---: | :--- | :---: |
| **V1** | Feedforward | Distance Fitness | Baseline kiểm tra độ khó của Deceptive Fitness | Rất thấp ($\approx 7.5\%$) |
| **V2** | Recurrent | Distance Fitness | Kiểm tra tác động của bộ nhớ khi chưa có Novelty | Thấp - Trung bình ($\approx 25\%$) |
| **V3** | Feedforward | Hybrid Novelty | Kiểm tra tác động của Novelty Search khi mạng không có nhớ | Khá ($\approx 75 - 80\%$) |
| **V4** | **Recurrent** | **Hybrid Novelty** | **Mô hình đề xuất tối ưu nhất của đồ án** | **Rất cao ($\approx 95 - 100\%$)** |

### 3.2. Code Quản lý Đánh giá Cá thể Linh hoạt trong Python
```python
import neat

def create_network(genome, config, is_recurrent: bool = False):
    """Tạo mạng tương ứng theo cấu hình"""
    if is_recurrent:
        return neat.nn.RecurrentNetwork.create(genome, config)
    return neat.nn.FeedForwardNetwork.create(genome, config)

def evaluate_genome_on_map(net, env, is_recurrent: bool = False):
    """Đánh giá 1 genome trên 1 bản đồ cụ thể"""
    obs, info = env.reset()
    
    # BẮT BUỘC: Reset bộ nhớ trong nếu là mạng Recurrent
    if is_recurrent and hasattr(net, "reset"):
        net.reset()
        
    terminated = truncated = False
    while not (terminated or truncated):
        action = net.activate(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
    return info
```

---

## 4. Quyết định của tôi
- [x] **Chọn Phương án A cho Kiến trúc thực nghiệm**: Xây dựng codebase hỗ trợ cả FF-NEAT và Recurrent-NEAT. Việc so sánh giữa V1, V2, V3, V4 sẽ là "vũ khí khoa học" sắc bén nhất để chứng minh giá trị của Recurrent NEAT khi bảo vệ đồ án.
- [x] **Chọn Phương án A cho Reset bộ nhớ**: Bắt buộc gọi `net.reset()` mỗi khi reset môi trường để triệt tiêu nhiễu dữ liệu chéo giữa các lượt chạy.

---

## 5. TODO tiếp theo
- [ ] Tạo 2 file cấu hình riêng biệt: `configs/neat_ff.ini` và `configs/neat_rec.ini`.
- [ ] Brainstorm chủ đề cuối cùng của nhóm NEAT: `02_neat/speciation_analysis.md` (Phân tích chi tiết cơ chế phân loài, khoảng cách tương thích $\delta$, và giải quyết Stagnation).
