# CHIẾT LƯỢC BẢO VỆ PHẠM VI ĐỀ TÀI: LẬP LUẬN 2D VS 3D VÀ NEAT VS DEEP RL
> **Ngày**: 2026-10-02  
> **Trạng thái**: [DECIDED]  
> **Liên quan**: overview.md phần 1, 6.2, 7.1  

---

## 1. Câu hỏi gốc tôi đã hỏi LLM
> "Chiến lược bảo vệ phạm vi đề tài trước hội đồng: cách lập luận thuyết phục lý do chọn mô phỏng 2D Vectorized thay vì 3D/Robot thật, và lý do chọn NeuroEvolution (NEAT) thay vì Deep Reinforcement Learning (PPO/SAC)."

---

## 2. Tóm tắt kết luận rút ra (Key Takeaways)
- **Chủ động đánh phủ đầu về Mô phỏng 2D**: Đưa hẳn 1 Slide chính thức ngay ở phần Đặt vấn đề giải thích lý do dùng 2D Vectorized:
  * Ngân sách đánh giá khoa học: $600,000$ episodes (10 seeds $\times 60,000$). Trên 2D đạt $> 20,000$ FPS (chạy trong vài chục phút); trên 3D Gazebo (50-100 FPS) sẽ mất $3 - 6$ tháng tính toán liên tục.
  * Tính kế thừa chuẩn mực từ các paper nền tảng của Kenneth Stanley & Joel Lehman (MIT Press).
  * Cầu nối thực tế: Đã tích hợp Robustness Test (nhiễu LiDAR $5\%$, trượt bánh $2\%$) và tập dữ liệu BARN.
- **Định vị ưu thế tuyệt đối của NEAT trước Deep RL**:
  * Miễn nhiễm Deceptive Gradient: Deep RL dựa trên đạo hàm phần thưởng $\nabla_\theta J$ sẽ sụp đổ ở bẫy chữ U, trong khi Novelty Search không phụ thuộc gradient mục tiêu.
  * Tối giản phần cứng (Occam's Razor): Mạng NEAT chỉ có $3 - 6$ nút ẩn (vài chục tham số), có thể chạy thời gian thực trên chip nhúng siêu rẻ (ESP32/STM32 tiêu thụ vài milliwatt), vượt trội hoàn toàn các mạng Deep RL cồng kềnh hàng trăm ngàn tham số đòi hỏi GPU.

---

## 3. Bảng So Sánh Chiến Lược Giữa NEAT và Deep RL (Dùng cho Slide)

| Tiêu chí | Deep RL (PPO / SAC / DDPG) | NEAT + Novelty Search (Đề xuất) |
| :--- | :--- | :--- |
| **Kiến trúc mạng** | Cố định trước (Human-designed), thường rất lớn | **Tự động tiến hóa từ tối giản (3 – 6 nút ẩn)** |
| **Hiện tượng bẫy chữ U** | Rất dễ kẹt cực trị địa phương, sụp đổ chính sách | **Vượt qua ngoạn mục nhờ cơ chế khuếch tán hành vi** |
| **Tính toán đạo hàm** | Bắt buộc Backpropagation, nhạy cảm vanishing gradient | **Không cần đạo hàm (Gradient-free), siêu ổn định** |
| **Triển khai phần cứng nhúng** | Đòi hỏi vi xử lý mạnh (Raspberry Pi 4, Jetson Nano) | **Nạp vừa khít vi điều khiển giá rẻ ($2) như ESP32** |
| **Khả năng song song hóa** | Phụ thuộc GPU batching | **Song song hóa hoàn hảo trên đa nhân CPU** |

---

## 4. Quyết định của tôi
- [x] **Chọn Phương án A cho Lập luận 2D**: Đưa thẳng slide "Lý do lựa chọn 2D Vectorized" vào phần Đặt vấn đề. Chủ động giải trình ngân sách $600,000$ evaluations để biến điều tưởng như nhược điểm thành điểm cộng về phương pháp luận nghiên cứu nghiêm túc.
- [x] **Chọn Phương án A cho So sánh với Deep RL**: Trình bày bảng so sánh trực diện NEAT vs Deep RL trên Slide. Khẳng định giá trị thực tiễn của NEAT trong bài toán điều khiển nhúng công suất thấp.

---

## 5. TODO tiếp theo
- [ ] Brainstorm chủ đề cuối cùng của toàn bộ dự án: `05_defense/qa_preparation.md` (Bộ ngân hàng 10 câu hỏi & kịch bản trả lời phản biện sắc bén nhất của Hội đồng bảo vệ).
