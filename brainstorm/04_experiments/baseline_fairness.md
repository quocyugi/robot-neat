# THIẾT KẾ ĐỐI CHUẨN CÔNG BẰNG VỚI 7 BASELINES & ĐỒNG NHẤT NGÂN SÁCH ĐÁNH GIÁ
> **Ngày**: 2026-10-02  
> **Trạng thái**: [DECIDED]  
> **Liên quan**: overview.md phần 6.1 & 6.2  

---

## 1. Câu hỏi gốc tôi đã hỏi LLM
> "Thiết kế đối chuẩn công bằng giữa mô hình NEAT và 7 baselines (B1 - B7), nguyên tắc đồng nhất ngân sách đánh giá (Evaluations Budget), và phân định rạch ròi về thông tin đầu vào (Local Sensing vs Global Oracle)."

---

## 2. Tóm tắt kết luận rút ra (Key Takeaways)
- **Đồng nhất Ngân sách Đánh giá (Evaluations Budget)**: Cố định cứng $\text{Budget} = 60,000 \text{ evaluations}$ ($150 \text{ cá thể} \times 100 \text{ thế hệ} \times 4 \text{ maps}$) cho mọi thuật toán tiến hóa (B1, B2, B3, NEAT). Loại bỏ sự thiên lệch về thời gian CPU hay cấu hình phần cứng.
- **Phân cấp Triển khai Thực tế**:
  * **Nhóm Cốt lõi Bắt buộc (Core 4)**: B1 (Fixed MLP GA), B3 (Random Search), B4 (Heuristic Rule-based), B5 (Artificial Potential Field). Đủ 100% cơ sở khoa học để trả lời Research Questions Q1 & Q2.
  * **Nhóm Nâng cao (Bonus Benchmark)**: B6 ($A^* + \text{DWA}$) làm cận trên lý thuyết (Oracle Upper Bound), B7 (PPO) so sánh với Deep RL nếu đủ thời gian.
- **Phân định rạch ròi Oracle vs Local Sensing**:
  * B1, B2, B3, B4, B5, B7, NEAT: Chỉ dùng cảm biến cục bộ (Local Sensors - 12 tia LiDAR).
  * B6 ($A^* + \text{DWA}$): Biết trước toàn bộ bản đồ vật cản (Global Map). Nếu NEAT đạt hiệu năng tiệm cận B6, đó là minh chứng xuất sắc cho khả năng tự hành của mạng nơ-ron.

---

## 3. Bảng Tổng Hợp 7 Baselines & Kế Hoạch Triển Khai

| Mã | Tên Phương pháp | Bản chất Thuật toán | Loại Thông tin | Vai trò Khoa học trong Đồ án |
| :---: | :--- | :--- | :---: | :--- |
| **B1** | Fixed MLP GA | Mạng cố định (1 tầng ẩn, 8 nút), chỉ tối ưu trọng số | Cục bộ (15D) | Trả lời **Q1**: Chứng minh NEAT tối ưu cấu trúc vượt trội GA |
| **B2** | Fixed MLP Deep | Mạng cố định (2 tầng ẩn, 16-8 nút), tối ưu trọng số | Cục bộ (15D) | Đánh giá mạng sâu hơn có tự giải được bẫy chữ U hay không |
| **B3** | Random Search | Trọng số ngẫu nhiên thuần túy | Cục bộ (15D) | Chứng minh bài toán không tầm thường (Non-triviality check) |
| **B4** | Heuristic Rule-based | Nếu tia trước $< 0.5\text{m} \to$ rẽ hướng thoáng nhất | Cục bộ (15D) | Phản xạ cơ bản, làm thang đo tối thiểu |
| **B5** | APF (Potential Field)| Lực hút về đích + Lực đẩy từ vật cản | Cục bộ (15D) | Thuật toán kinh điển, chứng minh bẫy chữ U là khắc tinh của APF |
| **B6** | $A^* + \text{DWA}$ | Quy hoạch toàn cục + Động học cục bộ | **Toàn cục (Oracle)** | **Cận trên lý thuyết (Theoretical Upper Bound)** |
| **B7** | Deep RL (PPO) | Proximal Policy Optimization (Actor-Critic) | Cục bộ (15D) | So sánh giữa Tiến hóa (NeuroEvolution) và Học sâu tăng cường |

---

## 4. Quyết định của tôi
- [x] **Chọn Phương án A cho Phân cấp Baselines**: Ưu tiên hoàn thành xuất sắc nhóm Core 4 (B1, B3, B4, B5), đưa B6 làm cận trên lý thuyết. Đảm bảo tiến độ đồ án không bị quá tải nhưng bài báo cáo vẫn cực kỳ chắc chắn.
- [x] **Chọn Phương án A cho Ngân sách Đánh giá**: Chốt chặn $60,000$ evaluations cho mọi thuật toán tiến hóa. Đảm bảo tính trung thực khoa học tuyệt đối.

---

## 5. TODO tiếp theo
- [ ] Brainstorm tiếp chủ đề `04_experiments/ablation_design.md`: Thiết kế ma trận Ablation Study chi tiết (bóc tách vai trò của từng thành phần: Cấu trúc mạng, Novelty Search, Curriculum Learning).
