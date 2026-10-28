# QUYẾT ĐỊNH THIẾT KẾ PROJECT — NEAT ROBOT NAVIGATION
> Cập nhật lần cuối: 2026-10-02
> File này tổng hợp MỌI quyết định thiết kế đã được chốt sau khi brainstorm.
> Mỗi khi brainstorm xong 1 chủ đề → cập nhật bảng tương ứng ở đây.

---

## 🟢 Môi trường & Robot (01_env/)

| # | Vấn đề | Quyết định | Lý do | File tham khảo | Ngày |
|---|--------|-----------|-------|----------------|------|
| 1 | Biểu diễn khoảng cách LiDAR | Proximity Index: $1 - d/R_{\max}$ | Kích thích nơ-ron mạnh khi nguy hiểm, an toàn = 0 | [ray_casting.md](01_env/ray_casting.md) | 2026-10-02 |
| 2 | Thuật toán Ray-casting | NumPy Vectorized (Broadcasting `(12, M)`) | Đạt > 20,000 FPS trên CPU, loại bỏ loop Python | [ray_casting.md](01_env/ray_casting.md) | 2026-10-02 |
| 3 | Xử lý vật cản tròn | Xấp xỉ đa giác đều (8 cạnh) | Đồng nhất toàn bộ môi trường thành line segments | [ray_casting.md](01_env/ray_casting.md) | 2026-10-02 |
| 4 | Cung cấp bản đồ khi Train | Pre-generation Dataset (JSON) | 1,000 train + 100 val + 200 test; load < 0.1ms, tái lập 100% | [map_generator.md](01_env/map_generator.md) | 2026-10-02 |
| 5 | Tiêu chuẩn lọc bản đồ | Lọc kép: $A^*$ liên thông + Dải Tortuosity $\tau$ | Đảm bảo đúng độ khó cho từng cấp độ Curriculum | [map_generator.md](01_env/map_generator.md) | 2026-10-02 |
| 6 | Phát hiện va chạm | Point-to-Segment Euclidean Distance | Chính xác tuyệt đối, tránh lỗi xuyên tường của LiDAR | [gymnasium_api.md](01_env/gymnasium_api.md) | 2026-10-02 |
| 7 | Render Engine | Pygame đa chế độ (`human` + `rgb_array`) | 60 FPS xem trực tiếp + Xuất video MP4/GIF làm slide | [gymnasium_api.md](01_env/gymnasium_api.md) | 2026-10-02 |

## 🟡 NEAT Engine & Fitness (02_neat/)

| # | Vấn đề | Quyết định | Lý do | File tham khảo | Ngày |
|---|--------|-----------|-------|----------------|------|
| 8 | Tiến độ khoảng cách trong Fitness | Dùng $d_{\min}$ (khoảng cách gần đích nhất đạt được) | Bảo tồn nỗ lực tiếp cận đích, không xóa sạch điểm khi né tránh ở cuối | [fitness_design.md](02_neat/fitness_design.md) | 2026-10-02 |
| 9 | Đảm bảo Fitness luôn dương ($F > 0$) | Thêm Offset $+3.0$ & Hàm sàn $\max(F_{\text{raw}} + 3.0, 0.01)$ | Đảm bảo tương thích ổn định với Roulette Selection / Speciation neat-python | [fitness_design.md](02_neat/fitness_design.md) | 2026-10-02 |
| 10 | Kiểm soát số lượng loài | Adaptive Speciation Threshold (duy trì 8 - 15 loài) | Bảo vệ cấu trúc mới sinh, tránh 1 siêu loài áp đảo | [config_tuning.md](02_neat/config_tuning.md) | 2026-10-02 |
| 11 | Hàm kích hoạt mạng | Cố định `tanh` cho toàn mạng (Hidden + Output) | Đối xứng qua 0, tương thích góc lệch và động học vi sai | [config_tuning.md](02_neat/config_tuning.md) | 2026-10-02 |
| 12 | Kiến trúc Mạng đối chuẩn | Hỗ trợ song song cả FF-NEAT và Recurrent-NEAT | Tạo ma trận thực nghiệm 4 biến thể (V1 -> V4) cho Ablation Study | [recurrent_vs_ff.md](02_neat/recurrent_vs_ff.md) | 2026-10-02 |
| 13 | Quản lý Bộ nhớ Recurrent | Bắt buộc gọi `net.reset()` tại mỗi `env.reset()` | Xóa sạch trạng thái kích hoạt cũ, bảo đảm tính độc lập giữa các map | [recurrent_vs_ff.md](02_neat/recurrent_vs_ff.md) | 2026-10-02 |
| 14 | Chuẩn hóa số gen $N$ | Chuẩn hóa động: $N = \max(\|G_A\|, \|G_B\|)$ khi gen $\ge 20$ | Tránh méo mó khoảng cách tương thích $\delta$ khi mạng còn nhỏ | [speciation_analysis.md](02_neat/speciation_analysis.md) | 2026-10-02 |
| 15 | Chính sách Trì trệ | `max_stagnation = 15` kết hợp `species_elitism = 2` | Đào thải nhánh bế tắc sau 15 gen, luôn bảo tồn 2 loài tốt nhất | [speciation_analysis.md](02_neat/speciation_analysis.md) | 2026-10-02 |

## 🔴 Novelty Search & Deceptive Fitness (03_novelty/)

| # | Vấn đề | Quyết định | Lý do | File tham khảo | Ngày |
|---|--------|-----------|-------|----------------|------|
| 16 | Bộ mô tả hành vi $b(x)$ | Final Position: $(x_{\text{final}}, y_{\text{final}}) \in \mathbb{R}^2$ | Tránh Curse of Dimensionality, ngăn gian lận uốn lượn | [behavior_descriptor.md](03_novelty/behavior_descriptor.md) | 2026-10-02 |
| 17 | Số lượng láng giềng Novelty | $k_{\text{nn}} = 15$ láng giềng | Chuẩn Lehman & Stanley, làm mượt nhiễu cục bộ | [behavior_descriptor.md](03_novelty/behavior_descriptor.md) | 2026-10-02 |
| 18 | Ngưỡng nạp Archive | Adaptive Threshold ($\pm 5\%$) + 1% ngẫu nhiên | Giữ tốc độ nạp ổn định, tránh bão hòa/đóng băng Archive | [archive_management.md](03_novelty/archive_management.md) | 2026-10-02 |
| 19 | Quản lý dung lượng Archive | Trần $M_{\max} = 2,000$ (FIFO) + Scipy `cKDTree` | Truy vấn $< 0.5\text{ms}$, không rò rỉ RAM | [archive_management.md](03_novelty/archive_management.md) | 2026-10-02 |
| 20 | Trọng số Hybrid $\lambda$ | Adaptive $\lambda$ ($0.6 \to 0.1$ khi kẹt $\ge 5$ gen) | Tự động kích hoạt cơ chế phá bẫy khi quần thể gặp khó | [adaptive_lambda.md](03_novelty/adaptive_lambda.md) | 2026-10-02 |
| 21 | Tiêu chuẩn sinh tồn (MCNS) | Sống sót an toàn tối thiểu 30 bước đầu | Ngăn chặn việc khen thưởng các hành vi tự sát sớm | [adaptive_lambda.md](03_novelty/adaptive_lambda.md) | 2026-10-02 |
| 22 | Hình học Bẫy chữ U Cấp 4 | Tỉ lệ $Depth / Width \ge 1.0$ (sâu $2\text{m}$, rộng $2\text{m}$) | Bẫy đủ sâu để Deceptive Gradient loại bỏ hoàn toàn may rủi | [utrap_analysis.md](03_novelty/utrap_analysis.md) | 2026-10-02 |
| 23 | Trực quan hóa Bẫy lừa | 2D Scatter Plot / Heatmap đối chứng 2 cột | Minh chứng trực quan Fitness kẹt ở đáy vs Novelty tràn ra ngoài | [utrap_analysis.md](03_novelty/utrap_analysis.md) | 2026-10-02 |

## 🔵 Thực nghiệm & Thống kê (04_experiments/)

| # | Vấn đề | Quyết định | Lý do | File tham khảo | Ngày |
|---|--------|-----------|-------|----------------|------|
| 24 | Phân cấp Baselines triển khai | Core 4 (B1, B3, B4, B5) + Bonus (B6 Oracle, B7 PPO) | Đảm bảo trả lời trọn vẹn Q1 & Q2 mà không bị quá tải | [baseline_fairness.md](04_experiments/baseline_fairness.md) | 2026-10-02 |
| 25 | Ngân sách Đánh giá (Budget) | Chốt cứng $60,000$ evaluations cho GA/NEAT | Đảm bảo tính công bằng khoa học tuyệt đối | [baseline_fairness.md](04_experiments/baseline_fairness.md) | 2026-10-02 |
| 26 | Ma trận Ablation Study | 6 cấu hình đối chứng (M1 đến M6) | Bóc tách độc lập 3 trục: Recurrent, Novelty, Curriculum | [ablation_design.md](04_experiments/ablation_design.md) | 2026-10-02 |
| 27 | Chỉ số Chất lượng Quỹ đạo | Tính chỉ số chuẩn quốc tế SPL | Phạt đường đi vòng vèo, đo độ tiệm cận đường $A^*$ tối ưu | [ablation_design.md](04_experiments/ablation_design.md) | 2026-10-02 |
| 28 | Chuẩn báo cáo số liệu | Median [IQR] + Boxplot | Phản ánh chính xác phân phối lệch của giải thuật di truyền | [statistical_methods.md](04_experiments/statistical_methods.md) | 2026-10-02 |
| 29 | Kiểm định thống kê | Mann-Whitney U Test + Bonferroni Correction | Khẳng định tính vượt trội có ý nghĩa thống kê ($p < 0.05$) | [statistical_methods.md](04_experiments/statistical_methods.md) | 2026-10-02 |
| 30 | Phân tích thất bại (Failure Analysis) | Dành riêng 1 Slide phân tích 3 ca lỗi điển hình | Thể hiện tư duy khoa học trung thực, định hướng Future Work | [failure_analysis.md](04_experiments/failure_analysis.md) | 2026-10-02 |
| 31 | Đánh giá độ bền (Robustness) | Thêm 5% nhiễu LiDAR + 2% trượt bánh xe | Kiểm tra tính bền bỉ, sẵn sàng chuyển giao Sim-to-Real | [failure_analysis.md](04_experiments/failure_analysis.md) | 2026-10-02 |

## 🟣 Phản biện & Scope (05_defense/)

| # | Vấn đề | Quyết định | Lý do | File tham khảo | Ngày |
|---|--------|-----------|-------|----------------|------|
| 32 | Lập luận bảo vệ 2D | Slide riêng: Ngân sách $600k$ evals & Stanley Benchmark | Đánh phủ đầu câu hỏi về 3D, chứng minh tính khả thi khoa học | [scope_negotiation.md](05_defense/scope_negotiation.md) | 2026-10-02 |
| 33 | Định vị NEAT vs Deep RL | So sánh trực diện Occam's Razor & Chip nhúng | Mạng 3-6 nút ẩn, chạy trên ESP32 $2, miễn nhiễm bẫy chữ U | [scope_negotiation.md](05_defense/scope_negotiation.md) | 2026-10-02 |
| 34 | Slide Backup Q&A | Soạn sẵn 5 Slide Backup Q&A minh họa chi tiết | Bấm nhảy slide lập tức khi giám khảo hỏi, tạo ấn tượng áp đảo | [qa_preparation.md](05_defense/qa_preparation.md) | 2026-10-02 |
| 35 | Kế hoạch Tương lai | Mở rộng Vật cản di động & Hybrid RL Fine-tuning | Thể hiện tầm nhìn dài hạn và tiềm năng ứng dụng thực tế | [qa_preparation.md](05_defense/qa_preparation.md) | 2026-10-02 |
