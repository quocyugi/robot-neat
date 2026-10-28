# TINH CHỈNH SIÊU THAM SỐ CẤU HÌNH NEAT-PYTHON (CONFIG TUNING)
> **Ngày**: 2026-10-02  
> **Trạng thái**: [DECIDED]  
> **Liên quan**: overview.md phần 3.1, 3.2, 3.3  

---

## 1. Câu hỏi gốc tôi đã hỏi LLM
> "Tinh chỉnh các siêu tham số trong file config của neat-python: quy mô quần thể, cơ chế kiểm soát số lượng loài, tỉ lệ đột biến cấu trúc và hàm kích hoạt."

---

## 2. Tóm tắt kết luận rút ra (Key Takeaways)
- **Quy mô quần thể chuẩn mực**: $N = 150$, giữ nguyên theo thiết lập kinh điển của Kenneth Stanley (2002), cân bằng giữa độ phong phú di truyền và tốc độ mô phỏng.
- **Adaptive Speciation Threshold**: Không cố định $\delta_t$, mà dùng cơ chế thích nghi điều chỉnh $\delta_t$ tăng/giảm động theo bước $0.1$ nhằm giữ ổn định số lượng loài trong khoảng **$8 - 15$ loài**, bảo toàn các ngách sinh thái (ecological niches).
- **Hàm kích hoạt Tanh toàn mạng**: Sử dụng `tanh` cho cả tầng ẩn và tầng ra. Tính đối xứng qua gốc $0$ của `tanh` giúp việc điều khiển 2 bánh vi sai (rẽ trái/phải, tiến/lùi) mượt mà và trực quan hơn so với sigmoid/relu.
- **Phức tạp hóa dần (Gradual Complexification)**: Bắt đầu từ mạng tối giản (15 input $\to$ 2 output, 0 nút ẩn). Tỉ lệ đột biến cấu trúc giữ ở mức kiểm soát: $p_{\text{add\_conn}} = 0.05$, $p_{\text{add\_node}} = 0.03$ để mạng có đủ thời gian tối ưu trọng số trước khi phình to.
- **Bảo tồn tinh hoa (Elitism = 2)**: Mỗi loài giữ lại 2 cá thể vô địch sao chép nguyên vẹn sang thế hệ sau.

---

## 3. File Cấu hình Mẫu Chuẩn (`config-neat.ini`)

```ini
[NEAT]
fitness_criterion     = mean
fitness_threshold     = 18.0
pop_size              = 150
reset_on_extinction   = True

[DefaultGenome]
# Không gian đầu vào và đầu ra
num_inputs              = 15
num_hidden              = 0
num_outputs             = 2
initial_connection      = full_direct
feed_forward            = True

# Hàm kích hoạt (Activation)
activation_default      = tanh
activation_mutate_rate  = 0.0
activation_options      = tanh

# Hàm tổng hợp tín hiệu (Aggregation)
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# Đột biến cấu trúc (Topology Mutations)
conn_add_prob           = 0.05
conn_delete_prob        = 0.01
node_add_prob           = 0.03
node_delete_prob        = 0.01

# Đột biến trọng số liên kết (Weight Mutations)
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30.0
weight_min_value        = -30.0
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

# Đột biến ngưỡng kích hoạt (Bias Mutations)
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# Hệ số tương thích loài (Speciation Distance)
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 15
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
min_species_size   = 2
```

---

## 4. Quyết định của tôi
- [x] **Chọn Phương án A cho Phân loài**: Dùng Adaptive Speciation Threshold để giữ số loài trong dải $8 - 15$. Tránh hiện tượng một siêu cá thể độc chiếm toàn bộ quần thể, bảo vệ các cấu trúc mạng nơ-ron mới sinh.
- [x] **Chọn Phương án A cho Hàm kích hoạt**: Cố định hàm `tanh` cho mọi nơ-ron. Miền giá trị $[-1, 1]$ hoàn toàn tương thích với cả dải góc lệch $(\sin\varphi, \cos\varphi)$ lẫn dải vận tốc bánh xe.

---

## 5. TODO tiếp theo
- [ ] Lưu file `configs/neat_feedforward.ini` và `configs/neat_recurrent.ini` vào thư mục dự án.
- [ ] Brainstorm tiếp chủ đề `02_neat/recurrent_vs_ff.md`: So sánh Feedforward NEAT vs Recurrent NEAT (Mạng hồi quy có bộ nhớ trong để thoát bẫy chữ U).
