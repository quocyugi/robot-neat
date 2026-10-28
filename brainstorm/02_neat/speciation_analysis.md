# PHÂN TÍCH CƠ CHẾ PHÂN LOÀI (SPECIATION), KHOẢNG CÁCH TƯƠNG THÍCH VÀ XỬ LÝ TRÌ TRỆ
> **Ngày**: 2026-10-02  
> **Trạng thái**: [DECIDED]  
> **Liên quan**: overview.md phần 3.2  

---

## 1. Câu hỏi gốc tôi đã hỏi LLM
> "Phân tích chi tiết cơ chế phân loài (Speciation), công thức khoảng cách tương thích di truyền delta, chuẩn hóa số gen N, cơ chế chia sẻ fitness và chính sách xử lý trì trệ (stagnation) trong NEAT."

---

## 2. Tóm tắt kết luận rút ra (Key Takeaways)
- **Bảo vệ đột biến cấu trúc non trẻ**: Phân loài là "vườn ươm" giúp các kết nối và nơ-ron mới sinh ra có không gian tối ưu trọng số mà không bị các mạng cũ đào thải sớm.
- **Chuẩn hóa động $N$**: Chuẩn hóa $N = \max(|G_A|, |G_B|)$ khi số kết nối $\ge 20$, và giữ $N = 1$ khi mạng còn tối giản ở các thế hệ đầu để tránh làm méo mó khoảng cách tương thích khi đồ thị mạng còn nhỏ.
- **Explicit Fitness Sharing**: Chia nhỏ điểm thích nghi cho quy mô loài $f'_i = f_i / |S_s|$, ngăn chặn tuyệt đối hiện tượng độc chiếm tài nguyên sinh sản của một loài duy nhất.
- **Chính sách Trì trệ Cân bằng**: Đặt `max_stagnation = 15` thế hệ kết hợp bảo tồn `species_elitism = 2`. Đảm bảo đào thải dứt khoát các nhánh tiến hóa bế tắc để giải phóng CPU, đồng thời luôn duy trì ít nhất 2 loài có thành tích tốt nhất lịch sử.

---

## 3. Công thức Toán học & Triển khai Kỹ thuật

### 3.1. Khoảng cách Tương thích Di truyền (Compatibility Distance)
$$\delta = \frac{c_1 E}{N} + \frac{c_2 D}{N} + c_3 \bar{W}$$

Trong đó:
* $E$: Số gen dư thừa (Excess)
* $D$: Số gen rời rạc (Disjoint)
* $\bar{W}$: Độ lệch trọng số trung bình của các gen trùng khớp
* Hệ số: $c_1 = 1.0, \; c_2 = 1.0, \; c_3 = 0.5$
* Hệ số chuẩn hóa kích thước:
  $$N = \begin{cases} 1 & \text{nếu } \max(|G_A|, |G_B|) < 20 \\ \max(|G_A|, |G_B|) & \text{nếu } \max(|G_A|, |G_B|) \ge 20 \end{cases}$$

### 3.2. Phân bổ Chỉ tiêu Sinh sản (Offspring Allocation)
Số con thế hệ tiếp theo cấp cho loài $s$:
$$n_s = \frac{\bar{f}'_s}{\sum_{k=1}^{K} \bar{f}'_k} \cdot N_{\text{pop}}, \quad \text{với } \bar{f}'_s = \frac{1}{|S_s|} \sum_{i \in S_s} f'_i$$

### 3.3. Module Giám sát Phân loài (Speciation Logger)
```python
def log_speciation_stats(population, generation: int):
    """
    Theo dõi phân bố các loài trong từng thế hệ
    """
    species_set = population.species
    num_species = len(species_set.species)
    species_sizes = [len(s.members) for s in species_set.species.values()]
    stagnation_ages = [s.fitness_history for s in species_set.species.values()]
    
    print(f"[Gen {generation}] Số lượng loài: {num_species} | Phân bố kích thước: {species_sizes}")
    return {
        "num_species": num_species,
        "max_size": max(species_sizes) if species_sizes else 0,
        "min_size": min(species_sizes) if species_sizes else 0
    }
```

---

## 4. Quyết định của tôi
- [x] **Chọn Phương án A cho Chuẩn hóa $N$**: Kích hoạt chuẩn hóa động khi gen $\ge 20$. Giúp thuật toán đo lường chính xác khoảng cách di truyền ở cả 2 giai đoạn: khi mạng mới bắt đầu (tối giản) và khi mạng đã phân nhánh phức tạp.
- [x] **Chọn Phương án A cho Xử lý Trì trệ**: Cố định `max_stagnation = 15` và `species_elitism = 2`. Vừa dọn dẹp các nhánh cụt, vừa không lo bị tuyệt chủng đột ngột toàn quần thể.

---

## 5. TODO tiếp theo
- [ ] Hoàn thành 100% nhóm 02_neat.
- [ ] Chuyển sang Nhóm 3: **`03_novelty/behavior_descriptor.md`** (Thiết kế bộ mô tả hành vi $b(x)$, trajectory sampling, và KD-Tree Archive).
