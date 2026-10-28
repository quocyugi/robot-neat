# THIẾT KẾ PIPELINE SINH BẢN ĐỒ TỰ ĐỘNG & KIỂM ĐỊNH LIÊN THÔNG BẰNG A*
> **Ngày**: 2026-10-02  
> **Trạng thái**: [DECIDED]  
> **Liên quan**: overview.md phần 5.1 & 5.2  

---

## 1. Câu hỏi gốc tôi đã hỏi LLM
> "Thiết kế Pipeline sinh bản đồ tự động 4 cấp độ & Thuật toán kiểm tra tính liên thông A*, cơ chế sinh bẫy chữ U và chiến lược lưu trữ dữ liệu."

---

## 2. Tóm tắt kết luận rút ra (Key Takeaways)
- **Pre-generation Dataset**: Sinh sẵn và lưu trữ toàn bộ tập bản đồ ra định dạng `.json` (1,000 bản đồ train, 100 validation, 200 test) thay vì sinh on-the-fly lúc train. Điều này giúp loại bỏ hoàn toàn độ trễ của $A^*$ trong vòng lặp tiến hóa và đảm bảo tính tái lập 100%.
- **C-Space Inflation**: Khi chạy $A^*$, vật cản được phình to bán kính $r_{\text{inflation}} = r_{\text{robot}} + \text{margin} = 0.25\text{m} + 0.05\text{m} = 0.30\text{m}$ trên lưới $5\text{cm}$ ($200 \times 200$ cells) để đảm bảo robot kích thước thật chắc chắn đi lọt.
- **Tiêu chuẩn nghiệm ngặt (Re-roll Policy)**: Bản đồ sinh ra bắt buộc thỏa mãn 2 điều kiện: (1) $A^*$ tìm được đường hợp lệ và (2) Hệ số quanh co $\tau = L_{A^*} / d_{\text{Euclid}}$ phải nằm đúng dải quy định của cấp độ. Nếu vi phạm $\to$ Hủy và sinh lại với seed mới.
- **Bẫy chữ U có chủ đích (Oriented U-Trap)**: Đáy bẫy được sinh vuông góc với vector nối $S \to G$ và đặt ở khoảng cách $40\% - 60\%$ hành trình, hai cánh chữ U mở về phía Start để đảm bảo tạo ra lực hút deceptive fitness tối đa.

---

## 3. Chi tiết Kỹ thuật & Thiết kế Cấu trúc

### 3.1. Bảng Thông số 4 Cấp độ Bản đồ
| Cấp độ | Tên gọi | Đặc trưng địa hình | Hệ số quanh co $\tau$ | Quy tắc vật cản |
| :---: | :--- | :--- | :---: | :--- |
| **1** | **Empty** | Chỉ có 4 tường biên bao $10\text{m} \times 10\text{m}$ | $\tau = 1.00$ | 0 vật cản bên trong, $d(S, G) \ge 5\text{m}$ |
| **2** | **Sparse** | Các khối hộp/trụ tròn rải rác | $1.05 \le \tau \le 1.25$ | 5 – 8 khối, khe hở $\ge 0.8\text{m}$ |
| **3** | **Corridors** | Các bức tường zíc-zắc tạo hẻm nhỏ | $1.25 < \tau \le 1.60$ | 3 – 5 vách tường dài, khe hẹp $0.6 - 0.9\text{m}$ |
| **4** | **U-Traps** | Bẫy chữ U chắn trực diện đường thẳng $S-G$ | $\tau > 1.60$ | 1–2 bẫy chữ U + vật cản phân tán |

### 3.2. Thuật toán Sinh Bẫy chữ U (Oriented U-Pocket)
```python
def generate_u_trap(start_pos, goal_pos, trap_width=2.0, trap_depth=2.0, margin=0.5):
    """
    Tạo bẫy chữ U có đáy chắn ngang đường nối Start -> Goal, miệng bẫy quay về Start
    """
    # Vector đơn vị từ Start tới Goal
    vec = goal_pos - start_pos
    dist = np.linalg.norm(vec)
    u_dir = vec / dist
    ortho_dir = np.array([-u_dir[1], u_dir[0]])  # Vuông góc

    # Đặt tâm đáy bẫy tại 50% quãng đường
    trap_center = start_pos + 0.5 * vec

    # Tọa độ 2 góc của đáy chữ U
    bottom_left = trap_center - 0.5 * trap_width * ortho_dir
    bottom_right = trap_center + 0.5 * trap_width * ortho_dir

    # Hai cánh vươn ngược về phía Start
    wing_left = bottom_left - trap_depth * u_dir
    wing_right = bottom_right - trap_depth * u_dir

    # 3 đoạn thẳng tạo thành chữ U
    segments = [
        [bottom_left, bottom_right],  # Đáy bẫy
        [bottom_left, wing_left],      # Cánh trái
        [bottom_right, wing_right]     # Cánh phải
    ]
    return segments
```

### 3.3. Định dạng Schema File JSON Lưu Bản đồ
Mỗi file được lưu tại `data/maps/level_{1..4}/map_{seed}.json`:
```json
{
  "map_id": "lvl4_seed_00123",
  "difficulty_level": 4,
  "seed": 123,
  "bounds": [10.0, 10.0],
  "start": [1.5, 1.5],
  "goal": [8.5, 8.5],
  "robot_radius": 0.25,
  "inflation_radius": 0.30,
  "metrics": {
    "euclidean_distance": 9.899,
    "optimal_path_length": 16.85,
    "tortuosity": 1.702,
    "grid_resolution": 0.05
  },
  "segments": [
    [[0.0, 0.0], [10.0, 0.0]],
    [[10.0, 0.0], [10.0, 10.0]],
    [[10.0, 10.0], [0.0, 10.0]],
    [[0.0, 10.0], [0.0, 0.0]],
    [[4.0, 5.0], [6.0, 5.0]],
    [[4.0, 5.0], [4.0, 3.0]],
    [[6.0, 5.0], [6.0, 3.0]]
  ]
}
```

---

## 4. Quyết định của tôi
- [x] **Chọn Phương án A cho việc lưu trữ**: Pre-generate toàn bộ dataset (1,000 train + 100 val + 200 test) dạng JSON. Tải bản đồ vào RAM trong $< 0.1\text{ms}$, giúp NEAT tập trung 100% tài nguyên CPU cho việc đánh giá mạng nơ-ron.
- [x] **Chọn Phương án A cho tiêu chuẩn chấp nhận**: Lọc kép ngặt nghèo bằng cả $A^*$ liên thông và dải giá trị quanh co $\tau$. Đảm bảo tính nhất quán khoa học cho Curriculum Learning.

---

## 5. TODO tiếp theo
- [ ] Viết module `src/env/map_generator.py` và script tạo dataset `scripts/pregenerate_maps.py`.
- [ ] Brainstorm tiếp chủ đề cuối cùng của nhóm 01: Thiết kế `RobotNavEnv` chuẩn Gymnasium API (`01_env/gymnasium_api.md`).
