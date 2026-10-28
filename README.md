# 🤖 NEAT-Based Autonomous Robot Navigation

> **Học phần**: Tính toán tiến hóa (Evolutionary Computation) — IT4906  
> **Đề tài**: Điều hướng robot tự hành trong môi trường có vật cản bằng Neuroevolution (NEAT)  
> **Nhóm**: Nhóm 3

---

## 📋 Mô tả dự án

Dự án sử dụng thuật toán **NEAT (NeuroEvolution of Augmenting Topologies)** để tiến hóa bộ điều khiển mạng nơ-ron cho robot vi sai 2 bánh, điều hướng tự động trong không gian 2D 10m × 10m có vật cản tĩnh.

### Đặc điểm nổi bật
- **Tiến hóa đồng thời** cả cấu trúc mạng (topology) và trọng số (weights)
- **Novelty Search** lai ghép thích nghi (Adaptive Hybrid) giải quyết bẫy chữ U (deceptive fitness)
- **Curriculum Learning** 4 cấp độ khó (Empty → Sparse → Corridors → U-Traps)
- **7 Baselines** đối chuẩn + Ablation Study 6 cấu hình (M1–M6)
- Kiểm định thống kê nghiêm ngặt (Mann-Whitney U + Bonferroni correction)

---

## 🏗️ Cấu trúc dự án

```
robot-neat/
├── configs/                    # Cấu hình NEAT (feedforward & recurrent)
├── data/raw/                   # Dữ liệu gốc tải về (BARN, mê cung Lehman) — không đưa lên git
├── data/maps/                  # Bản đồ đã xử lý (có sẵn trong repo): train/val/test, test_barn, test_lehman, manifest.csv
├── src/
│   ├── env/                    # Môi trường Gymnasium 2D
│   │   ├── kinematics.py       #   Hằng số vật lý & động học vi sai
│   │   ├── lidar.py            #   LiDAR 12 tia vectorised ray-casting
│   │   ├── collision.py        #   Phát hiện va chạm point-to-segment
│   │   ├── renderer.py         #   Pygame renderer (human + rgb_array)
│   │   └── robot_env.py        #   RobotNavEnv chuẩn Gymnasium API
│   ├── maps/                   # Sinh bản đồ & kiểm định
│   │   ├── generator.py        #   MapGenerator 4 cấp độ
│   │   ├── utrap.py            #   Sinh bẫy chữ U/L
│   │   ├── validator.py        #   A* connectivity + tortuosity (+ làm mượt đường)
│   │   ├── map_metrics.py      #   Chỉ số độ khó: τ, khe hẹp nhất, độ sâu bẫy...
│   │   └── public_maps.py      #   Đọc/chuyển BARN & mê cung Lehman–Stanley
│   ├── neat_engine/            # Engine tiến hóa
│   │   ├── evaluator.py        #   Hàm fitness 5 thành phần
│   │   ├── novelty.py          #   NoveltyArchive + AdaptiveHybridManager
│   │   ├── curriculum.py       #   Quản lý thăng cấp độ khó
│   │   └── runner.py           #   Vòng lặp tiến hóa chính (orchestrator)
│   ├── baselines/              # Phương pháp đối chuẩn
│   │   ├── fixed_ga.py         #   B1/B2: Fixed MLP + GA
│   │   ├── random_search.py    #   B3: Tìm kiếm ngẫu nhiên
│   │   ├── rule_based.py       #   B4: Điều khiển phản xạ
│   │   └── apf.py              #   B5: Trường thế nhân tạo
│   └── experiments/            # Thực nghiệm & thống kê
│       ├── metrics.py          #   SR, SPL, MTT, Collision Rate
│       ├── stats.py            #   Mann-Whitney U + Boxplot
│       └── run_ablation.py     #   Chạy Ablation M1–M6
├── scripts/                    # Entry points
│   ├── download_public_data.py #   Tải BARN + mê cung Lehman–Stanley
│   ├── pregenerate_maps.py     #   Sinh 1,300 bản đồ
│   ├── preprocess_data.py      #   Tiền xử lý + kiểm tra chất lượng dữ liệu
│   ├── train_neat.py           #   Huấn luyện NEAT
│   └── visualize.py            #   Replay genome trên Pygame
├── brainstorm/                 # Kết quả brainstorm (35 quyết định)
├── docs/tien_xu_ly_du_lieu.md  # Báo cáo phần tiền xử lý dữ liệu
├── reports/data/               # Thống kê + hình của bộ dữ liệu
├── overview.md                 # Tài liệu thiết kế chi tiết
└── requirements.txt            # Dependencies
```

---

## 🚀 Hướng dẫn cài đặt & chạy

### 1. Tạo môi trường ảo & cài đặt
```bash
git clone https://github.com/quocyugi/robot-neat.git
cd robot-neat

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```
Với Python 3.14, `pygame` chưa có bản cài sẵn: dùng `pip install pygame-ce` (bản thay thế, code vẫn `import pygame`).

### 2. Chuẩn bị dữ liệu bản đồ
Bộ bản đồ đã xử lý (1.602 bản đồ, ~12 MB) **có sẵn trong `data/maps/`**, clone về là dùng được ngay.
Các lệnh dưới đây chỉ cần khi muốn tạo lại từ đầu. Chi tiết từng bước: [docs/tien_xu_ly_du_lieu.md](docs/tien_xu_ly_du_lieu.md)
```bash
# 2a. Tải dữ liệu công khai: BARN (300 môi trường) + 2 mê cung Lehman & Stanley
python scripts/download_public_data.py

# 2b. Sinh bộ bản đồ tự sinh: 1,000 train + 100 val + 200 test, 4 cấp độ khó
python scripts/pregenerate_maps.py --data-dir data/maps

# 2c. Tiền xử lý: chuyển BARN/Lehman về schema dự án, kiểm tra chất lượng, manifest, hình
python scripts/preprocess_data.py
```
Kết quả: `data/maps/{train,val,test}/level_k/`, `data/maps/test_barn/` (50 bản đồ BARN),
`data/maps/test_lehman/` (2 mê cung), `data/maps/manifest.csv` và `reports/data/`.

### 3. Huấn luyện NEAT

```bash
# Mô hình đề xuất đầy đủ (M6): Recurrent + Novelty + Curriculum
python scripts/train_neat.py \
    --config configs/neat_recurrent.ini \
    --recurrent --novelty --curriculum \
    --generations 300

# Cấu hình cơ bản (M1): Feedforward, không Novelty, không Curriculum
python scripts/train_neat.py \
    --config configs/neat_feedforward.ini \
    --no-recurrent --no-novelty --no-curriculum
```

### 4. Xem kết quả
```bash
python scripts/visualize.py \
    --genome output/train/winner_genome.pkl \
    --map data/maps/test/level_4/map_200050.json
```

### 5. Chạy Ablation Study (10 seeds × 6 cấu hình)
```bash
python src/experiments/run_ablation.py
```

---

## 📐 Thông số kỹ thuật chính

| Thông số | Giá trị |
|:---|:---|
| Kích thước arena | 10 m × 10 m |
| Bán kính robot | 0.25 m |
| Khoảng cách trục bánh (L) | 0.40 m |
| Vận tốc tối đa | 0.50 m/s |
| Bước thời gian (Δt) | 0.10 s |
| Số bước tối đa / episode | 500 (50 giây) |
| Số tia LiDAR | 12 (360°, tầm 3.0 m) |
| Observation space | 15-D liên tục |
| Action space | 2-D liên tục [-1, 1]² |
| Population size | 150 |
| Số thế hệ tối đa | 300 |

---

## 🧪 Thiết kế thực nghiệm

### Ma trận Ablation Study
| Mã | Kiến trúc | Tìm kiếm | Curriculum | Mục tiêu |
|:---:|:---:|:---:|:---:|:---|
| M1 | FF | Distance | ✗ | Sàn đối chứng |
| M2 | Recurrent | Distance | ✓ | Vai trò bộ nhớ |
| M3 | FF | Hybrid Novelty | ✓ | Vai trò Novelty |
| M4 | Recurrent | Pure Novelty | ✓ | Hybrid vs Pure |
| M5 | Recurrent | Hybrid Novelty | ✗ | Vai trò Curriculum |
| **M6** | **Recurrent** | **Hybrid Novelty** | **✓** | **Mô hình đề xuất** |

### Baselines đối chuẩn
- **B1**: Fixed MLP 1×8 + GA
- **B2**: Fixed MLP 2×16-8 + GA
- **B3**: Random Search
- **B4**: Heuristic Rule-based
- **B5**: Artificial Potential Field (APF)
- **B6**: A* + DWA (Oracle — có bản đồ toàn cục)
- **B7**: Deep RL PPO/SAC (tùy chọn nâng cao)

---

## 📚 Tham khảo

- Stanley, K.O. & Miikkulainen, R. (2002). *Evolving Neural Networks through Augmenting Topologies*. Evolutionary Computation.
- Lehman, J. & Stanley, K.O. (2011). *Abandoning Objectives: Evolution through the Search for Novelty Alone*. Evolutionary Computation.
- Anderson, P. et al. (2018). *On Evaluation of Embodied Navigation Agents*. arXiv:1807.06757.

---

## 📄 Giấy phép

Dự án phục vụ mục đích học tập tại trường Đại học Bách khoa Hà Nội.
