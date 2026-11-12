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
│   │   ├── geodesic.py         #   Khoảng cách theo đường đi thật tới đích (fitness, L26)
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
│   │   ├── curriculum.py       #   Curriculum + lịch học đều 4 cấp
│   │   ├── parallel.py         #   Đánh giá song song (multiprocessing)
│   │   ├── neat_fixes.py       #   Vá quanh neat-python 2.0.0 (khóa nút ẩn, ...)
│   │   └── runner.py           #   Vòng lặp tiến hóa chính (orchestrator)
│   ├── baselines/              # Phương pháp đối chuẩn
│   │   ├── fixed_ga.py         #   B1/B2: Fixed MLP + GA
│   │   ├── random_search.py    #   B3: Tìm kiếm ngẫu nhiên
│   │   ├── rule_based.py       #   B4: Điều khiển phản xạ
│   │   └── apf.py              #   B5: Trường thế nhân tạo
│   └── experiments/            # Thực nghiệm & thống kê
│       ├── metrics.py          #   SR, SPL, MTT, Collision Rate
│       ├── stats.py            #   Mann-Whitney U + Boxplot
│       ├── evaluation.py       #   Chấm bộ điều khiển trên các tập test
│       └── run_ablation.py     #   Chạy Ablation M1–M6
├── scripts/                    # Entry points
│   ├── download_public_data.py #   Tải BARN + mê cung Lehman–Stanley
│   ├── pregenerate_maps.py     #   Sinh 1,300 bản đồ
│   ├── preprocess_data.py      #   Tiền xử lý + kiểm tra chất lượng dữ liệu
│   ├── train_neat.py           #   Huấn luyện NEAT
│   ├── evaluate.py             #   Chấm champion trên tập test
│   ├── run_baselines.py        #   Huấn luyện + chấm baseline B1–B5
│   ├── summarize_results.py    #   Gom seed, kiểm định, boxplot
│   └── visualize.py            #   Replay genome trên Pygame
├── tests/                      # Kiểm thử các bản sửa lỗi
├── brainstorm/                 # Kết quả brainstorm (35 quyết định)
├── docs/tien_xu_ly_du_lieu.md  # Báo cáo phần tiền xử lý dữ liệu
├── docs/sua_loi_code.md        # Các lỗi đã sửa (mã L01–L34) và cách kiểm
├── docs/sua_cau_hinh_NEAT_GA.md # Cấu hình NEAT/GA đã sửa và kết quả kiểm thử
├── reports/                    # Báo cáo khả thi (Word) và các báo cáo khảo sát
├── kiem_thu/                   # Script và kết quả kiểm thử của báo cáo
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

Bản đồ cấp 3 đã được sinh lại với tường dày 0,20 m nối liền tường biên (lỗi L27, xem
[docs/sua_loi_code.md](docs/sua_loi_code.md)). `manifest.csv` và `reports/data/` vẫn ghi số của bộ cấp 3 cũ;
chạy lại bước 2c (cần tải `data/raw` ở bước 2a) để cập nhật.

### 3. Huấn luyện NEAT

```bash
# Mô hình đề xuất đầy đủ (M6): Recurrent + Novelty + Curriculum
python scripts/train_neat.py --config configs/neat_recurrent.ini \
    --novelty --curriculum --generations 300 --seed 0 --workers 4

# Cấu hình cơ bản (M1): Feedforward, không Novelty, học đều cả 4 cấp
python scripts/train_neat.py --config configs/neat_feedforward.ini \
    --no-novelty --no-curriculum --seed 0 --workers 4
```
Kiểu mạng (hồi quy hay truyền thẳng) suy ra từ `feed_forward` của file config. Mỗi thế hệ dùng chung
8 bản đồ cho cả quần thể; fitness đo tiến độ theo đường đi thật (trường geodesic, lưu đệm ở `output/geo_cache/`).
Kết quả trong `output/train/`: `champion_genome.pkl` (genome có SR cao nhất trên bản đồ val của cả 4 cấp),
`summary.json` (cấp cao nhất đã học, các lần lên cấp, SR val của champion), `training_history.json`, `neat_config.ini`.

### 4. Chấm trên tập test và xem kết quả
```bash
python scripts/evaluate.py --run-dir output/train          # SR, CR, hết giờ, SPL, MTT trên mọi tập test
python scripts/visualize.py --genome output/train/champion_genome.pkl \
    --map data/maps/test/level_4/map_200000.json
```

### 5. Chạy Ablation Study (10 seeds × 6 cấu hình) và baseline
```bash
python -m src.experiments.run_ablation --seeds 10 --workers 4          # output/ablation/M*/seed_*/
python scripts/run_baselines.py --baseline ga_b1 --seed 0 --workers 4   # cũng có ga_b2, random, rule, apf
python scripts/evaluate.py --run-dir output/ablation/M6/seed_0
python scripts/summarize_results.py --runs M6=output/ablation/M6 GA=output/baselines/ga_b1 --set test/level_3
```
Baseline dùng cùng lịch học, cùng số bản đồ mỗi thế hệ, cùng fitness, cùng ngân sách và cùng cách chọn
champion với NEAT, để phép so Q1 công bằng.

### 6. Kiểm thử các bản sửa lỗi
```bash
python tests/test_fixes.py          # 14 phép thử nhanh, mỗi phép một mã lỗi (vài giây)
python tests/test_reproducible.py   # cùng seed cho cùng log với 1 hay 4 worker (1–3 phút)
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
