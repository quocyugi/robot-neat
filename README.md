# 🤖 NEAT-Based Autonomous Robot Navigation

> **Học phần**: Tính toán tiến hóa (Evolutionary Computation) — IT4906  
> **Đề tài**: Điều hướng robot tự hành trong môi trường có vật cản bằng Neuroevolution (NEAT)  
> **Nhóm**: Nhóm 3

---

## 📋 Mô tả dự án

Dự án sử dụng thuật toán **NEAT (NeuroEvolution of Augmenting Topologies)** để tiến hóa bộ điều khiển mạng nơ-ron cho robot vi sai 2 bánh, điều hướng tự động trong không gian 2D 10m × 10m có vật cản tĩnh.

### Đặc điểm nổi bật
- **Tiến hóa đồng thời** cả cấu trúc mạng (topology) và trọng số (weights)
- **Novelty Search** lai ghép thích nghi (Adaptive Hybrid) nhắm vào bẫy chữ U (deceptive fitness); benchmark sau sửa lỗi chưa cho thấy nó giúp NEAT ([docs/benchmark_sau_sua_loi.md](docs/benchmark_sau_sua_loi.md))
- **Curriculum Learning** 4 cấp độ khó (Empty → Sparse → Corridors → U-Traps)
- **3 baseline** đối chuẩn (GA B1, GA B2, APF) + Ablation Study 6 cấu hình (M1–M6)
- Kiểm định thống kê nghiêm ngặt (Mann-Whitney U + Bonferroni correction)

---

## 🏗️ Cấu trúc dự án

```
robot-neat/
├── configs/                    # Cấu hình NEAT (feedforward, recurrent; neat_lehman_ref.ini cho LNS_ref; *_n24, *_n36: cảm biến 24, 36 hướng)
├── data/raw/                   # Dữ liệu gốc tải về (BARN, mê cung Lehman) — không đưa lên git
├── data/maps/                  # Bản đồ đã xử lý (có sẵn trong repo): train/val/test, test_barn, barn_all, test_lehman, manifest.csv
├── src/
│   ├── env/                    # Môi trường Gymnasium 2D
│   │   ├── kinematics.py       #   Hằng số vật lý & động học vi sai
│   │   ├── lidar.py            #   LiDAR 12 tia vectorised ray-casting (tùy chọn: N hướng hình quạt)
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
│   │   ├── novelty.py          #   NoveltyArchive + AdaptiveHybridManager + LehmanArchive (luật Lehman & Stanley)
│   │   ├── curriculum.py       #   Curriculum + lịch học đều 4 cấp
│   │   ├── parallel.py         #   Đánh giá song song (multiprocessing)
│   │   ├── neat_fixes.py       #   Vá quanh neat-python 2.0.0 (khóa nút ẩn, ...)
│   │   └── runner.py           #   Vòng lặp tiến hóa chính (orchestrator)
│   ├── batch/                  # Đánh giá theo lô (tùy chọn): NumPy trên CPU, CuPy trên GPU, trùng từng bit
│   ├── baselines/              # Phương pháp đối chuẩn
│   │   ├── fixed_ga.py         #   B1/B2: Fixed MLP + GA
│   │   └── apf.py              #   B5: Trường thế nhân tạo
│   ├── experiments/            # Thực nghiệm & thống kê
│   │   ├── stats.py            #   Mann-Whitney U + Boxplot
│   │   ├── evaluation.py       #   Chấm bộ điều khiển trên các tập test
│   │   └── run_ablation.py     #   Chạy Ablation M1–M6 (và LNS, LNS_ref: novelty search kiểu Lehman)
│   ├── console.py              # Ghi UTF-8 khi output chuyển ra file hoặc pipe
│   └── visual_report.py        # Ảnh bảng, ảnh biểu đồ (PNG) và trang HTML của các kết quả
├── scripts/                    # Entry points
│   ├── download_public_data.py #   Tải BARN + mê cung Lehman–Stanley
│   ├── pregenerate_maps.py     #   Sinh 1.300 bản đồ
│   ├── preprocess_data.py      #   Tiền xử lý + kiểm tra chất lượng dữ liệu
│   ├── train_neat.py           #   Huấn luyện NEAT
│   ├── evaluate.py             #   Chấm champion trên tập test
│   ├── run_baselines.py        #   Huấn luyện + chấm baseline GA B1, B2 và APF (B5)
│   ├── make_sensor_configs.py  #   Sinh configs/neat_*_n{N}.ini cho cảm biến N hướng
│   ├── summarize_results.py    #   Gom seed, kiểm định, boxplot
│   ├── plot_benchmark.py       #   Biểu đồ, ảnh bảng, trang HTML so sánh các phương pháp
│   ├── demo.py                 #   Cửa sổ demo (HUD tiếng Việt, xuất GIF)
│   └── visualize.py            #   Replay genome trên Pygame
├── tests/                      # Kiểm thử các bản sửa lỗi
├── lehman_maze/                # Cài lại thí nghiệm mê cung Lehman–Stanley (novelty search), xem lehman_maze/README.md
├── brainstorm/                 # Kết quả brainstorm (35 quyết định)
├── docs/tien_xu_ly_du_lieu.md  # Báo cáo phần tiền xử lý dữ liệu
├── docs/sua_loi_code.md        # Các lỗi đã sửa (mã L01–L34) và cách kiểm
├── docs/sua_cau_hinh_NEAT_GA.md # Cấu hình NEAT/GA đã sửa và kết quả kiểm thử
├── docs/benchmark_sau_sua_loi.md # Kết quả benchmark sau sửa lỗi (NEAT, GA, APF)
├── docs/benchmark/             # Bảng, biểu đồ, trang HTML và số liệu benchmark
├── docs/huong_dan_demo_vscode.md # Hướng dẫn demo bằng VS Code
├── demo/                       # Champion mặc định của demo (demo/README.md)
├── .vscode/                    # Cấu hình Run and Debug cho demo
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

# 2b. Sinh bộ bản đồ tự sinh: 1.000 train + 100 val + 200 test, 4 cấp độ khó
python scripts/pregenerate_maps.py --data-dir data/maps

# 2c. Tiền xử lý: chuyển BARN/Lehman về schema dự án, kiểm tra chất lượng, manifest, hình
python scripts/preprocess_data.py
```
Kết quả: `data/maps/{train,val,test}/level_k/`, `data/maps/test_barn/` (50 bản đồ BARN),
`data/maps/barn_all/` (đủ 300 bản đồ BARN), `data/maps/test_lehman/` (2 mê cung), `data/maps/manifest.csv` và
`reports/data/`.

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

# Novelty search thuần kiểu Lehman & Stanley 2011 (LNS): luật archive của bài báo, không curriculum
python scripts/train_neat.py --config configs/neat_recurrent.ini \
    --novelty --no-curriculum --novelty-rule lehman --seed 0 --workers 4
```
`--val-key per_level` đổi khóa chọn champion thành trung bình SR val của từng cấp (mỗi bộ dữ liệu nặng như nhau),
dùng khi các thư mục cấp có số bản đồ val khác nhau; mặc định là SR trên mọi bản đồ val gộp lại.
`--levels` chọn các thư mục cấp (`train/level_k`, `val/level_k`) trong `--data-dir` dùng để học và chọn champion
(mặc định `1,2,3,4`); ví dụ `--levels 1` cho một thư mục dữ liệu chỉ đặt BARN ở `level_1`, `--levels 1,2,3,4,5,6`
cho sáu bộ.
Kiểu mạng (hồi quy hay truyền thẳng) suy ra từ `feed_forward` của file config. Mỗi thế hệ dùng chung
8 bản đồ cho cả quần thể; fitness đo tiến độ theo đường đi thật (trường geodesic, lưu đệm ở `output/geo_cache/`).
Kết quả trong `output/train/`: `champion_genome.pkl` (genome có SR cao nhất trên bản đồ val của các cấp đang học, mặc định cả 4 cấp),
`summary.json` (cấp cao nhất đã học, các lần lên cấp, SR val của champion), `training_history.json`, `neat_config.ini`,
và log huấn luyện dạng hình: `training_history_bieu_do.png` (biểu đồ theo thế hệ), `training_history_bang.png`
(bảng), `training_history.html` (trang mở bằng trình duyệt).

**Cảm biến hình quạt (tùy chọn, mặc định tắt).** Mặc định robot đo 12 tia điểm. Đặt hai biến môi trường TRƯỚC khi
chạy script: `NEAT_LIDAR_MODE=sector` (mặc định `rays`) để mỗi đầu vào đo khoảng cách nhỏ nhất tới vật cản trong cả
hình quạt 360°/N quanh hướng đó (vật cản mảnh lọt giữa hai tia điểm vẫn được thấy), `NEAT_LIDAR_N=N` (mặc định 12)
để đổi số hướng; quan sát có N + 3 chiều. Config NEAT phải có `num_inputs = N + 3`: N = 24, 36 dùng
`configs/neat_*_n24.ini`, `configs/neat_*_n36.ini` (bản sao file gốc, chỉ đổi `num_inputs`; sinh lại bằng
`python scripts/make_sensor_configs.py 24 36`), N = 12 dùng file gốc. GA (`scripts/run_baselines.py`) tự lấy lớp vào
N + 3, APF chạy với mọi N. `summary.json` ghi `"sensor": {"mode": ..., "n": ...}` khi cảm biến khác mặc định; chấm
lại champion phải đặt đúng hai biến đó (`scripts/evaluate.py` dừng nếu lệch). Không đặt biến nào thì code chạy trùng
từng bit với bản chỉ có 12 tia điểm.
```bash
# PowerShell: $env:NEAT_LIDAR_MODE = "sector"; $env:NEAT_LIDAR_N = "24"
NEAT_LIDAR_MODE=sector NEAT_LIDAR_N=24 python scripts/train_neat.py --config configs/neat_recurrent_n24.ini \
    --no-novelty --no-curriculum --seed 0 --workers 4
```

### 4. Chấm trên tập test và xem kết quả
```bash
python scripts/evaluate.py --run-dir output/train          # SR, CR, TC, hết giờ, SPL, MTT trên mọi tập test
python scripts/visualize.py --genome output/train/champion_genome.pkl \
    --map data/maps/test/level_4/map_200000.json
```
`evaluate.py` ghi `test_results.json` và, cạnh nó, `test_results_bang.png`, `test_results_bieu_do.png`,
`test_results.html` (bảng và biểu đồ TỚI ĐÍCH / VA CHẠM / HẾT GIỜ từng tập; `--no-visual` để bỏ). Các ảnh và trang
HTML của buổi demo: [docs/huong_dan_demo_vscode.md](docs/huong_dan_demo_vscode.md), mục *Kết quả dạng hình*.

### 5. Chạy Ablation Study (10 seeds × 6 cấu hình) và baseline
```bash
python -m src.experiments.run_ablation --seeds 10 --workers 4          # output/ablation/M*/seed_*/
python -m src.experiments.run_ablation --models LNS,LNS_ref --seeds 5 --workers 4   # novelty search kiểu Lehman
python scripts/run_baselines.py --baseline ga_b1 --seed 0 --workers 4   # cũng có ga_b2, apf
python scripts/evaluate.py --run-dir output/ablation/M6/seed_0
python scripts/summarize_results.py --runs M6=output/ablation/M6 GA=output/baselines/ga_b1 --set test/level_3
```
Baseline GA dùng cùng fitness, cùng số bản đồ mỗi thế hệ và cùng cách chọn champion với NEAT; muốn so công bằng (Q1)
thì đặt cùng lịch học (`--schedule` của `run_baselines.py`) và cùng số thế hệ (`--generations`: mặc định của
`run_ablation` là 300, của `run_baselines.py` là 150; benchmark dùng 150 cho mọi phương pháp, tức 180.000 lượt đánh
giá mỗi lượt chạy). `run_baselines.py` cũng nhận `--levels` và `--val-key` như `train_neat.py`. APF (B5) không học,
chỉ chấm trên tập test.

### 6. Kiểm thử các bản sửa lỗi
```bash
python tests/test_fixes.py          # 16 phép thử nhanh, mỗi phép một mã lỗi hoặc một luật (vài giây)
python tests/test_reproducible.py   # cùng seed cho cùng log với 1 hay 4 worker (1–3 phút)
```

### 7. Demo và kết quả benchmark
- Demo bằng VS Code (cấu hình Run and Debug, kịch bản 10–15 phút): [docs/huong_dan_demo_vscode.md](docs/huong_dan_demo_vscode.md);
  champion mặc định của demo: [demo/README.md](demo/README.md).
- Kết quả benchmark trên code đã sửa (NEAT R0, M1–M6; GA B1, GA B2; APF; cấp 1–4, BARN, mê cung Lehman–Stanley):
  [docs/benchmark_sau_sua_loi.md](docs/benchmark_sau_sua_loi.md).

### 8. Chạy theo lô và trên GPU (tùy chọn)
Mặc định mỗi lượt chạy riêng (`ParallelEvaluator`, `--workers` tiến trình) và **không đổi gì**. Tùy chọn `--backend`
của `train_neat.py`, `run_baselines.py`, `evaluate.py` và `python -m src.experiments.run_ablation` (hoặc tham số
`backend` của `NEATRunner`, `evaluate_sets`, `parallel.make_evaluator`; hay biến môi trường `NEAT_EVAL_BACKEND`, tham số
tường minh thắng biến) chọn cách đánh giá:
- `cpu` (mặc định): như cũ;
- `batch`: mọi lượt của một lần đánh giá chạy cùng lúc theo lô bằng NumPy (`src/batch/`); `--workers W > 1` chia lô cho
  W tiến trình;
- `gpu`: như `batch`, nhưng LiDAR và va chạm chạy trên GPU NVIDIA bằng CuPy (kernel biên dịch `--fmad=false`), phần còn
  lại ở CPU; chạy trong một tiến trình (bỏ qua `--workers`). Không nên cho nhiều job cùng dùng GPU (mã không chặn việc
  này; job hết bộ nhớ GPU sẽ dừng với lỗi `GpuUnavailable`): các job chạy song song dùng `batch`.
```bash
python scripts/train_neat.py --config configs/neat_recurrent.ini --no-novelty --no-curriculum --seed 0 --backend gpu
python scripts/run_baselines.py --baseline ga_b1 --seed 0 --backend batch --workers 4
```
`batch` và `gpu` chạy ở chế độ exact, được thiết kế để từng lượt cho đúng cùng dict kết quả với mặc định (cùng thứ tự
phép tính, cùng kiểu float32/float64, `math.tanh` và tổng bù của `sum()` như neat-python), tức cùng lịch sử tiến hóa,
champion và kết quả test, **khi chạy trong cùng môi trường Python/NumPy** (môi trường khác, ví dụ phiên bản NumPy hay
SciPy khác, có thể cho số khác ngay cả với đường mặc định). Đã kiểm ngày 26/11/2026:
`tests/test_batch_backend.py --pool --gpu` đạt 16/16; `tests/verify_batch_full.py --device cpu,gpu --workers 4` cho 0 lệch
với `rays:12`, `sector:12` và `sector:24` (cấp 1–4, BARN, Lehman; cả khi đổi thứ tự, chia khúc và chạy Pool 4 tiến trình);
huấn luyện 3 thế hệ có chấm test (R0, GA B1, M6, LNS với `rays:12`; R0, GA B1, LNS với `sector:12`, `sector:24`) cho
cùng `training_history.json`, champion, `summary.json` và `test_results.json` ở `cpu` 5 worker, `batch` 1 và 5 worker,
`gpu`, và ở bản mã đóng băng trước khi có backend. Với `rays:12`, hai môi trường NumPy 2.4.6 / SciPy 1.17.1 và
NumPy 2.5.5 / SciPy 1.18.1 cũng cho cùng số. Lúc khởi tạo, chương trình tự kiểm các giả định số học trên
máy đang chạy và so trọn vài lượt trên bản đồ tổng hợp với đường từng lượt (lệch thì in cảnh báo và quay về `cpu`; đặt
biến môi trường `NEAT_EVAL_STRICT=1` để báo lỗi thay vì quay về); `summary.json` (và file kết quả của `evaluate.py`) ghi
`"backend"` khi khác mặc định. Sửa môi trường mô phỏng (`src/env`, ví dụ `RobotNavEnv.step`), `compute_episode_fitness`,
`run_episode` hay cách dùng kết quả lượt (ví dụ công thức novelty đọc thêm khóa) thì phải chạy lại
`python tests/test_batch_backend.py` trước khi dùng `batch` / `gpu`. Kiểm nhanh: `python tests/test_batch_backend.py`
(`--pool`, `--gpu` để thêm Pool và GPU); kiểm đầy đủ trước khi dùng cho số liệu báo cáo:
`python tests/verify_batch_full.py --device cpu,gpu` (mọi cấp, BARN, Lehman, cảm biến `rays:12`, `sector:12`,
`sector:24`, phải 0 lệch). `--inexact` (chỉ với `gpu`) tính cả hàm siêu việt trên GPU: KHÔNG trùng bit, kết cục vài
lượt đổi, chỉ để thử, không dùng cho benchmark.

Thời gian một thế hệ 150 genome × 8 bản đồ (1.200 lượt), đo ngày 26/11/2026 trên mã này bằng
`tests/verify_batch_full.py --timing` (i5-14400F, RTX 3060; máy đang chạy benchmark nền ở mức Idle nên số CPU chỉ để so
tương đối; cột "từng lượt" ước từ 24 genome; số bước trung bình mỗi lượt phụ thuộc các genome được đo nên khác nhau giữa
các dòng):

| bản đồ, cảm biến | TB bước/lượt | từng lượt, 1 tiến trình | `batch`, 1 tiến trình | `batch`, Pool 4 | `gpu` |
|---|---|---|---|---|---|
| cấp 1–4, rays:12 | 352 | ~79,6 s | 3,8 s | 1,6 s | 2,2 s |
| cấp 1–4, sector:12 | 385 | ~103,6 s | 7,4 s | 2,8 s | 2,5 s |
| cấp 1–4, sector:24 | 190 | ~42,8 s | 6,5 s | 2,5 s | 1,5 s |
| BARN, rays:12 | 235 | ~65,0 s | 19,2 s | 6,6 s | 1,7 s |
| BARN, sector:12 | 464 | ~157,1 s | 80,1 s | 27,3 s | 3,1 s |
| BARN, sector:24 | 195 | ~58,4 s | 63,7 s | 23,2 s | 2,3 s |

Khi huấn luyện thật (3 thế hệ, máy bận như trên), một thế hệ R0 với `rays:12` mất trung vị 13,8 s ở `cpu` 5 worker,
1,4 s ở `batch` 5 worker và 1,6 s ở `gpu`.

Với BARN (bản đồ nhiều đoạn) và cảm biến hình quạt, `batch` một tiến trình có thể chậm hơn đường mặc định nhiều worker:
khi đó dùng `gpu`, hoặc `batch --workers W`.

GPU cần CuPy cho CUDA và không nằm trong `requirements.txt`. Đang dùng (cài với `--no-deps` để giữ nguyên numpy; không
cần cuBLAS):
`python -m pip install --no-deps --only-binary :all: cupy-cuda13x==14.2.0 cuda-pathfinder==1.9.1 nvidia-cuda-runtime==13.5.37 nvidia-cuda-nvrtc==13.5.37`.
Khi thư mục repo có dấu (ví dụ `...\môn tính toán tiến hóa\...`),
NVRTC không mở được header của CuPy: `src/batch/backend.py` tự nạp CuPy qua tên ngắn 8.3 của thư mục `site-packages`,
cách chắc chắn nhất là gọi python của `.venv` bằng đường dẫn ngắn, dạng
`C:\Users\<tên>~1\Desktop\MNTNHT~1\robot-neat\.venv\Scripts\python.exe` (xem tên ngắn của từng thư mục bằng `dir /x`).

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
| Số tia LiDAR | 12 (360°, tầm 3.0 m); tùy chọn N hướng hình quạt (mục 3) |
| Observation space | 15-D liên tục (N + 3 với N hướng) |
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
| LNS | Recurrent | Pure Novelty, luật archive Lehman & Stanley | ✗ | Novelty search thuần theo bài báo gốc, mô tả hành vi trên 4 bản đồ probe (ngoài ma trận, gọi `--models LNS`) |
| LNS_ref | Recurrent | Pure Novelty, luật archive Lehman & Stanley | ✗ | Như LNS, với cấu hình NEAT tham chiếu của thí nghiệm mê cung `configs/neat_lehman_ref.ini` (gọi `--models LNS_ref`) |

### Baselines đối chuẩn
- **B1**: Fixed MLP 1×8 + GA
- **B2**: Fixed MLP 2×16-8 + GA
- **B5**: Artificial Potential Field (APF), không học

---

## 📚 Tham khảo

- Stanley, K.O. & Miikkulainen, R. (2002). *Evolving Neural Networks through Augmenting Topologies*. Evolutionary Computation.
- Lehman, J. & Stanley, K.O. (2011). *Abandoning Objectives: Evolution through the Search for Novelty Alone*. Evolutionary Computation.
- Anderson, P. et al. (2018). *On Evaluation of Embodied Navigation Agents*. arXiv:1807.06757.

---

## 📄 Giấy phép

Dự án phục vụ mục đích học tập tại trường Đại học Bách khoa Hà Nội.
