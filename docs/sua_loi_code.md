# Các lỗi code đã sửa (12/11/2026)

Tài liệu này ghi lại các sửa đổi trong code huấn luyện, môi trường, bản đồ và baseline, theo mã lỗi của báo cáo khả thi (file Word `Bao_cao_pipeline_NEAT_robot.docx` của nhóm, mục 9; file này không đưa lên repo). Phần cấu hình NEAT và GA nằm ở [sua_cau_hinh_NEAT_GA.md](sua_cau_hinh_NEAT_GA.md).

## Kiểm tra đã làm

- `python tests/test_fixes.py`: 14 phép thử, mỗi phép kiểm một mã lỗi, 14/14 qua. Hai phép thử L12 và L20 được chạy lại trên code cũ và đều hỏng ở đó, tức chúng bắt đúng lỗi.
- `python tests/test_reproducible.py`: M6 chạy hai lần cùng seed, một lần 1 worker, một lần 4 worker. Lịch sử huấn luyện và champion giống hệt nhau.
- Chạy thử toàn bộ đường đi, mỗi lượt 3–4 thế hệ:
  - huấn luyện M1, M4, M6 qua `run_ablation`;
  - `evaluate.py` trên 6 tập test;
  - `run_baselines.py` với GA B1, random search, luật rẽ, APF;
  - `summarize_results.py` có kiểm định và boxplot.
- Số của hai bộ điều khiển không học khớp báo cáo: luật rẽ sau khi sửa dấu đạt 80% ở cấp 1, APF đạt 34% trên BARN.
- Tốc độ: M6 (8 bản đồ huấn luyện và 4 bản đồ probe mỗi thế hệ) mất khoảng 10 giây mỗi thế hệ với 6 worker trên i5-14400F, tức khoảng 50 phút cho 300 thế hệ.
- Chưa chạy lượt dài bằng `runner.py`. Các số trong báo cáo đến từ bộ mô phỏng nhanh có cùng logic (script kiểm thử riêng, không đưa lên repo).

## Lỗi chặn thí nghiệm

| Mã | File | Trước | Sau |
|---|---|---|---|
| L02 | `src/neat_engine/runner.py` | Winner là genome có điểm hybrid cao nhất mọi thế hệ; winner M6 chỉ đạt 4% ở cấp 1 | Cứ `val_every` (10) thế hệ và ở thế hệ cuối: lấy 5 genome có fitness **thô** cao nhất, chấm trên 100 bản đồ val (25 mỗi cấp), giữ `deepcopy` của genome có SR val cao nhất (hòa thì so fitness val). Lưu ở `champion_genome.pkl`; `winner_genome.pkl` giữ tên cũ, cùng nội dung |
| L03 | `src/experiments/run_ablation.py`, `src/neat_engine/novelty.py` | M4 "novelty thuần" có tham số y hệt M6 | `AdaptiveHybridManager(fixed_lambda=...)`; M4 dùng `fixed_lambda = 0` (λ nhân với fitness, nên chỉ còn novelty) |
| L04 | `scripts/evaluate.py`, `scripts/run_baselines.py`, `scripts/summarize_results.py`, `src/experiments/evaluation.py` | Không có script kiểm thử và so sánh; `SimpleGA`, `random_search`, APF, luật rẽ, `pairwise_tests` không được gọi | `evaluate.py` chấm champion NEAT, vector GA, APF hoặc luật trên mọi tập test (SR, CR, hết giờ, SPL, MTT). `run_baselines.py` huấn luyện B1, B2, B3 theo đúng giao thức của NEAT và chấm B4, B5. `summarize_results.py` gom theo seed, gọi `pairwise_tests` và `plot_boxplot` |
| L05 | `src/neat_engine/curriculum.py` (`UniformSchedule`) | Tắt curriculum thì chỉ học trên 20 bản đồ đầu của cấp 1 (phòng trống) | Mỗi thế hệ bốc đều từ cả 4 cấp (8 bản đồ = 2 mỗi cấp), dùng chung cho cả quần thể |
| L13 | `src/neat_engine/parallel.py`, `runner.py` | Đánh giá tuần tự | `multiprocessing.Pool` kiểu spawn (chạy được trên Windows), `--workers`. Tiến trình cha chỉ gửi genome và đường dẫn bản đồ; worker tự nạp config, bản đồ, trường geodesic. Kết quả không phụ thuộc số worker |
| L14 | `runner.py`, `curriculum.py`, `novelty.py`, `fixed_ga.py`, `random_search.py`, `run_ablation.py`, `train_neat.py` | `seed` chỉ dùng đặt tên thư mục | Seed đặt cho `random`, `numpy`, `neat.Population(config, seed=...)` và một `numpy.random.Generator` cho mọi lần bốc bản đồ, nhận vào archive, khởi tạo và đột biến GA. Có `--seed` |

## Lỗi làm lệch kết quả

| Mã | File | Trước | Sau |
|---|---|---|---|
| L06 | `novelty.py` (`behavior_descriptor`) | Descriptor là trung bình vị trí cuối qua K bản đồ khác nhau | Nối K vị trí thành vector 2K chiều, chia cho kích thước arena |
| L07 | `runner.py`, `novelty.py` | Archive giữ vị trí đo trên bản đồ của thế hệ trước | Mặc định (`novelty_space="probe"`): hành vi đo trên 4 bản đồ probe cố định (bản đồ train đầu tiên của mỗi cấp), nên archive so được giữa các thế hệ; tốn thêm 4 lượt mỗi genome. Tùy chọn `"population"`: chỉ so trong quần thể hiện tại, không archive |
| L08 | `novelty.py` (`combine_scores`), `runner.py` | Chuẩn hóa min–max theo từng thế hệ | Thang cố định: fitness chia cho fitness lớn nhất có thể (28 với tiến độ geodesic); novelty chia cho √(2K), khoảng cách lớn nhất có thể giữa hai descriptor. Ngưỡng ban đầu của archive đặt theo phân vị 92,5% của thế hệ đầu (`calibrate`), nên nhận khoảng 5–10% mỗi thế hệ như Decision #18 |
| L09 | `novelty.py` (`update`, `reset`) | So với kỷ lục toàn run, không reset khi lên cấp; λ kẹt ở 0,1 tới 87–88% số thế hệ | So trung bình 5 thế hệ gần nhất với 5 thế hệ trước đó; `reset()` khi lên cấp; λ khởi đầu 0,5, bình thường 0,6, khi trì trệ 0,2 (sàn) |
| L10 | `curriculum.py`, `runner.py` | Chỉ học cấp hiện tại nên quên cấp cũ; xét lên cấp bằng điểm hybrid | Từ cấp 2: nửa số bản đồ mỗi thế hệ lấy từ các cấp thấp hơn. Xét lên cấp bằng 5 genome có fitness thô cao nhất trên val của cấp hiện tại |
| L12 | `src/baselines/rule_based.py` | Rẽ về phía có vật cản | Đổi dấu hai nhánh: bên trái thoáng thì rẽ trái (`turn > 0`) |
| L20 | `src/env/robot_env.py` | Bước vừa tới đích vừa chạm vật cản bị tính cả thành công lẫn va chạm (fitness +10 rồi −2) | `is_collision = collided and not reached_goal`; điều kiện kết thúc lượt giữ nguyên |
| L24 | `src/baselines/fixed_ga.py`, `random_search.py` | GA dùng tanh(z), NEAT dùng tanh(2,5·z) | `FixedMLP` dùng tanh(2,5·z) như neat-python. Khởi tạo N(0; 0,2) và σ 0,02 tương đương **chính xác** với N(0; 0,5) và σ 0,05 đã kiểm (θ với hệ số 2,5 cho cùng mạng như 2,5·θ với hệ số 1), nên kết quả GA trong báo cáo vẫn đúng |
| L26 | `src/env/geodesic.py` (mới), `robot_env.py`, `src/neat_engine/evaluator.py` | Tiến độ đo theo đường chim bay; vào ngõ cụt gần đích vẫn được điểm cao | Trường khoảng cách theo đường đi thật trên lưới 5 cm (Dijkstra, vật cản phình 0,30 m), lưu đệm ở `output/geo_cache/` (đổi bằng biến môi trường `NEAT_GEO_CACHE`). Env ghi `g0`, `g_min`; fitness dùng 10·(g0 − g_min)/g0, không giảm nửa khi va chạm. Đây đúng là fitness "geo" của báo cáo. `--progress euclid` giữ kiểu cũ để làm ablation |
| L27 | `src/maps/generator.py`, `scripts/pregenerate_maps.py`, `data/maps/{train,val,test}/level_3/` | Tường cấp 3 không bề dày, cách tường biên 0,5 m | Tường là hình chữ nhật dày 0,20 m, nối liền tường biên. 325 bản đồ cấp 3 được sinh lại và trùng khớp từng bản đồ với bộ "tường dày" đã dùng cho mọi kết quả từ mục 8.10 |
| L34 | `curriculum.py`, `runner.py` | Với ngưỡng 80%, run 1,5 triệu lượt chưa bao giờ học cấp 4 | Thêm `max_gens_per_level` (mặc định số thế hệ / 5): ở một cấp quá chừng đó thì lên cấp, nên mọi run đều học tới cấp 4. `summary.json` ghi cấp cao nhất đã học và từng lần lên cấp kèm lý do. Lịch học chung `--no-curriculum` (đều 4 cấp) dùng được cho cả NEAT và baseline |

## Sửa kèm theo

- **Cấu hình** (`configs/*.ini`):
  - L01: `no_fitness_termination = True`.
  - L29: đột biến cấu trúc 0,5/0,1/0,1/0,1.
  - L31: khai báo rõ `single_structural_mutation`.
  - L32: power trọng số 0,1, replace 0,01.
  - Có thêm nhóm khóa `response_*` mà neat-python 2.0.0 bắt buộc; thiếu nhóm này là lý do repo cũ không nạp được cấu hình.
  - Chú thích viết không dấu, vì neat-python mở file `.ini` bằng mã hóa mặc định của Windows và chữ có dấu gây `UnicodeDecodeError`.
- **L30** (`neat_fixes.py`): runner gọi `unify_initial_hidden_keys()`, không làm gì khi `num_hidden = 0`.
- **L18** (`requirements.txt`): ghim `neat-python==2.0.0`, dùng `pygame-ce`.
- **L21**: `train_neat.py`, `visualize.py`, `evaluate.py` suy ra mạng hồi quy hay truyền thẳng từ `feed_forward` của config.
- **L22** (`run_ablation.py`): lỗi được ghi kèm traceback vào `error.txt`; cuối chương trình in danh sách lượt hỏng, ghi `failures.json`, trả mã lỗi 1. Bỏ qua lượt đã có `summary.json`.
- **L23**: thiếu bản đồ thì báo lỗi ngay thay vì lặng lẽ cho fitness 0,01.
- **Hướng xuất phát**: env đọc `start_heading` của bản đồ (BARN đặt π/2, quay mặt về đích), như repo `Project_IT4906`.
- **In ra màn hình**:
  - runner in toàn ký tự ASCII;
  - mọi script gọi `src/console.py` để ghi UTF-8 khi output chuyển ra file hoặc pipe.
  - Trước đây λ, ★, ✓ làm chương trình dừng với `UnicodeEncodeError`.
- **Bộ sinh bản đồ**: `pregenerate_maps.py` luôn sinh đủ số bản đồ (thử seed kế tiếp khi seed hỏng); có `--levels` để chỉ sinh lại một cấp.

## Thay đổi hành vi cần biết khi so với số cũ

1. **Thang fitness đổi:** với tiến độ geodesic, fitness lớn nhất là 28 (trước là 19). `fitness_threshold` không còn tác dụng vì đã tắt dừng sớm.
2. **Số bản đồ mỗi thế hệ** mặc định 8 (trước 4), dùng chung cho cả quần thể.
3. **File kết quả chính** là `champion_genome.pkl` (chọn trên val). `library_best_genome.pkl` là genome thư viện trả về, chỉ để tham khảo.
4. **Bản đồ cấp 3 đã đổi** (tường dày). Mọi số cấp 3 đo trên bộ cũ không so trực tiếp được.
5. **Novelty tốn thêm 4 lượt mỗi genome** cho bản đồ probe. M4 nay thật sự là novelty thuần.
6. **Curriculum có giới hạn số thế hệ mỗi cấp.** Muốn giữ curriculum thuần theo SR thì đặt `--max-gens-per-level` rất lớn.

## Chưa làm

- **L11 (phân loài):** giữ ngưỡng 3,0. Thử điều chỉnh ngưỡng theo số loài đã làm kết quả kém đi (báo cáo mục 8.13), và neat-python 2.0.0 không gộp loài.
- **Các lỗi khác chưa sửa:**
  - L15: cho lượt chạy tiếp sau va chạm, chỉ cần cho thí nghiệm mê cung;
  - L16: tài liệu ghi 600 bước, code dùng 500;
  - L17: Holm và Fisher thay Bonferroni;
  - L22, phần checkpoint;
  - L25: slide;
  - L28: bản đồ thoáng cho 3D.
- **Xếp độ khó các cấp bằng planner tham chiếu** (phần thứ hai của cách sửa L27) chưa làm. Cấp 3 tường dày vẫn khó hơn cấp 4 với các bộ điều khiển phản xạ.
- **`data/maps/manifest.csv` và `reports/data/`** vẫn ghi số của bộ cấp 3 cũ. Cần tải `data/raw` rồi chạy lại `scripts/preprocess_data.py`.
- **Chưa chạy lượt dài và nhiều seed bằng code này.** Câu hỏi Q1 vẫn cần ít nhất 10 seed mỗi phương pháp.
