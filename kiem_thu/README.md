# Kiểm thử pipeline NEAT điều hướng robot

Script và số liệu gốc của phần kiểm thử trong báo cáo Word. Các script không sửa repo của nhóm.
Cách chạy xem Phụ lục A của báo cáo.

| File | Nội dung |
|---|---|
| `t01_config_versions.py` | Nạp cấu hình NEAT trên từng phiên bản neat-python |
| `t10_env_checks.py` | Kiểm tra môi trường: Gymnasium, động học, LiDAR, va chạm, fitness |
| `t11_data_checks.py` | Kiểm tra lại bộ bản đồ, trùng lặp, robot bám đường A* |
| `t12_misc.py` | Bộ sinh bản đồ, thống kê, novelty, λ, GA, visualize |
| `t13_speciation_probe.py` | Số loài theo ngưỡng compatibility |
| `t20_speed.py` | Tốc độ mô phỏng |
| `t30_baselines.py` | B4, B5 trên tập test |
| `run_asis.py` | Chạy nguyên trạng `NEATRunner` của nhóm cho một cấu hình và seed |
| `run_fixed.py` | Bản tham chiếu đã sửa lỗi: chạy song song, chọn champion trên val, novelty đúng |
| `run_maze.py` | Thí nghiệm trên mê cung Lehman & Stanley |

Thư mục `results/` chứa số liệu của lần chạy được dùng trong báo cáo.

## Điều tra tỉ lệ tới đích thấp (`tests2/`)

| File | Nội dung |
|---|---|
| `tests2/fastsim.py` | Bản vector hóa của `RobotNavEnv`, cho kết quả trùng khớp bản gốc |
| `tests2/d00_equivalence.py` | Kiểm tra bản nhanh và bản gốc cho cùng kết quả |
| `tests2/d01_diagnose.py` | Kiểm tra độ khó bản đồ, thuật toán tham chiếu, dạng thất bại |
| `tests2/d02_viz_l3.py` | Vẽ và thống kê chỗ va chạm ở cấp 3 |
| `tests2/d03_geo_cache.py` | Tính trước trường khoảng cách geodesic cho mọi bản đồ |
| `tests2/train_fast.py` | Huấn luyện nhanh GA, CMA-ES, NEAT với fitness và cách chọn bản đồ tùy chọn |
| `tests2/run_queue.py`, `tests2/jobs*.json` | Hàng đợi các lượt huấn luyện đã chạy |

## Kiểm thử 3D trên bản đồ DUnE (`sim3d/`)

| File | Nội dung |
|---|---|
| `sim3d/world3d.py` | Đọc mesh và SDF của DUnE thành lưới tam giác (không cần ROS) |
| `sim3d/sim3d.py` | Rover động học bám địa hình, LiDAR dò tia bằng Open3D, tiêu chí kết thúc như `rover_env.py` |
| `sim3d/d10_audit3d.py` | Kiểm tra từng thế giới và tạo tập lượt chạy cố định |
| `sim3d/d11_transfer3d.py` | Chạy bộ điều khiển 2D và thuật toán viết tay trên các thế giới |
| `sim3d/d12_mock_bridge.py` | Kiểm tra cầu nối DUnE bằng môi trường giả |
| `sim3d/d13_finetune3d.py` | Thêm pitch, roll và tinh chỉnh 30 thế hệ trên 3D |

Số liệu nằm ở `results/dieu_tra_SR/` và `results/sim3d/`. Kết quả 3D là mô phỏng gần đúng, không phải Gazebo.

## Cài lại baseline đã công bố (`baselines/`)

| File | Nội dung |
|---|---|
| `baselines/b01_xor.py` | XOR trên neat-python, kiểm tra bộ NEAT |
| `baselines/b02_lahouazi.py` | Viết lại môi trường của `GayaLahouazi/rl-robot-navigation`, chạy lại trọng số công bố và huấn luyện lại |
| `baselines/b03_lehman.py` | Viết lại mê cung Lehman–Stanley theo mã C++ gốc, chạy NEAT và Novelty Search |
| `baselines/neat_ref.ini` | Cấu hình NEAT tham chiếu dùng cho mê cung |

Cần tải riêng về thư mục khác: repo `GayaLahouazi/rl-robot-navigation` (MIT) và các file `hard_maze.txt`, `medium_maze.txt` từ `jal278/novelty-search-cpp`. Số liệu nằm ở `results/baselines/`.

## Sửa cấu hình NEAT, GA và bản đồ kiểu Butterworth (đợt 3)

| File | Nội dung |
|---|---|
| `tests2/d30_indoor_maps.py` | Bộ sinh bản đồ trong nhà theo mã TU Delft (McGuire 2019, Butterworth 2019), thu nhỏ về 10 × 10 m, lưu như cấp 5 |
| `tests2/d31_eval_champ.py` | Chấm champion đã lưu trên bất kỳ thư mục bản đồ nào |
| `tests2/d32_render_maps.py` | Vẽ bản đồ ra PNG |
| `tests2/d33_make_fixed_configs.py`, `tests2/fixed_config_changes.json` | Sinh cấu hình NEAT đã sửa từ file gốc của nhóm, kèm diff |
| `tests2/d34_fix_summary.py` | Gom kết quả thành `results/dieu_tra_SR/fix_summary.json` |
| `tests2/d35_readme.py` | Sinh README của `sua_cau_hinh_NEAT_GA/` từ kết quả |
| `tests2/d36_md_update.py` | Chèn khối "Cập nhật sau thí nghiệm" vào `reports/Cài đặt NEAT trong các benchmark.md` |
| `tests2/jobs8.json` … `jobs13c.json` | Các lượt sàng lọc và kiểm chứng; `jobs13*.json` là các lượt chạy lại sau lần dừng máy ngày 10/11 (lượt 1,5 triệu chạy với 8 worker, kết quả không phụ thuộc số worker) |
| `tests2/show_fix.py` | In nhanh các số chính của `fix_summary.json` và điểm chuyển giao |

`train_fast.py` có thêm `--levels` (huấn luyện trên cấp bất kỳ, ví dụ `--levels 5`), `--fixes` và `--ga-module` (nạp `neat_fixes.py`, `fixed_ga.py` từ thư mục giao cho nhóm). Bản đồ cấp 5 sinh lại được bằng
`python d30_indoor_maps.py <repo> <thư mục ra> <thư mục bản đồ tường dày> 250 25 100` (khoảng 3 phút), rồi `python d03_geo_cache.py <repo> <thư mục ra> <thư mục geo> 2`.
Mê cung: `baselines/b03_lehman.py run ... --config file --team-cfg <file .ini>` chạy một file cấu hình nguyên trạng, chỉ đổi số đầu vào, đầu ra.
