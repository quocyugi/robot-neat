# Mê cung Lehman–Stanley: cài lại thí nghiệm novelty search

Thư mục này cài lại thí nghiệm mê cung của Lehman & Stanley (2011), *Abandoning Objectives: Evolution Through the Search for Novelty Alone*, để kiểm tra bộ NEAT của dự án (neat-python 2.0.0) có tái hiện được kết quả của bài báo không: novelty search giải được mê cung khó, còn tìm theo fitness (khoảng cách tới đích) thì bị đánh lừa.

## Nội dung

| File | Nội dung |
|:--|:--|
| `lehman_maze_ns.py` | Bộ mô phỏng mê cung viết lại theo `maze.h` của mã C++ gốc ([jal278/novelty-search-cpp](https://github.com/jal278/novelty-search-cpp)) và vòng tiến hóa bằng neat-python 2.0.0 |
| `ket_qua_cai_lai.json` | Kết quả mọi lượt đã chạy (`summary`: tổng hợp theo cấu hình; `runs`: từng lượt) |

Bộ mô phỏng giữ đúng mã gốc: robot bán kính 8 đơn vị; 5 cảm biến khoảng cách (−90°, −45°, 0°, 45°, 90°, tầm 100 đơn vị) và 4 cảm biến radar hình quạt chỉ hướng đích, cộng bias, tổng 10 đầu vào; 400 bước mỗi lượt; tới đích khi cách đích dưới 5 đơn vị; va chạm chỉ chặn bước đi. Tiến hóa theo thế hệ của neat-python (bài báo dùng rtNEAT).

Novelty search: novelty = trung bình khoảng cách tới 15 láng giềng gần nhất trong quần thể và archive; hành vi = vị trí cuối (x, y). Archive theo luật của mã gốc: nhận khi novelty > ngưỡng (ngưỡng đầu 6, sàn 0,25); 10 thế hệ không nhận ai thì ngưỡng × 0,95; nhận quá 4 cá thể trong một thế hệ thì ngưỡng × 1,2. Chế độ fitness dùng 300 − khoảng cách cuối tới đích.

## Chạy

Cần hai file mê cung `hard_maze.txt` và `medium_maze.txt` của [jal278/novelty-search-cpp](https://github.com/jal278/novelty-search-cpp), đặt chung một thư mục (truyền bằng `--maze-dir`). Hai file này không đi kèm repo.

```
mkdir ket_qua
python -I lehman_maze/lehman_maze_ns.py run --maze-dir <thư mục mê cung> --maze hard --method ns --config ref --seed 0 --out ket_qua/hard_ns_ref_s0.json
python -I lehman_maze/lehman_maze_ns.py run --maze-dir <thư mục mê cung> --maze hard --method ns --config file --team-cfg configs/neat_recurrent.ini --seed 0 --out ket_qua/hard_ns_final_s0.json
python -I lehman_maze/lehman_maze_ns.py check
```

- `--method ns | fit`: novelty search thuần, hoặc tìm theo fitness.
- `--config ref`: cấu hình NEAT tham chiếu nhúng trong script (quần thể 250, sigmoid, hồi quy, 1 nút ẩn).
- `--config file --team-cfg <file .ini>`: chạy nguyên một file cấu hình của nhóm, chỉ đổi số đầu vào, đầu ra, quần thể và hàm kích hoạt sigmoid. `--config team` là cách cũ, đặt thêm compatibility 2,0, survival 0,5, replace 0,02.
- Mặc định quần thể 250 × 300 thế hệ, tối đa 75.000 lượt đánh giá (bài báo: 250.000); lượt chạy dừng ngay khi có cá thể tới đích.
- `check`: so bộ kích mạng theo lô với `neat.nn.RecurrentNetwork`.
- `--out`: file JSON kết quả; thư mục chứa file phải có sẵn (script không tự tạo).

## Kết quả đã có (20 seed mỗi dòng, seed 0–19)

| Mê cung | Cách tìm | Cấu hình NEAT | Giải được | Trung vị số lượt đánh giá tới khi giải |
|:--|:--|:--|--:|--:|
| Khó | novelty search | tham chiếu (`ref`) | 20/20 | 8.183 |
| Khó | fitness | tham chiếu (`ref`) | 0/20 | – |
| Trung bình | novelty search | tham chiếu (`ref`) | 20/20 | 6.433 |
| Trung bình | fitness | tham chiếu (`ref`) | 10/20 | 37.338 |
| Khó | novelty search | cấu hình gốc của nhóm (trước khi sửa) | 9/20 | 31.911 |
| Khó | novelty search | cấu hình nhóm, chế độ `team` | 12/20 | 30.477 |
| Khó | novelty search | cấu hình sửa, bản trung gian | 17/20 | 29.127 |
| Khó | novelty search | cấu hình sửa, bản cuối (= `configs/neat_recurrent.ini`) | 18/20 | 23.310 |

Bài báo báo: novelty search giải mê cung trung bình 40/40 và mê cung khó 39/40 lượt; NEAT theo fitness giải mê cung khó 3/40 lượt.

## Liên hệ với benchmark của dự án

- `data/maps/test_lehman` chứa hai mê cung bản [yaricom/goNEAT_NS](https://github.com/yaricom/goNEAT_NS) (tải bằng `scripts/download_public_data.py`), chuyển sang bộ mô phỏng của dự án theo tỉ lệ 0,03125 m mỗi đơn vị. Bản này lệch vài điểm hình học so với bản jal278 ở trên (mê cung khó: 11 so với 13 đoạn tường), và bộ mô phỏng của dự án khác bản gốc (12 tia, 500 bước, va chạm kết thúc lượt, đích 0,30 m). Vì vậy số ở bảng trên không so trực tiếp với cột Lehman của benchmark.
- Trong pipeline chính, novelty search kiểu Lehman là cấu hình `LNS` (`src/experiments/run_ablation.py`, `NEATRunner(novelty_rule="lehman")`, `scripts/train_neat.py --novelty-rule lehman --no-curriculum`): cùng luật archive, ngưỡng đổi sang thang mô tả hành vi của dự án (vị trí cuối trên 4 bản đồ probe), dùng được với mọi bộ dữ liệu của benchmark. `LNS_ref` là cùng thuật toán với cấu hình tham chiếu ở trên, chỉ đổi cho robot của dự án: 15 đầu vào, tanh, quần thể 150 (`configs/neat_lehman_ref.ini`). Kết quả benchmark của LNS và LNS_ref chưa có trong `docs/benchmark/`.
