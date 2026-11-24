# Thư mục `demo/`

Thư mục này chứa **champion mặc định cho buổi demo**. `scripts/demo.py` dùng nó khi không
chỉ định bộ điều khiển, và các cấu hình 2, 4, 7 và 9 trong `.vscode/launch.json` cũng dùng nó.

| File | Nội dung |
|---|---|
| `champion_genome.pkl` | genome NEAT (pickle của `neat.DefaultGenome`, neat-python 2.0.0) |
| `neat_config.ini` | cấu hình NEAT đi cùng genome; có `feed_forward = False` nên demo tự dùng mạng hồi quy |

## Champion hiện tại: M6 seed 1 của benchmark

Champion đang để ở đây là champion của lượt benchmark **M6 seed 1** (mô hình đầy đủ: mạng hồi quy, novelty lai,
curriculum; 150 thế hệ, 180.000 lượt đánh giá; code của commit 56f4204). Hai file được chép nguyên từ thư mục
của lượt đó ngày 24/11/2026, theo mục *Lấy champion nào* và *Cách thay champion* dưới đây. Champion tạm trước đó
(M6 chỉ huấn luyện 3 thế hệ) không còn dùng.

| Mục | Giá trị |
|---|---|
| Lượt chạy | `<bench>/M6/seed_1/`, huấn luyện xong ngày 23/11/2026 |
| `champion_val_mean` (SR trung bình trên 100 bản đồ val) | 0,50, cao nhất trong 3 seed M6 (seed 0: 0,46; seed 2: 0,49) |
| SR val cấp 1 / 2 / 3 / 4 | 0,72 / 0,60 / 0,28 / 0,40 |
| Thế hệ của champion | 120 (trong 150 thế hệ) |
| Kích thước mạng | 15 nút (2 nút ra, 13 nút ẩn; không kể 15 đầu vào), 19 kết nối đang bật (66 kết nối nếu kể cả kết nối đã tắt) |
| `neat_config.ini` | file của chính lượt đó; nội dung trùng `configs/neat_recurrent.ini`, chỉ khác kiểu xuống dòng (CRLF) |
| Ngày đưa vào `demo/` | 24/11/2026 |

Kết quả trên tập test (`scripts/evaluate.py`, 250 bản đồ), trùng từng số với `test_results.json` của lượt benchmark:

| Tập | SR | CR | Hết giờ |
|---|---:|---:|---:|
| Cấp 1 | 0,88 | 0,00 | 0,12 |
| Cấp 2 | 0,66 | 0,08 | 0,26 |
| Cấp 3 | 0,28 | 0,10 | 0,62 |
| Cấp 4 | 0,44 | 0,10 | 0,46 |
| BARN (50 bản đồ) | 0,34 | 0,52 | 0,14 |

Lượt benchmark còn chấm 2 mê cung Lehman–Stanley (0/2, cả hai va chạm); tập này đã bỏ khỏi code ngày 24/11/2026
theo yêu cầu của nhóm, nên bảng trên và cấu hình 4 không còn dòng đó.

Kiểm tra nhanh trước buổi demo: chạy cấu hình 4 trong Run and Debug (hoặc
`python scripts/evaluate.py --genome demo/champion_genome.pkl --config demo/neat_config.ini`), phải ra đúng bảng
trên; cấu hình 4 ghi thêm `output/demo/test_results_bieu_do.png` và `output/demo/test_results_bang.png` (bảng này
ở dạng hình, dùng khi chiếu) cùng trang `output/demo/test_results.html`. `python scripts/demo.py --no-window` (bản đồ test đầu tiên của mỗi cấp) phải cho cấp 1, 2 và 4 TỚI ĐÍCH sau
270, 334 và 380 bước, cấp 3 HẾT GIỜ.

## Lấy champion nào

Champion cuối cùng lấy từ lượt benchmark. Benchmark ghi mỗi lượt chạy vào
`<bench>/<cấu hình>/seed_<k>/` (`<bench>` là thư mục kết quả benchmark, nằm ngoài repo).
Lấy cấu hình **M6** (mô hình đầy đủ: hồi quy, novelty, curriculum). Mở `summary.json` của từng
seed (`<bench>/M6/seed_0/summary.json`, `<bench>/M6/seed_1/summary.json`, ...) và chọn seed có
`champion_val_mean` (SR trung bình trên 100 bản đồ val) cao nhất. Chọn theo val, không chọn
theo kết quả test, để số liệu test của champion vẫn khách quan.

## Cách thay champion

Chép **cả hai file** `champion_genome.pkl` và `neat_config.ini` từ thư mục kết quả đã chọn vào
đây, ghi đè hai file cũ. Thư mục đó có thể là `<bench>/M6/seed_<k>/` của benchmark, thư mục
`--output-dir` của `scripts/train_neat.py`, hoặc `output/ablation/<cấu hình>/seed_<k>/` của
`python -m src.experiments.run_ablation`. Luôn lấy hai file **từ cùng một thư mục**: genome phải
đi với đúng config của nó (số đầu vào, số đầu ra, kiểu mạng hồi quy hay truyền thẳng). Không cần
chép các file khác (`summary.json` ghi đường dẫn tuyệt đối của máy chạy benchmark).

Ví dụ seed 1 có `champion_val_mean` cao nhất (đúng trường hợp của lần thay ngày 24/11/2026), chạy ở thư mục gốc
repo (thay `D:\bench_out` bằng đường dẫn thật của thư mục benchmark và `seed_1` bằng seed đã chọn):

```powershell
# PowerShell
Copy-Item D:\bench_out\M6\seed_1\champion_genome.pkl, D:\bench_out\M6\seed_1\neat_config.ini -Destination demo\ -Force
```

```bat
:: Command Prompt
copy /Y D:\bench_out\M6\seed_1\champion_genome.pkl demo\
copy /Y D:\bench_out\M6\seed_1\neat_config.ini demo\
```

Cũng có thể chép bằng chuột trong thanh Explorer của VS Code (Ctrl+C, Ctrl+V vào thư mục `demo/`).

## Kiểm tra sau khi thay

1. Không mở cửa sổ, in bảng kết quả 4 bản đồ:

   ```bash
   python scripts/demo.py --no-window
   ```

2. Chạy cấu hình 4 (chấm trên toàn bộ tập test): bảng SR phải trùng `test_results.json` trong thư mục của lượt
   đã chọn (file đó của benchmark còn mục `test_lehman`, bỏ qua mục này) và khác bảng ở mục *Champion hiện tại*;
   sau đó sửa mục đó cho đúng champion mới.
3. Chạy lại cấu hình 9 để ghi lại GIF dự phòng `output/demo/demo.gif`: GIF cũ vẫn là champion trước đó.

Không muốn chép file thì chạy thẳng từ thư mục kết quả:

```bash
python scripts/demo.py --run-dir D:/bench_out/M6/seed_1
```

Thư mục `demo/` được đưa lên git (khác `output/`), nên sau khi thay champion hãy commit hai
file này nếu muốn repo mang theo champion cuối cùng.
