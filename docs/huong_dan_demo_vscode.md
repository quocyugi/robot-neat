# Hướng dẫn demo dự án bằng Visual Studio Code

Tài liệu cho buổi trình bày với giáo viên: cài đặt một lần, tạo môi trường, kiểm tra nhanh, kịch bản
demo 10–15 phút, các cấu hình **Run and Debug** có sẵn trong `.vscode/launch.json` và cách xử lý sự cố.

> **Lưu ý về số liệu và mã nguồn.** Benchmark đã chạy xong ngày 24/11/2026. Champion trong thư mục `demo/` là
> champion của benchmark (M6 seed 1, chọn theo SR val, xem `demo/README.md`); file
> `docs/benchmark/benchmark_results.json` và báo cáo `docs/benchmark_sau_sua_loi.md` lấy từ chính lượt benchmark
> đó (bước 1, 2, 3 và 5 của mục 10 đã làm ngày 24/11/2026). Các file demo (`.vscode/`, `demo/`, `scripts/demo.py`,
> `scripts/plot_benchmark.py`, `src/visual_report.py` và chính tài liệu này) **chưa được đưa lên GitHub**. Việc còn
> lại: bước 4 của mục 10.
>
> **Thay đổi ngày 24/11/2026 (theo yêu cầu của nhóm).** Tập test mê cung Lehman–Stanley (`test_lehman`, 2 bản đồ)
> đã bỏ khỏi code: tập test giờ là 4 cấp và BARN (250 bản đồ), cấu hình 7 chạy 4 bản đồ BARN. Các kết quả cần trình
> bày (cấu hình 2–9 và bảng benchmark) ghi thêm **ảnh bảng, ảnh biểu đồ và trang HTML** bên cạnh file JSON/MD cũ:
> mục *Kết quả dạng hình* ngay dưới bảng file.

## Các file dùng cho demo

| File | Vai trò |
|---|---|
| `.vscode/launch.json` | 9 cấu hình Run and Debug, đánh số theo thứ tự chạy: 1 và 9 chạy trước buổi demo để kiểm tra, 2–6 là phần chính của kịch bản (mục 4), 7–8 chạy thêm nếu còn thời gian |
| `.vscode/settings.json` | chọn interpreter `.venv`, đặt `PYTHONUTF8=1` cho terminal, mã hóa UTF-8 |
| `.vscode/extensions.json` | VS Code gợi ý cài extension Python và Python Debugger |
| `scripts/demo.py` | cửa sổ demo: robot chạy lần lượt trên nhiều bản đồ, có dải thông tin (HUD) tiếng Việt, xuất GIF; cuối lượt chạy ghi bảng kết quả dạng hình |
| `scripts/plot_benchmark.py` | vẽ biểu đồ, ảnh bảng và trang HTML so sánh các phương pháp (cấu hình 6) |
| `src/visual_report.py` | phần vẽ chung: ảnh bảng, ảnh biểu đồ, trang HTML cho cấu hình 2–9 và bảng benchmark |
| `demo/` | champion mặc định của demo: `champion_genome.pkl`, `neat_config.ini` |
| `docs/benchmark/benchmark_results.json` | số liệu benchmark đã gom, đầu vào của cấu hình 6 (**có sau benchmark**, mục 10) |
| `docs/benchmark/benchmark_SR.png`, `benchmark_SR_bang.png`, `benchmark_SR.html` | biểu đồ, ảnh bảng và trang HTML của bảng benchmark (`benchmark_SR.md` là cùng bảng ở dạng markdown) |
| `docs/benchmark_sau_sua_loi.md` | báo cáo benchmark, mở ở bước 6 của kịch bản (**có sau benchmark**, mục 10) |

Mọi cấu hình đều chạy ở thư mục gốc repo và đặt `PYTHONUTF8=1`; cấu hình nào ghi kết quả thì ghi vào
`output/demo/` (thư mục `output/` không đưa lên git).

### Kết quả dạng hình (ảnh PNG và trang HTML)

Bảng và biểu đồ ở terminal hay trong file `.md` khó đọc khi chiếu, nên mỗi kết quả cần trình bày ghi thêm ba file
cạnh file JSON/MD cũ (các file cũ vẫn ghi như trước):

- `<tên>_bang.png`: ảnh bảng số liệu, rộng 1600 px, chữ đủ to cho máy chiếu;
- `<tên>_bieu_do.png`: ảnh biểu đồ, rộng 1600 px;
- `<tên>.html`: một trang tự chứa (hai ảnh nhúng sẵn, không cần file nào khác) để mở bằng trình duyệt hoặc gửi cho
  người khác.

Cuối terminal in đường dẫn các file vừa ghi, dạng `Đã ghi ảnh: output/demo/test_results_bang.png` và
`Đã ghi trang HTML: output/demo/test_results.html`. **Ctrl+bấm** vào đường dẫn ảnh để mở ngay trong VS Code; trang
HTML mở bằng trình duyệt (kéo thả file vào Edge hoặc Chrome, hoặc chuột phải trên file trong Explorer của VS Code →
**Reveal in File Explorer** rồi bấm đúp). Màu kết quả giống cửa sổ demo: **TỚI ĐÍCH** xanh lá, **VA CHẠM** đỏ,
**HẾT GIỜ** cam; số dùng dấu phẩy thập phân; NEAT đứng trước, baseline ghi nhãn GA B1, GA B2.

| Cấu hình | File ảnh và trang HTML | Nội dung |
|---|---|---|
| 2, 3, 7, 8, 9 (`scripts/demo.py`) | `output/demo/ket_qua_cau_hinh_<k>_bang.png`, `_bieu_do.png`, `.html` (k là số cấu hình, đặt bằng `--report`) | bảng từng bản đồ (kết quả, số bước, thời gian, cách đích) và SR theo nhóm; biểu đồ số bước của từng bản đồ, màu theo kết quả, kèm ô SR từng cấp |
| 4 (`scripts/evaluate.py`) | `output/demo/test_results_bang.png`, `test_results_bieu_do.png`, `test_results.html` (cạnh `test_results.json`) | bảng SR, CR, hết giờ, SPL, MTT từng tập; biểu đồ tỉ lệ TỚI ĐÍCH, VA CHẠM, HẾT GIỜ từng tập |
| 5 (`scripts/train_neat.py`) | `output/demo/train_m6/training_history_bang.png`, `training_history_bieu_do.png`, `training_history.html` | log huấn luyện: tóm tắt champion và bảng từng thế hệ; biểu đồ fitness thô, tỉ lệ tới đích, SR val, số loài theo thế hệ (trang HTML có thêm bảng đủ mọi thế hệ, chép được chữ) |
| 6 (`scripts/plot_benchmark.py`) | `output/demo/benchmark_sr.png` (biểu đồ, như trước), `benchmark_sr_bang.png`, `benchmark_sr.html` | bảng phương pháp × tập test: mỗi ô là SR trung bình theo seed và min–max, nền xanh càng đậm SR càng cao, NEAT trước, baseline sau |
| bảng benchmark (mục 10) | `docs/benchmark/benchmark_SR.png`, `benchmark_SR_bang.png`, `benchmark_SR.html` (cạnh `benchmark_SR.md`) | như cấu hình 6 |

Không muốn ghi các file này thì thêm `--no-visual` vào lệnh (có ở cả bốn script).

---

## 1. Chuẩn bị (làm một lần)

### 1.1. Cài Visual Studio Code

1. Vào <https://code.visualstudio.com>, bấm **Download for Windows** (bản **User Installer**, không cần quyền quản trị).
2. Chạy bộ cài. Ở bước *Select Additional Tasks*, tick **Add to PATH** (để gõ `code .` trong terminal là mở
   thư mục hiện tại). Nên tick thêm *Add "Open with Code" action to Windows Explorer directory context menu*.

### 1.2. Cài Python 3.14

1. Vào <https://www.python.org/downloads/windows/>. Trong danh sách **Stable Releases**, tìm dòng
   **Python 3.14.x** đầu tiên (tháng 11/2026 là *Python 3.14.10*), rồi bấm **Windows installer (64-bit)**
   ngay dưới dòng đó.
2. **Không** dùng nút *Download Python install manager* ở trang <https://www.python.org/downloads/> (bộ cài
   kiểu mới, khác hẳn các bước dưới đây) và **không** cài Python 3.15 (kể cả dòng *standalone installer for
   Python 3.15.x* trên trang đó): dự án chưa kiểm thử trên 3.15.
3. Chạy file vừa tải (`python-3.14.x-amd64.exe`). Ở màn hình đầu của bộ cài, tick **Add python.exe to PATH**,
   rồi bấm **Install Now**.
4. Kiểm tra trong terminal: `py -3.14 --version` phải in `Python 3.14.x`.

Dự án đã kiểm thử với **Python 3.14.0**. Python 3.14 chưa có bản `pygame` gốc nên dự án dùng `pygame-ce`
(cùng tên import `pygame`, đã ghi trong `requirements.txt`). Đừng cài cả `pygame` lẫn `pygame-ce`.

### 1.3. Cài extension Python của Microsoft

1. Trong VS Code bấm **Ctrl+Shift+X**, tìm **Python** (nhà phát hành Microsoft, mã `ms-python.python`), bấm
   **Install**. Extension này tự cài kèm **Python Debugger** (`ms-python.debugpy`, cần cho Run and Debug), Pylance và
   **Python Environments** (`ms-python.vscode-python-envs`: hiện môi trường Python đang dùng trên thanh trạng thái,
   tự activate môi trường trong terminal).
2. Khi mở thư mục dự án, VS Code đọc `.vscode/extensions.json` và hiện thông báo gợi ý cài hai extension này
   nếu máy chưa có: bấm **Install**.

### 1.4. Lấy mã nguồn và mở đúng thư mục

1. Lấy **thư mục dự án do nhóm chuẩn bị** (file zip hoặc USB) và giải nén ra một chỗ cố định, ví dụ
   `D:\robot-neat`. Thư mục phải có đủ `.vscode\`, `demo\`, `scripts\demo.py` và `docs\huong_dan_demo_vscode.md`.

   Chỉ dùng `git clone https://github.com/quocyugi/robot-neat.git` (hoặc **Code → Download ZIP** trên GitHub)
   **sau khi** nhóm đã đưa các file demo lên GitHub: mở trang repo, thấy thư mục `.vscode` và file
   `scripts/demo.py` thì mới clone. Lúc viết tài liệu này GitHub chưa có các file đó, nên bản clone không có
   cấu hình Run and Debug, không có `scripts/demo.py` và champion trong `demo/`.
2. Trong VS Code: **File > Open Folder...**, chọn **thư mục gốc repo**, tức thư mục chứa `README.md`, `src/`,
   `scripts/` và `.vscode/`.
3. Lần đầu mở, VS Code hỏi *Do you trust the authors of the files in this folder?*: chọn
   **Yes, I trust the authors**. Ở chế độ hạn chế (Restricted Mode) không chạy được Run and Debug.

Mở sai thư mục (thư mục cha của repo, hoặc mở thẳng `scripts/`) thì VS Code không thấy `.vscode/` nên không có
các cấu hình demo, và lệnh gõ trong terminal ở thư mục đó báo không tìm thấy file (`can't open file ... [Errno 2]`
hoặc `FileNotFoundError`, xem mục 8).

---

## 2. Tạo môi trường Python cho dự án (`.venv`)

**Cách A, trong VS Code (giao diện có thể khác theo phiên bản extension):**

1. **Ctrl+Shift+P**, gõ **Python: Create Environment...**, Enter.
2. Chọn **Venv** (tạo thư mục `.venv` ngay trong dự án).
3. Chọn interpreter **Python 3.14**.
4. Ở bước chọn thư viện cần cài, tick **requirements.txt**, bấm **OK**. Chờ thông báo ở góc dưới phải báo xong
   (lần đầu mất vài phút, cần mạng). VS Code tự chọn `.venv` làm interpreter cho dự án.

Nếu thư mục đã có `.venv`, VS Code hỏi trước: chọn *Use Existing* nếu `.venv` được tạo trên chính máy này, *Delete and
Recreate* nếu thư mục chép từ máy khác. Nếu hiện *Select environment creation mode*: chọn **Custom**, rồi chọn venv,
Python 3.14 và cài `requirements.txt`.

**Cách B, trong terminal (khuyên dùng, đã kiểm thử)** (**Terminal > New Terminal**, hoặc **Ctrl+Shift+`**), đứng ở
thư mục gốc repo:

```bat
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Lệnh thứ hai gọi thẳng `python.exe` trong `.venv` nên không cần "activate" môi trường, và vì thế không bị chính
sách thực thi (execution policy) của PowerShell chặn. Hai lệnh chạy được cả trong PowerShell lẫn Command Prompt.

**Chọn interpreter:** **Ctrl+Shift+P → Python: Select Interpreter**, chọn dòng `.venv`
(`.\.venv\Scripts\python.exe`). `.vscode/settings.json` đã trỏ `python.defaultInterpreterPath` vào `.venv` nên
thường VS Code tự chọn đúng. Kiểm tra: mở một file `.py`, thanh trạng thái phía dưới phải hiện Python 3.14 kèm `.venv`.

**Đã có môi trường `venv/`** (tạo theo `README.md` bằng `python -m venv venv`, đã cài `requirements.txt` với
Python 3.14) thì không cần tạo `.venv`: chọn nó ở **Python: Select Interpreter** (dòng `.\venv\Scripts\python.exe`).
Các cấu hình Run and Debug dùng interpreter đang chọn; trong các lệnh ở mục 7 và 8 thì gõ `venv\Scripts\python`
thay cho `.venv\Scripts\python`.

Thư viện trong `requirements.txt`, bản đã kiểm thử: neat-python 2.0.0, numpy 2.4.6, gymnasium 1.3.0,
pygame-ce 2.5.8, scipy 1.17.1, matplotlib 3.11.2, seaborn 0.13.2, pandas 3.0.6, imageio 2.38.1 (dùng Pillow 13.0.0),
imageio-ffmpeg 0.8.1, tqdm 4.70.1. `requirements.txt` chỉ ghim neat-python, nên môi trường tạo sau này có thể
nhận bản mới hơn; bản mới gây lỗi thì cài lại đúng các bản trên bằng lệnh ở cuối mục 8.
Ngày 24/11/2026 cũng đã chạy đủ 9 cấu hình với các bản mới hơn mà pip cài lúc đó: numpy 2.5.5, gymnasium 1.4.0,
scipy 1.18.1, pandas 3.1.1, imageio 2.38.2 (các gói khác như trên).

---

## 3. Kiểm tra nhanh

1. Bấm **Ctrl+Shift+D** để mở **Run and Debug**.
2. Ở ô chọn cấu hình phía trên, chọn **1. Kiểm thử các bản sửa lỗi**.
3. Bấm **F5** (chạy có debug) hoặc **Ctrl+F5** (chạy không debug).
4. Terminal phía dưới in 13 dòng `OK    test_...` và dòng cuối **`13/13 passed`** (khoảng 2 giây với Ctrl+F5, 4–5
   giây với F5).

Thấy `13/13 passed` nghĩa là đã mở đúng thư mục, đúng interpreter, đủ thư viện tính toán. Muốn chắc chắn thư viện
`pygame` và `imageio` đã đủ: chạy cấu hình **9** (không mở cửa sổ, khoảng 10 giây), cuối terminal phải có dòng
`Đã ghi GIF output/demo/demo.gif: ...`, sau đó ba dòng `Đã ghi ảnh: ...`, `Đã ghi trang HTML: ...` (thư viện
`matplotlib` cũng đủ). Muốn thử cả cửa sổ demo: chạy cấu hình **7** (khoảng 30 giây), cửa sổ phải hiện robot và HUD.

---

## 4. Kịch bản demo 10–15 phút

Khi demo, chạy mọi cấu hình bằng **Ctrl+F5** (chạy không debug): nhanh hơn, và cửa sổ pygame không bị treo nếu
lỡ còn breakpoint. Cửa sổ demo có thể mở phía sau VS Code: bấm biểu tượng của nó trên taskbar, rồi bấm chuột vào
cửa sổ để nó nhận phím (Space, N, +, -, Esc). Từ bước 2 đến bước 6, bước k chạy cấu hình k.

| Bước | Cấu hình Run and Debug | Nói gì với giáo viên | Thời gian |
|---|---|---|---|
| 1 | (không chạy) mở `README.md`, bấm **Ctrl+Shift+V** để xem dạng trình bày | Bài toán: robot vi sai 2 bánh trong sân 10 m × 10 m, chỉ có 12 tia LiDAR tầm 3 m cùng khoảng cách và hướng tới đích (15 số đầu vào); mạng ra 2 lệnh tốc độ bánh trái, phải; mỗi lượt tối đa 500 bước (50 giây). NEAT tiến hóa cả cấu trúc lẫn trọng số mạng. 4 cấp bản đồ: cấp 1 trống, cấp 2 vật cản thưa, cấp 3 hành lang tường dày, cấp 4 bẫy chữ U. Thêm 50 bản đồ BARN (dữ liệu công khai) để thử khả năng tổng quát hóa. | 2 phút |
| 2 | **2. Demo: champion NEAT trên 4 cấp** | Chỉ trên cửa sổ: robot (tròn xanh dương, vạch trắng là hướng đầu), 12 tia LiDAR màu vàng (tia ngắn là vật cản gần): đó là tất cả những gì mạng "nhìn thấy"; quỹ đạo là nét đậm màu xanh tím; đích là vòng xanh lá. HUD ghi bộ điều khiển, cấp, mã bản đồ, bước/500, khoảng cách tới đích; cuối lượt hiện TỚI ĐÍCH, VA CHẠM hoặc HẾT GIỜ. Bấm Space để dừng giải thích, + để tua nhanh. Chạy xong 8 bản đồ, terminal in bảng kết quả và SR theo cấp, rồi ghi bảng đó dạng hình: Ctrl+bấm `output/demo/ket_qua_cau_hinh_2_bieu_do.png` ở cuối terminal để chiếu biểu đồ số bước và kết quả từng bản đồ. | 3 phút |
| 3 | **3. Demo: champion NEAT ở bẫy chữ U (cấp 4)** | Champion NEAT chạy trên hai bản đồ cấp 4 đầu tiên của tập test, nơi bẫy chữ U chắn giữa điểm xuất phát và đích. Robot chỉ thấy 3 m quanh mình nên dễ đi thẳng vào bẫy; đây là cấp khó nhất của dự án (SR test của R0 ở cấp 4 là 26,5%). | 1–2 phút |
| 4 | **4. Chấm champion trên toàn bộ tập test** | Không đánh giá bằng mắt mà chấm trên 250 bản đồ test không dùng khi huấn luyện (4 cấp × 50 và 50 bản đồ BARN): SR (tỉ lệ tới đích), CR (tỉ lệ va chạm), tỉ lệ hết giờ, SPL (vừa tới đích vừa đi ngắn: độ dài đường tối ưu chia độ dài đường đi thật, lượt không tới đích tính 0), MTT (thời gian trung bình của các lượt tới đích). Kết quả ghi vào `output/demo/test_results.json`; khi trình bày, Ctrl+bấm `output/demo/test_results_bieu_do.png` (tỉ lệ TỚI ĐÍCH, VA CHẠM, HẾT GIỜ từng tập) và `output/demo/test_results_bang.png` (bảng đủ chỉ số) ở cuối terminal. | 1 phút |
| 5 | **5. Huấn luyện thử M6 (5 thế hệ)** | Vừa chạy vừa giải thích log. Mỗi thế hệ in một khối khoảng 10 dòng; dòng cần đọc là `Gen k \| L1 \| best_raw=... mean_raw=... solved=... species=... lambda=...` của dự án, các dòng tiếng Anh còn lại do thư viện neat-python in (cách đọc: *Đọc log huấn luyện* ngay dưới bảng). Chạy xong, Ctrl+bấm `output/demo/train_m6/training_history_bieu_do.png`: cùng log đó dạng biểu đồ theo thế hệ (fitness thô, tỉ lệ tới đích, SR val, số loài), kèm ô tóm tắt champion; `training_history_bang.png` là bảng từng thế hệ. 5 thế hệ chỉ để xem cơ chế; các kết quả benchmark dùng 150 thế hệ (150 cá thể × 8 bản đồ × 150 thế hệ = 180.000 lượt đánh giá mỗi lượt chạy). | 3 phút |
| 6 | **6. Vẽ biểu đồ benchmark**, rồi mở `docs/benchmark_sau_sua_loi.md` (Ctrl+P, gõ tên file, Ctrl+Shift+V) | Biểu đồ và bảng so các phương pháp trên từng tập test, nhiều seed (cột là trung bình, thanh lỗi là min–max theo seed, mỗi chấm là một seed): NEAT (R0, M1–M6) với các baseline GA B1, GA B2. Nghĩa của từng tên: *Tên các phương pháp trong biểu đồ* dưới bảng. Mở ảnh: Ctrl+bấm vào đường dẫn ở các dòng `Đã ghi ảnh: ...` cuối terminal (`output/demo/benchmark_sr.png` là biểu đồ, `output/demo/benchmark_sr_bang.png` là bảng có tô màu theo SR), hoặc mở thư mục `output/demo/` trong Explorer (Ctrl+P không tìm file trong `output/` vì thư mục này nằm trong `.gitignore`). Bảng markdown ở terminal rất rộng nên bị xuống dòng: khi trình bày, dùng ảnh bảng. | 2–3 phút |
| 7 | (không chạy) mở code các phần đã sửa | **Ctrl+P**, gõ `geodesic.py`: trường khoảng cách theo đường đi thật (Dijkstra trên lưới 5 cm) thay cho đường chim bay, để fitness không còn thưởng việc lao vào ngõ cụt gần đích (lỗi L26). **Ctrl+P**, gõ `runner.py`, rồi **Ctrl+G**, gõ `268`, Enter: dòng `champ_check = self._champion_check(objs, raw, gen)`, champion chọn theo SR trên val (lỗi L02). Bấm chuột vào chữ `_champion_check` trên dòng đó rồi bấm **F12** để nhảy tới định nghĩa hàm (dòng 311), **Alt+←** để quay lại dòng 268. Danh sách đầy đủ các lỗi đã sửa: `docs/sua_loi_code.md`. | 2 phút |

Còn thời gian thì chạy thêm **7. Demo: champion trên 4 bản đồ BARN** (bản đồ công khai, khác hẳn bản đồ huấn
luyện) và **8. Xem champion vừa huấn luyện** (champion 5 thế hệ của bước 5, so với champion benchmark để thấy tác
dụng của huấn luyện dài). Hai cấu hình này cũng ghi ảnh kết quả `output/demo/ket_qua_cau_hinh_7_*` và
`output/demo/ket_qua_cau_hinh_8_*`.

**Đọc log huấn luyện (bước 5).** Mỗi thế hệ in một khối như sau (thế hệ đầu của một lần chạy thử):

```text
 ****** Running generation 0 ******

  Gen    1 | L1 | best_raw=26.28 mean_raw=5.84 solved=1.00 species=1  lambda=0.60 | 19.1s
Population's average fitness: 0.17417 stdev: 0.13940
Best fitness: 0.64243 - size: (2, 32) - species 1 - id 59
Average adjusted fitness: 0.136
Mean genetic distance 0.946, standard deviation 0.252
Population of 150 members in 1 species (after reproduction)
Total extinctions: 0
Generation time: 19.096 sec
```

- Dòng `Gen k | ...` là log của dự án: `L1` là cấp curriculum đang học; `best_raw`, `mean_raw` là fitness thô cao
  nhất và trung bình của 150 cá thể; `solved` là tỉ lệ bản đồ huấn luyện tới đích cao nhất trong quần thể;
  `species` là số loài (NEAT chia loài để bảo vệ cấu trúc mới); `lambda` là tỉ trọng fitness so với novelty
  (0,6 bình thường, hạ xuống 0,2 khi fitness chững lại); số giây cuối dòng là thời gian của thế hệ đó. Ở thế hệ 5
  có thêm `val_SR=...`: champion được chọn theo SR trên 100 bản đồ val của cả 4 cấp, không theo fitness huấn luyện.
- Các dòng tiếng Anh do thư viện neat-python in ra (`neat.StdOutReporter`), không phải code của nhóm:
  - `****** Running generation k ******`: neat-python đếm thế hệ từ 0, nên `Running generation 0` đi cùng
    `Gen 1`, `Running generation 4` đi cùng `Gen 5`.
  - `Best fitness`, `Population's average fitness`: điểm NEAT dùng để chọn lọc. Khi bật novelty (M6), đó là điểm
    kết hợp `lambda × fitness thô (chia cho mức tối đa) + (1 − lambda) × novelty (chuẩn hóa)`, nằm trong khoảng
    0–1, nên không so trực tiếp với `best_raw` (0,64 khác 26,28 là bình thường). `size: (2, 32)` là số nút (không
    kể 15 nút đầu vào; ở đây chỉ có 2 nút đầu ra, chưa có nút ẩn) và số kết nối đang bật.
  - `Average adjusted fitness`: fitness đã điều chỉnh theo loài, dùng để chia số con cho các loài.
    `Mean genetic distance`: khoảng cách di truyền trung bình, dùng để chia loài. `Population of 150 members in
    1 species`: số loài sau sinh sản. `Total extinctions`: số lần cả quần thể tuyệt chủng. `Generation time`:
    thời gian của thế hệ.
- Gần cuối log có dòng `Best individual in generation 4 meets fitness threshold - complexity: (2, 31)`, câu cố định
  của neat-python: config đặt `no_fitness_termination = True` (lỗi L01, xem `docs/sua_cau_hinh_NEAT_GA.md`) nên thư
  viện chạy đủ số thế hệ rồi luôn in câu này. Nó **không** có nghĩa đã giải xong bài toán: điểm kết hợp không quá 1
  nên không bao giờ đạt `fitness_threshold = 16`. Dòng cuối cùng
  `Training complete. Champion saved to output/demo/train_m6/champion_genome.pkl` cho biết file champion mà cấu hình
  8 sẽ chạy.

**Tên các phương pháp trong biểu đồ (bước 6).** Mọi phương pháp học dùng cùng ngân sách 180.000 lượt đánh giá
(150 cá thể × 8 bản đồ × 150 thế hệ) và champion của mọi phương pháp đều chọn theo SR trên val.

- **R0**: NEAT hồi quy, fitness thường (không novelty), học đều cả 4 cấp (không curriculum). Đây là cấu hình
  "NEAT đã sửa, hồi quy" trong `docs/sua_cau_hinh_NEAT_GA.md`, làm mốc để so với M1–M6.
- **M1–M6**: thí nghiệm bóc tách (`src/experiments/run_ablation.py`): bật, tắt từng thành phần (mạng hồi quy,
  novelty, curriculum) để xem thành phần nào có tác dụng. M1: truyền thẳng, không novelty, không curriculum. M2: hồi quy, curriculum. M3: truyền thẳng, novelty lai,
  curriculum. M4: hồi quy, novelty thuần (lambda = 0), curriculum. M5: hồi quy, novelty lai, không curriculum.
  **M6**: hồi quy, novelty lai, curriculum (mô hình đầy đủ; champion trong `demo/` là M6).
- **GA B1**, **GA B2** (baseline): thuật toán di truyền chỉ tiến hóa trọng số của mạng có cấu trúc cố định 15-8-2 (B1) và
  15-16-8-2 (B2). Cột GA B2 cùng màu với R0 nhưng có gạch chéo.

---

## 5. Phím tắt trong cửa sổ demo

| Phím | Tác dụng |
|---|---|
| **Space** | Tạm dừng / chạy tiếp. Ô trạng thái hiện **TẠM DỪNG** (vàng). Bấm lúc đang hiện kết quả thì giữ khung hình kết quả tới khi bấm lại. |
| **N** hoặc **→** | Sang bản đồ tiếp. Phần còn lại của lượt hiện tại vẫn được chạy (không hiển thị, dưới 0,1 giây) để bảng kết quả cuối cùng đúng; dòng đó có dấu `*`. |
| **+** (cũng nhận phím **=** và **+** bên bàn phím số) | Nhanh hơn, theo các mức 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240 khung hình/giây (mặc định 30). |
| **-** (cũng nhận **-** bên bàn phím số) | Chậm hơn, theo các mức trên. |
| **Esc** hoặc **Q** | Thoát; bấm nút đóng (X) của cửa sổ cũng vậy. Bảng kết quả các bản đồ đã chạy vẫn in ra terminal. |

Mỗi khung hình là một bước mô phỏng (0,1 giây), nên 30 khung/giây là nhanh gấp 3 lần thời gian thật. Tốc độ hiện ở
cuối dòng thứ hai của HUD (khi còn chỗ) và luôn có trên thanh tiêu đề cửa sổ, cùng số thứ tự `Bản đồ k/n`.

**Đọc màn hình demo:**

- HUD dòng 1: tên bộ điều khiển (ví dụ `NEAT hồi quy · champion trong demo/`). Ô bên phải hiện ĐANG CHẠY, TẠM DỪNG, rồi kết quả: **TỚI ĐÍCH** (xanh lá), **VA CHẠM** (đỏ),
  **HẾT GIỜ** (cam); viền sân đổi theo màu kết quả. Kết quả được giữ 1,5 giây (`--pause`) rồi sang bản đồ tiếp.
- HUD dòng 2: `Cấp k` (hoặc `BARN`) · mã bản đồ · `bước t/500` · `cách đích x m`.
- Sân: robot là hình tròn xanh dương (vạch trắng là hướng đầu robot), 12 tia LiDAR màu vàng, quỹ đạo là nét đậm
  màu xanh tím (vẽ dưới robot và các tia), đích là vòng tròn xanh lá (bán kính 0,3 m), tường và vật cản màu xám.

**Tùy chọn hay dùng của `scripts/demo.py`** (thêm vào `"args"` của một cấu hình trong `.vscode/launch.json`, hoặc
gõ trong terminal; xem đủ bằng `python scripts/demo.py -h`):

| Tùy chọn | Ý nghĩa |
|---|---|
| `--run-dir DIR` · `--genome FILE --config FILE` | chọn bộ điều khiển; không ghi gì thì dùng champion trong `demo/` |
| `--levels 1,2,3,4` · `--maps-per-level N` · `--start-index K` | N bản đồ mỗi cấp, bắt đầu từ bản đồ thứ K theo thứ tự tên file (mặc định 4 cấp, 1 bản đồ, K = 0) |
| `--split test` / `val` / `train` | lấy bản đồ ở tập nào (mặc định `test`) |
| `--barn N` | thêm N bản đồ BARN đầu tiên của `data/maps/test_barn` |
| `--maps FILE1,FILE2` | chạy đúng các file bản đồ này (thay cho chọn theo cấp) |
| `--fps 30` · `--pause 1.5` · `--frame-skip N` | tốc độ cửa sổ; số giây giữ khung kết quả; số bước mô phỏng mỗi khung hình |
| `--scale 1.3` | phóng to cửa sổ khi chiếu (cửa sổ tự thu lại nếu cao hơn màn hình) |
| `--label "M6 seed 0"` | tên bộ điều khiển hiện trên HUD |
| `--save-gif FILE` · `--no-window` | ghi GIF động có HUD; chạy không mở cửa sổ. Mặc định GIF lấy một khung mỗi 5 bước, nên mỗi bản đồ cho tối đa 101 khung; ước tính quá 1.000 khung (với mặc định là từ 10 bản đồ trở lên) thì script tự lấy thưa khung hơn và in ra số bước mỗi khung, để không tốn quá khoảng 0,9 GB RAM lúc ghi file. Có cửa sổ thì cửa sổ đóng trước khi ghi GIF |
| `--report TIỀN_TỐ` · `--no-visual` | tên các file kết quả dạng hình: `TIỀN_TỐ_bang.png`, `TIỀN_TỐ_bieu_do.png`, `TIỀN_TỐ.html` (mặc định `output/demo/ket_qua_demo`; các cấu hình trong `launch.json` đặt `output/demo/ket_qua_cau_hinh_<k>`); `--no-visual` để không ghi |

---

## 6. Bảng các cấu hình Run and Debug

Thời gian "đo" là đo trên máy kiểm thử. Cấu hình 4, 6 và 9 đo lại ngày 24/11/2026 với champion hiện tại trong
`demo/` (M6 seed 1), lúc máy không chạy việc nặng khác; các cấu hình còn lại đo trước đó, với champion tạm cũ và
trong lúc máy chạy benchmark nền (cấu hình 1, 3, 5, 8 không dùng champion trong `demo/`). Champion hiện tại sống
lâu hơn champion tạm trên bản đồ khó (nhiều lượt hết giờ đủ 500 bước thay vì va chạm sớm), nên cấu hình 2 và 7 lâu
hơn số đo cũ. Chỗ ghi "ước tính" là chưa đo trực tiếp: tính bằng số bước mô phỏng chia 30 khung hình/giây, cộng
1,5 giây giữ kết quả mỗi bản đồ (cách tính này khớp các số đo với champion tạm). Từ ngày 24/11/2026 cấu hình 2–9 ghi
thêm ảnh kết quả và trang HTML (mục *Kết quả dạng hình*), thêm khoảng 1–2 giây vào cuối mỗi lần chạy; cấu hình 4
và 6 dưới đây đã đo lại cùng phần đó.

| Cấu hình | Lệnh tương đương trong terminal (ở thư mục gốc repo) | Thời gian |
|---|---|---|
| 1. Kiểm thử các bản sửa lỗi | `python tests/test_fixes.py` | đo 1,6 giây |
| 2. Demo: champion NEAT trên 4 cấp | `python scripts/demo.py --levels 1,2,3,4 --maps-per-level 2 --report output/demo/ket_qua_cau_hinh_2` | ước tính 1,8 phút với champion hiện tại (2.822 bước; 5/8 bản đồ tới đích, 3 lượt hết giờ); tối đa 2,5 phút (8 lượt × 500 bước ở 30 khung/giây); champion tạm cũ đo 44 giây |
| 3. Demo: champion NEAT ở bẫy chữ U (cấp 4) | `python scripts/demo.py --levels 4 --maps-per-level 2 --report output/demo/ket_qua_cau_hinh_3` | dưới 40 giây (2 bản đồ, mỗi bản đồ tối đa 500 bước) |
| 4. Chấm champion trên toàn bộ tập test | `python scripts/evaluate.py --genome demo/champion_genome.pkl --config demo/neat_config.ini --workers 4 --out output/demo/test_results.json` | đo 5–7 giây (250 bản đồ, kể cả ghi ảnh và trang HTML) |
| 5. Huấn luyện thử M6 (5 thế hệ) | `python scripts/train_neat.py --generations 5 --workers 4 --val-every 5 --output-dir output/demo/train_m6` | đo 3 phút với 2 worker (khoảng 35 giây mỗi thế hệ); ước tính 1,5–2 phút với 4 worker |
| 6. Vẽ biểu đồ benchmark | `python scripts/plot_benchmark.py --from-json docs/benchmark/benchmark_results.json --out output/demo/benchmark_sr.png` | đo 3 giây với `docs/benchmark/benchmark_results.json` của benchmark (9 phương pháp, mục 10), gồm biểu đồ, ảnh bảng và trang HTML |
| 7. Demo: champion trên 4 bản đồ BARN | `python scripts/demo.py --maps-per-level 0 --barn 4 --report output/demo/ket_qua_cau_hinh_7` | ước tính 32 giây với champion hiện tại (766 bước; 3/4 bản đồ tới đích, 1 va chạm); tối đa 1,2 phút |
| 8. Xem champion vừa huấn luyện | `python scripts/demo.py --run-dir output/demo/train_m6 --levels 1,2 --maps-per-level 1 --report output/demo/ket_qua_cau_hinh_8` | đo 13 giây (cần chạy cấu hình 5 trước) |
| 9. Xuất GIF demo (không mở cửa sổ) | `python scripts/demo.py --no-window --save-gif output/demo/demo.gif --levels 1,2,3,4 --report output/demo/ket_qua_cau_hinh_9` | đo 9 giây, GIF 1,8 MB (301 khung hình; 3/4 bản đồ tới đích, cấp 3 hết giờ) |

**F5 hay Ctrl+F5?** F5 chạy dưới debugger (debugpy): dừng được ở breakpoint nhưng chậm hơn, và với cấu hình 4, 5
(nhiều tiến trình) debugger còn gắn vào từng tiến trình con. Khi demo dùng **Ctrl+F5**; chỉ dùng F5 khi muốn cho
giáo viên xem chạy từng bước.

---

## 7. Chạy không cần VS Code

Mở PowerShell hoặc Command Prompt ở thư mục gốc repo. Các lệnh gọi thẳng Python trong `.venv` nên không cần
activate (môi trường tên `venv` thì gõ `venv\Scripts\python`). Thứ tự các lệnh là thứ tự cấu hình 1–9:

```powershell
# PowerShell (Command Prompt thì thay dòng đầu bằng:  set PYTHONUTF8=1)
$env:PYTHONUTF8 = "1"
.venv\Scripts\python tests\test_fixes.py
.venv\Scripts\python scripts\demo.py --levels 1,2,3,4 --maps-per-level 2 --report output\demo\ket_qua_cau_hinh_2
.venv\Scripts\python scripts\demo.py --levels 4 --maps-per-level 2 --report output\demo\ket_qua_cau_hinh_3
.venv\Scripts\python scripts\evaluate.py --genome demo\champion_genome.pkl --config demo\neat_config.ini --workers 4 --out output\demo\test_results.json
.venv\Scripts\python scripts\train_neat.py --generations 5 --workers 4 --val-every 5 --output-dir output\demo\train_m6
.venv\Scripts\python scripts\plot_benchmark.py --from-json docs\benchmark\benchmark_results.json --out output\demo\benchmark_sr.png
.venv\Scripts\python scripts\demo.py --maps-per-level 0 --barn 4 --report output\demo\ket_qua_cau_hinh_7
.venv\Scripts\python scripts\demo.py --run-dir output\demo\train_m6 --levels 1,2 --maps-per-level 1 --report output\demo\ket_qua_cau_hinh_8
.venv\Scripts\python scripts\demo.py --no-window --save-gif output\demo\demo.gif --levels 1,2,3,4 --report output\demo\ket_qua_cau_hinh_9
```

Mỗi lệnh in đường dẫn ảnh và trang HTML vừa ghi ở cuối (mục *Kết quả dạng hình*); bỏ `--report ...` thì
`scripts/demo.py` ghi vào `output/demo/ket_qua_demo_*`.

Đã activate môi trường (hoặc Python cài sẵn đủ thư viện) thì gõ `python` thay cho `.venv\Scripts\python`.

---

## 8. Xử lý sự cố

| Hiện tượng | Nguyên nhân và cách sửa |
|---|---|
| `can't open file '...\\scripts\\demo.py': [Errno 2] No such file or directory`, `FileNotFoundError: [Errno 2] No such file or directory: 'demo/champion_genome.pkl'`, hoặc `ModuleNotFoundError: No module named 'src'` | Không đứng ở thư mục gốc repo. Hai lỗi đầu: mở sai thư mục trong VS Code hoặc `cd` sai trong terminal (Python tìm `scripts\...`, `demo/...` theo thư mục hiện tại). Lỗi thứ ba: bấm nút ▶ *Run Python File* trên một file trong `src/` (đó là thư viện, không chạy trực tiếp). Sửa: **File > Open Folder** đúng thư mục gốc (chứa `README.md`, `src/`, `scripts/`) và chạy bằng các cấu hình Run and Debug (đã đặt `cwd` là thư mục gốc); trong terminal thì `cd` về thư mục gốc rồi chạy `python scripts/...` hoặc `python -m src.experiments.run_ablation`. |
| `ModuleNotFoundError: No module named 'neat'` (hoặc `'pygame'`, `'imageio'`, `'matplotlib'`) | Chưa chọn interpreter `.venv` hoặc chưa cài `requirements.txt`. Sửa: **Ctrl+Shift+P → Python: Select Interpreter → .venv**, rồi chạy lại `.venv\Scripts\python -m pip install -r requirements.txt`. Gói tên `neat-python` nhưng import là `neat`; gói `pygame-ce` import là `pygame`. |
| Lỗi lạ trong numpy, pygame, matplotlib, pandas, ... sau khi tạo môi trường mới, dù máy khác chạy được | `requirements.txt` chỉ ghim `neat-python`, các gói khác lấy bản mới nhất lúc cài, có thể khác bản đã kiểm thử (mục 2). Cài lại đúng các bản đã kiểm thử bằng lệnh ngay dưới bảng này. |
| PowerShell báo *running scripts is disabled on this system* khi mở terminal (thường chỉ gặp khi tổ chức khóa execution policy bằng Group Policy) | Execution policy của PowerShell chặn script activate. Không cần activate: gọi thẳng `.venv\Scripts\python ...` như mục 7, hoặc đổi terminal mặc định sang Command Prompt: **Ctrl+Shift+P → Terminal: Select Default Profile → Command Prompt**, rồi mở terminal mới. Các cấu hình Run and Debug vẫn chạy bình thường vì gọi thẳng interpreter. |
| Cửa sổ pygame "Not Responding" | Chương trình đang dừng ở breakpoint (chạy bằng F5) nên cửa sổ không xử lý được sự kiện. Bấm Continue (F5) để chạy tiếp; khi demo hãy chạy bằng **Ctrl+F5** (Run Without Debugging). |
| `UnicodeDecodeError` khi nạp file `.ini` | Windows đọc file bằng bảng mã cp1252. Đã xử lý: chú thích trong các file `.ini` viết không dấu, và `launch.json`, `settings.json` đặt `PYTHONUTF8=1`. Chạy ngoài VS Code mà vẫn gặp thì đặt `PYTHONUTF8=1` như mục 7. |
| Chạy chậm, máy nóng | Giảm `--workers` (sửa `"4"` thành `"2"` trong `"args"` của cấu hình 4, 5 ở `.vscode/launch.json`), đóng ứng dụng nặng, cắm sạc. Cửa sổ demo: bấm **+** hoặc thêm `--frame-skip 2`. |
| Máy chiếu không hiện cửa sổ pygame | Bấm **Win+P → Duplicate** để chiếu giống màn hình laptop, hoặc kéo cửa sổ sang màn hình chiếu. Vẫn không được thì dùng GIF dự phòng của cấu hình 9: mở `output/demo/demo.gif` bằng trình duyệt (kéo thả file vào Edge hoặc Chrome). |
| Cửa sổ demo bị cắt mất phần dưới, hoặc quá nhỏ khi chiếu | Cửa sổ tự thu nhỏ nếu cao hơn màn hình; muốn to hơn thêm `--scale 1.3` vào `"args"`. |
| Bấm phím không có tác dụng | Bấm chuột vào cửa sổ demo trước. Nếu đang bật bộ gõ tiếng Việt (Unikey, EVKey), chuyển sang chế độ gõ tiếng Anh. |
| F5 báo lỗi debugger | Cập nhật extension **Python Debugger**, hoặc dùng **Ctrl+F5**. Ctrl+F5 vẫn chạy qua extension này; nếu cũng lỗi thì chạy lệnh tương ứng ở mục 7 trong terminal. |
| Chạy bằng F5, VS Code dừng lại và hiện khung *Exception has occurred: SystemExit* (ví dụ cấu hình 6 khi chưa có file JSON, cấu hình 8 khi chưa chạy cấu hình 5) | Không phải lỗi debugger: chương trình tự thoát, thông báo `Lỗi: ...` nằm ở terminal. Bấm **Continue** (F5) tới khi chạy xong (cấu hình 6 dừng 2 lần) hoặc **Stop** (Shift+F5); khi demo dùng **Ctrl+F5**. |
| Cấu hình 8 báo `không thấy thư mục output/demo/train_m6` (hoặc `không thấy file ...champion_genome.pkl`) | Chưa có kết quả huấn luyện: chạy cấu hình 5 trước. |
| Cấu hình 6 báo `không đọc được docs\benchmark\benchmark_results.json` | File này chỉ có sau khi lượt benchmark chạy xong (mục 10). |
| `Lỗi: chưa có champion mặc định của demo` | Thiếu `demo/champion_genome.pkl` hoặc `demo/neat_config.ini`: chép lại theo `demo/README.md`. |
| `Cảnh báo: không ghi được ảnh và trang HTML ...` ở cuối terminal | Phần vẽ ảnh gặp lỗi (thường do thiếu `matplotlib` hoặc thư mục `output/demo/` đang bị khóa); các kết quả chính (bảng ở terminal, file JSON, GIF, champion) vẫn ghi bình thường. Cài lại `requirements.txt`, rồi chạy lại cấu hình đó. |
| `demo.py: error: unrecognized arguments: --lehman` | Tập mê cung Lehman–Stanley đã bỏ khỏi code ngày 24/11/2026; bỏ `--lehman` khỏi lệnh (cấu hình 7 giờ dùng `--barn 4`). |

Lệnh cài đúng các bản thư viện đã kiểm thử (chạy ở thư mục gốc repo; môi trường tên `venv` thì đổi `.venv` thành `venv`):

```bat
.venv\Scripts\python -m pip install neat-python==2.0.0 numpy==2.4.6 gymnasium==1.3.0 pygame-ce==2.5.8 scipy==1.17.1 matplotlib==3.11.2 seaborn==0.13.2 pandas==3.0.6 imageio==2.38.1 imageio-ffmpeg==0.8.1 pillow==13.0.0 tqdm==4.70.1
```

---

## 9. Checklist trước buổi demo

- [ ] Champion trong `demo/` đúng là champion của benchmark (M6 seed 1, `demo/README.md`): chạy cấu hình 4, bảng SR
      phải trùng bảng ghi trong `demo/README.md`.
- [ ] Chạy thử **mọi cấu hình 1–9 một lần** trên đúng máy sẽ dùng (cấu hình 5 trước cấu hình 8). Lần chạy thử cũng
      tính sẵn trường geodesic vào `output/geo_cache/` (lúc demo cấu hình 5 chỉ nhanh hơn vài giây).
- [ ] Có `docs/benchmark/benchmark_results.json` và `docs/benchmark_sau_sua_loi.md`, cấu hình 6 vẽ được biểu đồ và
      ảnh bảng.
- [ ] Mở thử các ảnh kết quả (`output/demo/*_bieu_do.png`, `*_bang.png`) trong VS Code và một trang `.html` bằng
      trình duyệt trên đúng máy chiếu, chữ đọc được từ cuối phòng.
- [ ] Đã có GIF dự phòng `output/demo/demo.gif` của champion cuối (chạy lại cấu hình 9 sau khi thay champion) và
      đã mở thử bằng trình duyệt; nên chép thêm ra USB.
- [ ] Phóng to giao diện VS Code cho người ngồi xa đọc được: **Ctrl+=** (thu nhỏ **Ctrl+-**); kiểm tra cả chữ
      trong terminal.
- [ ] Đóng ứng dụng nặng (trình duyệt nhiều tab, game, phần mềm đồng bộ), cắm sạc, tắt thông báo
      (Do not disturb / Focus).
- [ ] Thử máy chiếu: **Win+P → Duplicate**, cửa sổ demo hiện đủ cả HUD và đáy sân.
- [ ] Mở sẵn các tab: `README.md`, `docs/benchmark_sau_sua_loi.md`, `src/env/geodesic.py`,
      `src/neat_engine/runner.py`, `docs/sua_loi_code.md`.

---

## 10. Khi benchmark chạy xong (người chuẩn bị demo làm)

Gọi thư mục kết quả benchmark là `<bench>` (nằm ngoài repo): mỗi lượt chạy ở `<bench>/<tên>/seed_<k>/`, với các
tên `ga_b1`, `ga_b2`, `R0`, `M1` … `M6`; mỗi thư mục seed có `test_results.json`.
Các lệnh dưới đây chạy ở thư mục gốc repo.

1. **Champion cho demo:** chọn seed của M6 có `champion_val_mean` cao nhất trong `<bench>/M6/seed_<k>/summary.json`
   (chọn theo val, không theo test), chép `champion_genome.pkl` và `neat_config.ini` của seed đó vào `demo/`:
   mục *Lấy champion nào* và *Cách thay champion* của `demo/README.md`.
2. **Số liệu cho cấu hình 6** (PowerShell; dòng đầu đặt đường dẫn thật của `<bench>`):

   ```powershell
   $B = "D:\bench_out"
   .venv\Scripts\python scripts\plot_benchmark.py --runs "R0=$B/R0" "M1=$B/M1" "M2=$B/M2" "M3=$B/M3" "M4=$B/M4" "M5=$B/M5" "M6=$B/M6" "GA B1=$B/ga_b1" "GA B2=$B/ga_b2" `
     --metric SR --sets test/level_1,test/level_2,test/level_3,test/level_4,test_barn,mean_levels_1_4 `
     --json docs/benchmark/benchmark_results.json --md docs/benchmark/benchmark_SR.md --out docs/benchmark/benchmark_SR.png
   ```

   Lệnh in bảng, ghi bảng markdown (`benchmark_SR.md`), ảnh biểu đồ (`benchmark_SR.png`), ảnh bảng
   (`benchmark_SR_bang.png`: mỗi ô là SR trung bình và min–max theo seed, nền xanh càng đậm SR càng cao, NEAT trước,
   baseline sau), trang HTML (`benchmark_SR.html`, hai ảnh nhúng sẵn) và file JSON mà cấu hình 6 đọc (`--from-json`
   chỉ nhận file do `--json` ghi; file JSON chỉ giữ các tập trong `--sets`). Tập `test_lehman` không còn trong
   `--sets` vì đã bỏ khỏi code ngày 24/11/2026 theo yêu cầu của nhóm; các file trong `docs/benchmark/` đã tạo lại
   bằng đúng lệnh trên cùng ngày (số của các tập còn lại không đổi). Ngày 25/11/2026 nhóm bỏ các baseline tìm kiếm ngẫu nhiên (B3), luật phản xạ (B4) và APF (B5) khỏi code; `docs/benchmark/` được vẽ lại từ file JSON đã bỏ ba phương pháp này, số của NEAT và GA không đổi. Tên trong `--runs` là tên trên biểu đồ, trong bảng và trong file JSON, khớp *Tên các phương pháp
   trong biểu đồ* ở mục 4. Thứ tự trong `--runs` là thứ tự cột và quyết định màu: NEAT trước (R0, M1–M6), rồi các
   baseline theo thứ tự GA B1, GA B2; giữ đúng thứ tự này để màu khớp
   với `docs/benchmark/benchmark_SR.png` đã tạo ngày 24/11/2026. Bảng và nhãn số trên cột dùng dấu phẩy thập phân;
   file JSON vẫn lưu số (tỉ lệ 0–1). Trong file JSON, đường dẫn tuyệt đối nằm ngoài thư mục hiện tại chỉ giữ 2 cấp cuối
   (`.../bench_out/M6`), nên file không mang tên người dùng của máy chạy benchmark. Phương pháp chưa có kết quả bị
   bỏ qua kèm cảnh báo: khi đó chờ benchmark xong rồi chạy lại.
3. **Báo cáo:** viết `docs/benchmark_sau_sua_loi.md` (dùng bảng trong `docs/benchmark/benchmark_SR.md`) và kiểm
   tra ngân sách ghi ở mục 4 (150 thế hệ, 180.000 lượt đánh giá mỗi lượt chạy) khớp với báo cáo.
4. **Đưa lên GitHub:** commit và push các file demo (`.vscode/`, `demo/`, `scripts/demo.py`,
   `scripts/plot_benchmark.py`, `src/visual_report.py`, `docs/huong_dan_demo_vscode.md`), `docs/benchmark/` và
   `docs/benchmark_sau_sua_loi.md`; thêm vào `README.md` một dòng dẫn tới tài liệu này. Từ lúc đó mới dùng được
   cách `git clone` ở mục 1.4; sửa lại lưu ý ở đầu tài liệu và mục 1.4 (bỏ câu "chưa được đưa lên GitHub").
5. Chạy lại cấu hình 4, 6 và 9 (chấm champion mới, vẽ biểu đồ, ghi lại GIF dự phòng) và sửa số đo thời gian ở
   mục 6 nếu khác nhiều.
