# Cấu hình NEAT và GA đã sửa (nhóm 3, IT4906)

Tài liệu này mô tả bản sửa cấu hình NEAT và GA. Bản sửa được kiểm trên repo `hoangthcslt/Project_IT4906`, khác repo này ở vài chỗ nhỏ đã được đưa vào cùng đợt sửa (khóa `response_*` trong config, hướng xuất phát `start_heading`, in ra ký tự ASCII). Phân tích đầy đủ ở mục 8.13–8.14 của `reports/Bao_cao_pipeline_NEAT_robot.docx`; script và kết quả kiểm thử ở `kiem_thu/`; các lỗi code đã sửa ở [sua_loi_code.md](sua_loi_code.md).

## Trạng thái trong repo này

Mọi bước áp dụng đã làm trong cùng commit:

- `configs/neat_recurrent.ini`, `configs/neat_feedforward.ini` là bản đã sửa. Mỗi dòng đổi có chú thích `SUA` (viết không dấu, xem mục Giới hạn).
- `src/neat_engine/neat_fixes.py` được `NEATRunner` gọi (`unify_initial_hidden_keys`). Champion được chọn trên val ngay trong runner (L02). Mặc định 8 bản đồ mỗi thế hệ.
- `src/baselines/fixed_ga.py` dùng các mặc định mới:
  - cắt ngọn 20%, đột biến mọi gen, lai ghép đều, 1 elite;
  - tanh hệ số 2,5 như NEAT (lỗi L24), nên khởi tạo N(0; 0,2) và σ 0,02;
  - hai giá trị này tương đương chính xác với N(0; 0,5) và σ 0,05 trên tanh thường, là cấu hình đã kiểm.
- Ghim `neat-python==2.0.0` trong `requirements.txt`.

## Đã sửa gì

| Tham số | Gốc | Sửa | Vì sao |
|---|---|---|---|
| `no_fitness_termination` | False | True | Lỗi L01: dừng ở thế hệ 0–1 khi một genome đạt 16 điểm trên phòng trống |
| `conn_add` / `conn_delete` / `node_add` / `node_delete` | 0,05 / 0,01 / 0,03 / 0,01 | 0,5 / 0,1 / 0,1 / 0,1 | Lỗi L29: với giá trị cũ XOR chỉ giải 87/100 lần |
| `single_structural_mutation` | không khai báo | True | neat-python 2.0.0 đọc khóa vắng mặt thành chuỗi `'false'`, chuỗi khác rỗng là True (L31); ghi rõ để hành vi không đổi |
| `max_stagnation` | 15 | 20 | Fitness đo trên bản đồ mới mỗi thế hệ nên ồn |
| `weight/bias_mutate_power`; `weight/bias_replace_rate` | 0,5; 0,1 | 0,1; 0,01 | L32: bước 0,5 quá thô để tinh chỉnh |
| GA: chọn lọc | tournament k = 3 | cắt ngọn 20% | Cường độ chọn lọc 0,85 so với 1,40 |
| GA: đột biến | 10% gen, σ 0,5 | mọi gen, σ 0,05 (= σ 0,02 với tanh hệ số 2,5) | σ 0,5 bằng nửa thang trọng số |
| GA: khởi tạo; elite | N(0, 1); 2 | N(0; 0,5); 1 (= N(0; 0,2) với tanh hệ số 2,5) | N(0, 1) làm tanh bão hòa |

## Kết quả kiểm thử

Kiểm thử chạy **chính các file trong thư mục này**: bộ huấn luyện nhanh của báo cáo đọc các file `.ini`, nạp `neat_fixes.py` và `fixed_ga.py` từ đây. Điều kiện chung, trừ chỗ ghi khác trong tên dòng: bản đồ cấp 3 tường dày 0,20 m, fitness geodesic, học đều cả bốn cấp với 8 bản đồ mỗi thế hệ (2 mỗi cấp), 180.000 lượt đánh giá, champion chọn trên val, 50 bản đồ test mỗi cấp. Nhiều seed: trung bình (thấp nhất–cao nhất).

| Cấu hình | Seed | Cấp 1 | Cấp 2 | Cấp 3 | Cấp 4 | BARN |
|---|---|---|---|---|---|---|
| NEAT, cấu hình gốc | 3 | 97% (90%–100%) | 45% (36%–52%) | 13% (8%–16%) | 1% (0%–2%) | 9% (8%–10%) |
| NEAT đã sửa, hồi quy | 3 | 97% (96%–98%) | 68% (50%–78%) | 33% (26%–46%) | 37% (26%–50%) | 21% (16%–26%) |
| NEAT đã sửa, truyền thẳng | 3 | 97% (94%–100%) | 61% (48%–78%) | 25% (12%–36%) | 29% (2%–50%) | 23% (14%–36%) |
| NEAT đã sửa, hồi quy, 1,5 triệu lượt | 1 | 92% | 72% | 28% | 50% | 26% |
| NEAT tốt nhất trước đây, 1,5 triệu lượt, curriculum | 1 | 96% | 62% | 16% | 38% | 16% |
| GA, cấu hình gốc | 3 | 95% (90%–98%) | 68% (66%–70%) | 16% (4%–28%) | 14% (0%–26%) | 17% (10%–20%) |
| GA đã sửa | 3 | 91% (86%–100%) | 67% (64%–68%) | 34% (26%–42%) | 29% (24%–34%) | 9% (4%–14%) |
| CMA-ES (tham chiếu) | 3 | 98% (96%–100%) | 65% (62%–68%) | 35% (34%–38%) | 48% (30%–72%) | 23% (20%–26%) |

- XOR, 100 seed: cấu hình gốc 87/100, bản sửa 96/100 (trung bình 118 thế hệ; cấu hình mẫu của thư viện 66 thế hệ).
- Mê cung hard của Lehman–Stanley, novelty search, quần thể 250: cấu hình gốc giữ nguyên 9/20 lần giải, bản sửa 18/20; cấu hình tham chiếu Packt 20/20.
- Bản đồ trong nhà kiểu McGuire/Butterworth (cấp 5, 100 bản đồ test, học và test trên cùng họ bản đồ, 180.000 lượt nếu không ghi khác, 1 seed): NEAT gốc 47%; NEAT sửa (hồi quy) 63%; NEAT sửa (truyền thẳng) 61%; NEAT sửa (hồi quy, 750.000 lượt) 73%; GA sửa 69%; CMA-ES 68%. Bug đạt 80% trong 500 bước và 90% trong 1.000 bước.

## Giới hạn

- Mỗi cấu hình chỉ có 1–3 seed; khác biệt dưới khoảng 15 điểm phần trăm chưa chắc là thật.
- Kiểm thử dùng bộ mô phỏng nhanh của báo cáo (khớp `RobotNavEnv` trên mọi lượt đã so), chưa chạy toàn bộ bằng `runner.py` của nhóm.
- `TargetSpeciesReporter` có trong `neat_fixes.py` nhưng không bật mặc định: neat-python giữ đại diện cho mọi loài cũ nên số loài chỉ giảm khi loài bị loại vì trì trệ; xem mục 8.13.
- Chú thích trong file `.ini` viết không dấu vì neat-python 2.0.0 mở file bằng mã hóa mặc định của Windows (cp1252 hoặc cp1258), và chữ có dấu gây `UnicodeDecodeError` khi không bật chế độ UTF-8. Bản giao ngày 11/11/2026 còn chú thích có dấu nên không nạp được khi chạy `python` bình thường; bản ngày 12/11/2026 đã sửa, các giá trị cấu hình không đổi. Nếu tự thêm chú thích tiếng Việt, hãy viết không dấu hoặc chạy với `python -X utf8`.
