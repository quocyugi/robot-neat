# Benchmark sau khi sửa lỗi (đợt 5)

Benchmark chạy lại toàn bộ ma trận thí nghiệm trên code đã sửa (commit `56f4204`) từ ngày 23 đến 24/11/2026;
lượt cuối cùng xong lúc 12:12 ngày 24/11/2026. Mọi số dưới đây tính từ `test_results.json` và `summary.json` của
từng lượt chạy (`<bench>/<phương pháp>/seed_<k>/`) sau khi mọi lượt đã xong. Bảng và biểu đồ do script ghi:
`docs/benchmark/benchmark_SR.md`, `docs/benchmark/benchmark_SR.png` (biểu đồ), `docs/benchmark/benchmark_SR_bang.png`
(cùng bảng ở dạng hình, tô màu theo SR, dùng khi chiếu) và `docs/benchmark/benchmark_SR.html` (biểu đồ và ảnh bảng
trong một trang, mở bằng trình duyệt); số liệu đã gom: `docs/benchmark/benchmark_results.json` (cách tạo lại: mục 10
của `docs/huong_dan_demo_vscode.md`).

## 1. Giao thức

- **Code:** bản sao đóng băng của commit `56f4204`, tức đúng code trong repo. Huấn luyện bằng `NEATRunner` (NEAT)
  và `scripts/run_baselines.py` (các baseline), chấm test bằng `src/experiments/evaluation.py` (như
  `scripts/evaluate.py`).
- **Ngân sách mỗi lượt chạy:** quần thể 150, 150 thế hệ, 8 bản đồ huấn luyện mỗi thế hệ, tức
  150 × 8 × 150 = 180.000 lượt đánh giá, như nhau cho NEAT và GA. M3–M6 (có novelty) chạy
  thêm 4 bản đồ probe cố định cho mỗi cá thể mỗi thế hệ, chỉ để tính novelty: thêm 90.000 lượt, không tính vào
  180.000.
- **Chọn champion:** cứ 10 thế hệ, 5 cá thể có fitness huấn luyện cao nhất được chấm trên 100 bản đồ val (25 mỗi
  cấp); champion là cá thể có SR val trung bình 4 cấp cao nhất (thêm 7.500 lượt val mỗi lượt chạy). Không chọn
  theo tập test.
- **Curriculum** (M2, M3, M4, M6): ở cả 12 lượt, quần thể lên cấp 2 ở thế hệ 10 nhờ SR val ≥ 80%, nhưng lên cấp 3
  và 4 ở thế hệ 40 và 70 chỉ vì chạm giới hạn 30 thế hệ mỗi cấp; lịch thực tế là 10, 30, 30 và 80 thế hệ ở cấp 1–4.
- **Tập test:** 50 bản đồ mỗi cấp 1–4, 50 bản đồ BARN, 2 mê cung Lehman–Stanley (252 lượt mỗi champion), tối đa
  500 bước mỗi lượt. Chỉ số chính là SR (tỉ lệ tới đích); file kết quả có thêm CR, tỉ lệ hết giờ, SPL, MTT. Tập mê
  cung Lehman–Stanley (mọi lượt đều 0/2) đã bỏ khỏi code ngày 24/11/2026 theo yêu cầu của nhóm, nên các bảng dưới và
  `docs/benchmark/` không còn cột này; số của các tập khác không đổi.
- **Số seed:** NEAT R0 4 seed; M1–M6 mỗi cấu hình 3 seed; GA B1 3 seed; GA B2 1 seed.

**Các phương pháp.** NEAT: **R0** là NEAT hồi quy, chỉ dùng fitness, học đều 4 cấp, làm mốc so sánh. **M1–M6** là
thí nghiệm bóc tách: M1 như R0 nhưng mạng truyền thẳng; M2 hồi quy, curriculum; M3 truyền thẳng, novelty lai, curriculum; M4 hồi
quy, novelty thuần (lambda = 0), curriculum; M5 hồi quy, novelty lai, không curriculum; M6 hồi quy, novelty lai,
curriculum (mô hình đầy đủ). Baseline: **GA B1**, **GA B2** là GA chỉ tiến hóa trọng số của mạng cố định 15-8-2 và
15-16-8-2 (ngày 25/11/2026 nhóm bỏ các baseline tìm kiếm ngẫu nhiên, luật phản xạ và APF khỏi code, nên tài liệu chỉ còn so NEAT với GA). Bảng dưới, `docs/benchmark/benchmark_SR.md` và biểu đồ dùng đúng các tên
này, cùng thứ tự: NEAT trước, rồi các baseline.

## 2. SR trên tập test

SR (%): trung bình các seed (seed thấp nhất–cao nhất). Cấp 1–4 và BARN mỗi tập 50 bản đồ. Cùng số với
`docs/benchmark/benchmark_SR.md` (file đó do script ghi; ảnh `docs/benchmark/benchmark_SR_bang.png` là cùng bảng ở
dạng hình). Bảy dòng đầu là NEAT (R0, M1–M6), năm dòng sau là baseline.

| Phương pháp | Seed | Cấp 1 | Cấp 2 | Cấp 3 | Cấp 4 | BARN | TB cấp 1–4 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| **R0** | 4 | 96,5 (94–98) | 71,5 (54–82) | 26,0 (18–46) | 26,5 (4–50) | 24,0 (16–30) | 55,1 (48–67,5) |
| M1 | 3 | 94,7 (92–98) | 64,7 (54–82) | 30,0 (28–34) | 30,0 (2–50) | 30,0 (16–58) | 54,8 (45,5–65) |
| M2 | 3 | 86,7 (64–100) | 52,0 (44–58) | 22,0 (16–26) | 28,7 (16–40) | 16,0 (14–20) | 47,3 (43–52,5) |
| M3 | 3 | 95,3 (94–98) | 48,7 (42–52) | 28,0 (18–34) | 30,7 (18–42) | 16,0 (8–20) | 50,7 (46,5–55) |
| M4 | 3 | 86,7 (82–92) | 40,0 (34–48) | 8,7 (2–14) | 1,3 (0–4) | 6,7 (6–8) | 34,2 (30,5–36) |
| M5 | 3 | 94,7 (90–98) | 66,0 (44–78) | 30,0 (22–34) | 30,0 (2–66) | 28,0 (16–36) | 55,2 (47,5–68,5) |
| M6 | 3 | 96,0 (88–100) | 60,0 (54–66) | 17,3 (6–28) | 17,3 (0–44) | 20,7 (10–34) | 47,7 (40–56,5) |
| GA B1 | 3 | 99,3 (98–100) | 60,7 (48–68) | 29,3 (18–38) | 46,7 (32–54) | 14,7 (10–24) | 59,0 (55–64,5) |
| GA B2 | 1 | 100 | 70 | 26 | 46 | 14 | 60,5 |

**NEAT.** R0 đạt trung bình 55,1% trên 4 cấp, 26,0% ở cấp 3, 26,5% ở cấp 4 và 24,0% trên BARN, với dao động giữa
các seed rất lớn (cấp 4 từ 4% đến 50%). Trong các biến thể bóc tách, M1 (truyền thẳng, 54,8%) và M5 (novelty lai,
không curriculum, 55,2%) ngang R0; ba cấu hình có curriculum M2 (47,3%), M3 (50,7%) và M6 (47,7%) thấp hơn R0
4–8 điểm; M4 (novelty thuần, cũng có curriculum) thấp nhất ở cấp 2, 3, 4, BARN và trung bình 4 cấp (34,2%; cấp 4 chỉ
1,3%, BARN 6,7%; ở cấp 1 ngang M2, cùng 86,7%), và ở trung
bình 4 cấp cả 3 seed M4 đều thấp hơn mọi seed của mọi phương pháp học khác có từ 3 seed. Cấp 4 là
chỗ NEAT kém ổn định nhất: 8/22 lượt NEAT chỉ đạt 0–8% ở cấp 4 (R0 seed 3, M1 seed 2, cả 3 seed M4, M5 seed 1, M6
seed 0 và 2), trong khi seed thấp nhất của GA B1 đạt 32%. Champion trong `demo/` là M6 seed 1, seed có
`champion_val_mean` cao nhất trong 3 seed M6 (0,50; hai seed kia 0,46 và 0,49); trên tập test nó đạt 88%, 66%, 28%
và 44% ở cấp 1–4 (trung bình 56,5%) và 34% trên BARN.

**Baseline.** GA B1 (3 seed) có SR cao nhất ở cấp 4 (46,7%) và ở trung bình 4 cấp (59,0%) trong các phương pháp có từ
3 seed, nhưng thấp hơn R0 trên BARN (14,7% so với 24,0%); GA B2 (1 seed) cho số gần GA B1 (60,5%).

## 3. Kiểm định và giới hạn

Kiểm định hoán vị chính xác hai phía trên giá trị SR từng seed: xét mọi cách chia lại nhãn các seed của hai phương
pháp, p = min(1, 2 × p một phía nhỏ hơn). Đơn vị lặp là seed huấn luyện. Vì số seed ít, p có mức sàn: 3 so với 3
seed thì p không nhỏ hơn 0,1; 4 so với 3 thì không nhỏ hơn 2/35 ≈ 0,057; 4 so với 4 thì không nhỏ hơn 2/70 ≈ 0,029.
GA B2 chỉ có 1 seed nên không kiểm định được với phương pháp khác, chỉ so bằng mô tả.

Chênh lệch SR (điểm %, A trừ B) và p hoán vị:

| Câu hỏi (A so với B) | Seed A / B | Cấp 3 | Cấp 4 | BARN | TB cấp 1–4 | p nhỏ nhất có thể |
|:--|:-:|--:|--:|--:|--:|--:|
| NEAT so với GA: R0 so với GA B1 | 4 / 3 | −3,3; p = 0,74 | −20,2; p = 0,17 | +9,3; p = 0,17 | −3,9; p = 0,51 | 0,057 |
| Hồi quy: R0 so với M1 | 4 / 3 | −4,0; p = 0,74 | −3,5; p = 0,91 | −6,0; p = 0,86 | +0,3; p = 0,91 | 0,057 |
| Curriculum: M2 so với R0 | 3 / 4 | −4,0; p = 0,97 | +2,2; p = 0,97 | −8,0; p = 0,11 | −7,8; p = 0,23 | 0,057 |
| Novelty lai: M5 so với R0 | 3 / 4 | +4,0; p = 0,69 | +3,5; p = 0,91 | +4,0; p = 0,57 | +0,0; p = 0,91 | 0,057 |
| Mô hình đầy đủ: M6 so với R0 | 3 / 4 | −8,7; p = 0,51 | −9,2; p = 0,57 | −3,3; p = 0,69 | −7,5; p = 0,34 | 0,057 |
| Novelty thuần so với lai: M4 so với M6 | 3 / 3 | −8,7; p = 0,40 | −16,0; p = 0,40 | −14,0; p = 0,10 | −13,5; p = 0,10 | 0,1 |
| Truyền thẳng so với hồi quy: M3 so với M6 | 3 / 3 | +10,7; p = 0,30 | +13,3; p = 0,50 | −4,7; p = 0,80 | +3,0; p = 0,70 | 0,1 |

**Kết luận.**

- Với số seed hiện có, không phép so nào giữa các phương pháp của benchmark này có thể đạt p < 0,05, nên không
  chênh lệch nào có ý nghĩa thống kê. Trên 168 phép so (28 cặp của 8 phương pháp có từ 3 seed, mỗi cặp 6 tập: cấp
  1–4, BARN, trung bình 4 cấp), p nhỏ nhất là 0,057: R0 so với M4 (cấp 1, 2, 3, BARN và trung bình 4 cấp) và R0 so
  với M3 ở cấp 2. Đó là các trường hợp mọi seed bên này cao hơn mọi seed bên kia, mức mạnh nhất mà số seed cho
  phép (tính cả các cặp 3 so với 3, có p = 0,1, thì có 33/168 phép so như vậy, 28 trong số đó có M4), nhưng vẫn
  trên 0,05 và chưa hiệu chỉnh cho nhiều phép so.
- Chưa kết luận được NEAT hơn hay kém GA mạng cố định (R0 thấp hơn GA B1 20,2 điểm ở cấp 4 nhưng cao hơn 9,3 điểm
  trên BARN, cả hai p = 0,17), và chưa kết luận được mạng hồi quy, curriculum hay novelty có giúp NEAT không. Chỉ để
  tham khảo về hướng: các cấu hình có curriculum thấp hơn R0, novelty thuần (M4) thấp hơn hẳn, novelty lai không
  curriculum (M5) ngang R0.
- Muốn tách được các chênh lệch cỡ vài điểm cần nhiều seed hơn hẳn; mục 8.16 ước tính chênh lệch phát hiện được với 10 seed mỗi
  bên.

## 4. Đọc thêm

- Báo cáo Word `Bao_cao_pipeline_NEAT_robot.docx` của nhóm, bản ngày 24/11/2026 (file Word không đưa lên repo):
  - **8.16** Benchmark code đã sửa (đợt 5): giao thức, bảng kết quả, kiểm định, so theo từng bản đồ. Số liệu ở đó
    là số cuối, tính sau khi mọi lượt đã xong: bảng SR, bảy phép so và các số đếm gộp mọi lượt (ví dụ 8/22 lượt NEAT
    chỉ đạt 0–8% ở cấp 4) trùng với mục 2 và 3 ở trên. Bản tạm trước đó của báo cáo chốt số lúc 02:15 ngày
    24/11/2026, khi M3, M4 mới có 1 seed và GA B2 chưa xong.
  - **8.17** Kiểm tra lại lượt R0 và GA B1: chấm lại, diễn biến huấn luyện, phân loài, cảm biến, BARN.
  - **8.18** BARN thấp: nguyên nhân và thử cảm biến khác (biến thể R0_sec12, R0_sec24).
  - **8.19** Hai cải tiến cho NEAT: giữ cảm biến (R0_keep) và bật phân loài (R0_spec).
- Bốn biến thể NEAT của đợt 5, R0_sec12 (4 seed), R0_sec24 (4 seed), R0_keep (3 seed) và R0_spec (3 seed), chạy
  cùng ngân sách với R0 trên một bản sao code có thêm công tắc và đã xong ngày 24/11/2026. Mục 8.18 và 8.19 mô tả
  thiết kế và kết quả của chúng; các biến thể này không đưa vào bảng trên.
