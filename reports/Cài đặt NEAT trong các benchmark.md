# NEAT thua CMA-ES, nhưng phép so còn lệch

<!-- cap-nhat-sau-thi-nghiem:start -->
> **Cập nhật sau thí nghiệm (2026-11-11).** Các ý dưới đây đã được kiểm bằng thí nghiệm trên máy của nhóm (mục 8.13–8.14 của `Bao_cao_pipeline_NEAT_robot.docx`, file giao `sua_cau_hinh_NEAT_GA/`). Chúng sửa một số dự đoán của báo cáo này, nhất là về phân loài và survival. Điều kiện chung: bản đồ tường dày, fitness geodesic, học đều cả bốn cấp, 8 bản đồ mỗi thế hệ, champion chọn trên val, 50 bản đồ test mỗi cấp. Số viết theo dạng cấp 3 / cấp 4.
>
> - **Đòn bẩy lớn nhất là bước đột biến trọng số, không phải phân loài.** Giảm `weight_mutate_power` 0,5 → 0,1 và `weight_replace_rate` 0,1 → 0,01 (bias cũng vậy), cùng bỏ dừng sớm, tăng đột biến cấu trúc và stagnation 20, đưa NEAT từ 13% / 1% (3 seed, cấu hình gốc) lên 33% / 37% (3 seed) ở 180.000 lượt. Chỉ sửa dừng sớm, đột biến cấu trúc và stagnation thì được 6% / 2% (1 seed).
> - **Sau khi sửa, NEAT cùng dải với GA và CMA-ES ở cấp 3:** GA đã sửa (cắt ngọn 20%, σ 0,05 trên mọi gen, khởi tạo N(0; 0,5), 1 elite) 34% / 29% (3 seed), GA gốc 16% / 14% (3 seed), CMA-ES 35% / 48% (3 seed). Cấp 4 CMA-ES vẫn nhỉnh hơn; với 3 seed, chênh dưới khoảng 10–15 điểm chưa tách được khỏi nhiễu.
> - **Hai đề xuất lấy từ tài liệu không giúp trên bài này:** giữ 8 loài bằng ngưỡng động 4% / 14% (1 seed); survival 0,5 như Butterworth 8% / 1% (2 seed), đều kém bản đã sửa (33% / 37%). neat-python 2.0.0 không bao giờ gộp loài, nên nâng ngưỡng không giảm được số loài.
> - **Ở cùng ngân sách 1,5 triệu lượt**, NEAT đã sửa đạt 28% / 50% (1 seed), so với 16% / 38% của run cũ (cấu hình Packt, học curriculum, chưa từng gặp cấp 4) và 46% / 68% của CMA-ES 1 triệu lượt. Thêm ngân sách nâng cấp 4 nhưng không nâng cấp 3 (trung bình 33% ở 180.000 lượt); ở cấp 3, 66% số lượt hết giờ, tức robot quẩn trong ngõ cụt chứ không đâm tường.
> - **Trên họ bản đồ McGuire/Butterworth dựng lại cho robot của nhóm** (cấp 5, 100 bản đồ test, 180.000 lượt nếu không ghi khác, 1 seed mỗi cấu hình): NEAT gốc 47%; NEAT đã sửa 63%; NEAT đã sửa truyền thẳng 61%; NEAT đã sửa 750.000 lượt 73%; GA đã sửa 69%; CMA-ES 68%. Bug đạt 80% trong 500 bước, gần mức khoảng 90% của Bug có nhớ trong McGuire. Bài Butterworth báo 90,4% với genome tốt nhất của 20 run chọn ngay trên test, lượt 300 s và robot nhỏ hơn nhiều so với khe cửa.
> - **Bài chuẩn:** cấu hình NEAT giao cho nhóm giải XOR 96/100 lần (gốc 87/100) và mê cung hard của Lehman–Stanley bằng novelty 18/20 lần (gốc 9/20).
>
> **Kết luận sau thí nghiệm:** 16% / 38% không phải giới hạn của NEAT trên bài này. Mức đó chủ yếu do cấu hình (bước đột biến trọng số quá thô) và giao thức (curriculum chưa từng gặp cấp 4). Sau khi sửa, NEAT không còn dấu hiệu cài đặt sai; câu hỏi NEAT có hơn mạng cố định hay không cần ≥ 10 seed để trả lời.
<!-- cap-nhat-sau-thi-nghiem:end -->

Chưa có bài báo nào đo NEAT trên bản đồ held-out dựng quanh ngõ cụt dài hay bẫy chữ U với giới hạn 50 s. Vì vậy 16% (cấp 3) và 38% (cấp 4) không thể gọi là "tương đương công bố". So với các mốc gần đúng loại, **16% chỉ ngang bộ điều khiển phản xạ trong điều kiện chặt** (Bug của nhóm 28%, FTG 26% trên dead-end của Xiong 2025) và thua rõ CMA-ES 46% trên chính các map đó. Ngân sách không phải lý do. Butterworth et al. 2019, bài gần nhất với cùng 15 đầu vào, 2 bánh và pop 150, cũng dùng đúng 1,5 triệu episode. Họ nới survival lên 0,55, giữ add-node 0,005, cho cả quần thể chạy chung 10 map mới mỗi thế hệ, và đạt 90,4–97,1% trên loại map dễ hơn nhiều. Khác biệt nằm ở cấu hình và giao thức. Run NEAT tốt nhất dùng cấu hình chuyển từ sách Packt với ngưỡng compatibility 3,0, và với ngưỡng đó distance chuẩn hóa của neat-python 2.0.0 chỉ cho **1–3 loài (trung bình 1,3)**, trong khi mọi nhánh code gần tác giả gốc giữ khoảng 10 loài. Run đó còn học theo curriculum và dừng ở cấp 3, nên **chưa từng gặp cấp 4**, trong khi CMA-ES học đều cả bốn cấp; mỗi bên chỉ có 1 seed. Giao thức nên theo là: map mới mỗi thế hệ dùng chung cho cả quần thể; champion chọn trên validation cố định khoảng 200 map tách khỏi test, như ProGAb của Pinville 2011 (170/180 trên validation, 150/180 trên test); test 200 map; ít nhất 10 seed. Butterworth chọn ứng viên ngay trên test, nên số của họ có thể lạc quan tới khoảng 5 điểm. GA của nhóm lệch thực hành công bố ở bốn chỗ: lai ghép đều trên mọi con; tournament k = 3 có cường độ chọn lọc chỉ khoảng 0,85 so với 1,40 của truncation 20%; khởi tạo N(0,1) gấp 3–4 lần thang Xavier với bước đột biến bằng nửa thang trọng số; và elite giữ điểm may mắn. Một bản sửa theo hướng đó đưa cấp 3 từ 16% lên 34–38% trong các run một seed. NEAT-NC (arXiv 2604.15076) chỉ là neat-python 1.1.0 với 21 đầu vào "navigation cell", chạy 50 cá thể × 10 thế hệ trên ba map cố định, không tách train/test, nên 93–100% của nó không so được với số của nhóm. Vì vậy vẫn cần cải thiện, nhưng đích là một NEAT đúng tài liệu trong giao thức khớp với CMA-ES. Nếu sau đó NEAT vẫn thua, kết quả ấy khớp Igel 2003 và Pagliuca 2019 và vẫn có giá trị báo cáo.

## Các bài NEAT thành công giữ khoảng 10 loài và nới chọn lọc, NEAT của nhóm chạy với 1–3 loài

Quy ước dùng trong báo cáo:

- **Đo cục bộ**: kết quả chạy trên máy của nhóm, 50 map test mỗi cấp, 1 seed trừ khi ghi khác, trên bộ map tường dày 0,20 m trừ khi ghi "tường mỏng". File kết quả nằm ở `kiem_thu/results/dieu_tra_SR/` (bản sao kết quả của máy chạy thử), không có URL.
- **Tự tính**: phép tính của người viết.
- **NR**: bài không báo cáo.
- Số thập phân viết bằng dấu phẩy; tên khóa và giá trị trong ô mã giữ dấu chấm như trong file `.ini`.

Các bài NEAT điều hướng chia thành hai trường phái cấu hình. Trường phái thứ nhất đi theo NEAT gốc và code của Lehman–Stanley: thêm node 0,003–0,03, thêm link 0,05–0,1, phân loài có kiểm soát và chạy rất nhiều thế hệ. Butterworth, Silva, Lehman & Stanley, Gomes và MCC thuộc nhóm này, và **hai bài có kết quả held-out tốt nhất, Butterworth và Silva, đều dùng tỉ lệ cấu trúc thấp** ([Butterworth §4.2.1](https://ar5iv.labs.arxiv.org/html/1904.06239); [Silva §4.1](https://repositorio.iscte-iul.pt/bitstream/10071/10504/5/p1059_single.pdf)). Trường phái thứ hai chép file ví dụ của neat-python: thêm connection 0,5, thêm node 0,2–0,4, quần thể 30–100 và ít thế hệ. Shrestha & Bhattarai 2025, NEAT-NC và Ishtiaq 2022 thuộc nhóm này, và không bài nào vừa có map held-out vừa có thống kê theo seed ([Shrestha & Bhattarai §3.2](https://arxiv.org/pdf/2504.17794); [NEAT-NC Table 2](https://arxiv.org/html/2604.15076); [Ishtiaq Table II](https://arxiv.org/abs/2209.09007)). Tỉ lệ cao của trường phái thứ hai vì thế là dấu vết của file ví dụ, không phải bằng chứng về cấu hình tốt cho điều hướng. Trong phạm vi đã tìm, không có bài NEAT mapless dùng LiDAR nào vừa có tập map held-out, vừa chạy nhiều seed, vừa báo trung bình theo seed.

### Butterworth 2019 là cấu hình gần nhất nhưng bỏ trống phần phân loài

**Butterworth, Savani & Tuyls 2019 có thiết lập gần đề tài nhất**: robot vi sai Foot-bot trong ARGoS, 12 cảm biến proximity cách đều tầm 0,2 m, 1 khoảng cách và 2 bearing tới đích (15 đầu vào), 2 đầu ra tốc độ bánh, pop 150, 1.000 thế hệ, 20 run NEAT và 20 run NEAT-GRU ([Butterworth §4, §4.2.1](https://ar5iv.labs.arxiv.org/html/1904.06239)). Mỗi thế hệ sinh mới **10 maze dùng chung cho mọi genome**, nên ngân sách là 150 × 1.000 × 10 = 1,5 triệu episode, đúng bằng run NEAT lớn nhất của nhóm (tự tính). So với "tham số NEAT chuẩn", bài chỉ nêu các thay đổi sau:

- add node 0,005 (add GRU 0,003), vì đặt cao hơn gây bloat đáng kể;
- **survival threshold 0,55**, vì ngưỡng cao hơn giảm áp lực chọn lọc và cải thiện hiệu năng;
- weight mutation power 1,5, nhiễu đều trong ±power;
- NEAT-GRU không dùng lai ghép;
- khởi đầu tối thiểu, không có nút ẩn.

Fitness f1 = 1/√l chỉ có giá trị khi tới đích, với l là độ dài quỹ đạo chia độ dài A*. Không tới đích thì fitness bằng 0, đâm tường thì chia 10. Tác giả bỏ tổng có trọng số vì với mọi bộ trọng số, policy chỉ tối ưu một trong hai thước đo: hoặc lao thẳng tới đích rồi kẹt trong phòng, hoặc bám tường với đường rất dài. Bài **không báo** ngưỡng compatibility, hệ số distance, số loài, add connection, stagnation, elitism, tỉ lệ đột biến trọng số, activation hay thư viện, và code chỉ cấp theo yêu cầu. Code công khai gần nhất là luận văn EvolvableRNN do Butterworth kèm cặp, cho một task thưa chỉ có đầu vào khoảng cách. Luận văn này dùng pop 512, ngưỡng phân loài khởi đầu 2,0 tự tăng khi số loài vượt mục tiêu 5 (tối đa 15), add node 0,005, add connection 0,02, stagnation 25 ([population_config.py](https://raw.githubusercontent.com/RubenPants/EvolvableRNN/master/configs/population_config.py); [genome_config.py](https://raw.githubusercontent.com/RubenPants/EvolvableRNN/master/configs/genome_config.py)).

### Dòng mê cung Lehman–Stanley điều khiển số loài và gần như tắt stagnation

Dòng mê cung Lehman–Stanley cho biết phần Butterworth bỏ trống. File `maze.ne` gốc đặt pop 250, add link 0,10, add node 0,005, `recur_prob` 0,15, xác suất đột biến trọng số 0,6 với power 0,8, hệ số distance 1,0/1,0/3,0, survival 0,4 và `dropoff_age` 2.000. Giá trị dropoff này gần như tắt phạt loài già ([maze.ne](https://raw.githubusercontent.com/jal278/novelty-search-cpp/master/maze.ne)). Vòng lặp steady-state trong `noveltyexp.cpp` **chỉnh ngưỡng compatibility về mục tiêu pop/15 loài**: cứ pop/20 con thì cộng hoặc trừ 0,1, sàn 0,3. Mỗi cá thể chạy 1 lần 400 bước với fitness 300 trừ khoảng cách tới đích, trên một maze cố định ([noveltyexp.cpp](https://raw.githubusercontent.com/jal278/novelty-search-cpp/master/noveltyexp.cpp)).

Mốc so sánh là NEAT 2002: pop 150, c3 0,4, δt 3,0 cố định, add node 0,03, add link 0,05, stagnation 15 ([Stanley & Miikkulainen §4.1](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)). Từ mốc đó, cấu hình maze tăng c3 lên 3,0, tăng add link lên 0,1, giảm add node sáu lần, tăng pop lên 250 và nới survival. Các thư viện gần tác giả gốc cũng điều khiển số loài: JAIR 2004 nhắm 10 loài với bước 0,3, MultiNEAT 5–10, SharpNEAT cố định 10 loài, và template "canonical" của neat-python 2.1 đặt `target_num_species = 10` ([JAIR 2004 App. A](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume21/stanley04a-html/node9.html); [MultiNEAT](https://github.com/peter-ch/MultiNEAT/blob/master/src/Parameters.cpp); [SharpNEAT](https://github.com/colgreen/sharpneat/blob/master/src/SharpNeat/Neat/EvolutionAlgorithm/NeatEvolutionAlgorithmSettings.cs); [neat-python docs](https://neat-python.readthedocs.io/en/latest/academic_research.html)). neat-python 2.0.0 mà nhóm dùng chưa có cơ chế này; nó chỉ có trên nhánh master 2.1.0, chưa phát hành ([CHANGELOG](https://github.com/CodeReclaimers/neat-python/blob/master/CHANGELOG.md)).

Cũng cần nhớ rằng chi tiết implementation nặng ký không kém các tham số được liệt kê. Novelty search trên hard maze cho 39/40 run ở bản gốc, 93/100 ở Gravina và 10/10 ở goNEAT_NS, nhưng chỉ 1–4/30 ở Sarti, người dùng code của sách Packt ([Sarti Table 2](https://link.springer.com/article/10.1007/s42979-022-01064-6); [Gravina §VII](https://ar5iv.labs.arxiv.org/html/1706.02556); [goNEAT_NS](https://raw.githubusercontent.com/yaricom/goNEAT_NS/master/README.md)).

| Nguồn | Pop × thế hệ (ngân sách) | Thêm node / link (xóa) | Kiểm soát số loài | Survival; stagnation | Đột biến trọng số | Khởi đầu; recurrent | Map mỗi lượt đánh giá | Chọn champion |
|---|---|---|---|---|---|---|---|---|
| [Butterworth 2019](https://ar5iv.labs.arxiv.org/html/1904.06239) | 150 × 1.000 (1,5 triệu episode) | 0,005 (GRU 0,003) / NR | NR | 0,55; NR | power 1,5, nhiễu đều | tối thiểu, không ẩn; NEAT-GRU | 10 map mới mỗi thế hệ, dùng chung | 6 genome mỗi 25 thế hệ, chấm trên chính 209 map test |
| [Lehman & Stanley 2011](https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/lehman_ecj11.pdf), [code](https://raw.githubusercontent.com/jal278/novelty-search-cpp/master/maze.ne) | 250, ≤ 250.000 eval | 0,005 / 0,10 | ngưỡng động, mục tiêu pop/15 | 0,4; dropoff 2.000 | xác suất 0,6, power 0,8 | start genome từ file; `recur_prob` 0,15 | 1 maze cố định | dừng ở lời giải đầu tiên |
| [NEAT 2002](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) | 150 | 0,03 / 0,05 | δt 3,0 cố định, không chuẩn hóa | –; 15 | 80% genome; 90% nhiễu, 10% gán mới | nối đủ, không ẩn | – | – |
| [Silva 2015](https://repositorio.iscte-iul.pt/bitstream/10071/10504/5/p1059_single.pdf) | 40 mỗi robot (odNEAT), 200 (rtNEAT) | 0,03 / 0,05 | δ 3,0; thử mục tiêu 1–10 loài, kết quả tương tự | – | rate 0,1, độ lớn 0,5 | NR | tiến hóa online | 100 lần khởi động lại mỗi run |
| [Packt chương 6](https://github.com/PacktPublishing/Hands-on-Neuroevolution-with-Python/blob/master/Chapter6/maze_config.ini), chuyển thành cấu hình tham chiếu của run 16%/38% (đo cục bộ) | 150 × 1.250 (1,5 triệu) | 0,1 / 0,5 (0,1 / 0,1) | 3,0 cố định → 1–3 loài, trung bình 1,3 | 0,1; 100 | 0,8 / 0,5 / 0,1, Gauss | `partial_direct 0.5`, 1 ẩn; recurrent | 8 map | validation 25 map mỗi cấp |
| Cấu hình gốc của nhóm | 150 | 0,03 / 0,05 (0,01 / 0,01) | 3,0 cố định → 1 loài | 0,2; 15 | 0,8 / 0,5 / 0,1, Gauss | `full_direct`, 0 ẩn; recurrent, output có self-loop | 8 map | validation; `fitness_threshold` 16 gây dừng sớm |
| [NEAT-NC 2026](https://raw.githubusercontent.com/HHNM/NEAT-NC/main/config/config-neatPC.txt) | 50 × 10 (≤ 500 episode) | 0,2 / 0,5 (0,2 / 0,2) | 3,0 cố định | 0,2; 20 | 0,8 / 0,5 / 0,1 | `partial_direct 0.5`; recurrent | 1 episode, map cố định | cá thể tốt nhất thế hệ cuối |

Bảng cho thấy một điều dễ bị bỏ qua: **con số 16%/38% không đến từ cấu hình gốc của nhóm mà từ cấu hình tham chiếu chuyển từ Packt chương 6** (pop 150, 15 đầu vào, tanh, `no_fitness_termination = True`; đọc từ `patched.ini` của run). Cấu hình này đã có tỉ lệ cấu trúc cao và stagnation dài, nên những chỗ lệch còn lại phải tìm ở nơi khác.

### Sáu chỗ cấu hình của nhóm lệch khỏi các bài thành công

**Thứ nhất, phân loài là chỗ lệch lớn nhất.** neat-python 2.0.0 tính distance bằng phần node cộng phần connection. Mỗi phần bằng tổng chênh lệch của gene khớp cộng hệ số disjoint nhân số gene không khớp, rồi chia cho số gene của genome lớn hơn. Gene được ghép theo khóa và không có hệ số excess riêng ([genome.py L523-572](https://github.com/CodeReclaimers/neat-python/blob/v2.0.0/neat/genome.py#L523-L572)). Bài 2002 cho phép đặt N = 1 khi cả hai genome dưới 20 gene, nên mỗi gene không khớp cộng nguyên 1,0 và chỉ ba gene khác nhau đã vượt δt = 3,0 ([Stanley §3.3](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)). Genome 15 đầu vào có 30–32 connection ngay từ đầu, nên dưới phép chuẩn hóa mỗi gene không khớp chỉ cộng khoảng 1/32. Vì vậy **ngưỡng 3,0 của bài báo không cùng thang với 3,0 trong neat-python** (tự tính).

Số đo khớp với phép tính. Với cấu hình gốc của nhóm, distance ở thế hệ 0 trung bình 1,081, lớn nhất 2,165, chỉ có 1 loài, và khi chạy thử với fitness ngẫu nhiên thì tới thế hệ 29 mới lên 2 loài (đo cục bộ). Hai dòng dõi cùng cấu trúc, ở trạng thái dừng của đột biến trọng số, cũng chỉ cách nhau khoảng 1,9 (tự tính). Run NEAT 1,5 triệu với cấu hình tham chiếu ghi 1–3 loài, trung bình 1,3 qua 125 lần log (đo cục bộ).

Cực ngược lại xuất hiện khi `num_hidden > 0`. Khóa nút ẩn ban đầu lấy từ một bộ đếm dùng chung cho mọi genome, nên với mạng 15-8-2 `full_direct`, mọi nút ẩn và mọi connection chạm vào chúng đều không khớp giữa hai genome. Distance khi đó khoảng 3,45 và mỗi genome thành một loài riêng ([genome.py L112-124](https://github.com/CodeReclaimers/neat-python/blob/v2.0.0/neat/genome.py#L112-L124); tự tính). Hành vi này có từ bản 0.92, vẫn còn trên master, và không tìm thấy issue nào báo. Hệ quả là NEAT của nhóm trên thực tế là một GA có mọc cấu trúc: `species_elitism`, stagnation và cơ chế bảo vệ cấu trúc mới gần như không có tác dụng.

Bằng chứng ở đây có giới hạn. Silva thử cả ngưỡng cố định lẫn ngưỡng động với mục tiêu 1–10 loài và thấy kết quả tương tự, và chưa nghiên cứu nào cô lập tác động của phân loài lên success held-out trong điều hướng ([Silva §4.1](https://repositorio.iscte-iul.pt/bitstream/10071/10504/5/p1059_single.pdf)). Một run thử của nhóm với mục tiêu 10 loài giữ được 8–14 loài nhưng đạt 12% và 0% ở cấp 3 và 4 (180.000 episode, 1 seed). Log của run đó ghi ngưỡng bị đẩy lên 14, nên bộ điều khiển cần được kiểm tra trước khi rút kết luận (đo cục bộ). Phân loài đúng nghĩa là điều kiện để gọi thuật toán là NEAT, nhưng chưa phải đòn bẩy hiệu năng đã được chứng minh.

**Thứ hai, áp lực chọn lọc đi ngược hướng các bài thành công.** Trong neat-python, mọi con đều sinh từ lai ghép hai cha mẹ rút trong top `survival_threshold` của loài rồi mới đột biến. Không có phần 25% con chỉ đột biến như NEAT 2002 ([reproduction.py L254-272](https://github.com/CodeReclaimers/neat-python/blob/v2.0.0/neat/reproduction.py#L254-L272); [Stanley §4.1](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)). Cấu hình gốc dùng survival 0,2 và cấu hình tham chiếu dùng 0,1, tức với một loài và pop 150 chỉ 15–30 genome được làm cha mẹ. Lehman dùng 0,4, còn Butterworth dùng 0,55 và nêu rõ lý do là giảm áp lực chọn lọc.

Stagnation cũng lệch hướng. Cấu hình gốc loại loài sau 15 thế hệ không cải thiện, trong khi fitness là trung bình 8 map ngẫu nhiên, nên "15 thế hệ không cải thiện" chủ yếu phản ánh nhiễu. Lehman đặt `dropoff_age` 2.000, MultiNEAT đặt stagnation 25.000, goNEAT_NS 200, Packt chương 6 là 100 ([maze.ne](https://raw.githubusercontent.com/jal278/novelty-search-cpp/master/maze.ne); [MultiNEAT](https://github.com/peter-ch/MultiNEAT/blob/master/src/Parameters.cpp)). Stagnation của neat-python so với `max(fitness_history)` và không tự reset. Khi curriculum lên cấp và thang fitness đổi, loài đang học vẫn có thể bị loại. Tuy vậy, điểm này chỉ có nghĩa khi đã có nhiều hơn một loài.

**Thứ ba, tỉ lệ đột biến cấu trúc chép từ NEAT C++ nhưng chạy theo ngữ nghĩa khác.** Ba thư viện hiểu "xác suất đột biến" theo ba cách:

- NEAT C++: có con chỉ đột biến, còn con lai chỉ bị đột biến theo một xác suất.
- SharpNEAT: mỗi con vô tính nhận đúng một loại đột biến (94% trọng số, 1% thêm node, 2,5% thêm và 2,5% xóa connection), và chỉ một nửa số con là vô tính.
- neat-python: đột biến **mọi** con, với bốn lần rút độc lập cho thêm/xóa node và thêm/xóa connection ([genome.py L338-366](https://github.com/CodeReclaimers/neat-python/blob/v2.0.0/neat/genome.py#L338-L366); [SharpNEAT](https://github.com/colgreen/sharpneat/blob/master/src/SharpNeat/Neat/Reproduction/Asexual/NeatAsexualReproductionSettings.cs)).

Vì vậy cặp 0,03/0,05 hợp lý trong NEAT C++ lại thấp trong neat-python, nhất là khi xác suất xóa gần bằng 0. Số đo khớp với nhận định này. Cấu hình gốc của nhóm giải XOR 87/100 lần, so với 100/100 của config mẫu. Trên hard maze có novelty, nó giải 12/20 run, so với 20/20 của cấu hình tham chiếu (đo cục bộ). Nhưng tỉ lệ cao chưa đủ: cấu hình tham chiếu vẫn chỉ đạt 16% và 38%, và genome tốt nhất các thế hệ cuối của run 1,5 triệu đã có 26–34 node (đo cục bộ). Trong khi đó Butterworth cảnh báo add-node cao gây bloat, và một genome không có nút ẩn của họ đã đạt 90,4%.

**Thứ tư, thang đột biến trọng số không chép được giữa các nguồn.** neat-python cộng nhiễu Gauss N(0, power) lên từng gene với xác suất `weight_mutate_rate` và gán giá trị mới với xác suất `replace_rate` ([attributes.py L68-80](https://github.com/CodeReclaimers/neat-python/blob/v2.0.0/neat/attributes.py#L68-L80)). Các nguồn khác dùng thang khác:

- Lehman: power 0,8.
- Butterworth: power 1,5 với nhiễu đều, tương đương độ lệch chuẩn khoảng 0,87 (tự tính).
- MultiNEAT: power tối đa 1,0.

Thêm vào đó, `tanh` của neat-python tính tanh(2,5z), nên mỗi đơn vị trọng số dốc gấp 2,5 lần ([activations.py L11-18](https://github.com/CodeReclaimers/neat-python/blob/v2.0.0/neat/activations.py#L11-L18)). Gomes thấy NEAT dùng fitness tốt nhất khi mutation rate 0,25–0,8 ([Gomes §4.3](https://repositorio.iscte-iul.pt/bitstream/10071/23144/1/conferenceObject_26611.pdf)). Số đo cục bộ đi ngược hướng "tăng power": hạ power của trọng số và bias từ 0,5 xuống 0,1, kèm replace 0,01, cho 22% và 30%, so với 6% và 20% (lịch đều, 180.000 episode, 1 seed mỗi bên). Power vì thế phải được quét, không chép.

**Thứ năm, điều kiện dừng và champion đang đo độ may.** neat-python dừng khi max fitness trên toàn quần thể chạm `fitness_threshold`. Với 8 map ngẫu nhiên, chỉ cần một genome gặp 8 map dễ là dừng. Cấu hình gốc với ngưỡng 16 từng dừng ngay ở thế hệ 0–1 trên phòng trống (đo cục bộ), còn Butterworth chạy đủ ngân sách. `best_genome` mà `run()` trả về là fitness cao nhất từng thấy qua mọi thế hệ, tức max của khoảng 22.500 giá trị nhiễu ở 180.000 episode, và chỉ là một tham chiếu chứ không phải bản sao ([population.py](https://github.com/CodeReclaimers/neat-python/blob/v2.0.0/neat/population.py); [#271](https://github.com/CodeReclaimers/neat-python/issues/271); tự tính). Pipeline hiện tại chọn champion trên 25 map validation mỗi cấp. Cách này tốt hơn `best_genome` nhưng vẫn rất nhiễu (xem mục sau).

**Thứ sáu, recurrent của neat-python không phải GRU.** Butterworth cho thấy chính nút GRU tường minh mới tạo khác biệt. Số run có genome thắng I-Bug trên hai thước đo là 10/20 với NEAT-GRU so với 3/20 với NEAT. Khi bỏ bearing, buộc agent tích hợp thông tin theo thời gian, NEAT-GRU giải 10/10 run so với 0/10 của NEAT ([Butterworth §5.2, §6](https://ar5iv.labs.arxiv.org/html/1904.06239)). `RecurrentNetwork` của neat-python chỉ là mạng rời rạc. Mỗi lần `activate` là một bước, mỗi hop qua nút ẩn trễ một bước, và phải `reset()` giữa các episode. Với `feed_forward = False`, mọi node không phải input có self-connection ngay từ thế hệ 0 ([recurrent.py L21-42](https://github.com/CodeReclaimers/neat-python/blob/v2.0.0/neat/nn/recurrent.py#L21-L42); [genome.py L636-664](https://github.com/CodeReclaimers/neat-python/blob/v2.0.0/neat/genome.py#L636-L664)).

Ở 180.000 episode, bản feed-forward của cấu hình tham chiếu đạt 12% và 40%, so với 6% và 20% của bản recurrent (đo cục bộ, 1 seed). Vậy kết nối hồi quy kiểu này chưa giúp ở ngân sách đó. Seriani 2021, bài duy nhất dùng LiDAR tầm 3 m như đề tài, thấy lookback lớn hơn 0 tốt hơn lookback 0, nhưng bài chỉ báo fitness, không có success held-out ([Seriani §3–4](https://link.springer.com/article/10.1007/s10846-020-01308-8)). 15 đầu vào của nhóm trùng Butterworth, và CMA-ES với cùng đầu vào đạt 46–68%, nên đầu vào không phải nút thắt chính của NEAT.

Từ các nguồn trên, điểm khởi đầu hợp lý cho neat-python 2.0.0 như sau (đề xuất, chưa kiểm trên task):

- Viết một reporter chỉnh `compatibility_threshold` mỗi thế hệ ±0,1–0,3 về mục tiêu 8–12 loài với pop 150, sàn 0,3, như JAIR 2004 và code Lehman. Cách này làm được vì `speciate` đọc lại ngưỡng từ config mỗi lần gọi ([species.py L79](https://github.com/CodeReclaimers/neat-python/blob/v2.0.0/neat/species.py#L79-L125)).
- Dùng `num_hidden = 0`, hoặc thống nhất khóa nút ẩn bằng một lớp con của `DefaultGenome`, giống cách code Lehman sinh cả quần thể từ một start genome chung có 1 nút ẩn ([mazestartgenes](https://raw.githubusercontent.com/jal278/novelty-search-cpp/master/mazestartgenes)).
- Survival 0,3–0,55.
- Stagnation 30–100 hoặc tắt hẳn; xóa `fitness_history` khi lên cấp.
- Giữ tỉ lệ cấu trúc của cấu hình tham chiếu, vì cấu hình này đã qua XOR và hard maze; sau đó ablation node add 0,03 so với 0,1.
- Quét power trọng số trong khoảng 0,1–1,0.
- Đặt `no_fitness_termination = True` và khai báo tường minh `single_structural_mutation = False`. Default của khóa này là chuỗi `'false'`, mà chuỗi này lại mang giá trị truthy ([genome.py L38](https://github.com/CodeReclaimers/neat-python/blob/v2.0.0/neat/genome.py#L38)).

Muốn dùng `target_num_species` có sẵn thì phải cài neat-python từ master 2.1.0. Bản này còn đổi cách ghép gene khi tính distance và quy tắc disable 75%, nên phải chạy lại XOR và hard maze trước khi dùng ([CHANGELOG](https://github.com/CodeReclaimers/neat-python/blob/master/CHANGELOG.md)).

## Giao thức đáng theo là map mới mỗi thế hệ, validation 200 map và test chỉ chạm một lần

Về sinh dữ liệu, các bài gần đề tài dùng generator thủ tục có kiểm tra giải được. Generator của McGuire, mà Butterworth dùng lại cho cả train lẫn test, hoạt động như sau. Hai agent xuất phát đúng tại vị trí start và target, đi trên lưới thô, đi thẳng với xác suất p_str và dừng khi hành lang chiếm t_cor = 0,4. Generator kiểm tra liên thông và sinh lại nếu hỏng, rồi dựng tường, chia phòng và đục cửa ([McGuire §4.2](https://arxiv.org/pdf/1808.05050)). Các bài mê cung có ba cách khác:

- recursive backtracking trên lưới 4 × 4, 15 maze cho mỗi độ dài đường tối ưu từ 4 đến 14 ô (Lehman 2013);
- randomized Prim đã lọc trùng (Shorten);
- sinh hàng trăm maze rồi chỉ giữ 60 maze mà objective search thất bại còn NS-LC giải được ít nhất 1/50 run (Gravina).

Nguồn của ba cách trên: [Lehman 2013 §3.2](https://nn.cs.utexas.edu/downloads/papers/lehman.gecco13.pdf); [Shorten §2.3](https://people.cs.uct.ac.za/~gnitschke/projects/papers/2015-Evolving%20Generalised%20Maze%20Solvers.pdf); [Gravina §IV-C](https://arxiv.org/abs/1807.02397). Pinville không sinh map mà sinh "context", tức bộ giá trị của mọi tham số task như vị trí xuất phát, hướng và kích thước môi trường ([Pinville §2.1, Table 4](https://groups.csail.mit.edu/EVO-DesignOpt/gecco2011Proceedings/proceedings/p259.pdf)). BARN lọc map bằng flood-fill trên C-space đã phình theo kích thước robot, và xếp độ khó bằng chính planner tham chiếu thay vì suy từ tham số sinh ([Perille §III-A, §IV-A](https://arxiv.org/html/2008.13315v3)).

Về chia tập, Butterworth sinh map train bằng đúng generator và tham số đã sinh bộ test 209 map 14 × 14 m, nên train và test cùng phân phối. Map train là dòng sinh mới liên tục, còn test là bộ đã lưu, nên test gần như chắc chắn chưa thấy, dù bài không khẳng định ([Butterworth §4.1, §4.2.1](https://ar5iv.labs.arxiv.org/html/1904.06239)). McGuire chạy mỗi thuật toán đúng một lần trên mỗi môi trường (300 s, tới đích khi vào bán kính 1 m), chuẩn hóa quãng đường bằng A* 8 hướng và so sánh bằng bootstrap 10.000 mẫu. Đây là khuôn mẫu đúng cho các baseline không học ([McGuire §5.1, App. C](https://arxiv.org/pdf/1808.05050)).

Điểm yếu của Butterworth nằm ở khâu chọn. Cứ 25 thế hệ, 6 genome được chấm ngay trên 209 map test, tức khoảng 240 ứng viên và 50.160 episode test mỗi run (tự tính). Bài không có validation, không nêu số lần chạy mỗi map và không có khoảng tin cậy. Ở mức 93%, sai số chuẩn trên 209 map khoảng 1,8 điểm. Lấy max của 240 ứng viên ngang tài sẽ phồng điểm tối đa khoảng +5 điểm (tự tính), **cùng cỡ với khoảng cách 3,8 điểm giữa genome tốt nhất (203/209) và I-Bug (195/209)**. Bài cũng gọi bộ test là 209 map trong khi bản arXiv của McGuire mô tả 200, và chênh lệch này chưa giải thích được.

Bằng chứng trực tiếp nhất về chia tập trong evolutionary robotics là Pinville et al. 2011. Bài định nghĩa train, validation và test theo nghĩa machine learning, rồi so năm cách cùng ngân sách:

- **Fit-only**: một context cố định;
- **Eval-all**: mọi context validation;
- **FRI**: một tập con ngẫu nhiên chọn đầu run rồi giữ cố định;
- **Jakobi**: tập con mới mỗi thế hệ;
- **ProGAb**: chọn theo khả năng tổng quát hóa đo trên validation.

Nguồn: [Pinville §4.3](https://groups.csail.mit.edu/EVO-DesignOpt/gecco2011Proceedings/proceedings/p259.pdf). Trên T-maze, Fit-only chỉ đạt trung vị 3/180 context validation. Số run đạt ít nhất 150/180 là 0, 1, 6, 12 và 14 trên 20, theo đúng thứ tự năm cách trên. **ProGAb đạt 170/180 trên validation và 150/180 trên test**, không khác nhau có ý nghĩa (p = 0,45). Jakobi so với FRI là 153 so với 80, nhưng p = 0,15 ([Pinville §5.1, Table 5](https://groups.csail.mit.edu/EVO-DesignOpt/gecco2011Proceedings/proceedings/p259.pdf)). Trên bài gom bóng, test lệch phân phối so với validation: Jakobi tụt từ 2,13 bóng trên validation xuống 0,74 trên test, còn ProGAb giữ 2,63 và 2,23 ([Pinville §5.2](https://groups.csail.mit.edu/EVO-DesignOpt/gecco2011Proceedings/proceedings/p259.pdf)).

Pagliuca & Nolfi 2019 thêm hai số định lượng. Thứ nhất, đổi điều kiện giúp CMA-ES, xNES, sNES và SSS nhưng không ảnh hưởng NEAT; **CMA-ES tốt nhất khi đổi với tần suất 0,04–0,1 (0,571) và kém hơn khi đổi mỗi thế hệ (0,455)**. Thứ hai, khi giữ ngân sách, bước nhảy lớn nhất về số episode mỗi cá thể là từ 1 lên 10 (CMA-ES từ 0,125 lên 0,413); hiệu năng đạt đỉnh ở 20–200 episode và giảm nhẹ ở 1.000 ([Pagliuca Table 1–2](https://pmc.ncbi.nlm.nih.gov/articles/PMC6396973/)). Milano et al. cũng thấy biến thiên vừa phải là tốt nhất ([arXiv 1710.07913](https://arxiv.org/abs/1710.07913)).

Các cách khác đều có điểm yếu riêng. Shorten & Nitschke giữ cố định 100 perfect maze, chọn genome đầu tiên giải đủ 100 map rồi test trên 10.000 maze 21 × 21. Khoảng cách train–test gần bằng 0 (97,4–99,5%), có lẽ vì perfect maze có một chiến lược phổ quát là bám tường ([Shorten §3.2, Table 1](https://people.cs.uct.ac.za/~gnitschke/projects/papers/2015-Evolving%20Generalised%20Maze%20Solvers.pdf)). Các benchmark khác tách test bằng seed hoặc level rời nhưng hiếm khi có validation để chọn checkpoint:

- BARN Challenge công bố 300 map để phát triển và chấm trên 50 map ẩn, mỗi map 10 trial ([Report 2024](https://arxiv.org/html/2407.01862)).
- Repo Lahouazi dùng 4 map mới mỗi thế hệ, chung cho cả quần thể. Validation 50 map từ seed 5.000.000 chỉ để ghi log, test là 200 map từ seed 10.000.000 + i, và controller trả về là mean của CMA-ES ([neuroevolution.py](https://raw.githubusercontent.com/GayaLahouazi/rl-robot-navigation/main/robotnav/neuroevolution.py)).
- Với mạng CNN lớn, CoinRun overfit đáng kể khi train dưới 4.000 level, và Procgen cần khoảng 10.000 level để đóng khoảng cách tổng quát hóa ([CoinRun §3.2](https://ar5iv.labs.arxiv.org/html/1812.02341); [Procgen §3.1](https://ar5iv.labs.arxiv.org/html/1912.01588)).

Về chọn champion, các nghiên cứu cẩn thận đều chấm lại một số ít ứng viên trên một tập lớn và cố định. Deep GA chạy thêm 30 episode cho top 10 mỗi thế hệ. World Models chạy 1.024 rollout cho agent tốt nhất mỗi 25 thế hệ. Pagliuca hậu kiểm cá thể tốt nhất mỗi thế hệ trên một ma trận điều kiện cố định và tính phần đó vào ngân sách ([Deep GA §3.1](https://ar5iv.labs.arxiv.org/html/1712.06567); [World Models App. A.4](https://ar5iv.labs.arxiv.org/html/1803.10122); [Pagliuca §3](https://pmc.ncbi.nlm.nih.gov/articles/PMC6396973/)).

Giao thức hiện tại của nhóm gồm 8 map rút từ pool 250 mỗi cấp, champion chọn trên 25 map validation mỗi cấp và test 50 map. Hướng đi đúng, nhưng hai khâu còn quá nhiễu. Ở p khoảng 0,4–0,6, sai số chuẩn trên 25 map khoảng 10 điểm. Chọn ứng viên tốt nhất trong 20 ứng viên ngang tài làm điểm validation phồng khoảng 18 điểm. Hai ứng viên thật sự cách nhau 10 điểm bị xếp ngược với xác suất khoảng 24% ở 25 map, so với khoảng 2% ở 200 map (tự tính). Log của run NEAT 1,5 triệu minh họa đúng điều này: success validation ở cấp 4 nhảy từ 24% xuống 12% giữa hai lần ghi liền nhau (đo cục bộ). Trên test, độ lệch chuẩn theo seed của CMA-ES ở cấp 4 khoảng 13 điểm (70%, 44% và 58% với 3 seed, tường mỏng), lớn hơn nhiễu do lấy mẫu 50 map, khoảng 7 điểm. Vì vậy, khi test đã có khoảng 200 map, thêm seed có giá trị hơn thêm map (tự tính).

| Thành phần | Hiện tại | Đề xuất | Căn cứ |
|---|---|---|---|
| Map train | 8 map rút từ pool 250 mỗi cấp | 8 map mới mỗi thế hệ từ dải seed train riêng, lọc A*, dùng chung cho mọi cá thể trong thế hệ | Butterworth, Lahouazi, CoinRun bản không giới hạn; Pinville không thấy lấy lại map kém hơn giữ cố định |
| Số map mỗi lượt | 8 | Giữ 8 ở 180.000 episode; K = 16 làm CMA-ES tụt còn 22% và 24% vì mất nửa số thế hệ (đo cục bộ, tường mỏng) | Pagliuca: lợi lớn nhất từ 1 lên 10 episode |
| Lịch train | NEAT curriculum, CMA-ES đều | Một lịch chung cho mọi thuật toán; curriculum chỉ là ablation | Run NEAT 1,5 triệu chưa từng gặp cấp 4 (đo cục bộ) |
| Validation | 25 map mỗi cấp | 200 map cố định mỗi cấp, dải seed rời, chỉ dùng để chọn champion | Xác suất xếp ngược giảm từ khoảng 24% xuống khoảng 2% |
| Champion NEAT | theo validation, quy tắc chưa chốt | Mỗi 10 thế hệ chấm top 3 (từ ít nhất 2 loài) trên validation, giữ hall-of-fame bằng deepcopy; không dùng `best_genome` | Deep GA, World Models, Pagliuca |
| Champion CMA-ES | theo validation | Mỗi 25 thế hệ chấm mean `xfavorite` và mẫu tốt nhất của thế hệ | Lahouazi trả về mean |
| Ngân sách | 1 triệu (CMA-ES) so với 1,5 triệu (NEAT) | Cùng số episode, tính cả episode validation (khoảng 5% với NEAT, tự tính) | Pagliuca, Such |
| Test | 50 map mỗi cấp | 200 map trong file cố định có start/goal, 1 episode mỗi map, cùng map cho mọi thuật toán, kiểm định cặp (McNemar) | McGuire |
| Test lệch phân phối | BARN, mê cung Lehman–Stanley | Thêm một họ map held-out kiểu McGuire, báo riêng cùng họ và khác họ | Pinville: Jakobi 2,13 → 0,74 |
| Seed | 1–3 | Ít nhất 10 cho kết luận chính, 5 cho ablation; seed là đơn vị phân tích, báo trung vị, IQR và CI theo seed | Độ lệch chuẩn seed khoảng 13 điểm |
| Kỷ luật test | chưa nêu | Chốt giao thức trước khi chạy; nếu test đã dùng cho bất kỳ quyết định nào thì nó thành validation và phải sinh test mới | Thiên lệch kiểu Butterworth |

Tài liệu neuroevolution không cho con số "cần bao nhiêu map", nên nhóm nên tự đo bằng một thí nghiệm quét kích thước pool. Thiết kế đề xuất: N ∈ {25, 100, 250, 1.000, không giới hạn}, chạy trên cấp 3 và 4, cho NEAT và CMA-ES, 5 seed mỗi ô. Tổng cộng 50 run mỗi cấp, khoảng 8,3 giờ ở 10 phút mỗi run. Với 5 seed, thí nghiệm chỉ phát hiện được khác biệt cỡ 15 điểm trở lên (tự tính). Riêng CMA-ES nên thêm ablation tần suất đổi map (mỗi 1, 10 và 25 thế hệ), vì đó là chỗ Pagliuca đo được hiệu ứng rõ nhất.

## Không bài NEAT nào đo bẫy tương đương; so trong nhà, 16% là thấp còn 38% lệch lịch train

Tài liệu về mê cung đánh lừa dùng một thước đo khác với nhóm, và phải tách hai thước đo ra trước khi so. Thước đo kinh điển là **tỉ lệ run tìm được lời giải cho chính map dùng để train**:

- Hard maze: NEAT theo fitness khoảng cách chỉ giải 3/40 run (7,5%) trong 250.000 evaluation, ngang chọn ngẫu nhiên (4/40). NEAT + novelty search giải 39/40, trung bình 35.109 evaluation. Trên medium maze, cả hai cách giải 40/40 ([Lehman & Stanley 2011 §6](https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/lehman_ecj11.pdf)).
- 165 maze ngẫu nhiên: nhóm phương pháp có novelty giải nhiều run hơn có ý nghĩa (Fisher p < 0,001), nhưng khác biệt chỉ xuất hiện từ độ khó 7 trở lên ([Lehman 2013 §4](https://nn.cs.utexas.edu/downloads/papers/lehman.gecco13.pdf)).
- 60 maze procedural được lọc sao cho objective search thất bại 50/50 run: phương pháp QD tốt nhất chỉ đạt 451/3.000 run (15%), NS thuần 149/3.000 (5%) ([Gravina §IV-C, §V-B](https://arxiv.org/abs/1807.02397)).

Các số này nói rằng fitness khoảng cách rất yếu trước cấu trúc lừa. Nhưng chúng không phải **success của một controller cố định trên map chưa thấy**, là loại số mà 16% và 38% thuộc về.

Các bài NEAT có số held-out đều ở điều kiện dễ hơn nhiều:

- Butterworth: 90,4% với genome không có nút ẩn; 93,7% và 97,1% với hai genome NEAT-GRU tốt nhất; I-Bug đạt 93,3%. Thời gian là 300 s, trên loại map mà Bug đơn giản đạt khoảng 60–90%, và các số trên là best-of-20-run chọn ngay trên test ([Butterworth §5–6](https://ar5iv.labs.arxiv.org/html/1904.06239); [McGuire §5.1](https://arxiv.org/pdf/1808.05050)).
- Shorten: 97,4–99,5% trên 10.000 perfect maze với 8.000 bước, nơi bám tường là lời giải phổ quát ([Shorten Table 1](https://people.cs.uct.ac.za/~gnitschke/projects/papers/2015-Evolving%20Generalised%20Maze%20Solvers.pdf)).
- Silva: 69,4–73,4% trong arena 3 × 3 m với tiến hóa online ([Silva Table 7](https://repositorio.iscte-iul.pt/bitstream/10071/10504/5/p1059_single.pdf)).

Trên cấp 3 của nhóm, chính lớp chiến lược đó chỉ đạt 28%: Bug được 28% với 500 bước và 54% với 1.000 bước trên map tường dày (đo cục bộ). Cấp 3 vì thế khó hơn map McGuire rất nhiều, và 90% của Butterworth không làm mốc được cho 16%. **Không tìm thấy bài NEAT nào báo success trên map held-out dựng quanh hành lang cụt dài hay bẫy chữ U với giới hạn thời gian chặt.** Mốc gần nhất về độ khó đến từ các phương pháp khác:

| Nghiên cứu | Phương pháp | Bản đồ test chưa thấy | Success |
|---|---|---|---|
| [Xiong 2025](https://arxiv.org/abs/2511.22338) | SAC (MLP); TEB; FTG phản xạ; Hybrid A* | 900 trial ngõ cụt, 500 bước × 0,1 s | 71,85%; 37,33%; 25,73%; 7,20% |
| [Zhelo 2018](https://ar5iv.labs.arxiv.org/html/1804.00456) | A3C + LSTM (ICM + entropy); A3C rút gọn | floor plan Map2, Map3, Map4 | 82,7 / 72,7 / 55,0%; 66,0 / 51,3 / 44,3% |
| [Yang 2025](https://arxiv.org/html/2506.05997) | GRU, LSTM, SRU | 120 môi trường maze | 68,1–76,0% |
| [Xu 2023](https://arxiv.org/html/2210.04839v2) | MLP, GRU, Transformer; DWA | 50 môi trường BARN | 65%, 51%, 68%; DWA 82% |
| [Meijer 2024](https://arxiv.org/abs/2407.13530) | RMP + FFN / RNN | scan thật SUN3D; BundleFusion | 0,68 / 0,68; 0,74 / 0,72 |
| [McGuire 2019](https://arxiv.org/pdf/1808.05050) | Com (không nhớ); các Bug có nhớ | 200 môi trường trong nhà, 300 s | khoảng 60%; khoảng 90% |
| [Lahouazi](https://github.com/GayaLahouazi/rl-robot-navigation) | CMA-ES 738 trọng số; PPO; potential field | 200 map 10 × 10 m vật cản tròn (map dày) | 87% (56%); 90% (66%); 66% (35%) |

Hai quy luật từ tài liệu áp thẳng vào đề tài. Thứ nhất, chiến lược không nhớ kẹt nhiều hơn rõ rệt khi thiếu thông tin hướng: Com đạt 60% so với 90% của các Bug có nhớ, và NEAT đạt 0/10 so với 10/10 của NEAT-GRU khi bỏ bearing. Nhưng khi đã có bearing, trí nhớ chỉ cho lợi thế nhỏ hoặc không có. Butterworth gọi lợi thế của GRU là "slight", Meijer thấy RNN ngang FFN, còn ở Xu thì GRU thấp hơn MLP ([Butterworth §5.2](https://ar5iv.labs.arxiv.org/html/1904.06239); [Meijer Table I](https://arxiv.org/abs/2407.13530)).

Thứ hai, trên mạng nhỏ cố định, CMA-ES hiệu quả mẫu hơn NEAT nhiều lần: 895 so với 3.600 evaluation ở double pole có vận tốc, 6.061 so với 33.184 khi không có vận tốc, và 0,571 so với 0,194 ở double pole mở rộng cùng ngân sách ([Igel 2003 Table 4–5](https://christian-igel.github.io/paper/NfRLUES.pdf); [Pagliuca Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC6396973/)). Ngoại lệ là một maze có stepping stone, nơi NEAT qua khoảng 5/7 mốc so với khoảng 3/7 của CMA-ES ([Nisioti 2025 §5.1](https://arxiv.org/html/2505.22696v1)). Tỉ số CMA-ES/NEAT của nhóm là 2,9 lần ở cấp 3 (46/16) và 1,8 lần ở cấp 4 (68/38) (tự tính). Con số này cùng cỡ 2,9 lần của Pagliuca, dù hai thước đo khác nhau. Vậy **khoảng cách đi đúng hướng tài liệu dự đoán**, chứ không phải dấu hiệu engine NEAT hỏng; engine đã qua kiểm thử XOR và hard maze.

Bảng số đo cục bộ dưới đây đặt 16% và 38% cạnh các run khác của nhóm trên cùng bộ map tường dày 0,20 m, cùng 50 map test mỗi cấp.

| Run (đo cục bộ) | Ngân sách, lịch train | Seed | Cấp 3 | Cấp 4 |
|---|---|---|---|---|
| CMA-ES 15-8-2 | 1 triệu, đều | 1 | 46% | 68% |
| CMA-ES 15-8-2 | 180.000, đều | 3 | 34–38% | 30–72% |
| NEAT tham chiếu, recurrent | 1,5 triệu, curriculum, dừng ở cấp 3 | 1 | 16% | 38% |
| NEAT tham chiếu, recurrent | 180.000, curriculum, dừng ở cấp 2 | 3 | 10–16% | 0–2% |
| NEAT tham chiếu, recurrent / feed-forward | 180.000, đều | 1 mỗi biến thể | 6% / 12% | 20% / 40% |
| NEAT tham chiếu, power trọng số và bias 0,1, replace 0,01 | 180.000, đều | 1 | 22% | 30% |
| NEAT tham chiếu, mục tiêu 10 loài | 180.000, đều | 1 | 12% | 0% |
| GA của nhóm (SimpleGA) | 180.000, đều | 1 | 16% | 16% |
| GA2: truncation 20%, đột biến mọi gen σ = 0,05 | 180.000, đều | 1 | 34% | 16% |
| Planner occupancy online + A* | không học | – | 72% | 76% (tường mỏng) |
| Bug, 500 / 1.000 bước | không học | – | 28% / 54% | 40% ở 500 bước (tường mỏng) |

Bảng cho ba kết luận.

1. Ở cấp 4, phép so 38% với 68% trộn ba khác biệt: ngân sách (1,5 triệu so với 1 triệu), lịch train (NEAT học curriculum và chưa từng gặp cấp 4, CMA-ES học đều) và chỉ 1 seed mỗi bên. Khi chạy lịch đều ở 180.000 episode, NEAT đã đạt 20–40%, nằm trong dải 30–72% của ba seed CMA-ES.
2. Ở cấp 3, khoảng cách bền hơn: không run NEAT nào vượt 22%, trong khi cả ba seed CMA-ES và GA2 đều ở 34–38%. Khoảng Wilson 95% của 16% là [8,3; 28,5], so với [33,0; 59,6] của 46% (tự tính).
3. NEAT 16% cũng không tách được khỏi Bug 28%, có khoảng Wilson [17,5; 41,7].

Tóm lại, **16% ở cấp 3 là thấp so với chính các controller phản xạ học được trên cùng map, còn 38% ở cấp 4 chưa đủ dữ liệu để nói là thấp.**

Câu hỏi "nếu tương đương thì không cần cải thiện" vì thế không đứng được ở cả hai vế. Vế thứ nhất: không có số công bố tương đương để làm mốc, và các mốc gần đúng loại đều cao hơn. DRL có nhớ trên map chưa thấy khó đạt 55–76%, SAC trên ngõ cụt đạt 72%; chỉ các baseline phản xạ hoặc cổ điển trong điều kiện chặt mới ở mức 16–38% (FTG 26%, TEB 37%, A3C rút gọn 44% ở Map4). Vế thứ hai quan trọng hơn: mục tiêu của đề tài là so NEAT với tiến hóa trọng số trên mạng cố định. Phép so đó chỉ hợp lệ khi NEAT chạy đúng NEAT (nhiều loài), cùng lịch, cùng ngân sách và nhiều seed. Thứ cần cải thiện là tính hợp lệ của phép so, không phải bản thân con số. Nếu NEAT đã sửa vẫn thua CMA-ES, nhóm có một kết quả khớp Igel và Pagliuca, và kết quả đó có giá trị báo cáo hơn một con số cao đạt được trong giao thức lỏng.

Một ràng buộc chung kéo mọi số tuyệt đối xuống. 500 bước × 0,1 s × 0,5 m/s chỉ cho 25 m, tức 2,5 cạnh arena. Bug trên map McGuire đi 2,5–3,3 lần độ dài A*, nên một Bug đầy đủ chỉ kịp về đích khi đường A* ngắn hơn khoảng 7,6–10 m (tự tính từ [McGuire Fig. 12](https://arxiv.org/pdf/1808.05050)). Nhóm nên báo thêm success ở 1.000–2.000 bước, tách va chạm khỏi hết giờ, và báo theo độ dài hành lang hoặc độ sâu chữ U, để tách hiệu ứng thời gian khỏi thất bại thoát bẫy.

## GA mạng cố định đáng tin chọn cắt ngọn, đột biến nhỏ mọi gen và gần như bỏ lai ghép

Các GA tiến hóa trọng số đã công bố hội tụ về một công thức tối giản:

- **Deep GA (Such et al. 2017)**: chọn lọc truncation, giữ top 20/1.001 ở Atari, 625/12.501 ở Humanoid và 61/20.001 ở Image Hard Maze, tức khoảng 0,3–5%. Đột biến Gauss **mọi** tham số với σ 0,002–0,005 trên mạng khởi tạo Xavier, một elite, và **bỏ lai ghép "cho đơn giản"**. Với fitness nhiễu, top 10 được chạy thêm 30 episode để chọn elite ([Such §3.1, Table 2](https://ar5iv.labs.arxiv.org/html/1712.06567)). Trên Image Hard Maze, GA chỉ theo reward mắc bẫy, chỉ GA + novelty giải được, và ES kém GA ([Such §4.2](https://ar5iv.labs.arxiv.org/html/1712.06567)).
- **SSS trong evorobotpy2 (Nolfi)**: giữ nửa tốt nhất của 20 cá thể, sinh nửa còn lại bằng bản sao cộng N(0; 0,02²) trên mọi trọng số, không lai ghép, dùng một seed môi trường chung cho cả thế hệ và hậu kiểm cá thể tốt nhất bằng một seed khác ([sss.py](https://raw.githubusercontent.com/snolfi/evorobotpy2/master/bin/sss.py)).
- **PLOS ONE 2018 (cùng nhóm Nolfi)**: trên bộ điều khiển recurrent 151 trọng số, gần đúng cỡ 146 của nhóm, SSS, xNES và PSHC thắng NEAT, CoSyNE và CGPANN, và 4–8 episode mỗi lượt đánh giá là tốt nhất khi giữ ngân sách ([PLOS ONE 2018 Table 4–5, 11](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0198788)).
- **Khepera kinh điển**: quần thể 80–100, lai ghép hiếm (p = 0,1 ở Floreano & Mondada) hoặc bỏ hẳn (Nolfi thử p = 0,1; 0,5; 1,0 và không mức nào hơn đột biến thuần), chọn top 20/100 rồi nhân 5 bản sao ([Floreano & Mondada 1996, Appendix](https://infoscience.epfl.ch/record/63879/files/floreano.IEEE-SMC.pdf); [Nolfi TR 96-14](https://www.cs.swarthmore.edu/~meeden/cs63/f07/nolfi.emodular.pdf)).

Gomes 2015 còn cho thấy, khi dùng novelty, GA trên mạng Elman cố định 232 trọng số không khác có ý nghĩa so với NEAT-NS tốt nhất ([Gomes §4.3](https://repositorio.iscte-iul.pt/bitstream/10071/23144/1/conferenceObject_26611.pdf)).

Với ES, công thức mặc định đã rõ. OpenAI-ES dùng σ = 0,02, Adam 0,01, 20 cặp mẫu đối xứng mỗi thế hệ, centred rank và weight decay khoảng 0,005 ([evorobotpy2 es.py](https://raw.githubusercontent.com/snolfi/evorobotpy2/master/bin/es.py); [Salimans 2017 §2.1](https://arxiv.org/abs/1703.03864)). Trong so sánh năm 2020, OpenAI-ES thắng hoặc hòa CMA-ES, xNES và sNES trên mọi bài, một phần nhờ weight decay và chuẩn hóa đầu vào giữ trọng số nhỏ ([Pagliuca 2020 Table 1–3](https://www.frontiersin.org/articles/10.3389/frobt.2020.00098/full)). CMA-ES có các thiết lập chuẩn sau ([Igel 2003 §3.1, §4](https://christian-igel.github.io/paper/NfRLUES.pdf); [HM&I 2009 §4.3](https://christian-igel.github.io/paper/NSfERL.pdf)):

- λ mặc định 4 + ⌊3 ln n⌋ = 18 với n = 146;
- σ0 khoảng 1 cho mạng feed-forward và 0,1 cho recurrent;
- sàn σ_min = σ0/2, giúp ở mọi bảng của Heidrich-Meisner & Igel;
- restart IPOP, mỗi lần tăng gấp đôi quần thể.

Trên pole balancing, CMA-NeuroES cần 585 evaluation, NEAT 3.600 và GA thông thường (CNE) 22.100 ([HM&I Table 4](https://christian-igel.github.io/paper/NSfERL.pdf)), cùng thứ tự với kết quả của nhóm. λ = 60 của nhóm gấp khoảng 3,3 lần mặc định. Quần thể lớn hơn trung bình được nhiều nhiễu hơn, nhưng chỉ còn 375 thế hệ thay vì 1.250 ở 180.000 episode (tự tính).

Đặt cạnh các công thức đó, GA của nhóm lệch ở bốn chỗ mà tài liệu đã cảnh báo.

**Lai ghép đều trên mọi con** đi ngược mọi GA và ES đã xem. Với 8 nút ẩn tanh, cùng một hàm có 8! cách sắp thứ tự nút ẩn, nhân thêm 2⁸ cách đổi dấu, tức khoảng 10,3 triệu cách mã hóa tương đương (tự tính). Khi cha mẹ dùng hai cách mã hóa khác nhau, lai ghép tạo ra nút ẩn ghép từ hai mạng. Đây đúng là vấn đề competing conventions mà bài NEAT gốc mô tả ([Stanley §2.2](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)).

**Tournament k = 3 chọn lọc yếu.** Cường độ chọn lọc của nó khoảng 0,85 độ lệch chuẩn fitness, tương đương cắt ở top 47%, so với 1,40 của truncation 20% và 2,06 của top 5% (tự tính). Nhiễu của 8 map còn hạ thêm áp lực chọn lọc lên chất lượng thật. Lý thuyết chọn lọc dưới nhiễu cho thấy truncation tối ưu một thống kê thứ tự xác định, còn tournament có thể không tối ưu đại lượng nào và có thể dao động ([GECCO 2011](https://groups.csail.mit.edu/EVO-DesignOpt/gecco2011Proceedings/proceedings/p2059.pdf)).

**Thang trọng số và bước đột biến quá lớn.** N(0,1) gấp 3–4 lần thang Xavier, vốn có độ lệch chuẩn 0,26 cho lớp 15→8 và 0,35 cho lớp 8→2. Với khởi tạo này, tiền kích hoạt của nút ẩn có độ lệch chuẩn tới khoảng 3,9 khi các đầu vào gần 1, đủ làm bão hòa tanh. Mỗi gen bị đột biến nhảy 0,5 lần thang trọng số, so với khoảng 0,1 lần ở Such hay evorobotpy2 (tự tính). Riêng tỉ lệ 10% gen thì không sai: Floreano đột biến 20% gen, Nolfi đột biến 2% bit, và cả hai đều thành công.

**Elite có thể giữ điểm may mắn.** Nếu 2 elite mang điểm 8 map cũ sang thế hệ sau, quần thể giữ lại đúng những cá thể từng gặp map dễ. SSS sửa việc này bằng cách đánh giá lại mọi cá thể được giữ trên seed mới ở mỗi thế hệ.

Số đo cục bộ ủng hộ chẩn đoán này nhưng chưa tách được từng nguyên nhân. GA cũ đạt 14–16% ở cấp 3 và 0–16% ở cấp 4 trên map dày. GA2 (truncation 20%, đột biến mọi gen σ = 0,05, khởi tạo N(0; 0,5)) đạt 34% và 16%, còn bản bật lai ghép đạt 38% và 24%, ngang CMA-ES 180.000 episode ở cấp 3 (đo cục bộ, 1 seed mỗi run). Kết quả của bản bật lai ghép cũng có nghĩa là chưa có bằng chứng cục bộ rằng lai ghép có hại. Mặt khác, GA2 đổi năm thiết lập cùng lúc, và cùng họ đó với σ = 0,02 chỉ đạt 2% và 0%, với σ = 0,10 đạt 12% và 34%. Thứ tự hai cấp đảo theo σ, đúng dấu hiệu của nhiễu một seed cộng với độ nhạy theo bước đột biến. σ = 0,02 của evorobotpy2 vốn đi kèm rất nhiều thế hệ, nên trong 150 thế hệ nó gần như không đi xa khỏi điểm khởi tạo (suy luận).

| Thiết lập | GA của nhóm | GA theo tài liệu (đề xuất) | Căn cứ |
|---|---|---|---|
| Chọn lọc | tournament k = 3 | truncation 10–20% hoặc chọn theo hạng; nếu giữ tournament thì k ≥ 5 | Such; Nolfi 20/100 |
| Lai ghép | đều, mọi con | không lai ghép, hoặc ≤ 0,1 và hoán đổi nguyên nút ẩn | Nolfi; Such; Floreano 0,1 |
| Đột biến | 10% gen, σ = 0,5 | mọi gen, Gauss σ 0,05–0,1 trên thang Xavier, quét σ, có thể giảm dần | Such; evorobotpy2; GA2 (đo cục bộ) |
| Khởi tạo | N(0,1) | Xavier N(0; 1/fan_in): độ lệch chuẩn 0,26 và 0,35 | Such; evorobotpy2 |
| Elite | 2 | 1–2, đánh giá lại mỗi thế hệ trên map hiện tại | Such; SSS |
| Đánh giá | 8 map ngẫu nhiên | 4–8 map chung cho cả thế hệ, đổi mỗi thế hệ | PLOS ONE 2018; sss.py; Lahouazi |
| Champion | – | top 5–10 chấm lại trên validation 200 map | Such; Pagliuca |
| Đầu vào | – | chuẩn hóa | Pagliuca 2020 |
| Seed | 1 | ≥ 10 seed, Mann–Whitney kèm hiệu chỉnh Holm | Pagliuca 2018, 2020 |

Về công bằng, giữ GA ở cấu hình đã biết là yếu trong khi CMA-ES chạy gần chuẩn sẽ làm lệch phép so NEAT – GA – CMA-ES. Nên báo cả SimpleGA lẫn GA theo tài liệu. Bên cạnh đó cần một ablation tắt lần lượt từng thay đổi của GA2 (lai ghép, truncation, σ, khởi tạo), vì không nguồn nào trả lời câu hỏi này cho mạng 146 trọng số.

## NEAT-NC học thuộc ba bản đồ cố định, nhóm đã có gần hết những gì đáng mượn

Bài thành viên nhóm gửi là "NEAT-NC: NEAT guided Navigation Cells for Robot Path Planning" của Meliani, Slimani và Khoulji, arXiv v1 ngày 16/4/2026. Bản bình duyệt là bài ngắn 4 trang trong GECCO '26 Companion (tr. 553–556), và OpenAlex chưa ghi nhận trích dẫn nào ([arXiv](https://arxiv.org/abs/2604.15076); [OpenAlex](https://api.openalex.org/works/doi:10.1145/3795101.3805301)). "NC" nghĩa là navigation cells. Thay cho cảm biến khoảng cách, đầu vào gồm 21 giá trị: 9 "border cell" (lưới chiếm chỗ 3 × 3 quanh robot), 9 "place cell" của đích, 2 giá trị hướng [sin α, cos α] của góc tới đích và 1 "speed cell" v/v_max. Grid cell có trong abstract nhưng không nằm trong vector đầu vào ([arXiv HTML](https://arxiv.org/html/2604.15076v1)).

**Thuật toán NEAT không đổi.** Code là neat-python 1.1.0 với `DefaultGenome`, `DefaultReproduction`, `DefaultSpeciesSet`, `DefaultStagnation`, `feed_forward = False` và `RecurrentNetwork` ([config-neatPC.txt](https://raw.githubusercontent.com/HHNM/NEAT-NC/main/config/config-neatPC.txt); [main.py](https://raw.githubusercontent.com/HHNM/NEAT-NC/main/main.py)). Cấu hình:

- pop 50, 10 thế hệ;
- conn add 0,5, conn delete 0,2, node add/delete 0,2/0,2; trọng số 0,8/0,5/0,1;
- ngưỡng compatibility 3,0; `max_stagnation` 20, không bao giờ kích hoạt trong 10 thế hệ;
- elitism 4, survival 0,2;
- `fitness_threshold` 10.000, đạt được, nên run có thể dừng sớm.

Fitness là tổng theo từng bước gồm: tiến độ Euclid về đích; phạt −0,05·|ω|; −100 và kết thúc episode khi va chạm; +10 mỗi bước trong một "vùng thấy đích" đặt tay; và 5.000 + 5·(1.000 − t) khi tới đích.

Ba môi trường là ba map 800 × 600 px vẽ tay: một S-maze, một arena có 5 vật cản động và một zig-zag có 2 vật cản động. Start (80, 100) và goal cố định, chỉ pha của vật cản động là ngẫu nhiên. Mỗi genome chạy 1 episode mỗi thế hệ, 30 run không gieo seed, và **không có tách train/test**: success là của cá thể tốt nhất thế hệ cuối trên chính map train ([environment.py](https://raw.githubusercontent.com/HHNM/NEAT-NC/main/src/environment.py); [main.py](https://raw.githubusercontent.com/HHNM/NEAT-NC/main/main.py)). Table 3 báo success 93,33/100/70% cho NEAT-NC, 66,33/47/23,33% cho NEAT thường và 63,33/16,67/6,67% cho PPO (500.000 bước) ([Table 3](https://arxiv.org/html/2604.15076v1)). Kiểm Fisher theo cặp trên số run thành công, giả định n = 30, cho khác biệt rõ ở môi trường 2 và 3. Ở môi trường 1, khác biệt không qua hiệu chỉnh Bonferroni (p khoảng 0,021 và 0,010) (tự tính).

Bằng chứng còn yếu hơn bề ngoài:

- 66,33% và 47% không thể sinh từ 30 run.
- NEAT thường là mạng feed-forward 8 radar, và văn bản không nhắc tới đầu vào đích.
- PPO dùng γ = λ = 1 và không được tinh chỉnh.
- Cột thời gian thực chất là vòng pygame khóa 60 FPS cộng 2 s nghỉ mỗi thế hệ: 10 thế hệ mất khoảng 187 s, khớp 186,89 s ở môi trường 3 (tự tính).
- Đầu vào, kiểu mạng và tham số đổi cùng lúc, không có ablation.
- Repo không có code baseline, không gieo seed, và file kết quả chỉ có dòng tiêu đề ([CSV](https://raw.githubusercontent.com/HHNM/NEAT-NC/main/data/neatNC_results.csv)).
- Theo danh mục trích từ bản HTML (cần đối chiếu lại với PDF), bài không dẫn Butterworth 2019, bài NEAT điều hướng gần nhất.

**Số của NEAT-NC không so được với 16% và 38%.** Một bên là success trên chính map train sau tối đa 500 episode, bên kia là success trên 50 map chưa thấy sau 1,5 triệu episode. Tầm cảm biến cũng khác: 100 px trong arena rộng 800 px (12,5%) so với 3 m trong 10 m (30%).

Phần lớn những gì dùng được từ NEAT-NC nhóm đã có sẵn. Mã hóa [sin α, cos α] đã nằm trong 15 đầu vào, cùng khoảng cách tới đích mà NEAT-NC không có. Tỉ lệ cấu trúc cao đã có trong cấu hình tham chiếu. Recurrent chỉ là một dòng `feed_forward = False` cộng `reset()` mỗi episode. Những phần mới đáng thử, đều rẻ:

- thêm đầu vào tốc độ hoặc hành động ở bước trước, để mạng phản xạ có chút trạng thái;
- một phạt nhỏ |ω| chống quay vòng, cộng thưởng về sớm nếu fitness geodesic hiện tại chưa có thành phần thời gian;
- đánh giá cả thế hệ trên cùng mẫu ngẫu nhiên, như NEAT-NC làm với pha vật cản, nhưng với nhiều map mỗi genome;
- báo Fisher hoặc chi-square cho số run thành công và Kruskal–Wallis hoặc Mann–Whitney cho đại lượng liên tục, như bài đã làm, nhưng trên map held-out.

Ba thứ không nên lấy:

- tiến độ Euclid, vì nó thưởng việc đi vào ngõ cụt hướng về đích, đúng kiểu đánh lừa mà fitness geodesic đã sửa ở cấp 4;
- vùng thưởng đặt tay, vì không chuyển được sang map tự sinh;
- lưới 3 × 3 thay cho 12 tia, vì mất thông tin khoảng cách liên tục.

NEAT-NC không có gì cho novelty search hay curriculum.

## Kết luận

Đặt tài liệu cạnh số đo thì câu hỏi đúng không còn là "16% và 38% có ngang bài báo không", vì bài báo tương đương không tồn tại. Câu hỏi đúng là "phép so NEAT với CMA-ES đã công bằng chưa", và hiện còn ba mắt xích lệch. Thứ nhất, NEAT chạy với 1–3 loài nên chưa phải NEAT theo nghĩa của Stanley. Thứ hai, run NEAT tốt nhất học theo curriculum và chưa từng gặp cấp 4, trong khi CMA-ES học đều. Thứ ba, mỗi bên chỉ có 1 seed, chọn trên 25 map validation. Kể cả khi nắn xong ba mắt xích này, tài liệu vẫn dự đoán CMA-ES thắng trên mạng 146 tham số. Một NEAT đã sửa mà vẫn thua sẽ là kết quả đáng tin hơn một con số cao đạt được trong giao thức lỏng.

Hệ quả thứ hai là nếu NEAT có lợi thế riêng ở cấp 3–4, lợi thế đó không nằm ở việc mọc topology. Genome không nút ẩn của Butterworth đã đạt 90,4%, còn genome của nhóm mọc tới 26–34 node mà vẫn dừng ở 16% cấp 3. Lợi thế, nếu có, sẽ đến từ trí nhớ tường minh kiểu nút GRU hoặc từ novelty, hai thứ mà MLP cố định của CMA-ES không có sẵn. Đây cũng là thí nghiệm tài liệu còn thiếu: chưa bài NEAT nào cùng lúc có map held-out dựng quanh bẫy, nhiều seed và báo theo seed. Một so sánh công bằng giữa NEAT, NEAT-GRU hoặc NEAT + novelty với CMA-ES và GA đã sửa, trên 200 map test với 10 seed, là đóng góp mà một đồ án sinh viên làm được trên một máy.
