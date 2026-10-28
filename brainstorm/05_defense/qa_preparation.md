# NGÂN HÀNG CÂU HỎI PHẢN BIỆN (Q&A BANK) & KỊCH BẢN BẢO VỆ XUẤT SẮC
> **Ngày**: 2026-10-02  
> **Trạng thái**: [DECIDED]  
> **Liên quan**: overview.md toàn bộ các phần  

---

## 1. Câu hỏi gốc tôi đã hỏi LLM
> "Xây dựng ngân hàng câu hỏi phản biện của hội đồng chấm điểm, kịch bản ứng khẩu sắc bén, cấu trúc các slide phụ Backup Q&A, và chiến lược định hướng Future Work."

---

## 2. Tóm tắt kết luận rút ra (Key Takeaways)
- **Chuẩn bị 5 Slide Backup Q&A chuyên nghiệp**: Thiết kế sẵn 5 slide phụ ở cuối file thuyết trình. Khi giám khảo hỏi trúng chủ đề, chuyển ngay tới slide minh họa tương ứng, tạo hiệu ứng áp đảo về sự chuẩn bị chu đáo.
- **Làm chủ 5 câu hỏi học thuật kinh điển**:
  1. *Cơ chế Hybrid Novelty $\lambda$*: Giải thích tại sao robot không đi lạc vào các góc mới lạ mà vẫn về đích.
  2. *Lý do chọn $k_{\text{nn}} = 15$*: Cân bằng giữa triệt tiêu nhiễu cục bộ và duy trì độ nhạy không gian.
  3. *Khả năng tổng quát hóa (Generalization)*: Quy trình kiểm định Zero-Shot trên 200 bản đồ Test và 50 bản đồ thực tế BARN.
  4. *Cơ chế Innovation Number*: Xếp thẳng hàng gen, loại bỏ hoàn toàn hiện tượng Competing Conventions.
  5. *Vật cản di động (Dynamic Obstacles)*: Mạng Recurrent NEAT có bộ nhớ sẵn sàng nhận diện vận tốc tương đối.
- **Định hướng phát triển tương lai (Future Work)**: Mở rộng sang môi trường động và tích hợp học tăng cường lai (NeuroEvolution + RL fine-tuning).

---

## 3. Ngân Hàng 5 Câu Hỏi Phản Biện Trọng Tâm & Kịch Bản Trả Lời Mẫu

### Câu 1: Về cơ chế Novelty Search và khả năng cán đích
* **Giảng viên hỏi**: *"Novelty Search chỉ đi tìm cái mới lạ, lỡ robot đi tìm một góc tường kỳ quái xa xôi rồi đứng ở đó thì sao? Nó có thực sự tìm về đích không?"*
* **Kịch bản trả lời**:
  > *"Thưa Thầy/Cô, đó là nhược điểm của Pure Novelty Search. Để khắc phục, đề tài đề xuất **Adaptive Hybrid $\lambda$**: Trọng số cơ sở $\lambda = 0.6$ ưu tiên dẫn hướng về đích. Chỉ khi phát hiện quần thể bị tắc nghẽn (Stagnation $\ge 5$ thế hệ do bẫy chữ U), hệ thống mới tự động hạ $\lambda = 0.1$ để giải phóng sức mạnh phá bẫy. Khi robot đã thoát ra ngoài miệng bẫy, $\lambda$ tự động tăng trở lại để hút robot vào Goal."*

### Câu 2: Về siêu tham số $k_{\text{nn}} = 15$
* **Giảng viên hỏi**: *"Tại sao lại chọn $k = 15$? Nếu chọn $k = 1$ hay $k = 50$ thì sao?"*
* **Kịch bản trả lời**:
  > *"Thưa Thầy/Cô, $k$ quá nhỏ ($k=1, 3$) sẽ khiến thuật toán bị quá nhạy với nhiễu ngẫu nhiên (chỉ 1 robot trượt bánh cũng bị coi là mới lạ). $k$ quá lớn ($k=50$) sẽ làm phẳng các hốc hẹp địa hình. $k=15$ là giá trị tối ưu thực nghiệm theo công trình gốc của Lehman & Stanley (2011), đủ lớn để triệt tiêu nhiễu và đủ nhạy để nhận biết vùng thưa thớt."*

### Câu 3: Về tính tổng quát hóa (Generalization)
* **Giảng viên hỏi**: *"Robot học trên bản đồ tự sinh, có gì đảm bảo nó không học vẹt và chạy được trên môi trường thực tế?"*
* **Kịch bản trả lời**:
  > *"Thưa Thầy/Cô, mô hình được kiểm chứng nghiêm ngặt qua 3 tầng: (1) Quá trình train sinh mới bản đồ ngẫu nhiên liên tục; (2) Đánh giá Zero-Shot trên 200 bản đồ Test độc lập hoàn toàn; (3) Kiểm thử mở rộng trên 50 bản đồ thực tế trích xuất từ tập dữ liệu chuẩn BARN của Đại học Texas Austin."*

### Câu 4: Về cơ chế số đổi mới (Innovation Numbers) trong NEAT
* **Giảng viên hỏi**: *"Cơ chế Innovation Number giải quyết vấn đề Competing Conventions như thế nào?"*
* **Kịch bản trả lời**:
  > *"Thưa Thầy/Cô, mỗi gen kết nối mới được cấp một số thứ tự toàn cục (Global Innovation Number). Khi lai ghép giữa 2 mạng có cấu trúc khác nhau, thuật toán chỉ cần căn thẳng hàng các gen có cùng số đổi mới (Matching Genes) để trao đổi chéo, còn các gen dư thừa (Excess) hoặc rời rạc (Disjoint) được thừa hưởng từ cha mẹ có fitness cao hơn, giúp bảo toàn tính toàn vẹn cấu trúc."*

### Câu 5: Về vật cản di động (Dynamic Obstacles) & Future Work
* **Giảng viên hỏi**: *"Nếu trong môi trường có người đi lại hoặc xe khác di chuyển thì robot có né được không?"*
* **Kịch bản trả lời**:
  > *"Thưa Thầy/Cô, nhờ kiến trúc Recurrent NEAT có bộ nhớ trong (Hidden State), robot có khả năng cảm nhận chuỗi biến thiên khoảng cách LiDAR qua thời gian để suy đoán vận tốc tương đối. Đây chính là một trong hai hướng phát triển trọng tâm (Future Work) của nhóm em."*

---

## 4. Quyết định của tôi
- [x] **Chọn Phương án A cho Backup Slides**: Thiết kế sẵn 5 Slide Backup Q&A ở cuối bài thuyết trình. Sẵn sàng "nhảy slide" phản biện ngay lập tức khi được hỏi.
- [x] **Chọn Phương án A cho Kế hoạch tương lai**: Đưa ra 2 hướng Future Work rõ ràng (Vật cản di động + Hybrid RL Fine-tuning).

---

## 5. Hoàn tất Toàn bộ Giai đoạn Brainstorm
- [x] Đã hoàn thành 100% tài liệu tại 5 thư mục con của `brainstorm/`.
- [x] Đã chốt 35 quyết định kỹ thuật trong `brainstorm/_decisions.md`.
- [ ] Bắt tay vào giai đoạn xây dựng mã nguồn trong thư mục `src/`.
