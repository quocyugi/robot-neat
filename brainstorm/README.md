# HƯỚNG DẪN TỔ CHỨC FILE BRAINSTORM

## Nguyên tắc cốt lõi
1. **1 chủ đề = 1 file** — không gom nhiều chủ đề vào 1 file dài
2. **Đặt tên theo hành động** — `design_fitness.md` chứ không phải `brainstorm_1.md`
3. **Mỗi file có header chuẩn** — ghi ngày, câu hỏi gốc, kết luận rút ra
4. **Tách rõ: ý tưởng thô vs quyết định cuối cùng** — dùng tag `[DECIDED]` và `[OPEN]`

---

## Cấu trúc thư mục đề xuất

```
Slide TTTH/
├── overview.md                          # Bản tổng quan project (đã có)
├── SCRIPT_THUYET_TRINH_NEAT_ROBOT.md    # Script thuyết trình (đã có)
│
├── brainstorm/                          # ★ THƯ MỤC CHÍNH CHỨA KẾT QUẢ BRAINSTORM
│   │
│   ├── _decisions.md                    # ★ FILE QUAN TRỌNG NHẤT: tổng hợp mọi quyết định
│   │
│   ├── 01_env/                          # Nhóm 1: Môi trường & Mô phỏng
│   │   ├── observation_space.md         # Thiết kế 15D input, review từ LLM
│   │   ├── ray_casting.md              # Thuật toán ray-casting vector hóa
│   │   ├── map_generator.md            # Pipeline sinh bản đồ, tortuosity thresholds
│   │   └── gymnasium_api.md            # Thiết kế RobotNavEnv class
│   │
│   ├── 02_neat/                         # Nhóm 2: NEAT Engine & Config
│   │   ├── config_tuning.md            # Phân tích tham số neat-python
│   │   ├── fitness_design.md           # Review hàm fitness, edge cases, chỉnh w_i
│   │   ├── speciation_analysis.md      # Phân tích cơ chế phân loài
│   │   └── recurrent_vs_ff.md          # So sánh Feedforward vs Recurrent NEAT
│   │
│   ├── 03_novelty/                      # Nhóm 3: Novelty Search & Deceptive Fitness
│   │   ├── behavior_descriptor.md      # Thiết kế b(x), trajectory sampling
│   │   ├── archive_management.md       # KD-Tree, giới hạn archive, threshold
│   │   ├── adaptive_lambda.md          # Chiến lược hybrid fitness-novelty
│   │   └── utrap_analysis.md           # Phân tích bẫy chữ U, thí nghiệm
│   │
│   ├── 04_experiments/                  # Nhóm 4: Thực nghiệm & Thống kê
│   │   ├── ablation_design.md          # Bảng ablation study chi tiết
│   │   ├── baseline_fairness.md        # Lập luận so sánh công bằng
│   │   ├── statistical_methods.md      # Mann-Whitney U, Bonferroni, effect size
│   │   └── failure_analysis.md         # Cách trình bày khi kết quả không đạt
│   │
│   └── 05_defense/                      # Nhóm 5: Phản biện & Bảo vệ
│       ├── qa_preparation.md           # Câu hỏi giảng viên + câu trả lời
│       ├── related_work.md             # Tổng quan tài liệu liên quan
│       └── scope_negotiation.md        # Lập luận về phạm vi 2D vs 3D
│
└── src/                                 # Code thực tế (khi bắt tay code)
    └── ...
```

---

## Format chuẩn cho mỗi file brainstorm

Mỗi file nên có cấu trúc sau:

```markdown
# [Tên chủ đề]
> **Ngày**: 2026-10-02
> **Trạng thái**: [OPEN] / [DECIDED] / [NEEDS_CODE]
> **Liên quan**: overview.md phần X.X

## Câu hỏi gốc tôi đã hỏi LLM
> (Copy nguyên văn prompt đã gửi)

## Tóm tắt câu trả lời (3-5 dòng)
- Điểm chính 1
- Điểm chính 2
- Điểm chính 3

## Chi tiết đầy đủ
(Paste toàn bộ câu trả lời của LLM ở đây)

## Quyết định của tôi
- [x] Chọn phương án A vì ...
- [ ] ~~Phương án B bị loại vì ...~~

## TODO tiếp theo
- [ ] Code hàm XYZ theo thiết kế này
- [ ] Hỏi thêm LLM về vấn đề phụ ...
```

---

## File _decisions.md — Bản ghi quyết định tổng hợp

Đây là file quan trọng nhất, cập nhật liên tục mỗi khi brainstorm xong 1 chủ đề:

```markdown
# QUYẾT ĐỊNH THIẾT KẾ PROJECT

## Môi trường
| # | Vấn đề | Quyết định | Lý do | File tham khảo |
|---|--------|-----------|-------|----------------|
| 1 | Số tia LiDAR | 12 tia | Cân bằng giữa thông tin và tốc độ | 01_env/observation_space.md |
| 2 | Ray-casting | NumPy vectorized | > 5000 FPS trên CPU | 01_env/ray_casting.md |

## NEAT
| # | Vấn đề | Quyết định | Lý do | File tham khảo |
|---|--------|-----------|-------|----------------|
| 3 | Population size | 150 | Theo paper gốc | 02_neat/config_tuning.md |
| 4 | FF vs Recurrent | Cả hai, so sánh | Trả lời Q2 | 02_neat/recurrent_vs_ff.md |

## Novelty Search
...

## Thực nghiệm
...
```

---

## Quy trình làm việc thực tế

```
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │  Mở chat LLM │────>│  Hỏi 1 chủ  │────>│  Lưu file .md│
 │  + đính kèm  │     │  đề cụ thể   │     │  vào đúng    │
 │  overview.md  │     │              │     │  thư mục     │
 └──────────────┘     └──────────────┘     └──────┬───────┘
                                                   │
                                                   v
                                            ┌──────────────┐
                                            │ Cập nhật      │
                                            │ _decisions.md │
                                            │ + overview.md │
                                            └──────┬───────┘
                                                   │
                                                   v
                                            ┌──────────────┐
                                            │  Bắt tay code │
                                            │  trong src/   │
                                            └──────────────┘
```

**Vòng lặp**: Brainstorm → Ghi file → Quyết định → Code → Gặp vấn đề → Brainstorm tiếp

---

## Mẹo thực tế

1. **Khi LLM cho code**: Lưu code snippet vào file brainstorm, nhưng khi quyết định dùng
   thì copy sang `src/` và chỉnh sửa — KHÔNG code trực tiếp trong thư mục brainstorm.

2. **Khi LLM cho nhiều phương án**: Ghi hết vào file, đánh dấu `[OPTION A]`, `[OPTION B]`,
   sau đó ghi rõ `[DECIDED]: Option A` khi đã chọn.

3. **Khi brainstorm nhiều lần cùng 1 chủ đề**: Thêm ngày vào cuối file (append),
   KHÔNG tạo file mới `fitness_design_v2.md` — dễ loạn.

4. **Mỗi tuần**: Đọc lại `_decisions.md` 1 lần, cập nhật trạng thái, xóa TODO đã xong.
