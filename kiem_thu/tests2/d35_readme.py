"""Write README.md of the delivered folder sua_cau_hinh_NEAT_GA from results2/fix_summary.json.

Usage: python d35_readme.py <fix_summary.json> <out_README.md>
"""
import json
import sys

S = json.load(open(sys.argv[1], encoding="utf-8"))
G = S["groups"]


def p(x):
    return "—" if x is None else f"{round(100 * x):d}%"


def cell(g, k):
    v = G.get(g, {}).get(k)
    if v is None:
        return "—"
    if len(v["per_seed"]) > 1:
        return f"{p(v['SR'])} ({p(v['min'])}–{p(v['max'])})"
    return p(v["SR"])


def row(g, label):
    if g not in G:
        return None
    n = len(G[g]["runs"])
    return f"| {label} | {n} | " + " | ".join(cell(g, k) for k in ("test_L1", "test_L2", "test_L3", "test_L4", "test_barn")) + " |"


xo, xf = S.get("xor_team_old") or {}, S.get("xor") or {}
mz, mzo = S.get("maze_hard_ns", {}), S.get("maze_hard_ns_earlier", {})
lines = []
w = lines.append
w("# Cấu hình NEAT và GA đã sửa (nhóm 3, IT4906)")
w("")
w("Thư mục này chứa bản sửa cấu hình NEAT và GA của repo `hoangthcslt/Project_IT4906`. Repo gốc **không bị sửa**: các file ở đây đặt đúng đường dẫn như trong repo để chép đè, kèm diff ở `diff/`. Phân tích đầy đủ nằm ở mục 8.13–8.14 của `Bao_cao_pipeline_NEAT_robot.docx`.")
w("")
w("## Có gì trong thư mục")
w("")
w("| File | Thay cho | Nội dung |")
w("|---|---|---|")
w("| `configs/neat_recurrent.ini`, `configs/neat_feedforward.ini` | file cùng tên trong repo | Cấu hình NEAT đã sửa; mỗi dòng đổi có chú thích `SUA` ngay phía trên (viết không dấu, xem mục Giới hạn) |")
w("| `src/baselines/fixed_ga.py` | file cùng tên | `SimpleGA` với mặc định mới (cắt ngọn 20%, đột biến mọi gen σ 0,05, khởi tạo N(0; 0,5), 1 elite); giữ nguyên giao diện lớp, tham số cũ vẫn chọn được |")
w("| `src/neat_engine/neat_fixes.py` | (mới) | `unify_initial_hidden_keys()`, `TargetSpeciesReporter`, `ValidationChampion` |")
w("| `diff/*.diff` | | Diff thống nhất so với bản gốc |")
w("")
w("## Cách áp dụng")
w("")
w("1. Ghim `neat-python==2.0.0` trong `requirements.txt`.")
w("2. Chép `configs/` và `src/` của thư mục này đè lên repo (hoặc áp các file trong `diff/`).")
w("3. Trong `NEATRunner.run()` (`src/neat_engine/runner.py`), ngay sau `population = neat.Population(self.neat_config)`, thêm:")
w("")
w("```python")
w("from src.neat_engine.neat_fixes import unify_initial_hidden_keys, ValidationChampion")
w("unify_initial_hidden_keys(population, self.neat_config)   # không làm gì khi num_hidden = 0")
w("```")
w("")
w("4. Thay `winner` của `population.run()` bằng genome có SR trung bình cao nhất trên bản đồ val (lớp `ValidationChampion`, gọi `update()` trong `_eval_genomes` sau khi gán fitness), và lưu genome đó.")
w("5. Đổi mặc định `--maps-per-eval` trong `scripts/train_neat.py` từ 4 lên 8, dùng cùng 8 bản đồ cho cả quần thể trong một thế hệ.")
w("")
w("Các sửa đổi ngoài cấu hình đã nêu trong báo cáo (tiến độ đo theo đường đi thật L26, tường cấp 3 có bề dày L27, cùng một lịch học cho mọi thuật toán) vẫn cần làm riêng.")
w("")
w("## Đã sửa gì")
w("")
w("| Tham số | Gốc | Sửa | Vì sao |")
w("|---|---|---|---|")
w("| `no_fitness_termination` | False | True | Lỗi L01: dừng ở thế hệ 0–1 khi một genome đạt 16 điểm trên phòng trống |")
w("| `conn_add` / `conn_delete` / `node_add` / `node_delete` | 0,05 / 0,01 / 0,03 / 0,01 | 0,5 / 0,1 / 0,1 / 0,1 | Lỗi L29: với giá trị cũ XOR chỉ giải " + str(xo.get("team_settings_sigmoid", {}).get("solved", 87)) + "/100 lần |")
w("| `single_structural_mutation` | không khai báo | True | neat-python 2.0.0 đọc khóa vắng mặt thành chuỗi `'false'`, chuỗi khác rỗng là True (L31); ghi rõ để hành vi không đổi |")
w("| `max_stagnation` | 15 | 20 | Fitness đo trên bản đồ mới mỗi thế hệ nên ồn |")
w("| `weight/bias_mutate_power`; `weight/bias_replace_rate` | 0,5; 0,1 | 0,1; 0,01 | L32: bước 0,5 quá thô để tinh chỉnh |")
w("| GA: chọn lọc | tournament k = 3 | cắt ngọn 20% | Cường độ chọn lọc 0,85 so với 1,40 |")
w("| GA: đột biến | 10% gen, σ 0,5 | mọi gen, σ 0,05 | σ 0,5 bằng nửa thang trọng số |")
w("| GA: khởi tạo; elite | N(0, 1); 2 | N(0; 0,5); 1 | N(0, 1) làm tanh bão hòa |")
w("")
w("## Kết quả kiểm thử")
w("")
w("Kiểm thử chạy **chính các file trong thư mục này**: bộ huấn luyện nhanh của báo cáo đọc các file `.ini`, nạp `neat_fixes.py` và `fixed_ga.py` từ đây. Điều kiện chung, trừ chỗ ghi khác trong tên dòng: bản đồ cấp 3 tường dày 0,20 m, fitness geodesic, học đều cả bốn cấp với 8 bản đồ mỗi thế hệ (2 mỗi cấp), 180.000 lượt đánh giá, champion chọn trên val, 50 bản đồ test mỗi cấp. Nhiều seed: trung bình (thấp nhất–cao nhất).")
w("")
w("| Cấu hình | Seed | Cấp 1 | Cấp 2 | Cấp 3 | Cấp 4 | BARN |")
w("|---|---|---|---|---|---|---|")
for g, label in (("neat_team", "NEAT, cấu hình gốc"), ("neat_fixB_rnn", "NEAT đã sửa, hồi quy"), ("neat_fixB_ff", "NEAT đã sửa, truyền thẳng"),
                 ("neat_fixB_rnn_1M5", "NEAT đã sửa, hồi quy, 1,5 triệu lượt"), ("neat_old_1M5", "NEAT tốt nhất trước đây, 1,5 triệu lượt, curriculum"),
                 ("ga_team", "GA, cấu hình gốc"), ("ga_fixed", "GA đã sửa"), ("cma", "CMA-ES (tham chiếu)")):
    r = row(g, label)
    if r:
        w(r)
w("")
xt = xo.get("team_settings_sigmoid", {})
xn = (S.get("xor_final") or xf).get("team_settings_sigmoid", {})
gm = lambda d: "—" if d.get("gens_mean") is None else f"{d['gens_mean']:.0f}"
w(f"- XOR, 100 seed: cấu hình gốc {xt.get('solved', '—')}/100, bản sửa {xn.get('solved', '—')}/100 (trung bình {gm(xn)} thế hệ; cấu hình mẫu của thư viện {gm(xf.get('docs_settings', {}))} thế hệ).")
mzf = mz.get("final") or mz.get("fixed") or {}
if mz:
    w(f"- Mê cung hard của Lehman–Stanley, novelty search, quần thể 250: cấu hình gốc giữ nguyên {mz.get('teamraw', {}).get('solved', '—')}/{mz.get('teamraw', {}).get('runs', '—')} lần giải, bản sửa {mzf.get('solved', '—')}/{mzf.get('runs', '—')}; cấu hình tham chiếu Packt {mzo.get('ref', {}).get('solved', '—')}/{mzo.get('ref', {}).get('runs', '—')}.")
ind = [(lbl, G[k]["test_L5"]["SR"]) for k, lbl in (("in_neat_team", "NEAT gốc"), ("in_neat_fixB_rnn", "NEAT sửa (hồi quy)"), ("in_neat_fixB_ff", "NEAT sửa (truyền thẳng)"),
                                                   ("in_neat_fixB_rnn_750k", "NEAT sửa (hồi quy, 750.000 lượt)"), ("in_ga_fixed_like", "GA sửa"), ("in_cma", "CMA-ES")) if k in G and "test_L5" in G[k]]
pl = (S.get("indoor_planners") or {}).get("L5", {})
if ind:
    w("- Bản đồ trong nhà kiểu McGuire/Butterworth (cấp 5, 100 bản đồ test, học và test trên cùng họ bản đồ, 180.000 lượt nếu không ghi khác, 1 seed): " + "; ".join(f"{a} {p(b)}" for a, b in ind)
      + (f". Bug đạt {p(pl['bug_500']['SR'])} trong 500 bước và {p(pl['bug_1000']['SR'])} trong 1.000 bước." if pl else "."))
w("")
w("## Giới hạn")
w("")
w("- Mỗi cấu hình chỉ có 1–3 seed; khác biệt dưới khoảng 15 điểm phần trăm chưa chắc là thật.")
w("- Kiểm thử dùng bộ mô phỏng nhanh của báo cáo (khớp `RobotNavEnv` trên mọi lượt đã so), chưa chạy toàn bộ bằng `runner.py` của nhóm.")
w("- `TargetSpeciesReporter` có trong `neat_fixes.py` nhưng không bật mặc định: neat-python giữ đại diện cho mọi loài cũ nên số loài chỉ giảm khi loài bị loại vì trì trệ; xem mục 8.13.")
w("- Chú thích trong file `.ini` viết không dấu vì neat-python 2.0.0 mở file bằng mã hóa mặc định của Windows (cp1252 hoặc cp1258), và chữ có dấu gây `UnicodeDecodeError` khi không bật chế độ UTF-8. "
  "Bản giao ngày 11/11/2026 còn chú thích có dấu nên không nạp được khi chạy `python` bình thường; bản ngày 12/11/2026 đã sửa, các giá trị cấu hình không đổi. "
  "Nếu tự thêm chú thích tiếng Việt, hãy viết không dấu hoặc chạy với `python -X utf8`.")
open(sys.argv[2], "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("written", sys.argv[2])
