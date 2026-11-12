"""Insert (or replace) the "Cập nhật sau thí nghiệm" block at the top of the research report, with numbers taken from fix_summary.json.

Usage: python d36_md_update.py <fix_summary.json> <report.md> <date_text>
The block sits between the title line and the first paragraph, fenced by HTML comments so a rerun replaces it.
"""
import json
import re
import sys

S = json.load(open(sys.argv[1], encoding="utf-8"))
md_path, date_text = sys.argv[2], sys.argv[3]
G = S["groups"]


def p(x):
    return "—" if x is None else f"{round(100 * x):d}%"


def sr(g, k):
    v = G.get(g, {}).get(k)
    return None if v is None else v["SR"]


def n(g):
    return len(G.get(g, {}).get("runs", []))


def pair(g):
    return f"{p(sr(g, 'test_L3'))} / {p(sr(g, 'test_L4'))}"


lines = []
w = lines.append
w("<!-- cap-nhat-sau-thi-nghiem:start -->")
w(f"> **Cập nhật sau thí nghiệm ({date_text}).** Các ý dưới đây đã được kiểm bằng thí nghiệm trên máy của nhóm (mục 8.13–8.14 của `Bao_cao_pipeline_NEAT_robot.docx`, file giao `sua_cau_hinh_NEAT_GA/`). "
  "Chúng sửa một số dự đoán của báo cáo này, nhất là về phân loài và survival. Điều kiện chung: bản đồ tường dày, fitness geodesic, học đều cả bốn cấp, 8 bản đồ mỗi thế hệ, champion chọn trên val, 50 bản đồ test mỗi cấp. Số viết theo dạng cấp 3 / cấp 4.")
w(">")
if "neat_fixB_rnn" in G and "neat_team" in G:
    w(f"> - **Đòn bẩy lớn nhất là bước đột biến trọng số, không phải phân loài.** Giảm `weight_mutate_power` 0,5 → 0,1 và `weight_replace_rate` 0,1 → 0,01 (bias cũng vậy), cùng bỏ dừng sớm, tăng đột biến cấu trúc và stagnation 20, đưa NEAT từ "
      f"{pair('neat_team')} ({n('neat_team')} seed, cấu hình gốc) lên {pair('neat_fixB_rnn')} ({n('neat_fixB_rnn')} seed) ở 180.000 lượt. Chỉ sửa dừng sớm, đột biến cấu trúc và stagnation thì được {pair('neat_fixA_rnn')} (1 seed).")
if "ga_fixed" in G and "cma" in G:
    w(f"> - **Sau khi sửa, NEAT cùng dải với GA và CMA-ES ở cấp 3:** GA đã sửa (cắt ngọn 20%, σ 0,05 trên mọi gen, khởi tạo N(0; 0,5), 1 elite) {pair('ga_fixed')} ({n('ga_fixed')} seed), "
      f"GA gốc {pair('ga_team')} ({n('ga_team')} seed), CMA-ES {pair('cma')} ({n('cma')} seed). Cấp 4 CMA-ES vẫn nhỉnh hơn; với 3 seed, chênh dưới khoảng 10–15 điểm chưa tách được khỏi nhiễu.")
neg = []
if "neat_fixC_ts8" in G:
    neg.append(f"giữ 8 loài bằng ngưỡng động {pair('neat_fixC_ts8')} ({n('neat_fixC_ts8')} seed)")
if "neat_fixD_sv05" in G:
    neg.append(f"survival 0,5 như Butterworth {pair('neat_fixD_sv05')} ({n('neat_fixD_sv05')} seed)")
if neg:
    w("> - **Hai đề xuất lấy từ tài liệu không giúp trên bài này:** " + "; ".join(neg) + f", đều kém bản đã sửa ({pair('neat_fixB_rnn')}). neat-python 2.0.0 không bao giờ gộp loài, nên nâng ngưỡng không giảm được số loài.")
if "neat_fixB_rnn_1M5" in G:
    l3 = G["neat_fixB_rnn_1M5"]["test_L3"]
    w(f"> - **Ở cùng ngân sách 1,5 triệu lượt**, NEAT đã sửa đạt {pair('neat_fixB_rnn_1M5')} (1 seed), so với 16% / 38% của run cũ (cấu hình Packt, học curriculum, chưa từng gặp cấp 4) và {pair('cma_1M')} của CMA-ES 1 triệu lượt. "
      f"Thêm ngân sách nâng cấp 4 nhưng không nâng cấp 3 (trung bình {p(sr('neat_fixB_rnn', 'test_L3'))} ở 180.000 lượt); ở cấp 3, {p(l3.get('timeout'))} số lượt hết giờ, tức robot quẩn trong ngõ cụt chứ không đâm tường.")
ind = [(lbl, sr(g, "test_L5")) for g, lbl in (("in_neat_team", "NEAT gốc"), ("in_neat_fixB_rnn", "NEAT đã sửa"), ("in_neat_fixB_ff", "NEAT đã sửa truyền thẳng"),
                                              ("in_neat_fixB_rnn_750k", "NEAT đã sửa 750.000 lượt"), ("in_ga_fixed_like", "GA đã sửa"), ("in_cma", "CMA-ES")) if sr(g, "test_L5") is not None]
pl = (S.get("indoor_planners") or {}).get("L5", {})
if ind:
    w("> - **Trên họ bản đồ McGuire/Butterworth dựng lại cho robot của nhóm** (cấp 5, 100 bản đồ test, 180.000 lượt nếu không ghi khác, 1 seed mỗi cấu hình): "
      + "; ".join(f"{a} {p(b)}" for a, b in ind) + "."
      + (f" Bug đạt {p(pl['bug_500']['SR'])} trong 500 bước, gần mức khoảng 90% của Bug có nhớ trong McGuire." if pl else "")
      + " Bài Butterworth báo 90,4% với genome tốt nhất của 20 run chọn ngay trên test, lượt 300 s và robot nhỏ hơn nhiều so với khe cửa.")
mz = S.get("maze_hard_ns", {})
xo, xf = (S.get("xor_team_old") or {}).get("team_settings_sigmoid", {}), (S.get("xor_final") or {}).get("team_settings_sigmoid", {})
if mz.get("final") and xf:
    w(f"> - **Bài chuẩn:** cấu hình NEAT giao cho nhóm giải XOR {xf['solved']}/100 lần (gốc {xo.get('solved', '—')}/100) và mê cung hard của Lehman–Stanley bằng novelty {mz['final']['solved']}/{mz['final']['runs']} lần (gốc {mz.get('teamraw', {}).get('solved', '—')}/{mz.get('teamraw', {}).get('runs', '—')}).")
w(">")
w("> **Kết luận sau thí nghiệm:** 16% / 38% không phải giới hạn của NEAT trên bài này. Mức đó chủ yếu do cấu hình (bước đột biến trọng số quá thô) và giao thức (curriculum chưa từng gặp cấp 4). "
  "Sau khi sửa, NEAT không còn dấu hiệu cài đặt sai; câu hỏi NEAT có hơn mạng cố định hay không cần ≥ 10 seed để trả lời.")
w("<!-- cap-nhat-sau-thi-nghiem:end -->")
block = "\n".join(lines)

md = open(md_path, encoding="utf-8").read()
if "<!-- cap-nhat-sau-thi-nghiem:start -->" in md:
    md = re.sub(r"<!-- cap-nhat-sau-thi-nghiem:start -->.*?<!-- cap-nhat-sau-thi-nghiem:end -->", lambda m: block, md, flags=re.S)
else:
    title, rest = md.split("\n", 1)
    md = title + "\n\n" + block + "\n" + rest
open(md_path, "w", encoding="utf-8").write(md)
print(block)
