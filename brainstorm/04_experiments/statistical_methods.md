# QUY CHUẨN THỐNG KÊ KHOA HỌC: MANN-WHITNEY U TEST, 10 SEEDS VÀ MEDIAN/IQR
> **Ngày**: 2026-10-02  
> **Trạng thái**: [DECIDED]  
> **Liên quan**: overview.md phần 6.3  

---

## 1. Câu hỏi gốc tôi đã hỏi LLM
> "Quy chuẩn thống kê khoa học trong đánh giá giải thuật tiến hóa: tại sao dùng Median/IQR thay cho Mean/Std, thiết kế 10 seeds độc lập, kiểm định phi tham số Mann-Whitney U Test và hiệu chỉnh Bonferroni."

---

## 2. Tóm tắt kết luận rút ra (Key Takeaways)
- **Chuẩn hóa Báo cáo Median [IQR]**: Vì phân phối kết quả của giải thuật di truyền thường bị lệch (skewed) hoặc có hai đỉnh (bimodal), việc dùng Mean/Std vi phạm giả định chuẩn. Toàn bộ bảng kết quả định lượng bắt buộc sử dụng Trung vị (Median) và Khoảng tứ phân vị (IQR: $Q_3 - Q_1$).
- **Giao thức 10 Seeds Độc lập**: Mỗi cấu hình chạy lặp lại đúng 10 lần với 10 random seeds cố định (`42, 101, 7, 13, 23, 88, 99, 123, 555, 777`). Mọi đường cong tiến hóa đều vẽ dải bóng mờ (Shaded ribbon IQR) bao quanh đường trung vị.
- **Kiểm định Phi tham số Mann-Whitney U**: Đánh giá sự khác biệt hiệu năng giữa mô hình đề xuất M6 và các baselines. Mức ý nghĩa tiêu chuẩn $p < 0.05$ (đánh dấu $*p < 0.05, **p < 0.01$).
- **Hiệu chỉnh Bonferroni khi so sánh chéo**: Khi so sánh M6 với đồng thời $m = 4$ baselines, ngưỡng ý nghĩa được siết chặt về $\alpha' = 0.05 / 4 = 0.0125$ để triệt tiêu hoàn toàn sai lầm loại I (False Positive).

---

## 3. Mã Nguồn Kiểm Định Thống Kê & Xuất Bảng (Scipy)

```python
import numpy as np
from scipy import stats

def compute_run_statistics(scores: list) -> dict:
    """Tính Median và IQR cho 1 danh sách 10 kết quả runs"""
    arr = np.array(scores)
    median = np.median(arr)
    q25 = np.percentile(arr, 25)
    q75 = np.percentile(arr, 75)
    iqr = q75 - q25
    return {"median": median, "q25": q25, "q75": q75, "iqr": iqr}

def compare_models_statistically(model_scores: list, baseline_scores: list, alpha: float = 0.05, num_comparisons: int = 1):
    """
    Kiểm định Mann-Whitney U Test kèm hiệu chỉnh Bonferroni
    """
    bonferroni_alpha = alpha / num_comparisons
    stat, p_value = stats.mannwhitneyu(model_scores, baseline_scores, alternative='greater')

    is_significant = p_value < bonferroni_alpha
    stars = "**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns")

    return {
        "U_statistic": stat,
        "p_value": p_value,
        "is_significant": is_significant,
        "marker": stars
    }
```

---

## 4. Quyết định của tôi
- [x] **Chọn Phương án A cho Định dạng Báo cáo**: Báo cáo Median [IQR] kèm biểu đồ Boxplot. Đưa chất lượng báo cáo đồ án lên tiệm cận một bài báo khoa học tại các hội thảo quốc tế như GECCO hoặc IROS.
- [x] **Chọn Phương án A cho Kiểm định Thống kê**: Sử dụng Mann-Whitney U Test kết hợp Bonferroni Correction. Đảm bảo mọi kết luận về sự vượt trội của mô hình đều có bảo chứng toán học vững chắc.

---

## 5. TODO tiếp theo
- [ ] Brainstorm tiếp chủ đề `04_experiments/failure_analysis.md`: Phân tích thất bại (Failure Analysis) và các tình huống biên cực đoan.
