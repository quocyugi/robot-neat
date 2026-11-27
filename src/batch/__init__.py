"""
Đánh giá theo lô (tùy chọn): NumPy trên CPU, CuPy trên GPU
===========================================================
Chạy nhiều lượt cùng lúc theo kiểu lockstep thay cho từng lượt một (src/neat_engine/parallel.py).
Ở chế độ exact (mặc định) kết quả từng lượt trùng từng bit với run_episode, nên lịch sử tiến hóa,
champion và kết quả test không đổi; chỉ thời gian chạy đổi.

Chọn bằng parallel.make_evaluator(workers, backend) hoặc tham số --backend của các script:
  "cpu"   (mặc định) ParallelEvaluator như cũ, không nạp gì của gói này;
  "batch" BatchEvaluator(device="cpu"): NumPy, workers > 1 thì chia lô cho Pool;
  "gpu"   BatchEvaluator(device="gpu"): LiDAR + va chạm trên GPU (CuPy, kernel biên dịch
          --fmad=false), phần còn lại ở host; vẫn exact.
Chế độ không exact (hàm siêu việt trên GPU) chỉ bật tường minh (--inexact), không dùng cho benchmark.

Mô-đun:
  backend.py   lớp xp (NumPy / CuPy), hàm siêu việt exact / device, import CuPy qua đường dẫn ASCII
  numerics.py  tự kiểm số học của chế độ exact lúc khởi tạo
  policies.py  mạng NEAT (bảng đệm theo genome, tổng Neumaier, math.tanh), FixedMLP, APF theo lô
  sim.py       vòng lockstep, LiDAR tia / hình quạt và va chạm theo bucket số đoạn
  gpu.py       kernel CUDA LiDAR + va chạm (biến thể lai)
  evaluator.py BatchEvaluator (map / close / with), Pool, quay về eval_task khi không hỗ trợ
"""

from src.batch.backend import GpuUnavailable
from src.batch.evaluator import BatchEvaluator, NumericsMismatch

__all__ = ["BatchEvaluator", "NumericsMismatch", "GpuUnavailable"]
