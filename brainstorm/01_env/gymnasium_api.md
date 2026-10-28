# THIẾT KẾ LỚP ROBOTNAVENV CHUẨN GYMNASIUM API & VA CHẠM HÌNH HỌC
> **Ngày**: 2026-10-02  
> **Trạng thái**: [DECIDED]  
> **Liên quan**: overview.md phần 2.1, 2.2, 2.3  

---

## 1. Câu hỏi gốc tôi đã hỏi LLM
> "Thiết kế kiến trúc lớp RobotNavEnv chuẩn Gymnasium API, cơ chế phân định terminated vs truncated, thuật toán phát hiện va chạm hình học và cơ chế render đồ họa."

---

## 2. Tóm tắt kết luận rút ra (Key Takeaways)
- **Tương thích Gymnasium v0.26+**: Kế thừa `gymnasium.Env`, phân định rạch ròi giữa `terminated` (đạt đích hoặc đâm va) và `truncated` (hết thời gian tối đa $T_{\max} = 500$ steps).
- **Phát hiện va chạm chính xác (Point-to-Segment)**: Không dựa vào các tia LiDAR (tránh góc chết gây xuyên tường), mà dùng thuật toán khoảng cách hình học chính xác từ tâm robot $\mathbf{p}$ tới tất cả đoạn thẳng trong bản đồ: $\min_j \text{dist}(\mathbf{p}, \text{segment}_j) \le r_{\text{robot}} = 0.25\text{m}$.
- **Metadata `info` đầy đủ**: Trả về dữ liệu phục vụ trực tiếp cho NEAT Fitness và Novelty Search: `final_pos` $(x, y)$ (bộ mô tả hành vi $b(x)$), `trajectory` (để tính chỉ số SPL), `spin_steps`, `is_success`, `is_collision`.
- **Render linh hoạt với Pygame**: Hỗ trợ 2 chế độ: `human` (mở cửa sổ tương tác 60 FPS) và `rgb_array` (xuất frame ảnh headless để ghi video MP4/GIF chất lượng cao phục vụ slide thuyết trình).

---

## 3. Chi tiết Kỹ thuật & Code Kiến trúc

### 3.1. Thuật toán Va chạm Điểm - Đoạn thẳng (Point-to-Segment Vectorized)
Cho điểm $\mathbf{p} = (x_r, y_r)$ và đoạn thẳng $\mathbf{AB}$. Hình chiếu vuông góc của $\mathbf{p}$ lên đoạn thẳng có tham số $t$:
$$t = \text{clip}\left(\frac{(\mathbf{p} - \mathbf{A}) \cdot (\mathbf{B} - \mathbf{A})}{\|\mathbf{B} - \mathbf{A}\|^2}, 0.0, 1.0\right)$$
Điểm gần nhất trên đoạn thẳng: $\mathbf{p}_{\text{proj}} = \mathbf{A} + t (\mathbf{B} - \mathbf{A})$.  
Khoảng cách: $d = \|\mathbf{p} - \mathbf{p}_{\text{proj}}\|$.

```python
def check_collision_vectorized(robot_pos: np.ndarray, segments: np.ndarray, robot_radius: float = 0.25) -> bool:
    """
    robot_pos: (2,)
    segments: (M, 2, 2)
    Trả về True nếu khoảng cách nhỏ nhất <= robot_radius
    """
    A = segments[:, 0, :]  # (M, 2)
    B = segments[:, 1, :]  # (M, 2)
    AB = B - A            # (M, 2)
    AP = robot_pos - A     # (M, 2)

    AB_squared = np.sum(AB ** 2, axis=1)  # (M,)
    AB_squared = np.maximum(AB_squared, 1e-8)

    t = np.sum(AP * AB, axis=1) / AB_squared  # (M,)
    t = np.clip(t, 0.0, 1.0)[:, np.newaxis]   # (M, 1)

    closest_pt = A + t * AB                    # (M, 2)
    dists = np.linalg.norm(robot_pos - closest_pt, axis=1) # (M,)

    return bool(np.any(dists <= robot_radius))
```

### 3.2. Cấu trúc Lớp `RobotNavEnv` (Skeleton)
```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RobotNavEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, map_data: dict, render_mode: str = None):
        super().__init__()
        self.map_data = map_data
        self.render_mode = render_mode
        self.robot_radius = 0.25
        self.goal_radius = 0.30
        self.max_steps = 500
        self.dt = 0.1
        self.v_max = 0.5
        self.L = 0.4  # Wheel base

        # 15D Observation & 2D Continuous Action
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(15,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.spin_count = 0
        self.pos = np.array(self.map_data["start"], dtype=np.float32)
        self.theta = 0.0
        self.trajectory = [self.pos.copy()]
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        oL, oR = np.clip(action, -1.0, 1.0)
        vL, vR = self.v_max * oL, self.v_max * oR

        # Kiểm tra hành vi quay tròn tại chỗ: |vL - (-vR)| < epsilon
        if abs(vL + vR) < 0.05 and abs(vL - vR) > 0.2:
            self.spin_count += 1

        # Cập nhật động học
        omega = (vR - vL) / self.L
        v = (vR + vL) / 2.0
        self.theta += omega * self.dt
        self.pos[0] += v * np.cos(self.theta) * self.dt
        self.pos[1] += v * np.sin(self.theta) * self.dt
        self.trajectory.append(self.pos.copy())

        # Kiểm tra điều kiện kết thúc
        is_collision = check_collision_vectorized(self.pos, self.map_data["segments"], self.robot_radius)
        dist_to_goal = np.linalg.norm(self.pos - np.array(self.map_data["goal"]))
        is_success = dist_to_goal <= self.goal_radius

        terminated = is_collision or is_success
        truncated = (self.step_count >= self.max_steps) and not terminated

        reward = 0.0  # Fitness sẽ được tính toán riêng sau episode cho NEAT

        info = {
            "is_success": is_success,
            "is_collision": is_collision,
            "steps": self.step_count,
            "final_pos": tuple(self.pos),
            "trajectory": self.trajectory,
            "spin_steps": self.spin_count,
            "dist_to_goal": dist_to_goal
        }
        return self._get_obs(), reward, terminated, truncated, info
```

---

## 4. Quyết định của tôi
- [x] **Chọn Phương án A cho Phát hiện va chạm**: Dùng hình học Point-to-Segment chính xác tuyệt đối. Loại bỏ hoàn toàn khả năng robot bị lỗi xuyên tường qua kẽ hở giữa các tia LiDAR.
- [x] **Chọn Phương án A cho Render**: Dùng Pygame đa chế độ (`human` + `rgb_array`). Hỗ trợ vừa xem trực tiếp lúc debug vừa tự động xuất video/GIF cho bài báo cáo và slide thuyết trình.

---

## 5. TODO tiếp theo
- [ ] Hoàn thiện code `src/env/robot_env.py` và `src/env/renderer.py`.
- [ ] Chuyển sang Nhóm 2: **`02_neat/fitness_design.md`** để mổ xẻ hàm Fitness 5 thành phần.
