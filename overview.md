# PROJECT OVERVIEW: NEAT-BASED AUTONOMOUS ROBOT NAVIGATION
> **Học phần**: Tính toán tiến hóa (Evolutionary Computation)  
> **Đề tài**: Điều hướng robot tự hành trong môi trường có vật cản bằng Neuroevolution (NEAT)  
> **Nhóm**: Nhóm 3  
> **Mục đích tài liệu**: Cung cấp toàn bộ bối cảnh học thuật, mô hình toán, kiến trúc hệ thống, dữ liệu, thiết kế thực nghiệm và các câu hỏi mở để phục vụ **brainstorming, phản biện và đồng phát triển cùng LLM**.

---

## 1. TỔNG QUAN DỰ ÁN & TRIẾT LÝ HỌC THUẬT

### 1.1. Phát biểu bài toán (Problem Statement)
* **Tác tử**: Robot vi sai 2 bánh (Differential Drive Robot) hoạt động trong không gian 2D liên tục kích thước $10\text{m} \times 10\text{m}$.
* **Nhiệm vụ**: Xuất phát từ điểm $S(x_0, y_0, \theta_0)$, điều hướng đến đích $G(x_g, y_g)$ trong thời gian tối đa $T_{\max} = 60\text{s}$ (tương đương 600 bước điều khiển với $\Delta t = 0.1\text{s}$) mà không được va chạm vào bất kỳ chướng ngại vật nào.
* **Ràng buộc thông tin**: **Robot hoàn toàn KHÔNG có bản đồ toàn cục (Global Map)**. Robot chỉ được trang bị các cảm biến khoảng cách cục bộ (Local Sensors - LiDAR giả lập) và tín hiệu định vị tương đối hướng về đích.
* **Bản chất bài toán**: Tối ưu hóa hộp đen (Black-box Optimization) trên không gian chính sách (Policy Space Search). Không có nhãn cho từng bước điều khiển $\to$ **Không thể dùng Học có giám sát (Supervised Learning)**; hàm mục tiêu chỉ chấm điểm sau cả một lượt chạy (Episode).

### 1.2. Tại sao lại chọn Neuroevolution & NEAT?
So với các hướng tiếp cận khác:
1. **Quy hoạch cổ điển ($A^*$, DWA, PRM)**: Đòi hỏi bản đồ toàn cục tĩnh đã biết trước.
2. **Trường thế nhân tạo (APF - Artificial Potential Field)**: Dễ tính toán nhưng thường xuyên mắc kẹt ở cực trị địa phương (bẫy chữ U, ngõ cụt).
3. **Học tăng cường sâu (Deep RL - PPO, SAC, TD3)**: Rất mạnh nhưng phụ thuộc nhiều vào reward shaping từng bước, sample inefficiency cao, nhạy cảm với siêu tham số.
4. **Genetic Algorithm với mạng cố định (Fixed-topology Neuroevolution)**: Mắc kẹt ở khâu thiết kế kiến trúc mạng (phải đoán số tầng/số nút ẩn). Nếu mạng nhỏ thì underfitting, mạng lớn thì không gian tìm kiếm trọng số quá lớn.
5. **NEAT (Stanley & Miikkulainen, 2002)**: Tiến hóa đồng thời cả **cấu trúc đồ thị (Topology)** và **trọng số (Weights)**.
   * Bắt đầu từ mạng tối giản nhất (minimal topology: chỉ nối thẳng Input $\to$ Output, không nút ẩn).
   * Tự phát triển thêm nút và liên kết khi cần $\to$ Mạng sinh ra nhỏ gọn, tốn ít tài nguyên tính toán, có khả năng giải thích hành vi.
   * Giải quyết triệt để vấn đề ghép đôi gen khác cấu trúc thông qua **Innovation Numbers**.
   * Bảo vệ các cấu trúc non trẻ thông qua **Phân loài (Speciation)**.

---

## 2. MÔ HÌNH TOÁN HỌC & ĐỘNG HỌC ROBOT

```
       +---------------------------------------------+
       |           CẢM BIẾN CỤC BỘ (15D)             |
       |  - 12 tia LiDAR quét 360 độ (0 - 3m)        |
       |  - Khoảng cách tới đích (chuẩn hóa)         |
       |  - Góc lệch tới đích (sin phi, cos phi)     |
       +---------------------------------------------+
                              |
                              v
                   +---------------------+
                   |   MẠNG NƠ-RON NEAT  |
                   | (Topology + Weights)|
                   +---------------------+
                              |
                              v
       +---------------------------------------------+
       |             HÀNH ĐỘNG ĐIỀU KHIỂN            |
       |   oL, oR in [-1, 1] qua hàm tanh            |
       |   vL = vmax * oL;  vR = vmax * oR           |
       +---------------------------------------------+
                              |
                              v
       +---------------------------------------------+
       |         ĐỘNG HỌC ROBOT VI SAI (2D)          |
       |     v = (vR + vL) / 2;  omega = (vR - vL)/L |
       |     theta_{t+1} = theta_t + omega * dt      |
       |     x_{t+1} = x_t + v * cos(theta) * dt     |
       |     y_{t+1} = y_t + v * sin(theta) * dt     |
       +---------------------------------------------+
```

### 2.1. Động học Robot vi sai (Differential Drive Kinematics)
* **Bán kính bánh xe / Thân xe**: Bán kính robot $r = 0.25\text{m}$, khoảng cách giữa 2 bánh $L = 0.4\text{m}$.
* **Vận tốc cực đại**: $v_{\max} = 0.5\text{m/s}$.
* **Bước thời gian mô phỏng**: $\Delta t = 0.1\text{s}$ (tần số 10 Hz).
* **Công thức cập nhật trạng thái** (Euler Integration):
  $$\omega_t = \frac{v_{R, t} - v_{L, t}}{L}, \quad v_t = \frac{v_{R, t} + v_{L, t}}{2}$$
  $$\theta_{t+1} = \theta_t + \omega_t \cdot \Delta t$$
  $$x_{t+1} = x_t + v_t \cdot \cos(\theta_{t+1}) \cdot \Delta t$$
  $$y_{t+1} = y_t + v_t \cdot \sin(\theta_{t+1}) \cdot \Delta t$$

### 2.2. Không gian Quan sát (Observation Space - 15 Chiều)
Toàn bộ đầu vào được chuẩn hóa về $[0, 1]$ hoặc $[-1, 1]$:
1. **$d_1 \dots d_{12}$ (12 chiều)**: Khoảng cách đo được từ 12 tia LiDAR giả lập phân bố đều $360^\circ$ quanh robot. Tầm quét tối đa $R_{\max} = 3.0\text{m}$. Giá trị đầu vào (Proximity Index): $s_i = 1.0 - \min(d_i, R_{\max}) / R_{\max} \in [0, 1]$ (an toàn = $0.0$, tiệm cận vật cản = $1.0$).
2. **$\rho$ (1 chiều)**: Khoảng cách đường chim bay từ robot tới đích, chuẩn hóa theo đường chéo bản đồ ($D = \sqrt{10^2 + 10^2} \approx 14.14\text{m}$): $\rho = \min(\|\mathbf{p} - \mathbf{p}_g\|_2 / D, 1.0) \in [0, 1]$.
3. **$\sin\varphi, \cos\varphi$ (2 chiều)**: Góc lệch $\varphi = \text{atan2}(y_g - y, x_g - x) - \theta$ giữa hướng đầu xe và hướng đích. 
   * *Lý do kỹ thuật*: Sử dụng cặp $(\sin\varphi, \cos\varphi)$ thay vì góc trực tiếp $\varphi \in [-\pi, \pi]$ để đảm bảo tính liên tục của hàm số, tránh hiện tượng bất liên tục tại điểm biên $\pm \pi$.

### 2.3. Không gian Hành động (Action Space - 2 Chiều)
* Mạng xuất ra 2 giá trị liên tục $o_L, o_R \in [-1, 1]$ qua hàm kích hoạt $\tanh$.
* Vận tốc thực tế của từng bánh:
  $$v_L = v_{\max} \cdot o_L, \quad v_R = v_{\max} \cdot o_R$$
* *Lưu ý*: Cho phép vận tốc âm để robot có thể lùi và xoay tại chỗ.

---

## 3. CƠ CHẾ CỐT LÕI CỦA NEAT

NEAT giải quyết 3 thách thức lớn nhất của việc tiến hóa mạng nơ-ron:

### 3.1. Số đổi mới (Historical Markings / Innovation Numbers)
* **Vấn đề (Competing Conventions)**: Hai mạng nơ-ron khác nhau có thể cùng biểu diễn một hàm logic nhưng cấu trúc nút ẩn bị xáo trộn. Phép lai ghép cắt chéo (crossover) thông thường sẽ tạo ra cá thể con bị đột biến khuyết tật.
* **Giải pháp NEAT**: Mỗi khi một gen kết nối mới xuất hiện do đột biến, một số thứ tự toàn cục (Global Innovation Number) được gán cho nó. Khi lai ghép giữa hai mạng khác cấu trúc, thuật toán chỉ cần xếp thẳng hàng các gen có cùng số đổi mới (Matching Genes) để trao đổi chéo, còn các gen dư thừa (Excess) hoặc rời rạc (Disjoint) sẽ được thừa hưởng từ cha/mẹ có độ thích nghi cao hơn.

### 3.2. Phân loài (Speciation) & Chia sẻ độ thích nghi (Fitness Sharing)
* **Vấn đề**: Một đột biến cấu trúc mới (thêm nút) thường cần thời gian để tối ưu trọng số. Nếu cạnh tranh sinh tồn ngay với các cá thể đã ổn định, nó sẽ bị loại bỏ sớm.
* **Giải pháp**:
  * **Khoảng cách tương thích (Compatibility Distance)** giữa hai genome:
    $$\delta = \frac{c_1 E}{N} + \frac{c_2 D}{N} + c_3 \bar{W}$$
    (Với $E$: số gen dư thừa, $D$: số gen rời rạc, $\bar{W}$: chênh lệch trọng số trung bình, $N$: số gen tối đa).
  * Gom các cá thể có $\delta < \delta_t$ vào cùng loài.
  * **Chia sẻ fitness (Explicit Fitness Sharing)**:
    $$f'_i = \frac{f_i}{\sum_{j=1}^{N_s} \text{sh}(\delta(i, j))}$$
    Ngăn chặn một loài phát triển áp đảo và bảo tồn các ngách sinh thái (ecological niches).

### 3.3. Phức tạp hóa dần (Complexification)
* Mọi cá thể bắt đầu với 15 inputs kết nối trực tiếp đến 2 outputs (không có hidden nodes).
* Đột biến thêm kết nối ($p_{\text{add\_conn}} = 0.05$) hoặc thêm nút ($p_{\text{add\_node}} = 0.03$) dần dần tạo ra các cấu trúc nơ-ron phong phú.

---

## 4. HÀM FITNESS & HIỆN TƯỢNG FITNESS ĐÁNH LỪA (DECEPTIVE FITNESS)

### 4.1. Hàm Fitness 5 Thành phần
Hàm thích nghi thô cho một lượt chạy đơn lẻ:
$$F_{\text{raw}} = w_1 \cdot \text{Progress} + w_2 \cdot I_{\text{goal}} + w_3 \cdot \left(1 - \frac{t_{\text{goal}}}{T_{\max}}\right) \cdot I_{\text{goal}} - w_4 \cdot I_{\text{collision}} - w_5 \cdot \frac{n_{\text{spin}}}{T_{\max}}$$
Trong đó:
$$\text{Progress} = \begin{cases} \frac{d_0 - d_{\min}}{d_0} & \text{nếu an toàn} \\ 0.5 \cdot \frac{d_0 - d_{\min}}{d_0} & \text{nếu va chạm} \end{cases}$$
Hàm thích nghi thực tế bảo đảm luôn dương cho `neat-python`:
$$F = \max(F_{\text{raw}} + 3.0, \; 0.01)$$

| Thành phần | Ý nghĩa kỹ thuật | Trọng số đề xuất |
| :--- | :--- | :--- |
| $\text{Progress}$ | Tỉ lệ quãng đường gần đích nhất từng đạt ($d_{\min}$) so với xuất phát ($d_0$) | $w_1 = 1.0$ |
| $I_{\text{goal}}$ | Biến chỉ thị: $+1$ nếu chạm vùng đích ($\|\mathbf{p} - \mathbf{p}_g\| < 0.3\text{m}$) | $w_2 = 10.0$ |
| $1 - \frac{t_{\text{goal}}}{T_{\max}}$ | Thưởng thêm nếu hoàn thành hành trình nhanh (tiết kiệm thời gian) | $w_3 = 5.0$ |
| $I_{\text{collision}}$ | Biến chỉ thị phạt nếu va chạm vật cản (dừng ngay lượt chạy) | $w_4 = 2.0$ |
| $\frac{n_{\text{spin}}}{T_{\max}}$ | Phạt hành vi gian lận quay tròn tại chỗ ($\|v_L - (-v_R)\| < \epsilon$ hoặc $\sum\|\Delta\theta\| > 4\pi$) | $w_5 = 1.0$ |
| Base Offset | Hằng số dịch chuyển đảm bảo Fitness không âm cho Roulette Selection | $+3.0$ |

*Mỗi genome được chạy trên $K = 4$ bản đồ ngẫu nhiên trong mỗi thế hệ; Fitness cá thể bằng trung bình cộng của 4 lượt chạy.*

### 4.2. Bẫy chữ U & Hiện tượng Deceptive Fitness
* **Bản chất**: Khi chướng ngại vật có dạng lòng chảo / ngõ cụt hình chữ U chắn giữa robot và đích, đáy của chữ U là vị trí có khoảng cách Euclid tới đích ngắn nhất.
* **Hậu quả**: Hàm fitness dựa trên khoảng cách sẽ liên tục thưởng cho robot tiến sâu vào đáy bẫy. Con đường giải đúng bắt buộc robot phải chấp nhận đi lùi hoặc đi vòng ra xa đích tạm thời. Do đó, quần thể bị mắc kẹt vĩnh viễn ở cực trị địa phương (local optima).
* **Kết quả kinh điển (Lehman & Stanley, 2011)**:
  * NEAT dùng fitness khoảng cách: chỉ giải được **3/40 lần chạy** (7.5%).
  * NEAT dùng Novelty Search: giải được **39/40 lần chạy** (97.5%).

### 4.3. Giải pháp: Novelty Search (Tìm kiếm tính mới lạ)
* **Ý tưởng**: Hoàn toàn bỏ qua mục tiêu khoảng cách, chỉ thưởng điểm cho **hành vi mới lạ** (độc đáo, chưa từng xuất hiện).
* **Bộ mô tả hành vi (Behavior Characterization)**: $b(x) = (x_{\text{final}}, y_{\text{final}})$ (tọa độ vị trí dừng chân cuối cùng của robot).
* **Điểm Novelty $\rho(x)$**:
  $$\rho(x) = \frac{1}{k} \sum_{i=1}^k \text{dist}(b(x), \mu_i)$$
  Trong đó $\mu_i$ là $k = 15$ hàng xóm gần nhất của $b(x)$ trong tập hợp gồm (Quần thể hiện tại $\cup$ Kho lưu trữ Archive các hành vi mới trong quá khứ).
* **Cơ chế lai (Hybrid / Minimal Criteria Novelty Search)**:
  $$F_{\text{hybrid}} = \lambda \cdot F_{\text{fitness}} + (1 - \lambda) \cdot \rho(x), \quad \lambda \in [0, 1]$$
* **Mạng hồi quy (Recurrent NEAT)**: Cho phép kết nối ngược để mạng có trạng thái ẩn (Internal Memory), giúp nhận biết nó đang đi vào ngõ cụt để quay đầu.

---

## 5. DỮ LIỆU: BỘ BẢN ĐỒ & CURRICULUM LEARNING

### 5.1. Pipeline 6 Bước Sinh Bản Đồ Tự Động (Procedural Generation)
1. **Khởi tạo**: Chọn cấp độ khó (1–4) và Random Seed (đảm bảo tính tái lập).
2. **Rải vật cản**: Đặt chướng ngại vật tĩnh (hình tròn, hình chữ nhật) theo tham số mật độ của cấp.
3. **Sinh điểm $S$ và $G$**: Đảm bảo khoảng cách tối thiểu $\|\mathbf{p}_s - \mathbf{p}_g\| \ge 5\text{m}$ và cách vật cản tối thiểu $r + 0.2\text{m}$.
4. **Kiểm tra tính liên thông bằng $A^*$**: Chạy thuật toán $A^*$ trên lưới $5\text{cm}$ với vật cản được phình to thêm một bán kính robot $r + 5\text{cm}$ (C-Space Inflation). **Nếu không có đường đi hợp lệ $\to$ Hủy và sinh lại**.
5. **Đo lường độ khó**: Tính hệ số quanh co (Tortuosity) $\tau = \frac{L_{A^*}}{d_{\text{Euclid}}}$, khe hở hẹp nhất, cờ ngõ cụt.
6. **Lưu trữ**: Xuất file định dạng JSON + CSV + Ảnh render 2D.

### 5.2. Phân cấp Bản đồ & Chiến lược Chia Tập

| Cấp độ | Tên gọi | Đặc trưng địa hình | Độ quanh co $\tau$ |
| :---: | :--- | :--- | :---: |
| **Cấp 1** | Phòng trống (Empty) | Không vật cản, chỉ có 4 bức tường biên; luyện kỹ năng đi thẳng tới đích. | $\approx 1.0$ |
| **Cấp 2** | Rải rác (Sparse) | Các khối hộp/trụ phân tán ngẫu nhiên; luyện phản xạ né vật cản đơn giản. | $1.05 - 1.25$ |
| **Cấp 3** | Hành lang hẹp (Corridors) | Các bức tường xếp zíc-zắc tạo hẻm nhỏ; luyện luồn lách trong không gian hẹp. | $1.25 - 1.60$ |
| **Cấp 4** | Ngõ cụt (U-Traps) | Chứa các bẫy chữ U, chữ L chắn trước đích; thử thách vượt bẫy đánh lừa. | $> 1.60$ |

* **Quy tắc phân chia tập dữ liệu**:
  * **Tập Train**: Sinh mới ngẫu nhiên mỗi thế hệ bằng dải seed `0 – 99,999` (chống học vẹt).
  * **Tập Validation**: Cố định 100 bản đồ (25 bản đồ/cấp, seed `100,000 – 100,099`) dùng để kiểm tra điều kiện thăng cấp trong Curriculum Learning.
  * **Tập Test**: Cố định 200 bản đồ độc lập (50 bản đồ/cấp, seed `200,000 – 200,199`), chỉ chạy kiểm thử 1 lần duy nhất để báo cáo kết quả.
  * **Tập Test đặc biệt**: 2 mê cung Lehman & Stanley + 50 bản đồ trích từ BARN.

### 5.3. Chiến lược Học theo Giáo trình (Curriculum Learning)
* Quần thể bắt đầu tiến hóa ở **Cấp 1**.
* Sau mỗi 10 thế hệ, đánh giá genome tốt nhất trên tập Validation của cấp hiện tại.
* **Quy tắc chuyển cấp**: Nếu Tỉ lệ thành công trên tập Val đạt $SR_{\text{val}} \ge 80\%$, hệ thống tự động nâng độ khó lên cấp tiếp theo ($1 \to 2 \to 3 \to 4$).

---

## 6. THIẾT KẾ THỰC NGHIỆM & PHƯƠNG PHÁP ĐÁNH GIÁ (BASELINES)

### 6.1. Ba Câu hỏi Nghiên cứu (Research Questions)
* **Q1 (Topology vs. Weight-only)**: Tiến hóa cả cấu trúc (NEAT) có đạt hiệu năng cao hơn và hội tụ nhanh hơn phương pháp chỉ tiến hóa trọng số trên mạng cố định (Fixed-topology GA/CMA-ES) khi cấp cùng ngân sách tính toán (Evaluations Budget)?
* **Q2 (Novelty Search vs. Deceptive Fitness)**: Trên các bản đồ có bẫy chữ U, Novelty Search có giải quyết thành công ở số lượng run vượt trội so với hàm fitness khoảng cách truyền thống?
* **Q3 (Generalization & Zero-Shot Transfer)**: Bộ điều khiển học trên bản đồ tự sinh có khả năng tổng quát hóa thành công sang bản đồ BARN chưa từng thấy?

### 6.2. Danh mục 7 Baseline Đối chuẩn (B1 – B7)
1. **B1**: Fixed MLP (1 tầng ẩn, 8 nút) tiến hóa bằng GA truyền thống.
2. **B2**: Fixed MLP (2 tầng ẩn, 16-8 nút) tiến hóa bằng GA.
3. **B3**: Random Search (Kiểm tra tính không tầm thường của bài toán).
4. **B4**: Heuristic Rule-based (Quy tắc phản xạ nếu-thì né tường đơn giản).
5. **B5**: Artificial Potential Field (APF - Trường thế nhân tạo).
6. **B6**: $A^* + \text{DWA}$ (Dynamic Window Approach - Quy hoạch cổ điển có bản đồ toàn cục).
7. **B7**: Deep RL (PPO/SAC) chạy trên cùng môi trường Gymnasium (Baseline tùy chọn nâng cao).

### 6.3. Tiêu chuẩn Đánh giá & Thống kê Khoa học
* **Các chỉ số chính**:
  * **Tỉ lệ thành công (Success Rate - SR)**: Tỉ lệ phần trăm robot chạm đích an toàn trong $T_{\max}$.
  * **Success weighted by Path Length (SPL)**: Thước đo chuẩn quốc tế kết hợp giữa tỉ lệ thành công và độ tối ưu quãng đường:
    $$\text{SPL} = \frac{1}{N} \sum_{i=1}^N S_i \cdot \frac{L_{\text{optimal}, i}}{\max(L_{\text{actual}, i}, L_{\text{optimal}, i})}$$
  * **Thời gian hoàn thành trung bình (MTT)** và **Tỉ lệ va chạm (Collision Rate)**.
* **Quy chuẩn thống kê**:
  * Chạy lặp lại **10 lần độc lập (10 seeds)** cho mỗi cấu hình thí nghiệm.
  * Báo cáo **Trung vị (Median)** và **Khoảng tứ phân vị (IQR)** (thay vì Mean/Std vì phân phối di truyền thường bị lệch).
  * Sử dụng kiểm định phi tham số **Mann–Whitney U test** với mức ý nghĩa $p < 0.05$ để khẳng định sự vượt trội có ý nghĩa thống kê.

---

## 7. MỞ RỘNG SIM-TO-REAL: BENCHMARK DUnE (3D OFF-ROAD)

### 7.1. Giới thiệu DUnE (University of Central Florida, 2025)
* **DUnE** (*Dynamic Unstructured Environment*): Nền tảng benchmark quốc tế đánh giá hệ thống điều hướng việt dã (Off-Road) trên mô phỏng vật lý 3D chân thực cao.
* **Công nghệ**: ROS 2 Humble + Gazebo Fortress + Wrapper giao diện Gymnasium.
* **Địa hình**: Bùn lầy, dốc nghiêng, đất đá gồ ghề, có chướng ngại vật động (mô hình người đi bộ).

### 7.2. Chiến lược Chuyển giao Hai Giai đoạn (Two-Stage Transfer)
Do Gazebo 3D tiêu tốn tài nguyên gấp hàng chục lần so với 2D, không thể chạy 180.000 lượt đánh giá trực tiếp trên Gazebo:
* **Giai đoạn 1**: Huấn luyện toàn bộ cấu trúc và trọng số NEAT cốt lõi trên môi trường 2D tự viết siêu tốc (Vectorized NumPy + Multiprocessing).
* **Giai đoạn 2**: Lấy genome tốt nhất chuyển giao (Sim-to-Sim) sang DUnE, bổ sung góc cảm biến Pitch/Roll từ IMU (chống lật xe) và chỉ cần **tinh chỉnh (fine-tune) thêm 20–30 thế hệ**.

### 7.3. Bộ 5 Chỉ số Đánh giá Chuẩn của DUnE
1. **SR (Success Rate)**: Tỉ lệ tới đích không đâm đụng, không lật xe.
2. **TC (Total Collisions)**: Tổng số va chạm với vật cản tĩnh/động.
3. **MTT (Mean Time to Traverse)**: Thời gian trung bình để hoàn thành lộ trình (giây).
4. **VORT (Velocity Over Rough Terrain)**: Tốc độ trung bình duy trì trên địa hình gồ ghề hiểm trở.
5. **OC (Obstacle Clearance)**: Khoảng cách an toàn nhỏ nhất duy trì tới chướng ngại vật (càng lớn càng an toàn).

---

## 8. CẤU TRÚC KỸ THUẬT & STACK CÔNG NGHỆ

```
Slide TTTH/
├── Slide_NEAT.pptx                  # Slide báo cáo chính thức (18 slides)
├── đề tài tính toán tiến hoá.pdf    # Đề cương đề tài gốc chi tiết
├── SCRIPT_THUYET_TRINH_NEAT_ROBOT.md# Toàn văn lời thoại thuyết trình & Q&A
├── overview.md                      # [Tài liệu hiện tại] Bản tổng lược cho LLM Brainstorm
└── src/ (Cấu trúc mã nguồn đề xuất)
    ├── env/
    │   ├── robot_env.py             # Môi trường 2D chuẩn Gymnasium (Vectorized Ray-casting)
    │   ├── kinematics.py            # Mô hình động học robot vi sai
    │   └── map_generator.py         # Pipeline 6 bước sinh bản đồ & kiểm tra A*
    ├── neat_engine/
    │   ├── neat_runner.py           # Vòng lặp tiến hóa sử dụng neat-python
    │   ├── novelty_search.py        # Module tính Novelty score & Archive
    │   └── curriculum.py            # Quản lý thăng cấp độ khó theo dõi Validation SR
    ├── baselines/
    │   ├── fixed_ga.py              # Baseline mạng cố định (B1, B2)
    │   ├── apf_controller.py        # Baseline trường thế nhân tạo (B5)
    │   └── astar_dwa.py             # Mốc tham chiếu cổ điển (B6)
    ├── configs/
    │   ├── config_feedforward.txt   # File cấu hình NEAT chuẩn
    │   └── config_recurrent.txt     # File cấu hình Recurrent NEAT
    └── experiments/
        ├── run_benchmarks.py        # Đánh giá 10 runs, xuất số liệu CSV
        └── statistical_tests.py     # Tính Mann-Whitney U test, vẽ Boxplot
```

* **Công nghệ cốt lõi**: Python 3.10+, NumPy, neat-python $\ge 1.0$, Gymnasium, Pygame, SciPy, Matplotlib, multiprocessing.

---

## 9. CÁC ĐỀ MỤC MỞ CẦN BRAINSTORM CÙNG LLM (DISCUSSION TOPICS)

Dưới đây là các chủ đề trọng tâm bạn có thể đào sâu thảo luận với LLM:

### 💡 Chủ đề 1: Tối ưu hóa hàm Fitness & Tránh hành vi gian lận (Reward Exploits)
* *Câu hỏi gợi mở*: Hiện tại hàm fitness đã phạt va chạm và phạt quay tròn. Liệu robot có thể phát hiện hành vi gian lận khác (ví dụ: tiến lùi liên tục tại chỗ, hoặc chạy vòng tròn rộng quanh vạch xuất phát)? Có nên tích hợp thêm thành phần phạt độ mượt của tay lái (smoothness penalty) $\Delta \omega = |\omega_t - \omega_{t-1}|$ không?

### 💡 Chủ đề 2: Tinh chỉnh cơ chế Novelty Search & Tránh bùng nổ Archive
* *Câu hỏi gợi mở*: Bộ mô tả hành vi $b(x)$ hiện chỉ lấy tọa độ điểm dừng $(x_{\text{final}}, y_{\text{final}})$. Liệu như vậy có đủ để phân biệt giữa một robot luồn lách qua mê cung và một robot đi vòng đường ngoài? Có nên mở rộng $b(x)$ thành chuỗi hành trình lấy mẫu đa điểm (Trajectory sampling) không? Cơ chế quản lý bộ nhớ Archive để không bị chậm thuật toán $k$-NN khi chạy hàng trăm thế hệ?

### 💡 Chủ đề 3: Thiết kế thí nghiệm Ablation Study
* *Câu hỏi gợi mở*: Để bài báo cáo đạt điểm tối đa, cần thiết kế bảng thực nghiệm Ablation như thế nào để chứng minh rõ vai trò của từng thành phần: (1) Bỏ Speciation, (2) Bỏ Curriculum Learning, (3) Bỏ Recurrent connections, (4) Thay đổi số lượng tia LiDAR (8 tia vs 12 tia vs 16 tia)?

### 💡 Chủ đề 4: Tăng tốc độ huấn luyện mô phỏng 2D
* *Câu hỏi gợi mở*: Với 180.000 lượt mô phỏng, thuật toán Ray-Casting giao cắt giữa tia và đoạn thẳng (Line-Segment Intersection) trên NumPy nên vector hóa thế nào để đạt tốc độ $> 1000$ FPS trên CPU? Khi nào thì việc chuyển sang GPU (TensorNEAT / JaxNEAT) thực sự đem lại lợi nhuận hiệu năng?

### 💡 Chủ đề 5: Lập luận bảo vệ đồ án trước Hội đồng
* *Câu hỏi gợi mở*: Chuẩn bị các kịch bản đối đáp nếu Giảng viên đặt các câu hỏi hóc búa về: tính khả thi của Sim-to-Real, việc sử dụng thư viện `neat-python` thay vì code từ đầu, và sự khác biệt giữa Neuroevolution và Deep Reinforcement Learning.
