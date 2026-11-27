"""
Bộ điều khiển chạy theo lô: mạng NEAT (hồi quy, truyền thẳng), FixedMLP (GA B1/B2), APF
========================================================================================
Mỗi lớp nhận quan sát float32 dạng (A, N + 3) của A lượt đang chạy và trả hành động (A, 2)
float64; lớp mô phỏng ép về float32 như run_episode (np.asarray(..., dtype=np.float32)).
keep(mask) nén theo các lượt còn chạy (cùng thứ tự với mảng trạng thái).

Mạng NEAT dựng từ CHÍNH đối tượng mạng của neat-python (net.input_nodes, net.output_nodes,
net.node_evals do RecurrentNetwork.create / FeedForwardNetwork.create tạo ra), nên thừa hưởng mọi
quy tắc của create: nút required (kể cả kết nối bị tắt với mạng hồi quy), thứ tự links, nút
"bias neuron" của mạng truyền thẳng, nút trong chu trình không bao giờ được tính.

Bảng đệm theo genome (mỗi lượt mang bảng của genome của nó), cột của ma trận giá trị V (A, C):
  0..n_in-1      đầu vào, theo thứ tự input_nodes
  n_in           cột Z luôn bằng 0.0: nguồn không phải đầu vào và không có trong node_evals (mạng hồi
                 quy: nút giữ 0.0 mãi), output không được tính (luôn ra 0.0), đệm của links
  n_in + 1 + j   nút thứ j của node_evals
  C - 1          cột chìm cho slot đệm (chỉ ghi, không bao giờ đọc)
Mạng hồi quy: một mức chứa mọi slot, đọc hết rồi mới ghi (cập nhật đồng bộ như hai dict values
của RecurrentNetwork). Mạng truyền thẳng: mức = độ sâu của nút (nút chỉ đọc nút ở mức thấp hơn, đã
tính trong cùng bước), cho cùng kết quả với tính lần lượt theo node_evals.
Tổng links = bản sao tổng bù Neumaier của sum() (Python >= 3.12), cùng thứ tự cộng; tanh =
math.tanh từng phần tử ở chế độ exact. Chỉ hỗ trợ tanh + sum (mọi config của repo); hàm khác
báo Unsupported và tác vụ được chạy lại bằng đường từng lượt. Trọng số phải là float (hoặc int) của
Python: sum() chỉ cộng bù khi mọi số hạng đúng kiểu float, nên trọng số kiểu khác (ví dụ np.float64 do
một toán tử đột biến cộng số NumPy vào) cũng báo Unsupported.
"""

from __future__ import annotations

import numpy as np

from src.env.kinematics import LIDAR_MAX_RANGE


class Unsupported(Exception):
    """Tác vụ không chạy được theo lô; BatchEvaluator chạy nó bằng parallel.eval_task (cùng kết quả)."""


# ── Mạng NEAT ─────────────────────────────────────────────────────────────
def neat_slots(net, recurrent: bool):
    """(n_in, slots, cột output, số cột) từ một mạng neat-python; slot = (mức, cột đích, cột nguồn,
    trọng số, bias, response). Báo Unsupported nếu có hàm khác tanh / sum, trọng số không phải float / int
    của Python, hoặc thứ tự lạ."""
    import neat.activations as A
    import neat.aggregations as G
    inputs, outputs = list(net.input_nodes), list(net.output_nodes)
    n_in = len(inputs)
    col = {k: i for i, k in enumerate(inputs)}
    if len(col) != n_in:
        raise Unsupported("khóa đầu vào lặp")
    zero = n_in                                   # cột luôn bằng 0.0
    for j, ev in enumerate(net.node_evals):
        if ev[0] in col:
            raise Unsupported("nút lặp trong node_evals")
        col[ev[0]] = n_in + 1 + j
    depth: dict = {}
    slots = []
    for j, (node, act, agg, bias, resp, links) in enumerate(net.node_evals):
        if act is not A.tanh_activation:
            raise Unsupported("hàm kích hoạt " + getattr(act, "__name__", str(act)) + " (chỉ hỗ trợ tanh)")
        if agg is not G.sum_aggregation:
            raise Unsupported("hàm gộp " + getattr(agg, "__name__", str(agg)) + " (chỉ hỗ trợ sum)")
        srcs = [col.get(i, zero) for i, _ in links]
        for _, w in links:
            # ivalues[i] * w với w kiểu np.float64 cho np.float64, khi đó sum() cộng thường từ trái sang phải chứ
            # không cộng bù như bản sao Neumaier (bias, response kiểu nào cũng cho cùng giá trị nên không cần chốt)
            if type(w) is not float and type(w) is not int:
                raise Unsupported(f"trọng số kiểu {type(w).__name__} (sum() chỉ cộng bù khi mọi số hạng là float)")
        ws = [float(w) for _, w in links]
        if recurrent:
            lvl = 0
        else:
            ds = []
            for i, _ in links:
                c = col.get(i)
                if c is not None and c > zero:            # nguồn là một nút được tính
                    if i not in depth:                   # mạng truyền thẳng đọc giá trị cũ: không mô phỏng
                        raise Unsupported("mạng truyền thẳng đọc nút được tính sau nó")
                    ds.append(depth[i])
            lvl = 1 + max(ds) if ds else 0
            depth[node] = lvl
        slots.append((lvl, n_in + 1 + j, srcs, ws, float(bias), float(resp)))
    outc = [col.get(o, zero) for o in outputs]
    return n_in, slots, outc, n_in + 1 + len(net.node_evals)


class NeatTables:
    """Bảng đệm theo genome, một khối cho mỗi mức (hồi quy: 1 mức; truyền thẳng: theo độ sâu)."""

    def __init__(self, nets, recurrent: bool):
        per = [neat_slots(n, recurrent) for n in nets]
        n_in = {p[0] for p in per}
        n_out = {len(p[2]) for p in per}
        if len(n_in) != 1 or len(n_out) != 1:
            raise Unsupported("các mạng trong lô có số đầu vào / đầu ra khác nhau")
        self.n_in = n_in.pop()
        self.n_out = n_out.pop()
        self.C = max(p[3] for p in per) + 1            # + cột chìm cho slot đệm
        sink = self.C - 1
        n_lv = 1 + max((s[0] for p in per for s in p[1]), default=0)
        G = len(per)
        self.levels = []
        for lv in range(n_lv):
            rows = [[s for s in p[1] if s[0] == lv] for p in per]
            J = max(1, max(len(r) for r in rows))
            K = max(1, max((len(s[2]) for r in rows for s in r), default=1))
            src = np.full((G, J, K), self.n_in, np.int64)     # đệm đọc cột Z
            w = np.zeros((G, J, K))
            fanin = np.zeros((G, J), np.int64)
            bias = np.zeros((G, J))
            resp = np.ones((G, J))
            dst = np.full((G, J), sink, np.int64)
            for g, r in enumerate(rows):
                for j, (_, d, srcs, ws, b, rp) in enumerate(r):
                    src[g, j, :len(srcs)] = srcs
                    w[g, j, :len(ws)] = ws
                    fanin[g, j] = len(srcs)
                    bias[g, j], resp[g, j], dst[g, j] = b, rp, d
            self.levels.append(dict(src=src, w=w, fanin=fanin, bias=bias, resp=resp, dst=dst))
        self.outc = np.array([p[2] for p in per], np.int64)


def neumaier_sum(xp, P, fanin):
    """sum() của Python >= 3.12 (tổng bù Neumaier, bắt đầu từ số nguyên 0) trên P[..., :fanin],
    từng phần tử: s = 0 + x0 (-0.0 thành +0.0), rồi cộng bù từng số hạng theo đúng thứ tự;
    danh sách rỗng cho 0.0 (sum([]) là số nguyên 0, mọi phép sau đó cho cùng giá trị)."""
    K = P.shape[-1]
    s = 0.0 + P[..., 0]
    s = xp.where(fanin >= 1, s, 0.0)
    c = xp.zeros_like(s)
    for k in range(1, K):
        m = fanin > k
        x = P[..., k]
        t = s + x
        corr = xp.where(xp.abs(s) >= xp.abs(x), (s - t) + x, (x - t) + s)
        c = xp.where(m, c + corr, c)
        s = xp.where(m, t, s)
    return xp.where((c != 0.0) & xp.isfinite(c), s + c, s)


class NeatPolicy:
    """Mạng NEAT theo lô; ep_task[e] = chỉ số mạng (trong nets) của lượt e."""

    def __init__(self, be, nets, recurrent: bool, ep_task):
        self.be = be
        xp = be.xp
        T = NeatTables(nets, recurrent)
        self.n_in, self.C = T.n_in, T.C
        idx = np.asarray(ep_task, np.int64)
        self.lv = [{k: be.dev(v[idx]) for k, v in L.items()} for L in T.levels]
        self.outc = be.dev(T.outc[idx])
        self.V = xp.zeros((len(idx), self.C))           # reset() đầu lượt: mọi giá trị 0.0

    def keep(self, m):
        self.lv = [{k: v[m] for k, v in L.items()} for L in self.lv]
        self.outc, self.V = self.outc[m], self.V[m]

    def step(self, obs32):
        be, xp = self.be, self.be.xp
        A = obs32.shape[0]
        V = self.V
        V[:, :self.n_in] = obs32.astype(xp.float64)    # obs.tolist(): float64 của giá trị float32
        flat = V.reshape(-1)
        base = xp.arange(A, dtype=xp.int64) * self.C
        for L in self.lv:                               # hồi quy: 1 mức, đọc hết rồi mới ghi
            P = flat[base[:, None, None] + L["src"]] * L["w"]          # ivalues[i] * w
            s = neumaier_sum(xp, P, L["fanin"])
            z = L["bias"] + L["resp"] * s
            z = 2.5 * z
            z = xp.where(z < 60.0, z, 60.0)              # min(60.0, z)
            z = xp.where(z > -60.0, z, -60.0)            # max(-60.0, z)
            flat[base[:, None] + L["dst"]] = be.tanh_neat(z)
        return flat[base[:, None] + self.outc]


# ── FixedMLP (GA B1, B2) ──────────────────────────────────────────────────
class MLPPolicy:
    """FixedMLP theo lô. exact: mỗi hàng một phép x @ W như FixedMLP.activate (np.matmul xếp chồng
    (A, 1, n) @ (A, n, m), cùng lời gọi BLAS gemv cho từng hàng) rồi np.tanh, tính ở host.
    device (GPU, không exact): cộng dồn từng phần tử trên GPU (không có cuBLAS), lệch ~1e-15."""

    def __init__(self, be, mlps, ep_task):
        self.be = be
        idx = np.asarray(ep_task, np.int64)
        gains = {float(m.gain) for m in mlps}
        if len(gains) != 1:
            raise Unsupported("các FixedMLP trong lô có gain khác nhau")
        self.gain = gains.pop()
        n_layers = len(mlps[0].weights)
        self.Wh = [np.stack([m.weights[l] for m in mlps])[idx] for l in range(n_layers)]
        self.bh = [np.stack([m.biases[l] for m in mlps])[idx] for l in range(n_layers)]
        self.on_device = be.gpu and not be.exact
        if self.on_device:
            self.W = [be.dev(w) for w in self.Wh]
            self.b = [be.dev(b) for b in self.bh]

    def keep(self, m):
        mh = self.be.host(m)
        self.Wh = [w[mh] for w in self.Wh]
        self.bh = [b[mh] for b in self.bh]
        if self.on_device:
            self.W = [w[m] for w in self.W]
            self.b = [b[m] for b in self.b]

    def step(self, obs32):
        be = self.be
        if not self.on_device:                           # đúng các lời gọi của FixedMLP.activate, ở host
            x = be.host(obs32).astype(np.float64)
            for Wh, bh in zip(self.Wh, self.bh):
                x = np.tanh(self.gain * (np.matmul(x[:, None, :], Wh)[:, 0, :] + bh))
            return be.dev(x)
        xp = be.xp
        x = obs32.astype(xp.float64)
        for W, b in zip(self.W, self.b):
            acc = x[:, 0:1] * W[:, 0, :]
            for i in range(1, W.shape[1]):
                acc = acc + x[:, i:i + 1] * W[:, i, :]
            x = xp.tanh(self.gain * (acc + b))
        return x


# ── APF (B5) ──────────────────────────────────────────────────────────────
class APFPolicy:
    """APFController.activate theo lô, đúng thứ tự phép tính (vòng i = 0..n-1 tuần tự).
    ctrl.vw (Waffle): đầu ra (speed, turn) thay cho hai bánh, như APFController.activate."""

    def __init__(self, be, n: int, ctrl):
        self.be, self.n = be, int(n)
        self.k_att, self.k_rep, self.d_safe = ctrl.k_att, ctrl.k_rep, ctrl.d_safe
        self.vw = bool(getattr(ctrl, "vw", False))
        ang = np.linspace(0, 2 * np.pi, self.n, endpoint=False)
        self.cos = [float(np.cos(a)) for a in ang]       # np.cos(angle) vô hướng như APFController
        self.sin = [float(np.sin(a)) for a in ang]

    def keep(self, m):
        pass

    def step(self, obs32):
        xp = self.be.xp
        o = obs32.astype(xp.float64)                     # np.array(obs[:n]) và obs[n+1], obs[n+2]
        n = self.n
        sin_phi, cos_phi = o[:, n + 1], o[:, n + 2]
        fx, fy = cos_phi * self.k_att, sin_phi * self.k_att
        rx = xp.zeros_like(fx)
        ry = xp.zeros_like(fx)
        inv_safe = 1.0 / self.d_safe
        for i in range(n):
            p = o[:, i]
            dist = (1.0 - p) * LIDAR_MAX_RANGE
            near = (p > 0.01) & (dist < self.d_safe)
            m = xp.where(0.05 > dist, 0.05, dist)        # max(dist, 0.05)
            mag = self.k_rep * (1.0 / m - inv_safe)
            rx = xp.where(near, rx - mag * self.cos[i], rx)
            ry = xp.where(near, ry - mag * self.sin[i], ry)
        tx, ty = fx + rx, fy + ry                        # f_total = f_att + f_rep
        speed = xp.clip(tx * 0.5, -1.0, 1.0)
        turn = xp.clip(ty * 0.5, -1.0, 1.0)
        if self.vw:                                      # Waffle: [float(speed), float(turn)]
            return xp.stack([speed, turn], axis=1)
        return xp.stack([xp.clip(speed - turn, -1.0, 1.0), xp.clip(speed + turn, -1.0, 1.0)], axis=1)
