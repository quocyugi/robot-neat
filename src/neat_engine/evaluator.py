"""
NEAT Genome Evaluator — Fitness Computation
=============================================
Runs a genome through one or more maps and computes the 5-component
fitness score.

Reference
---------
- brainstorm/02_neat/fitness_design.md  (Decisions #8, #9)
- overview.md §4.1

Sửa lỗi L26 (tiến độ đo theo đường chim bay)
---------------------------------------------
Khi môi trường có trường geodesic (src.env.geodesic), tiến độ đo theo đường đi
thật: (g0 - g_min) / g0, trọng số W_PROGRESS_GEO = 10 và không giảm nửa khi va
chạm. Đây đúng là fitness "geo" đã dùng cho mọi kết quả ở mục 8.10–8.14 của báo
cáo; cùng CMA-ES, nó cho SR cấp 3 và 4 cao hơn hẳn kiểu đo cũ. Kiểu đo cũ vẫn
chọn được bằng progress="euclid" (để làm ablation).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.env.kinematics import MAX_STEPS


# ── Fitness weights (Decision #8) ────────────────────────────────────────
W_PROGRESS: float = 1.0
W_GOAL: float = 10.0
W_TIME: float = 5.0
W_COLLISION: float = 2.0
W_SPIN: float = 1.0
F_OFFSET: float = 3.0       # ensures F > 0 for neat-python
W_PROGRESS_GEO: float = 10.0  # trọng số tiến độ khi đo theo đường đi thật (L26)

PROGRESS_MODES = ("geodesic", "euclid", "auto")


def compute_episode_fitness(
    info: dict[str, Any],
    w1: float = W_PROGRESS,
    w2: float = W_GOAL,
    w3: float = W_TIME,
    w4: float = W_COLLISION,
    w5: float = W_SPIN,
    max_steps: int = MAX_STEPS,
    progress: str = "auto",
    w_geo: float = W_PROGRESS_GEO,
) -> float:
    """Compute fitness for a single episode from the env's ``info`` dict.

    progress : "geodesic" | "euclid" | "auto"
        "auto" dùng geodesic khi info có "g0" (env được tạo với trường geodesic),
        nếu không thì đo theo đường chim bay như code cũ.

    Returns
    -------
    fitness : float ≥ 0.01   (always positive for neat-python)
    """
    if progress not in PROGRESS_MODES:
        raise ValueError(f"progress must be one of {PROGRESS_MODES}, got {progress!r}")
    is_goal: float = float(info["is_success"])
    is_col: float = float(info["is_collision"])
    steps: int = info["steps"]
    spin_steps: int = info["spin_steps"]

    use_geo = progress == "geodesic" or (progress == "auto" and info.get("g0") is not None)
    if use_geo:
        if info.get("g0") is None:
            raise ValueError("progress='geodesic' cần env tạo với trường geodesic (RobotNavEnv(..., geodesic=...))")
        g0: float = info["g0"]
        g_min: float = info["g_min"]
        # 1. Tiến độ theo đường đi thật, không giảm nửa khi va chạm (đã có phạt va chạm riêng)
        progress_term = w_geo * min(max((g0 - g_min) / max(g0, 1e-5), 0.0), 1.0)
    else:
        d0: float = info["d0"]
        d_min: float = info["min_distance_to_goal"]
        # 1. Progress (capped at [0, 1]) — kiểu đo cũ theo đường chim bay
        raw_progress = max(0.0, (d0 - d_min) / max(d0, 1e-5))
        progress_term = w1 * (0.5 * raw_progress if is_col else raw_progress)

    # 2. Goal + time bonus
    time_bonus = (1.0 - steps / max_steps) * is_goal

    # 3. Spin penalty
    spin_penalty = spin_steps / max_steps

    # 4. Raw fitness
    f_raw = (
        progress_term
        + w2 * is_goal
        + w3 * time_bonus
        - w4 * is_col
        - w5 * spin_penalty
    )

    # 5. Offset to guarantee positivity
    return max(f_raw + F_OFFSET, 0.01)


def max_episode_fitness(progress: str = "geodesic") -> float:
    """Fitness lớn nhất có thể của một lượt (dùng để chuẩn hóa theo thang cố định, L08)."""
    w_prog = W_PROGRESS_GEO if progress in ("geodesic", "auto") else W_PROGRESS
    return w_prog + W_GOAL + W_TIME + F_OFFSET


def evaluate_genome_on_maps(
    genome_net,
    maps: list[dict],
    env_cls,
    is_recurrent: bool = False,
    geodesics: list | None = None,
    progress: str = "auto",
) -> tuple[float, list[dict]]:
    """Evaluate a single genome across *K* maps and return mean fitness.

    Bản tuần tự, giữ để tương thích. Huấn luyện dùng src.neat_engine.parallel.

    Parameters
    ----------
    genome_net
        An activated NEAT network (FeedForwardNetwork or RecurrentNetwork).
    maps : list[dict]
        List of map_data dicts loaded from JSON.
    env_cls
        The RobotNavEnv class (passed to avoid circular imports).
    is_recurrent : bool
        If True, call ``genome_net.reset()`` at the start of each episode.
    geodesics : list[GeodesicField], optional
        Trường geodesic của từng bản đồ (cùng thứ tự với maps).

    Returns
    -------
    mean_fitness : float
    episode_infos : list[dict]
    """
    fitnesses: list[float] = []
    infos: list[dict] = []

    for i, map_data in enumerate(maps):
        geo = geodesics[i] if geodesics is not None else None
        env = env_cls(map_data=map_data, geodesic=geo)
        obs, _ = env.reset()

        if is_recurrent and hasattr(genome_net, "reset"):
            genome_net.reset()  # Decision #13: clear hidden state

        terminated = truncated = False
        while not (terminated or truncated):
            action = genome_net.activate(obs.tolist())
            action = np.array(action, dtype=np.float32)
            obs, _, terminated, truncated, info = env.step(action)

        fitnesses.append(compute_episode_fitness(info, progress=progress))
        infos.append(info)
        env.close()

    return float(np.mean(fitnesses)), infos
