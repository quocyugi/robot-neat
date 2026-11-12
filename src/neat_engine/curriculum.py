"""
Curriculum Learning — Automatic Level Promotion
=================================================
Monitors validation success rate and promotes the training level
when the champion genome consistently solves the current difficulty.

Reference
---------
- overview.md §5.3
- brainstorm evaluation: "logic simple enough to code directly"

Rules
-----
- Evaluate champion on validation set every *eval_interval* generations.
- Promote when SR_val ≥ *promotion_threshold* (default 80 %).
- Max level = 4 (U-Traps).

Các lỗi đã sửa (báo cáo mục 9)
-------------------------------
L10  Chỉ lấy bản đồ của cấp hiện tại nên quên cấp cũ. Nay từ cấp 2 trở đi, một
     nửa bản đồ mỗi thế hệ lấy từ các cấp thấp hơn (mix_lower=True). Việc chọn
     genome để xét lên cấp (top-k theo fitness THÔ, không theo điểm hybrid) do
     NEATRunner làm.
L34  Với ngưỡng 80%, run 1,5 triệu lượt chỉ lên cấp 3 ở thế hệ 540 và chưa bao giờ
     học cấp 4. Nay có thêm max_gens_per_level: ở một cấp quá số thế hệ này thì
     lên cấp, nên mọi run đều học tới cấp 4; lý do lên cấp được ghi lại.
L05  Khi tắt curriculum, code cũ chỉ học trên 20 bản đồ đầu của cấp 1. Nay dùng
     UniformSchedule: mỗi thế hệ lấy đều từ cả 4 cấp (k/4 bản đồ mỗi cấp).
L14  Mọi lần bốc bản đồ dùng bộ sinh số có seed.
L23  Thiếu bản đồ thì báo lỗi ngay thay vì lặng lẽ cho fitness 0,01.

Bản đồ được giữ dưới dạng ĐƯỜNG DẪN (worker tự nạp, xem src.neat_engine.parallel).
Mọi genome trong một thế hệ chạy trên cùng một bộ bản đồ.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

LEVELS: tuple[int, ...] = (1, 2, 3, 4)


def list_maps(data_dir: str | Path, split: str, level: int) -> list[str]:
    """Đường dẫn tuyệt đối các bản đồ JSON của một cấp, đã sắp xếp."""
    folder = Path(data_dir) / split / f"level_{level}"
    return [str(p.resolve()) for p in sorted(folder.glob("*.json"))]


def _load_split(data_dir, split: str, levels: Sequence[int]) -> dict[int, list[str]]:
    out = {lvl: list_maps(data_dir, split, lvl) for lvl in levels}
    missing = [f"{split}/level_{lvl}" for lvl, paths in out.items() if not paths]
    if missing:
        raise FileNotFoundError(
            f"Không thấy bản đồ trong {Path(data_dir).resolve()}: {', '.join(missing)}. "
            "Kiểm tra --data-dir (mặc định data/maps) hoặc chạy scripts/pregenerate_maps.py."
        )
    return out


class CurriculumManager:
    """Manages difficulty progression across 4 levels."""

    MAX_LEVEL: int = 4
    name = "curriculum"

    def __init__(
        self,
        data_dir: str | Path,
        eval_interval: int = 10,
        promotion_threshold: float = 0.80,
        rng: Optional[np.random.Generator] = None,
        mix_lower: bool = True,
        max_gens_per_level: Optional[int] = None,
        levels: Sequence[int] = LEVELS,
    ):
        self.data_dir = Path(data_dir)
        self.eval_interval = eval_interval
        self.promotion_threshold = promotion_threshold
        self.rng = rng if rng is not None else np.random.default_rng()
        self.mix_lower = mix_lower
        self.max_gens_per_level = max_gens_per_level
        self.levels = tuple(levels)
        self.MAX_LEVEL = max(self.levels)
        self.current_level: int = self.levels[0]
        self.highest_level: int = self.current_level
        self.gens_at_level: int = 0
        self.promotions: list[dict] = []

        self._train_paths = _load_split(data_dir, "train", self.levels)
        self._val_paths = _load_split(data_dir, "val", self.levels)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def get_train_maps(self, k: int = 8) -> list[str]:
        """Bốc k bản đồ cho thế hệ này (dùng chung cho cả quần thể).

        Cấp đầu tiên: k bản đồ của cấp đó. Từ cấp sau: k - k//2 bản đồ cấp hiện tại
        và k//2 bản đồ của các cấp thấp hơn (L10).
        """
        cur = self._train_paths[self.current_level]
        if self.current_level == self.levels[0] or not self.mix_lower or k < 2:
            return [str(p) for p in self.rng.choice(cur, size=min(k, len(cur)), replace=False)]
        lower = sum((self._train_paths[lvl] for lvl in self.levels if lvl < self.current_level), [])
        n_cur = k - k // 2
        return ([str(p) for p in self.rng.choice(cur, size=min(n_cur, len(cur)), replace=False)]
                + [str(p) for p in self.rng.choice(lower, size=min(k // 2, len(lower)), replace=False)])

    def tick(self):
        """Gọi một lần mỗi thế hệ để đếm số thế hệ ở cấp hiện tại."""
        self.gens_at_level += 1

    def get_val_maps(self) -> list[str]:
        """Return the fixed validation set for the current level."""
        return self._val_paths[self.current_level]

    def all_val_maps(self) -> dict[int, list[str]]:
        return dict(self._val_paths)

    def should_evaluate(self, generation: int) -> bool:
        """True if this generation should trigger a validation check."""
        return generation > 0 and generation % self.eval_interval == 0

    def check_promotion(self, success_rate: float, generation: int | None = None) -> Optional[str]:
        """Lên cấp nếu SR val ≥ ngưỡng, hoặc đã ở cấp này ≥ max_gens_per_level thế hệ (L34).

        Returns lý do lên cấp ("val_sr" / "max_gens") hoặc None.
        """
        if self.current_level >= self.MAX_LEVEL:
            return None
        if success_rate >= self.promotion_threshold:
            reason = "val_sr"
        elif self.max_gens_per_level and self.gens_at_level >= self.max_gens_per_level:
            reason = "max_gens"
        else:
            return None
        self.promotions.append({"generation": generation, "from": self.current_level,
                                "to": self.current_level + 1, "reason": reason, "val_sr": success_rate})
        self.current_level += 1
        self.highest_level = max(self.highest_level, self.current_level)
        self.gens_at_level = 0
        return reason

    @property
    def is_max_level(self) -> bool:
        return self.current_level >= self.MAX_LEVEL


class UniformSchedule:
    """Không curriculum (sửa L05): mỗi thế hệ bốc đều từ mọi cấp, cùng một bộ cho cả quần thể.

    Đây cũng là lịch học chung nên dùng khi so NEAT với GA, CMA-ES (L34).
    """

    name = "uniform"

    def __init__(
        self,
        data_dir: str | Path,
        rng: Optional[np.random.Generator] = None,
        levels: Sequence[int] = LEVELS,
    ):
        self.data_dir = Path(data_dir)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.levels = tuple(levels)
        self.current_level = max(self.levels)     # để ghi log: học mọi cấp
        self.highest_level = max(self.levels)
        self.promotions: list[dict] = []
        self._train_paths = _load_split(data_dir, "train", self.levels)
        self._val_paths = _load_split(data_dir, "val", self.levels)

    def get_train_maps(self, k: int = 8) -> list[str]:
        per = max(1, k // len(self.levels))
        maps: list[str] = []
        for lvl in self.levels:
            paths = self._train_paths[lvl]
            maps += [str(p) for p in self.rng.choice(paths, size=min(per, len(paths)), replace=False)]
        return maps

    def tick(self):
        pass

    def all_val_maps(self) -> dict[int, list[str]]:
        return dict(self._val_paths)

    def should_evaluate(self, generation: int) -> bool:
        return False

    def check_promotion(self, success_rate: float, generation: int | None = None) -> Optional[str]:
        return None

    @property
    def is_max_level(self) -> bool:
        return True
