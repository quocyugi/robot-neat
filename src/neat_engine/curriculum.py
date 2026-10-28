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
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np


class CurriculumManager:
    """Manages difficulty progression across 4 levels."""

    MAX_LEVEL: int = 4

    def __init__(
        self,
        data_dir: str | Path,
        eval_interval: int = 10,
        promotion_threshold: float = 0.80,
    ):
        self.data_dir = Path(data_dir)
        self.eval_interval = eval_interval
        self.promotion_threshold = promotion_threshold
        self.current_level: int = 1

        # Pre-load validation maps per level
        self._val_maps: dict[int, list[dict]] = {}
        self._train_maps: dict[int, list[dict]] = {}
        for lvl in range(1, self.MAX_LEVEL + 1):
            self._val_maps[lvl] = self._load_maps(lvl, "val")
            self._train_maps[lvl] = self._load_maps(lvl, "train")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def get_train_maps(self, k: int = 4) -> list[dict]:
        """Sample *k* training maps for the current level.

        Each genome is evaluated on K=4 random maps per generation
        (overview.md §4.1).
        """
        maps = self._train_maps.get(self.current_level, [])
        if not maps:
            return []
        indices = np.random.choice(len(maps), size=min(k, len(maps)), replace=False)
        return [maps[i] for i in indices]

    def get_val_maps(self) -> list[dict]:
        """Return the fixed validation set for the current level."""
        return self._val_maps.get(self.current_level, [])

    def should_evaluate(self, generation: int) -> bool:
        """True if this generation should trigger a validation check."""
        return generation > 0 and generation % self.eval_interval == 0

    def check_promotion(self, success_rate: float) -> bool:
        """Attempt to promote to the next level.

        Returns True if promotion occurred.
        """
        if (
            success_rate >= self.promotion_threshold
            and self.current_level < self.MAX_LEVEL
        ):
            self.current_level += 1
            return True
        return False

    @property
    def is_max_level(self) -> bool:
        return self.current_level >= self.MAX_LEVEL

    # ------------------------------------------------------------------ #
    #  Data loading                                                        #
    # ------------------------------------------------------------------ #

    def _load_maps(self, level: int, split: str) -> list[dict]:
        """Load all JSON maps for a given level and split."""
        folder = self.data_dir / split / f"level_{level}"
        if not folder.exists():
            return []
        maps = []
        for fp in sorted(folder.glob("*.json")):
            with open(fp, "r", encoding="utf-8") as f:
                maps.append(json.load(f))
        return maps
