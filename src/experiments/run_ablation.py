"""
Ablation Study Runner — M1 to M6 Experiment Configurations
=============================================================
Automates running all 6 ablation variants and collecting metrics.

Reference
---------
- brainstorm/04_experiments/ablation_design.md (Decision #26)
"""

from __future__ import annotations

import json
from pathlib import Path

from src.neat_engine.runner import NEATRunner

# ── Ablation matrix definition ───────────────────────────────────────────
ABLATION_CONFIGS: dict[str, dict] = {
    "M1": {
        "description": "FF + Distance Fitness + No Curriculum (baseline floor)",
        "config_file": "configs/neat_feedforward.ini",
        "is_recurrent": False,
        "use_novelty": False,
        "use_curriculum": False,
    },
    "M2": {
        "description": "Recurrent + Distance Fitness + Curriculum",
        "config_file": "configs/neat_recurrent.ini",
        "is_recurrent": True,
        "use_novelty": False,
        "use_curriculum": True,
    },
    "M3": {
        "description": "FF + Hybrid Novelty + Curriculum",
        "config_file": "configs/neat_feedforward.ini",
        "is_recurrent": False,
        "use_novelty": True,
        "use_curriculum": True,
    },
    "M4": {
        "description": "Recurrent + Pure Novelty + Curriculum",
        "config_file": "configs/neat_recurrent.ini",
        "is_recurrent": True,
        "use_novelty": True,  # pure novelty = lambda fixed at 0
        "use_curriculum": True,
    },
    "M5": {
        "description": "Recurrent + Hybrid Novelty + No Curriculum",
        "config_file": "configs/neat_recurrent.ini",
        "is_recurrent": True,
        "use_novelty": True,
        "use_curriculum": False,
    },
    "M6": {
        "description": "FULL MODEL: Recurrent + Hybrid Novelty + Curriculum",
        "config_file": "configs/neat_recurrent.ini",
        "is_recurrent": True,
        "use_novelty": True,
        "use_curriculum": True,
    },
}


def run_ablation(
    data_dir: str = "data/maps",
    output_root: str = "output/ablation",
    max_generations: int = 300,
    n_seeds: int = 10,
):
    """Run all ablation experiments across multiple seeds."""
    output_root = Path(output_root)

    for model_id, cfg in ABLATION_CONFIGS.items():
        for seed in range(n_seeds):
            run_dir = output_root / model_id / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"  {model_id} — Seed {seed} — {cfg['description']}")
            print(f"{'='*60}\n")

            runner = NEATRunner(
                config_path=cfg["config_file"],
                data_dir=data_dir,
                output_dir=str(run_dir),
                use_novelty=cfg["use_novelty"],
                use_curriculum=cfg["use_curriculum"],
                is_recurrent=cfg["is_recurrent"],
                max_generations=max_generations,
            )

            try:
                runner.run()
            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                error_log = run_dir / "error.txt"
                error_log.write_text(str(e))


if __name__ == "__main__":
    run_ablation()
