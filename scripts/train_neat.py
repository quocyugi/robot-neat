"""
Train NEAT — Main Entry Point
===============================
Launches the full NEAT evolution pipeline with configurable options.

Usage
-----
    python scripts/train_neat.py --config configs/neat_recurrent.ini --novelty --curriculum
    python scripts/train_neat.py --config configs/neat_feedforward.ini --no-novelty --no-curriculum
"""

import argparse
from pathlib import Path

from src.neat_engine.runner import NEATRunner


def main():
    parser = argparse.ArgumentParser(description="Train NEAT Robot Navigation")
    parser.add_argument(
        "--config", type=str, default="configs/neat_recurrent.ini",
        help="Path to NEAT config file",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/maps",
        help="Path to pre-generated map dataset",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/train",
        help="Output directory for results",
    )
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--maps-per-eval", type=int, default=4)
    parser.add_argument(
        "--novelty", dest="novelty", action="store_true", default=True,
    )
    parser.add_argument("--no-novelty", dest="novelty", action="store_false")
    parser.add_argument(
        "--curriculum", dest="curriculum", action="store_true", default=True,
    )
    parser.add_argument("--no-curriculum", dest="curriculum", action="store_false")
    parser.add_argument(
        "--recurrent", dest="recurrent", action="store_true", default=True,
    )
    parser.add_argument("--no-recurrent", dest="recurrent", action="store_false")

    args = parser.parse_args()

    runner = NEATRunner(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_novelty=args.novelty,
        use_curriculum=args.curriculum,
        is_recurrent=args.recurrent,
        maps_per_eval=args.maps_per_eval,
        max_generations=args.generations,
    )

    print("=" * 60)
    print("  NEAT Robot Navigation Training")
    print(f"  Config:     {args.config}")
    print(f"  Novelty:    {args.novelty}")
    print(f"  Curriculum: {args.curriculum}")
    print(f"  Recurrent:  {args.recurrent}")
    print(f"  Max Gen:    {args.generations}")
    print("=" * 60)

    winner = runner.run()
    print(f"\nTraining complete. Winner saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
