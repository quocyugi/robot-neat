"""
NEAT Evolution Runner — Main Training Orchestrator
====================================================
Wires together:
  neat-python  ←→  RobotNavEnv  ←→  Evaluator  ←→  Novelty  ←→  Curriculum

Supports 4 experiment variants (ablation matrix M1–M6):
  - Feedforward / Recurrent
  - Distance-only / Hybrid Novelty / Pure Novelty
  - With / without Curriculum Learning

Reference
---------
- brainstorm/04_experiments/ablation_design.md (Decision #26)
- brainstorm/02_neat/config_tuning.md
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Optional

import neat
import numpy as np

from src.env.robot_env import RobotNavEnv
from src.neat_engine.evaluator import (
    compute_episode_fitness,
    evaluate_genome_on_maps,
)
from src.neat_engine.novelty import AdaptiveHybridManager, NoveltyArchive
from src.neat_engine.curriculum import CurriculumManager


class NEATRunner:
    """High-level training driver for NEAT-based robot navigation."""

    def __init__(
        self,
        config_path: str | Path,
        data_dir: str | Path,
        output_dir: str | Path = "output",
        use_novelty: bool = True,
        use_curriculum: bool = True,
        is_recurrent: bool = False,
        maps_per_eval: int = 4,
        max_generations: int = 300,
    ):
        self.config_path = Path(config_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_novelty = use_novelty
        self.use_curriculum = use_curriculum
        self.is_recurrent = is_recurrent
        self.maps_per_eval = maps_per_eval
        self.max_generations = max_generations

        # NEAT config
        self.neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(self.config_path),
        )

        # Modules
        self.curriculum = CurriculumManager(data_dir) if use_curriculum else None
        self.archive = NoveltyArchive() if use_novelty else None
        self.hybrid_mgr = AdaptiveHybridManager() if use_novelty else None

        # Logging
        self._gen: int = 0
        self.history: list[dict] = []

    # ================================================================== #
    #  Main entry point                                                    #
    # ================================================================== #

    def run(self) -> neat.DefaultGenome:
        """Run the full evolutionary loop and return the best genome."""
        population = neat.Population(self.neat_config)

        # Add reporters
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)

        # Run evolution
        winner = population.run(
            self._eval_genomes, n=self.max_generations,
        )

        # Save winner
        self._save_winner(winner)
        self._save_history()

        return winner

    # ================================================================== #
    #  Per-generation evaluation callback                                  #
    # ================================================================== #

    def _eval_genomes(
        self,
        genomes: list[tuple[int, neat.DefaultGenome]],
        config: neat.Config,
    ):
        """Evaluate all genomes in the population (called by neat-python)."""
        self._gen += 1

        # ── Select training maps ──────────────────────────────────────
        if self.curriculum:
            train_maps = self.curriculum.get_train_maps(k=self.maps_per_eval)
        else:
            # Without curriculum: use level-1 maps only (fallback)
            train_maps = self._load_fallback_maps()

        if not train_maps:
            # Safety: assign minimal fitness if no maps available
            for gid, genome in genomes:
                genome.fitness = 0.01
            return

        # ── Evaluate each genome ──────────────────────────────────────
        raw_fitnesses = np.zeros(len(genomes))
        behaviors = np.zeros((len(genomes), 2))
        all_steps = np.zeros(len(genomes))
        all_collisions = np.zeros(len(genomes), dtype=bool)

        for idx, (gid, genome) in enumerate(genomes):
            net = self._create_network(genome, config)
            mean_fit, infos = evaluate_genome_on_maps(
                net, train_maps, RobotNavEnv, self.is_recurrent,
            )
            raw_fitnesses[idx] = mean_fit

            # Behavior descriptor: average final position across maps
            final_positions = [np.array(info["final_pos"]) for info in infos]
            behaviors[idx] = np.mean(final_positions, axis=0)

            # Collect episode stats for MCNS
            all_steps[idx] = np.mean([info["steps"] for info in infos])
            all_collisions[idx] = any(info["is_collision"] for info in infos)

        # ── Novelty scoring & hybrid combination ──────────────────────
        if self.use_novelty and self.archive and self.hybrid_mgr:
            novelty_scores = self.archive.compute_novelty(
                behaviors, population_behaviors=behaviors,
            )

            # Update archive
            for i in range(len(genomes)):
                self.archive.try_add(behaviors[i], novelty_scores[i])

            # Adaptive lambda
            self.hybrid_mgr.update(float(np.max(raw_fitnesses)))

            # Combine
            final_scores = self.hybrid_mgr.combine_scores(
                raw_fitnesses, novelty_scores,
                all_steps, all_collisions,
            )

            self.archive.end_generation()
        else:
            final_scores = raw_fitnesses

        # ── Assign fitness ────────────────────────────────────────────
        for idx, (gid, genome) in enumerate(genomes):
            genome.fitness = float(final_scores[idx])

        # ── Curriculum check ──────────────────────────────────────────
        promoted = False
        val_sr = None
        if self.curriculum and self.curriculum.should_evaluate(self._gen):
            val_sr = self._validate_champion(genomes, config)
            promoted = self.curriculum.check_promotion(val_sr)

        # ── Log ───────────────────────────────────────────────────────
        best_fit = float(np.max(final_scores))
        mean_fit = float(np.mean(final_scores))
        gen_log = {
            "generation": self._gen,
            "best_fitness": best_fit,
            "mean_fitness": mean_fit,
            "level": self.curriculum.current_level if self.curriculum else 1,
            "archive_size": self.archive.size if self.archive else 0,
            "lambda": self.hybrid_mgr.current_lambda if self.hybrid_mgr else 1.0,
            "val_sr": val_sr,
            "promoted": promoted,
        }
        self.history.append(gen_log)

        level_str = f"Lv{gen_log['level']}"
        lambda_str = f"λ={gen_log['lambda']:.2f}" if self.use_novelty else ""
        promo_str = " ★ PROMOTED!" if promoted else ""
        print(
            f"  Gen {self._gen:4d} | {level_str} | "
            f"Best={best_fit:.3f}  Mean={mean_fit:.3f}  "
            f"Archive={gen_log['archive_size']}  {lambda_str}{promo_str}"
        )

    # ================================================================== #
    #  Validation                                                          #
    # ================================================================== #

    def _validate_champion(
        self,
        genomes: list[tuple[int, neat.DefaultGenome]],
        config: neat.Config,
    ) -> float:
        """Evaluate the best genome on the validation set.

        Returns success rate ∈ [0, 1].
        """
        # Find champion
        best_genome = max(genomes, key=lambda g: g[1].fitness)[1]
        net = self._create_network(best_genome, config)

        val_maps = self.curriculum.get_val_maps() if self.curriculum else []
        if not val_maps:
            return 0.0

        successes = 0
        for map_data in val_maps:
            env = RobotNavEnv(map_data=map_data)
            obs, _ = env.reset()
            if self.is_recurrent and hasattr(net, "reset"):
                net.reset()

            terminated = truncated = False
            while not (terminated or truncated):
                action = np.array(net.activate(obs.tolist()), dtype=np.float32)
                obs, _, terminated, truncated, info = env.step(action)

            if info["is_success"]:
                successes += 1
            env.close()

        return successes / len(val_maps)

    # ================================================================== #
    #  Helpers                                                             #
    # ================================================================== #

    def _create_network(self, genome, config):
        if self.is_recurrent:
            return neat.nn.RecurrentNetwork.create(genome, config)
        return neat.nn.FeedForwardNetwork.create(genome, config)

    def _load_fallback_maps(self) -> list[dict]:
        """Load a few level-1 maps when curriculum is disabled."""
        folder = self.data_dir / "train" / "level_1"
        if not folder.exists():
            return []
        maps = []
        for fp in sorted(folder.glob("*.json"))[:20]:
            with open(fp, "r", encoding="utf-8") as f:
                maps.append(json.load(f))
        if maps:
            idx = np.random.choice(len(maps), size=min(4, len(maps)), replace=False)
            return [maps[i] for i in idx]
        return []

    def _save_winner(self, genome):
        path = self.output_dir / "winner_genome.pkl"
        with open(path, "wb") as f:
            pickle.dump(genome, f)

    def _save_history(self):
        path = self.output_dir / "training_history.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
