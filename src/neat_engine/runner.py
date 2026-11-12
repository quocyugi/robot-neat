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

Các lỗi đã sửa (báo cáo mục 9)
-------------------------------
L02  Winner cũ là genome có điểm hybrid cao nhất mọi thế hệ; điểm này ồn và chuẩn
     hóa theo thế hệ nên winner M6 chỉ đạt 4% ở cấp 1. Nay cứ val_every thế hệ
     (và ở thế hệ cuối) lấy top-k genome theo fitness THÔ, chấm trên toàn bộ bản đồ
     val (4 cấp), giữ bản sao (deepcopy) của genome có SR val cao nhất. File kết
     quả chính là champion_genome.pkl (winner_genome.pkl giữ tên cũ, cùng nội dung).
L05  Tắt curriculum: học đều trên cả 4 cấp (UniformSchedule), không còn 20 bản đồ
     phòng trống.
L13  Đánh giá song song bằng multiprocessing (src.neat_engine.parallel), workers > 1.
L14  seed đặt cho random, numpy, neat.Population và mọi lần bốc bản đồ: cùng seed
     cho cùng log, không phụ thuộc số worker.
L21  is_recurrent=None: suy ra từ feed_forward của file config.
L30  Gọi unify_initial_hidden_keys() (không làm gì khi num_hidden = 0).
L34  Curriculum có giới hạn số thế hệ mỗi cấp; summary.json ghi cấp cao nhất đã học.
In ra chỉ dùng ký tự ASCII (λ, ★ làm lỗi UnicodeEncodeError trên console cp1252).
"""

from __future__ import annotations

import copy
import json
import pickle
import random
import shutil
import time
from pathlib import Path
from typing import Optional

import neat
import numpy as np

from src.env.geodesic import DEFAULT_CACHE_DIR
from src.neat_engine.curriculum import LEVELS, CurriculumManager, UniformSchedule, list_maps
from src.neat_engine.evaluator import max_episode_fitness
from src.neat_engine.neat_fixes import unify_initial_hidden_keys
from src.neat_engine.novelty import AdaptiveHybridManager, NoveltyArchive, behavior_descriptor
from src.neat_engine.parallel import ParallelEvaluator, neat_spec


class NEATRunner:
    """High-level training driver for NEAT-based robot navigation."""

    def __init__(
        self,
        config_path: str | Path,
        data_dir: str | Path,
        output_dir: str | Path = "output",
        use_novelty: bool = True,
        use_curriculum: bool = True,
        is_recurrent: Optional[bool] = None,
        maps_per_eval: int = 8,
        max_generations: int = 300,
        seed: int = 0,
        workers: int = 1,
        fixed_lambda: Optional[float] = None,
        val_every: int = 10,
        val_top_k: int = 5,
        novelty_space: str = "probe",
        progress: str = "geodesic",
        geo_cache_dir: Optional[str] = None,
        max_gens_per_level: Optional[int] | str = "auto",
        promotion_threshold: float = 0.80,
        verbose: bool = True,
    ):
        self.config_path = Path(config_path).resolve()
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_novelty = use_novelty
        self.use_curriculum = use_curriculum
        self.maps_per_eval = maps_per_eval
        self.max_generations = max_generations
        self.seed = int(seed)
        self.workers = workers
        self.val_every = val_every
        self.val_top_k = val_top_k
        self.novelty_space = novelty_space
        self.fixed_lambda = fixed_lambda
        self.progress = progress
        self.verbose = verbose
        if novelty_space not in ("probe", "population"):
            raise ValueError("novelty_space must be 'probe' or 'population'")

        # NEAT config
        self.neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(self.config_path),
        )
        # L21: suy ra kiểu mạng từ config thay vì mặc định True
        if is_recurrent is None:
            is_recurrent = not bool(self.neat_config.genome_config.feed_forward)
        self.is_recurrent = is_recurrent

        # L14: một bộ sinh số có seed cho mọi lựa chọn ngẫu nhiên của runner
        self.rng = np.random.default_rng(self.seed)

        # Lịch học: curriculum (L10, L34) hoặc đều trên 4 cấp (L05)
        if use_curriculum:
            if max_gens_per_level == "auto":
                max_gens_per_level = max(10, int(np.ceil(max_generations / 5)))
            self.schedule = CurriculumManager(
                data_dir, rng=self.rng, max_gens_per_level=max_gens_per_level,
                promotion_threshold=promotion_threshold,
            )
        else:
            self.schedule = UniformSchedule(data_dir, rng=self.rng)
        self.curriculum = self.schedule if use_curriculum else None   # tên cũ

        # Bản đồ val của cả 4 cấp, dùng để chọn champion (L02)
        val = self.schedule.all_val_maps()
        self.val_maps = [p for lvl in sorted(val) for p in val[lvl]]
        self.val_levels = [lvl for lvl in sorted(val) for _ in val[lvl]]

        # Novelty (L06–L09): bản đồ probe cố định = bản đồ train đầu tiên của mỗi cấp
        self.probe_maps = (
            [list_maps(data_dir, "train", lvl)[0] for lvl in LEVELS]
            if use_novelty and novelty_space == "probe" else []
        )
        # Thang cố định của novelty (L08): khoảng cách lớn nhất có thể giữa hai descriptor
        # 2K chiều trong hình hộp đơn vị, sqrt(2K), nên điểm novelty chuẩn hóa không bị cắt.
        n_desc_maps = len(self.probe_maps) or maps_per_eval
        self.archive = NoveltyArchive(rng=self.rng) if use_novelty else None
        self.hybrid_mgr = (
            AdaptiveHybridManager(fixed_lambda=fixed_lambda, fitness_scale=max_episode_fitness(progress),
                                  novelty_scale=float(np.sqrt(2 * n_desc_maps)))
            if use_novelty else None
        )

        self.opts = {"progress": progress, "geo_cache": geo_cache_dir or DEFAULT_CACHE_DIR}

        # Champion theo validation (L02)
        self.champion = None
        self.champion_key: Optional[tuple] = None
        self.champion_gen: Optional[int] = None
        self.champion_val: Optional[dict] = None

        # Logging
        self._gen: int = 0
        self.history: list[dict] = []
        self.train_episodes: int = 0
        self.wall_seconds: float = 0.0
        self._population = None
        self._evaluator: Optional[ParallelEvaluator] = None

    # ================================================================== #
    #  Main entry point                                                    #
    # ================================================================== #

    def run(self) -> neat.DefaultGenome:
        """Run the full evolutionary loop and return the validated champion."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        population = neat.Population(self.neat_config, seed=self.seed)
        unify_initial_hidden_keys(population, self.neat_config)   # L30, không làm gì khi num_hidden = 0
        self._population = population

        # Add reporters
        if self.verbose:
            population.add_reporter(neat.StdOutReporter(False))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)

        t0 = time.time()
        self._evaluator = ParallelEvaluator(self.workers)
        try:
            library_best = population.run(self._eval_genomes, n=self.max_generations)
            self._evaluator.close()
        finally:
            self._evaluator.__exit__(None, None, None)
        self.wall_seconds = time.time() - t0

        champion = self.champion if self.champion is not None else library_best
        self._save(champion, library_best)
        return champion

    # ================================================================== #
    #  Per-generation evaluation callback                                  #
    # ================================================================== #

    def _tasks(self, genomes: list, map_paths: list[str]) -> list[tuple]:
        cfg = str(self.config_path)
        return [(neat_spec(g, cfg, self.is_recurrent), map_paths, self.opts) for g in genomes]

    def _eval_genomes(
        self,
        genomes: list[tuple[int, neat.DefaultGenome]],
        config: neat.Config,
    ):
        """Evaluate all genomes in the population (called by neat-python)."""
        self._gen += 1
        gen = self._gen
        t0 = time.time()

        # ── Bản đồ của thế hệ này, dùng chung cho cả quần thể ─────────
        train_paths = self.schedule.get_train_maps(k=self.maps_per_eval)
        self.schedule.tick()
        n_train = len(train_paths)
        objs = [g for _, g in genomes]

        # ── Đánh giá song song (L13); bản đồ probe chạy cùng lượt ─────
        results = self._evaluator.map(self._tasks(objs, train_paths + self.probe_maps))
        self.train_episodes += len(objs) * n_train

        raw = np.array([np.mean([r["fitness"] for r in res[:n_train]]) for res in results])
        solved = np.array([np.mean([r["success"] for r in res[:n_train]]) for res in results])
        steps = np.array([np.mean([r["steps"] for r in res[:n_train]]) for res in results])
        collisions = np.array([any(r["collision"] for r in res[:n_train]) for res in results])

        # ── Novelty scoring & hybrid combination ──────────────────────
        if self.use_novelty:
            if self.probe_maps:      # L07: hành vi đo trên bản đồ probe cố định, giữ archive
                beh = np.array([behavior_descriptor([r["final_pos"] for r in res[n_train:]]) for res in results])
                use_archive = True
            else:                    # L07: chỉ so trong quần thể hiện tại (cùng bộ bản đồ)
                beh = np.array([behavior_descriptor([r["final_pos"] for r in res[:n_train]]) for res in results])
                use_archive = False
            novelty = self.archive.compute_novelty(beh, population_behaviors=beh, use_archive=use_archive)
            if use_archive:
                if self.archive.size == 0:
                    self.archive.calibrate(novelty)   # ngưỡng ban đầu theo đúng thang của descriptor
                for b, s in zip(beh, novelty):
                    self.archive.try_add(b, s)
                self.archive.end_generation()
            self.hybrid_mgr.update(float(np.max(raw)))
            final_scores = self.hybrid_mgr.combine_scores(raw, novelty, steps, collisions)
        else:
            final_scores = raw

        # ── Assign fitness ────────────────────────────────────────────
        for g, s in zip(objs, final_scores):
            g.fitness = float(s)

        # ── Curriculum: xét lên cấp bằng top-k theo fitness THÔ (L10) ──
        promotion = None
        level_val_sr = None
        if self.use_curriculum and self.schedule.should_evaluate(gen) and not self.schedule.is_max_level:
            top = self._top_k(objs, raw, self.val_top_k)
            res = self._evaluator.map(self._tasks(top, self.schedule.get_val_maps()))
            level_val_sr = float(max(np.mean([r["success"] for r in rr]) for rr in res))
            promotion = self.schedule.check_promotion(level_val_sr, generation=gen)
            if promotion and self.hybrid_mgr is not None:
                self.hybrid_mgr.reset()     # L09: thang fitness đổi khi lên cấp

        # ── Chọn champion trên val của cả 4 cấp (L02) ─────────────────
        champ_check = None
        if gen % self.val_every == 0 or gen == self.max_generations:
            champ_check = self._champion_check(objs, raw, gen)

        # ── Log ───────────────────────────────────────────────────────
        n_species = len(self._population.species.species) if self._population is not None else None
        gen_log = {
            "generation": gen,
            "level": self.schedule.current_level,
            "best_raw_fitness": float(np.max(raw)),
            "mean_raw_fitness": float(np.mean(raw)),
            "best_solved": float(np.max(solved)),
            "mean_solved": float(np.mean(solved)),
            "best_score": float(np.max(final_scores)),
            "mean_score": float(np.mean(final_scores)),
            "n_species": n_species,
            "archive_size": self.archive.size if self.archive else 0,
            "lambda": self.hybrid_mgr.current_lambda if self.hybrid_mgr else 1.0,
            "level_val_sr": level_val_sr,
            "promotion": promotion,
            "champion_check": champ_check,
            "seconds": round(time.time() - t0, 2),
        }
        self.history.append(gen_log)

        if self.verbose:
            lam = f"  lambda={gen_log['lambda']:.2f}" if self.use_novelty else ""
            promo = f"  PROMOTED -> L{self.schedule.current_level} ({promotion})" if promotion else ""
            champ = f"  val_SR={champ_check['best_val_sr']:.2f}" if champ_check else ""
            print(
                f"  Gen {gen:4d} | L{gen_log['level']} | best_raw={gen_log['best_raw_fitness']:.2f} "
                f"mean_raw={gen_log['mean_raw_fitness']:.2f} solved={gen_log['best_solved']:.2f} "
                f"species={n_species}{lam}{promo}{champ} | {gen_log['seconds']:.1f}s",
                flush=True,
            )

    # ================================================================== #
    #  Validation                                                          #
    # ================================================================== #

    @staticmethod
    def _top_k(genomes: list, raw: np.ndarray, k: int) -> list:
        order = np.argsort(-raw, kind="stable")[:k]
        return [genomes[i] for i in order]

    def _champion_check(self, genomes: list, raw: np.ndarray, gen: int) -> dict:
        """Chấm top-k (theo fitness thô) trên toàn bộ bản đồ val; giữ genome tốt nhất (L02)."""
        top = self._top_k(genomes, raw, self.val_top_k)
        res = self._evaluator.map(self._tasks(top, self.val_maps))
        best_here = -1.0
        levels = np.array(self.val_levels)
        for g, rr in zip(top, res):
            succ = np.array([r["success"] for r in rr], dtype=float)
            fit = float(np.mean([r["fitness"] for r in rr]))
            per_level = {int(lvl): round(float(succ[levels == lvl].mean()), 4) for lvl in sorted(set(self.val_levels))}
            score = float(succ.mean())
            best_here = max(best_here, score)
            key = (score, fit)
            if self.champion_key is None or key > self.champion_key:
                self.champion = copy.deepcopy(g)
                self.champion_key = key
                self.champion_gen = gen
                self.champion_val = per_level
        return {"best_val_sr": round(best_here, 4), "champion_generation": self.champion_gen,
                "champion_val_sr": round(self.champion_key[0], 4) if self.champion_key else None}

    # ================================================================== #
    #  Helpers                                                             #
    # ================================================================== #

    def _create_network(self, genome, config):
        if self.is_recurrent:
            return neat.nn.RecurrentNetwork.create(genome, config)
        return neat.nn.FeedForwardNetwork.create(genome, config)

    def _save(self, champion, library_best):
        out = self.output_dir
        for name, obj in (("champion_genome.pkl", champion), ("winner_genome.pkl", champion),
                          ("library_best_genome.pkl", library_best)):
            with open(out / name, "wb") as f:
                pickle.dump(obj, f)
        shutil.copyfile(self.config_path, out / "neat_config.ini")
        with open(out / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        summary = {
            "config": str(self.config_path),
            "seed": self.seed,
            "recurrent": self.is_recurrent,
            "novelty": self.use_novelty,
            "novelty_space": self.novelty_space if self.use_novelty else None,
            "fixed_lambda": self.fixed_lambda,
            "schedule": self.schedule.name,
            "progress": self.progress,
            "maps_per_eval": self.maps_per_eval,
            "pop_size": self.neat_config.pop_size,
            "generations": self._gen,
            "train_episodes": self.train_episodes,
            "highest_level_trained": self.schedule.highest_level,
            "promotions": self.schedule.promotions,
            "champion_generation": self.champion_gen,
            "champion_val": self.champion_val,
            "champion_val_mean": self.champion_key[0] if self.champion_key else None,
            "champion_nodes": len(champion.nodes) if champion is not None else None,
            "champion_connections": sum(1 for c in champion.connections.values() if c.enabled) if champion is not None else None,
            "wall_seconds": round(self.wall_seconds, 1),
            "workers": self.workers,
        }
        with open(out / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
