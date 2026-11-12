"""
Run Baselines — B1, B2 (GA mạng cố định), B3 (random search), B4 (luật), B5 (APF)
====================================================================================
Sửa lỗi L04: trước đây SimpleGA, random_search, RuleBasedController, APFController
không được gọi ở đâu. Script này huấn luyện và chấm các baseline theo CÙNG giao
thức với NEATRunner, để phép so Q1 công bằng:
  - cùng lịch học (--schedule uniform|curriculum, cùng lớp với NEAT, L34),
  - cùng số bản đồ mỗi thế hệ, dùng chung cho cả quần thể,
  - cùng fitness (tiến độ geodesic, L26), cùng ngân sách pop × thế hệ × bản đồ,
  - champion chọn trên val của cả 4 cấp (top-5 theo fitness, mỗi --val-every thế hệ),
  - chấm champion trên cùng các tập test (src.experiments.evaluation).

Usage
-----
    python scripts/run_baselines.py --baseline ga_b1 --seed 0 --generations 150 --workers 4
    python scripts/run_baselines.py --baseline ga_b2 --seed 0
    python scripts/run_baselines.py --baseline random --seed 0      # B3: mỗi thế hệ một quần thể ngẫu nhiên mới
    python scripts/run_baselines.py --baseline rule                 # B4: không học, chỉ chấm
    python scripts/run_baselines.py --baseline apf                  # B5: không học, chỉ chấm

Kết quả: output/baselines/<baseline>/seed_<seed>/ gồm champion_params.npz,
summary.json, training_history.json, test_results.json.
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.baselines.fixed_ga import SimpleGA  # noqa: E402
from src.env.geodesic import DEFAULT_CACHE_DIR  # noqa: E402
from src.experiments.evaluation import evaluate_sets, find_sets, format_report  # noqa: E402
from src.neat_engine.curriculum import CurriculumManager, UniformSchedule  # noqa: E402
from src.neat_engine.parallel import ParallelEvaluator, mlp_spec  # noqa: E402

LAYERS = {"ga_b1": [15, 8, 2], "ga_b2": [15, 16, 8, 2], "random": [15, 8, 2]}


def train(args, ev: ParallelEvaluator, out_dir: Path) -> tuple[np.ndarray, list[int], dict]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    layers = LAYERS[args.baseline]
    if args.schedule == "curriculum":
        schedule = CurriculumManager(args.data_dir, rng=rng, max_gens_per_level=max(10, math.ceil(args.generations / 5)))
    else:
        schedule = UniformSchedule(args.data_dir, rng=rng)
    val = schedule.all_val_maps()
    val_paths = [p for lvl in sorted(val) for p in val[lvl]]
    val_levels = np.array([lvl for lvl in sorted(val) for _ in val[lvl]])
    ga = SimpleGA(layers, pop_size=args.pop, rng=rng)
    opts = {"progress": "geodesic", "geo_cache": args.geo_cache or DEFAULT_CACHE_DIR}

    champion, champion_key, champion_gen, champion_val = None, None, None, None
    history, episodes, t_start = [], 0, time.time()
    for gen in range(1, args.generations + 1):
        t0 = time.time()
        if args.baseline == "random" and gen > 1:
            ga.population = rng.standard_normal(ga.population.shape) * 0.2   # B3: quần thể ngẫu nhiên mới
        maps = schedule.get_train_maps(args.maps_per_eval)
        schedule.tick()
        res = ev.map([(mlp_spec(p, layers), maps, opts) for p in ga.population])
        fit = np.array([np.mean([r["fitness"] for r in rr]) for rr in res])
        solved = np.array([np.mean([r["success"] for r in rr]) for rr in res])
        ga.fitnesses = fit
        episodes += len(ga.population) * len(maps)

        promotion = None
        if isinstance(schedule, CurriculumManager) and schedule.should_evaluate(gen) and not schedule.is_max_level:
            top = np.argsort(-fit, kind="stable")[:5]
            rr = ev.map([(mlp_spec(ga.population[i], layers), schedule.get_val_maps(), opts) for i in top])
            promotion = schedule.check_promotion(max(np.mean([r["success"] for r in x]) for x in rr), generation=gen)

        check = None
        if gen % args.val_every == 0 or gen == args.generations:
            top = np.argsort(-fit, kind="stable")[:5]
            rr = ev.map([(mlp_spec(ga.population[i], layers), val_paths, opts) for i in top])
            best_here = -1.0
            for i, x in zip(top, rr):
                succ = np.array([r["success"] for r in x], dtype=float)
                key = (float(succ.mean()), float(np.mean([r["fitness"] for r in x])))
                best_here = max(best_here, key[0])
                if champion_key is None or key > champion_key:
                    champion, champion_key, champion_gen = ga.population[i].copy(), key, gen
                    champion_val = {int(l): round(float(succ[val_levels == l].mean()), 4) for l in sorted(set(val_levels.tolist()))}
            check = round(best_here, 4)

        history.append({"generation": gen, "level": schedule.current_level, "best_fitness": float(fit.max()),
                        "mean_fitness": float(fit.mean()), "best_solved": float(solved.max()),
                        "promotion": promotion, "val_check": check, "seconds": round(time.time() - t0, 2)})
        print(f"  Gen {gen:4d} | L{schedule.current_level} | best={fit.max():.2f} mean={fit.mean():.2f} "
              f"solved={solved.max():.2f}" + (f"  PROMOTED ({promotion})" if promotion else "")
              + (f"  val_SR={check:.2f}" if check is not None else "") + f" | {time.time() - t0:.1f}s", flush=True)
        if args.baseline != "random":
            ga.evolve()

    np.savez(out_dir / "champion_params.npz", params=champion, layers=np.array(layers))
    summary = {"baseline": args.baseline, "layers": layers, "seed": args.seed, "schedule": schedule.name,
               "pop": args.pop, "generations": args.generations, "maps_per_eval": args.maps_per_eval,
               "train_episodes": episodes, "highest_level_trained": schedule.highest_level,
               "promotions": schedule.promotions, "champion_generation": champion_gen,
               "champion_val": champion_val, "champion_val_mean": champion_key[0] if champion_key else None,
               "wall_seconds": round(time.time() - t_start, 1)}
    with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return champion, layers, summary


def main():
    from src.console import utf8_stdout
    utf8_stdout()
    ap = argparse.ArgumentParser(description="Huấn luyện và chấm các baseline B1–B5")
    ap.add_argument("--baseline", required=True, choices=["ga_b1", "ga_b2", "random", "rule", "apf"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generations", type=int, default=150)
    ap.add_argument("--pop", type=int, default=150)
    ap.add_argument("--maps-per-eval", type=int, default=8)
    ap.add_argument("--schedule", choices=["uniform", "curriculum"], default="uniform")
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data-dir", default="data/maps")
    ap.add_argument("--geo-cache", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    trained = args.baseline in LAYERS
    out_dir = Path(args.output_dir or (f"output/baselines/{args.baseline}" + (f"/seed_{args.seed}" if trained else "")))
    out_dir.mkdir(parents=True, exist_ok=True)
    sets = find_sets(args.data_dir)

    with ParallelEvaluator(args.workers) as ev:
        if trained:
            params, layers, summary = train(args, ev, out_dir)
            spec = mlp_spec(params, layers)
        else:
            spec, summary = {"kind": args.baseline}, {"baseline": args.baseline}
        report = evaluate_sets(spec, sets, evaluator=ev)
        ev.close()

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump({"policy": args.baseline, "results": report}, f, indent=2, ensure_ascii=False)
    print(f"\n{args.baseline} (seed {args.seed})")
    print(format_report(report))
    print(f"\nĐã ghi {out_dir}")


if __name__ == "__main__":
    main()
