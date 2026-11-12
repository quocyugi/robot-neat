"""Reproduce the Lehman & Stanley maze comparison inside the team's simulator.

A single maze; a run stops as soon as any genome reaches the goal.
method:  fit (team's fitness) | ns (novelty only) | hybrid (team's adaptive scheme)
env:     terminate  = project behaviour (a collision ends the episode)
         slide      = closer to the paper (robot keeps its previous pose and the episode continues)

Usage: python run_maze.py --repo R --maze hard --method ns --env slide --steps 750 --seed 0 --gens 400 --workers 4 --out FILE
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time

import numpy as np

_W = {}


def init_worker(repo, cfg_path, map_path, env_mode, steps):
    sys.path.insert(0, repo)
    import neat
    import src.env.robot_env as re_mod
    from src.env.robot_env import RobotNavEnv
    from src.neat_engine.evaluator import compute_episode_fitness
    re_mod.MAX_STEPS = steps

    class SlideEnv(RobotNavEnv):
        def step(self, action):
            prev_pos, prev_th = self.pos.copy(), self.theta
            obs, r, term, trunc, info = super().step(action)
            if info["is_collision"] and not info["is_success"]:
                self.pos = prev_pos
                self.trajectory[-1] = prev_pos.copy()
                obs = self._get_obs()
                d = float(np.linalg.norm(self.pos - self.goal_pos))
                info = dict(info, is_collision=False, final_pos=(float(prev_pos[0]), float(prev_pos[1])), dist_to_goal=d)
                term = False
                trunc = self.step_count >= steps
            return obs, r, term, trunc, info

    with open(map_path, encoding="utf-8") as f:
        _W["map"] = json.load(f)
    _W["Env"] = SlideEnv if env_mode == "slide" else RobotNavEnv
    _W["fit"] = compute_episode_fitness
    _W["steps"] = steps
    _W["neat"] = neat
    _W["cfg"] = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                            neat.DefaultStagnation, cfg_path)


def worker(genome):
    neat = _W["neat"]
    net = neat.nn.RecurrentNetwork.create(genome, _W["cfg"])
    env = _W["Env"](map_data=_W["map"])
    obs, _ = env.reset()
    net.reset()
    term = trunc = False
    info = None
    while not (term or trunc):
        act = np.array(net.activate(obs.tolist()), dtype=np.float32)
        obs, _, term, trunc, info = env.step(act)
    return (float(_W["fit"](info, max_steps=_W["steps"])), bool(info["is_success"]), bool(info["is_collision"]),
            int(info["steps"]), info["final_pos"], float(info["min_distance_to_goal"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--maze", choices=["hard", "medium"], default="hard")
    ap.add_argument("--method", choices=["fit", "ns", "hybrid"], default="ns")
    ap.add_argument("--env", choices=["terminate", "slide"], default="terminate")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gens", type=int, default=400)
    ap.add_argument("--pop", type=int, default=150)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, a.repo)
    random.seed(a.seed)
    np.random.seed(a.seed)
    import neat
    from src.neat_engine.novelty import AdaptiveHybridManager, NoveltyArchive

    cfg_src = os.path.join(a.repo, "configs", "neat_recurrent.ini")
    txt = open(cfg_src, encoding="utf-8").read()
    txt = re.sub(r"no_fitness_termination\s*=\s*\w+", "no_fitness_termination  = True", txt)
    txt = re.sub(r"pop_size\s*=\s*\d+", f"pop_size                = {a.pop}", txt)
    cfg_path = a.out + ".ini"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(txt)
    map_path = os.path.join(a.repo, "data", "maps", "test_lehman", f"lehman_{a.maze}_maze.json")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                         neat.DefaultStagnation, cfg_path)
    pop = neat.Population(config, seed=a.seed)
    archive = NoveltyArchive()
    mgr = AdaptiveHybridManager()
    state = {"gen": 0, "solved_gen": None, "best_min_d": 1e9, "t": time.time(), "evals": 0}

    class Solved(Exception):
        pass

    with mp.get_context("spawn").Pool(a.workers, initializer=init_worker,
                                      initargs=(a.repo, cfg_path, map_path, a.env, a.steps)) as pool:
        def eval_genomes(genomes, cfg):
            state["gen"] += 1
            gs = [g for _, g in genomes]
            res = pool.map(worker, gs, chunksize=max(1, len(gs) // (a.workers * 4)))
            state["evals"] += len(gs)
            raw = np.array([r[0] for r in res])
            beh = np.array([r[4] for r in res], dtype=float)
            state["best_min_d"] = min(state["best_min_d"], float(min(r[5] for r in res)))
            if a.method == "fit":
                final = raw
            else:
                nov = archive.compute_novelty(beh, population_behaviors=beh)
                for i in range(len(gs)):
                    archive.try_add(beh[i], float(nov[i]))
                archive.end_generation()
                if a.method == "ns":
                    final = np.maximum(nov, 0.01)
                else:
                    mgr.update(float(raw.max()))
                    final = mgr.combine_scores(raw, nov, np.array([r[3] for r in res]),
                                               np.array([r[2] for r in res], dtype=bool))
            for g, f in zip(gs, final):
                g.fitness = float(f)
            if any(r[1] for r in res):
                state["solved_gen"] = state["gen"]
                raise Solved()

        try:
            pop.run(eval_genomes, n=a.gens)
        except Solved:
            pass

    out = {"maze": a.maze, "method": a.method, "env": a.env, "steps": a.steps, "seed": a.seed, "pop": a.pop,
           "gens_limit": a.gens, "solved": state["solved_gen"] is not None, "solved_gen": state["solved_gen"],
           "evals_used": state["evals"], "best_min_dist": round(state["best_min_d"], 3),
           "archive_size": archive.size, "seconds": round(time.time() - state["t"], 1)}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
