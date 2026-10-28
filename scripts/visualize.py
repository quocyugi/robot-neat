"""
Visualise — Render a genome navigating a map
==============================================
Loads a saved genome and replays it on a given map with Pygame rendering.

Usage
-----
    python scripts/visualize.py --genome output/train/winner_genome.pkl --map data/maps/test/level_4/map_200050.json
"""

import argparse
import json
import pickle
import time

import neat
import numpy as np

from src.env.robot_env import RobotNavEnv


def main():
    parser = argparse.ArgumentParser(description="Visualise NEAT agent")
    parser.add_argument("--genome", type=str, required=True)
    parser.add_argument("--map", type=str, required=True)
    parser.add_argument(
        "--config", type=str, default="configs/neat_recurrent.ini",
    )
    parser.add_argument("--recurrent", action="store_true", default=True)
    parser.add_argument("--no-recurrent", dest="recurrent", action="store_false")
    args = parser.parse_args()

    # Load genome
    with open(args.genome, "rb") as f:
        genome = pickle.load(f)

    # Load NEAT config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        args.config,
    )

    # Create network
    if args.recurrent:
        net = neat.nn.RecurrentNetwork.create(genome, config)
    else:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Load map
    with open(args.map, "r", encoding="utf-8") as f:
        map_data = json.load(f)

    # Run with rendering
    env = RobotNavEnv(map_data=map_data, render_mode="human")
    obs, _ = env.reset()

    if args.recurrent and hasattr(net, "reset"):
        net.reset()

    terminated = truncated = False
    while not (terminated or truncated):
        action = np.array(net.activate(obs.tolist()), dtype=np.float32)
        obs, _, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.05)

    status = "✓ GOAL!" if info["is_success"] else ("✗ COLLISION" if info["is_collision"] else "— TIMEOUT")
    print(f"\nResult: {status}  |  Steps: {info['steps']}  |  Dist: {info['dist_to_goal']:.2f}m")

    time.sleep(2)
    env.close()


if __name__ == "__main__":
    main()
