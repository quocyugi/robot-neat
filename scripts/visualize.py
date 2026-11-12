"""
Visualise — Render a genome navigating a map
==============================================
Loads a saved genome and replays it on a given map with Pygame rendering.

Usage
-----
    python scripts/visualize.py --genome output/train/champion_genome.pkl --map data/maps/test/level_4/map_200000.json

Nếu không có --config, dùng neat_config.ini nằm cạnh file genome (runner lưu sẵn),
nếu không có thì configs/neat_recurrent.ini. Kiểu mạng (hồi quy hay truyền thẳng)
suy ra từ feed_forward của config (lỗi L21); --recurrent / --no-recurrent để ép.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

# Cho phép chạy "python scripts/visualize.py" từ thư mục gốc dự án
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import neat
import numpy as np

from src.env.robot_env import RobotNavEnv


def main():
    from src.console import utf8_stdout
    utf8_stdout()
    parser = argparse.ArgumentParser(description="Visualise NEAT agent")
    parser.add_argument("--genome", type=str, required=True)
    parser.add_argument("--map", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--recurrent", dest="recurrent", action="store_true", default=None)
    parser.add_argument("--no-recurrent", dest="recurrent", action="store_false")
    args = parser.parse_args()

    # Load genome
    with open(args.genome, "rb") as f:
        genome = pickle.load(f)

    config_path = args.config
    if config_path is None:
        beside = Path(args.genome).with_name("neat_config.ini")
        config_path = str(beside) if beside.exists() else "configs/neat_recurrent.ini"

    # Load NEAT config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    recurrent = args.recurrent if args.recurrent is not None else not config.genome_config.feed_forward

    # Create network
    if recurrent:
        net = neat.nn.RecurrentNetwork.create(genome, config)
    else:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Load map
    with open(args.map, "r", encoding="utf-8") as f:
        map_data = json.load(f)

    # Run with rendering
    env = RobotNavEnv(map_data=map_data, render_mode="human")
    obs, _ = env.reset()

    if recurrent and hasattr(net, "reset"):
        net.reset()

    terminated = truncated = False
    while not (terminated or truncated):
        action = np.array(net.activate(obs.tolist()), dtype=np.float32)
        obs, _, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.05)

    status = "GOAL" if info["is_success"] else ("COLLISION" if info["is_collision"] else "TIMEOUT")
    print(f"\nResult: {status}  |  Steps: {info['steps']}  |  Dist: {info['dist_to_goal']:.2f} m")

    time.sleep(2)
    env.close()


if __name__ == "__main__":
    main()
