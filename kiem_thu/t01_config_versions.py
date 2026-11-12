"""T01 - Parse both repos' NEAT configs against a given neat-python install.

Usage: python t01_config_versions.py <repo_root> [<repo_root> ...]
Prints one line per (repo, config) with OK or the exception text.
"""
import os
import sys
import traceback

import neat

try:
    from importlib.metadata import version
    v = version("neat-python")
except Exception:
    v = getattr(neat, "__version__", "?")
print(f"neat-python from {os.path.dirname(neat.__file__)} version={v}")
for repo in sys.argv[1:]:
    name = os.path.basename(repo)
    for cfg in ("configs/neat_feedforward.ini", "configs/neat_recurrent.ini"):
        path = os.path.join(repo, cfg)
        try:
            c = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation, path)
            print(f"  OK    {name}/{cfg}  pop={c.pop_size} thr={c.fitness_threshold}")
        except Exception as e:
            print(f"  FAIL  {name}/{cfg}  {type(e).__name__}: {e}")
