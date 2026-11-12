"""T13 - How many species does NEAT form for several compatibility thresholds?

No simulator is involved: fitness is random, so only mutation-driven genetic drift is measured.
Usage: python t13_speciation_probe.py <repo_root> <out_json>
"""
import json
import os
import random
import re
import sys
import tempfile

REPO, OUT = sys.argv[1], sys.argv[2]
import neat

src = open(os.path.join(REPO, "configs", "neat_recurrent.ini"), encoding="utf-8").read()
src = re.sub(r"no_fitness_termination\s*=\s*\w+", "no_fitness_termination = True", src)
res = {}
for thr in (3.0, 2.5, 2.0, 1.5, 1.2, 1.0):
    txt = re.sub(r"compatibility_threshold\s*=\s*[\d.]+", f"compatibility_threshold = {thr}", src)
    with tempfile.NamedTemporaryFile("w", suffix=".ini", delete=False, encoding="utf-8") as f:
        f.write(txt)
        path = f.name
    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                      neat.DefaultStagnation, path)
    series = []
    pop = neat.Population(cfg, seed=0)
    series.append(len(pop.species.species))

    class Rec(neat.reporting.BaseReporter):
        def post_evaluate(self, config, population, species, best):
            series.append(len(species.species))

    pop.add_reporter(Rec())

    def ev(genomes, config):
        for _, g in genomes:
            g.fitness = random.random() + 1.0

    pop.run(ev, n=40)
    os.unlink(path)
    res[str(thr)] = {"gen0": series[0], "gen10": series[10], "gen20": series[20], "gen40": series[-1],
                     "min": min(series), "max": max(series)}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1))
