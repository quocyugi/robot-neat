"""Score saved champions (champion.pkl + summary.json of train_fast.py runs) on any map folders.

Usage: python d31_eval_champ.py <repo> <map_root> <rels_csv> <out_json> <run_dir> [<run_dir> ...]
  rels_csv: folders under map_root, e.g. test/level_5,test/level_3
"""
import glob
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO, ROOT, RELS, OUT = sys.argv[1:5]
RUNS = sys.argv[5:]
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import fastsim as fs  # noqa: E402
from train_fast import summarize  # noqa: E402


def make_policy(run, payload, s):
    if s["algo"] == "neat":
        import neat
        cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, os.path.join(run, "patched.ini"))
        if s["net"] == "rnn":
            return lambda: fs.BatchRecurrentPolicy([payload], cfg)
        net = neat.nn.FeedForwardNetwork
        return lambda: fs.NetsPolicy([net.create(payload, cfg)])
    layers = [s["n_rays"] + 3] + [int(x) for x in str(s["layers"]).split(",")] + [2]
    if s.get("arch") == "elman":
        return lambda: fs.ElmanBatch(payload[None, :], layers)
    return lambda: fs.MLPBatch(payload[None, :], layers)


def main():
    maps = {}
    for rel in RELS.split(","):
        maps[rel] = [fs.load_json(p) for p in sorted(glob.glob(os.path.join(ROOT, rel, "*.json")))]
    out = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    for run in RUNS:
        s = json.load(open(os.path.join(run, "summary.json"), encoding="utf-8"))
        payload = pickle.load(open(os.path.join(run, "champion.pkl"), "rb"))
        mk = make_policy(run, payload, s)
        row = {}
        for rel, ms in maps.items():
            res = []
            for m in ms:
                fm = fs.FastMap(m, m["map_id"], n_rays=s["n_rays"])
                r = fs.rollout(fm, mk(), 1, max_steps=s["max_steps"], track_length=True)
                res.append((r.succ, r.col, r.steps, r.length, fm.l_opt))
            row[rel] = summarize(res)
        name = os.path.basename(os.path.normpath(run))
        out[name] = {"algo": s["algo"], "train_evals": s["train_evals"], "map_root_trained": s.get("levels", "1,2,3,4"), "scores": row}
        print(name, json.dumps({k: v["SR"] for k, v in row.items()}), flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
