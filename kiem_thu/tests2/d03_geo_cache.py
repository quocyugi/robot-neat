"""Pre-compute the geodesic (distance-to-go) field of every map.

Usage: python d03_geo_cache.py <repo_root> <map_root> <geo_dir> <workers>
"""
import glob
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def work(args):
    sys.path.insert(0, HERE)
    import fastsim as fs
    root, key, geo_dir = args
    out = os.path.join(geo_dir, key.replace("/", "__") + ".npy")
    if os.path.exists(out):
        return 0
    m = fs.load_json(os.path.join(root, key))
    np.save(out, fs.compute_geodesic(m["segments"], m["goal"]))
    return 1


def main():
    repo, root, geo_dir, workers = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    os.makedirs(geo_dir, exist_ok=True)
    keys = []
    for rel in ("train", "val", "test", "test_barn", "test_lehman"):
        for p in glob.glob(os.path.join(root, rel, "**", "*.json"), recursive=True):
            keys.append(os.path.relpath(p, root).replace("\\", "/"))
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        done = sum(pool.imap_unordered(work, [(root, k, geo_dir) for k in keys], chunksize=8))
    print(f"maps={len(keys)} computed={done} seconds={time.time()-t0:.0f}", flush=True)


if __name__ == "__main__":
    main()
