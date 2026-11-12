"""Run a list of train_fast.py experiments, at most MAX_JOBS at a time.

Usage: python queue.py <python_exe> <repo> <geo_dir> <results_dir> <max_jobs> <jobs.json>
jobs.json: {"name": ["--algo", "ga", ...], ...}
"""
import json
import os
import subprocess
import sys
import time

py, repo, geo, res, max_jobs, jobs_file = sys.argv[1:7]
max_jobs = int(max_jobs)
here = os.path.dirname(os.path.abspath(__file__))
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
jobs = list(json.load(open(jobs_file, encoding="utf-8")).items())
os.makedirs(res, exist_ok=True)
running, t0 = [], time.time()
while jobs or running:
    while jobs and len(running) < max_jobs:
        name, args = jobs.pop(0)
        out = os.path.join(res, name)
        if os.path.exists(os.path.join(out, "summary.json")):
            continue
        log = open(out + ".log", "w", encoding="utf-8")
        cmd = [py, "-u", "-X", "utf8", os.path.join(here, "train_fast.py"), "--repo", repo, "--geo-dir", geo, "--out", out] + args
        running.append((name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), time.time()))
        print(f"[{time.time()-t0:6.0f}s] start {name}", flush=True)
    time.sleep(2)
    keep = []
    for name, p, t in running:
        if p.poll() is None:
            keep.append((name, p, t))
        else:
            print(f"[{time.time()-t0:6.0f}s] done  {name} rc={p.returncode} in {time.time()-t:.0f}s", flush=True)
    running = keep
print("QUEUE DONE", round(time.time() - t0), flush=True)
