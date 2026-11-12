"""Run a list of commands, at most MAX_JOBS at a time. A job is skipped when its result file already exists.

Usage: python queue_cmds.py <max_jobs> <jobs.json>
jobs.json: {"name": {"cmd": [...], "out": "result_file", "log": "log_file"}, ...}
"""
import json
import os
import subprocess
import sys
import time

max_jobs, jobs_file = int(sys.argv[1]), sys.argv[2]
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
jobs = list(json.load(open(jobs_file, encoding="utf-8")).items())
running, t0 = [], time.time()
while jobs or running:
    while jobs and len(running) < max_jobs:
        name, j = jobs.pop(0)
        if os.path.exists(j["out"]):
            continue
        os.makedirs(os.path.dirname(j["out"]), exist_ok=True)
        log = open(j["log"], "w", encoding="utf-8")
        running.append((name, subprocess.Popen(j["cmd"], stdout=log, stderr=subprocess.STDOUT, env=env), time.time()))
        print(f"[{time.time() - t0:6.0f}s] start {name}", flush=True)
    time.sleep(2)
    keep = []
    for name, p, t in running:
        if p.poll() is None:
            keep.append((name, p, t))
        else:
            print(f"[{time.time() - t0:6.0f}s] done  {name} rc={p.returncode} in {time.time() - t:.0f}s", flush=True)
    running = keep
print("QUEUE DONE", round(time.time() - t0), flush=True)
