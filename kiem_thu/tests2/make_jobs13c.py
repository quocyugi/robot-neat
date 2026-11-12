"""Write a one-job list with a different number of pool workers. Results do not depend on the worker count
(checked: the 8-worker 1.5M run reproduced the 3-worker log exactly for the first 120 generations).

Usage: python make_jobs13c.py <scratchpad_new> [<source jobs file> <job name> <out jobs file> <workers>]
Default: jobs10.json fixB_rnn_1M5 -> jobs13c.json with 8 workers.
"""
import json
import os
import sys

t2 = os.path.join(sys.argv[1], "tests2")
src, name, out, workers = (sys.argv[2:6] if len(sys.argv) > 5 else ("jobs10.json", "fixB_rnn_1M5", "jobs13c.json", "8"))
J = json.load(open(os.path.join(t2, src), encoding="utf-8"))
args = list(J[name])
i = args.index("--workers")
args[i + 1] = str(workers)
with open(os.path.join(t2, out), "w", encoding="utf-8") as f:
    json.dump({name: args}, f, ensure_ascii=False, indent=1)
print(out, name, "workers", args[i + 1])
