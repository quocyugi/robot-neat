"""Build the resume job lists after the stop of 2026-11-10 and report which maze seeds are still missing.

Usage: python make_jobs13.py <scratchpad_new>
Writes tests2/jobs13.json (wave10 runs, long run first) and tests2/jobs13b.json (wave11, the 750k indoor run).
"""
import json
import os
import sys

new = sys.argv[1]
t2 = os.path.join(new, "tests2")
load = lambda f: json.load(open(os.path.join(t2, f), encoding="utf-8"))
J10, J11, J12 = load("jobs10.json"), load("jobs11.json"), load("jobs12.json")

w10 = {}
for n in ["fixB_rnn_1M5", "fixB_rnn_in", "fixB_ff_in", "ga_team_uni_s2"]:
    w10[n] = J10[n]
w10["fixD_sv05_s1"] = J12["fixD_sv05_s1"]
w11 = {"fixB_rnn_in_750k": J11["fixB_rnn_in_750k"]}

for name, jobs, res in (("jobs13.json", w10, "wave10"), ("jobs13b.json", w11, "wave11")):
    todo = [n for n in jobs if not os.path.exists(os.path.join(new, "results2", res, n, "summary.json"))]
    with open(os.path.join(t2, name), "w", encoding="utf-8") as f:
        json.dump({n: jobs[n] for n in todo}, f, ensure_ascii=False, indent=1)
    print(name, "->", res, todo)

M = json.load(open(os.path.join(new, "baselines", "jobs_b03_final.json"), encoding="utf-8"))
missing = [n for n, j in M.items() if not os.path.exists(j["out"])]
print("maze final:", len(M) - len(missing), "done of", len(M), "; missing:", missing)
if missing:
    print("example out:", M[missing[0]]["out"])
