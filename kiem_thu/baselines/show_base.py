"""Summaries of the re-implemented baselines. Usage: python show_base.py <results2/baselines>"""
import glob
import json
import math
import os
import re
import statistics as st
import sys

root = sys.argv[1]
print("== Lahouazi retrain (success on 200 maps, standard / dense / noisy)")
rows = {}
for p in sorted(glob.glob(os.path.join(root, "lahouazi", "*.json"))):
    j = json.load(open(p, encoding="utf-8"))
    name = re.sub(r"_s\d+$", "", os.path.basename(p)[:-5])
    v = [j["test"][k]["success"] for k in j["test"]]
    rows.setdefault(name, []).append((v, j["steps"], j["generations"], j["wall_s"]))
for n, r in rows.items():
    a = [sum(x[0][i] for x in r) / len(r) for i in range(3)]
    print(f"{n:12s} seeds={len(r)} mean={a[0]:.0f}/{a[1]:.0f}/{a[2]:.0f}  each={[x[0] for x in r]}  steps={r[0][1]:,} gens={r[0][2]} wall={r[0][3]:.0f}s")

print("== Lehman-Stanley")
g = {}
for p in sorted(glob.glob(os.path.join(root, "lehman", "*.json"))):
    if os.path.basename(p).startswith("_"):
        continue
    j = json.load(open(p, encoding="utf-8"))
    g.setdefault((j["maze"], j["method"], j["config"]), []).append(j)
for k, r in sorted(g.items()):
    sv = [x for x in r if x["solved"]]
    ev = [x["evals_to_solve"] for x in sv]
    print(f"{k[0]:7s}{k[1]:4s}{k[2]:5s} solved {len(sv)}/{len(r)}  evals mean={st.mean(ev) if ev else float('nan'):.0f} median={st.median(ev) if ev else float('nan'):.0f}  best_d_unsolved_median={st.median([x['best_distance'] for x in r if not x['solved']]) if len(sv) < len(r) else float('nan'):.1f}  species_med={st.median([x['species_median'] for x in r]):.0f}  wall_mean={st.mean([x['wall_s'] for x in r]):.0f}s")
