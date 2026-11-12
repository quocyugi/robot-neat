"""Print the headline numbers of fix_summary.json and the transfer scores.

Usage: python show_fix.py <results2_root>
"""
import json
import os
import sys

root = sys.argv[1]
S = json.load(open(os.path.join(root, "fix_summary.json"), encoding="utf-8"))
G = S["groups"]
for g, v in G.items():
    cells = []
    for k in ("test_L1", "test_L2", "test_L3", "test_L4", "test_barn", "test_L5"):
        if k in v:
            c = v[k]
            ps = c.get("per_seed", [])
            cells.append(f"{k[5:]}={c['SR']*100:.0f}" + (f"[{','.join(f'{x*100:.0f}' for x in ps)}]" if len(ps) > 1 else ""))
    print(f"{g:24s} n={len(v.get('runs', []))} ev={v.get('train_evals')} " + " ".join(cells))
print("other keys:", [k for k in S if k != "groups"])
T = os.path.join(root, "d31_indoor_old_champions.json")
if os.path.exists(T):
    for name, r in json.load(open(T, encoding="utf-8")).items():
        print("transfer", name, r["algo"], r["train_evals"], {k: round(v["SR"], 2) for k, v in r["scores"].items()})
