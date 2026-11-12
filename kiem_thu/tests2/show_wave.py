import glob
import json
import os
import sys

root = sys.argv[1]
print(f"{'run':30s} {'evals':>7} {'stepsM':>7} {'wall':>6} lvl  L1   L2   L3   L4   BARN  Lehm | CR_L3 CR_L4 TO_L3 TO_L4 | promos")
for d in sorted(glob.glob(os.path.join(root, "*"))):
    s = os.path.join(d, "summary.json")
    if not os.path.isdir(d):
        continue
    if not os.path.exists(s):
        log = d + ".log"
        tail = open(log, encoding="utf-8").read()[-300:] if os.path.exists(log) else ""
        print(f"{os.path.basename(d):30s} (no summary) {tail!r}")
        continue
    j = json.load(open(s, encoding="utf-8"))
    t = j["test"]
    g = lambda k, m: t[k][m]
    print(f"{os.path.basename(d):30s} {j['train_evals']:7d} {j['approx_steps_M']:7.1f} {j['wall_s']:6.0f} {j['final_level']:3d} "
          f"{g('test_L1','SR'):4.2f} {g('test_L2','SR'):4.2f} {g('test_L3','SR'):4.2f} {g('test_L4','SR'):4.2f} {g('test_barn','SR'):5.2f} {g('test_lehman','SR'):5.2f} | "
          f"{g('test_L3','CR'):5.2f} {g('test_L4','CR'):5.2f} {g('test_L3','timeout'):5.2f} {g('test_L4','timeout'):5.2f} | {j['promotions']}")
