"""Collect the config-fix experiments into one JSON (used by the README and the Word report).

Usage: python d34_fix_summary.py <results2_root> <out_json>
Groups are lists of run names (one per seed); a group's SR per test set is the mean over its seeds,
with the pooled Wilson 95% interval over all test episodes of all seeds.
"""
import glob
import json
import math
import os
import sys

root, out = sys.argv[1:3]
runs = {}
for s in glob.glob(os.path.join(root, "wave*", "*", "summary.json")):
    runs[os.path.basename(os.path.dirname(s))] = json.load(open(s, encoding="utf-8"))

GROUPS = {
    # NEAT, thick maps, geodesic fitness, uniform maps, 180k evaluations
    "neat_team": ["neat_team_uni", "neat_team_uni_s1", "neat_team_uni_s2"],
    "neat_fixA_rnn": ["fixA_rnn_s0"],
    "neat_fixA_ff": ["fixA_ff_s0"],
    "neat_fixB_rnn": ["fixB_rnn_s0", "fixB_rnn_s1", "fixB_rnn_s2"],
    "neat_fixB_ff": ["fixB_ff_s0", "fixB_ff_s1", "fixB_ff_s2"],
    "neat_fixB_rnn_1M5": ["fixB_rnn_1M5"],
    "neat_fixC_ts8": ["fixC_ts8_s0", "fixC_ts8_s1"],
    "neat_fixD_sv05": ["fixD_sv05_s0", "fixD_sv05_s1"],
    "neat_old_1M5": ["neat_ref_thick_1M5"],
    # screening, 1 seed each
    "scr_neat_ref": ["neat_ref_uni"],
    "scr_neat_ref_ts10": ["neat_ref_uni_ts10"],
    "scr_neat_ref_p300": ["neat_ref_uni_p300"],
    "scr_neat_ref_ff": ["neat_ref_uni_ff"],
    "scr_neat_ref_h8": ["neat_ref_uni_h8"],
    "scr_neat_ref_wp01": ["neat_ref_uni_wp01"],
    "scr_neat_ref_wp005": ["neat_ref_uni_wp005"],
    "scr_neat_team_wp01": ["neat_team_uni_wp01"],
    # GA
    "ga_team": ["ga_team_uni", "ga_team_uni_s1", "ga_team_uni_s2"],
    "ga_fixed": ["gafix_s0", "gafix_s1", "gafix_s2"],
    "ga_trunc_noxover": ["ga2_s05", "ga2_s05_s1", "ga2_s05_s2"],
    "scr_ga_s02": ["ga2_s02"],
    "scr_ga_s10": ["ga2_s10"],
    "scr_ga_s05_x": ["ga2_s05_x"],
    # CMA-ES reference on the same thick maps (earlier waves)
    "cma": ["cma_thick", "cma_thick_s1", "cma_thick_s2"],
    "cma_1M": ["cma_thick_1M"],
    # indoor (McGuire-style) family, trained and tested on level 5
    "in_cma": ["cma_in"],
    "in_neat_team": ["neat_team_in"],
    "in_neat_ref": ["neat_ref_in"],
    "in_neat_fixB_rnn": ["fixB_rnn_in"],
    "in_neat_fixB_ff": ["fixB_ff_in"],
    "in_neat_fixB_rnn_750k": ["fixB_rnn_in_750k"],
    "in_ga_fixed_like": ["ga2x_in"],
    "in_ga_team": ["ga_team_in"],
}
KEYS = ["test_L1", "test_L2", "test_L3", "test_L4", "test_L5", "test_barn"]


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


res = {"groups": {}, "missing": []}
for g, names in GROUPS.items():
    have = [n for n in names if n in runs]
    if len(have) < len(names):
        res["missing"].extend(sorted(set(names) - set(have)))
    if not have:
        continue
    row = {"runs": have, "train_evals": runs[have[0]]["train_evals"], "val_mean": round(sum(runs[n]["champion_val_mean"] for n in have) / len(have), 3)}
    for k in KEYS:
        vals = [runs[n]["test"][k] for n in have if k in runs[n]["test"]]
        if not vals:
            continue
        srs = [v["SR"] for v in vals]
        ks = sum(round(v["SR"] * v["n"]) for v in vals)
        ns = sum(v["n"] for v in vals)
        row[k] = {"SR": round(sum(srs) / len(srs), 3), "min": min(srs), "max": max(srs), "CR": round(sum(v["CR"] for v in vals) / len(vals), 3),
                  "timeout": round(sum(v.get("timeout", 0.0) for v in vals) / len(vals), 3),
                  "ci95": wilson(ks, ns), "n_episodes": ns, "per_seed": srs}
    res["groups"][g] = row


def load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


res["xor"] = load(os.path.join(root, "baselines", "xor_fixed.json"))
res["xor_final"] = load(os.path.join(root, "baselines", "xor_final.json"))
res["xor_team_old"] = load(os.path.join(root, "baselines", "xor.json"))
maze = {}
for p in glob.glob(os.path.join(root, "baselines", "lehman_fix", "*.json")):
    j = json.load(open(p, encoding="utf-8"))
    tag = os.path.basename(p).split("_")[2]
    maze.setdefault(tag, []).append(j)
res["maze_hard_ns"] = {t: {"runs": len(v), "solved": sum(r["solved"] for r in v),
                           "evals_mean_solved": round(sum(r["evals_to_solve"] for r in v if r["solved"]) / max(1, sum(r["solved"] for r in v))),
                           "species_median": sorted(r["species_median"] for r in v)[len(v) // 2]} for t, v in maze.items()}
old = {}
for p in glob.glob(os.path.join(root, "baselines", "lehman", "hard_ns_*.json")):
    j = json.load(open(p, encoding="utf-8"))
    old.setdefault(j["config"], []).append(j["solved"])
res["maze_hard_ns_earlier"] = {k: {"runs": len(v), "solved": sum(v)} for k, v in old.items()}
res["indoor_planners"] = load(os.path.join(root, "d21_ref_indoor.json"))
res["indoor_transfer_old_champions"] = load(os.path.join(root, "d31_indoor_old_champions.json"))
with open(out, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
print(json.dumps({g: {k: v[k]["SR"] for k in KEYS if k in v} for g, v in res["groups"].items()}, ensure_ascii=False))
print("missing:", res["missing"])
print("maze:", res["maze_hard_ns"], "earlier:", res["maze_hard_ns_earlier"])
