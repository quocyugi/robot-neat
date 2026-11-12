import json
import sys

j = json.load(open(sys.argv[1], encoding="utf-8"))
ctrls = []
for w in j:
    for k in j[w]:
        c = k.split("|")[0]
        if c not in ctrls:
            ctrls.append(c)
print("SR raw (ideal ground filter in brackets)")
print(f"{'world':14s}" + "".join(f"{c:>16s}" for c in ctrls))
for w in j:
    row = []
    for c in ctrls:
        a = j[w].get(c + "|raw", {}).get("SR")
        b = j[w].get(c + "|ideal_ground_filter", {}).get("SR")
        row.append((f"{a:.2f}" if a is not None else "-") + (f" ({b:.2f})" if b is not None else ""))
    print(f"{w:14s}" + "".join(f"{r:>16s}" for r in row))
print()
for w in j:
    print(w)
    for k, r in j[w].items():
        if k.endswith("|raw"):
            print(f"   {k[:-4]:12s} SR={r['SR']:.2f} fall={r['fall']:.2f} far={r['too_far']:.2f} stuck={r['stuck']:.2f} timeout={r['timeout']:.2f} TC={r['TC_blocks_per_ep']:.1f} close={r['close_lt_0.3_frac']:.3f} SPL={r['SPL']:.2f} MTT={r['MTT_s']} speed={r['mean_speed']:.2f} pitch={r['mean_abs_pitch_deg']:.1f} roll={r['mean_abs_roll_deg']:.1f} gnd3m={r['ground_hit_frac_3m']:.2f}")
