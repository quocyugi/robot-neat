"""Write the corrected NEAT configs for the team by applying a fixed change set to their own .ini files,
and write unified diffs, so that the delivered files differ from theirs only where intended.

Usage: python d33_make_fixed_configs.py <team_repo> <out_dir> <changes.json>
changes.json: {"header": [lines], "changes": {"key": {"value": "...", "comment": [lines]}}, "notes": {"key": [comment lines, value unchanged]}}

Every line taken from changes.json is written as ASCII (Vietnamese without diacritics): neat-python 2.0.0 opens the
.ini with plain open(), i.e. the Windows ANSI code page (cp1252/cp1258), and UTF-8 bytes such as 0x90 in "Đ" raise
UnicodeDecodeError there. The team's own lines are copied unchanged.
"""
import difflib
import json
import os
import re
import sys
import unicodedata

SPECIAL = {"Đ": "D", "đ": "d", "—": "-", "–": "-", "→": "->", "≥": ">=", "≤": "<=", "σ": "sigma", "λ": "lambda", "×": "x",
           "’": "'", "‘": "'", "“": '"', "”": '"', "…": "...", "≈": "~"}


def ascii_vi(s):
    s = unicodedata.normalize("NFD", "".join(SPECIAL.get(ch, ch) for ch in s))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    left = {ch for ch in s if ord(ch) > 127}
    if left:
        raise ValueError(f"non-ASCII characters left: {left}")
    return s


def asciify(obj):
    if isinstance(obj, str):
        return ascii_vi(obj)
    if isinstance(obj, list):
        return [asciify(x) for x in obj]
    if isinstance(obj, dict):
        return {k: asciify(v) for k, v in obj.items()}
    return obj


repo, out_dir, changes_path = sys.argv[1:4]
spec = asciify(json.load(open(changes_path, encoding="utf-8")))
spec["header"] = spec["header"][:-1] + ["# Chu thich viet khong dau: neat-python mo file .ini bang ma hoa mac dinh cua Windows (cp1252),",
                                        "# chu co dau o day se gay UnicodeDecodeError khi khong bat che do UTF-8."] + spec["header"][-1:]
os.makedirs(os.path.join(out_dir, "configs"), exist_ok=True)
os.makedirs(os.path.join(out_dir, "diff"), exist_ok=True)

for name in ("neat_recurrent.ini", "neat_feedforward.ini"):
    src = open(os.path.join(repo, "configs", name), encoding="utf-8").read().splitlines()
    out, header_done, changed = [], False, set()
    for line in src:
        if not header_done and line.strip().startswith("["):
            out.extend(spec["header"])
            out.append("")
            header_done = True
        m = re.match(r"^(\s*)([A-Za-z_]+)(\s*=\s*)(.*)$", line)
        if m and m.group(2) in spec["changes"]:
            key = m.group(2)
            c = spec["changes"][key]
            out.extend("# " + t for t in c.get("comment", []))
            out.append(f"{m.group(1)}{key}{m.group(3)}{c['value']}")
            changed.add(key)
            if key in spec.get("insert_after", {}):
                ins = spec["insert_after"][key]
                out.extend("# " + t for t in ins.get("comment", []))
                out.extend(ins["lines"])
            continue
        if m and m.group(2) in spec.get("notes", {}):
            out.extend("# " + t for t in spec["notes"][m.group(2)])
        out.append(line)
        if m and m.group(2) in spec.get("insert_after", {}):
            ins = spec["insert_after"][m.group(2)]
            out.extend("# " + t for t in ins.get("comment", []))
            out.extend(ins["lines"])
    missing = set(spec["changes"]) - changed
    assert not missing, f"keys not found in {name}: {missing}"
    dst = os.path.join(out_dir, "configs", name)
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    raw = open(dst, "rb").read()
    for enc in ("cp1252", "cp1258"):
        raw.decode(enc)  # raises if neat-python would fail to read the file without UTF-8 mode
    diff = difflib.unified_diff(src, out, fromfile=f"a/configs/{name}", tofile=f"b/configs/{name}", lineterm="")
    with open(os.path.join(out_dir, "diff", name + ".diff"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(diff) + "\n")
    print(name, "changed:", sorted(changed))
