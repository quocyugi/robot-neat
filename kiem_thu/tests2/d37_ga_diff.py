"""Regenerate diff/fixed_ga.py.diff (unified diff of the team's src/baselines/fixed_ga.py against the delivered copy).

Usage: python d37_ga_diff.py <team_repo> <delivered_folder>
"""
import difflib
import os
import sys

repo, out = sys.argv[1], sys.argv[2]
rel = "src/baselines/fixed_ga.py"
a = open(os.path.join(repo, *rel.split("/")), encoding="utf-8").read().splitlines(keepends=True)
b = open(os.path.join(out, *rel.split("/")), encoding="utf-8").read().splitlines(keepends=True)
d = "".join(difflib.unified_diff(a, b, fromfile="a/" + rel, tofile="b/" + rel))
with open(os.path.join(out, "diff", "fixed_ga.py.diff"), "w", encoding="utf-8", newline="\n") as f:
    f.write(d)
print("diff lines:", d.count("\n"))
