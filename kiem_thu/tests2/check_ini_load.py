"""(1) Compare parsed key/value pairs of two sets of .ini files; (2) load each file with neat.Config in the current
interpreter mode. Run WITHOUT -X utf8 to reproduce how the team's code reads the files on Windows.

Usage: python check_ini_load.py <before_dir> <after_dir>
"""
import configparser
import os
import sys

import neat

before, after = sys.argv[1], sys.argv[2]
print("UTF-8 mode:", sys.flags.utf8_mode)
for name in ("neat_recurrent.ini", "neat_feedforward.ini"):
    vals = []
    for d in (before, after):
        cp = configparser.ConfigParser()
        cp.read(os.path.join(d, name), encoding="utf-8")
        vals.append({s: dict(cp[s]) for s in cp.sections()})
    print(f"{name}: parsed values identical before/after = {vals[0] == vals[1]}")
    for d, tag in ((before, "before"), (after, "after")):
        try:
            neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, os.path.join(d, name))
            print(f"   neat.Config({tag}) OK")
        except Exception as e:
            print(f"   neat.Config({tag}) FAILED: {type(e).__name__}: {str(e)[:90]}")
