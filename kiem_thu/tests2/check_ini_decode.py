"""Check whether config files decode under the Windows ANSI code pages that neat-python's plain open() would use.

Usage: python check_ini_decode.py <file> [<file> ...]
"""
import locale
import sys

print("this machine's locale encoding:", locale.getencoding())
for p in sys.argv[1:]:
    raw = open(p, "rb").read()
    res = []
    for enc in ("utf-8", "cp1252", "cp1258"):
        try:
            raw.decode(enc)
            res.append(f"{enc}=ok")
        except UnicodeDecodeError as e:
            res.append(f"{enc}=FAIL(byte {raw[e.start]:#x} at {e.start})")
    print(p[-60:], " ".join(res))
