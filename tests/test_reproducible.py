"""
Kiểm thử tích hợp L14 + L13: cùng seed cho cùng log, không phụ thuộc số worker
==============================================================================
Chạy NEATRunner (M6: hồi quy + novelty + curriculum) hai lần, 2 thế hệ, 2 bản đồ
mỗi thế hệ: một lần 1 worker, một lần 4 worker. Lịch sử huấn luyện (trừ thời gian)
và champion phải giống hệt. Khoảng 1–3 phút.

Chạy:  python tests/test_reproducible.py
"""

import json
import pickle
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.neat_engine.runner import NEATRunner  # noqa: E402


def run_once(out: Path, workers: int, seed: int = 3) -> tuple[list, dict]:
    r = NEATRunner(ROOT / "configs" / "neat_recurrent.ini", ROOT / "data" / "maps", out,
                   max_generations=2, maps_per_eval=2, seed=seed, workers=workers, val_every=2, verbose=False)
    r.run()
    hist = json.load(open(out / "training_history.json", encoding="utf-8"))
    for h in hist:
        h.pop("seconds", None)
    champ = pickle.load(open(out / "champion_genome.pkl", "rb"))
    return hist, {"key": champ.key, "nodes": sorted(champ.nodes), "conns": sorted(champ.connections),
                  "weights": [round(c.weight, 12) for _, c in sorted(champ.connections.items())]}


def main():
    from src.console import utf8_stdout
    utf8_stdout()
    with tempfile.TemporaryDirectory() as tmp:
        h1, c1 = run_once(Path(tmp) / "w1", workers=1)
        h4, c4 = run_once(Path(tmp) / "w4", workers=4)
    same_hist, same_champ = h1 == h4, c1 == c4
    print(f"  lịch sử giống hệt: {same_hist}; champion giống hệt: {same_champ}")
    sys.exit(0 if same_hist and same_champ else 1)


if __name__ == "__main__":
    main()
