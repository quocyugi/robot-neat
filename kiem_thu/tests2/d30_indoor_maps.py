"""Indoor maps in the style of McGuire et al. 2019 / Butterworth et al. 2019 (corridors, rooms, door openings),
scaled to the team's 10 x 10 m arena, saved as difficulty level 5 in the team's map format.

The steps follow tudelft/indoor_environment_generator (src/random_environment_generator.cpp), with every length
multiplied by F = 10/14 because the original arena is 14 x 14 m:
  * square grid of 7 x 7 cells (2 m cells -> 1.43 m);
  * two corridor-carving walkers start in the start cell and the goal cell; each step they go straight with
    p = 0.7, else turn left or right (0.15 each), wrap around the grid edge, and mark the cell they leave;
  * carving stops when 40 % of the cells are corridor; if start and goal cells are not 4-connected, carving goes on
    without clearing the grid;
  * corridors are widened by 0.4 m -> 0.29 m on each side and a 4 m -> 2.86 m square is kept open around S and G;
  * walls follow the outline of the corridor region;
  * every free region wider (taller) than 40 % of the arena side is split in half by a wall;
  * square openings of 2.7 m -> 1.93 m are cut at random points of the corridor walls, 0.1 per metre of wall in the
    original (0.14 per metre after scaling, so a map gets as many openings as the original);
  * walls are 0.10 m thick boxes, as in the generator's SDF export.
Start and goal are random cell centres at least 5 m apart (the team's rule). The team's A* validator (0.30 m
inflation) and difficulty metrics are reused; no tortuosity filter is applied, as in McGuire et al.

Usage: python d30_indoor_maps.py <repo> <out_root> [<root_to_copy_test_levels_from>] [n_train n_val n_test]
"""
import json
import os
import shutil
import sys
import time

import numpy as np
from scipy import ndimage

REPO, OUT = sys.argv[1:3]
COPY = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None
N_TRAIN, N_VAL, N_TEST = (int(x) for x in sys.argv[4:7]) if len(sys.argv) > 6 else (250, 25, 100)
sys.path.insert(0, REPO)

from src.env.lidar import arena_wall_segments  # noqa: E402
from src.maps.map_metrics import compute_map_metrics  # noqa: E402
from src.maps.validator import INFLATION_RADIUS, validate_map  # noqa: E402
from src.env.kinematics import ROBOT_RADIUS  # noqa: E402

LEVEL = 5
ARENA = 10.0
F = 10.0 / 14.0
N = 7
CELL = ARENA / N
P_STR = 0.7
COVER = 0.4
DIL = 0.4 * F
SQ_HALF = 2.0 * F
SPLIT = 0.4 * ARENA
OPEN_HALF = 1.35 * F
OPEN_RATE = 0.1 / F
THICK = 0.10
RES = 0.05
NP = int(round(ARENA / RES))
MIN_SG = 5.0
MIN_PIECE = 0.10
PARAMS = dict(arena=ARENA, scale=round(F, 4), grid=N, cell=round(CELL, 3), p_straight=P_STR, corridor_cover=COVER, widen_each_side=round(DIL, 3),
              open_square_half=round(SQ_HALF, 3), split_if_longer_than=SPLIT, opening_half=round(OPEN_HALF, 3), openings_per_m=round(OPEN_RATE, 3),
              wall_thickness=THICK, raster=RES, min_start_goal=MIN_SG)


def connected(grid, a, b):
    seen = np.zeros_like(grid)
    stack = [a]
    seen[a[1], a[0]] = True
    while stack:
        x, y = stack.pop()
        if (x, y) == tuple(b):
            return True
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            u, v = x + dx, y + dy
            if 0 <= u < N and 0 <= v < N and grid[v, u] and not seen[v, u]:
                seen[v, u] = True
                stack.append((u, v))
    return False


def carve(rng, s_cell, g_cell):
    grid = np.zeros((N, N), bool)  # [y, x]
    agents = []
    for c in (s_cell, g_cell):
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        k = int(rng.integers(4))
        agents.append([list(c), moves[k:] + moves[:k]])
    for _ in range(200):
        for _it in range(100):
            for ag in agents:
                u = rng.random()
                if u > P_STR:
                    ag[1] = ag[1][1:] + ag[1][:1] if u <= P_STR + (1 - P_STR) / 2 else ag[1][-1:] + ag[1][:-1]
                x, y = ag[0]
                grid[y, x] = True
                dx, dy = ag[1][0]
                ag[0] = [(x + dx) % N, (y + dy) % N]
            if grid.mean() > COVER:
                break
        grid[s_cell[1], s_cell[0]] = grid[g_cell[1], g_cell[0]] = True
        if connected(grid, s_cell, g_cell):
            return grid
    return None


def runs(mask_1d):
    """(start, end_exclusive) of True runs."""
    d = np.diff(np.concatenate([[0], mask_1d.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def outline(a):
    """Pixel edges across which the mask changes, merged into maximal straight pieces.
    Returns horizontal pieces (y, x0, x1) and vertical pieces (x, y0, y1) in metres."""
    hs, vs = [], []
    dh = a[:-1, :] != a[1:, :]
    dv = a[:, :-1] != a[:, 1:]
    for i in range(dh.shape[0]):
        for j0, j1 in runs(dh[i]):
            hs.append([(i + 1) * RES, j0 * RES, j1 * RES])
    for j in range(dv.shape[1]):
        for i0, i1 in runs(dv[:, j]):
            vs.append([(j + 1) * RES, i0 * RES, i1 * RES])
    return hs, vs


def cut(pieces, lo_fixed, hi_fixed, lo, hi):
    """Remove [lo, hi] from pieces whose fixed coordinate lies in [lo_fixed, hi_fixed]."""
    out = []
    for c, p0, p1 in pieces:
        if lo_fixed <= c <= hi_fixed and p1 > lo and p0 < hi:
            if p0 < lo:
                out.append([c, p0, lo])
            if p1 > hi:
                out.append([c, hi, p1])
        else:
            out.append([c, p0, p1])
    return out


def rect(cx, cy, hw, hh):
    tl, tr, br, bl = [cx - hw, cy + hh], [cx + hw, cy + hh], [cx + hw, cy - hh], [cx - hw, cy - hh]
    return [[tl, tr], [tr, br], [br, bl], [bl, tl]]


def try_map(rng):
    cells = [(x, y) for x in range(N) for y in range(N)]
    centre = lambda c: np.array([(c[0] + 0.5) * CELL, (c[1] + 0.5) * CELL])
    for _ in range(100):
        s_cell, g_cell = (cells[i] for i in rng.choice(len(cells), 2, replace=False))
        if np.linalg.norm(centre(s_cell) - centre(g_cell)) >= MIN_SG:
            break
    else:
        return None
    grid = carve(rng, s_cell, g_cell)
    if grid is None:
        return None
    S, G = centre(s_cell), centre(g_cell)
    xs = (np.arange(NP) + 0.5) * RES
    X, Y = np.meshgrid(xs, xs)
    corr = np.zeros((NP, NP), bool)
    for y in range(N):
        for x in range(N):
            if grid[y, x]:
                corr |= (X >= x * CELL - DIL) & (X <= (x + 1) * CELL + DIL) & (Y >= y * CELL - DIL) & (Y <= (y + 1) * CELL + DIL)
    for p in (S, G):
        corr |= (np.abs(X - p[0]) <= SQ_HALF) & (np.abs(Y - p[1]) <= SQ_HALF)
    h_walls, v_walls = outline(corr)
    # room dividers inside each free region that is wider / taller than 40 % of the side
    h_div, v_div = [], []
    lab, n_lab = ndimage.label(~corr)
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        reg = lab == k
        r0, r1, c0, c1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        if (c1 - c0) * RES >= SPLIT:
            jm = c0 + (c1 - c0) // 2
            col = reg[:, jm - 1] & reg[:, jm]
            for i0, i1 in runs(col):
                v_div.append([jm * RES, i0 * RES, i1 * RES])
        if (r1 - r0) * RES >= SPLIT:
            im = r0 + (r1 - r0) // 2
            row = reg[im - 1, :] & reg[im, :]
            for j0, j1 in runs(row):
                h_div.append([im * RES, j0 * RES, j1 * RES])
    # openings along the corridor walls
    centres = []
    for c, p0, p1 in h_walls:
        for _ in range(rng.binomial(max(1, int(round((p1 - p0) / RES))), OPEN_RATE * RES)):
            centres.append((rng.uniform(p0, p1), c))
    for c, p0, p1 in v_walls:
        for _ in range(rng.binomial(max(1, int(round((p1 - p0) / RES))), OPEN_RATE * RES)):
            centres.append((c, rng.uniform(p0, p1)))
    H, V = h_walls + h_div, v_walls + v_div
    for ox, oy in centres:
        H = cut(H, oy - OPEN_HALF, oy + OPEN_HALF, ox - OPEN_HALF, ox + OPEN_HALF)
        V = cut(V, ox - OPEN_HALF, ox + OPEN_HALF, oy - OPEN_HALF, oy + OPEN_HALF)
    segs = []
    for c, p0, p1 in H:
        if p1 - p0 >= MIN_PIECE:
            segs.extend(rect((p0 + p1) / 2, c, (p1 - p0) / 2 + THICK / 2, THICK / 2))
    for c, p0, p1 in V:
        if p1 - p0 >= MIN_PIECE:
            segs.extend(rect(c, (p0 + p1) / 2, THICK / 2, (p1 - p0) / 2 + THICK / 2))
    all_segs = np.concatenate([arena_wall_segments(ARENA, ARENA), np.array(segs, dtype=np.float64).reshape(-1, 2, 2)], axis=0)
    val = validate_map(all_segs, S, G)
    if not val["valid"]:
        return None
    metrics = compute_map_metrics(all_segs, S, G, validation=val)
    path = metrics.pop("optimal_path")
    extra = dict(corridor_cells=int(grid.sum()), free_regions=int(n_lab), openings=len(centres), wall_pieces=len(segs) // 4)
    return S, G, all_segs, metrics, path, extra


def generate(seed, max_tries=50):
    for attempt in range(max_tries):
        rng = np.random.default_rng([LEVEL, seed, attempt])
        r = try_map(rng)
        if r is not None:
            S, G, segs, metrics, path, extra = r
            return {"map_id": f"lvl{LEVEL}_seed_{seed:06d}", "difficulty_level": LEVEL, "seed": seed, "source": "indoor_mcguire_style",
                    "bounds": [ARENA, ARENA], "start": S.tolist(), "goal": G.tolist(), "robot_radius": ROBOT_RADIUS,
                    "inflation_radius": INFLATION_RADIUS, "metrics": metrics, "generator": extra, "optimal_path": path, "segments": segs.tolist()}
    return None


def main():
    stats = {}
    t0 = time.time()
    for split, n, offset in (("train", N_TRAIN, 0), ("val", N_VAL, 100_000), ("test", N_TEST, 200_000)):
        out_dir = os.path.join(OUT, split, f"level_{LEVEL}")
        os.makedirs(out_dir, exist_ok=True)
        made, seed, skipped, rows = 0, offset, 0, []
        while made < n:
            m = generate(seed)
            if m is None:
                skipped += 1
            else:
                with open(os.path.join(out_dir, f"map_{seed:06d}.json"), "w", encoding="utf-8") as f:
                    json.dump(m, f, ensure_ascii=False)
                rows.append([m["metrics"]["tortuosity"], m["metrics"]["optimal_path_length"], m["generator"]["wall_pieces"], m["generator"]["openings"]])
                made += 1
            seed += 1
        r = np.array(rows)
        stats[split] = dict(maps=made, seeds_tried=seed - offset, seeds_without_valid_map=skipped, tau_median=round(float(np.median(r[:, 0])), 3),
                            path_median_m=round(float(np.median(r[:, 1])), 2), wall_pieces_median=float(np.median(r[:, 2])), openings_median=float(np.median(r[:, 3])))
        print(split, stats[split], f"{time.time() - t0:.0f}s", flush=True)
    if COPY:
        for lv in (1, 2, 3, 4):
            dst = os.path.join(OUT, "test", f"level_{lv}")
            if not os.path.exists(dst):
                shutil.copytree(os.path.join(COPY, "test", f"level_{lv}"), dst)
        for rel in ("test_barn", "test_lehman"):
            dst = os.path.join(OUT, rel)
            if not os.path.exists(dst):
                shutil.copytree(os.path.join(COPY, rel), dst)
    with open(os.path.join(OUT, "generation_stats.json"), "w", encoding="utf-8") as f:
        json.dump(dict(level=LEVEL, params=PARAMS, splits=stats), f, indent=1)


if __name__ == "__main__":
    main()
