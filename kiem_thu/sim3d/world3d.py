"""Load DUnE's real Gazebo worlds as triangle meshes for ray casting (no ROS / Gazebo needed).

Two separate meshes are kept per world because Gazebo itself keeps them apart:
  ground : <collision> geometry, which the wheels touch
  lidar  : <visual> geometry, which gpu_lidar renders

Usage: python -I world3d.py <dune_root> <cache_dir> [world ...]     (prints a summary, writes .npz caches)
"""
from __future__ import annotations

import math
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

W_GZ = "ros2_ws/src/roverrobotics_ros2/roverrobotics_gazebo"


def _tag(e):
    return e.tag.split("}")[-1]


def _floats(s):
    return np.fromstring(s, dtype=np.float64, sep=" ")


def _ints(s):
    return np.fromstring(s, dtype=np.int64, sep=" ")


# ------------------------------------------------------------------------------ COLLADA
def _parse_geometry(g):
    mesh = next((c for c in g if _tag(c) == "mesh"), None)
    if mesh is None:
        return None
    src, vert = {}, {}
    prims = []
    for c in mesh:
        t = _tag(c)
        if t == "source":
            fa = next((x for x in c if _tag(x) == "float_array"), None)
            if fa is None or not fa.text:
                continue
            stride = 3
            for x in c.iter():
                if _tag(x) == "accessor" and x.get("stride"):
                    stride = int(x.get("stride"))
            src[c.get("id")] = _floats(fa.text).reshape(-1, stride)[:, :3]
        elif t == "vertices":
            for x in c:
                if _tag(x) == "input" and x.get("semantic") == "POSITION":
                    vert[c.get("id")] = x.get("source").lstrip("#")
        elif t in ("triangles", "polylist", "polygons"):
            prims.append(c)
    Vs, Fs, base = [], [], 0
    pos = None
    for p in prims:
        inputs = [x for x in p if _tag(x) == "input"]
        stride = max(int(x.get("offset", 0)) for x in inputs) + 1
        v_off, v_src = 0, None
        for x in inputs:
            if x.get("semantic") == "VERTEX":
                v_off = int(x.get("offset", 0))
                v_src = vert[x.get("source").lstrip("#")]
        pos = src[v_src]
        t = _tag(p)
        if t == "triangles":
            pe = next(x for x in p if _tag(x) == "p")
            Fs.append(_ints(pe.text)[v_off::stride].reshape(-1, 3))
        elif t == "polylist":
            pe = next(x for x in p if _tag(x) == "p")
            vc = _ints(next(x for x in p if _tag(x) == "vcount").text)
            idx = _ints(pe.text)[v_off::stride]
            tri, k = [], 0
            for n in vc.tolist():
                poly = idx[k:k + n]
                tri.extend([[poly[0], poly[i], poly[i + 1]] for i in range(1, n - 1)])
                k += n
            Fs.append(np.array(tri, dtype=np.int64).reshape(-1, 3))
        else:
            tri = []
            for pe in (x for x in p if _tag(x) == "p"):
                poly = _ints(pe.text)[v_off::stride]
                tri.extend([[poly[0], poly[i], poly[i + 1]] for i in range(1, len(poly) - 1)])
            Fs.append(np.array(tri, dtype=np.int64).reshape(-1, 3))
    if pos is None or not Fs:
        return None
    return pos, np.concatenate(Fs, axis=0)


def _node_matrix(node):
    M = np.eye(4)
    for c in node:
        t = _tag(c)
        if t == "matrix":
            M = M @ _floats(c.text).reshape(4, 4)
        elif t == "translate":
            T = np.eye(4)
            T[:3, 3] = _floats(c.text)
            M = M @ T
        elif t == "rotate":
            x, y, z, deg = _floats(c.text)
            n = math.sqrt(x * x + y * y + z * z) or 1.0
            x, y, z = x / n, y / n, z / n
            a = math.radians(deg)
            s, co = math.sin(a), math.cos(a)
            R = np.eye(4)
            R[:3, :3] = [[co + x * x * (1 - co), x * y * (1 - co) - z * s, x * z * (1 - co) + y * s],
                         [y * x * (1 - co) + z * s, co + y * y * (1 - co), y * z * (1 - co) - x * s],
                         [z * x * (1 - co) - y * s, z * y * (1 - co) + x * s, co + z * z * (1 - co)]]
            M = M @ R
        elif t == "scale":
            S = np.eye(4)
            S[0, 0], S[1, 1], S[2, 2] = _floats(c.text)
            M = M @ S
    return M


def parse_dae(path):
    """Vertices (metres, after <unit> and scene-node transforms) and triangles of a COLLADA file, by streaming."""
    geoms, unit, scene = {}, 1.0, None
    context = ET.iterparse(path, events=("end",))
    for _, el in context:
        t = _tag(el)
        if t == "unit" and el.get("meter"):
            unit = float(el.get("meter"))
        elif t == "geometry":
            r = _parse_geometry(el)
            if r is not None:
                geoms[el.get("id")] = r
            el.clear()
        elif t in ("library_images", "library_effects", "library_materials", "library_animations", "library_controllers"):
            el.clear()
        elif t == "visual_scene":
            scene = el
    Vs, Fs, base = [], [], 0

    def add(g, M):
        nonlocal base
        V, F = g
        Vs.append((M[:3, :3] @ V.T).T + M[:3, 3])
        Fs.append(F + base)
        base += len(V)

    def walk(node, parent):
        M = parent @ _node_matrix(node)
        for c in node:
            t = _tag(c)
            if t == "instance_geometry":
                key = c.get("url").lstrip("#")
                if key in geoms:
                    add(geoms[key], M)
            elif t == "node":
                walk(c, M)

    if scene is not None:
        for n in scene:
            if _tag(n) == "node":
                walk(n, np.eye(4))
    if not Vs:
        for g in geoms.values():
            add(g, np.eye(4))
    return np.concatenate(Vs, axis=0) * unit, np.concatenate(Fs, axis=0)


# ------------------------------------------------------------------------------ primitives
def pose_matrix(p):
    x, y, z, r, pi, ya = p
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(pi), math.sin(pi), math.cos(ya), math.sin(ya)
    M = np.eye(4)
    M[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                 [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                 [-sp, cp * sr, cp * cr]]
    M[:3, 3] = [x, y, z]
    return M


def transformed(V, F, M, scale=(1.0, 1.0, 1.0)):
    return (M[:3, :3] @ (V * np.asarray(scale)).T).T + M[:3, 3], F


def box(sx, sy, sz):
    V = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)]) * [sx, sy, sz]
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return V, F


def cylinder(radius, length, n=14):
    a = np.arange(n) * 2 * np.pi / n
    ring = np.stack([radius * np.cos(a), radius * np.sin(a)], axis=1)
    V = np.vstack([np.c_[ring, np.full(n, -length / 2)], np.c_[ring, np.full(n, length / 2)], [[0, 0, -length / 2], [0, 0, length / 2]]])
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
    return V, np.array(F)


def plane(sx, sy):
    V = np.array([[-sx / 2, -sy / 2, 0], [sx / 2, -sy / 2, 0], [sx / 2, sy / 2, 0], [-sx / 2, sy / 2, 0]], float)
    return V, np.array([[0, 1, 2], [0, 2, 3]])


def heightmap(png, sx, sy, sz):
    from PIL import Image
    img = Image.open(png)
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3:
        a = a[:, :, 0]
    full = 65535.0 if a.max() > 255 else 255.0
    h, w = a.shape
    xs = np.linspace(-sx / 2, sx / 2, w)
    ys = np.linspace(sy / 2, -sy / 2, h)    # image rows run from +y to -y, as in Gazebo
    xx, yy = np.meshgrid(xs, ys)
    V = np.stack([xx.ravel(), yy.ravel(), (a / full).ravel() * sz], axis=1)
    i = np.arange(h * w).reshape(h, w)
    a0, b0, c0, d0 = i[:-1, :-1].ravel(), i[:-1, 1:].ravel(), i[1:, 1:].ravel(), i[1:, :-1].ravel()
    # winding keeps normals pointing up (z+)
    F = np.vstack([np.stack([a0, d0, c0], axis=1), np.stack([a0, c0, b0], axis=1)])
    return V, F


class Parts:
    def __init__(self):
        self.Vs, self.Fs, self.n = [], [], 0

    def add(self, V, F):
        self.Vs.append(np.asarray(V, float))
        self.Fs.append(np.asarray(F, np.int64) + self.n)
        self.n += len(V)

    def get(self):
        return np.concatenate(self.Vs), np.concatenate(self.Fs)


def _pose_of(el):
    p = el.find("pose")
    if p is None or not (p.text or "").strip():
        return np.zeros(6)
    return _floats(p.text)


def sdf_static_to_parts(sdf_path, parts):
    """Boxes, cylinders and planes of every <model> placed directly in the <world> of an SDF file."""
    root = ET.parse(sdf_path).getroot()
    world = root.find("world")
    for m in world.findall("model"):
        Mm = pose_matrix(_pose_of(m))
        for l in m.findall("link"):
            Ml = Mm @ pose_matrix(_pose_of(l))
            for c in l.findall("collision"):
                Mc = Ml @ pose_matrix(_pose_of(c))
                g = c.find("geometry")
                if g.find("box") is not None:
                    parts.add(*transformed(*box(*_floats(g.find("box/size").text)), Mc))
                elif g.find("cylinder") is not None:
                    parts.add(*transformed(*cylinder(float(g.find("cylinder/radius").text), float(g.find("cylinder/length").text)), Mc))
                elif g.find("plane") is not None:
                    parts.add(*transformed(*plane(200.0, 200.0), Mc))


# ------------------------------------------------------------------------------ worlds
# Ranges copied from ros2_ws/src/sb3/environments/rover_env.py
TASKS = {
    "inspect": dict(spawn=(-27, -19, -27, -19), goal=(-27, -19, -27, -19), too_far=(-29, -13, -29, -17), z0=None),
    "island": dict(spawn=(-4, 4, -4, 4), goal=(-4, 4, -4, 4), too_far=(-20, 10, -20, 20), z0=None),
    "maze": dict(spawn=(-4, 4, -4, 4), goal=(-4, 4, -4, 4), too_far=(-30, 30, -30, 30), z0=0.0),
    "pillars": dict(spawn=(-4, 4, -4, 4), goal=(-4, 4, -4, 4), too_far=(-30, 30, -30, 30), z0=0.0),
    "flat": dict(spawn=(-4, 4, -4, 4), goal=(-4, 4, -4, 4), too_far=(-30, 30, -30, 30), z0=0.0),
    "rubicon": dict(spawn=(-4, 4, -4, 4), goal=(-4, 4, -4, 4), too_far=(-18, 18, -18, 18), z0=None),
    "construction": dict(spawn=(-4, 4, -4, 4), goal=(-4, 4, -4, 4), too_far=(-30, 30, -30, 30), z0=0.0),
    # the same meshes with spawn/goal spread over the whole usable area (not used by the DUnE paper)
    "inspect_wide": dict(spawn=(-30, 10, -30, 10), goal=(-30, 10, -30, 10), too_far=(-31, 33, -31, 40), z0=None, max_geo=40.0, base="inspect"),
    "island_wide": dict(spawn=(-20, 10, -20, 20), goal=(-20, 10, -20, 20), too_far=(-20, 10, -20, 20), z0=None, max_geo=40.0, base="island"),
}


def build_world(name, dune):
    gz = os.path.join(dune, "RoboTerrain-main", W_GZ.replace("/", os.sep))
    fuel = os.path.join(dune, "fuel_models")
    ground, lidar = Parts(), Parts()
    if name == "inspect":
        V, F = parse_dae(os.path.join(gz, "meshes", "inspection_world.dae"))
        ground.add(V, F)
        lidar.add(V, F)
        Vw, Fw = parse_dae(os.path.join(gz, "meshes", "inspection_water.dae"))
        lidar.add(Vw, Fw)     # the water surface has a <visual> but no <collision> in inspection.world
    elif name == "island":
        ground.add(*parse_dae(os.path.join(fuel, "Null_Island", "meshes", "islandcollision.dae")))
        root = ET.parse(os.path.join(fuel, "Null_Island", "model.sdf")).getroot()
        for c in root.iter("collision"):
            b = c.find("geometry/box")
            if b is not None:
                ground.add(*transformed(*box(*_floats(b.find("size").text)), pose_matrix(_pose_of(c))))
        lidar.add(*parse_dae(os.path.join(fuel, "Null_Island", "meshes", "island.dae")))
    elif name == "maze":
        sdf_static_to_parts(os.path.join(gz, "worlds", "maze.sdf"), ground)
        V, F = ground.get()
        lidar.add(V, F)
    elif name == "pillars":
        sdf_static_to_parts(os.path.join(gz, "worlds", "maze_pillars.sdf"), ground)
        V, F = ground.get()
        lidar.add(V, F)
    elif name == "flat":
        sdf_static_to_parts(os.path.join(gz, "worlds", "maze_empty.sdf"), ground)
        V, F = ground.get()
        lidar.add(V, F)
    elif name == "construction":
        V, F = parse_dae(os.path.join(fuel, "cpr_office_construction", "meshes", "office_construction.dae"))
        ground.add(V, F)
        lidar.add(V, F)
    elif name == "rubicon":
        d = os.path.join(fuel, "Rubicon")
        ground.add(*heightmap(os.path.join(d, "materials", "textures", "Heightmap.png"), 37.5, 37.5, 5.0))
        root = ET.parse(os.path.join(d, "model.sdf")).getroot()
        cache = {}
        for c in root.iter("collision"):
            g = c.find("geometry")
            M = pose_matrix(_pose_of(c))
            if g.find("cylinder") is not None:
                ground.add(*transformed(*cylinder(float(g.find("cylinder/radius").text), float(g.find("cylinder/length").text)), M))
            elif g.find("mesh") is not None:
                uri = g.find("mesh/uri").text.strip()
                if uri not in cache:
                    cache[uri] = parse_dae(os.path.join(d, uri.replace("/", os.sep)))
                sc = g.find("mesh/scale")
                scale = _floats(sc.text) if sc is not None else (1.0, 1.0, 1.0)
                ground.add(*transformed(*cache[uri], M, scale))
        V, F = ground.get()
        lidar.add(V, F)       # the 197 MB visual mesh is replaced by the collision set; see report
    else:
        raise KeyError(name)
    return ground.get(), lidar.get()


def load_world(name, dune, cache_dir):
    name = TASKS[name].get("base", name)
    p = os.path.join(cache_dir, f"{name}.npz")
    if not os.path.exists(p):
        (Vg, Fg), (Vl, Fl) = build_world(name, dune)
        np.savez_compressed(p, Vg=Vg.astype(np.float32), Fg=Fg.astype(np.uint32), Vl=Vl.astype(np.float32), Fl=Fl.astype(np.uint32))
    d = np.load(p)
    return d["Vg"], d["Fg"], d["Vl"], d["Fl"]


if __name__ == "__main__":
    dune, cache = sys.argv[1], sys.argv[2]
    os.makedirs(cache, exist_ok=True)
    names = sys.argv[3:] or list(TASKS)
    for n in names:
        t0 = time.time()
        Vg, Fg, Vl, Fl = load_world(n, dune, cache)
        print(f"{n:13s} ground faces={len(Fg):8d} bounds={np.round(Vg.min(0), 1).tolist()}..{np.round(Vg.max(0), 1).tolist()} | "
              f"lidar faces={len(Fl):8d} bounds={np.round(Vl.min(0), 1).tolist()}..{np.round(Vl.max(0), 1).tolist()} | {time.time() - t0:.1f}s", flush=True)
