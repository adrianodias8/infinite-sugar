#!/usr/bin/env python3
"""Photoreceptor retinotopy for the eye render: data/raw + data/roles.json -> web/brain/retina.json

The `visual` pool is FlyWire's photoreceptors: R1-6 (lamina), R7 and R8 (medulla) and the
ocellar retinula cells. The lamina and medulla are retinotopic, so a photoreceptor's terminal
position says where in the visual field it looks. This is an INFERRED map, not the measured
ommatidial lattice: per eye and per layer, the mean recorded position of each cell is ranked
along the dorsoventral axis (FAFB y grows ventrally) for elevation and along the anterior-
posterior arc of the neuropil (the second principal axis in the x-z plane; FAFB z grows toward
the rear) for azimuth, and the ranks are spread uniformly over the eye's field: azimuth from
-15 deg (past the midline, the binocular strip) to 155 deg (rear), elevation from +65 (dorsal)
to -65 deg. Ranks, because ommatidia are close to evenly spaced and the recorded positions are
not. The ocelli look up: +75 deg elevation, the lateral ones 30 deg to their side.

Usage: python3 tools/build_retina.py [data/raw] [data] [web/brain]
"""
import csv, gzip, json, os, sys
from collections import defaultdict
import numpy as np

RAW = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
DATA = sys.argv[2] if len(sys.argv) > 2 else "data"
OUT = sys.argv[3] if len(sys.argv) > 3 else "web/brain"
AZ = (-15.0, 155.0)      # degrees, per eye, front (past the midline) to rear
EL = (65.0, -65.0)       # degrees, dorsal to ventral

def rows(name):
    with gzip.open(os.path.join(RAW, name), "rt", newline="") as f:
        r = csv.reader(f); next(r); yield from r

roles = json.load(open(os.path.join(DATA, "roles.json")))
visual = set(roles["visual"]); left = set(roles["visual_l"]); right = set(roles["visual_r"])
ids = [int(row[0]) for row in rows("classification.csv.gz")]
idx = {r: i for i, r in enumerate(ids)}
ptype = {}
for row in rows("consolidated_cell_types.csv.gz"):
    i = idx.get(int(row[0]))
    if i in visual: ptype[i] = row[1].strip()
acc = defaultdict(lambda: [np.zeros(3), 0])
for row in rows("coordinates.csv.gz"):
    i = idx.get(int(row[0]))
    if i in visual:
        acc[i][0] += np.array([float(t) for t in row[1].strip("[]").split()]); acc[i][1] += 1
pos = {i: a[0] / a[1] for i, a in acc.items()}
print(f"photoreceptors: {len(visual):,}, with positions: {len(pos):,}")

def ranks(x):
    order = np.argsort(x); r = np.empty(len(x)); r[order] = np.arange(len(x))
    return r / max(1, len(x) - 1)

cells = []   # (index, eye, azimuth, elevation, type)
TYPES = {"R1-6": 0, "R7": 1, "R8": 2, "ocellar_retinula_cell": 3}
for eye, side in ((0, left), (1, right)):
    for t in ("R1-6", "R7", "R8"):
        sel = [i for i in sorted(side) if ptype.get(i) == t and i in pos]
        P = np.array([pos[i] for i in sel])
        el = EL[0] + (EL[1] - EL[0]) * ranks(P[:, 1])                       # y: dorsal (low) -> ventral
        # the arc: principal axis of the x-z spread, oriented so z (front -> rear) increases
        Q = P[:, [0, 2]] - P[:, [0, 2]].mean(0)
        u, s, vt = np.linalg.svd(Q, full_matrices=False)
        axis = vt[0] if vt[0][1] >= 0 else -vt[0]
        arc = Q @ axis
        az = AZ[0] + (AZ[1] - AZ[0]) * ranks(arc)
        for k, i in enumerate(sel): cells.append((i, eye, float(az[k]), float(el[k]), TYPES[t]))
        print(f"  {'left' if eye == 0 else 'right':<5} {t:<5} {len(sel):>5}  arc axis (x,z) {axis.round(2)}  spread {s[0]/s[1]:.1f}:1")
    oc = [i for i in sorted(side) if ptype.get(i) == "ocellar_retinula_cell"]
    for i in oc: cells.append((i, eye, 30.0, 75.0, 3))
centre = [i for i in sorted(visual - left - right) if ptype.get(i) == "ocellar_retinula_cell"]
for i in centre: cells.append((i, 2, 0.0, 75.0, 3))
unplaced = visual - {c[0] for c in cells}
print(f"  ocellar: left {len(oc)}, centre {len(centre)}; unplaced (no side, not ocellar): {len(unplaced)}")
cells.sort()
out = {
    "source": "FlyWire FAFB v783 mean recorded positions of the photoreceptors; inferred retinotopy (see tools/build_retina.py)",
    "n": len(cells),
    "azimuthRange": list(AZ), "elevationRange": [EL[1], EL[0]],
    "types": ["R1-6", "R7", "R8", "ocellar"],
    "idx": [c[0] for c in cells], "eye": [c[1] for c in cells],
    "az": [round(c[2] * 10) for c in cells], "el": [round(c[3] * 10) for c in cells],   # tenths of a degree
    "type": [c[4] for c in cells],
}
os.makedirs(OUT, exist_ok=True)
p = os.path.join(OUT, "retina.json")
json.dump(out, open(p, "w"), separators=(",", ":"))
print(f"wrote {p}: {len(cells):,} cells, {os.path.getsize(p)/1024:.0f} KB")
