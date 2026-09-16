#!/usr/bin/env python3
"""FlyWire Codex FAFB v783 -> whole-brain graph for utilifly.

Unlike the borrowed desktop-fly etl.py, there is NO neuron selection step: every
classified neuron and every connection at the Codex >=5-synapse threshold is kept.
Roles tag the populations we wire I/O to; everything else runs anyway, unnamed.

Usage: python3 tools/build_brain.py [data/raw] [data]
"""
import csv, gzip, json, os, sys
from collections import defaultdict, deque
import numpy as np

RAW = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data"
os.makedirs(OUT, exist_ok=True)

# nt_type -> sign. The Codex prediction per synapse (see docs/04-roadmap.md honesty budget)...
NT_SIGN = {"ACH": 1.0, "GABA": -1.0, "GLUT": -1.0, "DA": 0.5, "SER": 0.5, "OCT": 0.5}
# ...corrected, where the literature knows a cell type's transmitter, by the ground truth of
# flyconnectome/drosophila_neurotransmitters (gt_data.csv, CC-BY 4.0; Eckstein et al.): a
# neuron whose FlyWire cell type has one verified fast transmitter at confidence >= 3 gets that
# transmitter's sign on every outgoing synapse. Histamine (the photoreceptors) and glycine are
# inhibitory in the fly; the monoamines stay modulatory at 0.5 as in the prediction map; nitric
# oxide and co-transmitting types are left to the prediction. Set GT_MIN_CONF = 99 to disable.
GT_SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0, "glycine": -1.0,
           "dopamine": 0.5, "serotonin": 0.5, "octopamine": 0.5, "tyramine": 0.5}
GT_FAST = ("acetylcholine", "gaba", "glutamate", "histamine", "glycine")
GT_MIN_CONF = 3

def rows(name):
    with gzip.open(os.path.join(RAW, name), "rt", newline="") as f:
        r = csv.reader(f); next(r); yield from r

# ---- classification: identity of every neuron -------------------------------
ids, sup, cls, sub, side = [], [], [], [], []
for row in rows("classification.csv.gz"):
    ids.append(int(row[0])); sup.append(row[2]); cls.append(row[3])
    sub.append(row[4]); side.append(row[6])
N = len(ids)
idx = {r: i for i, r in enumerate(ids)}
print(f"neurons in classification: {N:,}")

ptype = ["" ] * N
for row in rows("consolidated_cell_types.csv.gz"):
    i = idx.get(int(row[0]))
    if i is not None: ptype[i] = row[1].strip()

# ---- roles: the populations we wire I/O to ----------------------------------
role = [""] * N
for i in range(N):
    s, c, p = sub[i], cls[i], ptype[i]
    # --- outputs: brain motor neurons that map onto flybody joints -------------
    if   s == "proboscis_motor_neuron":                      role[i] = "mn_proboscis"
    elif s == "haustellum_motor_neuron":                     role[i] = "mn_haustellum"
    elif s == "ingestion_motor_neuron":                      role[i] = "mn_ingestion"
    elif s == "neck_motor_neuron":                           role[i] = "mn_neck"
    elif s == "antennal_motor_neuron":                       role[i] = "mn_antenna"
    # (neck + antenna are also split by side below, so left/right move independently)
    # --- descending neurons: identified command cells ---------------------------
    # These do NOT map to muscles. FAFB is brain-only; wing and leg motor neurons live in the
    # ventral nerve cord (MANC), which is not in this dataset. A DN says "escape" — the posture
    # it produces is supplied by us. Kept separate from the motor pools for that reason.
    elif p == "DNp01":                                       role[i] = "dn_gf"        # giant fiber
    elif p in ("DNp02", "DNp04", "DNp11"):                   role[i] = "dn_escwing"   # escape wing
    elif p in ("DNa01", "DNa02"):                            role[i] = "dn_steer"     # steering
    elif p == "DNp09":                                       role[i] = "dn_walk"
    elif p == "DNg11":                                       role[i] = "dn_groom"
    elif p == "MDN":                                         role[i] = "dn_back"
    # --- looming detectors: the biologically correct trigger for the giant fiber -
    elif p == "LC4":                                         role[i] = "lc4"
    elif p == "LPLC2":                                       role[i] = "lplc2"
    elif p == "LC11":                                        role[i] = "lc11"        # small-object motion detectors
    # --- inputs: sensory populations -------------------------------------------
    elif s == "sugar/water":                                 role[i] = "grn_sweet"
    elif s == "SA_VTV_pro_meso_meta" and c == "gustatory":   role[i] = "grn_sweet_leg"
    elif s == "bitter":                                      role[i] = "grn_bitter"
    elif c == "olfactory":                                   role[i] = "orn"
    elif c == "mechanosensory":                              role[i] = "mechano"
    elif c == "thermosensory":                               role[i] = "thermo"
    elif c == "hygrosensory":                                role[i] = "hygro"
    elif c == "visual":                                      role[i] = "visual"
    # --- the reward readout ----------------------------------------------------
    elif p.startswith("PAM"):                                role[i] = "pam"
groups = defaultdict(list)
for i, r in enumerate(role):
    if r: groups[r].append(i)
for i in range(N):
    if cls[i] == "mechanosensory" and sub[i] == "wind_gravity": groups["mechano_wind"].append(i)
# side-split the paired motor pools so the two sides can drive their joints independently
for r in ("mn_neck", "mn_antenna", "dn_steer", "dn_escwing"):
    for sd in ("left", "right"):
        g = [i for i in groups[r] if side[i] == sd]
        if g: groups[f"{r}_{sd[0]}"] = g
# Side-split the sensory populations the world can drive from one side: the terrarium can
# smell/see/loom/touch the fly on the left or the right, and the steering and escape DNs are
# split the same way, so a lateralised input has a chance of a lateralised output. Neurons
# without a side (30 ORNs, 76 'center' visual cells) stay in the union only.
for r in ("orn", "mechano", "mechano_wind", "visual", "lc4", "lplc2", "lc11", "grn_sweet", "grn_bitter", "hygro"):
    for sd in ("left", "right"):
        g = [i for i in groups[r] if side[i] == sd]
        if g: groups[f"{r}_{sd[0]}"] = g
# Split the temperature senses by FlyWire sub_class. Real flies have separate hot and cold
# cells on the arista/sacculus; v783 labels them: thermosensory 'heating' (TRN_VP2) vs 'cold'
# (TRN_VP3a/b), and the hygrosensory 'cooling' / 'evaporative_cooling' cells (HRN_VP1l/VP1d).
# `thermo` and `hygro` stay as the unions, so nothing that used them changes.
SUB_SPLIT = {
    "thermo_hot":  ("thermo", ("heating",)),
    "thermo_cold": ("thermo", ("cold",)),
    "hygro_cool":  ("hygro",  ("cooling", "evaporative_cooling")),
}
for r, (parent, subs) in SUB_SPLIT.items():
    g = [i for i in groups[parent] if sub[i] in subs]
    if g: groups[r] = g
print("\nroles:")
for r in sorted(groups): print(f"  {r:<16} {len(groups[r]):>4}")

# ---- connections: aggregate (pre,post) across neuropils ---------------------
pre_l, post_l, syn_l, nt_l = [], [], [], []
NT_CODE = {k: i for i, k in enumerate(NT_SIGN)}
for row in rows("connections.csv.gz"):
    a, b = idx.get(int(row[0])), idx.get(int(row[1]))
    if a is None or b is None: continue
    pre_l.append(a); post_l.append(b); syn_l.append(int(row[3]))
    nt_l.append(NT_CODE.get(row[4].strip().upper(), 0))
pre = np.array(pre_l, np.int32); post = np.array(post_l, np.int32)
syn = np.array(syn_l, np.int32); ntc = np.array(nt_l, np.int8)
del pre_l, post_l, syn_l, nt_l
print(f"\nconnection rows (per-neuropil): {len(pre):,}")

key = pre.astype(np.int64) * N + post
order = np.argsort(key, kind="stable")
key, pre, post, syn, ntc = key[order], pre[order], post[order], syn[order], ntc[order]
uniq, start = np.unique(key, return_index=True)
agg_syn = np.add.reduceat(syn, start)
agg_pre, agg_post, agg_nt = pre[start], post[start], ntc[start]   # nt of first (dominant) row
E = len(uniq)
print(f"aggregated edges (pre,post):   {E:,}")
print(f"total synapses:                {agg_syn.sum():,}")

sign = np.array([NT_SIGN[k] for k in NT_SIGN], np.float32)[agg_nt]
# ---- literature-verified transmitters override the prediction per presynaptic neuron -------
gt_path = os.path.join(RAW, "gt_data.csv")
gt_sign = {}
if os.path.exists(gt_path) and GT_MIN_CONF <= 5:
    with open(gt_path, newline="") as f:
        for row in csv.DictReader(f):
            try: conf = int(row["neurotransmitter_verified_confidence"] or 0)
            except ValueError: conf = 0
            if conf < GT_MIN_CONF: continue
            fast = [t for t in GT_FAST if row.get(t) == "1"]
            mod = [t for t in GT_SIGN if t not in GT_FAST and row.get(t) == "1"]
            if len(fast) == 1: gt_sign[row["cell_type"]] = GT_SIGN[fast[0]]
            elif not fast and len(mod) == 1: gt_sign[row["cell_type"]] = GT_SIGN[mod[0]]
            # co-transmitting or unknown: leave the prediction
    override = np.full(N, np.nan, np.float32)
    for i in range(N):
        v = gt_sign.get(ptype[i])
        if v is not None: override[i] = v
    has = ~np.isnan(override[agg_pre])
    before = sign.copy()
    sign = np.where(has, override[agg_pre], sign).astype(np.float32)
    changed = has & (before != sign)
    flipped = has & (np.sign(before) != np.sign(sign))
    print(f"neurotransmitter ground truth: {len(gt_sign):,} cell types, {int(np.sum(~np.isnan(override))):,} neurons, "
          f"{int(has.sum()):,} of {E:,} edges covered, {int(changed.sum()):,} signs changed, {int(flipped.sum()):,} flipped in sign")
else:
    print("neurotransmitter ground truth: not applied (no data/raw/gt_data.csv)")
w = (agg_syn * sign).astype(np.float32)

# ---- CSR ---------------------------------------------------------------------
indptr = np.zeros(N + 1, np.int64)
np.add.at(indptr, agg_pre + 1, 1)
np.cumsum(indptr, out=indptr)
colidx, weight = agg_post.astype(np.int32), w      # already sorted by (pre,post)

np.savez_compressed(os.path.join(OUT, "brain.npz"),
                    indptr=indptr, colidx=colidx, weight=weight,
                    role=np.array(role), side=np.array(side), ptype=np.array(ptype))
with open(os.path.join(OUT, "roles.json"), "w") as f:
    json.dump({r: groups[r] for r in groups}, f)
print(f"\nwrote {OUT}/brain.npz  ({os.path.getsize(OUT+'/brain.npz')/1048576:.1f} MB)")

# ---- GATE 1: synaptic drive onto each population ----------------------------
print("\nGATE 1 — in-circuit synaptic drive (must be hundreds, not single digits):")
absw = np.abs(w)
for r in sorted(groups):
    tgt = np.zeros(N, bool); tgt[groups[r]] = True
    indeg = absw[tgt[colidx]].sum()
    print(f"  onto {r:<16} {indeg:>10,.0f} syn   ({indeg/len(groups[r]):>8,.0f} per neuron)")

# ---- GATE 2: shortest path sweet GRN -> proboscis MN ------------------------
print("\nGATE 2 — shortest path, sweet GRN -> proboscis motor neuron:")
targets = set(groups["mn_proboscis"]) | set(groups["mn_haustellum"])
for src_role in ("grn_sweet", "grn_sweet_leg"):
    dist = np.full(N, -1, np.int32)
    q = deque()
    for i in groups[src_role]: dist[i] = 0; q.append(i)
    hit = None
    while q:
        u = q.popleft()
        if u in targets: hit = (u, dist[u]); break
        for k in range(indptr[u], indptr[u + 1]):
            v = colidx[k]
            if dist[v] < 0: dist[v] = dist[u] + 1; q.append(v)
    reach = sum(1 for t in targets if dist[t] >= 0)
    hops = [int(dist[t]) for t in targets if dist[t] >= 0]
    print(f"  {src_role:<14} -> {reach}/{len(targets)} MNs reachable, "
          f"min {min(hops) if hops else '-'} hops, median {int(np.median(hops)) if hops else '-'}")
