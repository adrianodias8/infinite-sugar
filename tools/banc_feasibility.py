#!/usr/bin/env python3
"""BANC (female brain + nerve cord, v888) feasibility: build the same CSR graph the browser uses
from the public bucket's simple edgelist, tag the descending neurons this project already
commands and the leg / wing / neck motor pools BANC labels, and drive each DN in the NumPy LIF
of tools/response_matrix.py to see whether the wiring moves the legs. Read-only research: the
browser still runs FAFB v783. Usage: python3 tools/banc_feasibility.py <dir with the feathers>
"""
import sys, os, json, numpy as np, pandas as pd, pyarrow.feather as feather
D = sys.argv[1] if len(sys.argv) > 1 else "data/banc"
MIN_SYN = 5
meta = feather.read_feather(f"{D}/banc_888_meta.feather")
meta = meta[meta["super_class"].astype(str) != "glia"]
meta = meta[~meta["super_class"].astype(str).isin(["not_a_neuron", "trachea"])]
ids = meta["root_888"].astype(np.int64).to_numpy(); N = len(ids); idx = {int(r): i for i, r in enumerate(ids)}   # the edgelist uses the v888 roots
print(f"BANC neurons kept: {N:,}")
NT = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0, "dopamine": 0.5, "serotonin": 0.5, "octopamine": 0.5, "histamine": -1.0, "glycine": -1.0, "tyramine": 0.5}
nt = meta["neurotransmitter_predicted"].astype(str).str.lower().map(NT).fillna(1.0).to_numpy(np.float32)
ver = meta["neurotransmitter_verified"].astype(str).str.lower().map(NT)
nt = np.where(ver.notna(), ver.fillna(1.0).to_numpy(np.float32), nt)
print(f"verified transmitters on {int(ver.notna().sum()):,} neurons")
cache = f"{D}/banc_graph.npz"
if os.path.exists(cache):
    g = np.load(cache); indptr, colidx, weight = g["indptr"], g["colidx"], g["weight"]
else:
    t = feather.read_table(f"{D}/banc_888_edgelist_simple_v3.feather", columns=["pre", "post", "count"]).to_pandas()
    t = t[t["count"] >= MIN_SYN]
    pre = t["pre"].astype(np.int64).map(idx); post = t["post"].astype(np.int64).map(idx); keep = pre.notna() & post.notna()
    pre = pre[keep].astype(np.int64).to_numpy(); post = post[keep].astype(np.int64).to_numpy(); syn = t["count"][keep].to_numpy(np.float32)
    order = np.lexsort((post, pre)); pre, post, syn = pre[order], post[order], syn[order]
    indptr = np.zeros(N + 1, np.int64); np.add.at(indptr, pre + 1, 1); indptr = np.cumsum(indptr)
    colidx = post.astype(np.int32); weight = (syn * nt[pre]).astype(np.float32)
    np.savez_compressed(cache, indptr=indptr, colidx=colidx, weight=weight)
E = len(colidx); print(f"edges (>= {MIN_SYN} synapses): {E:,}")
ct = meta["cell_type"].astype(str).to_numpy(); sub = meta["cell_sub_class"].astype(str).to_numpy(); cls = meta["cell_class"].astype(str).to_numpy(); sc = meta["super_class"].astype(str).to_numpy()
G = {}
for name in ["DNp09", "MDN", "DNa01", "DNa02", "DNp02", "DNp04", "DNp11", "DNp01", "DNg11"]: G[name] = np.where(ct == name)[0]
for name, key in [("front_leg_mn", "front_leg_motor_neuron"), ("middle_leg_mn", "middle_leg_motor_neuron"), ("hind_leg_mn", "hind_leg_motor_neuron"),
                  ("wing_power_mn", "wing_power_motor_neuron"), ("wing_steer_mn", "wing_steering_motor_neuron"), ("haltere_mn", "haltere_steering_neuron")]:
    G[name] = np.where(sub == key)[0]
G["neck_mn"] = np.where(np.char.startswith(sub.astype(str), "neck_"))[0]
G["leg_mn"] = np.concatenate([G["front_leg_mn"], G["middle_leg_mn"], G["hind_leg_mn"]])
G["all_mn"] = np.where(sc == "motor")[0]
print("pools:", {k: len(v) for k, v in G.items()})
# leg motor neurons by muscle group, for the gait question
muscle = {}
for i in G["leg_mn"]:
    key = ct[i]
    for m in ("tibia_flexor", "tibia_extensor", "femur_reductor", "trochanter_flexor", "trochanter_extensor", "tarsus_depressor", "tarsus_levator", "long_tendon", "sternal", "tergotrochanter"):
        if m in key: muscle.setdefault(m, []).append(i); break
for k in muscle: G["leg_" + k] = np.array(muscle[k])
print("leg muscle pools:", {k: len(v) for k, v in G.items() if k.startswith("leg_") and k != "leg_mn"})
# ---- the same LIF as tools/response_matrix.py
DECAY = np.float32(np.exp(-1 / 20)); TH = np.float32(1.0); RF = 2; ID = 4; WS, BASE, DRIVE = 0.0050, 0.06, 0.20
def run(ms, stim, seed=1, warm=400, gain=1.0):
    rng = np.random.default_rng(seed); w = (weight * WS).astype(np.float32)
    v = np.zeros(N, np.float32); refr = np.zeros(N, np.int8); baseline = rng.uniform(0, BASE, N).astype(np.float32)
    inhq = [np.zeros(N, np.float32) for _ in range(ID + 1)]; cnt = np.zeros(N, np.int64)
    tgt = np.concatenate([G[k] for k in stim]) if stim else None
    for t in range(-warm, ms):
        s = (t + warm) % (ID + 1); q = inhq[s]; v += q; q.fill(0); np.maximum(v, -2, out=v)
        a = refr == 0; v[a] = v[a] * DECAY + baseline[a]; v[~a] *= DECAY; refr[~a] -= 1
        if tgt is not None and t >= 0: v[tgt] += DRIVE * gain
        v[rng.integers(0, N, 300)] += 0.42
        sp = np.where(v >= TH)[0]
        if len(sp):
            v[sp] = 0; refr[sp] = RF
            if t >= 0: cnt[sp] += 1
            for i in sp:
                a0, b0 = indptr[i], indptr[i + 1]; js = colidx[a0:b0]; ws = w[a0:b0]
                ex = ws >= 0
                np.add.at(v, js[ex], ws[ex]); np.minimum(v, 2.5, out=v)
                np.add.at(inhq[(s + ID) % (ID + 1)], js[~ex], ws[~ex])
    return cnt * 1000.0 / ms
OUTS = ["front_leg_mn", "middle_leg_mn", "hind_leg_mn", "leg_tibia_flexor", "leg_femur_reductor", "leg_trochanter_flexor", "wing_power_mn", "wing_steer_mn", "neck_mn", "haltere_mn"]
INS = [("(none)", []), ("DNp09", ["DNp09"]), ("MDN", ["MDN"]), ("DNa01", ["DNa01"]), ("DNa02", ["DNa02"]), ("DNp02+04+11", ["DNp02", "DNp04", "DNp11"]), ("DNp01", ["DNp01"]), ("DNg11", ["DNg11"])]
MS = 1000
print(f"\n  {'stimulus':<12}{'pop':>6} | " + "".join(f"{o.replace('_mn', '').replace('leg_', ''):>14}" for o in OUTS))
base = None
for name, stim in INS:
    r = run(MS, stim, gain=3.0)   # each DN is 2-4 cells; the browser drives command pools at this gain for its measurements
    pop = r.mean(); outs = [r[G[o]].mean() if len(G[o]) else float('nan') for o in OUTS]
    if base is None: base = outs; print(f"  {name:<12}{pop:>6.1f} | " + "".join(f"{b:>12.1f}Hz" for b in outs)); continue
    print(f"  {name:<12}{pop:>6.1f} | " + "".join(f"{(100 * (o - b) / b if b > 0 else float('nan')):>+12.0f}% " for o, b in zip(outs, base)))
print("\n  rows are % change vs (none); DNs driven at 3x the browser's standard drive (they are 2-4 cells each)")
