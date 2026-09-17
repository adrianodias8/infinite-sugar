# Next moves — what the fly-connectome community has built, and what it means for this fly

Written 2026-09-16 against [cobanov/awesome-fly](https://github.com/cobanov/awesome-fly), the
community list this project appears on. Two things from it are done below; the rest is a plan,
in order, with what each would cost and what it would make true.

## Done from the list

### 1. Literature-verified neurotransmitters (flyconnectome/drosophila_neurotransmitters)

The honesty budget has always said the synapse signs were predicted, not measured. The
community's ground-truth table (Eckstein et al.; 6,107 rows over 71 studies, CC-BY 4.0) names
the transmitter of a cell type where the literature knows it. `tools/build_brain.py` now
applies it: a neuron whose FlyWire cell type has one verified fast transmitter at confidence
≥ 3 gets that transmitter's sign on every outgoing synapse; histamine (the photoreceptors)
and glycine are inhibitory, the monoamines stay modulatory at 0.5, co-transmitting types and
nitric oxide keep the prediction. `tools/fetch_flywire.sh` fetches the table.

Measured on FAFB v783: 3,930 cell types matched, 77,351 neurons, 1,171,570 of 2,700,513 edges
covered, 92,818 signs changed, 79,870 flipped. The largest single correction is the eye:
8,456 R1-6 photoreceptors were predicted cholinergic or glutamatergic and are histaminergic,
so light now *inhibits* the lamina, as it does in the fly. The whole-brain resting rate went
from 6.4 to 7.5 Hz. The NumPy response matrix with the corrected signs
(`python3 tools/response_matrix.py`):

    stimulus     pop |    proboscis        neck     antenna       dn_gf  dn_escwing    dn_steer     dn_walk    dn_groom         pam
    (none)       6.4 |       9.2Hz     10.8Hz      9.9Hz      0.8Hz      0.3Hz     53.3Hz      0.8Hz     26.4Hz      4.0Hz
    sweet        6.7 |       +146%*       -19%        +12%        +50%*       +50%*        -1%         +0%         +0%         +3%
    sweet lab    6.6 |       +319%*       -17%        +14%       +100%*       +50%*        +0%        -50%*        -3%         -0%
    bitter       6.5 |        -43%*        +1%         +1%       +150%*       +50%*       +16%        +50%*        +0%         -0%
    odour       11.0 |        -26%*        +8%         -3%       -100%*     +2100%*       +16%        -50%*        +1%         -2%
    touch       10.1 |        +60%*      +121%*      +424%*       -50%*      -100%*        -9%        -50%*       +62%*        -1%
    heat 2.5x    6.5 |        -11%        +20%         +2%       +100%*     +1050%*       +13%       +200%*        -5%         -3%
    wind         7.1 |        -17%         +7%        -12%        +50%*      -100%*       +12%       +200%*       +65%*        +0%
    light       20.8 |        +14%         +5%         -6%         +0%        +50%*       +13%       +100%*        +2%         +0%
    looming      6.8 |         +5%         -5%        -29%*    +15750%*    +79100%*       -14%        +0%         -6%         -2%
    object       6.5 |         -1%         +3%         -2%        -50%*       -50%*       +29%*        +0%         -4%         -0%

Every reflex the piece rests on survives: sugar drives the proboscis (+146 % / +319 % labellar),
bitter clamps it (−43 %), looming alone wakes the escape wing DNs, touch turns the neck and
sweeps the antennae. Browser kernel, same graph: looming → escape DNs 252 / 199 Hz, escape mean
227 within 300 ms (takeoff at 100); sugar → proboscis 4 → 30 Hz.

Two things the corrected wiring changed, and what was done about them:

- **DNp09 rests at ~1 Hz, not 0.6.** Each spike requested 0.6 s of walking, set against the
  old rest, and the fly would have walked without pause (measured: 99 % of a two-minute run).
  The request is now rest-relative, like MDN's gate: 0.6 s × (0.6 / calibrated rest) per
  spike, so the resting drive asks for the same share of walking whatever the kernel's rest.
- **The resting-rate calibration was a 25 ms snapshot.** For a two-cell steering pool that is
  a coin toss (it read L 67 / R 81 against a true 76 / 58 and steered the fly in circles).
  `brain.calibrate` now takes the MEAN spike rate over the last 1.5 s of the 2.5 s calibration.
  This was wrong before the sign correction too; the correction made it visible.

Behavioural budget with the corrected signs (120 s on the perch disc): walking 84.8 % by day,
88.5 % at night; standing 12.1 / 8.0 %; 27 bouts, no flights (nothing loomed), the sack not
reached. Against 74 / 83 % before: the same fly, a little busier.

### 2. Wind on the antennae (after flyverse)

FlyWire labels 486 mechanosensory cells `wind_gravity` (Johnston's organ). The draught that
carries the odour now deflects the antennae: a `wind` channel (`mechano_wind`, split by side)
at 0.3 on the antenna the draught comes from, plus the fly's own airspeed in flight, and a
Wind switch in the strip. Measured in the wiring: wind raises DNg11 (grooming) +65 % and the
walking DN +200 % (one-sided: the wind cells 168 / 4 Hz).

## The plan, in order

### 3. A whole-CNS fly: BANC (largest fidelity gain) — feasibility run done, see below

The gait, the wingbeat and every leg movement here are supplied because FAFB is brain-only.
The **BANC** connectome (Harvard Dataverse, CC-BY 4.0, no login; ~188,000 neurons, 199 M
synapses, v888) is the *female* brain **and** ventral nerve cord: the leg and wing motor
neurons and their premotor circuits are in it. With it, DNp09 would drive real leg motor
neuron pools and their rates would drive flybody's leg actuators the way the proboscis motor
neurons already do, and "supplied gait" would become "measured leg motor output through a
supplied joint mapping". webgpu-fly did the nearest thing with the *male* VNC (MANC) matched
to FlyWire by cell-type name and 369 leg motor neurons averaged into six leg groups that
modulate a CPG. BANC removes the sex mismatch and the name matching. Cost: a new build
pipeline (BANC's tables → the same CSR graph), roles for the leg/wing motor pools, and the
honesty work of stating which of the 1.4× more neurons the browser can afford (the JS kernel
is ~1.1 s per simulated second for 139k). First step: a feasibility run of BANC in
`tools/response_matrix.py` (does a DNp09 drive raise the T1–T3 motor pools?).

**Feasibility run, 2026-09-17** (`tools/banc_feasibility.py`, from the public bucket
`lee-lab_brain-and-nerve-cord-fly-connectome`, v888 meta and the simple edgelist, both
CC-BY 4.0 and never committed). 175,401 neurons after dropping glia, 1,924,810 edges at ≥ 5
synapses, signs from BANC's predicted transmitter with its verified column overriding
(60,651 neurons). BANC labels the pools this project would need: DNp09, MDN, DNa01, DNa02,
DNp02/04/11, DNp01, DNg11 all present (2–5 cells each); 393 leg motor neurons by leg and by
muscle (101 tibia flexors, 62 trochanter flexors, 34 femur reductors, …), 24 wing power and
24 wing steering motor neurons, 15 haltere, 81 neck. The same NumPy LIF as
`response_matrix.py`, each DN driven at 3× the browser's standard drive, 1 s each:

    stimulus       pop |  front_leg  middle_leg   hind_leg  tibia_flex  femur_red  troch_flex  wing_power  wing_steer    neck  haltere
    (none)        12.0 |    15.9Hz     19.0Hz     13.0Hz      6.5Hz      3.9Hz      9.8Hz     88.6Hz     70.9Hz   18.8Hz   18.1Hz
    DNp09         12.0 |      -7%       -10%       -13%        -5%        -3%        -7%       -12%        -7%      +5%      -7%
    MDN           12.0 |      +2%        -6%        -9%        -6%       +49%       -16%        -5%        -2%      -3%      -8%
    DNa01         12.0 |      +5%        -6%        -8%        -3%        +2%       -12%        -1%        -1%      +2%      +4%
    DNa02         12.0 |      -2%        -4%       -11%        +0%       +44%       -10%        -6%        -5%     +10%      -7%
    DNp02+04+11   12.2 |      -7%       -14%        +4%       +30%       -36%       +15%       +47%       +19%      -3%     +76%
    DNp01         12.1 |      +3%        +2%        -0%        +8%        +8%        -5%        +3%        +0%      +2%      +0%
    DNg11         12.0 |      -2%        -3%        -3%        +8%       -10%       -16%        -2%        -6%      -7%      -5%

What it says. The escape command reaches the flight motor through the cord: DNp02/04/11 raise
the wing power motor neurons +47 %, the wing steering +19 % and the haltere +76 %, and flex the
tibiae +30 % (the jump). The backward-walking command and the steering DNa02 both raise the
femur reductors (+49 %, +44 %). The forward-walking command DNp09 does NOT raise any leg pool
in this integrator: two cells into a cord whose motor pools already rest hot (wing power at
89 Hz, three times this brain's whole-brain rate) are not enough, and the walking central
pattern generators are exactly the temporal circuits a memoryless LIF without synaptic time
constants underplays. So: a BANC fly would give this project real wing and haltere motor
output for its escape flight today, and honest measured leg output for backing and turning;
a connectome gait would need the cord's own rhythm, which needs the dynamics of item 6 or a
kernel with synaptic time constants (Shiu's 5 ms). The graph is 1.26× this one; the
WebAssembly kernel would carry it.

### 4. Speed: a worker thread, not a GPU — done ([17-threads-and-kernels.md](17-threads-and-kernels.md))

webgpu-fly's WebGPU kernel reaches 0.25 kHz on an M2 Pro (memory-bound); this JS kernel does
~0.9 kHz on the build container, so the GPU is not the win. The win is parallelism: the brain
(1.1 s / s) and the physics (1.4 s / s) run in series on one thread. A Web Worker for the
brain, exchanging one frame's worth of stimulus levels and rates through a SharedArrayBuffer
(cross-origin isolation headers on the server), would overlap them and approach the slower of
the two. Sensory levels would lag the body by one frame (16 ms), inside a real fly's
transduction delays. Cost: a day; the determinism test must still pass.

### 5. The eye from the measured lattice — blocked on data

The retinotopy is inferred by rank ([15-vision.md](15-vision.md)). Closed-Loop Fly samples the
scene at the connectome's 1,771 column directions (MaleCNS). For FlyWire the optic-lobe paper
(Matsliah et al. 2024) assigned columns to visual neurons; if its per-neuron column table is
obtainable (Codex is blocked from this build machine), `tools/build_retina.py` should take it
in place of the rank map. The FlyGym numbers to hold the field to: 721 ommatidia per eye,
~270° combined field, ~5° acceptance. **Checked 2026-09-17:** the FlyWire annotations
repository's supplemental files carry no column or hexagonal coordinates, the optic-lobe
matching repository only cross-matches cell types, and Codex (the one place a column table
would be) is unreachable from the build machine. The rank map stays until someone with Codex
access exports the column assignments; `build_retina.py` is written to take them.

### 6. Optic-lobe dynamics (flyvis) — blocked on data

The eyes deliver an image but a static dark sphere does not reach LC4 through the wiring:
motion detection needs the temporal dynamics that connectome-constrained models (flyvis,
Lappalainen et al. 2024, MIT) give each of 64 optic-lobe cell types — a time constant and a
resting potential per type, trained once against the connectome and published. Applying
published per-type constants to the optic-lobe cells is not learning in this simulation, but
it does break "one LIF for every neuron". Worth an experiment behind a flag: do LC4/LPLC2
then respond to a real looming sphere through the wiring, so the geometric looming channel can
be retired? **Checked 2026-09-17:** the trained per-type constants live in flyvis's PyTorch
checkpoints and its documentation site is unreachable from here; the repository README carries
no parameter table. Nothing honest can be applied without them.

### 7. Interaction (Help the Fly Escape, NeuroTerrarium) — the sack is done

The ball can be picked up; the sack could be too (drag it and the odour plume, the ground
map's footprint and the fly's target move with it), and a second object to place (a rock)
would let a visitor build the fly a maze the way the escape game does. Cost: small.
**Done for the sack** ([08-artwork-presentation.md](08-artwork-presentation.md)): it slides to
the pointer and snaps to floor its footprint fits, never onto the fly; the floor map, the
plume and the fly's target follow. A movable rock would use the same placement.

### 8. Cross-checks against the community's kernels — the parameters, side by side

Shiu et al.'s reference code, Eon's fly-brain and FastFly all run the same v783 graph. A
one-page comparison of resting rates and the sugar → proboscis response across kernels would
tell how much of what this fly does is the wiring and how much is the integrator. Cost: a
script and an afternoon, if their parameter files are readable.

**Read 2026-09-17**, Shiu et al.'s reference model (`model.py`, Brian 2) against this kernel:

| | Shiu et al. (Brian 2) | this fly (`web/brain.ts` / `lif.c`) |
|---|---|---|
| membrane time constant | 20 ms | 20 ms (`DECAY = e^(-1/20)` per 1 ms step) |
| rest / reset / threshold | −52 / −52 / −45 mV (7 mV to threshold) | 0 / 0 / 1.0 (membrane units) |
| refractory | 2.2 ms | 2 ms |
| synapse | 0.275 mV per synapse, 5 ms synaptic time constant, 1.8 ms delay | 0.005 units per synapse (3.9 % of threshold, vs 3.9 % of 7 mV = 0.275 mV: the same), instantaneous, excitation now / inhibition 4 ms later |
| sign | "Excitatory × Connectivity" column (predicted transmitter) | predicted transmitter, overridden by the literature ground truth where known |
| drive | Poisson spiking at 150 Hz in the stimulated neurons | +0.20 units per ms to the stimulated pool |
| noise / baseline | none | 300 random kicks of 0.42 per ms; a tonic per-neuron baseline U(0, 0.06) per ms |
| step | Brian 2's ODE integration | fixed 1 kHz, exact integer bookkeeping |

The synaptic weight is the one number that agrees by construction; the delays, the synaptic
time constant and the drive model differ, and this kernel adds the background noise that gives
it a resting state at all (Shiu's model is silent at rest). Running their Brian 2 code here is
left undone: the point of the table is that a response measured here should be read as this
integrator's, and checked against theirs before it is called the fly's.
