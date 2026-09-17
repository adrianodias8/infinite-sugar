# Environment — the other senses, touch, heat and cold

Run 2026-09-13. Reproduce with `./tools/fetch_flywire.sh && python3 tools/build_brain.py &&
python3 tools/export_web.py && python3 tools/response_matrix.py`, then `./serve.sh`.

## The sensory switches were already built

`web/brain.ts` mapped eight stimulus channels to real FlyWire populations, and the click handler
already accepted any `[data-stim]` button, but only sugar had one. The remaining channels now
sit in an environment strip under the main controls, and an Inspect table (Senses) shows what
every sensory pool is doing, with the active ones highlighted.

Measured in-browser with real DOM clicks, 2 s of simulation per switch, averaged over the last
second. Nothing here was tuned — these are the responses the wiring already produced
(with the predicted synapse signs; the table with the literature-corrected signs, which
replaced them on 2026-09-16, is in [16-next-moves.md](16-next-moves.md)):

```
                 pop  sensory  proboscis  antenna  neck L/R  gf  escwing L/R  steer L/R  groom
  rest           6.4       -        5       23     14/15    15      0/0        70/53      21
  sweet          7.0     146       24       26     15/14     6      0/0        73/53      22
  touch          9.8     166       15       52     15/34     5      0/0        38/52      35
  light         21.1     174        7       25     14/15    12      0/0        71/52      20
  looming        6.8     150/105    4       21     13/14    10    218/169      62/41      24
  odour         11.2     148        7       25     14/12     2      0/0        76/61      24
  bitter         6.7     130        2       24     14/16    13      0/0        75/59      22
  damp           6.6     103        7       25     15/15    12      0/0        70/51      21
```

Sensory = the driven pool's own rate (looming lists LC4/LPLC2). The signatures match
[docs/05](05-results-phase1-3.md): touch inverts the steering asymmetry and sweeps the antennae,
bitter clamps feeding below rest, looming is the only thing that wakes the escape wing DNs.

## Touch by tapping

A short tap on the fly (pointer down and up, under 8 px of travel and 350 ms) is raycast against
the visible fly geoms — the same meshes the render bridge syncs — and the hit geom's MuJoCo body
index says what was touched. A drag stays with OrbitControls. The raycast reads the scene and
never writes physics: `qpos` before and after a tap differ by exactly 0.

The result is a transient pulse on the mechanosensory population the Touch switch drives: held
at level 1 for 0.15 s of brain time, 0.30 s for the head and antennae (they carry Johnston's
organ), then released with a 0.5 s exponential. Stepped with the brain tick rather than wall
time, so a device below real time delivers the same pulse to the same neurons. FAFB has one
mechanosensory pool, so a poke cannot tell the fly *where* it was touched; region changes only
how long the hold lasts.

Stimulus levels now have two sources per channel, the sustained switch and the transient pulse,
and the brain receives their maximum through a single function. The buttons show the switch; the
Senses rows show what the brain receives. A head tap, traced:

```
  ms    touch  mechano  antennal MN  steer L/R  groom
    0    1.00      12        26        82/43      13
  100    1.00     164        52        42/46      49
  300    1.00     166        53        24/33      24
  500    0.67     120        52        32/87      28
 1000    0.25      44        34        62/63      35
 1500    0.09      15        30        65/26      14
```

## Hot and cold cells — one data-level change

`roles.json` had a single `thermo` pool of 29 neurons. FlyWire v783 labels the thermosensory
`sub_class` as `heating` (TRN_VP2, 7 cells), `cold` (TRN_VP3a/b, 9) and `humid` (TRN_VP1m, 13),
and the hygrosensory class carries `cooling` (HRN_VP1l, 13) and `evaporative_cooling`
(HRN_VP1d, 16) alongside `moist` and `dry`. `tools/build_brain.py` now derives three sub-pools
from those labels, the same way the motor pools are split by side:

| pool | FlyWire label | n |
|---|---|---|
| `thermo_hot` | thermosensory / heating | 7 |
| `thermo_cold` | thermosensory / cold | 9 |
| `hygro_cool` | hygrosensory / cooling + evaporative_cooling | 29 |

`thermo` and `hygro` stay as the unions. The graph is unchanged: `brain.npz` and the three
web blobs were regenerated and their payloads are byte-identical to the previous build; only
`roles.json` and `meta.json` carry the new groups. `reflex_test.py --sweep` reproduces the
earlier sweep. `heat` now drives `thermo_hot`; a new `cool` channel drives `thermo_cold` plus
`hygro_cool`.

**A readout bug this exposed.** The browser kernel assigned each neuron a single role for rate
counting, last pool wins, so a union pool lost every neuron that also belonged to a sub-pool:
`mn_neck`, `dn_steer` and `dn_escwing` have read 0 Hz since the side split. Rates are now counted
for both memberships (a neuron may be in a union and one sub-pool; a third is a hard error). This
is telemetry only; nothing about the dynamics changed.

## The hot cells are tonically inhibited

Driven at the standard 0.20 units/ms, the hot cells never fire: their membrane sits at −1.5,
near the floor, and the NumPy reference shows exactly zero downstream change. They receive
2,500 inhibitory units/s at rest — 0.36 per ms, more than the drive — from eight antennal-lobe
local neurons (lLN2F_a/b, lLN2T_b, lLN2X04) that fire at the 333 Hz refractory ceiling in this
kernel. The cold cells receive 1,400/s and do fire (53 Hz) at the same drive.

That is a property of the model, not of the fly, so it is handled as an explicit input gain
rather than by touching the kernel: `brain.stimGain.heat = 2.5` (every other channel stays at 1),
and `response_matrix.py` applies the same factor so the reference agrees.

```
  drive    hot cells   proboscis   neck   antenna
  0.20        0 Hz        2.2       7.2    22.4
  0.30       17          3.3      13.8    25.2
  0.40       95          4.9      15.0    24.1
  0.50      170          7.9      16.5    23.8
  0.70      320         26.3      19.9    18.0
```

## Response matrix with the new channels (NumPy reference, % change vs unstimulated)

```
  stimulus     pop |   proboscis   neck   antenna   dn_gf  escwing   steer   groom   pam
  (none)       5.7 |     12.1Hz  14.6Hz    8.9Hz   1.7Hz    0.0Hz  51.0Hz  25.1Hz  5.4Hz
  thermo       5.7 |       -14%    -3%      +1%     +0%     0.0      -1%    +6%    -1%
  heat 1x      5.7 |        +0%    +0%      +0%     +0%     0.0      +0%    +0%    +0%
  heat 2.5x    5.7 |       -15%   -17%      -1%    -50%*    0.0      -9%    +4%    +1%
  cold         5.7 |       -14%    -3%      +1%     +0%     0.0      -1%    +6%    -1%
  cool         5.8 |       -31%*  -11%      +3%    +50%*    0.0      -7%   +12%    +1%
  humid        5.8 |        -8%   -15%      +0%     +0%     0.0      -3%    +3%    +0%
```

`thermo` and `cold` are identical to the decimal: the hot and humid cells contributed nothing to
the old channel, so "heat" was, in effect, cold all along. In the browser, heat now reads 168 Hz
on the hot cells and cool 53 / 66 Hz on the cold and cooling cells; both leave the body nearly
still (neck and proboscis within a few Hz of rest), which is what the wiring says. They are
distinct in Inspect and were left honest rather than amplified.

## Light and looming as things that happen in the scene

**Light.** The Light switch still drives the 11,426 visual neurons (5 → 174 Hz); the scene now
brightens with it — hemisphere, key and rim lights up 45/55/30 %, the sky 25 %, fog lifted —
so the stimulus and what the viewer sees agree. Rendering only: the terrarium still sends
nothing to the brain, and the light level is the switch, not the picture.

**Looming.** The Looming switch is now an event: one press sends a matte dark sphere
(radius 0.12) from 1.9 units ahead-and-above the fly to 0.32 over 1.1 s, holds it there for
0.25 s, and pulls it back. A sphere rather than a disc because it reads the same from every
camera angle and casts a real shadow onto the fly as it arrives; the 19° view is narrow, so it
is in frame for the last half second. The looming stimulus follows its angular size,
`(r/d ÷ r/d_min)^1.5`, so LC4 and LPLC2 see the slow-then-sharp expansion of an object on a
collision course, and it releases with an 80 ms time constant after the pass. Stepped in brain time alongside the poke. A squared
ramp was tried first and held the peak for only ~0.1 s; the escape DNs reached 84/38 Hz, well
short of the 218/169 Hz a sustained loom gives them. The hold fixes that (measured, not tuned
by eye):

```
  ms      d    level   LC4   escwing L/R   wing yaw L
     0   1.90  0.07      1       0/0        0.71
   600   1.04  0.17     12      22/3        0.72
   900   0.61  0.38     40      72/23       0.56
  1000   0.46  0.58     70     113/56       0.55
  1100   0.32  1.00    131     201/129      0.22
  1150   0.32  1.00    146     217/156      0.16
  1400   0.32  0.53    105     171/121      0.16
  1500   0.32  0.15     18      30/13       0.41
  1800   1.90  0        1       0/0         0.70
```

The wings deploy and fold back exactly, through the real circuit: sphere → LC4/LPLC2 → DNp02/04/11
→ the supplied wing posture. This is the trigger flight builds on (see [flight](12-flight.md)).
