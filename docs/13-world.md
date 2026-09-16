# World — the terrarium as the source of the senses, and the fly deciding where to go

Run 2026-09-13. `npm test` runs `tools/world_test.mjs` and the walking checks in
`tools/flight_test.mjs`; `python3 tools/response_matrix.py --lateral` reproduces the table below.

## The change in premise

Until now every sense was a switch and the terrarium sent nothing to the brain. Now the terrarium
is the source of the senses, the switches are overrides, and where the fly goes is decided by
its own descending neurons. Sugar is off by default: it is a sack on the floor, placed on the
nearest open floor to a spot a short walk ahead of the perch, and the fly has to get there.

Nothing in the world is a claim about the fly's brain. It is the environment, made explicit,
feeding the same populations the switches drive — at the levels the environment gives them.

## Senses are split by side

FlyWire labels the side of every sensory neuron. `tools/build_brain.py` now splits the ORNs
(1,117 L / 1,134 R), mechanosensory (1,370 / 1,304), visual (5,892 / 5,458), LC4 (54 / 50),
LPLC2 (108 / 102), sweet and bitter GRNs and hygrosensory cells into left and right pools; the
30 unsided ORNs and 76 central visual cells stay in the unions. The graph is untouched (payloads
byte-identical); `roles.json` and `meta.json` carry 49 pools. The brain kernel takes a left and
a right level per channel: the shared part drives the union, the excess drives the stronger
side's pool, and equal levels are exactly the old uniform drive. Rates are now counted for up to
three memberships per neuron (union, sub-pool, side).

**Does a one-sided input give a one-sided output?** Measured in the NumPy reference, driving
the left half of a population alone (% change vs unstimulated; `*` = |change| ≥ 25 %):

```
  stimulus     pop |  neck_l  neck_r  antenna_l  antenna_r  steer_l  steer_r  escwing_l  escwing_r  dn_walk  dn_back
  (none)       5.7 |  17.4Hz  11.9Hz   10.7Hz     7.2Hz    47.9Hz   54.2Hz     0.0Hz      0.0Hz     0.0Hz    1.0Hz
  odour L      7.9 |    -9%    -38%*    -14%      +12%       -0%     -22%      0.0        0.0       0.8     -100%*
  odour both  10.1 |    -2%    -43%*    -17%      +23%       +9%     -28%*     0.0        0.8       0.4      -80%*
  touch L      7.6 |    +0%   +166%*   +384%*    +351%*     -20%     -22%      0.0        0.0       0.4      +80%*
  loom L       6.0 |    +3%     -3%     -27%*      +7%      -19%     +14%    227.2        0.0       0.8     +420%*
  loom both    6.2 |    -7%    -43%*    -39%*     -40%*     -12%     -22%    215.6      179.2       1.2      -20%
  light L     13.8 |   -12%    -19%      -6%       +0%       +4%     -18%      0.0        0.0       0.8      +20%
  sweet L      5.9 |    -8%    -63%*    +19%      +33%*     +12%     -18%      0.0        0.0       1.2      -40%*
```

Three things fell out of the wiring, none of them written by us:

- **A loom on the left drives the left escape wing DN alone** — 227 Hz against 0. The escape
  command is lateralised, so the escape-DN asymmetry is used in flight as a turn *away* from
  the threat.
- **A loom on the left shifts the steering DNs right** (L −19 %, R +14 %): the fly turns away.
  **An odour on the left shifts them left** (L 0 %, R −22 %): it turns toward the smell. Both
  are the signs a fly would show.
- **A loom on the left drives MDN**, the backward-walking neuron, +420 %: the fly backs off.

**The browser kernel does not reproduce the odour sign.** The same one-sided inputs measured in
the kernel the page actually runs (3 s means after calibration, Hz):

```
  condition        steer_l  steer_r  escwing_l  escwing_r  neck_l  neck_r  antenna_l  antenna_r  walk/s  back/s  pop
  rest               71.5     50.7      0          0        14.8    15.0     22.0       26.4      0.67    23.7    7.1
  odour L            71.2     61.5      0          0        14.7    15.7     22.5       27.3      1.0     20.7    8.1
  odour R            69.7     56.6      0          0        15.0    15.5     22.5       26.9      1.3     11.0    8.1
  loom L             64.3     57.4    226.2        0        14.5    16.4     20.6       25.2      3.0     20.3    6.7
  touch L            46.5     51.0      0          0        15.8    36.4     49.9       31.9      0.33    20.7    8.1
  noon (light .35)   69.6     50.9      0          0        14.1    14.6     22.0       26.7      1.5     19.0   11.8
  night (dark, cool) 70.8     55.4      0.1        0        15.0    16.2     22.5       27.1      2.25    24.0    6.6
```

An odour on either side raises the *right* steering DN (50.7 → 61.5 / 56.6) and leaves the left
where it is, so the fly does not turn toward a smell here; it wanders, and reaches the sack when
its wandering takes it there. The escape lateralisation holds (226 / 0), a touch on the left
turns the head and sweeps the left antenna, and the walking DN asks for a few more steps at
night (0.67 → 2.25 spikes/s) while noon raises the whole-brain rate through the visual cells.
Reported, not tuned.

**A fourth visual population.** LC11, the small-object motion detectors (127 cells, 66 L /
61 R), now has its own channel, `object`, driven by anything small crossing the view — the ball
rolling past — by its angular velocity about the fly rather than its approach. Measured
downstream (NumPy): the giant fiber −75 %, the proboscis −13 % (−27 % one-sided), the steering
DNs +17 %; nothing that moves a joint on its own, which is the honest result.

## What the world supplies

| sense | source | level |
|---|---|---|
| sweet (labellar) | the labellum touching the sack's surface: within 0.03 cm of its footprint | 1 in contact, falling to 0 over 0.03, × the sugar left |
| sweetLeg (tarsal) | feet in the sugar at the sack's base, within 0.04 cm of its footprint | the fraction of the six feet in it, × the sugar left |
| odour | the plume: the still-air fall-off from the sack, carried downwind by a fixed draught (toward the perch) in a cone 0.12 cm wide at the sack widening by 0.35 per cm; a quarter of the still-air level elsewhere. Each antenna samples the field 0.035 cm to its side of the head, and the antenna facing the sack gets the larger share | 0.8 at the patch, zero 0.8 cm beyond it, × the sugar left |
| light | in the page: the eyes — the terrarium rendered from the head, each of the 11,424 photoreceptors driven by the luminance in its own direction plus a transient from its rate of change ([docs/15-vision.md](15-vision.md)). Without a renderer (the tests, the first frames): a 120 s day, starting at noon, shaded by anything between the fly and the sun, plus the same transient term | per cell: 0.35 × luminance + 0.05 × \|d lum/dt\| (50 ms filtered), capped at 1; scalar path: 0.35 × daylight × (1 − shade) + the transient |
| heat / cool / damp | the same day | hot cells above 65 % daylight, cold, cooling and hygrosensory cells below 35 % |
| looming | the beach ball, by time to collision from its real position and velocity, on the eye it approaches; the loom sphere by its own event | 1 − τ/0.6 s, scaled by angular size, within 1.2 cm |
| object (LC11) | a small object crossing the view, by its angular velocity about the fly, on that side | (ω / 2 rad/s), fading as it fills the view, within 1.5 cm |
| touch | the ball hitting the fly (cannon-es contact), on that flank; a tap | the poke pulse |
| bitter | the labellum against foliage or a flower (plant cells of the floor map, found by the glTF's material colours) | 1 in contact |

**Tasting is by contact, and the reflex is not there.** Labellar and tarsal sugar sensing are
now separate channels (`grn_sweet`, 129 cells; `grn_sweet_leg`, 74). Measured, the labellum
hangs 0.05 cm under the head and the proboscis extends it only 0.02 further down, so from a
standing body it never reaches the floor; it tastes the sack by touching its side. Real flies
find sugar with their feet and the tarsal input triggers proboscis extension. Measured in this
wiring (NumPy reference), labellar drive alone raises the proboscis motor neurons +205 %, while
tarsal drive alone *lowers* them 45 %, and both together give the +103 % reported before. There
is no proboscis-extension reflex from the feet in this model, and none is faked: the fly tastes
with its feet at the sack's base, and feeds only when it faces the sack and its labellum meets
the surface. The Sugar switch still drives both channels.

**The sugar is finite.** Feeding at full labellar contact empties the sack in 180 s and an
untouched sack refills in 600 s (a world property; the sack slumps to half height on screen
as it empties). Taste and smell scale with what is left, so the feeding counter simply stops
when there is nothing to taste, and the fly's day has a story: find, feed, wander, find again.

**The odour is a plume.** A radial gradient smells the same from every direction; a real
plume has a downwind axis. A fixed draught blows from the sack toward the perch, so a fly at
the start is downwind of the sugar. Measured half a centimetre downwind on the axis: 0.36;
the same distance upwind: 0.09; 0.12 cm off the axis the antenna nearer the axis smells more
(0.31 against 0.23). The browser kernel still turns toward no smell (below), so the plume
changes what the antennae report, not where the fly goes.

**The ball is alive.** The floor under the ball leans 0.05 rad on a slow 45 s circle (the
gravity vector in cannon-es, and nothing else), so the ball keeps rolling somewhere and the
fly gets recurring looming and touch events without a hand on the button. Cosmetic and
supplied; measured over 40 s of wall time it drifted 0.14 cm down the hill and stayed 1.1 cm
from the centre, inside the glass.

The light cap: 11,426 visual neurons driven flat out triple the whole-brain rate, and real
photoreceptors adapt to steady light, which this model cannot (adaptation is ruled out by
design). A fractional level is the explicit stand-in. The plants are the bitter source: the
flowers, the foliage and the fern are told apart from the hill by the glTF's material colours
(every mesh in this export is auto-named), and a labellum in a plant cell drives the bitter GRNs
(labellar only; FAFB has no leg bitter pool). Measured: proboscis −55 %, giant fiber +25 %, so a
fly that walks into the flowers pulls its mouthparts in.

**The folded wing rests on the hind tibia.** Re-measured with the whole body standing across six
resting wing biases: none inside the actuator band clears the wing from the hind leg (the roll
joint saturates near 1.0 rad). Roll 0.0018 / yaw −0.0020 had the least neural-off jitter
(0.022 rad/s against 0.058), but after a landing the tibia caught the more outward-swept
membrane and held both wings forward at 0.56 rad instead of the folded 0.8–0.9. The original
bias stays; the contact is real geometry and is left as it is. A raised standing pose for the
hind legs was measured too (femur −0.03 to −0.1 rad, tibia +0.1 to +0.2, and both): none clears
the contact (39–40 of 40 ticks touching in every case) and each lowers the body by 0.002–0.011
with no change in the hind claw height, so the standing pose stays as the model gives it.

Every level is brain-tick state, so a slow device still feeds the same world to the same brain.

**Physics runs five substeps per brain millisecond**, not ten. flybody's XML says 0.1 ms; the
model is set to 0.2 ms at reset (`PHYS_SUBSTEPS`, the upstream file untouched). Measured
standing 3 s: height drift 0.0001 cm, root velocity 0.0025 at rest, every suite unchanged;
the solver's iteration count made no difference (the Newton solver converges early). Cost per
simulated second in node on the build machine: physics 2.8 → 1.4 s, the brain 1.1 s, the world,
actuators and gait under 0.06 s; a simulated second went from 4.7 to about 3.3 s of wall time.
The brain's kernel is the next cost and stays JavaScript for now. A speed control (1× / 2× /
4×) only changes how much simulated time a frame asks for; Inspect shows what the machine
achieves.

## The fly decides

| decision | neurons | measured |
|---|---|---|
| walk forward | DNp09 (`dn_walk`, 2 cells): each spike requests 0.6 s of steps, up to 2 s | 0.6 Hz at rest in the browser kernel |
| walk backward | MDN (`dn_back`, 4 cells): a 100 ms mean above 2.5 × its calibrated rest | ~6 Hz per cell at rest in the browser kernel, 17 spikes/s for the pool (1 Hz in NumPy), hence rest-relative |
| turn | DNa01/DNa02 (`dn_steer_l/r`): (L/rest_L − R/rest_R) / 0.6, clamped | L 70 / R 53 Hz at rest; the resting bias now reads as straight |
| take off | DNp02/04/11: 100 ms mean of the stronger side above 100 Hz (a loom on one flank drives one side alone) | 0 at rest, ~200 under looming |
| turn away | DNp02/04/11 asymmetry: −(L − R)/200 | 227 / 0 for a loom on the left |
| land | escape DNs under 30 Hz for 4 s | 0 at rest |

Supplied, and not claims about the fly: a tripod stepping cycle at 2.2 Hz with the body carried
at 0.22 cm/s (0.13 backward) 0.006 cm above its stance, the 24 Hz wingbeat, airspeed, cruise
height, bank, the leg tuck, wall avoidance, and the landing approach. In the stepping cycle each
foot lifts through its swing half while its fore-aft joint carries it forward on a cosine, and
is planted through its stance half while that joint carries it back *linearly* at the body's
speed, so stance feet stay put under the moving body instead of sliding. The fore-aft joints
are coxa twist for the front and middle legs and femur twist (opposite sign) for the hind legs.
Each leg's stride amplitude is derived from its own foot travel per radian (`WALK.cmPerRad`,
0.076 / 0.148 / 0.077 cm per rad for T1 / T2 / T3), calibrated with the foot planted so the
servo's lag under load is included: measured along-heading slip per stance is within 0.003 cm
for every leg (the six-leg mean cancels the real brain's steering wander). What remains is the
arc of a single twist joint, a sideways component of 0.01 (T2) to 0.06 cm (T1, T3) per stance,
which one joint per leg cannot straighten. A walking bout ends where the feet are: the stride
blends out over 0.15 s while the body sinks its 0.006 lift, and the root is handed back as soon
as the legs carry the weight — no landing crouch (measured: on the ground 0.15 s after the
bout, no dip below the standing height).

**Grooming** is a further command mapping: DNg11 (`dn_groom`) is the identified antennal-grooming
descending neuron and rises under touch (22.9 → 33.8 Hz). When its 200 ms mean exceeds 1.35 ×
its calibrated rest while the fly stands, the front legs lift and rub forward over the head for
0.9 s at 4 Hz (the motion is ours); a quiet DNg11 never grooms. Walking stays on the
walkable floor: a 56 × 56 map sampled at load from the terrarium's own opaque meshes (floor
hits between −0.10 and +0.10; the hill, rocks, plants, frame and the sack are obstacles). The
body has a size: each floor cell records its distance to the nearest obstacle, and a cell is
clear for the root only when that distance exceeds `WORLD.bodyRadius` (0.14 cm; the thorax is
0.05 wide, the legs reach 0.18, the folded wings 0.21 behind). The root lands only on clear
cells and walks onto clear cells freely; from a tight spot — the perch is 0.1 from the fern's
base, so the start is not clear — it may step onto floor no nearer an obstacle than where it
stands, so it walks out but never further in, and never onto anything but floor. The sugar
sack is exempt from the erosion (soft; the fly leans on it to feed) and only its own
footprint is kept off-limits. Each obstacle cell also records how tall the thing is (the
first hit from above; the sack its own height), so a flight keeps `WORLD.bodyClearance`
(0.15, the body's half-height) above whatever is under it and under the point 0.35 cm ahead,
climbing when it can, and what is too tall to clear under the roof (the tall plants reach
3.8 cm) it slows for and turns away from, the direction chosen once per block so two tall
sides do not dither; the next step is never taken onto such a cell, whatever the target says.
A landing approach blocked for 2 s re-aims at the nearest spot the body fits on. A bout that
meets an obstacle is spent turning toward open floor (supplied), so the next bout can go
somewhere. A flight lands where it is if that is floor, else on the nearest floor cell. The root is kinematic while walking or flying and handed back to physics at
the standing pose over loaded legs, as before.

## Measured

`tools/world_test.mjs`, real brain and body: odour on the left only drives the left ORNs (148 vs
3 Hz); a sack whose side the labellum touches is tasted and smelled on both antennae, a foot at
its base tastes with the tarsus, two body lengths away neither; a sack to the left is smelled on
the left antenna only (0.12 / 0.00, off the plume's axis); downwind of the sack the plume is
four times the upwind level and the antenna nearer its axis smells more; the labellum on the
sack drives the feeding motor neurons and the counter, and spends the sugar (0.992 left after
1.5 s); a quick sack empties, is neither tasted nor smelled when empty, freezes the counter, and
refills while nobody feeds; a raised DNg11 lifts and rubs the front legs and a resting one never
does; noon lights and warms, midnight is dark, cold and damp; a shadow arriving over 0.1 s is a
visual transient of 0.88 against the 0.35 daylight; an object between the fly and the sun
shades it to zero; a ball
closing from the left looms on the left eye only (0.50 / 0.00) and a receding ball not at all; a
bump from the ball is a touch on that flank and releases; switching the world off clears every
level.

Clearance, measured: the terrarium's 929 floor cells erode to 600 the body fits on; in a
30 s headless run with two flights and eleven walking bouts the root was never on a cell it
did not fit (standing or walking) except the perch it started on and its first step away,
and never closer than 0.15 above what was under it in the air (`tools/world_test.mjs` and `tools/flight_test.mjs` cover the erosion, the top map, a ridge
flown over with the clearance kept, and a wall too tall for the glass never approached within
the body's radius). Before this the root could be carried through a rock or a plant: the floor
map was checked at the point under the root only, and the air had no obstacles at all.

`tools/flight_test.mjs`: a DNp09 spike starts a bout and the fly walks forward along its
heading; a scripted MDN spike walks it backward; a stronger left steering DN turns it left; at
the edge of the floor it turns instead of leaving; a landing ends on walkable floor at the standing height with the
wings folded and the shuffle resumes; a threat on the left turns the flight right.

Browser, 60 s of the fly's own life from a cold start (world on, Sugar switch off), sampled every
2 s; `taste` is the world's sweet level, `odour` the left/right ORN levels, `counter` the
feeding counter:

```
   t   state    x      y    sugar dist  taste  odour L/R  daylight  counter  pill
   2s  ground   0.12   0.00    0.26      0     0.71/0.45    1.00        0
   6s  walk     0.60   0.01    0.25      0.1   0.72/0.32    0.98     2608  Walking
  10s  ground   0.92   0.03    0.54      0     0.43/0.06    0.93     4391
  14s  ground   0.64   0.10    0.16      1     0.61/0.80    0.87     6028  Feeding
  18s  ground   0.56   0.12    0.08      1     0.70/0.80    0.79    11757  Feeding
  22s  ground   0.19  -0.01    0.31      0.2   0.67/0.59    0.70    16934  Feeding
  28s  ground  -0.48  -0.19    1.00      0     0/0          0.55    17615
  44s  walk    -0.89   0.77    1.52      0     0/0          0.17    17615  Walking
  60s  walk    -0.42   0.90    1.15      0     0/0          0.00    17615  Walking
```

Nobody pressed anything. The fly smelled the sack ahead and to its left, walked to it in bouts,
stood on it with its feet and mouthparts in the sugar for eight seconds while the feeding motor
neurons fired and the counter rose, walked off, and spent the rest of the minute wandering the
floor and turning at obstacles while the day faded to night. A first version of this run got
stuck facing an obstacle after the sack, because a blocked bout simply ended; a blocked bout is
now spent turning toward open floor. A loom during that first run did not launch the fly: the
sphere came from its flank, drove one escape DN alone, and the mean of the two sides sat just
under the trigger — hence the stronger-side rule above.

After the contact rule, an unattended 20 s run: the fly walked from the perch to the sack's base
in three bouts and stood with its front feet in the sugar at 6 s and again at 10 s ("Tasting with
the feet"; the tarsal cells drove, the counter did not move, since it counts the labellum), then
passed the sack on its right without turning to face it and walked on. Whether it turns to feed
is the steering DNs' response to the odour on one antenna, which is real but small (R −22 %).

## Known limits

- The world is what is modelled: no wind, no other flies, no water. A plant tastes bitter
  wherever it stands in its cell, from the floor up.
- The ground map is a height map: the fly walks around obstacles by stopping at them, and a
  flight can pass through the fern or the flowers.
- Locomotion rates are those of this kernel. MDN being tonic here is a property of the model,
  handled by reading it against its own rest, as the other resting pools are.
