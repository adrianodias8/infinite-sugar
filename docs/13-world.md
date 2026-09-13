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

## What the world supplies

| sense | source | level |
|---|---|---|
| sweet (labellar) | the labellum touching the sack's surface: within 0.03 cm of its footprint | 1 in contact, falling to 0 over 0.03 |
| sweetLeg (tarsal) | feet in the sugar at the sack's base, within 0.04 cm of its footprint | the fraction of the six feet in it |
| odour | distance from the head to the patch, on the antenna facing it | 0.8 at the patch, zero 0.8 cm beyond it |
| light | a 120 s day, starting at noon; anything between the fly and the sun shades it | 0.35 × daylight × (1 − shade) |
| heat / cool / damp | the same day | hot cells above 65 % daylight, cold, cooling and hygrosensory cells below 35 % |
| looming | the beach ball, by time to collision from its real position and velocity, on the eye it approaches; the loom sphere by its own event | 1 − τ/0.6 s, scaled by angular size, within 1.2 cm |
| touch | the ball hitting the fly (cannon-es contact), on that flank; a tap | the poke pulse |

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

The light cap: 11,426 visual neurons driven flat out triple the whole-brain rate, and real
photoreceptors adapt to steady light, which this model cannot (adaptation is ruled out by
design). A fractional level is the explicit stand-in. Bitter has no world source.

Every level is brain-tick state, so a slow device still feeds the same world to the same brain.

## The fly decides

| decision | neurons | measured |
|---|---|---|
| walk forward | DNp09 (`dn_walk`, 2 cells): each spike requests 0.6 s of steps, up to 2 s | 0.6 Hz at rest in the browser kernel |
| walk backward | MDN (`dn_back`, 4 cells): a 100 ms mean above 2.5 × its calibrated rest | 17 Hz at rest in the browser kernel (1 Hz in NumPy), hence rest-relative |
| turn | DNa01/DNa02 (`dn_steer_l/r`): (L/rest_L − R/rest_R) / 0.6, clamped | L 70 / R 53 Hz at rest; the resting bias now reads as straight |
| take off | DNp02/04/11: 100 ms mean of the stronger side above 100 Hz (a loom on one flank drives one side alone) | 0 at rest, ~200 under looming |
| turn away | DNp02/04/11 asymmetry: −(L − R)/200 | 227 / 0 for a loom on the left |
| land | escape DNs under 30 Hz for 4 s | 0 at rest |

Supplied, and not claims about the fly: a tripod stepping cycle at 2.2 Hz with the body carried
at 0.22 cm/s (0.13 backward) 0.006 cm above its stance, the 24 Hz wingbeat, airspeed, cruise
height, bank, the leg tuck, wall avoidance, and the landing approach. In the stepping cycle each
foot lifts through its swing half while its fore-aft joint carries it forward, and is planted
through its stance half while that joint carries it back, so stance feet stay put under the
moving body instead of sliding. The fore-aft joints were measured on a lifted foot: coxa twist
for the front and middle legs (±0.3 rad moves the foot 0.014–0.05 fore-aft), femur twist with
the opposite sign for the hind legs.

**Grooming** is a further command mapping: DNg11 (`dn_groom`) is the identified antennal-grooming
descending neuron and rises under touch (22.9 → 33.8 Hz). When its 200 ms mean exceeds 1.35 ×
its calibrated rest while the fly stands, the front legs lift and rub forward over the head for
0.9 s at 4 Hz (the motion is ours); a quiet DNg11 never grooms. Walking stays on the
walkable floor: a 56 × 56 map sampled at load from the terrarium's own opaque meshes (floor
hits between −0.10 and +0.10; the hill, rocks, plants, frame and the sack are obstacles). A
bout that meets an obstacle is spent turning toward open floor (supplied), so the next bout can
go somewhere. A flight lands where it is if that is floor, else on the nearest floor cell. The root is kinematic while walking or flying and handed back to physics at
the standing pose over loaded legs, as before.

## Measured

`tools/world_test.mjs`, real brain and body: odour on the left only drives the left ORNs (148 vs
3 Hz); a sack whose side the labellum touches is tasted and smelled on both antennae, a foot at
its base tastes with the tarsus, two body lengths away neither; a sack to the left is smelled on
the left antenna only (0.46 / 0.00); the labellum on the sack drives the feeding motor neurons
and the counter; a raised DNg11 lifts and rubs the front legs and a resting one never does; noon lights and warms,
midnight is dark, cold and damp; an object between the fly and the sun shades it to zero; a ball
closing from the left looms on the left eye only (0.50 / 0.00) and a receding ball not at all; a
bump from the ball is a touch on that flank and releases; switching the world off clears every
level.

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

- The world is what is modelled: no taste of the plants, no wind, no other flies. Bitter has
  no source.
- The ground map is a height map: the fly walks around obstacles by stopping at them, and a
  flight can pass through the fern or the flowers.
- Locomotion rates are those of this kernel. MDN being tonic here is a property of the model,
  handled by reading it against its own rest, as the other resting pools are.
