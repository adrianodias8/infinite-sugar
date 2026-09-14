# Flight — a command mapping, and the limits of it

Run 2026-09-13. `npm test` runs `tools/flight_test.mjs`; the browser cycle below is from headless
Chromium with a real click on Loom.

## The claim, stated first

FAFB is brain-only. The wing motor neurons, the flight power muscles and the ventral nerve cord
circuit that generates the wingbeat are not in the dataset, so flight here is a **command
mapping** — the weakest tier in this project (see [docs/05](05-results-phase1-3.md), Phase 4c) —
and deliberately the most visible one. What the brain contributes is real and measured; the rest
is supplied and says nothing about the fly.

| the brain decides | measured by |
|---|---|
| **takeoff** — DNp02/04/11 (`dn_escwing`), silent at rest, ~200 Hz under looming. A 100 ms mean of both sides above 100 Hz launches the fly. | 0 ± 0 Hz at rest across every other stimulus; 218/169 Hz under looming |
| **steering** — DNa01/DNa02 (`dn_steer_l/r`). Their documented function is asymmetric modulation of left/right wing amplitude. Yaw rate ∝ (L − R) / 60 Hz, and each wing's stroke amplitude follows its side. | L 70 ± 20 Hz, R 53 ± 19 Hz at rest, inverted by touch |
| **landing** — when the escape DNs stay under 30 Hz for 4 s after at least 3 s in the air. | escape DNs return to 0 Hz once the loom passes |

| supplied | value |
|---|---|
| wingbeat | 24 Hz stylized stroke: yaw sweeps 0.35 ± 0.85 rad, roll 1.0 ± 0.35, pitch −0.7 ± 0.35 (a real 200 Hz stroke is invisible at 60 fps) |
| airspeed, height | 0.5 cm/s, cruise 0.45 cm above the perch; altitude wanders on a seeded random walk (white kicks filtered into a slow vertical drift, pulled back to cruise over 2 s, clamped ±0.08) rather than a sine, so there is no single peak to spot (stylized; the terrarium is small) |
| bank, pitch | 0.35 rad into turns, led by yaw acceleration (bank ∝ yaw rate + 0.12 × yaw acceleration); 0.15 nose-down at speed |
| takeoff | 60 ms push-off first: the front and middle legs extend (the tuck offsets in reverse) and the body rises 0.012 with the wings still folded, then the climb and the wingbeat start |
| legs | front and middle legs tuck using the shuffle joints; hind legs stay at their standing targets |
| bounds | 72 % of the glass's inner radius, below its roof; a soft turn toward the centre inside a 0.3 band |
| landing path | fly back over the perch, descend, settle, touchdown |

The steering DNs' resting asymmetry is real, so the fly circles left. That is the data, not a
choice, and it is left as it is.

## Physics

Root motion is kinematic: the free joint's `qpos` and `qvel` are written every physics step,
so the body is carried rather than lifted by any force. Legs, head, proboscis, antennae and
abdomen keep simulating under their own neural drive throughout. During flight the six wing
joints are written too; on the ground they return to the neural mapping in `DRIVE`. This
relaxes the ground rule that nothing translates the root, for the duration of a flight only.

Near the glass, the soft avoidance turn is joined by a looming pulse proportional to proximity,
capped at 0.2, so the brain's own escape circuit sees the wall coming. The cap matters: at the
first value the wall pulse drove the escape DNs to ~50 Hz, above the calm threshold, so a fly
circling near the glass could never land — a closed loop that only the number fixed.

## Landing was the hard part

Four attempts, each one a physics lesson, all recorded because the next person will hit them:

1. **Released at the standing pose, wings at the flight base pose.** The wing tips sat on the
   floor and against the hind legs; the 0.01 N·cm springs could not fold them back.
2. **Wings blended to the folded pose, root teleported to the perch.** The body dropped onto
   unloaded legs, landed 0.013 lower than the standing stance, and the folded wing tips touched
   the floor at that height.
3. **Root pinned at the standing height with the legs at their targets.** Unloaded position-servo
   legs stand taller than the loaded stance: the feet were driven into the very stiff floor
   (`solref 0.0002`) and the stored contact force threw the body sideways on release.
4. **Descending while the wings still beat, hind legs tucked.** The tips hit the floor, and the
   tucked hind femora sat inside the wing stroke, pinning the right wing.

What works, in order: while still high, stop the stroke and unfold the front and middle legs
(hind legs never tuck); descend from the approach height to the standing height; fold the
wings over legs that are already in their standing configuration; then pin the root at the
standing height with the legs 12 % flexed so nothing penetrates, extend them slowly onto the
floor, and hand the root back the moment the constraint force on its vertical dof equals the
body's weight (0.966 in model units) — a static equilibrium, so nothing is stored to launch it.

```
  settle   t 0.30  flap 0.00  fold 0.01  z 0.103   wings 0.36 1.00 -0.70
  settle   t 0.57  flap 0.00  fold 0.99  z -0.005  wings 0.84 1.05 -0.90
  touchdown t 0.09           Fz 1.02 (weight 0.97)
  ground   +0.3 s            z -0.005 (standing -0.007), wings 0.85 1.03 -0.85
```

## Measured

`tools/flight_test.mjs`, real brain and body: looming on → takeoff 0.1 s after the escape mean
crosses 100 Hz; z 0.456 in cruise; wing yaw sweeps > 0.8 rad per 0.1 s; heading turns toward the
stronger steering DN; looming off → landed 4.96 s later on the perch, standing height within
0.002, heading within 0.01 rad, wings at 0.73 / 0.90; the foot shuffle resumes. Scripted-brain
checks: a fly at the wall never leaves the bounds and the wall pulse never exceeds its cap; with
flight disabled the escape drive launches nothing.

Browser, one press of Loom:

```
  t       state     x      y      z     yaw   asym  escape
  1.00 s  ground    0      0     -0.01  0     0     54
  1.25 s  takeoff   0      0      0.11  0.01  0.28  169
  1.75 s  flight    0.21   0.04   0.46  0.38  0.35    9
  3.50 s  flight    0.57   0.71   0.43  2.35  0.31    2
  6.00 s  landing  -0.28   0.29   0.41 -0.91  0.36    0
  7.00 s  settle   -0.02   0.02   0.01 -0.15  0.25    0
  7.50 s  ground   -0.01   0     -0.01  0     0.54    0
```

The orbit target follows the thorax while the camera stays put — a spectator, not a chase
camera; moving the camera with the fly flew it straight through the flowers. Recenter frames the
fly wherever it is. `qpos` stays finite throughout; no page or console errors.

## Known limits

- The terrarium decor has no physics for the fly (as for the ball's fly proxy): a flight path
  can pass through the fern or the flowers.
- The wingbeat is a picture of a wingbeat. Nothing about lift, drag or the stroke is modelled.
- A flight is triggered by whatever drives the escape circuit above threshold, so a sustained
  Looming switch (via `window.fly.stimSwitch`) keeps the fly airborne indefinitely.
