# Polish — how the simulation should feel, and what to do about it

A plan, with a record of what has been done under each item (marked **Done** with the
measurement, or **Measured, not done** with the reason). Each item names the observed problem,
the fix, and how it would be verified, in the order it should be done. The rules stay: the connectome and the 1 kHz timing
are fixed, nothing learns, every supplied movement stays labelled as supplied, and every number
is measured before it is trusted.

## 1. The fly's motion

- **Walking looks like gliding.** The tripod cycle now plants stance feet, but the body is still
  carried at a constant 0.22 cm/s while the stride is a fixed 2.2 Hz, so the feet and the body
  disagree by a few percent and the fly drifts over its own footsteps. Fix: derive the body speed
  from the stride (2 × swing amplitude × step rate, measured per leg once at load) instead of
  the other way round, and let the DNp09 bout length modulate step rate rather than speed.
  Verify: foot slip per stance under 0.005 cm in `tools/flight_test.mjs`.
  **Done** (the other way round from the plan, with the same effect): each leg's stride
  amplitude is derived from its own foot travel per radian, calibrated with the foot planted,
  and the stance sweep is linear at the body's speed. Measured along-heading slip per stance
  is within 0.003 cm for every leg; the test asserts a six-leg mean under 0.005 and no leg
  over 0.03. What the plan did not foresee: a single twist joint sweeps the foot on an arc, so
  a sideways component of 0.01–0.06 cm per stance remains and would need a second joint per
  leg to remove. Step rate stays fixed at 2.2 Hz.
- **Every stop is a "touchdown".** A walking bout ends with the same pinned handover as a
  landing, which reads as a small crouch after every few steps. Fix: for walking, hand the root
  back the moment the stance legs carry the weight (already measured through the vertical
  constraint force) without the tuck ramp. Verify: root velocity at handover below 0.3 cm/s,
  and no height dip beyond 0.005.
  **Done**: a bout ends by blending the stride out over 0.15 s while the body sinks its 0.006
  lift, and the root is released the moment the legs carry the weight. Measured: on the ground
  0.15 s after the bout, dip +0.002 (above the standing height, never below), root velocity
  0.23 cm/s at handover.
- **Turning in place at obstacles is abrupt.** The supplied turn is a step in yaw rate. Fix:
  ramp it over 0.2 s and add a head turn toward the open side through the neck targets only
  while turning (supplied, labelled). Verify by eye and by yaw-rate continuity.
  **Not done**: the head turn would put a supplied offset on the neck actuators, which are the
  one head mapping driven purely by the neck motor neurons; that stays neural. The yaw-rate
  ramp is still open.
- **Takeoff has no jump.** Real flies push off with the middle legs. Fix: 60 ms of leg
  extension before the wingbeat starts, taken from the existing tuck offsets in reverse.
  Verify: the root leaves the floor after the legs extend, not before.
  **Done**: 60 ms push-off (`FLIGHT.pushSec`) with the front and middle legs extending to half
  the tuck range in reverse, the wings folded and the body rising 0.012; the climb and the
  wingbeat start after it. The test checks flap = 0 and legs extending 20 ms into the takeoff.
- **Flight bobs on a sine.** Replace the fixed bob with a slow random walk in altitude
  (a filtered noise term, seeded, so runs are reproducible) and bank that follows yaw
  acceleration rather than yaw rate. Verify: altitude spectrum has no single peak.
  **Done**: a seeded linear-congruential random walk (white kicks of ±12 cm/s² filtered into a
  vertical drift with a 0.4 s time constant, pulled back to cruise over 2 s, clamped ±0.08)
  replaces the sine; bank follows yaw rate + 0.12 × yaw acceleration. Deterministic for a
  given seed, so runs repeat. Measured over 6 s of cruise: altitude drifts 0.443–0.459 with no
  periodicity.

## 2. The body

- **Wings rest on the hind tibia.** Measured: no resting bias inside the actuator band clears
  it. The honest fix is in the pose, not the mapping: a slightly raised standing posture for the
  hind legs (femur targets −0.03 rad, within the shuffle's own range) would drop the tibia
  clear. Verify: zero wing contacts at rest for 10 s, and the shuffle test's settle bound back
  to 0.05.
  **Measured, not done**: femur −0.03 / −0.06 / −0.1, tibia +0.1 / +0.2 and femur −0.06 with
  tibia +0.15 were each held for 4 s standing. The wing–hind-tibia contact persists in every
  case (39–40 of 40 ticks), the hind claws do not rise, and the body sinks 0.002–0.011. The
  contact is the model's geometry; the standing pose stays and the settle bound stays at 0.1.
- **Proboscis jitter is large.** The haustellum reaches 50 rad/s at rest under neural noise.
  This is the servo responding to a 25 ms rate estimate. A longer rate window is a kernel-side
  readout choice and must stay explicit; the alternative is a low-pass on the proboscis servo
  target only (body side, labelled). Verify: peak haustellum velocity under 10 rad/s with the
  same mean extension under sugar.
  **Done**: a 100 ms low-pass on the four proboscis servo targets (`smooth` in the DRIVE
  table; the rate estimate is untouched). Measured at sugar 0.6, where the rate estimate
  flickers most: peak haustellum velocity 73 → 8 rad/s, mean 22 → 2.2 rad/s, mean extension
  unchanged (−0.75 rad both ways). The shuffle test asserts peak < 10 and mean < 4.
- **Antennae and head should point at things.** The neck and antennal motor neurons are
  lateralised (a touch on the left turns the head right and sweeps the left antenna, measured).
  Nothing to supply; but the world should give them something to point at: an odour on the
  left already drives the left ORNs. Verify that head yaw follows the odour side in a 20 s run
  (it may not; report either way).
  **Measured, does not**: with odour 0.8 on one antenna for 2 s each way, the neck motor pools
  sit at 15.1 / 16.6 Hz (left / right) for odour left and 14.9 / 15.9 for odour right, against
  15.2 / 15.6 at rest, and the head yaw actuator moves under 0.001 rad. The neck follows touch
  (measured before), not smell. Nothing is supplied to change that.

## 3. The world

- **The fly should not be inside things.** The floor map was checked under the root only and
  flight had no obstacles, so the body was carried through rocks and plants. Done after this
  plan: the floor is eroded by the body's radius for walking and landing, obstacle heights
  give flight a clearance to keep and a rule to turn away from what it cannot clear, and the
  next step is never taken into such a thing (see [13-world.md](13-world.md)).
- **The fly should see.** The visual cells were one number. Done after this plan, in
  [15-vision.md](15-vision.md): the terrarium is rendered from the head and every photoreceptor
  is driven by the brightness in its own direction, placed from where its axon ends.

- **Odour needs a plume.** A radial gradient smells the same from every direction; a real
  plume has a downwind axis. Fix: a fixed "draught" direction across the terrarium, odour
  strongest downwind of the sack, with the level on each antenna from the angle to the plume
  axis. Verify: odour L/R differ by side of the axis in `tools/world_test.mjs`.
  **Done**: `odourAt(x, y)` is the still-air fall-off shaped into a downwind cone (0.12 cm
  wide at the sack, +0.35 per cm), a quarter of it elsewhere; each antenna samples the field
  0.035 cm to its side and the antenna facing the sack keeps the larger share. Measured 0.5 cm
  from the sack: 0.36 downwind on the axis, 0.09 upwind; 0.12 cm off the axis 0.31 / 0.23 with
  the nearer antenna higher, both sides tested.
- **The sugar should run out.** Sugar as a finite quantity that feeding depletes over minutes
  (a world property, not a brain one) makes the fly's day a story: find, feed, wander, find
  again as it is refilled. Verify: counter rate falls to zero as the sack empties.
  **Done**: `world.sugar.amount` falls at labellar contact / 180 s and refills at 1 / 600 s;
  taste and smell scale with it, the sack slumps to half height, Inspect shows the percentage.
  Test: a 1 s sack empties in 2 s of feeding, taste and smell read 0, the counter freezes for
  the next second, and it refills within 1.5 s of being left alone.
- **Light should have contrast, not just level.** Visual cells respond to change. Fix: drive
  the visual level with the *rate of change* of brightness (shadow edges, the ball's shadow
  passing) added to the capped ambient term, both explicit. Verify: a passing shadow gives a
  visual transient larger than the ambient level.
  **Done**: light = 0.35 × brightness + 0.05 × |d brightness / dt| (50 ms filtered), capped at
  1. Measured: a shadow arriving over 0.1 s peaks at 0.88 against the 0.35 daylight; the day's
  own ramp adds under 0.002. A jump in the clock is a jump in brightness (the tests let it
  pass before reading the level).
- **Night.** At night the walking DN asks for more steps (measured), which is backwards for a
  diurnal fly and is simply what this wiring does without a clock. Say so in About rather than
  hide it.
  **Done**: one sentence in About's technical summary.
- **The ball should be alive.** A nudge from the fly walking into it already works through the
  proxy; a gentle periodic roll (a tilt of the floor plane in cannon-es on a slow cycle) would
  give the fly recurring looming and touch events without a hand on the button.
  **Done**: the cannon-es gravity vector leans 0.05 rad on a 45 s circle. Measured live over
  40 s: the ball rolled 0.14 cm down the hill and stayed inside the glass (1.1 of 1.48 cm from
  the centre). Not covered by `tools/world_test.mjs` (the ball's physics lives in the page).

## 4. Camera and presentation

- **A spectator that knows where to stand.** The camera pans from a fixed spot; when the fly
  walks behind a rock it vanishes. Fix: keep the orbit target on the thorax and, when the line
  of sight is blocked (one raycast per frame against the terrarium), slide the camera along its
  orbit to the nearest clear angle. Verify: fraction of frames with the fly occluded under 5 %
  in a 60 s run.
  **Done**: one raycast from the thorax to the camera every sixth frame against the
  terrarium's opaque meshes; when blocked, the first of ±0.2 … ±1.6 rad round the orbit that
  is clear becomes the goal and the camera slides there at 1.5 rad/s. The visitor's drag is
  the starting orbit and still wins. Measured in headless Chromium (the page runs far below real time under
  software GL, so a live 60 s run is not possible there): the fly placed at five spots in the
  terrarium from the home framing, 1 of 20 line-of-sight checks blocked before the slide and 0
  of 21 after it; in a 60 s scripted walk-and-flight run, 5 of 21 checks were blocked because
  the scripted frames gave the slide no time. The 5 % budget in a live run is checked by eye.
- **The status pill should be a sentence.** "Walking" is a state; "walking toward the sugar" or
  "tasting the fern" is a story. The pill has the data (world levels, bearing to the sack).
  **Done**: `statusSentence()` — walking toward the sugar (bearing within 0.6 rad, under 1 cm,
  odour present), turning at an obstacle, walking backward, taking off, flying from a threat
  (escape drive above the calm threshold), feeding at the sack / at an empty sack, standing in
  the sugar, tasting a plant, grooming the antennae, bumped by the ball, smelling the sugar.
- **Inspect should show the loop.** One row per sense with the world level, the switch level
  and the population rate side by side, so a viewer can see the world reach the neurons.
  **Done**: the Senses table has world, switch and rate columns per population (a tap and the
  loom event show as such in the switch column).
- **Sound.** A wingbeat tone during flight and a soft tick per step, both synthesised, both
  driven by the same state that draws them; muted by default.
  **Done**: a sound button (off by default; the AudioContext is created on the first press).
  Wingbeat: a triangle at 190 Hz with a sine at 475 Hz, loudness 0.06 × `flight.flap` while
  airborne, pitch nudged by the steering asymmetry. Steps: a 30 ms sine tick at 1.4 / 1.7 kHz
  each time a tripod lands (`flight.walk.phase` crossing a half cycle). Nothing else.

## 5. Performance

- The world costs one raycast grid at load and a handful of vector operations per brain tick;
  nothing per neuron. Keep it that way. The ground map (56 × 56) is the only allocation.
  **Measured** (node, build machine, per simulated second): the brain 1.1 s, physics 2.8 s at
  ten substeps per millisecond, the world, actuators and gait 0.06 s. **Done**: five
  substeps (0.2 ms), stable and identical in every suite, physics 1.4 s; a simulated second
  from 4.7 to about 3.3 s of wall time. A speed control asks for 1×, 2× or 4×. **Then done**
  ([17-threads-and-kernels.md](17-threads-and-kernels.md)): the kernel in WebAssembly (1.5×,
  bit-identical) and the brain on a worker thread, overlapping the physics.
- Wing kinematics write six joints per physics step; fine. The stride writes twelve leg
  actuators per brain tick; fine.
- Mobile: re-run `docs/10` viewports with a flight and a walk; the follow camera's per-frame
  vector work is negligible, but the ripple sprite allocates per tap — pool it.
  **Done**: ripple sprites are pooled (a finished ripple's sprite is hidden and reused; nothing
  is disposed per tap). The occlusion raycast is one ray every sixth frame, and up to sixteen
  more only on the frames where the view is blocked. The plume adds three field evaluations
  per brain tick. Mobile re-run at 390 × 844 in headless Chromium: the page lays out, the
  control row wraps with the sound button, no console errors.

## 6. Tests to add as this lands

- A ten-minute unattended run recorded as a behavioural budget (fraction of time walking,
  feeding, grooming, flying; by day and night), committed to the doc each time the
  locomotion mapping changes, so drift in "feel" is visible as numbers.
  **Done**: `node tools/behaviour_budget.mjs [seconds]` (not part of `npm test`; a ten-minute
  run takes minutes). Latest run, on the perch disc in node (no terrarium mesh, so no plants
  or rocks to block bouts):

  | 2026-09-14, 600 s | time | walking | flying | settling | feeding | grooming | standing |
  |---|---|---|---|---|---|---|---|
  | day | 300 s | 74.3 % | 0.0 % | 4.8 % | 0.1 % | 0.0 % | 20.8 % |
  | night | 300 s | 82.6 % | 0.0 % | 3.9 % | 0.0 % | 0.0 % | 13.5 % |

  175 walking bouts, of which 62.5 s in total were spent turning at the disc's edge; no
  flights (nothing loomed), no shuffles (the shuffle only runs while standing), and the sack
  was found by contact for 0.1 % of the day, so it stayed full. The fly walks more at night
  than by day, as measured before. On open floor without obstacles this wiring walks about
  four fifths of the time; the terrarium's rocks and plants, absent here, are what make it
  stop, taste and turn.
- A determinism check: two runs from the same seed produce the same trajectory for 30 s.
  **Done**: `tools/determinism_test.mjs` (in `npm test`) runs two independent brains and
  bodies for 12 s with the world on and a loom at 3 s, and asserts bit-identical `qpos`,
  spike counts, locomotion states and world levels at every 100 ms; a different altitude seed
  changes the cruise height and nothing before it. The only `Math.random` left is the ball's
  starting spot on the hill, in the page and outside the controller.
