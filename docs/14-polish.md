# Polish — how the simulation should feel, and what to do about it

A plan, not a record. Each item names the observed problem, the fix, and how it would be
verified, in the order it should be done. The rules stay: the connectome and the 1 kHz timing
are fixed, nothing learns, every supplied movement stays labelled as supplied, and every number
is measured before it is trusted.

## 1. The fly's motion

- **Walking looks like gliding.** The tripod cycle now plants stance feet, but the body is still
  carried at a constant 0.22 cm/s while the stride is a fixed 2.2 Hz, so the feet and the body
  disagree by a few percent and the fly drifts over its own footsteps. Fix: derive the body speed
  from the stride (2 × swing amplitude × step rate, measured per leg once at load) instead of
  the other way round, and let the DNp09 bout length modulate step rate rather than speed.
  Verify: foot slip per stance under 0.005 cm in `tools/flight_test.mjs`.
- **Every stop is a "touchdown".** A walking bout ends with the same pinned handover as a
  landing, which reads as a small crouch after every few steps. Fix: for walking, hand the root
  back the moment the stance legs carry the weight (already measured through the vertical
  constraint force) without the tuck ramp. Verify: root velocity at handover below 0.3 cm/s,
  and no height dip beyond 0.005.
- **Turning in place at obstacles is abrupt.** The supplied turn is a step in yaw rate. Fix:
  ramp it over 0.2 s and add a head turn toward the open side through the neck targets only
  while turning (supplied, labelled). Verify by eye and by yaw-rate continuity.
- **Takeoff has no jump.** Real flies push off with the middle legs. Fix: 60 ms of leg
  extension before the wingbeat starts, taken from the existing tuck offsets in reverse.
  Verify: the root leaves the floor after the legs extend, not before.
- **Flight bobs on a sine.** Replace the fixed bob with a slow random walk in altitude
  (a filtered noise term, seeded, so runs are reproducible) and bank that follows yaw
  acceleration rather than yaw rate. Verify: altitude spectrum has no single peak.

## 2. The body

- **Wings rest on the hind tibia.** Measured: no resting bias inside the actuator band clears
  it. The honest fix is in the pose, not the mapping: a slightly raised standing posture for the
  hind legs (femur targets −0.03 rad, within the shuffle's own range) would drop the tibia
  clear. Verify: zero wing contacts at rest for 10 s, and the shuffle test's settle bound back
  to 0.05.
- **Proboscis jitter is large.** The haustellum reaches 50 rad/s at rest under neural noise.
  This is the servo responding to a 25 ms rate estimate. A longer rate window is a kernel-side
  readout choice and must stay explicit; the alternative is a low-pass on the proboscis servo
  target only (body side, labelled). Verify: peak haustellum velocity under 10 rad/s with the
  same mean extension under sugar.
- **Antennae and head should point at things.** The neck and antennal motor neurons are
  lateralised (a touch on the left turns the head right and sweeps the left antenna, measured).
  Nothing to supply; but the world should give them something to point at: an odour on the
  left already drives the left ORNs. Verify that head yaw follows the odour side in a 20 s run
  (it may not; report either way).

## 3. The world

- **Odour needs a plume.** A radial gradient smells the same from every direction; a real
  plume has a downwind axis. Fix: a fixed "draught" direction across the terrarium, odour
  strongest downwind of the sack, with the level on each antenna from the angle to the plume
  axis. Verify: odour L/R differ by side of the axis in `tools/world_test.mjs`.
- **The sugar should run out.** Sugar as a finite quantity that feeding depletes over minutes
  (a world property, not a brain one) makes the fly's day a story: find, feed, wander, find
  again as it is refilled. Verify: counter rate falls to zero as the sack empties.
- **Light should have contrast, not just level.** Visual cells respond to change. Fix: drive
  the visual level with the *rate of change* of brightness (shadow edges, the ball's shadow
  passing) added to the capped ambient term, both explicit. Verify: a passing shadow gives a
  visual transient larger than the ambient level.
- **Night.** At night the walking DN asks for more steps (measured), which is backwards for a
  diurnal fly and is simply what this wiring does without a clock. Say so in About rather than
  hide it.
- **The ball should be alive.** A nudge from the fly walking into it already works through the
  proxy; a gentle periodic roll (a tilt of the floor plane in cannon-es on a slow cycle) would
  give the fly recurring looming and touch events without a hand on the button.

## 4. Camera and presentation

- **A spectator that knows where to stand.** The camera pans from a fixed spot; when the fly
  walks behind a rock it vanishes. Fix: keep the orbit target on the thorax and, when the line
  of sight is blocked (one raycast per frame against the terrarium), slide the camera along its
  orbit to the nearest clear angle. Verify: fraction of frames with the fly occluded under 5 %
  in a 60 s run.
- **The status pill should be a sentence.** "Walking" is a state; "walking toward the sugar" or
  "tasting the fern" is a story. The pill has the data (world levels, bearing to the sack).
- **Inspect should show the loop.** One row per sense with the world level, the switch level
  and the population rate side by side, so a viewer can see the world reach the neurons.
- **Sound.** A wingbeat tone during flight and a soft tick per step, both synthesised, both
  driven by the same state that draws them; muted by default.

## 5. Performance

- The world costs one raycast grid at load and a handful of vector operations per brain tick;
  nothing per neuron. Keep it that way. The ground map (56 × 56) is the only allocation.
- Wing kinematics write six joints per physics step; fine. The stride writes twelve leg
  actuators per brain tick; fine.
- Mobile: re-run `docs/10` viewports with a flight and a walk; the follow camera's per-frame
  vector work is negligible, but the ripple sprite allocates per tap — pool it.

## 6. Tests to add as this lands

- A ten-minute unattended run recorded as a behavioural budget (fraction of time walking,
  feeding, grooming, flying; by day and night), committed to the doc each time the
  locomotion mapping changes, so drift in "feel" is visible as numbers.
- A determinism check: two runs from the same seed produce the same trajectory for 30 s.
