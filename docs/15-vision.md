# Vision — the fly sees the terrarium through its own photoreceptors

Run 2026-09-16. `python3 tools/build_retina.py` writes `web/brain/retina.json`; `npm test` runs
`tools/retina_test.mjs`.

## What changed

Until now the 11,426 visual cells were one number: a daylight level, shaded when something
stood between the fly and the sun, with a transient when that brightness changed. The optic
lobes got the same drive on every cell, so nothing in the scene had a shape.

FlyWire's `visual` class is the photoreceptors themselves — 8,456 R1-6 (whose axons end in
the lamina), 1,338 R7 and 1,357 R8 (medulla) and 273 ocellar retinula cells — and the lamina
and medulla are retinotopic. So each cell's recorded position says where in the visual field
it looks, and the page can render the terrarium from the head and drive every photoreceptor by
what it would see. That is what the fly sees now: the plants, the rocks, the sack, the ball and
its shadow, the loom sphere as it arrives, the dim night.

## The map (`tools/build_retina.py`)

An inferred retinotopy, not the measured ommatidial lattice. Per eye and per layer (R1-6, R7,
R8) the mean recorded position of each cell is ranked along the dorsoventral axis (FAFB y
grows ventrally: the ocelli sit at y ≈ 94k, the lamina at 307k) for elevation, and along the
anterior-posterior arc of the neuropil (the first principal axis of the x–z spread, oriented
so that FAFB z, which grows toward the rear — antennal lobe 39k, lamina 154k, lobula plate
209k — increases) for azimuth. Ranks, because ommatidia are close to evenly spaced and the
recorded positions are not. The ranks are spread over −15° (past the midline: the binocular
strip) to 155° (rear) of azimuth and +65° to −65° of elevation; the ocelli look up at +75°,
the lateral ones 30° to their side. Two side-less photoreceptors are left out. The lamina
arcs are clear (spread 3.3:1 left, 2.5:1 right); the right medulla layers are nearly round in
x–z (1.2:1), so the azimuth of the 1,330 right R7/R8 cells is the least certain part of the map.

| | cells | placed |
|---|---|---|
| R1-6 | 8,456 | 4,425 left, 4,031 right |
| R7 | 1,338 | 669 / 669 |
| R8 | 1,357 | 696 / 661 |
| ocellar | 273 | 97 left, 100 right, 76 centre |

## The eye (web/app.ts, vision)

Six 90° faces are rendered from the head body's position along the world axes at 24 × 24
pixels each, 4 / 8 / 12 times a second by quality level, with the fly's own meshes hidden (an
eye does not see its own head) and the sky, the lights, the shadows and the fog as the viewer
sees them. Each photoreceptor's direction in the head frame (from its azimuth and elevation,
left eye to the left, right eye mirrored) is rotated by the head's live quaternion, lands on
one face, and reads one pixel. Render targets hold linear light; the value that drives a cell
is its display-encoded (sRGB) brightness, close to the compressive response of a real
photoreceptor, so a mid-grey surface reads 0.4 rather than 0.12 and a daylit scene sits where
the scalar daylight level used to. The drive per cell is the same two terms as before:

    level = 0.35 × brightness + 0.05 × |d brightness / dt| (50 ms filtered), capped at 1

handed to the kernel per neuron (`brain.setCellDrive`, membrane units per ms = level × 0.20).
No adaptation, per cell or otherwise: the drive is the brightness, as everything else.

Without a renderer — the node tests, the first frames before the retina and terrarium load —
the scalar path still runs and drives the pool uniformly; with the world off the eyes are off
with everything else. The Light switch is an override on top: a uniform extra drive.

Inspect shows what the eyes see: one dot per R1-6 cell at its azimuth and elevation, left eye
on the left, front at the centre, dorsal at the top, plus the mean drive per eye and for the
ocelli.

## Measured

`tools/retina_test.mjs`, real brain and body: a frontal cell looks along the body axis (world
+x at rest), a left-eye cell at 90° looks along +y, a right-eye one along −y, an ocellus up;
each world axis lands on its own face at the centre pixel, up within a face is the top row; a
white face ahead lights 2,428 cells (the frontal minority) at exactly 0.35 × 0.20, alike in
both eyes; a wall going dark is a transient that decays; light on the left face only drives
the left eye's cells, and the left photoreceptor pool goes from 4.8 Hz at rest to 28.1 Hz
while the right stays at 4.6; the world hands the cells to the kernel and drops the scalar
level; world off clears the per-cell drive and the left eye returns to 5.0 Hz.

Headless Chromium, the real terrarium at noon from the perch: 10,685 of 11,424 cells see
something (brightness > 0.05), mean brightness 0.44, mean drive 0.16 left / 0.15 right /
0.19 ocelli; the visual pools run at 35.0 / 31.3 Hz (scalar path at the same moment: 33.1 /
28.9), so the change does not move the whole-brain budget, it gives the input a shape. By
azimuth band (front to rear) the left eye reads 0.44, 0.55, 0.50, 0.43, 0.38 and the right
0.36, 0.44, 0.29, 0.40, 0.50: the scene is not uniform and the two eyes do not agree, which
is the point.

**A dark sphere ahead does not reach the looming detectors through the wiring.** A static
black sphere (r 0.06) 0.22 cm in front of the head: LC4 0.87 → 0.92 Hz, LPLC2 0.75 → 0.81,
giant fiber 8.6 → 4.0 (noise), escape DNs 0 either way, and with the eyes off and the sphere
still there LC4 reads 0.82. The optic lobe's motion and looming circuits (T4/T5, the lobula
columnar cells) depend on temporal filtering between neighbouring columns that a
leaky-integrate-and-fire with one fixed delay does not reproduce, so the looming (LC4, LPLC2)
and small-object (LC11) channels stay told geometrically by the world, as documented in
[13-world.md](13-world.md). The eyes give the photoreceptors and everything downstream of them
a real image; they do not, by themselves, make the fly flinch.

## Honesty budget

- The retinotopy is inferred from terminal positions by rank; the field limits (−15..155°,
  ±65°) are textbook values for *Drosophila*, not measured here. A cell may be tens of degrees
  from where the real ommatidium points.
- Photoreceptors are driven by display-encoded brightness with no adaptation: a real
  photoreceptor adapts over seconds and reports contrast around its mean; this one reports
  the mean. Ruled out by design (no habituation), and said here.
- The eye is where the head is, with no interocular distance; at the terrarium's scale
  parallax is negligible.
- The render's 24-pixel faces are coarser in the corners than an ommatidium's ~5° acceptance
  angle, and finer at the centre; several cells share a pixel.
- Night is what the viewer sees: the lights scaled to 30 %, not darkness.
