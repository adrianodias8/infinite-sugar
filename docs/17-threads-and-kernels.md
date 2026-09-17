# Threads and kernels — the brain in WebAssembly, on its own thread

Written 2026-09-17. `tools/build_kernel.sh` compiles the kernel; `npm test` runs
`tools/determinism_test.mjs`, which steps one fly on the JavaScript kernel and one on the
WebAssembly kernel through the same 12 s and requires them to agree to the bit.

## Where the time went

Per simulated second on the build machine, before this: the brain 1.1–1.5 s (JavaScript),
the physics 1.4 s (five 0.2 ms substeps per brain millisecond), the world, gait and actuators
0.06 s — all in series on the main thread, which also renders. A simulated second cost about
3.3 s of wall time; the page ran at a third of real time.

## The WebAssembly kernel (`web/kernel/lif.c`)

The per-millisecond loop — delayed inhibition, noise, pool and per-cell drives, leak /
threshold / reset over every neuron, propagation, spike bookkeeping — is written once more in
C and compiled with clang to a 3 KB module. It does the SAME arithmetic as `brain.ts`: every
update is computed in double and rounded once into the float32 state, exactly as JavaScript
does with a `Float32Array`, the pool drives are passed as doubles, and the noise generator is
the same xorshift32 with the same seed. The state (membrane, refractory counters, baseline,
graph, inhibition queues, spike list, last-spike times, pool memberships) is moved into the
module's linear memory at load and the JavaScript fields become views into it, so nothing
that reads the brain — the tests, the neural view, Inspect — changes. Rates, calibration and
the feeding counters stay in JavaScript.

Measured (node, build machine, per simulated second): JavaScript 1.45–1.55 s, WebAssembly
0.95–1.05 s, a 1.5× speed-up; `-msimd128` and `-O2` made no difference (the loop is a gather
over the graph, memory-bound). Bit-identical: after 3.6 s with stimuli, a per-cell drive and
a calibration, every membrane potential, every last-spike time and every resting rate agree
between the two kernels; the determinism test holds this for a full body-in-the-world run.
The page loads the module when the browser has WebAssembly and falls back to the JavaScript
kernel otherwise (Inspect's kernel line says which).

## The worker (`web/brain-worker.ts`, `web/brain-proxy.ts`)

The brain runs on a Web Worker. The page holds a proxy with the same surface (rates, rest,
stimulus levels, counters, last-spike times for the neural view's sample) and asks the worker
for BATCHES of 16 ms with the stimulus levels it has now; the worker steps them and returns one
record per millisecond — every pool's rate and spike count, the whole-brain rate — and the page
replays the records one per brain tick, so the physics, the world and the gait run on the main
thread exactly as before, reading the record of the millisecond they are in. Message passing
with transferred buffers, no SharedArrayBuffer (so it works on a plain static host with no
cross-origin isolation). The worker runs up to 64 ms ahead.

What that costs in honesty: a change in the senses reaches the neurons 16–64 ms later than it
did in-thread. A fly's own sensory transduction and conduction delays are of that order (the
looming escape here needs a 100 ms mean anyway), and the fixed 1 kHz coupling inside the brain
is untouched; but the page's run is no longer reproducible across machines (batch boundaries
depend on frame timing). `?brain=sync` runs the brain in-thread as before, and that is what
the tests and the scripted checks use. `docs/13-world.md`'s numbers were measured in-thread.

What it buys: the brain (1.0 s / s in WebAssembly) and the physics (1.4 s / s) overlap, so a
simulated second costs about the slower of the two plus the rest, ~1.5–1.6 s, instead of their
sum. Headless Chromium on the build machine cannot show this (its software renderer holds the
page at one frame per second); a desktop browser should see roughly twice the real-time
factor it had, in Inspect's real-time row.

## What is not done

- SIMD across neurons (the leak/threshold loop could be 4-wide): the loop is not the cost.
- A second worker for the physics: MuJoCo's WebAssembly is single-threaded and the body is
  written by the gait every millisecond; it would need the same batch replay in reverse.
