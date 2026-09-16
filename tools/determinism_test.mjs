// Determinism: two independent brains and bodies, stepped through the same 12 s (world on, a
// loom at 3 s), produce bit-identical trajectories, spike counts and world levels. Nothing in
// the brain, the world or the locomotion mapping draws on Math.random; the one seeded term
// (the altitude wander) repeats for the same seed and differs for another.
// Usage: npm test (or build first, then node tools/determinism_test.mjs)
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';
import loadMujoco from '../web/vendor/mujoco_wasm.js';
import { Brain } from '../dist/brain.js';

process.chdir(fileURLToPath(new URL('..', import.meta.url)));
const mujoco = await loadMujoco();
mujoco.FS.mkdir('/w'); mujoco.FS.mkdir('/w/assets');
const manifest = JSON.parse(fs.readFileSync('web/model/manifest.json'));
for (const name of ['scene.xml', 'fruitfly.xml', ...manifest.assets.map(a => `assets/${a}`)]) {
  mujoco.FS.writeFile(`/w/${name}`, fs.readFileSync(`web/model/${name}`));
}
const model = mujoco.MjModel.loadFromXML('/w/scene.xml');
function blob(name, Type) {
  const buf = zlib.gunzipSync(fs.readFileSync(`web/brain/${name}.bin.gz`));
  return new Type(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}
const meta = JSON.parse(fs.readFileSync('web/brain/meta.json'));
const app = fs.readFileSync('dist/app.js', 'utf8');
const start = app.indexOf('// ---------------------------------------------------------------- pose hold');
const end = app.indexOf('// ---------------------------------------------------------------- main', start);
assert(start >= 0 && end > start, 'controller section must exist');

function run(seed, seconds) {
  const data = new mujoco.MjData(model);
  const brain = new Brain(meta, blob('indptr', Uint32Array), blob('colidx', Uint32Array),
    Float32Array.from(blob('w2', Int16Array), x => x * 0.005 / 2));
  brain.calibrate();
  const sim = { steps:0, brainStartMs:0 };
  const context = vm.createContext({ mujoco, model, data, brain, sim, performance });
  vm.runInContext(app.slice(start, end) + `
globalThis.controller = { resetSim, buildDriveMap, buildShuffleMap, buildFlightMap, buildWorldMap, stepSimulation, stepMs, flight, world, shuffle, groom };`, context);
  const c = context.controller;
  c.resetSim(); c.buildDriveMap(model, brain); c.buildShuffleMap(); c.buildFlightMap(); c.buildWorldMap();
  c.flight.seed = seed;
  const samples = [];
  for (let ms = 0; ms < seconds * 1000; ms++) {
    if (ms === 3000) brain.setStim('looming', 1);
    if (ms === 3300) brain.setStim('looming', 0);
    c.stepMs();
    if (ms % 100 === 99) samples.push({ ms, qpos: Array.from(data.qpos), spikes: brain.feedSpikes, state: c.flight.state, z: c.flight.z, levels: { ...c.world.levels } });
  }
  return { samples, flights: c.flight.count, walks: c.flight.walk.count, shuffles: c.shuffle.count, t: data.time, brainMs: brain.ms };
}
const seconds = 12;
const a = run(7, seconds), b = run(7, seconds);
console.log('run A', { flights: a.flights, walks: a.walks, shuffles: a.shuffles, t: a.t.toFixed(3), brainMs: a.brainMs, spikes: a.samples.at(-1).spikes });
assert.deepEqual({ flights: a.flights, walks: a.walks, shuffles: a.shuffles, t: a.t, brainMs: a.brainMs }, { flights: b.flights, walks: b.walks, shuffles: b.shuffles, t: b.t, brainMs: b.brainMs }, 'the same counts');
for (let i = 0; i < a.samples.length; i++) {
  const sa = a.samples[i], sb = b.samples[i];
  assert.equal(sa.spikes, sb.spikes, `same spike count at ${sa.ms} ms`);
  assert.equal(sa.state, sb.state, `same locomotion state at ${sa.ms} ms`);
  assert.deepEqual(sa.levels, sb.levels, `same world levels at ${sa.ms} ms`);
  for (let k = 0; k < sa.qpos.length; k++) assert.equal(sa.qpos[k], sb.qpos[k], `identical qpos[${k}] at ${sa.ms} ms`);
}
assert(a.flights >= 1, 'the loom launched a flight in the run being compared');
// the seed is the only free term: another seed changes the altitude wander and nothing else upstream of it
const c2 = run(11, seconds);
const zA = a.samples.filter(s => s.state === 'flight').map(s => s.z), zC = c2.samples.filter(s => s.state === 'flight').map(s => s.z);
console.log('altitude by seed', { seed7: zA.slice(0, 5).map(z => z.toFixed(4)), seed11: zC.slice(0, 5).map(z => z.toFixed(4)) });
assert(zA.length && zC.length && zA.some((z, i) => zC[i] !== undefined && Math.abs(z - zC[i]) > 1e-6), 'a different seed wanders differently');
console.log('PASS: bit-identical trajectory, spikes, states and world levels across two runs; the altitude seed is the only free term');
