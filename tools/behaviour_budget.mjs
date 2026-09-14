// Behavioural budget: an unattended run with the world on (the perch disc; no terrarium mesh
// in node, so no plants or rocks), reporting the fraction of time in each state by day and
// night, feeding, grooming, flights and walks. Recorded in docs/14-polish.md whenever the
// locomotion mapping changes, so drift in "feel" shows up as numbers.
// Usage: node tools/behaviour_budget.mjs [seconds]   (default 600; slow — minutes of wall time)
import fs from 'node:fs';
import vm from 'node:vm';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';
import loadMujoco from '../web/vendor/mujoco_wasm.js';
import { Brain } from '../dist/brain.js';

process.chdir(fileURLToPath(new URL('..', import.meta.url)));
const seconds = Number(process.argv[2] || 600);
const mujoco = await loadMujoco();
mujoco.FS.mkdir('/w'); mujoco.FS.mkdir('/w/assets');
const manifest = JSON.parse(fs.readFileSync('web/model/manifest.json'));
for (const name of ['scene.xml', 'fruitfly.xml', ...manifest.assets.map(a => `assets/${a}`)]) {
  mujoco.FS.writeFile(`/w/${name}`, fs.readFileSync(`web/model/${name}`));
}
const model = mujoco.MjModel.loadFromXML('/w/scene.xml');
const data = new mujoco.MjData(model);
function blob(name, Type) {
  const buf = zlib.gunzipSync(fs.readFileSync(`web/brain/${name}.bin.gz`));
  return new Type(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}
const meta = JSON.parse(fs.readFileSync('web/brain/meta.json'));
const brain = new Brain(meta, blob('indptr', Uint32Array), blob('colidx', Uint32Array),
  Float32Array.from(blob('w2', Int16Array), x => x * 0.005 / 2));
brain.calibrate();
const sim = { steps:0, brainStartMs:0 };
const context = vm.createContext({ mujoco, model, data, brain, sim, performance });
const app = fs.readFileSync('dist/app.js', 'utf8');
const start = app.indexOf('// ---------------------------------------------------------------- pose hold');
const end = app.indexOf('// ---------------------------------------------------------------- main', start);
vm.runInContext(app.slice(start, end) + `
globalThis.controller = { resetSim, buildDriveMap, buildShuffleMap, buildFlightMap, buildWorldMap, stepSimulation, flight, world, WORLD, shuffle, groom };`, context);
const c = context.controller;
c.resetSim(); c.buildDriveMap(model, brain); c.buildShuffleMap(); c.buildFlightMap(); c.buildWorldMap();

const budget = { day: {}, night: {} };
const add = (bucket, k, ms) => { bucket[k] = (bucket[k] || 0) + ms; };
let feedSpikesAtDay = 0, feedSpikesAtNight = 0, lastSpikes = brain.feedSpikes;
const t0 = performance.now();
for (let ms = 0; ms < seconds * 1000; ms++) {
  for (let i = 0; i < 10; i++) c.stepSimulation();
  const bucket = c.world.day >= 0.5 ? budget.day : budget.night;
  const st = c.flight.state;
  const k = st === 'walk' ? 'walking' : st === 'takeoff' || st === 'flight' || st === 'landing' || st === 'settle' ? 'flying'
          : st === 'touchdown' ? 'settling' : c.world.levels.sweet > 0 ? 'feeding' : c.groom.active ? 'grooming' : 'standing';
  add(bucket, k, 1); add(bucket, 'total', 1);
  const ds = brain.feedSpikes - lastSpikes; lastSpikes = brain.feedSpikes;
  if (c.world.day >= 0.5) feedSpikesAtDay += ds; else feedSpikesAtNight += ds;
  if (ms % 60000 === 59999) console.error(`${(ms + 1) / 1000} s simulated in ${((performance.now() - t0) / 1000).toFixed(0)} s`);
}
const row = (name, b) => {
  const T = b.total || 1;
  return `| ${name} | ${(T / 1000).toFixed(0)} s | ` + ['walking', 'flying', 'settling', 'feeding', 'grooming', 'standing'].map(k => `${(100 * (b[k] || 0) / T).toFixed(1)} %`).join(' | ') + ' |';
};
console.log(`Behavioural budget, ${seconds} s with the world on (perch disc, sack at the default spot), ${new Date().toISOString().slice(0, 10)}:`);
console.log('| | time | walking | flying | settling | feeding | grooming | standing |');
console.log('|---|---|---|---|---|---|---|---|');
console.log(row('day', budget.day)); console.log(row('night', budget.night));
console.log(`flights ${c.flight.count}, walks ${c.flight.walk.count} (${c.flight.walk.blocked} blocked), shuffles ${c.shuffle.count}, feeding spikes day ${feedSpikesAtDay} / night ${feedSpikesAtNight}, sugar left ${(c.world.sugar.amount * 100).toFixed(0)} %`);
