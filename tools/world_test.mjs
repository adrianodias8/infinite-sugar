// The terrarium as the source of the senses, against the real brain and body: sugar by contact
// and smell (on the antenna facing it), daylight and temperature, looming by approach geometry
// on the eye it approaches, contact as touch, and the lateralised drive reaching the neurons.
// Usage: npm test (or build first, then node tools/world_test.mjs)
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
assert(start >= 0 && end > start, 'controller section must exist');
vm.runInContext(app.slice(start, end) + `
globalThis.controller = { resetSim, buildDriveMap, buildShuffleMap, buildFlightMap, buildWorldMap, stepSimulation,
  stepWorld, world, WORLD, flight, stimWorld, poke, walkable, nearestWalkable };`, context);
const c = context.controller;
c.resetSim(); c.buildDriveMap(model, brain); c.buildShuffleMap(); c.buildFlightMap(); c.buildWorldMap();
const w = c.world, W = c.WORLD;
c.flight.enabled = false;   // senses only; locomotion is covered by tools/flight_test.mjs
const advance = s => { for (let i = 0; i < Math.round(s * 10000); i++) c.stepSimulation(); };
const lr = k => brain.stimLR[k];
const head = () => { const b = mujoco.mj_name2id(model, 1, 'head'); return [data.xpos[b * 3], data.xpos[b * 3 + 1]]; };

// 1. Sides: the split populations exist and a one-sided level reaches only that side's cells.
for (const k of ['orn_l', 'orn_r', 'mechano_l', 'mechano_r', 'lc4_l', 'lc4_r', 'lplc2_l', 'lplc2_r', 'visual_l', 'visual_r', 'grn_sweet_l', 'grn_sweet_r']) assert(brain.groups[k] && brain.groups[k].length > 0, k);
brain.setStimLR('odour', 1, 0);
brain.step(300);
console.log('odour left only', { orn_l: brain.rate.orn_l.toFixed(0), orn_r: brain.rate.orn_r.toFixed(0), orn: brain.rate.orn.toFixed(0) });
assert(brain.rate.orn_l > 5 * Math.max(1, brain.rate.orn_r), 'left odour drives the left ORNs');
brain.setStimLR('odour', 0, 0);
brain.step(300);

// 2. Sugar by contact: the sack under the head is tasted and smelled; far away it is neither.
advance(0.5);
const [hx, hy] = head();
w.sugar = { x: hx, y: hy, r: 0.06, placed: true };
c.stepWorld(data, 0.001);
assert.equal(lr('sweet')[0], 1, 'mouthparts over the sugar taste it');
assert(w.levels.odour > 0.7 && lr('odour')[0] > 0.5 && lr('odour')[1] > 0.5, 'sugar under the head smells on both antennae');
w.sugar = { x: hx + 2.0, y: hy, r: 0.06, placed: true };
c.stepWorld(data, 0.001);
assert.equal(lr('sweet')[0], 0, 'nothing to taste two body lengths away');
assert.equal(w.levels.odour, 0, 'and nothing to smell beyond the odour range');
// odour to the left lands on the left antenna
w.sugar = { x: data.qpos[0], y: data.qpos[1] + 0.4, r: 0.06, placed: true };
c.stepWorld(data, 0.001);
const od = lr('odour');
console.log('sugar to the left', { odour: w.levels.odour.toFixed(2), left: od[0].toFixed(2), right: od[1].toFixed(2) });
assert(od[0] > 0.3 && od[1] < 0.05, 'odour on the left antenna only');
// feeding: sugar under the head drives the proboscis through the real wiring
w.sugar = { x: hx, y: hy, r: 0.06, placed: true };
const restProb = brain.rate.mn_proboscis;
advance(1.5);
console.log('feeding', { proboscis: brain.rate.mn_proboscis.toFixed(1), rest: brain.rest.mn_proboscis.toFixed(1), counter: brain.sugarFeedSpikes });
assert(brain.rate.mn_proboscis > brain.rest.mn_proboscis, 'the sack drives the feeding motor neurons');
assert(brain.sugarFeedSpikes > 0, 'the counter counts feeding at the sack');
w.sugar = { x: hx + 2.0, y: hy, r: 0.06, placed: true };

// 3. Daylight: noon lights and warms, midnight is dark, cold and damp; an occluder shades.
w.t = 0; c.stepWorld(data, 0.001);
assert(Math.abs(lr('light')[0] - W.lightMax) < 1e-6 && lr('heat')[0] > 0 && lr('cool')[0] === 0, 'noon');
w.t = W.dayPeriod / 2; c.stepWorld(data, 0.001);
assert(lr('light')[0] < 1e-6 && lr('cool')[0] > 0 && lr('damp')[0] > 0 && lr('heat')[0] === 0, 'midnight');
w.t = 0;
w.loomers = [{ x: data.qpos[0] + w.sun[0] * 0.3, y: data.qpos[1] + w.sun[1] * 0.3, z: data.qpos[2] + w.sun[2] * 0.3, vx: 0, vy: 0, vz: 0, r: 0.1, name: 'occluder' }];
c.stepWorld(data, 0.001);
assert(w.shade > 0.99 && lr('light')[0] < 1e-6, 'an object between the fly and the sun casts a shadow');

// 4. Looming by approach geometry, on the eye it approaches; a receding object does not loom.
const yaw = Math.atan2(2 * (data.qpos[3] * data.qpos[6] + data.qpos[4] * data.qpos[5]), 1 - 2 * (data.qpos[5] ** 2 + data.qpos[6] ** 2));
const leftX = data.qpos[0] - Math.sin(yaw) * 0.3, leftY = data.qpos[1] + Math.cos(yaw) * 0.3;   // 0.3 to the fly's left
w.loomers = [{ x: leftX, y: leftY, z: data.qpos[2], vx: Math.sin(yaw) * 1.0, vy: -Math.cos(yaw) * 1.0, vz: 0, r: 0.12, name: 'ball' }];
c.stepWorld(data, 0.001);
const lo = lr('looming');
console.log('ball from the left', { level: w.levels.looming.toFixed(2), left: lo[0].toFixed(2), right: lo[1].toFixed(2) });
assert(lo[0] > 0.3 && lo[1] < 0.05, 'a ball closing from the left looms on the left eye');
w.loomers = [{ x: leftX, y: leftY, z: data.qpos[2], vx: -Math.sin(yaw) * 1.0, vy: Math.cos(yaw) * 1.0, vz: 0, r: 0.12, name: 'ball' }];
c.stepWorld(data, 0.001);
assert.equal(w.levels.looming, 0, 'a ball moving away does not loom');
w.loomers = [];

// 5. A contact queued by the ball physics is a touch on that flank; the switch-off clears everything.
w.touchHits.push(Math.PI / 2);
c.stepWorld(data, 0.001);
const t = lr('touch');
assert(t[0] === 1 && t[1] < 1 && c.poke.last === 'ball', 'a bump on the left is a touch on the left');
advance(1.2);
assert(lr('touch')[0] < 0.2, 'the touch releases');
w.enabled = false; c.stepWorld(data, 0.001);
for (const k of Object.keys(w.levels)) assert.equal(w.levels[k], 0, `${k} cleared`);
assert(Object.values(c.stimWorld).every(v => v[0] === 0 && v[1] === 0), 'world sources cleared');
w.enabled = true;

// 6. Walkable floor: without the terrarium a disc round the perch; the nearest walkable point is inside it.
assert(c.walkable(0, 0) && !c.walkable(2, 2));
const [nx, ny] = c.nearestWalkable(2, 2);
assert(c.walkable(nx, ny), 'nearest walkable point is walkable');
assert(Array.from(data.qpos).every(Number.isFinite) && Array.from(brain.v).every(Number.isFinite), 'finite');
console.log('PASS: side pools, one-sided drive, sugar by contact and smell, lateral odour, feeding at the sack, daylight, shade, lateral looming, receding object, ball touch, world off, walkable floor');
