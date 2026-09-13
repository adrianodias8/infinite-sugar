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
  stepWorld, world, WORLD, flight, stimWorld, poke, walkable, nearestWalkable, stepGroom, groom };`, context);
const c = context.controller;
c.resetSim(); c.buildDriveMap(model, brain); c.buildShuffleMap(); c.buildFlightMap(); c.buildWorldMap();
const w = c.world, W = c.WORLD;
c.flight.enabled = false;   // senses only; locomotion is covered by tools/flight_test.mjs
const advance = s => { for (let i = 0; i < Math.round(s * 10000); i++) c.stepSimulation(); };
const lr = k => brain.stimLR[k];
const head = () => { const b = mujoco.mj_name2id(model, 1, 'head'); return [data.xpos[b * 3], data.xpos[b * 3 + 1]]; };

// 1. Sides: the split populations exist and a one-sided level reaches only that side's cells.
for (const k of ['orn_l', 'orn_r', 'mechano_l', 'mechano_r', 'lc4_l', 'lc4_r', 'lplc2_l', 'lplc2_r', 'lc11_l', 'lc11_r', 'visual_l', 'visual_r', 'grn_sweet_l', 'grn_sweet_r']) assert(brain.groups[k] && brain.groups[k].length > 0, k);
brain.setStimLR('odour', 1, 0);
brain.step(300);
console.log('odour left only', { orn_l: brain.rate.orn_l.toFixed(0), orn_r: brain.rate.orn_r.toFixed(0), orn: brain.rate.orn.toFixed(0) });
assert(brain.rate.orn_l > 5 * Math.max(1, brain.rate.orn_r), 'left odour drives the left ORNs');
brain.setStimLR('odour', 0, 0);
brain.step(300);

// 2. Sugar by contact: a sack whose surface the labellum touches is tasted, the feet taste at
// its base, it is smelled; far away it is none of these.
advance(0.5);
const [hx, hy] = head();
const lab = mujoco.mj_name2id(model, 1, 'labrum_left');
const tip = [data.xpos[lab * 3], data.xpos[lab * 3 + 1]];
w.sugar = { x: tip[0] + 0.15, y: tip[1], r: 0.14, placed: true };          // surface 0.01 from the labellum
c.stepWorld(data, 0.001);
assert(lr('sweet')[0] > 0.6, `labellum against the sack tastes it (${lr('sweet')[0].toFixed(2)})`);
assert(lr('sweetLeg')[0] > 0 && lr('sweetLeg')[0] < 1, `the front feet stand in the sugar at its base, the hind feet do not (${lr('sweetLeg')[0].toFixed(2)})`);
assert(brain.sugar === lr('sweet')[0], 'the feeding counter follows the labellum, not the feet');
assert(w.levels.odour > 0.6 && lr('odour')[0] > 0.5 && lr('odour')[1] > 0.5, 'sugar ahead smells on both antennae');
const claw = mujoco.mj_name2id(model, 1, 'claw_T1_left');
w.sugar = { x: data.xpos[claw * 3] + 0.10, y: data.xpos[claw * 3 + 1], r: 0.08, placed: true };   // base within footReach of a foot
c.stepWorld(data, 0.001);
assert(lr('sweetLeg')[0] > 0 && lr('sweetLeg')[0] <= 1, `a foot at the base tastes with the tarsus (${lr('sweetLeg')[0].toFixed(2)})`);
w.sugar = { x: hx + 2.0, y: hy, r: 0.06, placed: true };
c.stepWorld(data, 0.001);
assert.equal(lr('sweet')[0], 0, 'nothing to taste two body lengths away');
assert.equal(lr('sweetLeg')[0], 0, 'nor with the feet');
assert.equal(w.levels.odour, 0, 'and nothing to smell beyond the odour range');
// odour to the left lands on the left antenna
w.sugar = { x: data.qpos[0], y: data.qpos[1] + 0.4, r: 0.06, placed: true };
c.stepWorld(data, 0.001);
const od = lr('odour');
console.log('sugar to the left', { odour: w.levels.odour.toFixed(2), left: od[0].toFixed(2), right: od[1].toFixed(2) });
assert(od[0] > 0.3 && od[1] < 0.05, 'odour on the left antenna only');
// feeding: the labellum on the sack drives the proboscis through the real wiring
w.sugar = { x: tip[0] + 0.15, y: tip[1], r: 0.14, placed: true };
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
// a small object crossing the view on the left drives the LC11 side, and does not loom
w.loomers = [{ x: leftX, y: leftY, z: data.qpos[2], vx: Math.cos(yaw) * 0.6, vy: Math.sin(yaw) * 0.6, vz: 0, r: 0.05, name: 'ball' }];
c.stepWorld(data, 0.001);
const ob = lr('object');
console.log('object crossing on the left', { level: w.levels.object.toFixed(2), left: ob[0].toFixed(2), right: ob[1].toFixed(2), looming: w.levels.looming });
assert(ob[0] > 0.3 && ob[1] < 0.05 && w.levels.looming === 0, 'a crossing object drives LC11 on that side only');
w.loomers = [];
assert(lr('object')[0] > 0, 'the level persists until the next world step');
c.stepWorld(data, 0.001);
assert.equal(w.levels.object, 0, 'nothing crossing, nothing seen');
assert.equal(lr('object')[0] + lr('object')[1], 0, 'and the channel is cleared');

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

// 7. Grooming: DNg11 above its rest while standing lifts and rubs the front legs; quiet, never.
c.flight.enabled = false;
const femurT1 = mujoco.mj_name2id(model, 19, 'coxa_T1_left'), swingT1 = mujoco.mj_name2id(model, 19, 'coxa_twist_T1_left');   // coxa extension raises the leg
const hold = [data.ctrl[femurT1], data.ctrl[swingT1]];
const groomer = { rate: { dn_groom: 60 }, rest: { dn_groom: 22 } };
for (let i = 0; i < 400; i++) c.stepGroom(groomer, data, 0.001);
assert(c.groom.active && c.groom.count === 1, 'a raised DNg11 starts a grooming bout');
let maxLift = 0, maxSwing = 0;
for (let i = 0; i < 500; i++) { c.stepGroom(groomer, data, 0.001); maxLift = Math.max(maxLift, Math.abs(data.ctrl[femurT1] - hold[0])); maxSwing = Math.max(maxSwing, Math.abs(data.ctrl[swingT1] - hold[1])); }
console.log('grooming', { lift: maxLift.toFixed(3), swing: maxSwing.toFixed(3) });
assert(maxLift > 0.3 && maxSwing > 0.2, 'front leg is raised at the coxa and rubs');
for (let i = 0; i < 1500; i++) c.stepGroom(groomer, data, 0.001);
assert(!c.groom.active || c.groom.count >= 2, 'bouts end');
const quiet = { rate: { dn_groom: 22 }, rest: { dn_groom: 22 } };
c.groom.active = false; c.groom.cooldown = 0; c.groom.ema = 22; const count = c.groom.count;
for (let i = 0; i < 2000; i++) c.stepGroom(quiet, data, 0.001);
assert.equal(c.groom.count, count, 'a resting DNg11 never grooms');
assert(Array.from(data.qpos).every(Number.isFinite) && Array.from(brain.v).every(Number.isFinite), 'finite');
console.log('PASS: side pools, one-sided drive, labellar and tarsal sugar by contact, smell, lateral odour, feeding at the sack, daylight, shade, lateral looming, receding object, crossing object on LC11, ball touch, world off, walkable floor, grooming');
