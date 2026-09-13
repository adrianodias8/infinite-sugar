// Flight state machine against the real brain, body and clock: takeoff from the escape circuit,
// steering from the DN asymmetry, bounds, landing back on the perch, and the disable switch.
// Usage: npm test (or build first, then node tools/flight_test.mjs)
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
globalThis.controller = { resetSim, buildDriveMap, buildShuffleMap, buildFlightMap, stepSimulation, stepFlight,
  flight, FLIGHT, shuffle, stimPulse, get wings() { return wingJoints; } };`, context);
const c = context.controller;
c.resetSim(); c.buildDriveMap(model, brain); c.buildShuffleMap(); c.buildFlightMap();
const f = c.flight, F = c.FLIGHT;

function advance(seconds) { for (let i = 0; i < Math.round(seconds * 10000); i++) c.stepSimulation(); }
function finite() { assert(Array.from(data.qpos).every(Number.isFinite), 'finite qpos'); }
const yawJoint = mujoco.mj_name2id(model, 3, 'wing_yaw_left');
const yawAdr = model.jnt_qposadr[yawJoint];

// 1. Rest: nothing launches the fly.
advance(1.5);
const standZ = data.qpos[2];
assert.equal(f.state, 'ground');
assert(f.escape < 5, 'escape DNs silent at rest');

// 2. Looming drives the escape circuit; the escape circuit launches the fly.
brain.setStim('looming', 1);
let takeoffAt = -1;
for (let t = 0; t < 200 && takeoffAt < 0; t++) { advance(0.01); if (f.state !== 'ground') takeoffAt = t * 0.01; }
assert(takeoffAt >= 0, 'looming causes takeoff within 2 s');
assert.equal(f.count, 1);
assert(f.escape > F.takeoffRate, 'trigger is the measured escape rate');
advance(1.0);
assert.equal(f.state, 'flight');
finite();
assert(data.qpos[2] > 0.3, `airborne (z=${data.qpos[2].toFixed(3)})`);
let yawMin = Infinity, yawMax = -Infinity;
for (let i = 0; i < 1000; i++) { c.stepSimulation(); yawMin = Math.min(yawMin, data.qpos[yawAdr]); yawMax = Math.max(yawMax, data.qpos[yawAdr]); }
assert(yawMax - yawMin > 0.8, `wings beat (sweep ${(yawMax - yawMin).toFixed(2)} rad over 0.1 s)`);
// heading follows the steering asymmetry sign
let asymSum = 0, n = 0; const yaw0 = f.yaw;
for (let i = 0; i < 100; i++) { advance(0.01); asymSum += f.asym; n++; }
const dyaw = Math.atan2(Math.sin(f.yaw - yaw0), Math.cos(f.yaw - yaw0));
console.log('in flight', { takeoffAt, z: data.qpos[2].toFixed(3), asym: (asymSum / n).toFixed(2), dyaw: dyaw.toFixed(2), escape: f.escape.toFixed(0) });
if (Math.abs(asymSum / n) > 0.1) assert(Math.sign(dyaw) === Math.sign(asymSum / n), 'turns toward the stronger steering DN');

// 3. Threat gone: the escape DNs fall silent and the fly returns to its perch.
brain.setStim('looming', 0);
let landedAt = -1;
for (let t = 0; t < 2500 && landedAt < 0; t++) { advance(0.01); if (f.state === 'ground') landedAt = t * 0.01; }
assert(landedAt >= 0, 'lands within 25 s of the threat passing');
finite();
const home = f.home;
assert(Math.hypot(data.qpos[0] - home.x, data.qpos[1] - home.y) < 0.03, 'lands on the perch');
advance(1.5);
finite();
const heading = Math.atan2(2 * (data.qpos[3] * data.qpos[6] + data.qpos[4] * data.qpos[5]), 1 - 2 * (data.qpos[5] ** 2 + data.qpos[6] ** 2));
// The root and legs must be still; the head and proboscis keep their own neural jitter.
const rootVel = Math.max(...Array.from(data.qvel.subarray(0, 6), Math.abs));
const wingRest = ['wing_yaw_left', 'wing_yaw_right'].map(n => data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, 3, n)]]);
console.log('landed', { landedAt, z: data.qpos[2].toFixed(3), standZ: standZ.toFixed(3), heading: heading.toFixed(2), rootVel: rootVel.toFixed(3), wingYaw: wingRest.map(v => v.toFixed(2)) });
assert(Math.abs(data.qpos[2] - standZ) < 0.02, 'standing height restored');
assert(Math.abs(Math.atan2(Math.sin(heading - home.yaw), Math.cos(heading - home.yaw))) < 0.15, 'heading restored');
assert(rootVel < 0.5, 'root settled, not tumbling');
assert(wingRest.every(v => v > 0.6), 'wings fold back under neural control after landing');
const shuffles = c.shuffle.count;
advance(3);
assert(c.shuffle.count > shuffles, 'foot shuffle resumes after landing');

// 4. Pure controller checks with a scripted brain: bounds and the disable switch.
const fake = { rate: { dn_escwing_l:300, dn_escwing_r:300, dn_steer_l:110, dn_steer_r:20 } };
for (let i = 0; i < 300; i++) c.stepFlight(fake, data, 0.001);
assert.notEqual(f.state, 'ground', 'scripted escape drive launches');
for (let i = 0; i < 1000; i++) c.stepFlight(fake, data, 0.001);
assert(f.asym > 0.9 && f.yawRate > 0, 'full left asymmetry turns left');
const B = f.bounds;
f.x = B.cx + B.r - 0.02; f.y = B.cy; f.yaw = 0;               // at the wall, heading out
let maxR = 0, loomSeen = 0;
for (let i = 0; i < 2000; i++) { c.stepFlight(fake, data, 0.001); maxR = Math.max(maxR, Math.hypot(f.x - B.cx, f.y - B.cy)); loomSeen = Math.max(loomSeen, brain.stim.looming); }
assert(maxR <= B.r + 1e-9, 'never leaves the bounds');
assert(loomSeen > 0.5 * F.wallLoom && loomSeen <= F.wallLoom + 1e-9, 'the wall looms at the brain, gently');
f.state = 'ground'; f.wallLoom = 0; c.stimPulse.looming = 0; brain.setStim('looming', 0);
f.enabled = false; f.escape = 0;
for (let i = 0; i < 1000; i++) c.stepFlight(fake, data, 0.001);
assert.equal(f.state, 'ground', 'disabled flight never launches');
f.enabled = true;
console.log('PASS: escape-circuit takeoff, wingbeat, steering sign, perch landing, shuffle resumes, bounds, wall looming, disable switch');
