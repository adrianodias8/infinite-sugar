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
globalThis.controller = { resetSim, buildDriveMap, buildShuffleMap, buildFlightMap, buildWorldMap, stepSimulation, stepMs, PHYS_SUBSTEPS, stepFlight,
  flight, FLIGHT, WALK, WORLD, shuffle, stimPulse, stimSwitch, applyStim, world, walkable, erodeOk, get wings() { return wingJoints; }, get hold() { return holdCtrl; } };`, context);
const c = context.controller;
c.resetSim(); c.buildDriveMap(model, brain); c.buildShuffleMap(); c.buildFlightMap(); c.buildWorldMap();
c.world.enabled = false;   // the senses here are scripted; tools/world_test.mjs covers the world
const f = c.flight, F = c.FLIGHT;

function advance(seconds) { for (let i = 0; i < Math.round(seconds * 1000); i++) c.stepMs(); }
function finite() { assert(Array.from(data.qpos).every(Number.isFinite), 'finite qpos'); }
const yawJoint = mujoco.mj_name2id(model, 3, 'wing_yaw_left');
const yawAdr = model.jnt_qposadr[yawJoint];

// 1. Rest: nothing launches the fly. (Walking bouts from DNp09 are possible at rest; hold them
// off for the flight checks and exercise them separately below.)
f.enabled = false;
advance(1.5);
const standZ = data.qpos[2];
f.enabled = true;
assert.equal(f.state, 'ground');
assert(f.escape < 5, 'escape DNs silent at rest');

// 2. Looming drives the escape circuit; the escape circuit launches the fly.
c.stimSwitch.looming = 1; c.applyStim('looming');   // through the switch, as the page does (a direct setStim is overwritten by the wall pulse while walking)
let takeoffAt = -1;
for (let t = 0; t < 200 && takeoffAt < 0; t++) { advance(0.01); if (f.state === 'takeoff' || f.state === 'flight') takeoffAt = t * 0.01; }   // a walking bout may start first; the takeoff is what counts
assert(takeoffAt >= 0, 'looming causes takeoff within 2 s');
assert.equal(f.count, 1);
assert(f.escape > F.takeoffRate, 'trigger is the measured escape rate');
// the legs push off before the wings start: no flap during FLIGHT.pushSec, front legs extending
if (f.t < F.pushSec - 0.02) {
  advance(0.02);
  assert.equal(f.state, 'takeoff'); assert.equal(f.flap, 0, 'wings still folded during the push-off'); assert(f.tuck < 0, 'front and middle legs extend in the push-off');
}
advance(1.0);
assert.equal(f.state, 'flight');
finite();
assert(data.qpos[2] > 0.3, `airborne (z=${data.qpos[2].toFixed(3)})`);
let yawMin = Infinity, yawMax = -Infinity;
for (let i = 0; i < 100; i++) { c.stepMs(); yawMin = Math.min(yawMin, data.qpos[yawAdr]); yawMax = Math.max(yawMax, data.qpos[yawAdr]); }
assert(yawMax - yawMin > 0.8, `wings beat (sweep ${(yawMax - yawMin).toFixed(2)} rad over 0.1 s)`);
// heading follows the steering asymmetry sign (with the real brain the looming threat's own
// turn-away term also steers, so the sign check is made with a scripted brain below; here the
// real values are recorded)
let asymSum = 0, n = 0; const yaw0 = f.yaw;
for (let i = 0; i < 100; i++) { advance(0.01); asymSum += f.asym; n++; }
const dyaw = Math.atan2(Math.sin(f.yaw - yaw0), Math.cos(f.yaw - yaw0));
console.log('in flight', { takeoffAt, z: data.qpos[2].toFixed(3), asym: (asymSum / n).toFixed(2), dyaw: dyaw.toFixed(2), escape: f.escape.toFixed(0) });
{
  f.x = f.bounds.cx; f.y = f.bounds.cy; f.yawRate = 0;   // mid-air at the centre: no wall or obstacle term in the way
  const yawA = f.yaw, escapeA = f.escape;
  const steerer = (l, r) => ({ rate: { dn_escwing_l: 150, dn_escwing_r: 150, dn_steer_l: l, dn_steer_r: r }, rest: { dn_steer_l: 70, dn_steer_r: 53 }, spikesOf: () => 0 });
  for (let i = 0; i < 300; i++) c.stepFlight(steerer(105, 53), data, 0.001);   // left DN 1.5x its rest, escape sides balanced
  const turned = Math.atan2(Math.sin(f.yaw - yawA), Math.cos(f.yaw - yawA));
  assert(turned > 0.05, `turns toward the stronger steering DN (${turned.toFixed(2)} rad in 0.3 s)`);
  for (let i = 0; i < 600; i++) c.stepFlight(steerer(70, 80), data, 0.001);   // the asymmetry and the yaw rate are both filtered: ~0.3 s to reverse
  assert(f.yawRate < 0 && Math.atan2(Math.sin(f.yaw - yawA), Math.cos(f.yaw - yawA)) < turned, 'and back the other way when the right DN is stronger');
  f.escape = escapeA;
}

// 3. Threat gone: the escape DNs fall silent and the fly lands where it is, on walkable floor.
c.stimSwitch.looming = 0; c.applyStim('looming');
let landedAt = -1;
for (let t = 0; t < 2500 && landedAt < 0; t++) { advance(0.01); if (f.state === 'ground') landedAt = t * 0.01; }
assert(landedAt >= 0, 'lands within 25 s of the threat passing');
finite();
const home = f.home;
console.log('landing spot', { at: [data.qpos[0].toFixed(3), data.qpos[1].toFixed(3)], home: [home.x.toFixed(3), home.y.toFixed(3)], r: Math.hypot(data.qpos[0], data.qpos[1]).toFixed(3), state: f.state });
assert(Math.hypot(data.qpos[0] - home.x, data.qpos[1] - home.y) < 0.03, 'lands on its chosen spot');
assert(c.walkable(data.qpos[0], data.qpos[1]), 'the spot is walkable floor');
f.enabled = false;   // no walking bouts while the stance is checked
advance(1.5);
finite();
const heading = Math.atan2(2 * (data.qpos[3] * data.qpos[6] + data.qpos[4] * data.qpos[5]), 1 - 2 * (data.qpos[5] ** 2 + data.qpos[6] ** 2));
// The root and legs must be still; the head and proboscis keep their own neural jitter.
const rootVel = Math.max(...Array.from(data.qvel.subarray(0, 6), Math.abs));
const wingRest = ['wing_yaw_left', 'wing_yaw_right'].map(n => data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, 3, n)]]);
console.log('landed', { landedAt, at: [data.qpos[0].toFixed(2), data.qpos[1].toFixed(2)], z: data.qpos[2].toFixed(3), standZ: standZ.toFixed(3), heading: heading.toFixed(2), rootVel: rootVel.toFixed(3), wingYaw: wingRest.map(v => v.toFixed(2)) });
assert(Math.abs(data.qpos[2] - standZ) < 0.02, 'standing height restored');
assert(Math.abs(Math.atan2(Math.sin(heading - home.yaw), Math.cos(heading - home.yaw))) < 0.15, 'heading kept from the approach');
assert(rootVel < 0.5, 'root settled, not tumbling');
assert(wingRest.every(v => v > 0.6), 'wings fold back under neural control after landing');
const shuffles = c.shuffle.count;
advance(3);
assert(c.shuffle.count > shuffles, 'foot shuffle resumes after landing');
f.enabled = true;

// 3b. Walking: DNp09 spikes request forward bouts, MDN spikes backward ones; steering turns;
// the fly stops at the edge of the walkable floor and hands the root back standing.
const stepper = (fwd, back, l, r) => ({ rate: { dn_escwing_l:0, dn_escwing_r:0, dn_steer_l:l, dn_steer_r:r }, rest: { dn_steer_l:70, dn_steer_r:53 }, spikesOf: role => role === 'dn_walk' ? fwd : role === 'dn_back' ? back : 0 });
c.resetSim(); advance(0.5);                                   // at the origin, heading +x, on the fallback disc
const x0 = data.qpos[0], y0 = data.qpos[1], yawA = 0;
c.stepFlight(stepper(1, 0, 70, 53), data, 0.001);          // one DNp09 spike
assert.equal(f.state, 'walk', 'a DNp09 spike starts a walking bout');
assert(f.walk.bout > 0.5 && f.walk.bout <= c.WALK.boutMax);
f.enabled = false;                                            // no further requests from the real brain while this bout is measured
f.walk.bout = 1.2;                                            // long enough for a few full strides before the disc edge
// the stride: the fore-aft joints sweep at the derived per-leg amplitude while walking, and the
// planted feet stay put along the heading as the body is carried forward. (A turn moves left
// and right feet in opposite directions along the heading, so the six-leg mean cancels the
// real brain's steering wander; the sideways arc of a single twist joint is not measured.)
const swingAi = mujoco.mj_name2id(model, 19, 'coxa_twist_T2_left');
const legs = ['T1_left', 'T2_left', 'T3_left', 'T1_right', 'T2_right', 'T3_right'];
const TRIPOD_A = new Set(['T1_left', 'T2_right', 'T3_left']);
const claws = legs.map(l => mujoco.mj_name2id(model, 1, `claw_${l}`));
const slips = legs.map(() => []), stanceStart = legs.map(() => null);
let swMin = Infinity, swMax = -Infinity, yawRateSum = 0;
for (let i = 0; i < 1000; i++) {
  c.stepMs(); swMin = Math.min(swMin, data.ctrl[swingAi]); swMax = Math.max(swMax, data.ctrl[swingAi]); yawRateSum += f.yawRate / 1000;
  const hx = Math.cos(f.yaw), hy = Math.sin(f.yaw);
  legs.forEach((l, k) => {
    const ph = (f.walk.phase + (TRIPOD_A.has(l) ? 0 : Math.PI)) % (2 * Math.PI), b = claws[k];
    const mid = ph > Math.PI * 1.1 && ph < Math.PI * 1.9;   // the middle 80% of the stance half
    const xy = [data.xpos[b * 3], data.xpos[b * 3 + 1]];
    if (mid && !stanceStart[k]) stanceStart[k] = xy;
    else if (!mid && stanceStart[k]) { slips[k].push((xy[0] - stanceStart[k][0]) * hx + (xy[1] - stanceStart[k][1]) * hy); stanceStart[k] = null; }
  });
}
const walked = (data.qpos[0] - x0) * Math.cos(yawA) + (data.qpos[1] - y0) * Math.sin(yawA);
const slipPerLeg = slips.map(s => s.slice(1).reduce((a, b) => a + b, 0) / Math.max(1, s.length - 1));
const slipMean = slipPerLeg.reduce((a, b) => a + b, 0) / 6;
console.log('walk', { state: f.state, walked: walked.toFixed(3), bouts: f.walk.count, z: data.qpos[2].toFixed(3), stride: (swMax - swMin).toFixed(3), slip: slipPerLeg.map(v => v.toFixed(4)), meanYawRate: yawRateSum.toFixed(2), asym: f.asym.toFixed(2), steer: `${brain.rate.dn_steer_l.toFixed(0)}/${brain.rate.dn_steer_r.toFixed(0)} rest ${brain.rest.dn_steer_l.toFixed(0)}/${brain.rest.dn_steer_r.toFixed(0)}` });
assert(walked > 0.1, 'walks forward along its heading');
const ampT2 = c.WALK.speed / (4 * c.WALK.cmPerRad.T2 * c.WALK.stepHz);
assert(Math.abs(swMax - swMin - 2 * ampT2) < 0.02, `the middle leg strides at its derived amplitude (${(swMax - swMin).toFixed(2)} vs ${(2 * ampT2).toFixed(2)} rad)`);
assert(slips.every(s => s.length >= 2), 'several stances measured per leg');
assert(Math.abs(slipMean) < 0.005, `planted feet keep pace with the body (mean along-heading slip ${slipMean.toFixed(4)} cm per stance)`);
assert(slipPerLeg.every(v => Math.abs(v) < 0.03), 'no leg drags its footstep');
assert(Math.abs(yawRateSum) < 0.8, `steering at the calibrated rest wanders rather than circles (mean yaw rate ${yawRateSum.toFixed(2)} rad/s over the bout)`);
// the bout ends with a short settle, not the landing crouch: the root is handed back standing,
// within WALK.stopSec + settling, without the body dipping below its standing height
let stopT = 0, stopMinZ = Infinity;
for (let t = 0; t < 4 && f.state === 'walk'; t += 0.01) advance(0.01);   // the rest of the bout
for (; stopT < 4 && f.state !== 'ground'; stopT += 0.01) { advance(0.01); stopMinZ = Math.min(stopMinZ, data.qpos[2]); }
console.log('walk stop', { after: stopT.toFixed(2), dip: (stopMinZ - standZ).toFixed(4), z: data.qpos[2].toFixed(3) });
assert.equal(f.state, 'ground', 'the bout ends on the ground');
assert(stopT < c.WALK.stopSec + 0.15, 'the walk stops with a short settle');
assert(stopMinZ > standZ - 0.01, 'no crouch when a walk ends');
assert(Math.abs(data.qpos[2] - standZ) < 0.02, 'standing height after walking');
assert(Math.abs(data.ctrl[swingAi] - c.hold[swingAi]) < 1e-9, 'the stride joint returns to its standing target');
f.enabled = true;
// backward and turning
c.stepFlight(stepper(0, 1, 105, 53), data, 0.001);
assert.equal(f.state, 'walk'); assert.equal(f.walk.dir, -1, 'an MDN spike walks backward');
for (let i = 0; i < 400; i++) c.stepFlight(stepper(0, 0, 105, 53), data, 0.001);
assert(f.yawRate > 0.3, 'a stronger left steering DN turns left');
// things in the air: a ridge ahead is flown over with the body's clearance; one too tall for the
// glass is turned away from; and the root is never carried into it
{
  const n = 40, cell = 0.05, x0 = -1, y0 = -1;
  const ok = new Uint8Array(n * n).fill(1), plant = new Uint8Array(n * n), top = new Float32Array(n * n).fill(c.WORLD.floorZ);
  for (let j = 0; j < n; j++) for (let i = 26; i < 30; i++) { ok[j * n + i] = 0; top[j * n + i] = 0.6; }   // a wall across x = 0.3..0.5, 0.6 tall
  c.world.ground = { x0, y0, cell, n, ok, plant, top, clear: c.erodeOk(ok, n, c.WORLD.bodyRadius / cell) };
  const quiet = () => ({ rate: { dn_escwing_l:0, dn_escwing_r:0, dn_steer_l:70, dn_steer_r:53 }, rest: { dn_steer_l:70, dn_steer_r:53 }, spikesOf: () => 0 });
  f.state = 'flight'; f.x = -0.3; f.y = 0; f.z = F.cruiseZ; f.yaw = 0; f.yawRate = 0; f.air = 0; f.quiet = 0; f.escape = 0; f.bob = 0; f.bobV = 0;
  f.bounds = { cx: 0, cy: 0, r: 1.4, zmin: 0.2, zmax: 1.2 };
  let minGap = Infinity, zMax = -Infinity;
  for (let i = 0; i < 2000 && f.state === 'flight'; i++) { c.stepFlight(quiet(), data, 0.001); if (f.x > 0.25 && f.x < 0.55) { minGap = Math.min(minGap, f.z - 0.6); zMax = Math.max(zMax, f.z); } }
  console.log('ridge ahead', { x: f.x.toFixed(2), z: f.z.toFixed(2), zMax: zMax.toFixed(2), minGap: minGap.toFixed(3), yaw: f.yaw.toFixed(2), obstacle: f.obstacle });
  assert(f.x > 0.55 || Math.abs(f.yaw) > 0.3, 'the fly crossed the ridge or turned');
  if (f.x > 0.55) assert(minGap >= c.WORLD.bodyClearance - 1e-6, `over the ridge the root kept its clearance (gap ${minGap.toFixed(3)})`);
  // too tall to fly over inside the glass: turn away, never enter
  for (let j = 0; j < n; j++) for (let i = 26; i < 30; i++) top[j * n + i] = 1.15;
  f.x = -0.3; f.y = 0; f.z = F.cruiseZ; f.yaw = 0; f.yawRate = 0;
  let entered = false, maxX = -Infinity;
  for (let i = 0; i < 3000 && f.state === 'flight'; i++) { c.stepFlight(quiet(), data, 0.001); maxX = Math.max(maxX, f.x); if (f.x > 0.3 - c.WORLD.bodyRadius && f.x < 0.5 + c.WORLD.bodyRadius) entered = true; }
  console.log('wall too tall', { x: f.x.toFixed(2), maxX: maxX.toFixed(3), z: f.z.toFixed(2), yaw: f.yaw.toFixed(2), obstacle: f.obstacle, entered });
  assert(!entered, 'the root never comes within its radius of something it cannot fly over');
  assert(Math.abs(f.yaw) > 0.3 || f.x < 0.3, 'it turns away from it');
  c.world.ground = null; f.obstacle = false;
}
// the edge of the floor stops a bout
f.state = 'ground'; f.walk.bout = 0;
f.walk.bout = 1; f.walk.dir = 1; f.state = 'walk'; f.x = 0.58; f.y = 0; f.yaw = 0; f.standZ = standZ;   // fallback disc edge (no ground map here)
const blocked = f.walk.blocked;
for (let i = 0; i < 300; i++) c.stepFlight(stepper(0, 0, 70, 53), data, 0.001);
assert(f.walk.blocked > blocked && f.x < 0.6 && Math.abs(f.yaw) > 0.2, 'turns at the edge of the walkable floor instead of leaving it');
f.state = 'ground'; f.walk.bout = 0; c.resetSim(); f.enabled = true;

// 4. Pure controller checks with a scripted brain: bounds and the disable switch.
const fake = { rate: { dn_escwing_l:300, dn_escwing_r:300, dn_steer_l:110, dn_steer_r:20 } };
for (let i = 0; i < 300; i++) c.stepFlight(fake, data, 0.001);
assert.notEqual(f.state, 'ground', 'scripted escape drive launches');
for (let i = 0; i < 1000; i++) c.stepFlight(fake, data, 0.001);
assert(f.asym > 0.9 && f.yawRate > 0, 'full left asymmetry turns left');
// a threat on the left drives the left escape DN alone (measured); the fly turns away, right
const leftThreat = { rate: { dn_escwing_l:220, dn_escwing_r:0, dn_steer_l:60, dn_steer_r:60 } };
for (let i = 0; i < 600; i++) c.stepFlight(leftThreat, data, 0.001);
assert(f.yawRate < -0.5, 'escape asymmetry turns away from the threat');
const B = f.bounds;
f.x = B.cx + B.r - 0.02; f.y = B.cy; f.yaw = 0;               // at the wall, heading out
let maxR = 0, loomSeen = 0;
for (let i = 0; i < 2000; i++) { c.stepFlight(fake, data, 0.001); maxR = Math.max(maxR, Math.hypot(f.x - B.cx, f.y - B.cy)); loomSeen = Math.max(loomSeen, brain.stim.looming); }
assert(maxR <= B.r + 1e-9, 'never leaves the bounds');
assert(loomSeen > 0.5 * F.wallLoom && loomSeen <= F.wallLoom + 1e-9, 'the wall looms at the brain, gently');
f.state = 'ground'; f.wallLoom = 0; c.stimPulse.looming = 0; c.stimSwitch.looming = 0; c.applyStim('looming');
f.enabled = false; f.escape = 0;
for (let i = 0; i < 1000; i++) c.stepFlight(fake, data, 0.001);
assert.equal(f.state, 'ground', 'disabled flight never launches');
f.enabled = true;
console.log('PASS: escape-circuit takeoff, wingbeat, steering sign, landing on walkable floor, shuffle resumes, walking bouts, backward and turning, floor edge, over and around obstacles, bounds, turn-away, wall looming, disable switch');
