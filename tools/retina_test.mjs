// The eyes: the inferred retinotopy file, the face/pixel sampling math, the per-cell kernel
// drive, and the world path that hands what the photoreceptors see to the brain.
// Usage: npm test (or build first, then node tools/retina_test.mjs)
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';
import loadMujoco from '../web/vendor/mujoco_wasm.js';
import { Brain } from '../dist/brain.js';

process.chdir(fileURLToPath(new URL('..', import.meta.url)));
// 1. the retinotopy file
const retina = JSON.parse(fs.readFileSync('web/brain/retina.json'));
const meta = JSON.parse(fs.readFileSync('web/brain/meta.json'));
const visual = new Set(meta.roles.visual), left = new Set(meta.roles.visual_l), right = new Set(meta.roles.visual_r);
assert.equal(retina.n, retina.idx.length); assert(retina.n > 11000, 'nearly every photoreceptor is placed');
assert(retina.idx.every(i => visual.has(i)), 'only photoreceptors are placed');
assert(retina.idx.every((i, k) => retina.eye[k] === 2 || (retina.eye[k] === 0) === left.has(i)), 'eye follows the FlyWire side');
const azs = retina.az.map(v => v / 10), els = retina.el.map(v => v / 10);
assert(Math.min(...azs) >= -15 && Math.max(...azs) <= 155 && Math.min(...els) >= -65 && Math.max(...els) <= 75, 'angles within the eye field');
const frontal = retina.type.filter((t, k) => t === 0 && Math.abs(azs[k]) < 20).length, ocelli = retina.type.filter(t => t === 3).length;
console.log('retina', { cells: retina.n, r16: retina.type.filter(t => t === 0).length, r7: retina.type.filter(t => t === 1).length, r8: retina.type.filter(t => t === 2).length, ocelli, frontalR16: frontal });
assert(ocelli > 200 && els.every((e, k) => retina.type[k] !== 3 || e === 75), 'the ocelli look up');

// 2. the controller section against the real brain and body
const mujoco = await loadMujoco();
mujoco.FS.mkdir('/w'); mujoco.FS.mkdir('/w/assets');
const manifest = JSON.parse(fs.readFileSync('web/model/manifest.json'));
for (const name of ['scene.xml', 'fruitfly.xml', ...manifest.assets.map(a => `assets/${a}`)]) mujoco.FS.writeFile(`/w/${name}`, fs.readFileSync(`web/model/${name}`));
const model = mujoco.MjModel.loadFromXML('/w/scene.xml');
const data = new mujoco.MjData(model);
function blob(name, Type) { const buf = zlib.gunzipSync(fs.readFileSync(`web/brain/${name}.bin.gz`)); return new Type(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)); }
const brain = new Brain(meta, blob('indptr', Uint32Array), blob('colidx', Uint32Array), Float32Array.from(blob('w2', Int16Array), x => x * 0.005 / 2));
brain.calibrate();
const sim = { steps:0, brainStartMs:0 };
const context = vm.createContext({ mujoco, model, data, brain, sim, performance });
const app = fs.readFileSync('dist/app.js', 'utf8');
const start = app.indexOf('// ---------------------------------------------------------------- pose hold');
const end = app.indexOf('// ---------------------------------------------------------------- main', start);
vm.runInContext(app.slice(start, end) + `
globalThis.controller = { resetSim, buildDriveMap, buildShuffleMap, buildFlightMap, buildWorldMap, stepSimulation, stepMs, stepWorld, world, WORLD, flight, vision, EYE, EYE_AXES, buildRetina, facePixel, sampleRetina, stimWorld };`, context);
const c = context.controller;
c.resetSim(); c.buildDriveMap(model, brain); c.buildShuffleMap(); c.buildFlightMap(); c.buildWorldMap();
c.flight.enabled = false;
const advance = (s) => { for (let i = 0; i < Math.round(s * 1000); i++) c.stepMs(); };

// the head basis at rest: forward is the world +x, up is +z
const R = c.buildRetina(retina);
const b = c.vision.basis;
const dirWorld = (k) => { const H = mujoco.mj_name2id(model, 1, 'head'); const q = data.xquat.subarray(H * 4, H * 4 + 4); const [w, x, y, z] = q;
  const v = [R.dir[k * 3], R.dir[k * 3 + 1], R.dir[k * 3 + 2]];
  return [(1 - 2 * (y * y + z * z)) * v[0] + 2 * (x * y - w * z) * v[1] + 2 * (x * z + w * y) * v[2], 2 * (x * y + w * z) * v[0] + (1 - 2 * (x * x + z * z)) * v[1] + 2 * (y * z - w * x) * v[2], 2 * (x * z - w * y) * v[0] + 2 * (y * z + w * x) * v[1] + (1 - 2 * (x * x + y * y)) * v[2]]; };
const kFront = azs.findIndex((a, k) => retina.type[k] === 0 && Math.abs(a) < 2 && Math.abs(els[k]) < 5);
const kLeftRear = azs.findIndex((a, k) => retina.type[k] === 0 && retina.eye[k] === 0 && a > 85 && a < 95 && Math.abs(els[k]) < 5);
const kRightRear = azs.findIndex((a, k) => retina.type[k] === 0 && retina.eye[k] === 1 && a > 85 && a < 95 && Math.abs(els[k]) < 5);
const kUp = retina.type.findIndex(t => t === 3);
const dF = dirWorld(kFront), dL = dirWorld(kLeftRear), dR = dirWorld(kRightRear), dU = dirWorld(kUp);
console.log('directions', { front: dF.map(v => v.toFixed(2)), leftSide: dL.map(v => v.toFixed(2)), rightSide: dR.map(v => v.toFixed(2)), ocellus: dU.map(v => v.toFixed(2)) });
assert(dF[0] > 0.95, 'a frontal cell looks along the body axis');
assert(dL[1] > 0.95, 'a left-eye cell at 90 deg looks to the left');
assert(dR[1] < -0.95, 'a right-eye cell at 90 deg looks to the right');
assert(dU[2] > 0.9, 'an ocellus looks up');

// 3. the face/pixel math: each world axis lands on its own face, at the centre; up is the top row
const size = 8;
assert.deepEqual(c.facePixel(1, 0, 0, size)[0], 0); assert.deepEqual(c.facePixel(-1, 0, 0, size)[0], 1);
assert.deepEqual(c.facePixel(0, 1, 0, size)[0], 2); assert.deepEqual(c.facePixel(0, -1, 0, size)[0], 3);
assert.deepEqual(c.facePixel(0, 0, 1, size)[0], 4); assert.deepEqual(c.facePixel(0, 0, -1, size)[0], 5);
assert.equal(c.facePixel(1, 0, 0, size)[1], (4 * size + 4) * 4, 'the axis itself is the centre pixel');
assert.equal(c.facePixel(1, 0, 0.99, size)[1], (7 * size + 4) * 4, 'looking up within the +x face reaches the top row');
assert.equal(c.facePixel(1, 0.99, 0, size)[1] % (size * 4) / 4, 0, 'looking left within the +x face (right = forward x up = -y) reaches column 0');

// 4. sampling: light only in the +x face -> frontal cells see it, the sides do not; the drive lands on those cells only
const faces = { size, data: c.EYE_AXES.map(() => new Uint8Array(size * size * 4)), t: 1 };
faces.data[0].fill(255);   // white ahead
c.sampleRetina(data, R, faces, 1.0, brain.stimDrive);
assert(R.lum[kFront] > 0.99 && R.lum[kLeftRear] === 0 && R.lum[kRightRear] === 0, 'luminance per cell follows its direction');
assert(Math.abs(R.amt[kFront] - c.WORLD.lightMax * brain.stimDrive) < 1e-6, 'drive = lightMax x luminance x stimDrive');
let lit = 0; for (let k = 0; k < R.n; k++) if (R.lum[k] > 0.5) lit++;
console.log('white wall ahead', { litCells: lit, meanLeft: R.meanEye[0].toFixed(3), meanRight: R.meanEye[1].toFixed(3), ocelli: R.meanEye[2].toFixed(3) });
assert(lit > 500 && lit < R.n / 2, 'the frontal field is a minority of the cells');
assert(Math.abs(R.meanEye[0] - R.meanEye[1]) < 0.02 * (R.meanEye[0] + R.meanEye[1]) + 1e-3, 'a wall straight ahead is seen alike by both eyes');
// a sudden change is a transient on top of the level
faces.data[0].fill(0); c.sampleRetina(data, R, faces, 1.05, brain.stimDrive);
assert(R.filt[kFront] > 5 && R.amt[kFront] > 0, 'a wall going dark is a transient on the cells that saw it');
c.sampleRetina(data, R, faces, 1.5, brain.stimDrive); c.sampleRetina(data, R, faces, 2.0, brain.stimDrive);
assert(R.amt[kFront] < 1e-3, 'and it decays');

// 5. the world hands the cells to the kernel: light on the left only drives the left eye's photoreceptors
faces.data[2].fill(255);   // world +y: the fly's left at rest
c.world.enabled = true; c.vision.faces = faces;
c.sampleRetina(data, R, faces, 3.0, brain.stimDrive);
for (let i = 0; i < 300; i++) c.stepWorld(data, 0.001);   // several ticks: the world adopts the eyes
assert(c.vision.active, 'the eyes are active with a retina, faces and the world on');
assert(c.stimWorld.light[0] === 0 && c.stimWorld.light[1] === 0, 'the scalar light level steps aside');
advance(1.5);
console.log('lit from the left', { visual_l: brain.rate.visual_l.toFixed(1), visual_r: brain.rate.visual_r.toFixed(1), rest: brain.rest.visual.toFixed(1), rest_l: brain.rest.visual_l.toFixed(1), level: c.world.levels.light.toFixed(3) });
assert(brain.rate.visual_l > 3 * Math.max(brain.rate.visual_r, 1), 'the left eye fires, the right rests');
assert(c.world.levels.light > 0 && c.world.levels.light < c.WORLD.lightMax, 'the reported light level is the mean drive');
// world off: the eyes stop with everything else
c.world.enabled = false; c.stepWorld(data, 0.001); advance(1.0); console.log('world off', { visual_l: brain.rate.visual_l.toFixed(1), rest_l: brain.rest.visual_l.toFixed(1) });
assert(!c.vision.active && brain._cellIdx === null, 'world off clears the per-cell drive');
assert(brain.rate.visual_l < brain.rest.visual_l + 3, 'and the left eye returns to rest');
c.world.enabled = true; c.vision.faces = null; c.stepWorld(data, 0.001);
assert(!c.vision.active && c.stimWorld.light[0] > 0, 'without a render the scalar light level drives the pool');
console.log('PASS: retinotopy file, head-frame directions, face sampling, per-cell drive, the world hands the eyes to the kernel, world off');
