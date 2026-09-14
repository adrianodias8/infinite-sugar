// Flybody (MuJoCo Menagerie) in MuJoCo WASM, rendered with three.js.
// Bridge pattern follows the established mujoco_wasm demos: build one three.js mesh per
// mjModel geom, then each frame copy data.geom_xpos / data.geom_xmat onto it.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import loadMujoco from './vendor/mujoco_wasm.js';
import { Brain } from './brain.js';
import { createNeuralMap } from './neural-map.js';
import { AdaptiveQuality } from './performance.js';
import * as CANNON from 'cannon-es';

import type { MujocoModule, MjModel, MjData } from './vendor/mujoco_wasm.js';
type GeomNode = { mesh: THREE.Mesh, group: number, gi: number, isFloor: boolean };
type PropDefinition = { file: string, target: number, abs?: [number, number, number], pos?: [number, number], floor?: [number, number], rot: number };
type BallState = { wrap: THREE.Group, body: CANNON.Body, radius: number, thoraxBody: number, flyProxy: CANNON.Body };
type DriveDefinition = { act: string, role: string, gain?: number, to?: number, peak?: number, band?: [number, number], raw?: [number, number], cmd?: boolean, smooth?: number };
type DriveEntry = DriveDefinition & { ai: number, qadr: number, dadr: number };

function requiredElement(id: string) {
  const element = document.getElementById(id);
  if (!element) throw new Error(`missing required element #${id}`);
  return element;
}

const boot = requiredElement('boot');
const bootmsg = requiredElement('bootmsg');
const say = (message: string) => {
  bootmsg.textContent = message.charAt(0).toUpperCase() + message.slice(1);
};
const $ = requiredElement;

const MODEL_DIR = './model';
const SCENE_XML = 'scene.xml';

// mjtGeom
const G = { PLANE:0, HFIELD:1, SPHERE:2, CAPSULE:3, ELLIPSOID:4, CYLINDER:5, BOX:6, MESH:7 };

let mujoco: MujocoModule;
let model: MjModel;
let data: MjData;
let brain: Brain;
const sim = { paused:false, steps:0, t0:0, brainStartMs:0 };
const geomNodes: GeomNode[] = [];
const tmpMat = new THREE.Matrix4();

// ---------------------------------------------------------------- filesystem
async function stageFiles(mj: MujocoModule) {
  const manifest = (await (await fetch(`${MODEL_DIR}/manifest.json`)).json() as { assets: string[] });
  mj.FS.mkdir('/w'); mj.FS.mkdir('/w/assets');
  let done = 0;
  const files = [SCENE_XML, 'fruitfly.xml', ...manifest.assets.map(a => 'assets/' + a)];
  await Promise.all(files.map(async (f) => {
    const buf = new Uint8Array(await (await fetch(`${MODEL_DIR}/${f}`)).arrayBuffer());
    mj.FS.writeFile('/w/' + f, buf);
    if (++done % 20 === 0) say(`loading body assets ${done}/${files.length}`);
  }));
  say(`loaded ${files.length} body assets`);
}

// ---------------------------------------------------------------- geometry
function meshGeometry(m: MjModel, dataid: number) {
  const va = m.mesh_vertadr[dataid], vn = m.mesh_vertnum[dataid];
  const fa = m.mesh_faceadr[dataid], fn = m.mesh_facenum[dataid];
  const pos = new Float32Array(vn * 3);
  const nrm = new Float32Array(vn * 3);
  pos.set(m.mesh_vert.subarray(va * 3, (va + vn) * 3));
  nrm.set(m.mesh_normal.subarray(va * 3, (va + vn) * 3));
  const idx = new Uint32Array(fn * 3);
  idx.set(m.mesh_face.subarray(fa * 3, (fa + fn) * 3));
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  return g;
}

function primitiveGeometry(type: number, sx: number, sy: number, sz: number) {
  switch (type) {
    case G.PLANE:     return new THREE.PlaneGeometry(40, 40, 1, 1);
    case G.SPHERE:    return new THREE.SphereGeometry(sx, 20, 14);
    case G.CAPSULE:   return new THREE.CapsuleGeometry(sx, 2 * sy, 6, 14);
    case G.ELLIPSOID: { const g = new THREE.SphereGeometry(1, 20, 14); g.scale(sx, sy, sz); return g; }
    case G.CYLINDER:  return new THREE.CylinderGeometry(sx, sx, 2 * sy, 20);
    case G.BOX:       return new THREE.BoxGeometry(2 * sx, 2 * sy, 2 * sz);
    default:          return null;
  }
}

// MuJoCo cylinders/capsules point along +Z; three.js primitives point along +Y.
const NEEDS_Z_UP = new Set([G.CAPSULE, G.CYLINDER]);

// ---------------------------------------------------------------- skybox
// Frutiger Aero: glossy, optimistic, pale blue-to-mint gradient with soft cloud blobs.
// Procedural — a canvas gradient, not a downloaded HDRI, so it costs nothing to keep in sync
// with the rest of the "no bundler" convention.
function skyTexture() {
  const W = 512, H = 512, cv = document.createElement('canvas');
  cv.width = W; cv.height = H;
  const g = cv.getContext('2d');
  if (!g) throw new Error('2D canvas is unavailable');
  const sky = g.createLinearGradient(0, 0, 0, H);
  sky.addColorStop(0.00, '#8fc7ef');
  sky.addColorStop(0.45, '#bfe3ef');
  sky.addColorStop(0.78, '#e3f6ee');
  sky.addColorStop(1.00, '#f3fff6');
  g.fillStyle = sky; g.fillRect(0, 0, W, H);

  // a soft glossy sun-glow, upper-left — subtle, not a lens flare
  const glow = g.createRadialGradient(W*0.28, H*0.20, 0, W*0.28, H*0.20, W*0.32);
  glow.addColorStop(0, 'rgba(255,255,255,0.55)');
  glow.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = glow; g.fillRect(0, 0, W, H);

  // a handful of soft cloud blobs, low contrast
  const clouds = [[0.62,0.30,0.14],[0.74,0.34,0.09],[0.20,0.55,0.11],[0.85,0.62,0.08],[0.45,0.68,0.10]];
  for (const [cx, cy, r] of clouds) {
    const cg = g.createRadialGradient(W*cx, H*cy, 0, W*cx, H*cy, W*r);
    cg.addColorStop(0, 'rgba(255,255,255,0.5)');
    cg.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = cg; g.fillRect(0, 0, W, H);
  }

  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

let _checker: THREE.CanvasTexture | null = null;
function checkerTexture() {
  if (_checker) return _checker;
  const N = 256, cv = document.createElement('canvas');
  cv.width = cv.height = N;
  const g = cv.getContext('2d');
  if (!g) throw new Error('2D canvas is unavailable');
  g.fillStyle = '#1a2430'; g.fillRect(0, 0, N, N);
  g.fillStyle = '#233042'; g.fillRect(0, 0, N/2, N/2); g.fillRect(N/2, N/2, N/2, N/2);
  _checker = new THREE.CanvasTexture(cv);
  _checker.wrapS = _checker.wrapT = THREE.RepeatWrapping;
  _checker.repeat.set(80, 80);
  _checker.colorSpace = THREE.SRGBColorSpace;
  _checker.anisotropy = 8;
  return _checker;
}

function buildScene(scene: THREE.Scene, m: MjModel) {
  let tris = 0;
  for (let i = 0; i < m.ngeom; i++) {
    const type = m.geom_type[i];
    const sx = m.geom_size[i*3], sy = m.geom_size[i*3+1], sz = m.geom_size[i*3+2];
    let geo = type === G.MESH ? meshGeometry(m, m.geom_dataid[i]) : primitiveGeometry(type, sx, sy, sz);
    if (!geo) continue;
    if (NEEDS_Z_UP.has(type)) geo.rotateX(Math.PI / 2);

    // colour: MuJoCo uses geom_rgba when it was set explicitly, otherwise the material.
    // Default geom_rgba is (.5,.5,.5,1) — treat that as "not set".
    const mid = m.geom_matid[i];
    let r = m.geom_rgba[i*4], g = m.geom_rgba[i*4+1], b = m.geom_rgba[i*4+2], a = m.geom_rgba[i*4+3];
    const isDefaultRgba = (r === 0.5 && g === 0.5 && b === 0.5 && a === 1);
    if (isDefaultRgba && mid >= 0) {
      r=m.mat_rgba[mid*4]; g=m.mat_rgba[mid*4+1]; b=m.mat_rgba[mid*4+2]; a=m.mat_rgba[mid*4+3];
    }
    if (a === 0) { geo.dispose(); continue; }       // e.g. the wing inertial boxes

    const grp = m.geom_group[i];
    const isFloor = type === G.PLANE;
    const mat = isFloor
      ? new THREE.MeshStandardMaterial({ map: checkerTexture(), roughness:.95, metalness:0 })
      : new THREE.MeshStandardMaterial({
          color: new THREE.Color(r, g, b), roughness:.55, metalness:.12,
          transparent: a < 1, opacity: a, depthWrite: a >= 1, side: THREE.DoubleSide });

    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = !isFloor; mesh.receiveShadow = true;
    mesh.frustumCulled = false;
    mesh.visible = grp <= 2 && !isFloor;           // collision groups (3+) hidden by default;
                                                    // checker floor hidden — terrarium.glb's own
                                                    // floor sits flush over it (see PROPS above)
    scene.add(mesh);
    geomNodes.push({ mesh, group: grp, gi: i, isFloor });   // gi = mjModel geom index
    if (geo.index) tris += geo.index.count / 3;
  }
  return tris;
}

function syncGeoms(m: MjModel, d: MjData) {
  const R = d.geom_xmat;                            // row-major 3x3 per geom
  for (let k = 0, n = geomNodes.length; k < n; k++) {
    const i = geomNodes[k].gi;
    const o = geomNodes[k].mesh;
    o.position.set(d.geom_xpos[i*3], d.geom_xpos[i*3+1], d.geom_xpos[i*3+2]);
    tmpMat.set(R[i*9+0], R[i*9+1], R[i*9+2], 0,
               R[i*9+3], R[i*9+4], R[i*9+5], 0,
               R[i*9+6], R[i*9+7], R[i*9+8], 0,
               0, 0, 0, 1);
    o.quaternion.setFromRotationMatrix(tmpMat);
  }
}

// ---------------------------------------------------------------- terrarium
// Purely decorative CC-BY props (web/model/props/CREDITS.md), staged around the fly by
// measured scale rather than eyeballed units: each prop's own bounding box is computed after
// load and rescaled against the fly's actual bounding box, so this survives model changes.
// No physics involvement — these do not exist in the MuJoCo model and cast/receive shadows only.
const PROPS: PropDefinition[] = [
  // abs: tuned by hand via a live debug panel (x/y/z inputs bound straight to this object's
  // position) — not derived, not guessed. z sits the terrarium's floor flush with the world floor.
  { file: 'terrarium.glb', target: 8.25, abs: [-0.5, -0.15, 1.58729], rot: 0 },
  // The world's sugar: a sack on the floor a short walk ahead of the perch. floor: x, y with
  // the bottom of the model on the physics floor. Its footprint becomes the taste patch.
  { file: 'sugar_sack.glb', target: 0.4, floor: [0.42, 0.10], rot: 0.5 },
];
const propObjs: Record<string, THREE.Group> = {};   // filename -> THREE.Group, for the live position debug panel
async function loadProps(scene: THREE.Scene, flyBox: THREE.Box3) {
  const flySize = new THREE.Vector3();
  flyBox.getSize(flySize);
  const flySpan = Math.max(flySize.x, flySize.y);   // footprint, not height — props are floor items
  const loader = new GLTFLoader();
  for (const p of PROPS) {
    let gltf;
    try {
      gltf = await loader.loadAsync(`./model/props/${p.file}`);
    } catch (err) {
      console.warn(`prop "${p.file}" failed to load — skipping`, err);
      continue;
    }
    const raw = gltf.scene;
    raw.rotation.x = Math.PI / 2;   // glTF is Y-up; this scene is Z-up (matches MuJoCo)

    // Old Google-Poly-era exports (which most free CC-BY props are) often carry a baked root
    // transform with the mesh sitting far from local (0,0,0) — trusting the file's own origin
    // put objects hundreds of units away. Measure the true bounding box and recenter blind to
    // whatever origin the file shipped with.
    const box0 = new THREE.Box3().setFromObject(raw);
    const size0 = new THREE.Vector3(), center0 = new THREE.Vector3();
    box0.getSize(size0); box0.getCenter(center0);
    raw.position.sub(center0);              // bbox center now sits at raw's local origin

    const rawSpan = Math.max(size0.x, size0.y, size0.z) || 1;
    const scale = (flySpan * p.target) / rawSpan;

    const wrap = new THREE.Group();
    wrap.add(raw);
    wrap.scale.setScalar(scale);

    if (p.abs) {
      wrap.position.set(...p.abs);
    } else if (p.floor) {
      const box1 = new THREE.Box3().setFromObject(wrap);
      const size1 = new THREE.Vector3(); box1.getSize(size1);
      wrap.position.set(p.floor[0], p.floor[1], WORLD.floorZ - box1.min.z);
      if (p.file === 'sugar_sack.glb') world.sugar = { x: p.floor[0], y: p.floor[1], r: 0.5 * Math.max(size1.x, size1.y), placed: true, amount: world.sugar.amount };   // r: the sack's surface
    } else {
      if (!p.pos) throw new Error(`prop "${p.file}" needs abs or pos coordinates`);
      const box1 = new THREE.Box3().setFromObject(wrap);  // re-measure post-scale, world origin
      wrap.position.set(p.pos[0] * flySpan, p.pos[1] * flySpan, -box1.min.z);
    }
    wrap.rotation.z = p.rot;
    wrap.traverse(n => { if (n instanceof THREE.Mesh) { n.castShadow = true; n.receiveShadow = true; } });
    scene.add(wrap);
    propObjs[p.file] = wrap;
  }
}

// ---------------------------------------------------------------- beach ball physics
// cannon-es (vendored single-file ESM, github.com/pmndrs/cannon-es), not a hand-rolled raycast
// sim. The terrarium's own geometry is baked directly into static trimesh colliders — the real
// hill mesh and the real glass mesh (isolated by its alpha-blend material, confirmed against the
// glTF's material table), not an approximated cylinder. cannon-es owns integration, friction,
// restitution and rolling (real angular velocity from contact friction, not a cosmetic spin);
// world.step()'s own fixed-timestep sub-stepping is what prevents tunneling through the glass at
// low render framerate, rather than a hand-clamped frame delta.
let ball: BallState | null = null;
let ballWorld: CANNON.World | null = null;
const ballTilt = { phase: 0 };
const viewOccluders: THREE.Mesh[] = [];   // the terrarium's opaque meshes, for the camera's line of sight
const _ray = new THREE.Raycaster();      // still used once at load, to find the "hilltop" start
const _down = new THREE.Vector3(0, 0, -1);
function groundUnder(meshes: THREE.Object3D[], x: number, y: number) {
  _ray.set(new THREE.Vector3(x, y, 50), _down);
  const hits = _ray.intersectObjects(meshes, true);
  return hits.length ? hits[0] : null;
}

// Bakes a THREE mesh's real triangles (in world space, at load time — the terrarium never
// moves) into a flat vertex/index pair, ready to append into a combined CANNON.Trimesh.
function bakeMeshTriangles(mesh: THREE.Mesh, vertsOut: number[], idxOut: number[]) {
  mesh.updateWorldMatrix(true, false);
  const geo = mesh.geometry;
  const pos = geo.attributes.position;
  const base = vertsOut.length / 3;
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i).applyMatrix4(mesh.matrixWorld);
    vertsOut.push(v.x, v.y, v.z);
  }
  if (geo.index) {
    for (let i = 0; i < geo.index.count; i++) idxOut.push(base + geo.index.getX(i));
  } else {
    for (let i = 0; i < pos.count; i++) idxOut.push(base + i);
  }
}
function trimeshFromMeshes(meshes: THREE.Mesh[]) {
  if (!meshes.length) return null;
  const verts: number[] = [];
  const idx: number[] = [];
  for (const m of meshes) bakeMeshTriangles(m, verts, idx);
  return new CANNON.Trimesh(verts, idx);
}

async function loadBall(scene: THREE.Scene, flyBox: THREE.Box3, hillMeshes: THREE.Mesh[], glassMeshes: THREE.Mesh[], solidMeshes: THREE.Mesh[]) {
  const flySize = new THREE.Vector3();
  flyBox.getSize(flySize);
  const flySpan = Math.max(flySize.x, flySize.y);
  const loader = new GLTFLoader();
  let gltf;
  try {
    gltf = await loader.loadAsync('./model/props/beach_ball.glb');
  } catch (err) {
    console.warn('beach ball failed to load — skipping', err);
    return;
  }
  const raw = gltf.scene;
  raw.rotation.x = Math.PI / 2;
  const box0 = new THREE.Box3().setFromObject(raw);
  const size0 = new THREE.Vector3(), center0 = new THREE.Vector3();
  box0.getSize(size0); box0.getCenter(center0);
  raw.position.sub(center0);
  const rawSpan = Math.max(size0.x, size0.y, size0.z) || 1;
  const scale = (flySpan * 0.55) / rawSpan;
  const wrap = new THREE.Group();
  wrap.add(raw);
  wrap.scale.setScalar(scale);
  wrap.traverse(n => { if (n instanceof THREE.Mesh) { n.castShadow = true; n.receiveShadow = true; } });
  scene.add(wrap);
  propObjs['beach_ball.glb'] = wrap;

  const box1 = new THREE.Box3().setFromObject(wrap);
  const radius = Math.max(0.001, (box1.max.z - box1.min.z) / 2);

  // sample the terrarium's own surface for its highest reachable point ("top of the hill") —
  // not hand-placed, found the same way a marble dropped at random would find it. One-time
  // query at load, so a plain raycast (not a physics query) is the right tool.
  let best: THREE.Intersection | null = null;
  for (let i = 0; i < 40; i++) {
    const x = -0.5 + (Math.random() - 0.5) * flySpan * 3.2;
    const y = -0.15 + (Math.random() - 0.5) * flySpan * 3.2;
    const hit = groundUnder(hillMeshes, x, y);
    if (hit && (!best || hit.point.z > best.point.z)) best = hit;
  }
  const start = best ? best.point : new THREE.Vector3(-0.5, -0.15, 1.6);
  wrap.position.set(start.x, start.y, start.z + radius + flySpan * 0.05);

  // ---- cannon-es world ----
  const physicsWorld = new CANNON.World({ gravity: new CANNON.Vec3(0, 0, -1.6) });   // stylized — small terrarium
  physicsWorld.broadphase = new CANNON.SAPBroadphase(physicsWorld);
  physicsWorld.allowSleep = true;
  ballWorld = physicsWorld;

  const groundMat = new CANNON.Material('ground');
  const ballMat = new CANNON.Material('ball');
  physicsWorld.addContactMaterial(new CANNON.ContactMaterial(groundMat, ballMat, {
    friction: 0.5, restitution: 0.35,
  }));

  const terrainBody = new CANNON.Body({ mass: 0, material: groundMat });
  // Collision uses every opaque mesh (rocks, floor, flowers, fern, frame) — NOT hillMeshes,
  // which is height-filtered down to short ground-level decor for a different purpose (finding
  // a "hilltop" to start the ball on, below). Reusing that filtered set as the collider was the
  // actual bug: the fern and flowers were excluded from collision entirely, so the ball rolled
  // straight through them. A wall doesn't stop being solid for being tall.
  const solidTri = trimeshFromMeshes(solidMeshes);
  if (solidTri) terrainBody.addShape(solidTri);
  const glassTri = trimeshFromMeshes(glassMeshes);
  if (glassTri) terrainBody.addShape(glassTri);

  // Backstop: old Google-Poly-era decorative exports are built for rendering, not physics, and
  // are routinely NOT watertight — measured, the real glass trimesh has a seam somewhere (a ball
  // rolled through it at a perfectly ordinary 2 u/s in testing, nowhere near a tunneling speed).
  // A ring of inward-facing infinite planes is defense in depth: the real mesh still gives the
  // close-up bounce its correct shape, this just guarantees nothing ever visibly escapes through
  // whatever gap the source model has. (Not a solid Cylinder shape — that's filled geometry, and
  // spawning the ball inside one gets it shoved OUTWARD as overlap resolution, the exact opposite
  // of containment. A CANNON.Plane's solid side is behind its local +Z normal, so a ring of them
  // with normals pointing inward is a real hollow boundary, not a solid object to be pushed out of.)
  if (glassMeshes.length) {
    const gbox = new THREE.Box3();
    for (const m of glassMeshes) gbox.expandByObject(m);
    const gsize = new THREE.Vector3(), gcenter = new THREE.Vector3();
    gbox.getSize(gsize); gbox.getCenter(gcenter);
    const backstopRadius = 0.85 * Math.min(gsize.x, gsize.y) / 2;
    const N = 12;
    for (let i = 0; i < N; i++) {
      const theta = (i / N) * Math.PI * 2;
      const dir = new THREE.Vector3(Math.cos(theta), Math.sin(theta), 0);   // outward
      const pos = dir.clone().multiplyScalar(backstopRadius).add(gcenter);
      const q3 = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), dir.clone().negate());
      terrainBody.addShape(new CANNON.Plane(),
        new CANNON.Vec3(pos.x, pos.y, pos.z),
        new CANNON.Quaternion(q3.x, q3.y, q3.z, q3.w));
    }
  }
  physicsWorld.addBody(terrainBody);

  const ballBody = new CANNON.Body({
    mass: 0.05, shape: new CANNON.Sphere(radius), material: ballMat,
    linearDamping: 0.05, angularDamping: 0.1,
    position: new CANNON.Vec3(wrap.position.x, wrap.position.y, wrap.position.z),
  });
  physicsWorld.addBody(ballBody);

  // fly proxy: a KINEMATIC body (moves the world, is never moved by it) at the thorax's live
  // MuJoCo position — genuinely one-way, not a hand-rolled approximation of one-way. The fly's
  // real physics is never written to; cannon-es's own kinematic/dynamic contact handles the push.
  const thoraxBody = mujoco.mj_name2id(model, 1 /* mjOBJ_BODY */, 'thorax');
  const flyProxy = new CANNON.Body({ mass: 0, type: CANNON.Body.KINEMATIC,
                                      shape: new CANNON.Sphere(flySpan * 0.5), material: groundMat });
  physicsWorld.addBody(flyProxy);
  // The ball touching the fly is a touch on that flank (cannon-es contact events; the fly's
  // physics is still never written to).
  ballBody.addEventListener('collide', (e: { body: CANNON.Body }) => {
    if (e.body === flyProxy) world.touchHits.push(bearingTo(data, ballBody.position.x, ballBody.position.y));
  });

  ball = { wrap, body: ballBody, radius, thoraxBody, flyProxy };
}

function stepBall(dt: number) {
  if (!ball || !ballWorld || dt <= 0) return;
  if (ball.thoraxBody >= 0 && data) {
    const bi = ball.thoraxBody * 3;
    ball.flyProxy.position.set(data.xpos[bi], data.xpos[bi + 1], data.xpos[bi + 2]);
  }
  // The ball is alive: the floor tilts very slightly on a slow circular cycle (gravity leans
  // 0.05 rad, once round every 45 s), so the ball keeps rolling somewhere and the fly gets
  // recurring looming and touch events without a hand on the button. Supplied, and cosmetic.
  ballTilt.phase += 2 * Math.PI * dt / 45;
  const lean = 0.05, g = 1.6;
  ballWorld.gravity.set(g * Math.sin(lean) * Math.cos(ballTilt.phase), g * Math.sin(lean) * Math.sin(ballTilt.phase), -g * Math.cos(lean));
  ballWorld.step(1 / 120, dt, 10);   // fixed-timestep sub-stepping — the engine's own tunneling fix
  ball.wrap.position.copy(ball.body.position);
  ball.wrap.quaternion.copy(ball.body.quaternion);
  // The ball is a moving object the fly can see: position and velocity for the world's looming.
  const p = ball.body.position, v = ball.body.velocity;
  world.loomers[0] = { x: p.x, y: p.y, z: p.z, vx: v.x, vy: v.y, vz: v.z, r: ball.radius, name: 'ball' };
}

type BrainLike = { rate: Record<string, number>, rest?: Record<string, number> | null, spikesOf?: (role: string) => number };
// ---------------------------------------------------------------- pose hold
// 64 of the 78 actuators are position-servos. Drive them to the keyframe pose so the
// fly stands instead of collapsing — no controller, no policy, just a held posture.
let holdCtrl: Float64Array;
function captureHoldPose(m: MjModel, d: MjData) {
  const c = new Float64Array(m.nu);
  for (let a = 0; a < m.nu; a++) {
    if (m.actuator_trntype[a] === 0) {            // joint transmission
      const j = m.actuator_trnid[a*2];
      c[a] = d.qpos[m.jnt_qposadr[j]];
    }
  }
  return c;
}
function resetSim() {
  mujoco.mj_resetDataKeyframe(model, data, 0);
  holdCtrl = captureHoldPose(model, data);
  data.ctrl.set(holdCtrl);
  mujoco.mj_forward(model, data);
  sim.steps = 0; sim.t0 = performance.now();
  sim.brainStartMs = brain.ms; // preserve the brain, restart the body's clock beneath it
  flight.state = 'ground'; flight.escape = 0; flight.wallLR = [0, 0]; flight.walk.bout = 0;
  resetShuffle();
}

// ------------------------------------------------------------- brain -> body
// Spike rate of an identified motor-neuron population -> position-servo target.
// No policy, nothing learned, nothing scripted: each of these motor pools is individually
// identified in FlyWire and the joint it drives physically exists in flybody.
//
//   proboscis_motor_neuron (24) -> rostrum / haustellum / labrum   (feeding)
//   neck_motor_neuron      (26) -> head / head_twist / head_abduct (13 per side)
//   antennal_motor_neuron  (10) -> antenna_* x6                    (5 per side)
//
// COMMAND mappings are a DIFFERENT and weaker claim, kept visibly separate. FAFB is brain-only:
// wing and leg motor neurons live in the ventral nerve cord, which is not in this dataset. A
// descending neuron says "escape"; the posture it produces is supplied by us, not measured.
//
//   DNp02/04/11 'dn_escwing' (6) -> wing_roll / wing_yaw   (escape wing raise)
//   DNg11       'dn_groom'   (6) -> abdomen                (grooming)
//
// `gain` is how many multiples of the RESTING rate map to full joint excursion. Resting rates
// are measured at load by brain.calibrate() — they depend on the kernel, not just the wiring.
const DRIVE: DriveDefinition[] = [
  // smooth: a body-side low-pass (seconds) on the servo TARGET. The 25 ms rate estimate of a
  // 24-cell pool is noisy and the haustellum servo reached 50 rad/s at rest chasing it. The
  // mean extension is unchanged; only the target's jitter is filtered. Explicit and body-side:
  // the neural rate estimate itself is untouched.
  { act:'rostrum',              role:'mn_proboscis', gain:2.0, to:-1.24, smooth:0.1 },
  { act:'haustellum',           role:'mn_proboscis', gain:2.0, to:-1.59, smooth:0.1 },
  { act:'labrum_left',          role:'mn_proboscis', gain:2.0, to: 1.05, smooth:0.1 },
  { act:'labrum_right',         role:'mn_proboscis', gain:2.0, to: 1.05, smooth:0.1 },
  { act:'head',                 role:'mn_neck_r',    gain:3.0, to:-0.30 },
  { act:'head_twist',           role:'mn_neck_r',    gain:2.0, to: 0.30 },
  { act:'head_abduct',          role:'mn_neck_l',    gain:1.3, to: 0.20 },
  { act:'antenna_left',         role:'mn_antenna_l', gain:4.0, to: 0.50 },
  { act:'antenna_abduct_left',  role:'mn_antenna_l', gain:4.0, to: 0.80 },
  { act:'antenna_twist_left',   role:'mn_antenna_l', gain:4.0, to: 0.09 },
  { act:'antenna_right',        role:'mn_antenna_r', gain:4.0, to: 0.50 },
  { act:'antenna_abduct_right', role:'mn_antenna_r', gain:4.0, to: 0.80 },
  { act:'antenna_twist_right',  role:'mn_antenna_r', gain:4.0, to: 0.09 },
  // --- command mappings (see note above): absolute-rate activation, silent at rest ---
  // Wing actuators are FORCE generals (gainprm, no position bias) against very weak joint
  // springs (stiffness 0.01). Measured torque balance: the usable control band is roughly
  // ctrl in [-0.004, +0.010] — past that the joint pins against its stop and stops responding,
  // which is what made the pose lock rigidly. A PD loop here is unstable (the actuator is far
  // too strong relative to the spring), so this stays open-loop inside the proportional band.
  // ctrl 0 = no torque = the spring's natural folded pose. Ranges below are measured.
  // The wings sum THREE populations per joint (see applyBrainToActuators — several DRIVE rows
  // may target one actuator and their contributions add). Escape alone gave only two states,
  // because dn_escwing genuinely measures 0.0 +/- 0.0 Hz under every stimulus except looming —
  // that is the real data, not a wiring gap. The life comes from adding pools that are actually
  // active at rest.
  //
  // WINGCLAMP keeps the sum inside the measured proportional band; past it the joint pins
  // against its stop and stops responding to the neurons at all.
  //
  // (a) postural / idle -- DNa01+DNa02 steering DNs. Measured at rest: L 70.2 +/- 19.6 Hz,
  //     R 54.3 +/- 18.6 Hz. Large jitter AND a real standing left/right asymmetry, which
  //     INVERTS under touch (L 38.5, R 58.9) -- so the wings visibly re-trim. Steering DNs
  //     modulating left/right wing amplitude asymmetrically is their documented function, so
  //     side->side is the correct mapping rather than an arbitrary one. Deliberately small
  //     amplitude: a resting fly adjusts its wings, it does not flap them.
  { act:'wing_yaw_left',    role:'dn_steer_l',   band:[20,110], raw:[0, -0.0018], cmd:true },
  { act:'wing_yaw_right',   role:'dn_steer_r',   band:[20,110], raw:[0, -0.0018], cmd:true },
  { act:'wing_roll_left',   role:'dn_steer_l',   band:[20,110], raw:[0,  0.0010], cmd:true },
  { act:'wing_roll_right',  role:'dn_steer_r',   band:[20,110], raw:[0,  0.0010], cmd:true },
  // (b) grooming flutter -- DNg11. 22.9 +/- 6.8 Hz at rest, 33.8 Hz on touch (+48%). Flies
  //     groom their wings with the hind legs; a pitch twitch is the visible correlate.
  { act:'wing_pitch_left',  role:'dn_groom',     band:[8,45],   raw:[0,  0.0030], cmd:true },
  { act:'wing_pitch_right', role:'dn_groom',     band:[8,45],   raw:[0,  0.0030], cmd:true },
  // (c) escape deploy -- DNp02/04/11. Silent at rest, 221 Hz (L) / 170 Hz (R) on looming; note
  //     even the escape response is naturally asymmetric between the two sides.
  { act:'wing_roll_left',   role:'dn_escwing_l', peak:300, raw:[0,  0.0030], cmd:true },
  { act:'wing_roll_right',  role:'dn_escwing_r', peak:300, raw:[0,  0.0030], cmd:true },
  { act:'wing_yaw_left',    role:'dn_escwing_l', peak:300, raw:[0, -0.0035], cmd:true },
  { act:'wing_yaw_right',   role:'dn_escwing_r', peak:300, raw:[0, -0.0035], cmd:true },
  { act:'wing_pitch_left',  role:'dn_escwing_l', peak:300, raw:[0,  0.0080], cmd:true },
  { act:'wing_pitch_right', role:'dn_escwing_r', peak:300, raw:[0,  0.0080], cmd:true },
  { act:'abdomen',          role:'dn_groom',     gain:1.5, to:-0.15, cmd:true },
];
let driveMap: DriveEntry[] = [];
let driveGroups: [number, DriveEntry[]][] = [];          // [actuatorIndex, entries[]] — several pools may drive one joint
const WINGCLAMP = [-0.0055, 0.0105];   // measured usable band; past this the joint pins

// Purely cosmetic resting offset. The folded pose clips the wings through the abdomen, so lift
// them (roll+) and sweep them outward (yaw-) a little. This is a constant added to the base,
// ON TOP of whatever the neurons are doing — it shifts the resting posture without touching any
// neural mapping, so every response above still plays out from the new rest position.
// Signs follow the measured ctrl->angle curves: roll+ raises, yaw- sweeps outward.
// Re-measured with the whole body standing (six bias pairs): no bias inside the actuator band
// clears the folded wing from the hind tibia (roll saturates near 1.0 rad). Roll 0.0018 / yaw
// -0.0020 had the least resting jitter, but after a landing the tibia caught the more
// outward-swept membrane and held the wings forward at 0.56 rad, so these values stay.
const WINGBIAS: Record<string, number> = {
  wing_roll_left:   0.0012, wing_roll_right:  0.0012,
  wing_yaw_left:   -0.0012, wing_yaw_right:  -0.0012,
  wing_pitch_left:  0.0000, wing_pitch_right: 0.0000,
};
function buildDriveMap(m: MjModel, b: Brain) {
  driveMap = DRIVE
    .map(d => {
      const ai = mujoco.mj_name2id(m, 19 /* mjOBJ_ACTUATOR */, d.act);
      const ji = mujoco.mj_name2id(m, 3 /* mjOBJ_JOINT */, d.act);
      return { ...d, ai, qadr: ji >= 0 ? m.jnt_qposadr[ji] : -1, dadr: ji >= 0 ? m.jnt_dofadr[ji] : -1 };
    })
    .filter(d => d.ai >= 0 && b.groups[d.role] && b.groups[d.role].length);
  const byAct: Map<number, DriveEntry[]> = new Map();
  for (const d of driveMap) {
    if (!byAct.has(d.ai)) byAct.set(d.ai, []);
    const entries = byAct.get(d.ai);
    if (entries) entries.push(d);
  }
  driveGroups = [...byAct.entries()];
}

// Three activation modes, one per kind of population:
//   peak — pools silent at rest (escape DNs): absolute rate / peak.
//   band — pools with a real resting rate (steering, grooming): a fixed window shared by both
//          sides, so the natural left/right rate difference survives as a real pose difference
//          rather than being normalised away.
//   gain — ratio to each pool's own measured resting rate (the motor-neuron pools).
function activation(b: Brain, k: DriveEntry) {
  let a;
  if (k.peak)      a = b.rate[k.role] / k.peak;
  else if (k.band) a = (b.rate[k.role] - k.band[0]) / (k.band[1] - k.band[0]);
  else {
    if (k.gain === undefined) throw new Error(`drive ${k.act} has no activation scale`);
    const rest = (b.rest && b.rest[k.role]) || 1;
    a = (b.rate[k.role] / rest - 1) / (k.gain - 1);
  }
  return a < 0 ? 0 : a > 1 ? 1 : a;
}

let neural = true;
function applyBrainToActuators(b: Brain, d: MjData) {
  for (const [ai, entries] of driveGroups) {
    const first = entries[0];
    const base = (first.raw ? first.raw[0] : holdCtrl[ai]) + (WINGBIAS[first.act] || 0);
    if (!neural) { d.ctrl[ai] = base; continue; }
    let v = base;
    for (const k of entries) {
      const a = activation(b, k);
      if (k.raw) v += a * (k.raw[1] - k.raw[0]);
      else {
        if (k.to === undefined) throw new Error(`drive ${k.act} has no target`);
        v += a * (k.to - holdCtrl[ai]);
      }
    }
    if (first.raw) v = Math.max(WINGCLAMP[0], Math.min(WINGCLAMP[1], v));
    if (first.smooth) { const k = 0.001 / first.smooth; v = d.ctrl[ai] + (v - d.ctrl[ai]) * k; }   // called once per brain ms
    d.ctrl[ai] = v;
  }
}

// ------------------------------------------------------------- neural foot shuffle
// A supplied command mapping, like the wings: FAFB contains descending neurons, not the
// leg motor circuit. DNg11 activity accumulates into a request; ipsilateral steering activity
// scales the lift. The small flexion-and-return pattern is ours, not a reconstructed gait.
// No random timer, root translation, force on the thorax, or change to the brain kernel.
type ShuffleLeg = { name: string, role: string, joints: { ai: number, offset: number }[], amount: number,
                    swing: { ai: number, amp: number },    // fore-aft joint for the walking stride (measured per segment)
                    raise: number };                       // coxa extension actuator: lifts the whole leg (grooming)
let shuffleLegs: ShuffleLeg[] = [];
const shuffle = { enabled:true, active:-1, next:0, phase:0, charge:0,
                  cooldown:0.65, duration:0.5, strength:0, count:0 };

function buildShuffleMap() {
  // Alternate sides and leg pairs, leaving five legs at their standing targets.
  const order = ['T1_left', 'T3_right', 'T2_left', 'T1_right', 'T3_left', 'T2_right'];
  shuffleLegs = order.map(name => {
    const offsets = name.startsWith('T1') ? [-0.30, -0.23]
                  : name.startsWith('T2') ? [-0.22, -0.20] : [-0.27, -0.23];
    const joints = ['femur', 'tibia'].map((joint, i) => {
      const ai = mujoco.mj_name2id(model, 19, `${joint}_${name}`);
      if (ai < 0) throw new Error(`missing shuffle actuator ${joint}_${name}`);
      return { ai, offset:offsets[i] };
    });
    // Fore-aft swing, measured on a lifted foot: coxa twist moves the front and middle feet
    // forward (+), femur twist moves the hind feet forward with the opposite sign.
    const swingJoint = name.startsWith('T3') ? `femur_twist_${name}` : `coxa_twist_${name}`;
    const sai = mujoco.mj_name2id(model, 19, swingJoint);
    if (sai < 0) throw new Error(`missing swing actuator ${swingJoint}`);
    // Linear stance sweep at joint rate 4·A·stepHz (rad/s) must move the foot at WALK.speed:
    // A = speed / (4 · cmPerRad · stepHz). Sign: femur twist moves the hind foot the other way.
    const seg = name.slice(0, 2) as 'T1' | 'T2' | 'T3';
    const amp = (WALK.speed / (4 * WALK.cmPerRad[seg] * WALK.stepHz)) * (seg === 'T3' ? -1 : 1);
    const raise = mujoco.mj_name2id(model, 19, `coxa_${name}`);
    if (raise < 0) throw new Error(`missing coxa actuator coxa_${name}`);
    return { name, role:name.endsWith('left') ? 'dn_steer_l' : 'dn_steer_r', joints, amount:0, swing: { ai: sai, amp }, raise };
  });
  resetShuffle();
}

function resetShuffle() {
  Object.assign(shuffle, { active:-1, next:0, phase:0, charge:0,
                           cooldown:0.65, strength:0, count:0 });
  for (const leg of shuffleLegs) {
    leg.amount = 0;
    if (leg.swing) data.ctrl[leg.swing.ai] = holdCtrl[leg.swing.ai];
    if (leg.raise !== undefined) data.ctrl[leg.raise] = holdCtrl[leg.raise];
  }
}

function stepShuffle(b: Brain, d: MjData, dt: number) {
  const enabled = neural && shuffle.enabled;
  if (!enabled) {
    shuffle.active = -1; shuffle.charge = 0; shuffle.cooldown = 0.2;
  } else if (shuffle.active >= 0) {
    shuffle.phase = Math.min(1, shuffle.phase + dt / shuffle.duration);
    if (shuffle.phase >= 1) { shuffle.active = -1; shuffle.cooldown = 0.2; }
  } else if (shuffle.cooldown > 0) {
    shuffle.cooldown = Math.max(0, shuffle.cooldown - dt);
  } else {
    const groom = Math.max(0, b.rate.dn_groom || 0);
    // Eight population-equivalent spikes per request; silent neurons cannot trigger a shuffle.
    shuffle.charge += Math.min(100, groom) * dt;
    if (shuffle.charge >= 8 && shuffleLegs.length) {
      shuffle.active = shuffle.next;
      shuffle.next = (shuffle.next + 1) % shuffleLegs.length;
      shuffle.charge = 0; shuffle.phase = 0; shuffle.count++;
      const steer = Math.max(0, b.rate[shuffleLegs[shuffle.active].role] || 0);
      shuffle.strength = 0.65 + 0.35 * Math.min(1, steer / 100);
      shuffle.duration = 0.42 + 0.14 / (1 + groom / 30);
    }
  }
  // A single smooth lift-and-return pulse, not a free-running idle oscillator. The short
  // filter also lowers a lifted foot gently when neural drive or shuffle is switched off.
  const blend = 1 - Math.exp(-dt / 0.025);
  for (let i = 0; i < shuffleLegs.length; i++) {
    const leg = shuffleLegs[i];
    const target = i === shuffle.active ? shuffle.strength * Math.sin(Math.PI * shuffle.phase) ** 2 : 0;
    leg.amount += (target - leg.amount) * blend;
    for (const { ai, offset } of leg.joints) {
      d.ctrl[ai] = Math.max(model.actuator_ctrlrange[2 * ai],
        Math.min(model.actuator_ctrlrange[2 * ai + 1], holdCtrl[ai] + offset * leg.amount));
    }
  }
}

// ------------------------------------------------------------- grooming
// A command mapping, like the shuffle: DNg11 (dn_groom) is the identified antennal-grooming
// descending neuron, and it rises under touch (22.9 -> 33.8 Hz measured). When its 200 ms mean
// exceeds GROOM.gain times its calibrated rest while the fly stands, the front legs are lifted
// and rubbed forward over the head for GROOM.duration seconds (supplied motion, ours). No
// change to the brain, no timer: a quiet DNg11 never grooms.
const GROOM = { gain: 1.35, duration: 0.9, hz: 4, cooldown: 1.5 };
const groom = { active: false, t: 0, ema: 0, cooldown: 0, count: 0 };
function stepGroom(b: BrainLike, d: MjData, dt: number) {
  groom.ema += ((b.rate.dn_groom || 0) - groom.ema) * (dt / 0.2);
  const rest = b.rest && b.rest.dn_groom || 0;
  if (!neural) {   // neural drive off stops every supplied pattern, the shuffle included
    if (groom.active) { groom.active = false; groom.cooldown = 0; resetShuffle(); }
    return;
  }
  if (groom.active) {
    groom.t += dt;
    const u = smoothstep(Math.min(groom.t / 0.15, 1)) * smoothstep(Math.min((GROOM.duration - groom.t) / 0.15, 1));
    const rub = Math.sin(2 * Math.PI * GROOM.hz * groom.t);
    // The foot is raised with the coxa (the femur can lift only 0.15 rad and a rub with the
    // foot still on the floor scraped it) and flexed at the tibia, then rubbed fore-aft.
    for (const leg of shuffleLegs) {
      if (!leg.name.startsWith('T1')) continue;
      for (const { ai, offset } of leg.joints) d.ctrl[ai] = ctrlClamp(ai, holdCtrl[ai] + offset * 1.2 * u);
      d.ctrl[leg.raise] = ctrlClamp(leg.raise, holdCtrl[leg.raise] + 0.6 * u);
      d.ctrl[leg.swing.ai] = ctrlClamp(leg.swing.ai, holdCtrl[leg.swing.ai] + leg.swing.amp * u * (0.5 + 0.3 * rub));
    }
    if (groom.t >= GROOM.duration) { groom.active = false; groom.cooldown = GROOM.cooldown; resetShuffle(); }
    return;
  }
  if (groom.cooldown > 0) { groom.cooldown -= dt; return; }
  if (neural && rest > 0 && groom.ema > GROOM.gain * rest && flight.state === 'ground') {
    groom.active = true; groom.t = 0; groom.count++;
  }
}

// ------------------------------------------------------------- stimulus levels
// Each sensory channel has three sources, each with a left and a right level: the sustained
// switch (a button, both sides), a transient pulse (a poke, the loom event, the wall) and the
// world (what the terrarium supplies: sugar, odour, daylight, moving objects). The brain
// receives the per-side maximum; brain.setStimLR is only ever called from here, so the
// sources cannot clobber each other and the UI can show each of them.
type LR = [number, number];
const stimSwitch: Record<string, number> = {};
const stimPulse: Record<string, LR> = {};
const stimWorld: Record<string, LR> = {};
function applyStim(k: string) {
  const sw = stimSwitch[k] || 0, p = stimPulse[k], w = stimWorld[k];
  const l = Math.max(sw, p ? p[0] : 0, w ? w[0] : 0), r = Math.max(sw, p ? p[1] : 0, w ? w[1] : 0);
  const cur = brain.stimLR[k];
  if (!cur || cur[0] !== l || cur[1] !== r) brain.setStimLR(k, l, r);
}
function setPulse(k: string, l: number, r: number) {
  const p = stimPulse[k];
  if (p && p[0] === l && p[1] === r) return;
  stimPulse[k] = [l, r]; applyStim(k);
}
function setWorld(k: string, l: number, r: number) {
  const w = stimWorld[k];
  if (w && w[0] === l && w[1] === r) return;
  stimWorld[k] = [l, r]; applyStim(k);
}
const clamp = (v: number, lo: number, hi: number) => v < lo ? lo : v > hi ? hi : v;
const smoothstep = (u: number) => { u = clamp(u, 0, 1); return u * u * (3 - 2 * u); };
const wrapAngle = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));
// A level split by the bearing of its source (radians from the heading, +left): straight
// ahead or behind drives both sides, a source on one flank drives that side alone.
function lateral(level: number, bearing: number): LR {
  const s = Math.sin(bearing);
  return [level * Math.min(1, 1 + s), level * Math.min(1, 1 - s)];
}
function yawOf(q: ArrayLike<number>) {   // heading of the body +x axis from a [w,x,y,z] quaternion
  const w = q[0], x = q[1], y = q[2], z = q[3];
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}
function bearingTo(d: MjData, x: number, y: number) {
  return wrapAngle(Math.atan2(y - d.qpos[1], x - d.qpos[0]) - yawOf(d.qpos.subarray(3, 7)));
}

// ------------------------------------------------------------- poke (direct touch)
// A touch on the fly's body is a transient mechanosensory stimulus: the same 2,674-neuron
// `mechano` population the Touch switch drives, held briefly and then released exponentially.
// Stepped in brain time (1 ms per call from stepSimulation), not wall time, so a device running
// below real time still delivers the same pulse to the same neurons. The population is split by
// side, so a touch on the left flank drives mostly the left cells; within a side FAFB cannot say
// WHERE, so region only scales the hold — the antennae carry the fly's mechanosensory organ
// (Johnston's organ), so a hit on the head or antennae holds longer than one on the abdomen.
const POKE = { hold:0.15, holdHead:0.30, release:0.5, offSide:0.35 };   // seconds of brain time
const poke = { level:0, hold:0, count:0, last:'', lr:[1, 1] as LR };
let headBody = -1;
function bodyIsHead(body: number) {
  for (let b = body; b > 0; b = model.body_parentid[b]) if (b === headBody) return true;
  return false;
}
// side: +1 left flank, -1 right flank, 0 midline.
function pokeTouch(hold: number, side: number, what: string) {
  poke.level = 1; poke.hold = hold; poke.count++; poke.last = what;
  poke.lr = side > 0.3 ? [1, POKE.offSide] : side < -0.3 ? [POKE.offSide, 1] : [1, 1];
  setPulse('touch', poke.lr[0], poke.lr[1]);
}
function pokeBody(body: number, side = 0) {
  if (headBody < 0) headBody = mujoco.mj_name2id(model, 1 /* mjOBJ_BODY */, 'head');
  const name = mujoco.mj_id2name(model, 1, body);
  if (side === 0) side = name.endsWith('_left') ? 1 : name.endsWith('_right') ? -1 : 0;
  pokeTouch(bodyIsHead(body) ? POKE.holdHead : POKE.hold, side, name);
}
function stepPoke(dt: number) {
  if (poke.level <= 0) return;
  if (poke.hold > 0) poke.hold = Math.max(0, poke.hold - dt);
  else {
    poke.level *= Math.exp(-dt / POKE.release);
    if (poke.level < 0.01) poke.level = 0;
  }
  setPulse('touch', poke.level * poke.lr[0], poke.level * poke.lr[1]);
}

// ------------------------------------------------------------- looming event
// A dark sphere of radius LOOM.r approaches the fly from distance d0 to dmin at constant speed
// over LOOM.approach seconds, hangs at closest approach for LOOM.hold, then passes. The looming
// stimulus (LC4 + LPLC2, the fly's own looming detectors) follows its angular size:
// level = (r/d / r/dmin)^1.5, rising slowly and then sharply the way an object on a collision
// course expands (a squared ramp peaked for only ~0.1 s and the escape DNs never reached the
// rate a sustained loom gives them; measured, not tuned by eye). After the pass the level
// releases exponentially, and it lands on the eye facing the sphere. Stepped in brain time from
// stepSimulation, like the poke; the sphere is drawn from this state in main.
const LOOM = { r:0.12, d0:1.9, dmin:0.32, approach:1.1, hold:0.25, release:0.08, fade:0.35, exp:1.5,
               dir:[0.85, -0.10, 0.50] };   // world direction from the fly, normalised below
{ const n = Math.hypot(...LOOM.dir); LOOM.dir = LOOM.dir.map(v => v / n); }
const loom = { active:false, t:0, d:LOOM.d0, level:0, count:0 };
function startLoom() {
  loom.active = true; loom.t = 0; loom.d = LOOM.d0; loom.count++;
}
function stepLoom(d: MjData, dt: number) {
  if (!loom.active) { if (loom.level !== 0) { loom.level = 0; applyLooming(d); } return; }
  loom.t += dt;
  const passT = LOOM.approach + LOOM.hold;
  if (loom.t <= LOOM.approach) {
    loom.d = LOOM.d0 + (LOOM.dmin - LOOM.d0) * (loom.t / LOOM.approach);
    loom.level = Math.min(1, Math.pow((LOOM.r / loom.d) / (LOOM.r / LOOM.dmin), LOOM.exp));
  } else if (loom.t <= passT) {
    loom.d = LOOM.dmin; loom.level = 1;
  } else {
    loom.level *= Math.exp(-dt / LOOM.release);
    if (loom.t > passT + LOOM.fade) { loom.active = false; loom.level = 0; loom.d = LOOM.d0; }
  }
  applyLooming(d);
}
// The loom event and flight's wall proximity share the looming pulse; the brain gets the max
// per side. (Moving objects in the world use the world source instead; see stepWorld.)
function applyLooming(d: MjData) {
  const ev = lateral(loom.level, bearingTo(d, d.qpos[0] + LOOM.dir[0], d.qpos[1] + LOOM.dir[1]));
  const wall = flight.wallLR;
  setPulse('looming', Math.max(ev[0], wall[0]), Math.max(ev[1], wall[1]));
}

// ------------------------------------------------------------- world
// The terrarium as the source of the senses. Nothing here is a claim about the fly's brain: it
// is the environment, made explicit, feeding the same populations the switches drive.
//
//   sugar   — a sugar sack on the floor. Sweet GRNs are driven when the mouthparts are over the
//             spilled sugar around it (labellar) or the feet stand in it (tarsal); odour (ORNs)
//             falls off with distance from the head and lands on the antenna facing it.
//   daylight — a slow day/night cycle. Visual neurons follow the brightness (capped at
//             WORLD.lightMax: 11,426 cells driven flat out triple the whole-brain rate, and real
//             photoreceptors adapt to steady light, which this model cannot). The hot cells warm
//             at noon, the cold and cooling cells and the hygrosensory cells respond at night.
//             Anything between the fly and the sun casts a shadow: a dip in the visual drive.
//   objects — the beach ball and the loom sphere are looming stimuli by their real approach
//             geometry (time to collision from their position and velocity), on the eye they
//             approach; the ball hitting the fly is a touch on that flank.
//
// Levels are brain-tick state, so a slow device still feeds the same world to the same brain.
const WORLD = {
  dayPeriod: 120,      // seconds per day; the simulation starts at noon
  lightMax: 0.35,      // visual level in full daylight (see above)
  odourRange: 0.8,     // cm beyond the sugar at which the odour reaches zero
  // A draught across the terrarium carries the odour downwind of the sack in a plume: full
  // strength inside a cone that starts plumeWidth wide at the sack and widens by plumeSpread
  // per cm downwind, and a fraction (upwind) of the still-air level elsewhere. The draught
  // blows from the sack toward the perch, so a fly at the start is downwind of the sugar.
  draught: [-0.97, -0.24] as [number, number],
  plumeWidth: 0.12, plumeSpread: 0.35, upwind: 0.25,
  antennaSpan: 0.035,  // each antenna samples the odour field this far to its side of the head centre
  sugarEmpty: 180,     // seconds of feeding at full contact that empty the sack
  sugarRefill: 600,    // seconds for an empty sack to fill again (it is refilled, slowly)
  contrastGain: 0.05,  // visual level per unit brightness change per second (a passing shadow is a transient)
  tasteReach: 0.03,    // the labellum tastes within this of the sack's surface; the feet within footReach of its base
  footReach: 0.04,
  loomRange: 1.2, loomTau: 0.6,   // objects closer than loomRange on a collision course within loomTau seconds loom
  objectRange: 1.5, objectRate: 2.0,   // a small object crossing the view within objectRange at objectRate rad/s drives LC11 fully
  floorZ: -0.132,      // the physics floor; the terrarium's visual floor undulates just above it
  floorBand: 0.23,     // ground hits up to this far above floorZ count as walkable floor (the floor undulates to +0.10)
};
type Loomer = { x: number, y: number, z: number, vx: number, vy: number, vz: number, r: number, name: string };
type GroundMap = { x0: number, y0: number, cell: number, n: number, ok: Uint8Array, plant: Uint8Array };   // plant: a cell occupied by foliage or flowers
const world = {
  enabled: true, t: 0, day: 1, shade: 0, sugarDist: 0,
  bright: 1, contrast: 0,                                // brightness at the fly and its filtered rate of change
  sugar: { x: 0.34, y: 0.08, r: 0.10, placed: false, amount: 1 },   // refined from the sack's real footprint once it loads; amount 0..1
  sun: [0, 0.69, 0.72],                                  // toward the key light
  loomers: [] as Loomer[],                               // written by main (ball) each frame
  touchHits: [] as number[],                             // bearings of contacts, queued by main
  ground: null as GroundMap | null,                      // walkable floor cells, sampled from the terrarium mesh
  levels: { sweet:0, sweetLeg:0, bitter:0, odour:0, light:0, heat:0, cool:0, damp:0, looming:0, object:0, touch:0 },
  headBody: -1, labrumBodies: [] as number[], clawBodies: [] as number[],
};
function buildWorldMap() {
  world.headBody = mujoco.mj_name2id(model, 1 /* mjOBJ_BODY */, 'head');
  world.labrumBodies = ['labrum_left', 'labrum_right'].map(n => mujoco.mj_name2id(model, 1, n)).filter(b => b >= 0);
  world.clawBodies = [];
  for (const leg of ['T1_left', 'T1_right', 'T2_left', 'T2_right', 'T3_left', 'T3_right']) {
    const b = mujoco.mj_name2id(model, 1, `claw_${leg}`);
    if (b >= 0) world.clawBodies.push(b);
  }
}
// Walkable floor: the sampled ground map when the terrarium has loaded, else a disc round the perch.
function walkable(x: number, y: number) {
  const g = world.ground;
  if (!g) return Math.hypot(x, y) < 0.6;
  const i = Math.floor((x - g.x0) / g.cell), j = Math.floor((y - g.y0) / g.cell);
  if (i < 0 || j < 0 || i >= g.n || j >= g.n) return false;
  return g.ok[j * g.n + i] === 1;
}
function plantAt(x: number, y: number) {
  const g = world.ground;
  if (!g) return false;
  const i = Math.floor((x - g.x0) / g.cell), j = Math.floor((y - g.y0) / g.cell);
  if (i < 0 || j < 0 || i >= g.n || j >= g.n) return false;
  return g.plant[j * g.n + i] === 1;
}
function nearestWalkable(x: number, y: number): [number, number] {
  const g = world.ground;
  if (!g) { const r = Math.hypot(x, y) || 1; return r > 0.5 ? [x / r * 0.5, y / r * 0.5] : [x, y]; }   // the fallback disc, with a margin from its edge
  if (walkable(x, y)) return [x, y];
  let best: [number, number] = [0, 0], bd = Infinity;
  for (let j = 0; j < g.n; j++) for (let i = 0; i < g.n; i++) {
    if (g.ok[j * g.n + i] !== 1) continue;
    const cx = g.x0 + (i + 0.5) * g.cell, cy = g.y0 + (j + 0.5) * g.cell;
    const dd = (cx - x) * (cx - x) + (cy - y) * (cy - y);
    if (dd < bd) { bd = dd; best = [cx, cy]; }
  }
  return best;
}
// The odour field at a point: the still-air fall-off with distance from the sack, shaped by
// the draught into a downwind plume, scaled by how much sugar is left.
function odourAt(x: number, y: number) {
  const S = world.sugar, w = WORLD.draught;
  const vx = x - S.x, vy = y - S.y, dist = Math.hypot(vx, vy);
  const along = vx * w[0] + vy * w[1], perp = Math.abs(vx * w[1] - vy * w[0]);
  const radial = clamp(1 - (dist - S.r) / WORLD.odourRange, 0, 1);
  const plume = along >= 0 ? clamp(1 - (perp - S.r) / (WORLD.plumeWidth + WORLD.plumeSpread * along), 0, 1) : 0;
  return 0.8 * S.amount * radial * Math.max(plume, WORLD.upwind);
}
function stepWorld(d: MjData, dt: number) {
  if (!world.enabled) {
    for (const k of Object.keys(world.levels)) { (world.levels as Record<string, number>)[k] = 0; setWorld(k, 0, 0); }
    world.touchHits.length = 0;
    return;
  }
  world.t += dt;
  const L = world.levels;
  // --- daylight, temperature, humidity
  world.day = 0.5 + 0.5 * Math.cos(2 * Math.PI * world.t / WORLD.dayPeriod);
  let shade = 0;
  const fx = d.qpos[0], fy = d.qpos[1], fz = d.qpos[2];
  for (const o of world.loomers) {
    const ox = o.x - fx, oy = o.y - fy, oz = o.z - fz;
    const along = ox * world.sun[0] + oy * world.sun[1] + oz * world.sun[2];
    if (along <= 0) continue;
    const perp = Math.sqrt(Math.max(0, ox * ox + oy * oy + oz * oz - along * along));
    shade = Math.max(shade, clamp(1 - (perp - o.r) / (0.5 * o.r + 1e-6), 0, 1));
  }
  world.shade = shade;
  // Visual cells respond to change as much as to level: the capped ambient term plus a
  // transient from the rate of change of brightness (a shadow's edge passing over the fly, the
  // day's own slow ramp is negligible). The transient is a 50 ms filtered |d brightness / dt|.
  const bright = world.day * (1 - shade);
  world.contrast += (Math.abs(bright - world.bright) / dt - world.contrast) * Math.min(1, dt / 0.05);
  world.bright = bright;
  L.light = Math.min(1, WORLD.lightMax * bright + WORLD.contrastGain * world.contrast);
  L.heat = 0.5 * clamp((world.day - 0.65) / 0.35, 0, 1);
  L.cool = 0.5 * clamp((0.35 - world.day) / 0.35, 0, 1);
  L.damp = 0.3 * clamp((0.35 - world.day) / 0.35, 0, 1);
  setWorld('light', L.light, L.light); setWorld('heat', L.heat, L.heat);
  setWorld('cool', L.cool, L.cool); setWorld('damp', L.damp, L.damp);
  // --- sugar: the labellum tastes by touching the sack's surface (it hangs 0.05 under the head
  //     and extends only 0.02 further down, so it never reaches the floor from a standing body);
  //     the feet taste the sugar at the sack's base; odour reaches the antennae from a distance.
  const S = world.sugar;
  const hb = world.headBody;
  const hx = hb >= 0 ? d.xpos[hb * 3] : fx, hy = hb >= 0 ? d.xpos[hb * 3 + 1] : fy;
  const dh = Math.hypot(hx - S.x, hy - S.y);
  world.sugarDist = dh;
  let tipX = hx, tipY = hy;
  if (world.labrumBodies.length) {
    tipX = 0; tipY = 0;
    for (const b of world.labrumBodies) { tipX += d.xpos[b * 3] / world.labrumBodies.length; tipY += d.xpos[b * 3 + 1] / world.labrumBodies.length; }
  }
  const labellar = clamp(1 - (Math.hypot(tipX - S.x, tipY - S.y) - S.r) / WORLD.tasteReach, 0, 1);
  // The plants taste bitter: the labellum against foliage or a flower drives the bitter GRNs
  // (labellar only; FAFB has no leg bitter pool). Measured: proboscis -55%, giant fiber +25%.
  L.bitter = plantAt(tipX, tipY) ? 1 : 0;
  setWorld('bitter', L.bitter, L.bitter);
  let feet = 0;
  for (const b of world.clawBodies) if (Math.hypot(d.xpos[b * 3] - S.x, d.xpos[b * 3 + 1] - S.y) < S.r + WORLD.footReach) feet++;
  const tarsal = world.clawBodies.length ? feet / world.clawBodies.length : 0;
  // The sugar is finite: feeding (labellar contact) empties the sack over WORLD.sugarEmpty
  // seconds and it refills slowly while nobody feeds, so the day has a story — find, feed,
  // wander, find again. Taste and smell scale with what is left. A world property, not a
  // brain one: the feeding counter simply stops when there is nothing to taste.
  S.amount = clamp(S.amount + (labellar > 0 ? -labellar / WORLD.sugarEmpty : 1 / WORLD.sugarRefill) * dt, 0, 1);
  L.sweet = labellar * S.amount; L.sweetLeg = tarsal * S.amount;
  setWorld('sweet', L.sweet, L.sweet); setWorld('sweetLeg', L.sweetLeg, L.sweetLeg);
  // Odour: each antenna samples the plume at its own position (the antenna nearer the plume's
  // axis smells more), and the antenna facing the sack gets the larger share of it, as before.
  const yaw = yawOf(d.qpos.subarray(3, 7)), sx = -Math.sin(yaw) * WORLD.antennaSpan, sy = Math.cos(yaw) * WORLD.antennaSpan;
  const antL = odourAt(hx + sx, hy + sy), antR = odourAt(hx - sx, hy - sy);
  const facing = lateral(1, bearingTo(d, S.x, S.y));
  L.odour = odourAt(hx, hy);
  setWorld('odour', facing[0] * antL, facing[1] * antR);
  // --- moving objects: looming by time to collision, on the eye they approach; and a small
  //     object crossing the view (angular velocity, not approach) for the LC11 detectors
  let lo = 0, ro = 0, ol = 0, or_ = 0; L.looming = 0; L.object = 0;
  for (const o of world.loomers) {
    const rx = o.x - fx, ry = o.y - fy, rz = o.z - fz, dist = Math.hypot(rx, ry, rz);
    if (dist < 1e-6) continue;
    const speed = Math.hypot(o.vx, o.vy, o.vz);
    if (speed > 0.02 && dist <= WORLD.objectRange) {
      const vApp = -(rx * o.vx + ry * o.vy + rz * o.vz) / dist;
      const vTan = Math.sqrt(Math.max(0, speed * speed - vApp * vApp));
      const level = clamp((vTan / dist) / WORLD.objectRate, 0, 1) * clamp(1 - Math.atan2(o.r, dist) / 0.5, 0, 1);   // small in the view
      if (level > 0) { const lr = lateral(level, bearingTo(d, o.x, o.y)); ol = Math.max(ol, lr[0]); or_ = Math.max(or_, lr[1]); L.object = Math.max(L.object, level); }
    }
    if (dist > WORLD.loomRange) continue;
    const vApp = -(rx * o.vx + ry * o.vy + rz * o.vz) / dist;
    if (vApp <= 0.02) continue;
    const tau = dist / vApp;
    const level = clamp(1 - tau / WORLD.loomTau, 0, 1) * clamp(o.r / (0.4 * dist), 0, 1);
    if (level <= 0) continue;
    const lr = lateral(level, bearingTo(d, o.x, o.y));
    lo = Math.max(lo, lr[0]); ro = Math.max(ro, lr[1]); L.looming = Math.max(L.looming, level);
  }
  setWorld('looming', lo, ro); setWorld('object', ol, or_);
  // --- contacts queued by the ball physics become touches on that flank
  if (world.touchHits.length) {
    const bearing = world.touchHits[world.touchHits.length - 1];
    world.touchHits.length = 0;
    pokeTouch(POKE.hold, Math.sin(bearing), 'ball');
    L.touch = 1;
  } else L.touch = poke.last === 'ball' ? poke.level : 0;
}

// ------------------------------------------------------------- locomotion
// COMMAND MAPPINGS, the weakest tier in this file, and deliberately the most visible ones.
// FAFB is brain-only: the leg and wing motor neurons and the VNC pattern generators are not in
// the dataset. What the brain contributes is real and measured; the gait and the wingbeat are
// supplied and are NOT claims about the fly.
//
//   walking  — DNp09 (dn_walk, the forward-walking DN, 2 cells) and MDN (dn_back, backward
//              walking, 4 cells). Both fire at ~1 Hz or less here; each spike requests a short
//              bout of steps, forward or backward, so the fly ambles in bursts on its own.
//   takeoff  — DNp02/04/11 (dn_escwing), silent at rest, ~200 Hz under looming. A 100 ms mean
//              of the stronger side above FLIGHT.takeoffRate is the trigger.
//   turning  — DNa01/DNa02 (dn_steer_l/r), documented as asymmetric modulation of left/right
//              wing amplitude. Yaw rate follows the left-right difference, each side normalised
//              to its own calibrated resting rate, so the standing asymmetry (L ~70 Hz, R ~53 Hz)
//              reads as straight and a lateralised input reads as a turn.
//   landing  — the escape DNs return to 0 Hz when the threat passes; FLIGHT.quietSec of calm
//              after FLIGHT.minFlightSec in the air ends the flight, wherever the fly is.
//
// Root motion is kinematic: the free joint's qpos/qvel are written every physics step, so the
// body is carried rather than lifted by any force; legs, head, proboscis, antennae and abdomen
// keep simulating under their own neural drive throughout. This relaxes the ground rule that
// nothing translates the root (see the shuffle note above) while the fly is walking or flying;
// on stopping, the root is handed back to physics at the standing pose over loaded legs.
type FlightState = 'ground' | 'walk' | 'takeoff' | 'flight' | 'landing' | 'settle' | 'touchdown';
const FLIGHT = {
  takeoffRate: 100,   // Hz, 100 ms mean of the stronger of dn_escwing L/R
  quietRate: 30,      // Hz, below which the escape DNs count as calm (0 at rest, ~200 escaping)
  quietSec: 4, minFlightSec: 3,
  speed: 0.5,         // cm/s, ~1.5 body lengths per second (stylized; the terrarium is small)
  cruiseZ: 0.45, bobAmp: 0.05, bobKick: 12,   // altitude wander: clamp (cm) and kick strength (cm/s²)
  yawRate: 2.4,       // rad/s at full steering asymmetry
  steerBand: 0.6,     // normalised L-R difference (L/rest_L - R/rest_R) that counts as full asymmetry
  bank: 0.35, pitch: 0.15,
  flapHz: 24, takeoffSec: 0.45, pushSec: 0.06, settleSec: 0.6, approachRadius: 0.12,
  touchdownSec: 0.4, touchdownMax: 0.8,      // leg extension ramp onto the floor; hard stop
  touchdownTuck: 0.12,                       // leg flexion that keeps the feet clear at the standing height
  wall: 0.3,          // soft avoidance band inside the bounds
  wallLoom: 0.2,      // looming level at the wall; measured escape response ~15 Hz, below quietRate,
                      // so the glass makes the fly turn but cannot keep it airborne forever
};
const WALK = {
  speed: 0.22,        // cm/s forward (backward at 0.6x)
  yawRate: 1.5,       // rad/s at full steering asymmetry
  boutPerSpike: 0.6,  // seconds of walking each DNp09 spike requests
  backGain: 2.5,      // MDN is tonic here (~17 Hz at rest); backward bouts while its 100 ms mean exceeds this multiple of rest
  boutMax: 2.0,
  stepHz: 2.2, lift: 0.006,   // tripod stepping cycle and the body lift while the feet swing
  // Fore-aft foot travel per radian of the stride joint (cm/rad): coxa twist for T1 and T2,
  // femur twist for T3. Calibrated with the foot planted (the servo lags the target a little
  // under load) so that a stance foot sweeps back at exactly the body's speed: measured
  // along-heading slip per stance is under 0.003 cm for every leg. The stride amplitude per
  // leg is derived from these rather than every leg sharing one amplitude. What remains is
  // the arc of a single twist joint (a sideways component of 0.01-0.06 cm per stance), which
  // one joint per leg cannot straighten.
  cmPerRad: { T1: 0.076, T2: 0.148, T3: 0.077 },
  stopSec: 0.15,      // a walk ends with a short settle, not the landing crouch
};
const flight = {
  enabled: true, state: 'ground' as FlightState, t: 0, escape: 0, quiet: 0, air: 0, count: 0,
  x: 0, y: 0, z: 0, yaw: 0, vx: 0, vy: 0, vz: 0, roll: 0, pitch: 0, yawRate: 0,
  asym: 0, tuck: 0, flap: 0, fold: 0, phase: 0, wallLR: [0, 0] as LR,   // fold: 0 stroke centre .. 1 folded rest
  home: { x: 0, y: 0, z: 0, yaw: 0 },   // where the current locomotion ends: landing target / stopping pose
  standZ: 0,                            // standing height of the root, captured when leaving the ground
  settleFrom: 0,                        // height at which the settle descent began
  wingFrom: [0, 0, 0, 0, 0, 0],        // wing joint angles at takeoff, blended into the stroke
  weight: 0,                            // whole-body weight, for the touchdown handover
  // Refined from the terrarium's real glass bounds once it loads; these defaults sit inside it.
  bounds: { cx: -0.5, cy: -0.15, r: 1.25, zmin: 0.2, zmax: 0.95 },
  walk: { bout: 0, dir: 1, phase: 0, count: 0, blocked: 0, back: 0, turning: false, turnSign: 1, settle: 0 },   // back: 100 ms mean of dn_back; settle: stride blend-out after a bout
  bob: 0, bobV: 0, seed: 7,   // altitude random walk (seeded, so a run repeats)
};
type WingJoint = { qadr: number, dadr: number, side: 'l' | 'r', axis: 'yaw' | 'roll' | 'pitch' };
// Stroke centre in flight, and the folded pose the wings are returned to before the root is
// released on landing (measured resting angles under neural control; released from the flight
// pose, the wing tips sat on the floor and against the hind legs and the weak springs could
// not fold them back).
const WING_STROKE = { yaw: 0.35, roll: 1.00, pitch: -0.70 };
const WING_REST   = { yaw: 0.85, roll: 1.05, pitch: -0.90 };
let wingJoints: WingJoint[] = [];
function buildFlightMap() {
  const thorax = mujoco.mj_name2id(model, 1 /* mjOBJ_BODY */, 'thorax');
  flight.weight = model.body_subtreemass[thorax] * Math.abs(model.opt.gravity[2]);
  wingJoints = [];
  for (const side of ['left', 'right'] as const) for (const axis of ['yaw', 'roll', 'pitch'] as const) {
    const ji = mujoco.mj_name2id(model, 3 /* mjOBJ_JOINT */, `wing_${axis}_${side}`);
    if (ji < 0) throw new Error(`missing wing joint wing_${axis}_${side}`);
    wingJoints.push({ qadr: model.jnt_qposadr[ji], dadr: model.jnt_dofadr[ji], side: side[0] as 'l' | 'r', axis });
  }
}
// Quaternions as [w, x, y, z], MuJoCo's convention.
function qmul(a: number[], b: number[]) {
  return [a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
          a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
          a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
          a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]];
}
const qaxis = (x: number, y: number, z: number, a: number) => [Math.cos(a / 2), x * Math.sin(a / 2), y * Math.sin(a / 2), z * Math.sin(a / 2)];

function leaveGround(d: MjData) {
  flight.standZ = d.qpos[2];
  flight.home = { x: d.qpos[0], y: d.qpos[1], z: d.qpos[2], yaw: yawOf(d.qpos.subarray(3, 7)) };
  flight.x = d.qpos[0]; flight.y = d.qpos[1]; flight.z = d.qpos[2]; flight.yaw = flight.home.yaw;
  flight.vx = flight.vy = flight.vz = 0; flight.roll = flight.pitch = flight.yawRate = 0;
  flight.tuck = 0; flight.flap = 0; flight.fold = 0; flight.wallLR = [0, 0];
  flight.t = 0;
}
function startFlight(d: MjData) {
  if (flight.state === 'ground') leaveGround(d);
  else { flight.home.z = flight.standZ; flight.t = 0; }
  flight.wingFrom = wingJoints.map(w => d.qpos[w.qadr]);
  flight.tuck = 0; flight.flap = 0; flight.fold = 0; flight.walk.bout = 0; flight.walk.settle = 0;
  flight.bob = 0; flight.bobV = 0;   // the altitude wander restarts from cruise (its seed carries on)
  flight.state = 'takeoff'; flight.air = 0; flight.quiet = 0; flight.count++;
}
function startWalk(d: MjData) {
  leaveGround(d);
  flight.state = 'walk'; flight.walk.phase = 0; flight.walk.settle = 0; flight.walk.count++; flight.walk.turning = false;
  flight.walk.turnSign = flight.asym >= 0 ? 1 : -1;   // which way to turn if both sides are blocked
}
function endLocomotion(d: MjData) {
  // Release the root to physics where touchdown left it: level, legs loaded.
  const q = qaxis(0, 0, 1, flight.home.yaw);
  for (let i = 0; i < 4; i++) d.qpos[3 + i] = q[i];
  for (let i = 0; i < 6; i++) d.qvel[i] = 0;
  flight.state = 'ground'; flight.wallLR = [0, 0]; applyLooming(d);
  resetShuffle();
}
// Turning signal: each steering side relative to its own calibrated rest, clamped to +-1.
function steering(b: BrainLike) {
  if (!neural) return 0;
  const l = b.rate.dn_steer_l || 0, r = b.rate.dn_steer_r || 0;
  const rl = b.rest && b.rest.dn_steer_l || 0, rr = b.rest && b.rest.dn_steer_r || 0;
  const diff = rl > 0 && rr > 0 ? l / rl - r / rr : (l - r) / 60;
  return clamp(diff / FLIGHT.steerBand, -1, 1);
}

// Brain-tick step (1 ms): the locomotion state machine, steering, and the path.
function stepFlight(b: BrainLike, d: MjData, dt: number) {
  // The stronger side, not the mean: a loom on one flank drives that side's escape DN alone
  // (measured 227 / 0 Hz), and a one-sided escape command is still an escape command.
  const escape = Math.max(0, b.rate.dn_escwing_l || 0, b.rate.dn_escwing_r || 0);
  flight.escape += (escape - flight.escape) * (dt / 0.1);
  const W = flight.walk;
  W.back += ((b.rate.dn_back || 0) - W.back) * (dt / 0.1);
  const afoot = flight.state === 'ground' || flight.state === 'walk';
  if (flight.enabled && neural && b.spikesOf && afoot) {   // a flying fly requests no steps
    // DNp09 is near silent (~0.6 Hz): each spike requests a bout. MDN is tonic in this kernel
    // (~17 Hz at rest, 1 Hz in the NumPy reference), so it is read like the other resting
    // pools, as a rise over its own calibrated rest; a scripted brain without a rest rate is
    // read per spike.
    const fwd = b.spikesOf('dn_walk');
    const backRest = b.rest && b.rest.dn_back || 0;
    const back = backRest > 0 ? (W.back > WALK.backGain * backRest ? dt * 1.5 : 0) : b.spikesOf('dn_back') * WALK.boutPerSpike * 0.67;
    if (fwd > 0) { W.bout = Math.min(WALK.boutMax, W.bout + fwd * WALK.boutPerSpike); W.dir = 1; }
    else if (back > 0) { W.bout = Math.min(WALK.boutMax, W.bout + back); W.dir = -1; }
  }
  if (flight.state === 'ground') {
    if (flight.enabled && neural && flight.escape > FLIGHT.takeoffRate) startFlight(d);
    else if (flight.enabled && W.bout > 0) startWalk(d);
    return;
  }
  if (flight.state === 'walk' && flight.escape > FLIGHT.takeoffRate) startFlight(d);
  flight.t += dt;
  flight.asym += (steering(b) - flight.asym) * (dt / 0.15);
  // Measured (tools/response_matrix.py --lateral): a loom on the left drives the LEFT escape
  // wing DN alone (227 vs 0 Hz), so their asymmetry is a turn away from the threat.
  const escL = b.rate.dn_escwing_l || 0, escR = b.rate.dn_escwing_r || 0;
  const away = neural ? -clamp((escL - escR) / 200, -1, 1) : 0;
  let speed = FLIGHT.speed, zTarget = FLIGHT.cruiseZ, yawRate = (flight.asym + 0.8 * away) * FLIGHT.yawRate;
  const B = flight.bounds;

  if (flight.state === 'walk') {
    // Supplied gait on the walkable floor, steered by the brain; stops when the bout the
    // walking DNs requested runs out or the ground ahead is not walkable.
    W.bout -= dt;
    speed = WALK.speed * (W.dir > 0 ? 1 : -0.6); yawRate = flight.asym * WALK.yawRate;
    const nx = flight.x + speed * Math.cos(flight.yaw) * 0.05, ny = flight.y + speed * Math.sin(flight.yaw) * 0.05;
    if (!walkable(nx, ny)) {
      // An obstacle or the edge of the floor: the bout is spent turning toward open floor
      // (supplied), so the brain's next bout can go somewhere. Ending the bout instead left
      // the fly facing the same obstacle for good.
      if (!W.turning) { W.blocked++; W.turning = true; }
      speed = 0;
      const look = 0.08, a = flight.yaw + (W.dir > 0 ? 0 : Math.PI);
      const left = walkable(flight.x + look * Math.cos(a + 1.0), flight.y + look * Math.sin(a + 1.0));
      const right = walkable(flight.x + look * Math.cos(a - 1.0), flight.y + look * Math.sin(a - 1.0));
      yawRate += (left && !right ? 1 : right && !left ? -1 : W.turnSign) * WALK.yawRate;
    } else W.turning = false;
    zTarget = flight.standZ + WALK.lift; W.settle = 0;
    W.phase += 2 * Math.PI * WALK.stepHz * dt;
    if (W.bout <= 0) {
      // A walk ends where the feet are: the stride blends out, the body sinks its 0.006 lift,
      // and the root is handed back as soon as the legs carry the weight. No tuck, no crouch.
      flight.home = { x: flight.x, y: flight.y, z: flight.standZ, yaw: flight.yaw };
      flight.tuck = 0; W.settle = 1; flight.settleFrom = flight.z;
      flight.state = 'touchdown'; flight.t = 0;
    }
  } else if (flight.state === 'takeoff') {
    // A push-off first: the front and middle legs extend for FLIGHT.pushSec (the tuck offsets
    // in reverse) and lift the body a little before the wings start; then the stroke and the
    // climb. Supplied, like the rest of the takeoff.
    const push = flight.t < FLIGHT.pushSec;
    const u = smoothstep((flight.t - FLIGHT.pushSec) / (FLIGHT.takeoffSec - FLIGHT.pushSec));
    flight.tuck = push ? -0.5 * (flight.t / FLIGHT.pushSec) : u; flight.flap = push ? 0 : u;
    speed = FLIGHT.speed * u;
    zTarget = push ? flight.home.z + 0.012 * (flight.t / FLIGHT.pushSec) : flight.home.z + 0.012 + (FLIGHT.cruiseZ - flight.home.z - 0.012) * u;
    if (flight.t >= FLIGHT.takeoffSec) { flight.state = 'flight'; flight.air = 0; }
  } else if (flight.state === 'flight') {
    flight.air += dt;
    // Altitude wanders on a seeded random walk rather than a sine (no single peak to spot):
    // white kicks (cm/s²) filtered into a slow vertical drift, which is pulled back to the
    // cruise height over a couple of seconds. Deterministic for a given seed.
    flight.seed = (Math.imul(flight.seed, 1664525) + 1013904223) >>> 0;
    const kick = ((flight.seed / 4294967296) - 0.5) * 2 * FLIGHT.bobKick;
    flight.bobV = clamp(flight.bobV + (kick - flight.bobV / 0.4) * dt, -0.2, 0.2);
    flight.bob = clamp(flight.bob + (flight.bobV - flight.bob / 2) * dt, -FLIGHT.bobAmp * 1.6, FLIGHT.bobAmp * 1.6);
    zTarget = FLIGHT.cruiseZ + flight.bob;
    flight.quiet = flight.escape < FLIGHT.quietRate ? flight.quiet + dt : 0;
    if (flight.air > FLIGHT.minFlightSec && flight.quiet > FLIGHT.quietSec) {
      // Land here if the ground below is walkable, else at the nearest floor that is.
      const [tx, ty] = nearestWalkable(flight.x, flight.y);
      flight.home = { x: tx, y: ty, z: flight.standZ, yaw: flight.yaw };
      flight.state = 'landing'; flight.t = 0;
    }
  } else if (flight.state === 'landing') {
    // Supplied: fly over the landing spot and descend as it gets close.
    const dx = flight.home.x - flight.x, dy = flight.home.y - flight.y, dist = Math.hypot(dx, dy);
    const want = dist > 0.02 ? Math.atan2(dy, dx) : flight.yaw, err = wrapAngle(want - flight.yaw);
    yawRate = clamp(err / 0.25, -1.2, 1.2) * FLIGHT.yawRate;
    speed = FLIGHT.speed * clamp(0.3 + dist / 0.4, 0, 0.8);
    zTarget = flight.home.z + 0.08 + Math.min(1, dist / 0.6) * (FLIGHT.cruiseZ - flight.home.z - 0.08);
    if (dist < FLIGHT.approachRadius) { flight.state = 'settle'; flight.t = 0; flight.settleFrom = flight.z; flight.home.yaw = flight.yaw; }
  } else if (flight.state === 'settle') {
    // Sink from the approach height to the standing height. Order matters: the stroke stops
    // and the legs unfold in the first half, while the body is still high and the wings are
    // held raised at the stroke centre; the wings fold in the second half, over legs already
    // in their standing configuration. (Descending while still beating put the wing tips on
    // the floor; folding first let the extending hind legs sweep through the membranes and
    // push the wings forward, where the weak springs could not recover.)
    const u = smoothstep(flight.t / FLIGHT.settleSec);
    const u1 = clamp(u * 2, 0, 1), u2 = clamp(u * 2 - 1, 0, 1);
    flight.flap = 1 - u1; flight.tuck = 1 - (1 - FLIGHT.touchdownTuck) * u1; flight.fold = u2;
    speed = 0; yawRate = 0;
    const k = dt / 0.15;
    flight.x += (flight.home.x - flight.x) * k; flight.y += (flight.home.y - flight.y) * k;
    zTarget = flight.settleFrom + (flight.home.z - flight.settleFrom) * u;
    if (flight.t >= FLIGHT.settleSec) { flight.state = 'touchdown'; flight.t = 0; }
  } else if (flight.state === 'touchdown') {
    // Standing in reverse: the root stays pinned at the standing height while the legs extend
    // slowly onto the floor, and the root is handed back once the constraint force on its
    // vertical dof shows the legs carrying the body's weight — a static equilibrium, so
    // nothing is stored to launch it. (Unloaded servo legs stand taller than the loaded
    // stance: pinning with the legs at their targets drove the feet into the very stiff floor
    // and threw the body on release; releasing above the floor dropped it onto splayed feet,
    // a lower stance where the folded wing tips touched the floor.) After a walk the legs are
    // already down: the stride blends out and the body sinks its lift over WALK.stopSec.
    flight.x = flight.home.x; flight.y = flight.home.y; flight.yaw = flight.home.yaw;
    flight.flap = 0; speed = 0; yawRate = 0; flight.roll = flight.pitch = 0;
    if (W.settle > 0) {
      const u = smoothstep(flight.t / WALK.stopSec);
      W.settle = 1 - u;
      W.phase += 2 * Math.PI * WALK.stepHz * dt * (1 - u);
      zTarget = flight.settleFrom + (flight.home.z - flight.settleFrom) * u;
      const carried = flight.t > 0.05 && d.qfrc_constraint[2] >= flight.weight;
      if ((u >= 1 && carried) || flight.t >= FLIGHT.touchdownMax) { W.settle = 0; endLocomotion(d); return; }
    } else {
      flight.tuck = FLIGHT.touchdownTuck * Math.max(0, 1 - flight.t / FLIGHT.touchdownSec);
      zTarget = flight.home.z;
      const carried = flight.t > 0.1 && d.qfrc_constraint[2] >= flight.weight;
      if (carried || flight.t >= FLIGHT.touchdownMax) { endLocomotion(d); return; }
    }
  }

  // Soft wall avoidance (supplied) plus a looming pulse proportional to proximity, on the eye
  // facing the glass, so the brain's own escape circuit also sees the wall coming.
  const rx = flight.x - B.cx, ry = flight.y - B.cy, rc = Math.hypot(rx, ry);
  const prox = flight.state === 'walk' ? 0 : clamp((rc - (B.r - FLIGHT.wall)) / FLIGHT.wall, 0, 1);
  if (prox > 0 && flight.state !== 'settle' && flight.state !== 'touchdown') {
    const inward = Math.atan2(-ry, -rx), err = wrapAngle(inward - flight.yaw);
    if (Math.abs(err) > 0.35) yawRate += Math.sign(err) * prox * FLIGHT.yawRate;
  }
  flight.wallLR = lateral(FLIGHT.wallLoom * prox * prox, wrapAngle(Math.atan2(ry, rx) - flight.yaw)); applyLooming(d);

  const yawRate0 = flight.yawRate;
  flight.yawRate += (yawRate - flight.yawRate) * (dt / 0.12);
  const yawAccel = (flight.yawRate - yawRate0) / dt;
  flight.yaw = wrapAngle(flight.yaw + flight.yawRate * dt);
  flight.vx = speed * Math.cos(flight.yaw); flight.vy = speed * Math.sin(flight.yaw);
  flight.x += flight.vx * dt; flight.y += flight.vy * dt;
  const rx2 = flight.x - B.cx, ry2 = flight.y - B.cy, rc2 = Math.hypot(rx2, ry2);
  if (rc2 > B.r) { flight.x = B.cx + rx2 / rc2 * B.r; flight.y = B.cy + ry2 / rc2 * B.r; }
  const z0 = flight.z;
  flight.z += (clamp(zTarget, Math.min(B.zmin, flight.home.z), B.zmax) - flight.z) * (dt / 0.35);
  const exact = flight.state === 'takeoff' || flight.state === 'settle' || flight.state === 'touchdown' || flight.state === 'walk';
  if (exact) flight.z = zTarget;   // exact lift-off, touchdown and walking heights
  flight.vz = (flight.z - z0) / dt;
  const airborne = flight.state === 'takeoff' || flight.state === 'flight' || flight.state === 'landing';
  // Bank into the turn, leading it slightly (a share of yaw acceleration), like a real banked turn.
  const rollTarget = airborne ? -FLIGHT.bank * clamp((flight.yawRate + 0.12 * yawAccel) / FLIGHT.yawRate, -1, 1) : 0;
  flight.roll += (rollTarget - flight.roll) * (dt / 0.25);
  flight.pitch += ((airborne ? FLIGHT.pitch * (speed / FLIGHT.speed) : 0) - flight.pitch) * (dt / 0.3);
}

// Physics-step write (0.1 ms): root pose and the wingbeat. Every other joint stays simulated.
function writeFlightPose(d: MjData) {
  if (flight.state === 'ground') return;
  d.qpos[0] = flight.x; d.qpos[1] = flight.y; d.qpos[2] = flight.z;
  const q = qmul(qmul(qaxis(0, 0, 1, flight.yaw), qaxis(0, 1, 0, flight.pitch)), qaxis(1, 0, 0, flight.roll));
  for (let i = 0; i < 4; i++) d.qpos[3 + i] = q[i];
  d.qvel[0] = flight.vx; d.qvel[1] = flight.vy; d.qvel[2] = flight.vz;
  d.qvel[3] = d.qvel[4] = d.qvel[5] = 0;
  // The wings stay under neural control while walking, and are free during touchdown: they
  // reach the folded pose by the end of settle, and a kinematically pinned wing against the
  // hind legs shoved the legs into the floor.
  if (flight.state === 'walk' || flight.state === 'touchdown') return;
  // Stylized stroke: yaw sweeps fore-aft, roll (elevation) and pitch follow a quarter cycle
  // behind. Amplitude per side follows the steering DN asymmetry, their documented function.
  // The stroke centre blends in from the takeoff pose and back out to the folded rest pose,
  // so the wings are handed back to neural control from where the springs can hold them.
  flight.phase += 2 * Math.PI * FLIGHT.flapHz * 1e-4;
  const s = Math.sin(flight.phase), c = Math.cos(flight.phase);
  const A = flight.flap;
  const takingOff = flight.state === 'takeoff';
  wingJoints.forEach((w, i) => {
    const amp = A * (1 + (w.side === 'l' ? 0.25 : -0.25) * flight.asym);
    const stroke = WING_STROKE[w.axis];
    const centre = takingOff ? flight.wingFrom[i] + (stroke - flight.wingFrom[i]) * A
                             : stroke + (WING_REST[w.axis] - stroke) * flight.fold;
    const v = w.axis === 'yaw' ? centre + 0.85 * amp * s : centre + 0.35 * amp * c;
    d.qpos[w.qadr] = v; d.qvel[w.dadr] = 0;
  });
}
// Legs while off the ground (supplied), using the shuffle joints and offsets. In flight the
// front and middle legs tuck under the body; the hind legs keep their standing targets
// (lifted, they fold up into the wing stroke and pin the wings against the femur). Walking is
// a tripod cycle: alternate sets of three feet swing while the body is carried forward.
const TRIPOD_A = new Set(['T1_left', 'T2_right', 'T3_left']);
const ctrlClamp = (ai: number, v: number) => Math.max(model.actuator_ctrlrange[2 * ai], Math.min(model.actuator_ctrlrange[2 * ai + 1], v));
function stepFlightLegs(d: MjData) {
  const walking = flight.state === 'walk';
  const dir = flight.walk.dir > 0 ? 1 : -1;
  for (const leg of shuffleLegs) {
    leg.amount = 0;
    let lift: number, swing = 0;
    if (walking || flight.walk.settle > 0) {
      // Tripod cycle. A foot lifts through the swing half while its fore-aft joint carries it
      // forward on a cosine, and is planted through the stance half while the joint carries it
      // back LINEARLY at the body's speed, so stance feet stay put as the body is carried
      // forward. After a bout the stride blends out over WALK.stopSec instead of snapping.
      const ph = (flight.walk.phase + (TRIPOD_A.has(leg.name) ? 0 : Math.PI)) % (2 * Math.PI);
      const inSwing = ph < Math.PI;
      lift = inSwing ? Math.sin(ph) : 0;
      swing = (inSwing ? -Math.cos(ph) : 1 - 2 * (ph - Math.PI) / Math.PI) * dir;
      if (!walking) { lift *= flight.walk.settle; swing *= flight.walk.settle; }
    } else lift = leg.name.startsWith('T3') ? 0 : flight.tuck * 2.2;
    for (const { ai, offset } of leg.joints) d.ctrl[ai] = ctrlClamp(ai, holdCtrl[ai] + offset * lift);
    d.ctrl[leg.swing.ai] = ctrlClamp(leg.swing.ai, holdCtrl[leg.swing.ai] + leg.swing.amp * swing);
  }
}

function stepSimulation() {
  writeFlightPose(data);
  mujoco.mj_step(model, data);
  sim.steps++;
  // Exactly one brain millisecond per ten physics steps, including the frame's compute budget.
  // Neither calibration nor a body reset can put the brain ahead and freeze neural updates.
  if (sim.steps % 10 === 0) {
    brain.step(1);
    stepWorld(data, 0.001);
    stepPoke(0.001);
    stepLoom(data, 0.001);
    applyBrainToActuators(brain, data);
    stepFlight(brain, data, 0.001);
    if (flight.state === 'ground') { stepShuffle(brain, data, 0.001); stepGroom(brain, data, 0.001); }
    else { stepFlightLegs(data); groom.active = false; }
  }
}

// ---------------------------------------------------------------- main
(async function main() {
  try {
    say('initializing physics');
    mujoco = await loadMujoco();
    await stageFiles(mujoco);

    say('compiling body model');
    model = mujoco.MjModel.loadFromXML('/w/' + SCENE_XML);
    data  = new mujoco.MjData(model);

    brain = await Brain.load('./brain', say);
    say('calibrating resting rates');
    await brain.calibrateResponsive(2500);
    say('initializing renderer');

    // ---- three.js
    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);        // MuJoCo is Z-up
    const scene = new THREE.Scene();
    scene.background = skyTexture();
    scene.fog = new THREE.Fog(0xcdeaf0, 3.5, 14.0);   // matches the sky's mid-tone

    const device = (navigator as Navigator & {deviceMemory?:number});
    const quality = new AdaptiveQuality({
      compact:matchMedia('(pointer:coarse)').matches || Math.min(innerWidth, innerHeight) <= 700,
      cores:device.hardwareConcurrency || 8, memory:device.deviceMemory || 8,
    });
    const renderer = new THREE.WebGLRenderer({ antialias:quality.level === 2, powerPreference:'low-power' });
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, quality.profile.pixelRatio));
    renderer.setSize(innerWidth, innerHeight);
    renderer.shadowMap.enabled = quality.profile.shadowSize > 0;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    document.body.appendChild(renderer.domElement);
    renderer.domElement.setAttribute('aria-label', `${document.body.dataset.namedFly || 'Fly'}, live fruit fly simulation. Drag to orbit; scroll to zoom; tap the fly to touch it.`);
    renderer.domElement.setAttribute('role', 'img');

    // Keep the fly's horizontal framing on narrow screens rather than cropping its wings.
    const viewFov = () => 2 * Math.atan(Math.tan(19 * Math.PI / 180) / Math.min(1, innerWidth / innerHeight)) * 180 / Math.PI;
    const camera = new THREE.PerspectiveCamera(viewFov(), innerWidth/innerHeight, 0.01, 100);
    camera.up.set(0, 0, 1);
    camera.position.set(0.62, -0.62, 0.22);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.dampingFactor = 0.07;
    controls.target.set(-0.03, 0, -0.04);
    controls.minDistance = 0.15; controls.maxDistance = 12;
    const homePosition = camera.position.clone();
    const homeTarget = controls.target.clone();
    $('b_home').onclick = () => {
      // Recenter on the fly wherever it is: the home framing, offset to the live thorax.
      const t = followTarget();
      camera.position.copy(homePosition).sub(homeTarget).add(t); controls.target.copy(t); controls.update();
    };
    const followPos = new THREE.Vector3(), followDelta = new THREE.Vector3();
    const followBody = mujoco.mj_name2id(model, 1 /* mjOBJ_BODY */, 'thorax');
    function followTarget() {
      return followPos.set(data.xpos[followBody * 3], data.xpos[followBody * 3 + 1], data.xpos[followBody * 3 + 2]);
    }
    // The orbit target tracks the thorax wherever the fly walks or flies, while the camera stays
    // where the user put it, panning to keep the fly in view: a spectator, not a chase camera.
    // Moving the camera with the fly was tried and flew it straight through the flowers and fern.
    function stepFollow(dt: number) {
      followDelta.copy(followTarget()).sub(controls.target).multiplyScalar(1 - Math.exp(-dt / 0.25));
      if (followDelta.lengthSq() > 1e-10) controls.target.add(followDelta);
      stepOcclusion(dt);
    }
    // A spectator that knows where to stand: when the terrarium hides the fly (one raycast
    // from the thorax to the camera, every sixth frame), the camera slides along its orbit
    // about the fly, toward the nearest angle with a clear view, at a bounded rate. It never
    // moves otherwise, and a drag still wins (the user's own orbit is the starting point).
    const occ = { frame: 0, blocked: false, goal: 0, frames: 0, blockedFrames: 0, ray: new THREE.Raycaster(), dir: new THREE.Vector3(), off: new THREE.Vector3(), cand: new THREE.Vector3() };
    function viewBlocked(from: THREE.Vector3, to: THREE.Vector3) {
      if (!viewOccluders.length) return false;
      occ.dir.copy(to).sub(from); const dist = occ.dir.length();
      if (dist < 0.05) return false;
      occ.ray.set(from, occ.dir.multiplyScalar(1 / dist)); occ.ray.near = 0.02; occ.ray.far = dist - 0.02;
      return occ.ray.intersectObjects(viewOccluders, false).length > 0;
    }
    function stepOcclusion(dt: number) {
      if (occ.goal !== 0) {
        const step = Math.sign(occ.goal) * Math.min(Math.abs(occ.goal), 1.5 * dt);
        occ.off.copy(camera.position).sub(controls.target).applyAxisAngle(_up, step);
        camera.position.copy(controls.target).add(occ.off); occ.goal -= step;
        return;
      }
      if (++occ.frame % 6 !== 0) return;
      occ.frames++;
      occ.blocked = viewBlocked(controls.target, camera.position);
      if (!occ.blocked) return;
      occ.blockedFrames++;
      for (let k = 1; k <= 8; k++) for (const sign of [1, -1]) {
        occ.cand.copy(camera.position).sub(controls.target).applyAxisAngle(_up, sign * k * 0.2).add(controls.target);
        if (!viewBlocked(controls.target, occ.cand)) { occ.goal = sign * k * 0.2; return; }
      }
    }
    const _up = new THREE.Vector3(0, 0, 1);

    const hemi = new THREE.HemisphereLight(0x9fc4ff, 0x1a2028, 1.15);
    scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 2.1);
    // The fly faces +X; +Y is its left. Preserve the sun's radius and elevation.
    key.position.set(0, Math.hypot(0.5, 0.7), 0.9); key.castShadow = true;
    key.shadow.mapSize.set(quality.profile.shadowSize || 512, quality.profile.shadowSize || 512);
    const c = key.shadow.camera; c.near = 0.05; c.far = 4;
    c.left = -0.5; c.right = 0.5; c.top = 0.5; c.bottom = -0.5;
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x6fa8ff, 0.7);
    rim.position.set(-0.8, 0.5, 0.35); scene.add(rim);

    say('building meshes');
    const tris = buildScene(scene, model);
    resetSim();
    syncGeoms(model, data);

    say('loading terrarium');
    const flyBox = new THREE.Box3();
    for (const g of geomNodes) if (g.group <= 2 && !g.isFloor) flyBox.expandByObject(g.mesh);
    loadProps(scene, flyBox).then(() => {
      const terrarium = propObjs['terrarium.glb'];
      const hillMeshes: THREE.Mesh[] = [];
      // Only the terrain/decor meshes are "ground" for the ball to land on — exclude the glass
      // shell (transparent) and the black metal frame (opaque but not something to rest on),
      // or a random hill-top search finds the outside of the glass roof, the tallest thing there.
      function worldZRange(n: THREE.Mesh): [number, number] {
        n.updateWorldMatrix(true, false);
        const geo = n.geometry;
        if (!geo.boundingBox) geo.computeBoundingBox();
        const bb = geo.boundingBox;
        if (!bb) throw new Error('mesh has no bounding box');
        let zmin = Infinity, zmax = -Infinity;
        for (const cx of [bb.min.x, bb.max.x]) for (const cy of [bb.min.y, bb.max.y]) for (const cz of [bb.min.z, bb.max.z]) {
          const v = new THREE.Vector3(cx, cy, cz).applyMatrix4(n.matrixWorld);
          if (v.z < zmin) zmin = v.z;
          if (v.z > zmax) zmax = v.z;
        }
        return [zmin, zmax];
      }
      const glassMeshes: THREE.Mesh[] = [];
      const solidMeshes: THREE.Mesh[] = [];
      const sack = propObjs['sugar_sack.glb'];
      const sackMeshes: THREE.Mesh[] = [];
      if (sack) sack.traverse(n => { if (n instanceof THREE.Mesh) sackMeshes.push(n); });
      if (terrarium) {
        // The tall fern is built from many short stacked segments, so no single mesh's own
        // span flags it as "tall" — filter by absolute height instead. Ground-level decor (the
        // hill mound, rocks, low flowers) sits in the bottom slice of the terrarium; anything
        // climbing toward the glass roof is foliage reaching upward, not something to land on.
        let zminAll = Infinity, zmaxAll = -Infinity;
        terrarium.traverse(n => {
          if (!(n instanceof THREE.Mesh)) return;
          const [zmin, zmax] = worldZRange(n);
          if (zmin < zminAll) zminAll = zmin;
          if (zmax > zmaxAll) zmaxAll = zmax;
        });
        const hillCeiling = zminAll + (zmaxAll - zminAll) * 0.30;
        // The glass shell is the model's one alpha-blend material (confirmed against the glTF's
        // own material table — everything else is opaque) — a real collider baked from that
        // mesh, not a cylinder approximating its footprint.
        terrarium.traverse(n => {
          if (!(n instanceof THREE.Mesh)) return;
          const material = Array.isArray(n.material) ? n.material[0] : n.material;
          if (material.transparent) glassMeshes.push(n);
        });
        terrarium.traverse(n => {
          if (!(n instanceof THREE.Mesh)) return;
          const material = Array.isArray(n.material) ? n.material[0] : n.material;
          if (material.transparent) return;
          solidMeshes.push(n);                     // everything opaque — the real collision set
          if (material instanceof THREE.MeshStandardMaterial && material.color.getHexString() === '191919') return;
          const [, zmax] = worldZRange(n);
          if (zmax > hillCeiling) return;
          hillMeshes.push(n);                       // short ground-level decor only — for the
        });                                          // hilltop-placement search below
      }
      if (glassMeshes.length) {
        // Flight stays inside the real glass: 72% of its inner radius and below its roof.
        const gbox = new THREE.Box3();
        for (const m of glassMeshes) gbox.expandByObject(m);
        const gsize = new THREE.Vector3(), gcenter = new THREE.Vector3();
        gbox.getSize(gsize); gbox.getCenter(gcenter);
        flight.bounds = { cx: gcenter.x, cy: gcenter.y, r: 0.72 * Math.min(gsize.x, gsize.y) / 2,
                          zmin: 0.2, zmax: Math.min(gbox.max.z - 0.35, 1.2) };
        // Walkable floor: sample the terrarium's own opaque meshes from above; a cell is
        // floor when what it hits is at floor height (the hill, rocks, plants and the sugar
        // sack are obstacles, the glass is ignored). Measured: the low-poly floor sits between
        // -0.10 and +0.10, the hill from 0.6 up, the plants and frame from 1.0 up. One-time
        // raycasts at load.
        const n = 56, span = 2 * flight.bounds.r, cell = span / n;
        const x0 = gcenter.x - span / 2, y0 = gcenter.y - span / 2;
        const ok = new Uint8Array(n * n), plant = new Uint8Array(n * n);
        // Plants by the glTF's material colours (every mesh is auto-named): the flowers are
        // f06193, the foliage and fern fronds 4bb150 and df9b45, and the small 8ec44b pieces
        // are fern tips; the large 8ec44b mound is the hill.
        const isPlant = (o: THREE.Object3D) => {
          if (!(o instanceof THREE.Mesh)) return false;
          const m = Array.isArray(o.material) ? o.material[0] : o.material;
          if (!(m instanceof THREE.MeshStandardMaterial)) return false;
          const c = m.color.getHexString();
          if (c === 'f06193' || c === '4bb150' || c === 'df9b45') return true;
          if (c !== '8ec44b') return false;
          const bb = new THREE.Box3().setFromObject(o), sz = new THREE.Vector3(); bb.getSize(sz);
          return Math.max(sz.x, sz.y) < 1.5;
        };
        for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
          const x = x0 + (i + 0.5) * cell, y = y0 + (j + 0.5) * cell;
          if (Math.hypot(x - gcenter.x, y - gcenter.y) > flight.bounds.r - cell) continue;
          const hit = groundUnder(solidMeshes, x, y);
          if (hit && hit.point.z < WORLD.floorZ + WORLD.floorBand) ok[j * n + i] = 1;
          else if (hit && isPlant(hit.object)) plant[j * n + i] = 1;
        }
        world.ground = { x0, y0, cell, n, ok, plant };
        // The sugar goes on open floor: the nearest cell to its nominal spot whose 5x5
        // surroundings are all floor (a first placement put it half inside a rock), and its
        // footprint then becomes an obstacle the feet stop at.
        if (sack) {
          let best: [number, number] | null = null, bd = Infinity;
          for (let j = 2; j < n - 2; j++) for (let i = 2; i < n - 2; i++) {
            let clear = true;
            for (let dj = -2; dj <= 2 && clear; dj++) for (let di = -2; di <= 2; di++) if (ok[(j + dj) * n + i + di] !== 1) { clear = false; break; }
            if (!clear) continue;
            const cx = x0 + (i + 0.5) * cell, cy = y0 + (j + 0.5) * cell;
            const dd = (cx - world.sugar.x) ** 2 + (cy - world.sugar.y) ** 2;
            if (dd < bd) { bd = dd; best = [cx, cy]; }
          }
          if (best) { sack.position.x = best[0]; sack.position.y = best[1]; world.sugar.x = best[0]; world.sugar.y = best[1]; }
          sack.updateMatrixWorld(true);
          for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
            const cx = x0 + (i + 0.5) * cell, cy = y0 + (j + 0.5) * cell;
            if (Math.hypot(cx - world.sugar.x, cy - world.sugar.y) < world.sugar.r + 0.02) ok[j * n + i] = 0;
          }
        }
      }
      solidMeshes.push(...sackMeshes);   // solid for the ball, after the sack has found its spot
      viewOccluders.push(...solidMeshes);
      if (hillMeshes.length) {
        loadBall(scene, flyBox, hillMeshes, glassMeshes, solidMeshes).catch(err => console.warn('loadBall failed', err));
      }
    }).catch(err => console.warn('loadProps failed', err));

    buildDriveMap(model, brain);
    buildShuffleMap();
    buildFlightMap();
    buildWorldMap();
    // Sugar comes from the world (the sack) unless the switch overrides it.
    stimSwitch.sweet = 0; applyStim('sweet'); stimSwitch.sweetLeg = 0; applyStim('sweetLeg');
    $('b_world').onclick = (e) => {
      world.enabled = !world.enabled;
      if (e.currentTarget instanceof HTMLElement) {
        e.currentTarget.classList.toggle('on', world.enabled);
        e.currentTarget.setAttribute('aria-pressed', String(world.enabled));
      }
    };
    $('s_neu').textContent = brain.N.toLocaleString();
    $('s_syn').textContent = brain.meta.E.toLocaleString();
    $('s_nbody').textContent = String(model.nbody);
    $('s_nu').textContent    = String(model.nu);
    $('s_tri').textContent   = tris.toLocaleString();

    // ---- controls
    $('b_pause').onclick = (e) => {
      sim.paused = !sim.paused;
      document.body.classList.toggle('is-paused', sim.paused);
      if (e.currentTarget instanceof HTMLElement) {
        e.currentTarget.setAttribute('aria-label', sim.paused ? 'Resume' : 'Pause');
        e.currentTarget.title = sim.paused ? 'Resume' : 'Pause';
      }
    };
    for (const btn of document.querySelectorAll('[data-stim]')) {
      if (!(btn instanceof HTMLButtonElement)) continue;
      const k = btn.dataset.stim;
      if (!k) continue;
      if (!(k in brain.stim)) {          // stale brain.js, or a typo'd data-stim
        btn.disabled = true; btn.title = 'no such stimulus in the loaded brain.js';
        console.warn(`stimulus "${k}" not in brain.js — stale cache?`);
        continue;
      }
      if (btn.dataset.event === 'loom') {      // an event, not a switch: one approach per press
        btn.onclick = () => { startLoom(); syncStimUI(); };
        continue;
      }
      btn.onclick = () => {
        stimSwitch[k] = (stimSwitch[k] || 0) > 0 ? 0 : 1;
        applyStim(k);
        if (k === 'sweet') { stimSwitch.sweetLeg = stimSwitch.sweet; applyStim('sweetLeg'); }   // infinite sugar: mouth and feet
        syncStimUI();
      };
    }
    // Buttons show the sustained switch; inspector rows show what the brain actually receives
    // (switch or transient pulse). Called on clicks, on the HUD tick, and by any code that sets
    // a stimulus programmatically, so the page never shows stale state.
    const stimElements = [...document.querySelectorAll('[data-stim]')].filter((el): el is HTMLElement => el instanceof HTMLElement);
    function syncStimUI() {
      for (const el of stimElements) {
        const k = el.dataset.stim || '';
        const on = el instanceof HTMLButtonElement && !el.dataset.event ? (stimSwitch[k] || 0) > 0 : brain.stim[k] > 0;
        if (el.classList.contains('on') !== on) el.classList.toggle('on', on);
        if (el instanceof HTMLButtonElement) el.setAttribute('aria-pressed', String(on));
      }
      $('sugar-label').textContent = brain.sugar > 0 ? 'Sugar on' : 'Sugar off';
      document.body.classList.toggle('is-sugar-off', brain.sugar <= 0);
    }
    syncStimUI();

    // ---- light: the visual stimulus and what the viewer sees agree
    // The scene brightens with the `light` level (11,426 visual neurons driven), from the
    // artwork's resting look at 0 to a sunlit terrarium at 1. Rendering only; the stimulus
    // itself is the switch, and the terrarium still sends nothing to the brain.
    const REST_LIGHT = { hemi: hemi.intensity, key: key.intensity, rim: rim.intensity, bg: 1.0 };
    const lighting = { cur: 0 };
    // The world's day/night cycle scales the resting look (night keeps 30%), and the Light
    // switch brightens on top of it.
    function stepLighting(dt: number) {
      const target = stimSwitch.light || 0;
      lighting.cur += (target - lighting.cur) * (1 - Math.exp(-dt / 0.35));
      const u = lighting.cur, day = world.enabled ? 0.3 + 0.7 * world.day : 1;
      hemi.intensity = REST_LIGHT.hemi * day * (1 + 0.45 * u);
      key.intensity  = REST_LIGHT.key  * day * (1 + 0.55 * u);
      rim.intensity  = REST_LIGHT.rim  * day * (1 + 0.3 * u);
      scene.backgroundIntensity = REST_LIGHT.bg * (0.45 + 0.55 * day) * (1 + 0.25 * u);
      if (scene.fog instanceof THREE.Fog) scene.fog.color.setRGB((0.80 + 0.12 * u) * day, (0.92 + 0.06 * u) * day, (0.94 + 0.04 * u) * day);
    }

    // ---- sound: a wingbeat tone while the wings stroke and a soft tick as each tripod lands,
    // both synthesised and driven by the same state that draws them (flight.flap, the stride
    // phase). Muted by default; the AudioContext is created on the first press of the button.
    const sound = { on: false, ctx: null as AudioContext | null, gain: null as GainNode | null, osc: null as OscillatorNode | null, stance: -1 };
    function soundStart() {
      const ctx = new AudioContext();
      const gain = ctx.createGain(); gain.gain.value = 0; gain.connect(ctx.destination);
      const osc = ctx.createOscillator(); osc.type = 'triangle'; osc.frequency.value = 190; osc.connect(gain); osc.start();
      const osc2 = ctx.createOscillator(); osc2.type = 'sine'; osc2.frequency.value = 475; const g2 = ctx.createGain(); g2.gain.value = 0.35; osc2.connect(g2); g2.connect(gain); osc2.start();
      sound.ctx = ctx; sound.gain = gain; sound.osc = osc;
    }
    function stepSound() {
      if (!sound.on || !sound.ctx || !sound.gain || !sound.osc) return;
      const ctx = sound.ctx, now = ctx.currentTime;
      const airborne = flight.state === 'takeoff' || flight.state === 'flight' || flight.state === 'landing';
      sound.gain.gain.setTargetAtTime(airborne ? 0.06 * flight.flap : 0, now, 0.05);
      sound.osc.frequency.setTargetAtTime(190 + 25 * flight.asym, now, 0.1);
      const stance = flight.state === 'walk' ? Math.floor(flight.walk.phase / Math.PI) : -1;   // a tripod lands every half cycle
      if (stance !== sound.stance) {
        if (stance >= 0 && sound.stance >= 0) {
          const tick = ctx.createOscillator(), tg = ctx.createGain();
          tick.type = 'sine'; tick.frequency.value = 1400 + 300 * (stance % 2);
          tg.gain.setValueAtTime(0.035, now); tg.gain.exponentialRampToValueAtTime(0.0005, now + 0.03);
          tick.connect(tg); tg.connect(ctx.destination); tick.start(now); tick.stop(now + 0.035);
        }
        sound.stance = stance;
      }
    }
    $('b_sound').onclick = (e) => {
      sound.on = !sound.on;
      if (sound.on && !sound.ctx) soundStart();
      if (sound.ctx) { if (sound.on) sound.ctx.resume(); else sound.ctx.suspend(); }
      if (e.currentTarget instanceof HTMLElement) { e.currentTarget.classList.toggle('on', sound.on); e.currentTarget.setAttribute('aria-pressed', String(sound.on)); e.currentTarget.title = sound.on ? 'Mute' : 'Sound: wingbeat and footsteps'; }
    };

    // ---- the sack shows how much sugar is left: it slumps to half height as it empties
    const sackScale = { z: -1 };
    function stepSack() {
      const sack = propObjs['sugar_sack.glb'];
      if (!sack) return;
      if (sackScale.z < 0) sackScale.z = sack.scale.z;
      sack.scale.z = sackScale.z * (0.5 + 0.5 * world.sugar.amount);
    }

    const SENSE_ROWS: [string, () => number][] = [
      ['sweet', () => brain.rate.grn_sweet], ['bitter', () => brain.rate.grn_bitter], ['odour', () => brain.rate.orn],
      ['touch', () => brain.rate.mechano], ['heat', () => brain.rate.thermo_hot], ['cool', () => brain.rate.thermo_cold],
      ['damp', () => brain.rate.hygro], ['light', () => brain.rate.visual], ['looming', () => (brain.rate.lc4 + brain.rate.lplc2) / 2],
      ['object', () => brain.rate.lc11]];

    // ---- the status pill: a sentence from the state the fly is in and what the world offers
    function statusSentence() {
      const W = world.enabled, wl = world.levels, S = flight.state;
      const towardSugar = W && world.sugar.placed && world.sugarDist < 1.0 && Math.abs(bearingTo(data, world.sugar.x, world.sugar.y)) < 0.6;
      if (S === 'walk' || (S === 'touchdown' && flight.flap === 0 && flight.fold === 0)) {
        if (flight.walk.turning) return 'Turning at an obstacle';
        if (flight.walk.dir < 0) return 'Walking backward';
        return towardSugar && wl.odour > 0 ? 'Walking toward the sugar' : 'Walking';
      }
      if (S === 'takeoff') return 'Taking off';
      if (S === 'flight') return flight.escape > FLIGHT.quietRate ? 'Flying from a threat' : 'Flying';
      if (S !== 'ground') return 'Landing';
      if (W && wl.sweet > 0) return world.sugar.amount < 0.05 ? 'At an empty sack' : 'Feeding at the sack';
      if (brain.sugar > 0) return 'Feeding';
      if (W && wl.bitter > 0) return 'Tasting a plant';
      if (W && wl.sweetLeg > 0) return 'Standing in the sugar';
      if (groom.active) return 'Grooming the antennae';
      if (W && wl.touch > 0) return 'Bumped by the ball';
      if (W && wl.odour > 0.3) return 'Smelling the sugar';
      return '';
    }

    // ---- looming object: the visible cause of the escape response
    // A matte dark sphere rather than a disc: it reads the same from every camera angle and
    // casts a real shadow onto the fly as it arrives. It approaches from ahead and above, in
    // the camera's frame for the last half second (the 19 degree view is narrow).
    const loomBall = new THREE.Mesh(
      new THREE.SphereGeometry(LOOM.r, 28, 18),
      new THREE.MeshStandardMaterial({ color: 0x1c1622, roughness: 0.9, metalness: 0, transparent: true, opacity: 1 }));
    loomBall.visible = false; loomBall.castShadow = true;
    scene.add(loomBall);
    const LOOM_DIR = new THREE.Vector3(LOOM.dir[0], LOOM.dir[1], LOOM.dir[2]);
    const thoraxBody = mujoco.mj_name2id(model, 1 /* mjOBJ_BODY */, 'thorax');
    const thoraxPos = new THREE.Vector3();
    function stepLoomDisc() {
      if (!loom.active) { loomBall.visible = false; world.loomers.length = Math.min(world.loomers.length, 1); return; }
      thoraxPos.set(data.xpos[thoraxBody * 3], data.xpos[thoraxBody * 3 + 1], data.xpos[thoraxBody * 3 + 2]);
      const passT = LOOM.approach + LOOM.hold;
      const d = loom.t <= passT ? loom.d : LOOM.dmin + (loom.t - passT) * 4.0;   // retreats the way it came
      loomBall.position.copy(thoraxPos).addScaledVector(LOOM_DIR, d);
      loomBall.material.opacity = loom.t <= passT ? 1 : Math.max(0, 1 - (loom.t - passT) / LOOM.fade);
      loomBall.visible = true;
      // Present to the world as an occluder only (velocity 0): its looming is the event's own pulse.
      if (!world.loomers[0]) world.loomers[0] = { x: 0, y: 0, z: -9, vx: 0, vy: 0, vz: 0, r: 0, name: 'none' };
      world.loomers[1] = { x: loomBall.position.x, y: loomBall.position.y, z: loomBall.position.z, vx: 0, vy: 0, vz: 0, r: LOOM.r, name: 'sphere' };
    }

    // ---- poke: tap the fly to touch it
    // A tap (short press, little movement) is raycast against the fly's visible geoms — the same
    // meshes the render bridge syncs, so the hit body is read straight off the MuJoCo geom index.
    // A drag is left to OrbitControls. The raycast reads the scene and never writes physics.
    const flyMeshes = geomNodes.filter(g => g.group <= 2 && !g.isFloor).map(g => g.mesh);
    const meshGeom = new Map(geomNodes.map(g => [g.mesh, g.gi]));
    const tapRay = new THREE.Raycaster();
    const ripples: { sprite: THREE.Sprite, t: number }[] = [];
    const ripplePool: THREE.Sprite[] = [];
    const rippleTexture = (() => {
      const N = 64, cv = document.createElement('canvas');
      cv.width = cv.height = N;
      const g = cv.getContext('2d');
      if (!g) throw new Error('2D canvas is unavailable');
      g.strokeStyle = '#fff6d8'; g.lineWidth = 5;
      g.beginPath(); g.arc(N / 2, N / 2, N / 2 - 4, 0, Math.PI * 2); g.stroke();
      const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace;
      return tex;
    })();
    const flySize = new THREE.Vector3(); flyBox.getSize(flySize);
    const rippleSize = Math.max(flySize.x, flySize.y) * 0.12;
    function pokeAtScreen(clientX: number, clientY: number) {
      tapRay.setFromCamera(new THREE.Vector2(clientX / innerWidth * 2 - 1, -(clientY / innerHeight) * 2 + 1), camera);
      const hit = tapRay.intersectObjects(flyMeshes, false)[0];
      if (!hit || !(hit.object instanceof THREE.Mesh)) return null;
      const gi = meshGeom.get(hit.object);
      if (gi === undefined) return null;
      // Side of the hit: the contact point's offset from the root along the body's left axis.
      const yaw = yawOf(data.qpos.subarray(3, 7));
      const ly = -(hit.point.x - data.qpos[0]) * Math.sin(yaw) + (hit.point.y - data.qpos[1]) * Math.cos(yaw);
      pokeBody(model.geom_bodyid[gi], ly > 0.01 ? 1 : ly < -0.01 ? -1 : 0);
      // sprites are pooled: a tap reuses a finished ripple's sprite rather than allocating one
      const sprite = ripplePool.pop() || new THREE.Sprite(new THREE.SpriteMaterial({ map: rippleTexture, transparent: true, depthTest: false, depthWrite: false }));
      sprite.position.copy(hit.point); sprite.renderOrder = 10; sprite.visible = true;
      scene.add(sprite); ripples.push({ sprite, t: 0 });
      return poke.last;
    }
    function stepRipples(dt: number) {
      for (let i = ripples.length - 1; i >= 0; i--) {
        const r = ripples[i];
        r.t += dt;
        const u = Math.min(1, r.t / 0.45);
        r.sprite.scale.setScalar(rippleSize * (0.5 + 1.8 * u));
        r.sprite.material.opacity = 1 - u;
        if (u >= 1) { scene.remove(r.sprite); r.sprite.visible = false; ripplePool.push(r.sprite); ripples.splice(i, 1); }
      }
    }
    const tap = { x:0, y:0, t:0, id:-1 };
    renderer.domElement.addEventListener('pointerdown', e => { tap.x = e.clientX; tap.y = e.clientY; tap.t = performance.now(); tap.id = e.pointerId; });
    renderer.domElement.addEventListener('pointerup', e => {
      if (e.pointerId !== tap.id) return;
      tap.id = -1;
      if (Math.hypot(e.clientX - tap.x, e.clientY - tap.y) > 8 || performance.now() - tap.t > 350) return;
      pokeAtScreen(e.clientX, e.clientY);
    });

    for (const [triggerId, dialogId] of [['b_about', 'about'], ['b_inspect', 'inspect']]) {
      const dialog = $(dialogId);
      if (!(dialog instanceof HTMLDialogElement)) throw new Error(`missing dialog ${dialogId}`);
      $(triggerId).onclick = () => dialog.showModal();
      dialog.addEventListener('click', (e) => {
        const r = dialog.getBoundingClientRect();
        if (e.target === dialog && (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom)) dialog.close();
      });
    }

    const neuralCanvas = $('neural-map');
    const neuralMap = neuralCanvas instanceof HTMLCanvasElement
      ? await createNeuralMap(neuralCanvas, brain).catch(err => {
        console.warn('Neural view unavailable', err);
        $('neural-map-title').textContent = 'Neural view unavailable';
        return null;
      }) : null;
    function applyQuality() {
      const profile = quality.profile;
      document.body.dataset.quality = profile.name;
      renderer.setPixelRatio(Math.min(devicePixelRatio || 1, profile.pixelRatio));
      const shadows = profile.shadowSize > 0;
      renderer.shadowMap.enabled = shadows;
      key.castShadow = shadows;
      key.shadow.map?.dispose(); key.shadow.map = null;
      key.shadow.mapSize.set(profile.shadowSize || 512, profile.shadowSize || 512);
      scene.traverse(object => {
        if (!(object instanceof THREE.Mesh)) return;
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        for (const material of materials) material.needsUpdate = true;
      });
      neuralMap?.setQuality(profile.mapRatio);
    }
    applyQuality();
    boot.remove();
    $('artwork').removeAttribute('inert');
    $('b_about').focus({ preventScroll:true });
    $('b_about').click();

    // ---- loop
    const timestep = 1e-4;                          // flybody's opt.timestep
    let last = performance.now(), acc = 0, fps = 0, fpsT = last, frames = 0, sps = 0, spsN = 0, spsT = last;
    let nextFrameAt = last, mapAt = 0, hudAt = 0, loomShown = false, flightShown = '';
    // Browser suspension must never become a backlog of simulation work on return.
    document.addEventListener('visibilitychange', () => {
      last = nextFrameAt = performance.now();
      acc = 0; frames = 0; spsN = 0; fpsT = spsT = last;
      quality.resetSampling();
    });

    function frame() {
      requestAnimationFrame(frame);
      // Use performance.now() rather than the rAF timestamp: the two can sit on different
      // origins, which yields a negative first delta and stalls the accumulator.
      const now = performance.now();
      if (document.hidden) { last = nextFrameAt = now; acc = 0; return; }
      if (now + 1 < nextFrameAt) return;
      nextFrameAt = Math.max(nextFrameAt + 1000 / quality.profile.fps, now);
      const frameMs = now - last;
      const wall = Math.max(0, Math.min(frameMs / 1000, 0.05));
      last = now;

      if (!sim.paused) {
        acc = Math.min(acc + wall, 0.05);   // never try to 'catch up' more than 50 ms
        const budget = quality.profile.budgetMs;
        const tStart = performance.now();
        let n = 0;
        while (acc > timestep && performance.now() - tStart < budget) {
          stepSimulation();
          acc -= timestep; n++;
        }
        spsN += n;

        syncGeoms(model, data);
      }

      frames++;
      if (now - fpsT > 500) { fps = frames * 1000 / (now - fpsT); frames = 0; fpsT = now; }
      if (now - spsT > 500) {
        sps = spsN * 1000 / (now - spsT); spsN = 0; spsT = now;
        $('s_sps').textContent = Math.round(sps).toLocaleString();
        $('s_rt').textContent  = (sps * timestep).toFixed(2) + '×';
      }
      // Hidden inspector statistics do not need per-frame DOM work.
      if (now >= hudAt && $('inspect').hasAttribute('open')) {
        hudAt = now + 200;
        syncStimUI();
        $('s_fps').textContent  = fps.toFixed(0);
        $('s_time').textContent = data.time.toFixed(2) + ' s';
        $('s_bms').textContent  = ((brain.ms - sim.brainStartMs) / 1000).toFixed(2) + ' s';
        $('s_pop').textContent  = brain.popRate.toFixed(1) + ' Hz';
        for (const [k, rate] of SENSE_ROWS) {   // the loop: world -> switch -> neurons
          const wl = (world.levels as Record<string, number>)[k];
          $('w_' + k).textContent = world.enabled && wl !== undefined ? wl.toFixed(2) : '–';
          $('x_' + k).textContent = (stimSwitch[k] || 0) > 0 ? (stimSwitch[k] || 0).toFixed(2) : k === 'touch' && stimPulse.touch[0] + stimPulse.touch[1] > 0 ? 'tap' : k === 'looming' && loom.active ? 'loom' : '–';
          $('s_' + k).textContent = rate().toFixed(0) + ' Hz';
        }
        $('s_mnp').textContent  = brain.rate.mn_proboscis.toFixed(1) + ' Hz';
        $('s_mni').textContent  = brain.rate.mn_ingestion.toFixed(1) + ' Hz';
        $('s_neck').textContent = ((brain.rate.mn_neck_l + brain.rate.mn_neck_r) / 2).toFixed(1) + ' Hz';
        $('s_ant').textContent  = ((brain.rate.mn_antenna_l + brain.rate.mn_antenna_r) / 2).toFixed(1) + ' Hz';
        $('s_gf').textContent   = brain.rate.dn_gf.toFixed(1) + ' Hz';
        $('s_esc').textContent  = ((brain.rate.dn_escwing_l + brain.rate.dn_escwing_r) / 2).toFixed(0) + ' Hz';
        $('s_steer').textContent = brain.rate.dn_steer_l.toFixed(0) + ' / ' + brain.rate.dn_steer_r.toFixed(0) + ' Hz';
        $('s_groom').textContent = brain.rate.dn_groom.toFixed(0) + ' Hz';
        $('s_pam').textContent  = brain.rate.pam.toFixed(1) + ' Hz';
        $('s_flight').textContent = `${flight.state}; ${flight.count} flights, ${flight.walk.count} walks`;
        $('s_escape').textContent = flight.escape.toFixed(0) + ' Hz';
        const wl = world.levels;
        $('s_day').textContent = world.enabled ? `${(world.day * 100).toFixed(0)}% daylight` : 'off';
        $('s_sugar_d').textContent = world.sugar.placed ? `${world.sugarDist.toFixed(2)} cm · ${Math.round(world.sugar.amount * 100)} % left` : '–';
        $('s_wtaste').textContent = `${wl.sweet.toFixed(2)} / ${wl.sweetLeg.toFixed(2)} / ${wl.bitter.toFixed(0)} / ${wl.odour.toFixed(2)}`;
        $('s_wlight').textContent = `${wl.light.toFixed(2)} / ${wl.heat.toFixed(2)} / ${wl.cool.toFixed(2)}`;
        $('s_wloom').textContent = `${wl.looming.toFixed(2)} / ${wl.object.toFixed(2)} / ${wl.touch.toFixed(2)}`;
        $('s_steerside').textContent = flight.asym.toFixed(2);
      }
      $('s_cnt').textContent  = brain.sugarFeedSpikes.toLocaleString();

      if (!sim.paused) stepBall(wall);
      stepSack();
      stepSound();
      stepRipples(wall);
      stepLighting(wall);
      stepLoomDisc();
      if (loom.active !== loomShown) { loomShown = loom.active; syncStimUI(); }
      stepFollow(wall);
      const doing = statusSentence();
      if (doing !== flightShown) {
        flightShown = doing;
        $('flight-state').hidden = doing === '';
        $('flight-state').textContent = doing;
        document.body.classList.toggle('is-flying', flight.state !== 'ground');
      }
      controls.update();
      renderer.render(scene, camera);
      if (now >= mapAt) {
        neuralMap?.draw();
        mapAt = now + 1000 / quality.profile.mapFps;
      }
      if (!sim.paused && quality.sample(frameMs, performance.now() - now)) applyQuality();
    }
    requestAnimationFrame(frame);

    addEventListener('resize', () => {
      camera.fov = viewFov();
      camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
      renderer.setSize(innerWidth, innerHeight);
      renderer.setPixelRatio(Math.min(devicePixelRatio || 1, quality.profile.pixelRatio));
      neuralMap?.draw();
    });
    const flyWindow = (window as Window & typeof globalThis & { fly?: unknown, __flyReady?: boolean, __flyError?: string });
    flyWindow.fly = { mujoco, model, data, brain, sim, scene, camera, renderer, controls, geomNodes, quality,
                   applyBrain: () => applyBrainToActuators(brain, data), driveMap, shuffle, shuffleLegs,
                   stepSimulation, stimSwitch, stimPulse, applyStim, syncStimUI, poke, pokeBody, pokeAtScreen,
                   loom, startLoom, lighting, lights: { hemi, key, rim }, flight, FLIGHT, WALK,
                   world, WORLD, walkable, nearestWalkable, plantAt, stimWorld, stepWorld, groom, GROOM,
                   sync: () => syncGeoms(model, data),
                   stepBall, get ball() { return ball; }, get ballWorld() { return ballWorld; },
                   occlusion: occ, statusSentence, sound,
                   dbg: () => ({ paused: sim.paused, acc, steps: sim.steps, time: data.time,
                                 nodes: geomNodes.length,
                                 brainMs: brain.ms, sugar: brain.sugar,
                                 rates: { ...brain.rate }, pop: brain.popRate }) };
    const exitMessage = `${document.body.dataset.namedFly || 'The fly'} exists fully within this browser tab. If you close the tab this is the relative equivalent of killing an insect. Do you accept this moral hazard?`;
    $('b_exit').onclick = () => {
      if (!window.confirm(exitMessage)) return;
      // Pages cannot close a tab they did not open. Navigation destroys this
      // document and its simulation.
      window.location.replace('about:blank');
    };
    flyWindow.__flyReady = true;
  } catch (err) {
    console.error(err);
    const message = err instanceof Error ? err.stack || err.message : String(err);
    boot.setAttribute('role', 'alert');
    boot.innerHTML = '<div class="err"><h2>Unable to start simulation.</h2><p>The simulation could not start. Try reloading in a browser with WebGL enabled.</p><a href="./">Reload</a></div>';
    const flyWindow = (window as Window & typeof globalThis & { __flyError?: string });
    flyWindow.__flyError = message;
  }
})();
