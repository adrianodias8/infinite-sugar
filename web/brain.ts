// whole-brain leaky integrate-and-fire over the FlyWire FAFB v783 connectome.
// 139,255 neurons / 2,700,513 edges, 1 kHz. Kernel traced from desktop-fly's Sim.swift
// (MIT, Denis Shiryaev); parameters matched to tools/reflex_test.py, the reference implementation.
// Wiring is measured; gains are hand-tuned — see docs/04-roadmap.md.
//
// Kernel performance constraints:
//   - the dense pass is fused: decay + baseline + threshold in ONE loop over N
//   - noise and the delayed-inhibition queue are SPARSE (touched-index lists, never a full scan)
// The naive version costs ~1050 us/step; this one costs ~100-240 us/step.

const DECAY = Math.fround(Math.exp(-1 / 20));   // 20 ms membrane tau at 1 ms steps
const THRESH = 1.0;
const REFRACT = 2;                              // ms
const INH_DELAY = 4;                            // ms — inhibition arrives late, excitation doesn't
const INH_SLOTS = INH_DELAY + 1;

const WSCALE = 0.0050;      // synapse-count -> membrane units
const BASE_MAX = 0.06;      // per-neuron tonic drive ~ U(0, BASE_MAX)
const NOISE_PER_STEP = 300; // sparse random kicks per ms
const NOISE_KICK = 0.42;

const ROLE_SLOTS = 4;       // pools a neuron may be counted in: union, sub-pool, side, sub-pool side (a JO wind cell is in mechano, mechano_l, mechano_wind, mechano_wind_l)

type BrainMeta = { N: number, E: number, roles: Record<string, number[]> };
// What the page needs from a brain: the real one (this file) or BrainProxy (brain-proxy.ts,
// the same brain on a worker thread).
export type BrainAPI = Pick<Brain, 'N' | 'meta' | 'rate' | 'rest' | 'stim' | 'stimLR' | 'stimDrive' | 'stimGain' | 'sugar' | 'feedSpikes' | 'sugarFeedSpikes'
  | 'popRate' | 'ms' | 'totalSpikes' | 'lastSpikeMs' | 'roleNames' | 'groups' | 'STIM' | 'STIM_SIDES' | 'kernel' | '_cellAmt' | '_cnt'
  | 'setStim' | 'setStimLR' | 'setCellDrive' | 'syncCellDrive' | 'spikesOf' | 'calibrateResponsive' | 'step'>;
export type LR = [number, number];
type WasmExports = {
  setup: (...args: number[]) => void; setCellDrive: (idx: number, amt: number, n: number) => void;
  setState: (seed: number, slot: number, ms: number, total: number) => void; step: (ai: number, al: number, aa: number, n: number) => number;
};
type WasmKernel = { ex: WasmExports, memory: WebAssembly.Memory, off: Record<string, number>, activeIdx: Int32Array, activeLen: Int32Array, activeAmt: Float64Array,
                    cellIdx: Int32Array, cellAmt: Float32Array, activeDirty: boolean, cellSrc: Int32Array | null, cellN: number, nactive?: number };
type ActiveStimulus = { idx: Int32Array, amt: number };

async function loadBlob<T extends ArrayBufferView>(url: string, Ctor: { new(buffer: ArrayBuffer): T }): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  if (!res.body) throw new Error(`${url}: response has no body`);
  const ds = res.body.pipeThrough(new DecompressionStream('gzip'));
  return new Ctor(await new Response(ds).arrayBuffer());
}

export class Brain {
  declare meta: BrainMeta;
  declare N: number;
  declare indptr: Uint32Array;
  declare colidx: Uint32Array;
  declare w: Float32Array;
  declare v: Float32Array;
  declare refr: Uint8Array;
  declare baseline: Float32Array;
  declare _rng: number;
  declare inhVal: Float32Array[];
  declare inhIdx: Int32Array[];
  declare inhCnt: Int32Array;
  declare spiked: Int32Array;
  declare lastSpikeMs: Float64Array;
  declare slot: number;
  declare ms: number;
  declare totalSpikes: number;
  declare groups: Record<string, Int32Array>;
  declare roleNames: string[];
  declare roleOf: Int8Array;      // N x ROLE_SLOTS pool indices per neuron, -1 = none
  declare rate: Record<string, number>;
  declare _cnt: Int32Array;
  declare popRate: number;
  declare rateAlpha: number;
  declare STIM: Record<string, string[]>;
  declare STIM_SIDES: Record<string, [string[], string[]]>;
  declare stim: Record<string, number>;
  declare stimLR: Record<string, [number, number]>;
  declare stimDrive: number;
  declare stimGain: Record<string, number>;
  declare _active: ActiveStimulus[];
  declare _cellIdx: Int32Array | null;   // per-cell drive (the eyes): neuron indices and membrane units per ms
  declare _cellAmt: Float32Array | null;
  declare kernel: string;
  declare _wasm: WasmKernel | null;
  declare sugar: number;
  declare feedSpikes: number;
  declare sugarFeedSpikes: number;
  declare rest: Record<string, number> | null;

  constructor(meta: BrainMeta, indptr: Uint32Array, colidx: Uint32Array, w: Float32Array) {
    this.meta = meta;
    this.N = meta.N;
    this.indptr = indptr; this.colidx = colidx; this.w = w;
    const N = this.N;

    this.v = new Float32Array(N);
    this.refr = new Uint8Array(N);
    this.baseline = new Float32Array(N);
    let s = 22222;                               // deterministic: same brain every load
    const rnd = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 4294967296; };
    for (let i = 0; i < N; i++) this.baseline[i] = rnd() * BASE_MAX;
    this._rng = s;

    this.inhVal = [];
    this.inhIdx = [];
    this.inhCnt = new Int32Array(INH_SLOTS);
    for (let k = 0; k < INH_SLOTS; k++) {
      this.inhVal.push(new Float32Array(N));
      this.inhIdx.push(new Int32Array(N));
    }
    this.spiked = new Int32Array(N);
    // Read-only visualization telemetry; simulation time keeps flashes still when paused.
    this.lastSpikeMs = new Float64Array(N).fill(-Infinity);
    this.slot = 0; this.ms = 0; this.totalSpikes = 0;

    this.groups = {};
    for (const [k, arr] of Object.entries(meta.roles)) this.groups[k] = Int32Array.from(arr);
    this.roleNames = Object.keys(this.groups);
    // A neuron can belong to a union pool, a sub-pool and a side pool (hygro + hygro_cool +
    // hygro_l). Rates are counted per pool, so every membership is kept, up to ROLE_SLOTS; one
    // more would silently drop counts, hence the hard error. Readout only — no effect on dynamics.
    this.roleOf = new Int8Array(N * ROLE_SLOTS).fill(-1);
    this.roleNames.forEach((k, ri) => {
      for (const i of this.groups[k]) {
        let slot = 0;
        while (slot < ROLE_SLOTS && this.roleOf[i * ROLE_SLOTS + slot] >= 0) slot++;
        if (slot === ROLE_SLOTS) throw new Error(`neuron ${i} is in more than ${ROLE_SLOTS} role pools (${k})`);
        this.roleOf[i * ROLE_SLOTS + slot] = ri;
      }
    });

    this.rate = {};
    for (const k of this.roleNames) this.rate[k] = 0;
    this._cnt = new Int32Array(this.roleNames.length);
    this.popRate = 0;
    this.rateAlpha = 1 / 25;   // ~25 ms; 1/80 was so smooth that driven pools looked frozen

    // Sensory switches. Each maps to real FlyWire sensory populations; levels are 0..1.
    // Response of every motor pool to each of these was measured before wiring:
    // see tools/response_matrix.py and docs/05-results-phase1-3.md.
    this.STIM = {
      // Labellar and tarsal sugar sensing are separate channels: the world drives the feet
      // from the sugar under them and the labellum only by contact. Measured: labellar drive
      // alone raises the proboscis MNs +205%, tarsal drive alone LOWERS them 45% (no
      // proboscis-extension reflex from the feet in this wiring); the Sugar switch drives both.
      sweet:    ['grn_sweet'],
      sweetLeg: ['grn_sweet_leg'],
      bitter:  ['grn_bitter'],
      odour:   ['orn'],
      touch:   ['mechano'],
      wind:    ['mechano_wind'],   // JO wind/gravity cells: the antennae deflected by moving air
      // Hot and cold cells are separate populations in v783 (tools/build_brain.py splits
      // them by sub_class): heat drives the 'heating' thermosensory cells; cool drives the
      // 'cold' thermosensory cells plus the hygrosensory cells FlyWire labels as
      // cooling / evaporative-cooling responsive. `thermo` remains the union for readouts.
      heat:    ['thermo_hot'],
      cool:    ['thermo_cold', 'hygro_cool'],
      damp:    ['hygro'],
      light:   ['visual'],
      looming: ['lc4', 'lplc2'],   // LC4 + LPLC2: the fly's actual looming detectors
      object:  ['lc11'],           // LC11: small-object motion detectors (something crossing the view)
    };
    // Channels the world can drive from one side. Pools split by FlyWire's side label
    // (tools/build_brain.py); a channel not listed here drives both sides equally.
    this.STIM_SIDES = {
      sweet:   [['grn_sweet_l'], ['grn_sweet_r']],
      bitter:  [['grn_bitter_l'], ['grn_bitter_r']],
      odour:   [['orn_l'], ['orn_r']],
      touch:   [['mechano_l'], ['mechano_r']],
      wind:    [['mechano_wind_l'], ['mechano_wind_r']],
      damp:    [['hygro_l'], ['hygro_r']],
      light:   [['visual_l'], ['visual_r']],
      looming: [['lc4_l', 'lplc2_l'], ['lc4_r', 'lplc2_r']],
      object:  [['lc11_l'], ['lc11_r']],
    };
    this.stim = {};
    this.stimLR = {};
    for (const k of Object.keys(this.STIM)) { this.stim[k] = 0; this.stimLR[k] = [0, 0]; }
    this.stimDrive = 0.20;       // membrane units at level 1
    // Per-channel multiplier on stimDrive. Explicit input gains, not a kernel change; the
    // default is 1 for every channel. heat: the seven hot cells receive ~0.36 units/ms of
    // tonic inhibition at rest from eight interneurons firing at the refractory ceiling, so
    // the standard 0.20 drive never lifts them above threshold (measured: 0 Hz at 1x, 17 Hz at
    // 1.5x, 170 Hz at 2.5x). 2.5x puts them in the band of the other driven sensory pools.
    // tools/response_matrix.py applies the same gain so the NumPy reference agrees.
    this.stimGain = { heat: 2.5 };
    this._active = [];           // rebuilt by setStim()
    this._cellIdx = null; this._cellAmt = null;
    this.kernel = 'js'; this._wasm = null;
    this.sugar = 0;
    this.feedSpikes = 0;         // running total: proboscis + ingestion MN spikes
    this.sugarFeedSpikes = 0;    // same populations, counted only while sugar is enabled
    this.rest = null;            // resting rate per role, filled by calibrate()
  }

  static async load(base = './brain', say: (message: string) => void = () => {}) {
    say('fetching connectome');
    const meta = (await (await fetch(`${base}/meta.json`)).json()) as BrainMeta;
    say(`connectome: ${meta.N.toLocaleString()} neurons, ${meta.E.toLocaleString()} edges`);
    const [indptr, colidx, w2] = await Promise.all([
      loadBlob(`${base}/indptr.bin.gz`, Uint32Array),
      loadBlob(`${base}/colidx.bin.gz`, Uint32Array),
      loadBlob(`${base}/w2.bin.gz`, Int16Array),
    ]);
    say('unpacking weights');
    const w = new Float32Array(w2.length);
    const k = WSCALE / 2;                        // w2 = syn * sign * 2
    for (let i = 0; i < w2.length; i++) w[i] = w2[i] * k;
    const brain = new Brain(meta, indptr, colidx, w);
    // The WebAssembly kernel, when the browser has it: the same arithmetic as the JavaScript
    // one (bit-identical runs, tools/determinism_test.mjs), faster. Falls back silently.
    try {
      const bytes = await (await fetch(`${base}/../kernel/lif.wasm`)).arrayBuffer();
      brain.useKernel(await WebAssembly.compile(bytes));
      say('kernel: WebAssembly');
    } catch (err) { console.warn('WebAssembly kernel unavailable, using the JavaScript kernel', err); }
    return brain;
  }

  // Move the state into a WebAssembly module's memory and step there. Every array the page or
  // the tests read (v, refr, lastSpikeMs, _cnt, ...) becomes a view into that memory, so nothing
  // else changes. The module's step is web/kernel/lif.c, compiled by tools/build_kernel.sh.
  useKernel(module: WebAssembly.Module) {
    const N = this.N, E = this.colidx.length;
    const lay: [string, number, number][] = [   // name, bytes, alignment
      ['v', N * 4, 4], ['refr', N, 1], ['baseline', N * 4, 4], ['indptr', (N + 1) * 4, 4], ['colidx', E * 4, 4], ['w', E * 4, 4],
      ['inhVal', INH_SLOTS * N * 4, 4], ['inhIdx', INH_SLOTS * N * 4, 4], ['inhCnt', INH_SLOTS * 4, 4], ['spiked', N * 4, 4],
      ['lastSpikeMs', N * 8, 8], ['roleOf', N * ROLE_SLOTS, 1], ['cnt', this.roleNames.length * 4, 4],
      ['activeIdx', N * 4, 4], ['activeLen', 64 * 4, 4], ['activeAmt', 64 * 8, 8], ['cellIdx', N * 4, 4], ['cellAmt', N * 4, 4],
    ];
    // The module's own globals and stack sit below __heap_base; the state goes above it.
    let total = 0; for (const [, bytes] of lay) total += bytes + 8;
    const memory = new WebAssembly.Memory({ initial: Math.ceil((total + (1 << 20)) / 65536) + 4 });
    const instance = new WebAssembly.Instance(module, { env: { memory } });
    const ex = instance.exports as unknown as WasmExports & { __heap_base?: WebAssembly.Global };
    const off: Record<string, number> = {}; let p = ex.__heap_base ? (ex.__heap_base.value as number) : 1 << 20;
    for (const [name, bytes, align] of lay) { p = Math.ceil(p / align) * align; off[name] = p; p += bytes; }
    if (p > memory.buffer.byteLength) memory.grow(Math.ceil((p - memory.buffer.byteLength) / 65536) + 1);
    const buf = memory.buffer;
    const f32 = (name: string, n: number) => new Float32Array(buf, off[name], n);
    const i32 = (name: string, n: number) => new Int32Array(buf, off[name], n);
    // copy the state in, then re-point the fields at the views
    const v = f32('v', N); v.set(this.v); this.v = v;
    const refr = new Uint8Array(buf, off.refr, N); refr.set(this.refr); this.refr = refr;
    const baseline = f32('baseline', N); baseline.set(this.baseline); this.baseline = baseline;
    const indptr = new Uint32Array(buf, off.indptr, N + 1); indptr.set(this.indptr); this.indptr = indptr;
    const colidx = new Uint32Array(buf, off.colidx, E); colidx.set(this.colidx); this.colidx = colidx;
    const w = f32('w', E); w.set(this.w); this.w = w;
    const inhVal: Float32Array[] = [], inhIdx: Int32Array[] = [];
    for (let k = 0; k < INH_SLOTS; k++) {
      const a = new Float32Array(buf, off.inhVal + k * N * 4, N); a.set(this.inhVal[k]); inhVal.push(a);
      const b = new Int32Array(buf, off.inhIdx + k * N * 4, N); b.set(this.inhIdx[k]); inhIdx.push(b);
    }
    this.inhVal = inhVal; this.inhIdx = inhIdx;
    const inhCnt = i32('inhCnt', INH_SLOTS); inhCnt.set(this.inhCnt); this.inhCnt = inhCnt;
    const spiked = i32('spiked', N); spiked.set(this.spiked); this.spiked = spiked;
    const last = new Float64Array(buf, off.lastSpikeMs, N); last.set(this.lastSpikeMs); this.lastSpikeMs = last;
    const roleOf = new Int8Array(buf, off.roleOf, N * ROLE_SLOTS); roleOf.set(this.roleOf); this.roleOf = roleOf;
    const cnt = i32('cnt', this.roleNames.length); cnt.set(this._cnt); this._cnt = cnt;
    ex.setup(N, E, INH_SLOTS, INH_DELAY, ROLE_SLOTS, this.roleNames.length, NOISE_PER_STEP, DECAY, THRESH, REFRACT, NOISE_KICK,
      off.v, off.refr, off.baseline, off.indptr, off.colidx, off.w, off.inhVal, off.inhIdx, off.inhCnt, off.spiked, off.lastSpikeMs, off.roleOf, off.cnt, this._rng >>> 0);
    ex.setState(this._rng >>> 0, this.slot, this.ms, this.totalSpikes);
    this._wasm = { ex, memory, off, activeIdx: i32('activeIdx', N), activeLen: i32('activeLen', 64), activeAmt: new Float64Array(buf, off.activeAmt, 64),
                   cellIdx: i32('cellIdx', N), cellAmt: f32('cellAmt', N), activeDirty: true, cellSrc: null, cellN: 0 };
    this.kernel = 'wasm';
  }

  setStim(name: string, level: number) { this.setStimLR(name, level, level); }

  // Left and right levels for a channel. Where the sides differ and the channel has
  // side-split pools, the shared part drives the union pool and the excess drives the
  // stronger side's pool, so unsided neurons (30 ORNs, 76 central visual cells) still see the
  // common level. Equal levels are exactly the old uniform drive.
  setStimLR(name: string, left: number, right: number) {
    if (!(name in this.stim)) return;
    this.stimLR[name] = [left, right];
    this.stim[name] = Math.max(left, right);
    this._active = [];
    for (const k of Object.keys(this.stim)) {
      const [l, r] = this.stimLR[k];                 // each channel's OWN levels, not the argument
      const common = Math.min(l, r), gain = this.stimDrive * (this.stimGain[k] || 1);
      if (common > 0) {
        for (const p of this.STIM[k]) { const g = this.groups[p]; if (g) this._active.push({ idx: g, amt: common * gain }); }
      }
      const sides = this.STIM_SIDES[k];
      if (!sides || l === r) continue;
      const extra = Math.abs(l - r);
      for (const p of sides[l > r ? 0 : 1]) { const g = this.groups[p]; if (g) this._active.push({ idx: g, amt: extra * gain }); }
    }
    this.sugar = this.stim.sweet;   // the counter counts feeding: labellar sugar, not the feet
    if (this._wasm) this._wasm.activeDirty = true;
  }

  // A drive per neuron rather than per pool, in membrane units per millisecond: the eyes drive
  // each photoreceptor by the brightness it sees (web/app.ts, vision). Pass null to clear. The
  // arrays are used in place and must stay the same length; the caller owns them.
  setCellDrive(idx: Int32Array | null, amt: Float32Array | null) {
    if (idx && amt && idx.length !== amt.length) throw new Error('setCellDrive: idx and amt lengths differ');
    this._cellIdx = idx; this._cellAmt = idx ? amt : null;
    if (this._wasm) this.syncCellDrive();
  }
  // The WebAssembly kernel reads the per-cell drive from its own memory: copy the caller's arrays
  // in. Called by setCellDrive and again by whoever rewrites the amounts in place (the eyes).
  syncCellDrive() {
    const k = this._wasm; if (!k) return;
    const idx = this._cellIdx, amt = this._cellAmt;
    if (!idx || !amt) { k.ex.setCellDrive(0, 0, 0); k.cellSrc = null; k.cellN = 0; return; }
    if (idx.length > this.N) throw new Error('setCellDrive: more entries than neurons');
    k.cellIdx.set(idx); k.cellAmt.set(amt); k.cellSrc = idx; k.cellN = idx.length;
    k.ex.setCellDrive(k.off.cellIdx, k.off.cellAmt, idx.length);
  }

  // Spikes in a pool during the last millisecond stepped (0 for an unknown pool).
  spikesOf(role: string) {
    const r = this.roleNames.indexOf(role);
    return r < 0 ? 0 : this._cnt[r];
  }

  // Resting rates depend on the kernel, not just the wiring — the NumPy reference and this
  // one differ. Measure them here at load rather than hard-coding numbers from elsewhere.
  // Deliberately a one-shot calibration, NOT a running adaptation: a slow adaptive baseline
  // would make a sustained stimulus fade, i.e. habituation, which is ruled out by design.
  // The resting rate of each pool is the MEAN spike rate over the last `measure` ms of the
  // calibration (spike counts, not the 25 ms rate estimate at the final tick): a two-cell
  // pool's 25 ms estimate is a coin toss, and a rest taken from it steered the fly in circles.
  calibrate(ms = 2500, measure = 1500) {
    const saved = { ...this.stim };
    for (const k of Object.keys(this.stim)) this.setStim(k, 0);
    measure = Math.min(measure, ms);
    this.step(ms - measure);
    this._calibrationCounts(measure);
    for (const [k, v] of Object.entries(saved)) this.setStim(k, v);
    return this.rest;
  }
  _calibrationCounts(measure: number) {
    const acc = new Float64Array(this.roleNames.length);
    for (let t = 0; t < measure; t++) { this.step(1); for (let r = 0; r < acc.length; r++) acc[r] += this._cnt[r]; }
    this.rest = {};
    for (let r = 0; r < this.roleNames.length; r++) {
      const k = this.roleNames[r], n0 = this.groups[k].length;
      this.rest[k] = Math.max(0.5, measure > 0 && n0 > 0 ? acc[r] * 1000 / (measure * n0) : this.rate[k]);
    }
  }

  // Same one-shot calibration, yielded in small batches so mobile loading UI stays responsive.
  async calibrateResponsive(ms = 2500, measure = 1500) {
    const saved = { ...this.stim };
    for (const k of Object.keys(this.stim)) this.setStim(k, 0);
    measure = Math.min(measure, ms);
    const warm = ms - measure;
    for (let elapsed = 0; elapsed < warm; elapsed += 25) {
      this.step(Math.min(25, warm - elapsed));
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    const acc = new Float64Array(this.roleNames.length);
    for (let elapsed = 0; elapsed < measure; elapsed += 25) {
      const n = Math.min(25, measure - elapsed);
      for (let t = 0; t < n; t++) { this.step(1); for (let r = 0; r < acc.length; r++) acc[r] += this._cnt[r]; }
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    this.rest = {};
    for (let r = 0; r < this.roleNames.length; r++) {
      const k = this.roleNames[r], n0 = this.groups[k].length;
      this.rest[k] = Math.max(0.5, measure > 0 && n0 > 0 ? acc[r] * 1000 / (measure * n0) : this.rate[k]);
    }
    for (const [k, v] of Object.entries(saved)) this.setStim(k, v);
    return this.rest;
  }

  _rand(): number { let s = this._rng; s ^= s << 13; s ^= s >>> 17; s ^= s << 5; this._rng = s; return s >>> 0; }

  // Rate bookkeeping shared by both kernels, once per millisecond after the spikes are known.
  _account(ns: number, rFeedA: number, rFeedB: number) {
    for (let r = 0; r < this.roleNames.length; r++) {
      const k = this.roleNames[r], n0 = this.groups[k].length;
      this.rate[k] += (this._cnt[r] * 1000 / n0 - this.rate[k]) * this.rateAlpha;
    }
    const feeding = this._cnt[rFeedA] + this._cnt[rFeedB];
    this.feedSpikes += feeding;
    if (this.sugar > 0) this.sugarFeedSpikes += feeding;
    this.popRate += (ns * 1000 / this.N - this.popRate) * this.rateAlpha;
    this.totalSpikes += ns;
    this.ms++;
    this.slot = (this.slot + 1) % INH_SLOTS;
  }

  _stepWasm(n: number) {
    const k = this._wasm!;
    const rFeedA = this.roleNames.indexOf('mn_proboscis');
    const rFeedB = this.roleNames.indexOf('mn_ingestion');
    if (k.activeDirty) {   // flatten the pool drives into the module's memory
      let o = 0, a = 0;
      for (const { idx, amt } of this._active) {
        if (a >= k.activeLen.length) throw new Error('too many active pool drives');
        k.activeIdx.set(idx, o); k.activeLen[a] = idx.length; k.activeAmt[a] = amt; o += idx.length; a++;
      }
      k.nactive = a; k.activeDirty = false;
    }
    for (let it = 0; it < n; it++) {
      const ns = k.ex.step(k.off.activeIdx, k.off.activeLen, k.off.activeAmt, k.nactive || 0);
      this._account(ns, rFeedA, rFeedB);
    }
  }

  step(n: number) {
    if (this._wasm) { this._stepWasm(n); return; }
    const { N, v, refr, baseline, indptr, colidx, w, spiked, inhVal, inhIdx, inhCnt } = this;
    const active = this._active;
    const rFeedA = this.roleNames.indexOf('mn_proboscis');
    const rFeedB = this.roleNames.indexOf('mn_ingestion');

    for (let it = 0; it < n; it++) {
      const slot = this.slot;

      const q = inhVal[slot], qi = inhIdx[slot], qn = inhCnt[slot];
      for (let k = 0; k < qn; k++) { const j = qi[k]; const nv = v[j] + q[j]; v[j] = nv < -2 ? -2 : nv; q[j] = 0; }
      inhCnt[slot] = 0;

      for (let k = 0; k < NOISE_PER_STEP; k++) v[this._rand() % N] += NOISE_KICK;

      for (let a = 0; a < active.length; a++) {
        const idx = active[a].idx, amt = active[a].amt;
        for (let k = 0; k < idx.length; k++) v[idx[k]] += amt;
      }
      const cidx = this._cellIdx, camt = this._cellAmt;
      if (cidx && camt) for (let k = 0; k < cidx.length; k++) v[cidx[k]] += camt[k];

      let ns = 0;
      for (let i = 0; i < N; i++) {
        const r = refr[i];
        if (r !== 0) { refr[i] = r - 1; v[i] *= DECAY; continue; }
        const vi = v[i] * DECAY + baseline[i];
        if (vi >= THRESH) { v[i] = 0; refr[i] = REFRACT; spiked[ns++] = i; }
        else v[i] = vi;
      }

      const is = (slot + INH_DELAY) % INH_SLOTS, iq = inhVal[is], iqi = inhIdx[is];
      let ic = inhCnt[is];
      for (let s = 0; s < ns; s++) {
        const i = spiked[s], a = indptr[i], b = indptr[i + 1];
        for (let k = a; k < b; k++) {
          const j = colidx[k], x = w[k];
          if (x >= 0) { const nv = v[j] + x; v[j] = nv < -2 ? -2 : nv; }
          else { if (iq[j] === 0) iqi[ic++] = j; iq[j] += x; }
        }
      }
      inhCnt[is] = ic;

      this._cnt.fill(0);
      for (let s = 0; s < ns; s++) {
        const i = spiked[s];
        this.lastSpikeMs[i] = this.ms;
        for (let slot = i * ROLE_SLOTS, e = slot + ROLE_SLOTS; slot < e; slot++) {
          const r = this.roleOf[slot];
          if (r < 0) break;
          this._cnt[r]++;
        }
      }
      this._account(ns, rFeedA, rFeedB);
    }
  }
}
