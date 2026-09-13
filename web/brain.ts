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

const ROLE_SLOTS = 3;       // pools a neuron may be counted in: union, sub-pool, side

type BrainMeta = { N: number, E: number, roles: Record<string, number[]> };
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
      sweet:   ['grn_sweet', 'grn_sweet_leg'],
      bitter:  ['grn_bitter'],
      odour:   ['orn'],
      touch:   ['mechano'],
      // Hot and cold cells are separate populations in v783 (tools/build_brain.py splits
      // them by sub_class): heat drives the 'heating' thermosensory cells; cool drives the
      // 'cold' thermosensory cells plus the hygrosensory cells FlyWire labels as
      // cooling / evaporative-cooling responsive. `thermo` remains the union for readouts.
      heat:    ['thermo_hot'],
      cool:    ['thermo_cold', 'hygro_cool'],
      damp:    ['hygro'],
      light:   ['visual'],
      looming: ['lc4', 'lplc2'],   // LC4 + LPLC2: the fly's actual looming detectors
    };
    // Channels the world can drive from one side. Pools split by FlyWire's side label
    // (tools/build_brain.py); a channel not listed here drives both sides equally.
    this.STIM_SIDES = {
      sweet:   [['grn_sweet_l'], ['grn_sweet_r']],
      bitter:  [['grn_bitter_l'], ['grn_bitter_r']],
      odour:   [['orn_l'], ['orn_r']],
      touch:   [['mechano_l'], ['mechano_r']],
      damp:    [['hygro_l'], ['hygro_r']],
      light:   [['visual_l'], ['visual_r']],
      looming: [['lc4_l', 'lplc2_l'], ['lc4_r', 'lplc2_r']],
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
    return new Brain(meta, indptr, colidx, w);
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
    this.sugar = this.stim.sweet;
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
  calibrate(ms = 2500) {
    const saved = { ...this.stim };
    for (const k of Object.keys(this.stim)) this.setStim(k, 0);
    this.step(ms);
    this.rest = {};
    for (const k of this.roleNames) this.rest[k] = Math.max(0.5, this.rate[k]);
    for (const [k, v] of Object.entries(saved)) this.setStim(k, v);
    return this.rest;
  }

  // Same one-shot calibration, yielded in small batches so mobile loading UI stays responsive.
  async calibrateResponsive(ms = 2500) {
    const saved = { ...this.stim };
    for (const k of Object.keys(this.stim)) this.setStim(k, 0);
    for (let elapsed = 0; elapsed < ms; elapsed += 25) {
      this.step(Math.min(25, ms - elapsed));
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    this.calibrate(0); // Capture the final resting rates without advancing or adapting the brain.
    for (const [k, v] of Object.entries(saved)) this.setStim(k, v);
    return this.rest;
  }

  _rand(): number { let s = this._rng; s ^= s << 13; s ^= s >>> 17; s ^= s << 5; this._rng = s; return s >>> 0; }

  step(n: number) {
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
      for (let r = 0; r < this.roleNames.length; r++) {
        const k = this.roleNames[r], n0 = this.groups[k].length;
        this.rate[k] += (this._cnt[r] * 1000 / n0 - this.rate[k]) * this.rateAlpha;
      }
      const feeding = this._cnt[rFeedA] + this._cnt[rFeedB];
      this.feedSpikes += feeding;
      if (this.sugar > 0) this.sugarFeedSpikes += feeding;
      this.popRate += (ns * 1000 / N - this.popRate) * this.rateAlpha;

      this.totalSpikes += ns;
      this.ms++;
      this.slot = (slot + 1) % INH_SLOTS;
    }
  }
}
