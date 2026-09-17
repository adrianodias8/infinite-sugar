// The brain on another thread. BrainProxy has the surface the page uses (rates, rest, stimulus
// levels, counters, last-spike times for the neural view) and runs the real Brain in a Web
// Worker (brain-worker.ts) in BATCHES of milliseconds: the page asks for a batch with the
// stimulus levels it has now, the worker steps it and returns one record per millisecond
// (every pool's rate and spike count, the whole-brain rate), and the page replays those records
// one per brain tick — physics, world and gait run on the main thread exactly as before,
// reading the record of the millisecond they are in. The worker runs ahead by up to
// `queueMax` milliseconds, so a change in the senses reaches the neurons `batchMs` to
// `queueMax` ms later: within a real fly's transduction delays, and stated in docs/17.
// Nothing in the brain changes; the kernel (JavaScript or WebAssembly) is the worker's.
import type { LR } from './brain.js';   // [left, right]

type BatchMessage = { type: 'batch', ms: number, n: number, rates: Float32Array, counts: Int32Array, pop: Float32Array, feed: Int32Array,
                      tracked: Float64Array, totalSpikes: number };
type LoadedMessage = { type: 'loaded', meta: { N: number, E: number, roles: Record<string, number[]> }, roleNames: string[],
                       STIM: Record<string, string[]>, STIM_SIDES: Record<string, [string[], string[]]>, stimDrive: number, stimGain: Record<string, number>, kernel: string };

export class BrainProxy {
  worker: Worker;
  meta!: { N: number, E: number, roles: Record<string, number[]> };
  N = 0;
  roleNames: string[] = [];
  groups: Record<string, Int32Array> = {};
  STIM: Record<string, string[]> = {};
  STIM_SIDES: Record<string, [string[], string[]]> = {};
  stimDrive = 0.2;
  stimGain: Record<string, number> = {};
  kernel = 'worker';
  rate: Record<string, number> = {};
  rest: Record<string, number> | null = null;
  stim: Record<string, number> = {};
  stimLR: Record<string, LR> = {};
  sugar = 0;
  feedSpikes = 0;
  sugarFeedSpikes = 0;
  popRate = 0;
  ms = 0;
  totalSpikes = 0;
  lastSpikeMs!: Float64Array;   // filled for the tracked neurons only (the neural view's sample)
  _cnt!: Int32Array;
  _cellIdx: Int32Array | null = null;
  _cellAmt: Float32Array | null = null;
  _cellDirty = false;
  _stimDirty = true;
  _tracked: Int32Array = new Int32Array(0);
  _trackDirty = false;
  // the queue of millisecond records
  batchMs = 16;
  queueMax = 64;
  _queue: BatchMessage[] = [];
  _cursor = 0;
  available = 0;      // millisecond records not yet replayed
  pending = 0;        // batches requested and not yet returned
  starved = 0;        // ticks the page wanted a record and had none (it then holds the last one)
  workerMs = 0;       // the worker's own clock: how far ahead it is
  private _rFeedA = -1; private _rFeedB = -1;
  private _resolvers: Record<string, ((m: unknown) => void)[]> = {};
  onSay: (message: string) => void = () => {};

  constructor(worker: Worker) {
    this.worker = worker;
    worker.onmessage = (e: MessageEvent) => this._receive(e.data);
    worker.onerror = (e) => console.error('brain worker', e.message);
  }

  static async load(base = './brain', say: (m: string) => void = () => {}) {
    const worker = new Worker(new URL('./brain-worker.js', import.meta.url), { type: 'module' });
    const proxy = new BrainProxy(worker);
    proxy.onSay = say;
    const loaded = proxy._wait<LoadedMessage>('loaded');
    worker.postMessage({ type: 'load', base });
    const m = await loaded;
    proxy.meta = m.meta; proxy.N = m.meta.N; proxy.roleNames = m.roleNames;
    for (const [k, arr] of Object.entries(m.meta.roles)) proxy.groups[k] = Int32Array.from(arr);
    proxy.STIM = m.STIM; proxy.STIM_SIDES = m.STIM_SIDES; proxy.stimDrive = m.stimDrive; proxy.stimGain = m.stimGain; proxy.kernel = `worker (${m.kernel})`;
    for (const k of proxy.roleNames) proxy.rate[k] = 0;
    for (const k of Object.keys(proxy.STIM)) { proxy.stim[k] = 0; proxy.stimLR[k] = [0, 0]; }
    proxy._cnt = new Int32Array(proxy.roleNames.length);
    proxy.lastSpikeMs = new Float64Array(proxy.N).fill(-Infinity);
    proxy._rFeedA = proxy.roleNames.indexOf('mn_proboscis'); proxy._rFeedB = proxy.roleNames.indexOf('mn_ingestion');
    return proxy;
  }

  private _wait<T>(type: string) { return new Promise<T>(resolve => { (this._resolvers[type] ||= []).push(resolve as (m: unknown) => void); }); }
  private _receive(m: { type: string } & Record<string, unknown>) {
    if (m.type === 'say') { this.onSay(String(m.message)); return; }
    if (m.type === 'batch') { const b = m as unknown as BatchMessage; this._queue.push(b); this.available += b.n; this.pending--; this.workerMs = b.ms; }
    const rs = this._resolvers[m.type]; if (rs && rs.length) rs.shift()!(m);
  }

  // Same semantics as Brain.setStimLR for the levels the page reads; the worker applies them.
  setStim(name: string, level: number) { this.setStimLR(name, level, level); }
  setStimLR(name: string, left: number, right: number) {
    if (!(name in this.stim)) return;
    const cur = this.stimLR[name];
    if (cur && cur[0] === left && cur[1] === right) return;
    this.stimLR[name] = [left, right]; this.stim[name] = Math.max(left, right);
    this.sugar = this.stim.sweet; this._stimDirty = true;
  }
  setCellDrive(idx: Int32Array | null, amt: Float32Array | null) {
    if (idx === this._cellIdx && amt === this._cellAmt) return;
    this._cellIdx = idx; this._cellAmt = amt; this._cellDirty = true;
  }
  syncCellDrive() { this._cellDirty = true; }
  // Which neurons' last-spike times the page wants back each batch (the neural view's sample).
  trackSpikes(ids: ArrayLike<number>) { this._tracked = Int32Array.from(ids); this._trackDirty = true; }
  spikesOf(role: string) { const r = this.roleNames.indexOf(role); return r < 0 ? 0 : this._cnt[r]; }

  async calibrateResponsive(ms = 2500) {
    const done = this._wait<{ rest: Record<string, number>, rate: Record<string, number>, ms: number }>('calibrated');
    this.worker.postMessage({ type: 'calibrate', ms });
    const m = await done;
    this.rest = m.rest; for (const k of Object.keys(m.rate)) this.rate[k] = m.rate[k];
    this.ms = m.ms; this.workerMs = m.ms;
    return this.rest;
  }

  // Keep the worker busy: ask for the next batch whenever fewer than queueMax ms are waiting.
  pump() {
    while (this.pending < 2 && this.available + this.pending * this.batchMs < this.queueMax) {
      const msg: Record<string, unknown> = { type: 'step', ms: this.batchMs };
      const transfer: Transferable[] = [];
      if (this._stimDirty) { msg.stimLR = this.stimLR; this._stimDirty = false; }
      if (this._cellDirty) {
        if (this._cellIdx && this._cellAmt) { const idx = Int32Array.from(this._cellIdx), amt = Float32Array.from(this._cellAmt); msg.cellIdx = idx; msg.cellAmt = amt; transfer.push(idx.buffer, amt.buffer); }
        else msg.cellClear = true;
        this._cellDirty = false;
      }
      if (this._trackDirty) { const t = Int32Array.from(this._tracked); msg.track = t; transfer.push(t.buffer); this._trackDirty = false; }
      this.worker.postMessage(msg, transfer); this.pending++;
    }
  }

  // One brain millisecond on the main thread: replay the next record. With none waiting, the
  // last record stands and `starved` counts the tick (the page then slows to the worker).
  step(n: number) {
    for (let it = 0; it < n; it++) {
      const b = this._queue[0];
      if (!b) { this.starved++; continue; }
      const R = this.roleNames.length, base = this._cursor * R;
      for (let r = 0; r < R; r++) { this.rate[this.roleNames[r]] = b.rates[base + r]; this._cnt[r] = b.counts[base + r]; }
      this.popRate = b.pop[this._cursor];
      const feeding = this._cnt[this._rFeedA] + this._cnt[this._rFeedB];
      this.feedSpikes += feeding; if (this.sugar > 0) this.sugarFeedSpikes += feeding;
      this.ms++; this.available--; this._cursor++;
      if (this._cursor >= b.n) {
        // the batch is spent: take its last-spike times for the tracked neurons, then drop it
        const t = this._tracked; for (let k = 0; k < t.length; k++) this.lastSpikeMs[t[k]] = b.tracked[k];
        this.totalSpikes = b.totalSpikes;
        this._queue.shift(); this._cursor = 0;
      }
    }
  }
}
