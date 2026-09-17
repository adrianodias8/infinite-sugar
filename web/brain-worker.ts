// The brain's thread. Loads the connectome and the kernel, calibrates, and steps batches of
// milliseconds on request, returning one record per millisecond (see brain-proxy.ts).
import { Brain } from './brain.js';

let brain: Brain | null = null;
let tracked = new Int32Array(0);
const post = (m: unknown, transfer?: Transferable[]) => (self as unknown as Worker).postMessage(m, transfer || []);

self.onmessage = async (e: MessageEvent) => {
  const m = e.data as Record<string, unknown>;
  if (m.type === 'load') {
    brain = await Brain.load(String(m.base), (message) => post({ type: 'say', message }));
    post({ type: 'loaded', meta: brain.meta, roleNames: brain.roleNames, STIM: brain.STIM, STIM_SIDES: brain.STIM_SIDES, stimDrive: brain.stimDrive, stimGain: brain.stimGain, kernel: brain.kernel });
    return;
  }
  if (!brain) return;
  if (m.type === 'calibrate') {
    const rest = await brain.calibrateResponsive(Number(m.ms));
    post({ type: 'calibrated', rest, rate: { ...brain.rate }, ms: brain.ms });
    return;
  }
  if (m.type === 'step') {
    if (m.stimLR) for (const [k, lr] of Object.entries(m.stimLR as Record<string, [number, number]>)) brain.setStimLR(k, lr[0], lr[1]);
    if (m.cellClear) brain.setCellDrive(null, null);
    if (m.cellIdx && m.cellAmt) brain.setCellDrive(m.cellIdx as Int32Array, m.cellAmt as Float32Array);
    if (m.track) tracked = Int32Array.from(m.track as ArrayLike<number>);
    const n = Number(m.ms), R = brain.roleNames.length;
    const rates = new Float32Array(n * R), counts = new Int32Array(n * R), pop = new Float32Array(n), feed = new Int32Array(n);
    for (let t = 0; t < n; t++) {
      brain.step(1);
      for (let r = 0; r < R; r++) { rates[t * R + r] = brain.rate[brain.roleNames[r]]; counts[t * R + r] = brain._cnt[r]; }
      pop[t] = brain.popRate;
    }
    const last = new Float64Array(tracked.length);
    for (let k = 0; k < tracked.length; k++) last[k] = brain.lastSpikeMs[tracked[k]];
    post({ type: 'batch', ms: brain.ms, n, rates, counts, pop, feed, tracked: last, totalSpikes: brain.totalSpikes }, [rates.buffer, counts.buffer, pop.buffer, feed.buffer, last.buffer]);
  }
};
