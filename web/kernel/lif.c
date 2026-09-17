// The whole-brain leaky integrate-and-fire kernel, in C for WebAssembly. The SAME arithmetic as
// web/brain.ts (double intermediates, single rounding into the float32 state), so a run is
// bit-identical between the two kernels; the JavaScript one stays as the reference and the
// fallback. Memory is owned by the JS side: every array lives in the module's linear memory and
// the offsets are passed in once (setup). No allocation, no libc.
//
// Per millisecond (step):
//   1. apply the delayed inhibition queued for this slot (clamped at -2)
//   2. NOISE_PER_STEP random kicks (xorshift32, the same generator and seed as brain.ts)
//   3. the pool drives (uniform per pool) and the per-cell drive (the eyes)
//   4. leak / threshold / reset for every neuron, collecting the spikes
//   5. propagate: excitation lands now, inhibition into the queue INH_DELAY ms ahead
//   6. per-pool spike counts (ROLE_SLOTS memberships per neuron), last-spike timestamps
typedef unsigned int u32; typedef int i32; typedef unsigned char u8; typedef signed char i8;

static i32 N, E, INH_SLOTS, INH_DELAY, ROLE_SLOTS, NROLES, NOISE_PER_STEP;
static double DECAY, THRESH, NOISE_KICK; static i32 REFRACT;
static float *v, *baseline, *w; static u8 *refr; static u32 *indptr, *colidx;
static float *inhVal; static i32 *inhIdx, *inhCnt;   // INH_SLOTS x N each (inhCnt: INH_SLOTS)
static i32 *spiked; static double *lastSpikeMs; static i8 *roleOf; static i32 *cnt;
static i32 *cellIdx; static float *cellAmt; static i32 cellN;
static u32 rng; static i32 slot; static double ms; static i32 totalSpikes;

__attribute__((export_name("setup")))
void setup(i32 n, i32 e, i32 inhSlots, i32 inhDelay, i32 roleSlots, i32 nroles, i32 noisePerStep, double decay, double thresh, i32 refract, double noiseKick,
           float *v_, u8 *refr_, float *baseline_, u32 *indptr_, u32 *colidx_, float *w_,
           float *inhVal_, i32 *inhIdx_, i32 *inhCnt_, i32 *spiked_, double *lastSpikeMs_, i8 *roleOf_, i32 *cnt_, u32 seed) {
  N = n; E = e; INH_SLOTS = inhSlots; INH_DELAY = inhDelay; ROLE_SLOTS = roleSlots; NROLES = nroles; NOISE_PER_STEP = noisePerStep;
  DECAY = decay; THRESH = thresh; REFRACT = refract; NOISE_KICK = noiseKick;
  v = v_; refr = refr_; baseline = baseline_; indptr = indptr_; colidx = colidx_; w = w_;
  inhVal = inhVal_; inhIdx = inhIdx_; inhCnt = inhCnt_; spiked = spiked_; lastSpikeMs = lastSpikeMs_; roleOf = roleOf_; cnt = cnt_;
  rng = seed; slot = 0; ms = 0; totalSpikes = 0; cellIdx = 0; cellAmt = 0; cellN = 0;
}
__attribute__((export_name("setCellDrive"))) void setCellDrive(i32 *idx, float *amt, i32 n) { cellIdx = idx; cellAmt = amt; cellN = n; }
__attribute__((export_name("setState"))) void setState(u32 seed, i32 slot_, double ms_, i32 total) { rng = seed; slot = slot_; ms = ms_; totalSpikes = total; }
__attribute__((export_name("getRng"))) u32 getRng(void) { return rng; }
__attribute__((export_name("getSlot"))) i32 getSlot(void) { return slot; }
__attribute__((export_name("getMs"))) double getMs(void) { return ms; }
__attribute__((export_name("getTotalSpikes"))) i32 getTotalSpikes(void) { return totalSpikes; }

static inline u32 xorshift(void) { u32 s = rng; s ^= s << 13; s ^= s >> 17; s ^= s << 5; rng = s; return s; }

// One millisecond with `nactive` uniform pool drives (idx lists concatenated: activeIdx, per-pool
// counts activeLen, per-pool amounts activeAmt as doubles, as brain.ts adds them). Returns the spike count.
__attribute__((export_name("step")))
i32 step(i32 *activeIdx, i32 *activeLen, double *activeAmt, i32 nactive) {
  // 1. delayed inhibition for this slot
  float *q = inhVal + (long)slot * N; i32 *qi = inhIdx + (long)slot * N; i32 qn = inhCnt[slot];
  for (i32 k = 0; k < qn; k++) { i32 j = qi[k]; double nv = (double)v[j] + (double)q[j]; v[j] = (float)(nv < -2 ? -2 : nv); q[j] = 0; }
  inhCnt[slot] = 0;
  // 2. noise: JS does `v[i] += NOISE_KICK` on a Float32Array (double add, one rounding)
  for (i32 k = 0; k < NOISE_PER_STEP; k++) { i32 i = (i32)(xorshift() % (u32)N); v[i] = (float)((double)v[i] + NOISE_KICK); }
  // 3. drives
  i32 off = 0;
  for (i32 a = 0; a < nactive; a++) { double amt = activeAmt[a]; i32 len = activeLen[a]; for (i32 k = 0; k < len; k++) { i32 i = activeIdx[off + k]; v[i] = (float)((double)v[i] + amt); } off += len; }
  if (cellIdx) for (i32 k = 0; k < cellN; k++) { i32 i = cellIdx[k]; v[i] = (float)((double)v[i] + (double)cellAmt[k]); }
  // 4. leak, threshold, reset
  i32 ns = 0;
  for (i32 i = 0; i < N; i++) {
    u8 r = refr[i];
    if (r) { refr[i] = r - 1; v[i] = (float)((double)v[i] * DECAY); continue; }
    double vi = (double)v[i] * DECAY + (double)baseline[i];
    if (vi >= THRESH) { v[i] = 0; refr[i] = (u8)REFRACT; spiked[ns++] = i; }
    else v[i] = (float)vi;
  }
  // 5. propagate
  i32 is = (slot + INH_DELAY) % INH_SLOTS; float *iq = inhVal + (long)is * N; i32 *iqi = inhIdx + (long)is * N; i32 ic = inhCnt[is];
  for (i32 s = 0; s < ns; s++) {
    i32 i = spiked[s]; u32 a = indptr[i], b = indptr[i + 1];
    for (u32 k = a; k < b; k++) {
      i32 j = colidx[k]; float x = w[k];
      if (x >= 0) { double nv = (double)v[j] + (double)x; v[j] = (float)(nv < -2 ? -2 : nv); }
      else { if (iq[j] == 0) iqi[ic++] = j; iq[j] = (float)((double)iq[j] + (double)x); }
    }
  }
  inhCnt[is] = ic;
  // 6. bookkeeping
  for (i32 r = 0; r < NROLES; r++) cnt[r] = 0;
  for (i32 s = 0; s < ns; s++) {
    i32 i = spiked[s]; lastSpikeMs[i] = ms;
    for (i32 sl = i * ROLE_SLOTS, e = sl + ROLE_SLOTS; sl < e; sl++) { i32 r = roleOf[sl]; if (r < 0) break; cnt[r]++; }
  }
  totalSpikes += ns; ms += 1; slot = (slot + 1) % INH_SLOTS;
  return ns;
}
