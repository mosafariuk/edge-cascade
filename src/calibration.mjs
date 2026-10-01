// calibration.mjs — turn shadow-labeled (signal, correct) pairs into a calibrated
// threshold and the operating-point metrics the paper reports:
//   - escalation_rate at the chosen threshold
//   - local_precision  = P(correct | kept local)   ← the "verified precision floor"
//   - error_recall     = P(escalated | edge wrong) ← fraction of edge errors caught
// Plus an isotonic map g: signal → P(correct) (PAVA, O(n)) and its ECE.
//
// signal = mean value-token surprisal (nats); HIGHER = less confident → escalate.
'use strict';

/** Pool Adjacent Violators — monotone LEAST-squares fit. Here signal↑ ⇒ correctness↓,
 *  so we fit a NON-INCREASING map by negating x order (fit non-decreasing on -signal). */
export function pavaDecreasing(signal, correct) {
  const pairs = signal.map((s, i) => [-s, correct[i]]).sort((a, b) => a[0] - b[0]); // ascending in -signal
  const blocks = [];
  for (const [x, y] of pairs) {
    blocks.push([y, 1, x]);
    while (blocks.length >= 2 && blocks[blocks.length - 2][0] / blocks[blocks.length - 2][1] > blocks[blocks.length - 1][0] / blocks[blocks.length - 1][1]) {
      const [s2, w2] = blocks.pop(); const [s1, w1, l1] = blocks.pop();
      blocks.push([s1 + s2, w1 + w2, l1]);
    }
  }
  const knotsNegSig = blocks.map((b) => b[2]);       // in -signal space
  const knotsP = blocks.map((b) => b[0] / b[1]);     // P(correct)
  return { knotsNegSig, knotsP };
}

/** P(correct | signal) from the fitted isotonic map. */
export function isotonicPredict(model, s) {
  const x = -s;
  let lo = 0, hi = model.knotsNegSig.length - 1, idx = 0;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (model.knotsNegSig[m] <= x) { idx = m; lo = m + 1; } else hi = m - 1; }
  return model.knotsP[idx];
}

/** Expected Calibration Error of a probability array vs binary labels. */
export function ece(probs, labels, M = 10) {
  const bins = Array.from({ length: M }, () => []);
  probs.forEach((p, i) => bins[Math.min(M - 1, Math.floor(p * M))].push([p, labels[i]]));
  const n = probs.length; let e = 0;
  for (const b of bins) { if (!b.length) continue; const conf = b.reduce((a, [p]) => a + p, 0) / b.length; const acc = b.reduce((a, [, y]) => a + y, 0) / b.length; e += (b.length / n) * Math.abs(acc - conf); }
  return e;
}

// Threshold that escalates the top `rate` fraction under the guard's strict `>`.
// Interpolates to the MIDPOINT between the boundary points so tie-valued clusters
// at the cut are escalated cleanly (a plain quantile returns the boundary datapoint,
// which strict `>` would then miss).
function cutThreshold(xs, rate) {
  const s = xs.slice().sort((a, b) => a - b);
  const n = s.length, cut = Math.floor((1 - rate) * n);
  if (cut <= 0) return s[0] - 1;              // escalate all
  if (cut >= n) return s[n - 1] + 1;          // escalate none
  return (s[cut - 1] + s[cut]) / 2;
}

/** Wilson score 95% CI for a binomial proportion k/n (reviewers ask for these). */
export function wilson(k, n, z = 1.959964) {
  if (!n) return [null, null];
  const p = k / n, z2 = z * z, d = 1 + z2 / n;
  const c = (p + z2 / (2 * n)) / d, h = z * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / d;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}

// deterministic [0,1) hash of a key → reproducible train/test split
// FNV-1a + murmur3 finalizer. Plain FNV does not avalanche on sequential ids (syn_0001,
// syn_0002, …): on the first live run it split one workload 201/49 and the other 128/122.
function fnv01(s) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < String(s).length; i++) { h ^= String(s).charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; }
  h ^= h >>> 16; h = Math.imul(h, 0x85ebca6b) >>> 0; h ^= h >>> 13; h = Math.imul(h, 0xc2b2ae35) >>> 0; h ^= h >>> 16;
  return (h >>> 0) / 2 ** 32;
}

/**
 * Full analysis with Wilson CIs and an optional HELD-OUT split (for out-of-sample
 * reporting, per Threats §iii). Threshold + isotonic map are FIT on train; the
 * operating point is EVALUATED on the held-out test set when `holdout` > 0.
 * pairs: [{signal, correct, req_id?}]. Default holdout=0 → in-sample (backwards-compatible).
 */
export function analyzeCalibration(pairs, targetEscalationRate = 0.2, holdout = 0) {
  const usable = pairs.filter((p) => p.signal != null);
  if (usable.length === 0) return { n: 0 };

  let train = usable, evalSet = usable;
  if (holdout > 0) {
    // stratified by workload (`kind`) so both splits keep the population mix; within a
    // stratum the held-out records are the `holdout` fraction with the smallest id hash
    train = []; evalSet = [];
    const strata = new Map();
    usable.forEach((p, i) => { const k = p.kind ?? ''; if (!strata.has(k)) strata.set(k, []); strata.get(k).push([fnv01(p.req_id ?? i), p]); });
    for (const rows of strata.values()) {
      rows.sort((a, b) => a[0] - b[0]);
      const nEval = Math.round(holdout * rows.length);
      rows.forEach(([, p], i) => (i < nEval ? evalSet : train).push(p));
    }
    if (!train.length || !evalSet.length) { train = usable; evalSet = usable; holdout = 0; }
  }

  // FIT on train
  const trSignal = train.map((p) => p.signal), trCorrect = train.map((p) => (p.correct ? 1 : 0));
  const threshold = cutThreshold(trSignal, targetEscalationRate);
  const model = pavaDecreasing(trSignal, trCorrect);

  // EVALUATE on eval set
  let escalated = 0, keptCorrect = 0, kept = 0, errors = 0, errorsCaught = 0, corr = 0;
  for (const p of evalSet) {
    if (p.correct) corr++;
    const isErr = !p.correct; if (isErr) errors++;
    if (p.signal > threshold) { escalated++; if (isErr) errorsCaught++; }
    else { kept++; if (p.correct) keptCorrect++; }
  }
  const nE = evalSet.length;
  const calibrated = evalSet.map((p) => isotonicPredict(model, p.signal));

  return {
    n: usable.length, n_train: train.length, n_eval: nE, holdout,
    baseline_edge_accuracy: corr / nE, baseline_ci: wilson(corr, nE),
    threshold, target_escalation_rate: targetEscalationRate,
    escalation_rate: escalated / nE, escalation_ci: wilson(escalated, nE),
    local_precision: kept ? keptCorrect / kept : null, precision_ci: wilson(keptCorrect, kept),
    error_recall: errors ? errorsCaught / errors : null, recall_ci: wilson(errorsCaught, errors),
    n_errors: errors,
    ece_calibrated: ece(calibrated, evalSet.map((p) => (p.correct ? 1 : 0))),
    model,
  };
}
