// cascade-lib.mjs — pure, dependency-free logic (unit-testable, no I/O)
'use strict';

/**
 * Real-time arrival-rate estimator (lambda_obs).
 * Non-blocking: a single setInterval tick does O(1) work and is unref'd so it
 * never keeps the process alive on its own. The hot path only does record()++.
 */
export class RateEstimator {
  constructor({ alpha = 0.3, tickMs = 250 } = {}) {
    this.alpha = alpha;          // EWMA smoothing (higher = more reactive)
    this.tickMs = tickMs;        // measurement window
    this.count = 0;              // arrivals since last tick (hot-path counter)
    this.lambda = 0;             // smoothed req/s
    this._timer = null;
  }
  record(n = 1) { this.count += n; }               // called per consumed message
  _tick() {                                        // exposed for deterministic tests
    const inst = this.count / (this.tickMs / 1000); // instantaneous req/s this window
    this.lambda = this.alpha * inst + (1 - this.alpha) * this.lambda;
    this.count = 0;
    return this.lambda;
  }
  start() {
    this._timer = setInterval(() => this._tick(), this.tickMs);
    if (this._timer.unref) this._timer.unref();    // don't hold the event loop open
    return this;
  }
  stop() { if (this._timer) clearInterval(this._timer); this._timer = null; }
  get rps() { return this.lambda; }
}

/**
 * Adaptive batch size B*(t) = min(Bmax, max(1, ceil(lambda * Wmax))).
 * @param lambda  observed req/s
 * @param WmaxSec collection-window budget in SECONDS (SLA_p99 - worst batch service)
 * @param Bmax    hard cap on batch size
 */
export function adaptiveBatchSize(lambda, WmaxSec, Bmax) {
  return Math.min(Bmax, Math.max(1, Math.ceil(lambda * WmaxSec)));
}

/** Shannon entropy (nats) of a vLLM top_logprobs map {token: logprob}.
 *  Renormalized over the returned top-k, so H is a top-k-truncated estimate;
 *  calibrate H_max to the same `logprobs` k you request from vLLM. */
export function shannonEntropy(topLogprobs) {
  const lps = Object.values(topLogprobs);
  if (lps.length === 0) return 0;
  const ps = lps.map(Math.exp);
  const Z = ps.reduce((a, b) => a + b, 0) || 1;
  let H = 0;
  for (const p0 of ps) { const p = p0 / Z; if (p > 0) H -= p * Math.log(p); }
  return H;
}
