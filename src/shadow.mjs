// shadow.mjs — shadow-labeling harness. For each sampled payload:
//   1. run the 4-bit edge model in SHADOW mode (no abort) → extraction + guard signal
//   2. adjudicate the SAME payload with the heavy model (ground truth)
//   3. judge whether the edge extraction matched → (signal, correct) pair
// Feeds the pairs to the PAVA calibrator to produce the reported operating point.
//
// Async and non-blocking by construction: it consumes a *sample*, never the live
// queue, and each payload is independent. Inject edgeFetch / heavyOpts for tests.
'use strict';
import { admit } from './schemas.mjs';
import { extractStructured } from './constrained.mjs';
import { extractHeavy } from './heavy.mjs';
import { analyzeCalibration } from './calibration.mjs';

const NUM_TOL = 0.05;
/** Did the edge extraction match the heavy (ground-truth) extraction on the
 *  fields that decide the outcome? Numbers within tolerance; arrays set-equal. */
function sameField(a, b) {
  if (Array.isArray(a) || Array.isArray(b)) {
    const norm = (x) => String(x).trim().toLowerCase().replace(/\s+/g, ' ');
    const sa = new Set((a ?? []).map(norm)), sb = new Set((b ?? []).map(norm));
    return sa.size === sb.size && ![...sa].some((x) => !sb.has(x));
  }
  if (typeof a === 'number' && typeof b === 'number') return Math.abs(a - b) <= NUM_TOL;
  return (a ?? null) === (b ?? null);
}
/** Keys on which two extractions disagree (numbers within tolerance; arrays set-equal,
 *  case/whitespace-insensitive; null ≡ absent). */
export function fieldDiff(edge, heavy, keys) { return keys.filter((k) => !sameField(edge?.[k], heavy?.[k])); }
export function judge(edge, heavy, keys) { return fieldDiff(edge, heavy, keys).length === 0; }

/**
 * Produce (signal, correct) pairs for a batch of payloads.
 * @param payloads       array of raw ingestion payloads (the tapped sample)
 * @param edgeFetch      fetch impl for the edge vLLM (injectable)
 * @param heavyOpts      options passed to extractHeavy (provider/model/clientImpl)
 * @param vllmUrl        edge endpoint
 */
export async function shadowLabel({ payloads, edgeFetch = fetch, heavyOpts = {}, edgeOpts = {}, vllmUrl, onProgress, concurrency = 1, trace = null } = {}) {
  const pairs = [];
  const skipped = { not_admitted: 0, edge_failed: 0, heavy_failed: 0, no_signal: 0 };
  let next = 0, finished = 0;
  const one = async (payload) => {
    const gate = admit(payload);
    if (!gate.ok) { skipped.not_admitted++; return; }       // malformed → not a labeling sample
    // edge shadow pass and heavy adjudication concurrently on the same payload.
    // edgeOpts carries e.g. a long deadlineMs: the production 800 ms abort would discard
    // every sample when the edge model is reached over a tunnel or generates slowly.
    const [edge, heavy] = await Promise.all([
      extractStructured({ endpoint: vllmUrl, payload, schemaEntry: gate.entry, shadow: true, fetchImpl: edgeFetch, ...edgeOpts }),
      extractHeavy(payload, gate.entry, heavyOpts).then((d) => ({ status: 'done', data: d }), (e) => ({ status: 'error', error: String(e) })),
    ]);
    if (edge.status !== 'done') { skipped.edge_failed++; skipped.last = `edge: ${edge.reason ?? edge.status} ${edge.error ?? ''}`; return; }
    if (heavy.status !== 'done') { skipped.heavy_failed++; skipped.last = `heavy: ${heavy.error}`; return; }
    if (edge.signal == null) { skipped.no_signal++; return; }
    const keys = gate.entry.judgeKeys;
    const pair = { req_id: payload.req_id, kind: gate.entry.key, signal: edge.signal, correct: judge(edge.data, heavy.data, keys) };
    // Constructed ground truth (synthetic corpus): score BOTH models against it. `correct`
    // stays the judge-relative label; correct_truth / judge_truth are the truth-relative ones.
    const truth = payload.metadata?.synthetic_truth;
    if (truth) {
      pair.correct_truth = judge(edge.data, truth, keys);
      pair.judge_truth = judge(heavy.data, truth, keys);
      pair.style = payload.metadata.style;
      pair.diff_edge = fieldDiff(edge.data, truth, keys);      // which fields the edge model got wrong
      pair.diff_judge = fieldDiff(heavy.data, truth, keys);    // which fields the judge got wrong
    }
    if (edge.surprisals) pair.s = edge.surprisals.map((x) => Math.round(x * 1e4) / 1e4);   // full value-token trace
    if (trace) trace.push({ req_id: payload.req_id, kind: gate.entry.key, text: payload.raw_text ?? payload.raw_telemetry,
      truth: truth ?? null, edge: edge.raw, judge: Object.fromEntries(keys.map((k) => [k, heavy.data[k]])), value_tokens: edge.valueTokenTexts });
    pairs.push(pair);
  };
  const worker = async () => {
    while (next < payloads.length) {
      const p = payloads[next++];
      await one(p);
      finished++;
      if (onProgress && (finished % 25 === 0 || finished === payloads.length)) onProgress({ i: finished, n: payloads.length, labeled: pairs.length, skipped: { ...skipped } });
    }
  };
  await Promise.all(Array.from({ length: Math.max(1, concurrency) }, worker));
  pairs.sort((a, b) => String(a.req_id).localeCompare(String(b.req_id)));   // deterministic artifact order
  pairs.skipped = skipped;
  return pairs;
}

/** One-shot: label a sample and return the calibration operating point. */
export async function shadowCalibrate(opts, targetEscalationRate = 0.2) {
  const pairs = await shadowLabel(opts);
  return { pairs, calibration: analyzeCalibration(pairs, targetEscalationRate) };
}
