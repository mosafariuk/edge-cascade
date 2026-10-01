// verify-shadow.mjs — proves the shadow-labeling harness MECHANICS with mocks:
// signal capture, judge, PAVA isotonic calibration, and the operating-point metrics.
// NOTE: numbers here are from a controlled synthetic construction — they prove the
// harness computes correctly, NOT a production precision floor (that needs a real run).
import assert from 'node:assert/strict';
import { SCHEMAS } from '../src/schemas.mjs';
import { extractStructured } from '../src/constrained.mjs';
import { judge, shadowLabel } from '../src/shadow.mjs';
import { pavaDecreasing, isotonicPredict, analyzeCalibration, wilson } from '../src/calibration.mjs';

let pass = 0; const ok = (n) => { console.log(`  ✓ ${n}`); pass++; };

// mock edge stream: JSON tokens with per-value logprob controlling the signal
function jsonTokens(json, valueLogprob) {
  const toks = []; let run = '';
  const flush = () => { if (run) { toks.push({ text: run, lp: valueLogprob }); run = ''; } };
  for (const c of json) { if ('{}[],:"'.includes(c)) { flush(); toks.push({ text: c, lp: -0.001 }); } else run += c; }
  flush(); return toks;
}
const mockEdge = (json, lp) => async () => ({ ok: true, status: 200, body: (async function* () {
  const enc = new TextEncoder();
  for (const t of jsonTokens(json, lp)) yield enc.encode(`data: ${JSON.stringify({ choices: [{ text: t.text, logprobs: { token_logprobs: [t.lp], top_logprobs: [{ [t.text]: t.lp }] } }] })}\n\n`);
  yield enc.encode('data: [DONE]\n\n');
})() });

// ── 1. shadow mode captures signal without aborting ──────────────────────────
console.log('1. shadow mode: signal captured, no abort');
{
  const json = '{"current_value_gbp":18200,"previous_value_gbp":14500,"cohort_avg_increase_pct":12,"effective_date_raw":null,"assets":["x"]}';
  const r = await extractStructured({ payload: { raw_text: 'x', source: 's' }, schemaEntry: SCHEMAS.property_audit,
    fetchImpl: mockEdge(json, -4.0), shadow: true, k: 3, maxSurprisal: 3.0 });
  assert.equal(r.status, 'done', 'shadow never escalates');           // would have escalated normally (surprisal 4>3)
  assert.ok(Math.abs(r.signal - 4.0) < 1e-6, `signal≈4.0, got ${r.signal}`);
  ok(`signal=${r.signal.toFixed(2)} captured; status=done (guard suppressed in shadow)`);
}

// ── 2. judge: primitive match within tolerance ───────────────────────────────
console.log('2. judge (ground-truth comparison)');
{
  const keys = SCHEMAS.property_audit.judgeKeys;
  assert.equal(judge({ current_value_gbp: 18200, previous_value_gbp: 14500, cohort_avg_increase_pct: 12, assets: ['a'] },
                      { current_value_gbp: 18200, previous_value_gbp: 14500, cohort_avg_increase_pct: 12.02, assets: ['a'] }, keys), true);
  assert.equal(judge({ current_value_gbp: 18200, assets: ['a'] }, { current_value_gbp: 1820, assets: ['a'] }, keys), false); // 10× off
  assert.equal(judge({ current_value_gbp: 1, assets: ['a', 'b'] }, { current_value_gbp: 1, assets: ['b'] }, keys), false);   // asset mismatch
  ok('numeric tolerance + array set-equality + mismatch detection');
}

// ── 3. PAVA isotonic: higher surprisal → lower P(correct), monotone ──────────
console.log('3. PAVA isotonic map (signal↑ ⇒ P(correct)↓)');
{
  // synthetic: low signal mostly correct, high signal mostly wrong
  const sig = [], cor = [];
  for (let i = 0; i < 400; i++) { const s = i / 100; sig.push(s); cor.push(Math.random ? 0 : 0); }
  // deterministic construction (no RNG): correctness = signal < 2 ? 1 : 0 with a little overlap
  sig.length = 0; cor.length = 0;
  for (let i = 0; i < 400; i++) { const s = (i % 40) / 10; sig.push(s); cor.push(s < 2.0 ? 1 : 0); }
  const model = pavaDecreasing(sig, cor);
  assert.ok(isotonicPredict(model, 0.5) > isotonicPredict(model, 3.5), 'P(correct) decreases with surprisal');
  ok(`g(0.5)=${isotonicPredict(model, 0.5).toFixed(2)} > g(3.5)=${isotonicPredict(model, 3.5).toFixed(2)}`);
}

// ── 4. end-to-end shadowLabel + analyzeCalibration (mock edge + mock heavy) ──
console.log('4. shadowLabel → analyzeCalibration operating point');
{
  // 20 payloads; edge is WRONG (value off by 10×) on the high-surprisal ones
  const payloads = Array.from({ length: 20 }, (_, i) => ({ req_id: `p${i}`, source: 's', raw_text: `rec ${i}`,
    metadata: { ref_year: 2026 }, _hard: i >= 15 }));
  // edge mock: hard ones get low-confidence + wrong value (1820 not 18200); easy ones confident + right
  const edgeFetch = (url, opts) => {
    const p = JSON.parse(opts.body).prompt;
    const hard = /rec 1[5-9]/.test(p);
    const json = hard ? '{"current_value_gbp":1820,"previous_value_gbp":null,"cohort_avg_increase_pct":null,"effective_date_raw":null,"assets":[]}' : '{"current_value_gbp":18200,"previous_value_gbp":null,"cohort_avg_increase_pct":null,"effective_date_raw":null,"assets":[]}';
    return mockEdge(json, hard ? -4.5 : -0.1)();
  };
  // heavy mock = ground truth: always 18200
  const heavyClient = { messages: { create: async () => ({ stop_reason: 'end_turn', content: [{ type: 'text', text: '{"current_value_gbp":18200,"previous_value_gbp":null,"cohort_avg_increase_pct":null,"effective_date_raw":null,"assets":[]}' }] }) } };
  const pairs = await shadowLabel({ payloads, edgeFetch, heavyOpts: { provider: 'anthropic', clientImpl: heavyClient }, vllmUrl: 'http://mock' });
  assert.equal(pairs.length, 20);
  const c = analyzeCalibration(pairs, 0.25);           // target ~25% escalation (the 5 hard ones)
  assert.ok(c.local_precision >= 0.99, `kept-local should be ~all correct, got ${c.local_precision}`);
  assert.ok(c.error_recall >= 0.99, `all 5 edge errors should be escalated, got ${c.error_recall}`);
  ok(`escalation=${(c.escalation_rate*100).toFixed(0)}%, precision floor=${(c.local_precision*100).toFixed(0)}%, error recall=${(c.error_recall*100).toFixed(0)}% (harness computes correctly)`);
}

// ── 5. Wilson CI matches the paper's reported intervals ──────────────────────
console.log('5. Wilson 95% CI (matches paper Table III)');
{
  const r = (x) => Math.round(x * 1000) / 10;
  assert.deepEqual(wilson(389, 396).map(r), [96.4, 99.1]);   // local precision
  assert.deepEqual(wilson(78, 85).map(r), [84.0, 96.0]);     // edge-error recall (wide)
  assert.deepEqual(wilson(104, 500).map(r), [17.5, 24.6]);   // escalation rate
  ok('precision [96.4,99.1], recall [84.0,96.0], escalation [17.5,24.6] — reproduces the table');
}

// ── 6. held-out split evaluates out-of-sample; CIs attached ──────────────────
console.log('6. held-out train/test split');
{
  // 200 pairs: signal<2 correct, signal>2 wrong (clean separation), req_id keys for split
  const pairs = Array.from({ length: 200 }, (_, i) => ({ req_id: `r${i}`, signal: (i % 20) / 5, correct: (i % 20) / 5 < 2 }));
  const c = analyzeCalibration(pairs, 0.25, 0.3);            // 30% held out
  assert.ok(c.n_eval > 0 && c.n_train > 0 && c.n_train + c.n_eval === 200, 'disjoint split covering all pairs');
  assert.ok(c.n_eval >= 40 && c.n_eval <= 80, `~30% held out, got ${c.n_eval}`);
  assert.ok(c.precision_ci[0] <= c.local_precision && c.local_precision <= c.precision_ci[1], 'point estimate inside its CI');
  ok(`train=${c.n_train} eval=${c.n_eval}; precision=${(c.local_precision*100).toFixed(0)}% CI [${(c.precision_ci[0]*100).toFixed(0)},${(c.precision_ci[1]*100).toFixed(0)}]`);
}

console.log(`\nALL ${pass} SHADOW-HARNESS ASSERTIONS PASSED ✅`);
