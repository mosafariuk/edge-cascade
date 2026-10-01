// verify.mjs — executable proof that the batching math and the entropy guard
// behave correctly, using MOCKS (no Redis / vLLM / ONNX needed).
import assert from 'node:assert/strict';
import { RateEstimator, adaptiveBatchSize, shannonEntropy } from '../src/cascade-lib.mjs';
import { guardedGenerate } from '../src/entropy-guard.mjs';

let pass = 0;
const ok = (name) => { console.log(`  ✓ ${name}`); pass++; };

// ── 1. adaptiveBatchSize formula ─────────────────────────────────────────────
console.log('1. adaptiveBatchSize B*(t)=min(Bmax,max(1,ceil(λ·Wmax)))');
assert.equal(adaptiveBatchSize(10,   0.019, 32), 1);   // low load → batch of 1
assert.equal(adaptiveBatchSize(1000, 0.019, 32), 19);  // 1000·0.019=19
assert.equal(adaptiveBatchSize(5000, 0.019, 32), 32);  // clipped at Bmax
assert.equal(adaptiveBatchSize(0,    0.019, 32), 1);   // floor at 1
ok('formula matches derivation at low/mid/saturated load');

// ── 2. RateEstimator EWMA (deterministic _tick, no timers) ───────────────────
console.log('2. RateEstimator EWMA');
const re = new RateEstimator({ alpha: 0.5, tickMs: 250 }); // 250ms window
re.record(250); const l1 = re._tick();   // inst = 250/0.25 = 1000 ; λ=0.5·1000=500
assert.equal(l1, 500);
re.record(250); const l2 = re._tick();   // inst=1000 ; λ=0.5·1000+0.5·500=750
assert.equal(l2, 750);
ok(`converges toward true rate (λ: ${l1} → ${l2} req/s)`);

// ── 3. shannonEntropy bounds ─────────────────────────────────────────────────
console.log('3. shannonEntropy (nats)');
const ln = Math.log;
const uniform5 = { a: ln(0.2), b: ln(0.2), c: ln(0.2), d: ln(0.2), e: ln(0.2) };
const peaked   = { a: ln(0.96), b: ln(0.01), c: ln(0.01), d: ln(0.01), e: ln(0.01) };
const Hu = shannonEntropy(uniform5), Hp = shannonEntropy(peaked);
assert.ok(Math.abs(Hu - ln(5)) < 1e-9, `uniform H should be ln5=${ln(5).toFixed(3)}, got ${Hu}`);
assert.ok(Hp < 0.3, `peaked H should be small, got ${Hp}`);
ok(`H(uniform-5)=${Hu.toFixed(3)}≈ln5 ; H(peaked)=${Hp.toFixed(3)}≪`);

// ── mock vLLM SSE stream builder ─────────────────────────────────────────────
function sseBody(frames) {                     // async-iterable of Uint8Array
  const enc = new TextEncoder();
  return (async function* () {
    for (const f of frames) { yield enc.encode(`data: ${JSON.stringify(f)}\n\n`); }
    yield enc.encode('data: [DONE]\n\n');
  })();
}
const tokenFrame = (text, dist) => ({ choices: [{ text, logprobs: { top_logprobs: [dist] } }] });
const UNIF = { a: ln(0.2), b: ln(0.2), c: ln(0.2), d: ln(0.2), e: ln(0.2) }; // H=ln5≈1.61
const HIGH = { a: ln(0.11),b: ln(0.11),c: ln(0.11),d: ln(0.11),e: ln(0.11),
               f: ln(0.11),g: ln(0.11),h: ln(0.12),i: ln(0.11),j: ln(0.10) }; // H≈ln10≈2.30
const SHARP = { a: ln(0.97), b: ln(0.01), c: ln(0.01), d: ln(0.01) };         // H≈0.15

const mockFetch = (frames) => async () => ({ ok: true, status: 200, body: sseBody(frames) });
async function runGuard(frames, opts) {
  const gen = guardedGenerate({ prompt: 'x', fetchImpl: mockFetch(frames), ...opts });
  const tokens = [];
  while (true) { const n = await gen.next(); if (n.done) return { verdict: n.value, tokens }; tokens.push(n.value); }
}

// ── 4. guard ESCALATES on sustained high entropy ─────────────────────────────
console.log('4. entropy guard: high-entropy stream → escalate');
{
  const frames = Array.from({ length: 20 }, (_, i) => tokenFrame(`t${i}`, HIGH));
  const { verdict, tokens } = await runGuard(frames, { k: 5, Hmax: 2.5 });
  // wait — HIGH is H≈2.30 < 2.5, so it should NOT abort on entropy. Verify that:
  assert.equal(verdict.status, 'done', `H≈2.30<Hmax2.5 must NOT abort, got ${verdict.reason}`);
  ok(`H≈2.30 < Hmax2.5 → completes (no false abort); ${tokens.length} tokens streamed`);
}
{
  const VHIGH = {}; for (let i = 0; i < 20; i++) VHIGH[`k${i}`] = ln(0.05); // H≈ln20≈3.0
  const frames = Array.from({ length: 20 }, (_, i) => tokenFrame(`t${i}`, VHIGH));
  const { verdict, tokens } = await runGuard(frames, { k: 5, Hmax: 2.5 });
  assert.equal(verdict.status, 'escalate');
  assert.equal(verdict.reason, 'entropy');
  assert.equal(tokens.length, 5, 'must abort exactly at the k-th token');
  ok(`H≈3.0 > Hmax2.5 → escalate:entropy, aborted at token k=${tokens.length} (meanH=${verdict.meanH.toFixed(3)})`);
}

// ── 5. guard COMPLETES on confident (low-entropy) stream ─────────────────────
console.log('5. entropy guard: confident stream → done');
{
  const frames = Array.from({ length: 12 }, (_, i) => tokenFrame(` w${i}`, SHARP));
  const { verdict, tokens } = await runGuard(frames, { k: 5, Hmax: 2.5 });
  assert.equal(verdict.status, 'done');
  assert.ok(verdict.text.includes('w0'));
  ok(`low-entropy → done, full text streamed (${tokens.length} tokens)`);
}

// ── 6. repetition guard ──────────────────────────────────────────────────────
console.log('6. repetition guard: looping stream → escalate:repetition');
{
  const loop = ['A','B','C','D','E','F','G','H'];
  const seq = [...loop, ...loop];  // 8-token block repeated → triggers repWindow=8
  const frames = seq.map((t) => tokenFrame(t, SHARP)); // low entropy so only rep can fire
  const { verdict } = await runGuard(frames, { k: 5, Hmax: 5.0, repWindow: 8 });
  assert.equal(verdict.status, 'escalate');
  assert.equal(verdict.reason, 'repetition');
  ok('exact 8-gram repetition detected → escalate:repetition');
}

console.log(`\nALL ${pass} ASSERTIONS PASSED ✅`);
