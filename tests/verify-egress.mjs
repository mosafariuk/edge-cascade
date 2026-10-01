// verify-egress.mjs — executable proof of the egress layer (mock sinks, no I/O).
import assert from 'node:assert/strict';
import { EgressBuffer, pgvectorSink, redisStreamSink, toVectorLiteral, assertVectorDim } from '../src/egress.mjs';

let pass = 0;
const ok = (n) => { console.log(`  ✓ ${n}`); pass++; };
const tick = () => new Promise((r) => setImmediate(r));       // let microtasks drain

// ── 1. size-triggered flush + correct batching ───────────────────────────────
console.log('1. EgressBuffer flushes at maxBatch and passes exact records');
{
  const seen = [];
  const eb = new EgressBuffer({ maxBatch: 3, maxInflight: 4, sink: async (b) => { seen.push(b.map((r) => r.id)); } });
  eb.add({ id: 1 }); eb.add({ id: 2 });
  assert.equal(seen.length, 0, 'no flush below maxBatch');
  eb.add({ id: 3 });                                          // hits maxBatch → flush
  await tick();
  assert.deepEqual(seen, [[1, 2, 3]]);
  assert.equal(eb.committed, 3);
  ok('flush at size=3, batch contents exact, committed counter correct');
}

// ── 2. drain() flushes the tail below maxBatch ───────────────────────────────
console.log('2. drain flushes remaining partial batch');
{
  const seen = [];
  const eb = new EgressBuffer({ maxBatch: 100, sink: async (b) => { seen.push(b.length); } });
  eb.add({ id: 'a' }); eb.add({ id: 'b' });
  await eb.drain();
  assert.deepEqual(seen, [2]);
  ok('partial batch of 2 committed on drain');
}

// ── 3. backpressure: paused=true while flush slots are full ───────────────────
console.log('3. backpressure signal under a slow sink');
{
  let release;
  const gate = new Promise((r) => { release = r; });
  const eb = new EgressBuffer({ maxBatch: 1, maxInflight: 2, sink: async () => { await gate; } });
  assert.equal(eb.paused, false);
  eb.add({ id: 1 });                                          // inflight → 1
  eb.add({ id: 2 });                                          // inflight → 2 (full)
  await tick();
  assert.equal(eb.paused, true, 'must signal paused when inflight == maxInflight');
  release();                                                  // let sinks finish
  await tick(); await tick();
  assert.equal(eb.paused, false, 'un-pauses after flushes drain');
  ok('paused asserts under load and clears when sinks complete');
}

// ── 4. ack-after-commit ordering (delivery contract) ─────────────────────────
console.log('4. source ack happens AFTER downstream commit');
{
  const order = [];
  const ackInputs = async (batch) => { order.push('ack:' + batch.map((r) => r.id).join(',')); };
  const fakePg = { query: async () => { order.push('commit'); } };
  const sink = pgvectorSink(fakePg, 'documents', ackInputs);
  await sink([{ id: 7, vec: [0.1, 0.2], output: { text: 'hi', meta: {} } }]);
  assert.deepEqual(order, ['commit', 'ack:7'], 'commit must precede ack');
  ok('commit precedes ack → no input lost if crash before commit');
}

// ── 5. pgvector SQL shape (params + upsert + vector literal) ─────────────────
console.log('5. pgvector sink builds a single idempotent multi-row upsert');
{
  let captured;
  const fakePg = { query: async (sql, params) => { captured = { sql, params }; } };
  const sink = pgvectorSink(fakePg, 'documents', null);
  await sink([
    { id: 'x', vec: [0.1, 0.2], output: { text: 'A', meta: { k: 1 } } },
    { id: 'y', vec: [0.3, 0.4], output: { text: 'B', meta: {} } },
  ]);
  assert.match(captured.sql, /INSERT INTO documents/);
  assert.match(captured.sql, /ON CONFLICT \(id\) DO UPDATE/);    // idempotent
  assert.match(captured.sql, /\$1.*\$5.*\$8/s);                  // 2 rows × 4 params = 8
  assert.equal(captured.params.length, 8);
  assert.equal(captured.params[2], '[0.1,0.2]');                // vector literal
  ok('one INSERT, upsert clause present, 8 bound params, vector literal correct');
}
assert.equal(toVectorLiteral(Float32Array.from([1, 2, 3])), '[1,2,3]');

// ── 6. redis egress pipelines one round-trip per batch ───────────────────────
console.log('6. redis stream sink uses a single pipelined round-trip');
{
  let xadds = 0, execs = 0;
  const pipe = { xadd() { xadds++; return this; }, exec: async () => { execs++; return [[null, '1-0'], [null, '2-0']]; } };
  const fakeRedis = { pipeline: () => pipe };
  const sink = redisStreamSink(fakeRedis, 'out', null);
  await sink([{ id: 1, output: {} }, { id: 2, output: {} }]);
  assert.equal(xadds, 2); assert.equal(execs, 1, 'exactly one exec() for the batch');
  ok('2 XADD queued, 1 exec() → one network round-trip per batch');
}

// ── 7. sink failure on a SIZE-triggered flush must not crash the process ────
console.log('7. sink failure on size-triggered flush → onError, requeue, no unhandled rejection');
{
  const unhandled = [];
  const trap = (e) => unhandled.push(e);
  process.on('unhandledRejection', trap);
  const errors = [];
  let calls = 0;
  const eb = new EgressBuffer({ maxBatch: 2, maxInflight: 4,
    sink: async () => { calls++; if (calls === 1) throw new Error('pg down'); },
    onError: (e) => errors.push(String(e)) });
  eb.add({ id: 1 }); eb.add({ id: 2 });                       // size-triggered flush → throws
  await tick(); await tick();
  process.off('unhandledRejection', trap);
  assert.equal(unhandled.length, 0, 'flush rejection must be handled inside the buffer');
  assert.equal(errors.length, 1, 'onError must see the sink failure');
  assert.equal(eb.buf.length, 2, 'failed batch requeued for retry');
  assert.equal(eb.committed, 0);
  await eb.drain();                                           // second attempt succeeds
  assert.equal(eb.committed, 2);
  ok('sink error reported via onError, batch requeued, later retry commits');
}

// ── 8. boot-time vector-dimension check ──────────────────────────────────────
console.log('8. assertVectorDim refuses a schema/embedder mismatch at boot');
{
  const pg = (t) => ({ query: async () => ({ rows: t ? [{ t }] : [] }) });
  await assertVectorDim(pg('vector(384)'), 'documents', 384);   // ok
  await assert.rejects(() => assertVectorDim(pg('vector(8)'), 'documents', 384), /vector\(8\).*vector\(384\)/);
  await assert.rejects(() => assertVectorDim(pg(null), 'documents', 384), /missing/);
  ok('vector(384) accepted; vector(8) and missing column rejected with actionable message');
}

// ── 9. jittered exponential backoff on persistent sink failure ──────────────
console.log('9. retry backoff: min(base·2^n + jitter, max), no retry before the deadline, reset on success');
{
  let clock = 1_000_000;                                     // injected clock: deterministic timing
  const now = () => clock;
  const reported = [];
  let attempts = 0, failUntil = 6;
  const eb = new EgressBuffer({ maxBatch: 1, maxInflight: 1, now,
    backoff: { baseMs: 100, maxMs: 10_000, jitter: () => 0.5 },  // jitter = 50 ms, fixed
    sink: async () => { attempts++; if (attempts <= failUntil) throw new Error('sink down'); },
    onError: (e, info) => reported.push(info) });
  eb.add({ id: 1 });                                          // size-triggered flush → attempt 1 fails
  await tick(); await tick();
  assert.equal(attempts, 1);
  assert.deepEqual(reported[0], { retries: 1, retryInMs: 250 }, '100·2^1 + 50');
  assert.equal(eb.backingOff, true);

  // hammering _flushNow before the deadline must NOT touch the sink (this is the log-spam fix)
  for (let i = 0; i < 50; i++) eb._flushNow();
  await tick();
  assert.equal(attempts, 1, 'no retry before nextRetryAt');

  const expected = [250, 450, 850, 1650, 3250, 6450];        // 100·2^n + 50, n = 1..6 (all < 10 s cap)
  for (let n = 2; n <= failUntil; n++) {
    clock += reported[n - 2].retryInMs;                       // advance exactly to the deadline
    eb._flushNow(); await tick(); await tick();
    assert.equal(attempts, n, `attempt ${n} happens once the backoff elapses`);
    assert.equal(reported[n - 1].retryInMs, expected[n - 1]);
  }
  assert.deepEqual(reported.map((r) => r.retryInMs), expected);

  // cap: a deep retry count never exceeds maxMs
  eb.retries = 20; assert.equal(eb._backoffMs(), 10_000);
  eb.retries = failUntil;

  // sink recovers → commits, counters reset, next failure starts the ladder again
  clock += reported[failUntil - 1].retryInMs;
  eb._flushNow(); await tick(); await tick();
  assert.equal(eb.committed, 1);
  assert.equal(eb.retries, 0); assert.equal(eb.backingOff, false);
  ok(`backoff ladder ${expected.join('→')} ms, capped at 10 s, sink untouched while backing off, reset on success`);
}

// ── 10. bounded drain under a dead sink ──────────────────────────────────────
console.log('10. drain({timeoutMs}) returns the undelivered count instead of hanging');
{
  // real clock, millisecond-scale backoff so the test finishes fast
  let attempts = 0;
  const eb = new EgressBuffer({ maxBatch: 2, backoff: { baseMs: 5, maxMs: 20, jitter: () => 0 },
    sink: async () => { attempts++; throw new Error('dead'); } });
  eb.add({ id: 1 }); eb.add({ id: 2 }); eb.add({ id: 3 });
  await tick();
  const t0 = Date.now();
  const left = await eb.drain({ timeoutMs: 150 });
  const took = Date.now() - t0;
  assert.ok(left >= 3, `expected ≥3 undelivered, got ${left}`);
  assert.equal(eb.committed, 0);
  assert.ok(took >= 140 && took < 2_000, `drain must give up near the timeout (took ${took} ms)`);
  assert.ok(attempts >= 3 && attempts <= 12, `attempts paced by backoff, not by the 1 ms yield loop (got ${attempts})`);
  ok(`drain gave up after ${took} ms with ${left} undelivered; ${attempts} paced attempts (unacked → broker redelivers)`);
}

console.log(`\nALL ${pass} EGRESS ASSERTIONS PASSED ✅`);
