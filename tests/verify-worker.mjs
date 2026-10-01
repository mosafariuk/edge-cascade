// verify-worker.mjs — worker-level delivery contract, without Redis or a model:
// router reachability, stable record ids, per-message failure isolation, and
// pending-entry (PEL) recovery on a mocked Redis Streams client.
import assert from 'node:assert/strict';
import { makeRouter, cosine, idOf, processBatch, redisSource, resolveGuardConfig } from '../src/pipeline-worker.mjs';

let pass = 0; const ok = (n) => { console.log(`  ✓ ${n}`); pass++; };
const tick = () => new Promise((r) => setImmediate(r));

// ── 1. the local path is reachable ───────────────────────────────────────────
console.log('1. router: local-first reaches the local path at the default threshold');
{
  const r = makeRouter('local-first');
  const unit = Array.from({ length: 384 }, (_, i) => (i === 0 ? 1 : 0));
  assert.equal(r(unit), 1);
  assert.ok(r(unit) >= 0.75, 'default ROUTE_TAU=0.75 must be satisfiable by a confident vector');
  assert.ok(!(r(unit) >= 2), 'ROUTE_TAU=2 disables the local path');
  ok('score=1 ≥ 0.75 → local attempted; τ=2 → escalate (bench mode)');
}

// ── 2. centroid router: near a hard centroid → low score → escalate ──────────
console.log('2. router: centroid distance');
{
  const hard = [[1, 0, 0], [0, 1, 0]];
  const r = makeRouter('centroid', { centroids: hard });
  assert.ok(Math.abs(cosine([1, 0, 0], [1, 0, 0]) - 1) < 1e-12);
  assert.ok(r([1, 0, 0]) < 0.01, 'on a hard centroid → p≈0 → escalate');
  assert.ok(r([0, 0, 1]) > 0.99, 'orthogonal to every hard centroid → p≈1 → local');
  assert.throws(() => makeRouter('centroid', { centroids: [] }));
  assert.throws(() => makeRouter('nope'));
  ok('p = 1 − max cosine; empty centroids and unknown router rejected');
}

// ── 3. record id is stable across redelivery ─────────────────────────────────
console.log('3. idOf prefers payload ids, then the broker entry id');
{
  const msg = { srcId: '1700000000000-0', payload: {} };
  assert.equal(idOf({ req_id: 'r1', id: 'x' }, msg), 'r1');
  assert.equal(idOf({ id: 'x' }, msg), 'x');
  assert.equal(idOf({}, msg), 'src-1700000000000-0');
  assert.equal(idOf({}, msg), idOf({}, msg), 'same source entry → same id on redelivery');
  assert.notEqual(idOf({}, { srcId: 'a' }), idOf({}, { srcId: 'b' }));
  ok('req_id > id > src entry id; identical on redelivery');
}

// ── 4. one failing message does not strand its batch-mates ───────────────────
console.log('4. processBatch isolates per-message failures');
{
  const added = [];
  const egress = { add: (r) => added.push(r) };
  const embed = async (texts) => texts.map(() => [0.1, 0.2]);
  const msgs = [
    { srcId: '1-0', payload: { req_id: 'a', text: 'ok' } },
    { srcId: '1-1', payload: { req_id: 'b', text: 'boom' } },
    { srcId: '1-2', payload: { req_id: 'c', text: 'ok' } },
  ];
  const route = async (m) => { if (m.payload.text === 'boom') throw new Error('heavy refused'); return { data: { v: 1 }, meta: { path: 'local' } }; };
  await processBatch(msgs, { embed, calibrate: (s) => s, egress, route });
  assert.equal(added.length, 3, 'all three records handed to egress');
  const byId = Object.fromEntries(added.map((r) => [r.id, r]));
  assert.equal(byId.a.output.meta.path, 'local');
  assert.equal(byId.c.output.meta.path, 'local');
  assert.equal(byId.b.output.meta.path, 'dead_letter');
  assert.equal(byId.b.output.meta.reason, 'route_error');
  assert.match(byId.b.output.meta.error, /heavy refused/);
  assert.equal(byId.b._msg.srcId, '1-1', 'dead-lettered record still carries its source msg → gets acked after commit');
  ok('a, c committed as local; b becomes a dead_letter record; nothing left pending');
}

// ── 5. PEL recovery on a mocked Redis Streams client ─────────────────────────
console.log('5. redisSource.reclaim: stale entries reprocessed, poison entries dead-lettered');
{
  const calls = [];
  const entries = {
    '5-0': ['json', JSON.stringify({ req_id: 'stale' })],
    '6-0': ['json', JSON.stringify({ req_id: 'poison' })],
    '7-0': ['json', '{not json'],
  };
  const pipe = { ops: [], xadd(...a) { this.ops.push(['xadd', ...a]); return this; }, xack(...a) { this.ops.push(['xack', ...a]); return this; },
                 async exec() { calls.push(...this.ops); this.ops = []; return []; } };
  const reader = {
    async xpending(stream, group, IDLE, minIdle, s, e, count) {
      calls.push(['xpending', minIdle]);
      return [['5-0', 'dead-worker', '90000', '1'], ['6-0', 'dead-worker', '90000', '5'], ['8-0', 'dead-worker', '90000', '9']];
    },
    async xclaim(stream, group, consumer, minIdle, ...ids) {
      calls.push(['xclaim', consumer, minIdle, ...ids]);
      return ids.map((id) => (entries[id] ? [id, entries[id]] : null));   // 8-0 was trimmed → null
    },
    async xreadgroup() { return [['ingest', [['7-0', entries['7-0']], ['9-0', ['json', '{"req_id":"fine"}']]]]]; },
  };
  const writer = { pipeline: () => pipe, async xack(...a) { calls.push(['xack', ...a]); } };
  const src = redisSource(reader, writer, { stream: 'ingest', group: 'gw', consumer: 'host-3', minIdleMs: 60000, maxDeliveries: 5 });

  const got = await src.reclaim();
  assert.deepEqual(got.map((m) => m.payload.req_id), ['stale'], 'only the under-limit stale entry is returned for reprocessing');
  const dl = calls.filter((c) => c[0] === 'xadd' && c[1] === 'ingest:dead');
  assert.equal(dl.length, 1, 'poison entry (5 deliveries) goes to ingest:dead');
  assert.equal(dl[0][3], 'src_id'); assert.equal(dl[0][4], '6-0');
  assert.ok(calls.some((c) => c[0] === 'xack' && c.includes('6-0')), 'poison entry acked after dead-lettering');
  assert.ok(calls.some((c) => c[0] === 'xack' && c.includes('8-0')), 'trimmed entry acked so the PEL is clean');
  assert.ok(calls.some((c) => c[0] === 'xclaim' && c[1] === 'host-3' && c[2] === 60000 && c.includes('5-0')), 'stale entry claimed by this consumer');

  calls.length = 0;
  const pulled = await src.pull(10, 5);
  assert.deepEqual(pulled.map((m) => m.payload.req_id), ['fine'], 'malformed entry is not returned and does not throw');
  const dl2 = calls.filter((c) => c[0] === 'xadd' && c[1] === 'ingest:dead');
  assert.equal(dl2.length, 1); assert.equal(dl2[0][4], '7-0'); assert.match(dl2[0][6], /unparseable_json/);
  assert.ok(calls.some((c) => c[0] === 'xack' && c.includes('7-0')));
  ok('stale → reprocessed; poison → dead-lettered + acked; trimmed → acked; bad JSON → dead-lettered, batch-mates delivered');
}

// ── 6. the worker runs the evaluated guard configuration or refuses to start ──
console.log('6. guard configuration is enforced');
{
  const d = resolveGuardConfig({});
  assert.deepEqual({ extract: d.extract, shots: d.shots, maxSurprisal: d.maxSurprisal, statistic: d.statistic, evaluated: d.evaluated },
                   { extract: true, shots: 2, maxSurprisal: 0, statistic: 'mean-all', evaluated: true }, 'default = paper Table VIII configuration');
  assert.ok(d.deadlineMs >= 2000, 'deadline must not truncate a 1–2 s extraction');
  assert.equal(resolveGuardConfig({ MAX_SURPRISAL: '0' }).evaluated, true);
  assert.throws(() => resolveGuardConfig({ MAX_SURPRISAL: '0.5' }), /differs from the evaluated/);
  assert.throws(() => resolveGuardConfig({ MAX_SURPRISAL: 'abc' }), /not a number/);
  const o = resolveGuardConfig({ MAX_SURPRISAL: '0.021', ALLOW_GUARD_OVERRIDE: '1' });
  assert.equal(o.maxSurprisal, 0.021); assert.equal(o.evaluated, false);
  const legacy = resolveGuardConfig({ EXTRACT: '0', MAX_SURPRISAL: '9' });
  assert.equal(legacy.extract, false); assert.equal(legacy.evaluated, false);
  ok('default: 2-shot, s_max=0, mean-all; a different threshold is refused without ALLOW_GUARD_OVERRIDE=1');
}

console.log(`\nALL ${pass} WORKER ASSERTIONS PASSED ✅`);
