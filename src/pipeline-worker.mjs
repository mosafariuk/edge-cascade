// pipeline-worker.mjs — platform-agnostic LLM routing gateway.
//
//   [Redis Stream | RabbitMQ]  --pull batch-->  embed(ONNX)  --> route
//        route: calibrate(router(vec)) ≥ ROUTE_TAU ? local vLLM (surprisal-guarded) : escalate
//        (default router is local-first: the guard is the deferral rule; see makeRouter)
//        --> EgressBuffer --> [pgvector | Redis Stream | RabbitMQ], ack-after-commit
//
// The batching/entropy/rate primitives are REUSED UNCHANGED from the chat build,
// proving the core is infra, not platform. Only the SOURCE and SINK adapters differ.
//
// Run:  UV_THREADPOOL_SIZE=8 node pipeline-worker.mjs
// deps: npm i ioredis amqplib pg @xenova/transformers
'use strict';
process.env.UV_THREADPOOL_SIZE ||= '8';                // libuv fix (prefer CLI env)

// NOTE: @xenova/transformers is imported DYNAMICALLY in boot() only when
// EMBED_MODE !== 'hash', so the pipeline can run (and be load-tested) without the
// ~90MB ONNX model download when you only want to stress the plumbing.
import os from 'node:os';
import { readFileSync } from 'node:fs';
import { RateEstimator, adaptiveBatchSize } from './cascade-lib.mjs';
import { guardedGenerate } from './entropy-guard.mjs';
import { EgressBuffer, pgvectorSink, redisStreamSink, assertVectorDim } from './egress.mjs';
import { admit } from './schemas.mjs';
import { extractStructured } from './constrained.mjs';
import { extractHeavy } from './heavy.mjs';
import { EDGE_MODEL, GUARD_EVALUATED } from './models.mjs';

const B_MAX      = Number(process.env.B_MAX || 32);
const VLLM_URL   = process.env.VLLM_URL || 'http://127.0.0.1:8000/v1/completions';
const SINK_MS    = Number(process.env.SINK_MS || 0);   // simulate slow downstream (bench)
// ── guard configuration ───────────────────────────────────────────────────────
// The worker runs the configuration the paper evaluated (Table VIII: 2-shot prompt,
// s_max = 0 on the mean surprisal over all value tokens) or it refuses to start. A
// different threshold or shot count is allowed only with ALLOW_GUARD_OVERRIDE=1 and is
// logged as NOT the evaluated configuration.
//   EXTRACT=0             legacy unconstrained path (entropy guard) — benchmarks only
//   EDGE_DEADLINE_MS      abort + escalate an extraction slower than this (default 5000;
//                         it must exceed the stack's object latency — the evaluation ran
//                         without a binding deadline, and 1.1–1.9 s/object was measured)
export function resolveGuardConfig(env = process.env) {
  const extract = env.EXTRACT !== '0';
  const unset = env.MAX_SURPRISAL === undefined || env.MAX_SURPRISAL === '';
  const maxSurprisal = unset ? GUARD_EVALUATED.maxSurprisal : Number(env.MAX_SURPRISAL);
  const shots = EDGE_MODEL.few_shot;
  const deadlineMs = Number(env.EDGE_DEADLINE_MS || 5000);
  const deviations = [];
  if (Number.isNaN(maxSurprisal)) deviations.push(`MAX_SURPRISAL="${env.MAX_SURPRISAL}" is not a number`);
  else if (maxSurprisal !== GUARD_EVALUATED.maxSurprisal) deviations.push(`MAX_SURPRISAL=${maxSurprisal} (evaluated: ${GUARD_EVALUATED.maxSurprisal})`);
  if (shots !== GUARD_EVALUATED.shots) deviations.push(`EDGE_MODEL.few_shot=${shots} (evaluated: ${GUARD_EVALUATED.shots})`);
  if (extract && deviations.length && env.ALLOW_GUARD_OVERRIDE !== '1') {
    throw new Error(`guard configuration differs from the evaluated one (${GUARD_EVALUATED.source}): ${deviations.join('; ')}. ` +
                    'Set ALLOW_GUARD_OVERRIDE=1 to run it anyway.');
  }
  return { extract, maxSurprisal, shots, deadlineMs, statistic: GUARD_EVALUATED.statistic, evaluated: extract && deviations.length === 0, deviations };
}
const GUARD = resolveGuardConfig();
const EXTRACT = GUARD.extract;
const W_MAX_MS   = Number(process.env.W_MAX_MS || 19);
// Router score p ∈ [0,1] is compared against ROUTE_TAU (see makeRouter). ROUTE_TAU > 1
// disables the local path entirely (bench: pure embed-bound workload, heavy path stubbed).
const ROUTE_TAU  = Number(process.env.ROUTE_TAU || 0.75);
// Pending-entry recovery (Redis Streams): entries read by a consumer that crashed before
// acking sit in the group's PEL forever unless someone reclaims them. Every worker sweeps
// the PEL on boot and every RECLAIM_MS, claims entries idle longer than RECLAIM_MIN_IDLE_MS,
// and dead-letters any entry that has already been delivered MAX_DELIVERIES times.
const RECLAIM_MS          = Number(process.env.RECLAIM_MS || 30_000);
const RECLAIM_MIN_IDLE_MS = Number(process.env.RECLAIM_MIN_IDLE_MS || 60_000);
const MAX_DELIVERIES      = Number(process.env.MAX_DELIVERIES || 5);
// NOTE: ONNX intra-op threads are pinned to 1 inside boot() where transformers is
// dynamically imported (non-hash mode only) — not here, since `env` isn't in scope.

// ── SOURCE adapter interface: { pull(maxBatch,maxWaitMs) -> msg[], ack(msg[]) } ──
// msg = { srcId, payload }  (payload = raw parsed JSON object from the hose)

// Redis Streams source (pull-based; adaptive batch = COUNT, window = BLOCK).
// `consumer` must be STABLE across restarts (hostname+slot, not pid) so a restarted
// worker sees its own pending entries; reclaim() covers entries of consumers that never
// come back.
export function redisSource(reader, writer, { stream, group, consumer, deadLetter = `${stream}:dead`,
                                              minIdleMs = RECLAIM_MIN_IDLE_MS, maxDeliveries = MAX_DELIVERIES }) {
  // Malformed entries are dead-lettered and acked immediately instead of throwing out of
  // pull(): a single bad producer must not stall the consumer or strand its batch-mates.
  const parse = async (entries) => {
    const msgs = [], bad = [];
    for (const [id, kv] of entries) {
      try { msgs.push({ srcId: id, payload: JSON.parse(kvField(kv, 'json')) }); }
      catch (e) { bad.push([id, kv, String(e)]); }
    }
    if (bad.length) await toDeadLetter(bad.map(([id, kv, reason]) => ({ id, kv, reason: 'unparseable_json: ' + reason })));
    return msgs;
  };
  const toDeadLetter = async (items) => {
    const pipe = writer.pipeline();
    for (const { id, kv, reason } of items) pipe.xadd(deadLetter, '*', 'src_id', id, 'reason', reason, ...(kv ?? []));
    pipe.xack(stream, group, ...items.map((i) => i.id));
    await pipe.exec();
  };
  return {
    async ensure() {
      await reader.xgroup('CREATE', stream, group, '$', 'MKSTREAM')
        .catch((e) => { if (!String(e).includes('BUSYGROUP')) throw e; });
    },
    async pull(maxBatch, maxWaitMs) {
      const res = await reader.xreadgroup('GROUP', group, consumer, 'COUNT', maxBatch,
                                          'BLOCK', maxWaitMs, 'STREAMS', stream, '>');
      if (!res) return [];
      return parse(res[0][1]);
    },
    async ack(msgs) { if (msgs.length) await writer.xack(stream, group, ...msgs.map((m) => m.srcId)); },
    // Sweep the PEL: claim stale entries for reprocessing; dead-letter poison entries.
    async reclaim() {
      const pend = await reader.xpending(stream, group, 'IDLE', minIdleMs, '-', '+', 1000);
      if (!pend?.length) return [];
      const poison = pend.filter(([, , , n]) => Number(n) >= maxDeliveries).map(([id]) => id);
      const stale  = pend.filter(([, , , n]) => Number(n) <  maxDeliveries).map(([id]) => id);
      if (poison.length) {
        const entries = (await reader.xclaim(stream, group, consumer, 0, ...poison)).filter(Boolean);
        await toDeadLetter(entries.map(([id, kv]) => ({ id, kv, reason: `max_deliveries_${maxDeliveries}` })));
        // ids XCLAIM did not return were trimmed from the stream; ack them so the PEL is clean
        const gone = poison.filter((id) => !entries.some(([e]) => e === id));
        if (gone.length) await writer.xack(stream, group, ...gone);
      }
      if (!stale.length) return [];
      const claimed = (await reader.xclaim(stream, group, consumer, minIdleMs, ...stale)).filter(Boolean);
      return parse(claimed);
    },
  };
}

// RabbitMQ source (push-based; prefetch caps in-flight, local buffer forms batches).
export function rabbitSource(channel, { queue }) {
  const pending = [];
  channel.consume(queue, (m) => { if (m) pending.push({ srcId: m, payload: JSON.parse(m.content.toString()) }); },
                  { noAck: false });
  return {
    async ensure() { await channel.assertQueue(queue, { durable: true }); },
    async pull(maxBatch, maxWaitMs) {
      await channel.prefetch(maxBatch);                // adaptive flow control
      if (pending.length === 0) await new Promise((r) => setTimeout(r, maxWaitMs));
      return pending.splice(0, maxBatch);
    },
    async ack(msgs) { for (const m of msgs) channel.ack(m.srcId); },  // ack raw delivery
  };
}

// ── worker ────────────────────────────────────────────────────────────────────
export async function runWorker({ source, sinkFactory, embed, calibrate, router = defaultRouter, onMetrics }) {
  await source.ensure();
  const rate = new RateEstimator({ alpha: 0.3, tickMs: 250 }).start();

  // ack-after-commit: the sink acks source messages only once egress is durable.
  const egress = new EgressBuffer({
    maxBatch: Number(process.env.EGRESS_MAX_BATCH || 256),
    maxDelayMs: 25,
    maxInflight: Number(process.env.EGRESS_MAX_INFLIGHT || 4),
    sink: sinkFactory((batch) => source.ack(batch.map((r) => r._msg))),
    backoff: { baseMs: Number(process.env.EGRESS_BACKOFF_BASE_MS || 100),
               maxMs:  Number(process.env.EGRESS_BACKOFF_MAX_MS  || 10_000) },
    // one line per failed attempt, at the backoff cadence (100 ms → 10 s), not every 25 ms
    onError: (e, { retries, retryInMs }) =>
      console.error(`egress: sink error (attempt ${retries}), retrying in ${retryInMs} ms: ${String(e).split('\n')[0]}`),
  }).start();

  let running = true, inflight = 0, curB = 1;
  const MAX_INFLIGHT = 4;
  // telemetry: emit rps / current batch size / backpressure flags every 500ms
  const mt = setInterval(() => onMetrics?.({
    rps: rate.rps, B: curB, Bmax: B_MAX, batchMaxed: curB >= B_MAX,
    egressPaused: egress.paused, inflight, committed: egress.committed,
  }), 500);
  if (mt.unref) mt.unref();
  const stop = async () => {
    running = false; rate.stop(); clearInterval(mt);
    const t0 = Date.now(); while (inflight && Date.now() - t0 < 10_000) await sleep(50);
    // Bounded: with the sink down, waiting forever would just hold the process open.
    // Whatever is still undelivered was never acked, so the broker redelivers it.
    const left = await egress.drain({ timeoutMs: Number(process.env.EGRESS_DRAIN_TIMEOUT_MS || 30_000) });
    if (left) console.error(`egress: drain timed out with ~${left} records undelivered (unacked; broker will redeliver)`);
    process.exit(left ? 1 : 0);
  };
  process.on('SIGTERM', stop); process.on('SIGINT', stop);

  // Pending-entry recovery: on boot (entries of a consumer that died) and periodically.
  let lastReclaim = 0;
  const reclaimDue = () => source.reclaim && Date.now() - lastReclaim >= RECLAIM_MS;

  while (running) {
    if (inflight >= MAX_INFLIGHT || egress.paused) { await sleep(1); continue; }  // backpressure
    curB = adaptiveBatchSize(rate.rps, W_MAX_MS / 1000, B_MAX);
    let msgs;
    if (reclaimDue()) {
      lastReclaim = Date.now();
      msgs = await source.reclaim().catch((e) => { console.error('reclaim error', e); return []; });
      if (msgs.length) console.log(`[pel] reclaimed ${msgs.length} stale entries`);
    }
    if (!msgs?.length) msgs = await source.pull(curB, W_MAX_MS);
    if (msgs.length === 0) continue;
    rate.record(msgs.length);
    inflight++;
    processBatch(msgs, { embed, calibrate, egress, router })   // fire-and-forget
      .catch((e) => console.error('batch error (entries stay pending; reclaim will retry)', e))
      .finally(() => { inflight--; });
  }
}

// One batch: one embed call, then route each message. A per-message failure produces a
// dead-letter record for THAT message only; its batch-mates still commit and ack.
// (A rejection here would strand every message of the batch in the PEL.)
export async function processBatch(msgs, { embed, calibrate, egress, router = defaultRouter, route }) {
  const texts = msgs.map((m) => extractText(m.payload));
  const vecs = await embed(texts, { pooling: 'mean', normalize: true });   // one async ONNX call
  const list = vecs.tolist ? vecs.tolist() : vecs;
  const routeFn = route ?? ((m, i) => (EXTRACT
    ? routeStructured(m.payload, list[i], calibrate, router)
    : routeOne(texts[i], list[i], calibrate, router)));
  await Promise.all(msgs.map(async (m, i) => {
    const vec = list[i];
    const id = idOf(m.payload, m);
    let output;
    try { output = await routeFn(m, i); }
    catch (e) { output = { data: null, meta: { path: 'dead_letter', reason: 'route_error', error: String(e).slice(0, 500) } }; }
    // echo t_ingest + id so the load generator can compute true end-to-end TTA
    output.t_ingest = m.payload.t_ingest;
    output.id = id;
    egress.add({ id, vec, output, _msg: m });                              // non-blocking handoff
  }));
}

// constrained structured-extraction routing (EXTRACT=1)
async function routeStructured(payload, vec, calibrate, router = defaultRouter) {
  const gate = admit(payload);                       // route + validate inbound envelope
  if (!gate.ok) return { data: null, meta: { path: 'dead_letter', reason: gate.reason } };
  const p = calibrate(router(vec));
  if (p >= ROUTE_TAU) {
    const r = await extractStructured({ endpoint: VLLM_URL, payload, schemaEntry: gate.entry,
      maxSurprisal: GUARD.maxSurprisal, shots: GUARD.shots, statistic: GUARD.statistic,
      deadlineMs: GUARD.deadlineMs });                       // guided_json + 2-shot prompt + value-surprisal guard
    if (r.status === 'done') return { data: r.data, meta: { path: 'local', kind: gate.entry.key } };
    // guard trip / parse / schema failure → fall through to the heavy model
  }
  const data = await extractHeavy(payload, gate.entry);
  return { data, meta: { path: 'escalated', kind: gate.entry.key } };
}
// extractHeavy is imported from ./heavy.mjs (Anthropic Opus 4.8 / OpenAI GPT-4o,
// same schema → validate → derive, so egress is identical to the local path).

async function routeOne(text, vec, calibrate, router = defaultRouter) {
  const p = calibrate(router(vec));
  if (p >= ROUTE_TAU) {
    const gen = guardedGenerate({ endpoint: VLLM_URL, prompt: buildPrompt(text), k: 5, Hmax: 2.5, deadlineMs: 600 });
    let v; while (true) { const n = await gen.next(); if (n.done) { v = n.value; break; } }
    if (v.status === 'done') return { text: v.text, meta: { path: 'local' } };
  }
  const heavy = await callHeavyAPI(buildPrompt(text));                      // escalation
  return { text: heavy, meta: { path: 'escalated' } };
}

// ── router ────────────────────────────────────────────────────────────────────
// A router maps the embedding to p ∈ [0,1] = P(the local model suffices); the worker
// attempts the local path iff calibrate(p) ≥ ROUTE_TAU.
//   'local-first' (default): p = 1. Every admitted payload is tried locally and the
//     value-surprisal guard in constrained.mjs is the deferral rule — this IS the cascade
//     described in the paper. ROUTE_TAU > 1 turns the local path off (bench mode).
//   'centroid': p = 1 − max_c cos(vec, c) over hard-payload centroids loaded from
//     ROUTER_CENTROIDS (JSON: array of vectors), so payloads near known-hard examples
//     escalate before any local tokens are spent. Requires L2-normalised inputs.
export function makeRouter(kind = process.env.ROUTER || 'local-first', { centroids } = {}) {
  if (kind === 'local-first') return () => 1;
  if (kind === 'centroid') {
    const cs = centroids ?? JSON.parse(readFileSync(process.env.ROUTER_CENTROIDS, 'utf8'));
    if (!Array.isArray(cs) || !cs.length) throw new Error('centroid router: ROUTER_CENTROIDS must be a non-empty array of vectors');
    return (vec) => { let best = -1; for (const c of cs) best = Math.max(best, cosine(vec, c)); return Math.min(1, Math.max(0, 1 - best)); };
  }
  throw new Error(`unknown ROUTER=${kind} (local-first | centroid)`);
}
export function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return na && nb ? dot / Math.sqrt(na * nb) : 0;
}
const defaultRouter = makeRouter('local-first');

// ── stubs to wire ─────────────────────────────────────────────────────────────
const kvField = (kv, f) => { for (let i = 0; i < kv.length; i += 2) if (kv[i] === f) return kv[i + 1]; return '{}'; };
const extractText = (o) => o.text ?? o.message ?? JSON.stringify(o);
// Record id = the upsert key. It must be identical on every redelivery of the same source
// message, or ON CONFLICT never fires and duplicates are committed. Prefer the payload's
// own identifiers, then the broker's entry id (stable across redelivery); a generated id
// is the last resort and is only reached for sources with no entry id.
export const idOf = (o, msg) => o.req_id ?? o.id ?? o.uuid ?? (typeof msg?.srcId === 'string' ? `src-${msg.srcId}` : cryptoId());
const buildPrompt = (t) => `Extract structured data.\nInput: ${t}\nOutput:`;
async function callHeavyAPI(_p) { return '{"structured":"…heavy…"}'; }
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let _c = 0; const cryptoId = () => `gen-${process.pid}-${_c++}`;

// Deterministic fake embedding (EMBED_MODE=hash): isolates pipeline plumbing from
// the ONNX model so you can load-test Redis I/O, batching, egress & backpressure
// without downloading MiniLM. NOT a substitute for measuring real ONNX contention.
function hashEmbed(texts) {
  return texts.map((t) => {
    let h = 2166136261 >>> 0;
    for (let i = 0; i < t.length; i++) { h ^= t.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; }
    const v = new Array(8);
    for (let d = 0; d < 8; d++) { h = Math.imul(h ^ (h >>> 15), 2246822507) >>> 0; v[d] = (h / 2 ** 32) * 2 - 1; }
    return v;
  });
}

// ── boot ────────────────────────────────────────────────────────────────────────
if (import.meta.url === `file://${process.argv[1]}`) {
  const Redis = (await import('ioredis')).default;
  const reader = new Redis(process.env.REDIS_URL), writer = new Redis(process.env.REDIS_URL);
  const metricsRedis = new Redis(process.env.REDIS_URL);

  // embedding backend:
  //   EMBED_MODE=hash → fast deterministic pseudo-embedding (plumbing tests)
  //   EMBED_MODE=real → native onnxruntime-node, STRICTLY pinned via ORT_INTRA_OP
  //                     (default 1 → 1 core/worker → linear horizontal scaling).
  let embed;
  if (process.env.EMBED_MODE === 'hash') { embed = async (t) => hashEmbed(t); }
  else {
    const { createNativeEmbedder } = await import('./embed-native.mjs');
    const intra = Number(process.env.ORT_INTRA_OP ?? 1);   // C++-level intraOpNumThreads
    // ORT_ALLOW_SPINNING=0|1 sets session.intra_op.allow_spinning; unset/empty = runtime default.
    const spinEnv = process.env.ORT_ALLOW_SPINNING ?? '';
    const spin = spinEnv === '' ? null : spinEnv !== '0';
    const ne = await createNativeEmbedder({ intraOpNumThreads: intra, allowSpinning: spin });
    embed = ne.embed;                                       // returns Array<Array<number>>
    console.log(`[embed] native onnxruntime-node, intraOpNumThreads=${intra}, allow_spinning=${spin === null ? 'default' : spin ? 1 : 0}`);
  }
  const calibrate = (s) => (process.env.EMBED_MODE === 'hash' ? 1 : s); // hash: always attempt local so guard routes
  const router = makeRouter();
  console.log(GUARD.extract
    ? `[guard] constrained extraction: few_shot=${GUARD.shots}, s_max=${GUARD.maxSurprisal} nats (${GUARD.statistic}), deadline=${GUARD.deadlineMs} ms — ` +
      (GUARD.evaluated ? 'EVALUATED configuration (paper Table VIII)' : `OVERRIDDEN, not the evaluated configuration: ${GUARD.deviations.join('; ')}`)
    : '[guard] EXTRACT=0: legacy unconstrained path — NOT the evaluated cascade (benchmarks only)');
  console.log(`[router] ${process.env.ROUTER || 'local-first'}, ROUTE_TAU=${ROUTE_TAU}` +
              (ROUTE_TAU > 1 ? ' (local path DISABLED: every payload escalates)' : ''));

  // egress: Redis Stream 'results' (default, for TTA measurement) or pgvector.
  let sinkFactory;
  if (process.env.EGRESS === 'pgvector') {
    const { Pool } = await import('pg');
    const pool = new Pool({ connectionString: process.env.PG_URL });
    await assertVectorDim(pool, 'documents', process.env.EMBED_MODE === 'hash' ? 8 : 384);
    sinkFactory = (ackInputs) => pgvectorSink(pool, 'documents', ackInputs);
  } else {
    const base = redisStreamSink(writer, 'results', null);
    // optional artificial slowdown to DEMONSTRATE egress backpressure under load
    sinkFactory = (ackInputs) => async (batch) => {
      if (SINK_MS) await sleep(SINK_MS);
      await base(batch); await ackInputs(batch);
    };
  }

  const onMetrics = (m) => {
    metricsRedis.xadd('metrics', 'MAXLEN', '~', '1000', '*',
      'pid', String(process.pid), 'rps', m.rps.toFixed(1), 'B', String(m.B),
      'batchMaxed', m.batchMaxed ? '1' : '0', 'egressPaused', m.egressPaused ? '1' : '0',
      'inflight', String(m.inflight), 'committed', String(m.committed)).catch(() => {});
  };

  // Stable consumer name: a restarted worker must find its own pending entries.
  const consumer = process.env.WORKER_NAME || `${os.hostname()}-${process.env.WORKER_SLOT ?? process.pid}`;
  await runWorker({
    source: redisSource(reader, writer, { stream: 'ingest', group: 'gw', consumer }),
    sinkFactory, embed, calibrate, router, onMetrics,
  });
}
