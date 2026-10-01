// egress.mjs — non-blocking downstream handoff.
//
// EgressBuffer decouples egress from ingestion: records are micro-batched and
// flushed by size OR time, with bounded in-flight flushes providing BACKPRESSURE
// (the worker checks `.paused` and stops ingesting instead of buffering forever).
//
// Delivery contract: the sink commits downstream, THEN acks the source ids it
// carried. Input is acked only after output is durably committed ⇒ at-least-once
// with NO loss. Duplicates on crash-redelivery are absorbed by idempotent sinks
// (pgvector ON CONFLICT upsert / message dedup). This is stated, not hand-waved.
'use strict';

export class EgressBuffer {
  constructor({ maxBatch = 256, maxDelayMs = 25, maxInflight = 4, sink, onError = () => {},
                // retry backoff after a sink failure: wait = min(base·2^retries + jitter, max)
                backoff = {}, now = Date.now } = {}) {
    this.maxBatch = maxBatch;
    this.maxDelayMs = maxDelayMs;
    this.maxInflight = maxInflight;
    this.sink = sink;                 // async (record[]) => void ; MUST be idempotent-safe
    this.onError = onError;           // (err, { retries, retryInMs }) => void
    this.backoff = { baseMs: 100, maxMs: 10_000, jitter: Math.random, ...backoff };
    this.now = now;
    this.buf = [];
    this.inflight = 0;
    this._committed = 0;
    this._timer = null;
    this.retries = 0;                 // consecutive sink failures (reset on any success)
    this.nextRetryAt = 0;             // no flush is attempted before this timestamp
  }
  // Backpressure signal: full flush slots, or buffer already holds several batches.
  get paused() {
    return this.inflight >= this.maxInflight || this.buf.length >= this.maxBatch * this.maxInflight;
  }
  get committed() { return this._committed; }
  get backingOff() { return this.now() < this.nextRetryAt; }

  add(record) { this.buf.push(record); if (this.buf.length >= this.maxBatch) this._flushNow(); }

  start() {
    this._timer = setInterval(() => this._flushNow(), this.maxDelayMs);
    if (this._timer.unref) this._timer.unref();
    return this;
  }
  stop() { if (this._timer) clearInterval(this._timer); this._timer = null; }

  // Delay before the (retries)-th retry: jittered exponential, capped.
  _backoffMs() {
    const { baseMs, maxMs, jitter } = this.backoff;
    return Math.min(baseMs * 2 ** this.retries + Math.floor(jitter() * baseMs), maxMs);
  }

  // Never rejects: a sink failure requeues the batch, schedules a backoff and is reported
  // ONCE through onError. (add() and the timer fire-and-forget this promise, so a rejection
  // here would be an unhandled rejection and take the worker down with unacked messages
  // in flight.) While backing off, every call is a no-op, so a dead sink is probed at the
  // backoff cadence rather than every maxDelayMs.
  _flushNow() {
    if (this.buf.length === 0 || this.inflight >= this.maxInflight || this.backingOff) return null;
    const batch = this.buf.splice(0, this.maxBatch);   // take up to one batch
    this.inflight++;
    return Promise.resolve()
      .then(() => this.sink(batch))
      .then(() => { this._committed += batch.length; this.retries = 0; this.nextRetryAt = 0; },
            (e) => {
              this.buf.unshift(...batch);                 // requeue on failure (retry)
              this.retries++;
              const retryInMs = this._backoffMs();
              this.nextRetryAt = this.now() + retryInMs;
              this.onError(e, { retries: this.retries, retryInMs });
            })
      .finally(() => { this.inflight--; });
  }

  // Flush everything before shutdown (drain in-flight + remaining buffer). Returns the
  // number of records still undelivered when `timeoutMs` elapses (0 on a clean drain);
  // those records' source messages were never acked, so the broker redelivers them.
  async drain({ timeoutMs = Infinity } = {}) {
    this.stop();
    const deadline = this.now() + timeoutMs;
    while (this.buf.length || this.inflight) {
      if (this.now() >= deadline) return this.buf.length + this.inflight * this.maxBatch;
      const p = this._flushNow();
      if (p) await p;
      else await new Promise((r) => setTimeout(r, 1));   // inflight full or backing off → yield
    }
    return 0;
  }
}

// ── concrete sinks (constructed with an injected client → unit-testable) ──────

/** Redis Stream egress: one pipelined round-trip per batch. */
export const redisStreamSink = (redis, key, ackInputs) => async (batch) => {
  const pipe = redis.pipeline();
  for (const r of batch) pipe.xadd(key, '*', 'id', r.id, 'data', JSON.stringify(r.output));
  const res = await pipe.exec();                       // single RTT for the whole batch
  const failed = res.find(([e]) => e);
  if (failed) throw failed[0];
  if (ackInputs) await ackInputs(batch);               // ack source only after commit
};

/** RabbitMQ egress on a CONFIRM channel: publish all, await confirms ONCE. */
export const rabbitSink = (confirmCh, exchange, routingKey = '', ackInputs) => async (batch) => {
  for (const r of batch) {
    confirmCh.publish(exchange, routingKey, Buffer.from(JSON.stringify(r.output)),
                      { persistent: true, messageId: String(r.id) });
  }
  await confirmCh.waitForConfirms();                   // batched durability, not per-msg await
  if (ackInputs) await ackInputs(batch);
};

/** pgvector bulk upsert: single multi-row parameterized INSERT ... ON CONFLICT. */
export const pgvectorSink = (pool, table = 'documents', ackInputs) => async (batch) => {
  const rows = [], params = [];
  batch.forEach((r, i) => {
    const o = i * 4;
    rows.push(`($${o + 1},$${o + 2},$${o + 3}::vector,$${o + 4}::jsonb)`);
    params.push(r.id, r.output.text ?? '', toVectorLiteral(r.vec), JSON.stringify(r.output.meta ?? {}));
  });
  const sql =
    `INSERT INTO ${table} (id, content, embedding, meta) VALUES ${rows.join(',')} ` +
    `ON CONFLICT (id) DO UPDATE SET content=EXCLUDED.content, embedding=EXCLUDED.embedding, meta=EXCLUDED.meta`;
  await pool.query(sql, params);                       // idempotent ⇒ dup-safe
  if (ackInputs) await ackInputs(batch);
};

// pgvector text input format: '[0.1,0.2,...]'
export function toVectorLiteral(vec) { return '[' + Array.from(vec).join(',') + ']'; }

/**
 * Boot-time check that `table.embedding` is declared with the dimension the embedder
 * produces. A mismatch is otherwise only discovered at the first flush, where every
 * INSERT fails, the batch requeues forever and nothing is ever acked.
 */
export async function assertVectorDim(pool, table, expectedDim) {
  const { rows } = await pool.query(
    `SELECT format_type(atttypid, atttypmod) AS t FROM pg_attribute
      WHERE attrelid = $1::regclass AND attname = 'embedding' AND NOT attisdropped`, [table]);
  const declared = rows[0]?.t ?? null;                          // e.g. 'vector(384)'
  const m = /^vector\((\d+)\)$/.exec(declared ?? '');
  if (!m || Number(m[1]) !== expectedDim) {
    throw new Error(`pgvector: ${table}.embedding is ${declared ?? 'missing'}, embedder produces vector(${expectedDim}); ` +
                    `fix bench/init.sql (or set EMBED_MODE to match)`);
  }
}
