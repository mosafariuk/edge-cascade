// load-gen.mjs — ruthless ramping load generator + end-to-end telemetry.
//
//  Producer:  XADD 'ingest' at a rate that RAMPS from RATE_START..RATE_END over
//             DURATION seconds (open-loop: fires on a schedule, does NOT wait for
//             completions — so it exposes the real capacity cliff).
//  Collector: reads 'results' stream, computes TTA = t_done - t_ingest per payload.
//  Monitor:   tails the 'metrics' stream and logs when batching maxes out / egress
//             pauses ingestion (backpressure).
//
//  Self-test: `node load-gen.mjs --selftest` verifies the percentile math offline.
import Redis from 'ioredis';

function percentiles(samples) {
  if (!samples.length) return null;
  const s = samples.slice().sort((a, b) => a - b);
  const q = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  const mean = s.reduce((a, b) => a + b, 0) / s.length;
  return { n: s.length, mean, p50: q(0.5), p95: q(0.95), p99: q(0.99), max: s[s.length - 1] };
}

if (process.argv.includes('--selftest')) {
  // ints 1..1000: p50≈500, p95≈950, p99≈990, mean≈500
  const r = percentiles(Array.from({ length: 1000 }, (_, i) => i + 1));
  const near = (a, b, t) => Math.abs(a - b) <= t;
  const assert = (c, m) => { if (!c) { console.error('FAIL', m); process.exit(1); } };
  assert(near(r.mean, 500, 2), 'mean'); assert(near(r.p50, 501, 5), 'p50');
  assert(near(r.p95, 951, 5), 'p95');   assert(near(r.p99, 991, 5), 'p99');
  console.log('percentile self-test PASSED', r); process.exit(0);
}

const URL         = process.env.REDIS_URL || 'redis://127.0.0.1:6379';
const RATE_START  = Number(process.env.RATE_START || 10);
const RATE_END    = Number(process.env.RATE_END || 5000);
const DURATION    = Number(process.env.DURATION || 30);      // seconds
const HARD_FRAC   = Number(process.env.HARD_FRAC || 0.2);    // fraction routed to escalate

const pub = new Redis(URL), col = new Redis(URL), mon = new Redis(URL);
const ttas = [];
let sent = 0, done = 0, startWall = 0;

// ── collector: end-to-end TTA from the 'results' stream ──────────────────────
(async () => {
  let last = '$';
  while (true) {
    const res = await col.xread('BLOCK', 1000, 'COUNT', 1000, 'STREAMS', 'results', last);
    if (!res) continue;
    const now = Date.now();
    for (const [, entries] of res) for (const [id, kv] of entries) {
      last = id;
      const data = JSON.parse(field(kv, 'data'));
      if (data.t_ingest) { ttas.push(now - data.t_ingest); done++; }
    }
  }
})().catch(() => {});

// ── monitor: backpressure events from the 'metrics' stream ───────────────────
(async () => {
  let last = '$', lastPaused = false, lastMaxed = false;
  while (true) {
    const res = await mon.xread('BLOCK', 1000, 'COUNT', 100, 'STREAMS', 'metrics', last);
    if (!res) continue;
    for (const [, entries] of res) for (const [id, kv] of entries) {
      last = id;
      const paused = field(kv, 'egressPaused') === '1', maxed = field(kv, 'batchMaxed') === '1';
      if (paused && !lastPaused) log(`⚠️  BACKPRESSURE: egress paused ingestion (pid ${field(kv, 'pid')}, B=${field(kv, 'B')}, rps=${field(kv, 'rps')})`);
      if (!paused && lastPaused) log(`✅ egress recovered (pid ${field(kv, 'pid')})`);
      if (maxed && !lastMaxed) log(`🔶 adaptive batch MAXED at Bmax (pid ${field(kv, 'pid')}, rps=${field(kv, 'rps')})`);
      lastPaused = paused; lastMaxed = maxed;
    }
  }
})().catch(() => {});

// ── producer: open-loop ramp ─────────────────────────────────────────────────
async function ramp() {
  startWall = Date.now();
  const end = startWall + DURATION * 1000;
  let id = 0;
  while (Date.now() < end) {
    const t = (Date.now() - startWall) / (DURATION * 1000);          // 0..1
    const rate = RATE_START + (RATE_END - RATE_START) * t;           // linear ramp
    const perTick = Math.max(1, Math.round(rate / 50));              // 50 ticks/sec
    const pipe = pub.pipeline();
    for (let i = 0; i < perTick; i++) {
      const hard = Math.random() < HARD_FRAC;
      const payload = { id: `p${id++}`, t_ingest: Date.now(),
        text: (hard ? '[HARD] ' : '') + 'classify this record ' + id };
      pipe.xadd('ingest', 'MAXLEN', '~', '100000', '*', 'json', JSON.stringify(payload));
      sent++;
    }
    await pipe.exec();
    await new Promise((r) => setTimeout(r, 20));                     // 20ms tick
  }
}

function report() {
  const r = percentiles(ttas);
  const wall = (Date.now() - startWall) / 1000;
  log('─'.repeat(64));
  log(`sent=${sent}  completed=${done}  wall=${wall.toFixed(1)}s  achieved≈${(done / wall).toFixed(0)} req/s`);
  if (r) log(`TTA(ms)  mean=${r.mean.toFixed(1)}  p50=${r.p50}  p95=${r.p95}  p99=${r.p99}  max=${r.max}  (n=${r.n})`);
  log('─'.repeat(64));
}
const field = (kv, f) => { for (let i = 0; i < kv.length; i += 2) if (kv[i] === f) return kv[i + 1]; return null; };
const log = (...a) => console.log(`[${((Date.now() - startWall) / 1000).toFixed(1)}s]`, ...a);

// periodic progress + final report
const prog = setInterval(report, 5000);
ramp().then(async () => {
  await new Promise((r) => setTimeout(r, 3000));   // let tail drain
  clearInterval(prog); report();
  await Promise.allSettled([pub.quit(), col.quit(), mon.quit()]);
  process.exit(0);
});
