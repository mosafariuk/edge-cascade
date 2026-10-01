// shadow-run.mjs — RUN THIS AGAINST YOUR LIVE STACK to calibrate MAX_SURPRISAL.
//
// Samples payloads (from a Redis stream tap OR the payloads/ dir), runs each through
// the real 4-bit edge model (shadow mode) and the real heavy API (ground truth),
// writes the (signal, correct) pairs to shadow-pairs.jsonl, and prints the operating
// point: escalation rate, local precision floor, and edge-error recall.
//
// Requirements for a REAL run:
//   VLLM_URL=http://<edge>:8000/v1/completions   (your 4-bit guided_json server)
//   HEAVY_PROVIDER=anthropic|openai + ANTHROPIC_API_KEY / OPENAI_API_KEY
//   optional: SAMPLE_STREAM=<redis stream>  SAMPLE_N=500  TARGET_ESCALATION=0.2
//
// Sampling sources:
//   - default: every *.json under payloads/  (small; for a smoke run)
//   - SAMPLE_STREAM set: XRANGE a representative slice of live traffic (non-blocking tap)
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { shadowLabel } from '../src/shadow.mjs';
import { analyzeCalibration, wilson } from '../src/calibration.mjs';
import { EDGE_MODEL, JUDGE_MODEL, provenanceHeader, assertModelsDeclared } from '../src/models.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const VLLM_URL = process.env.VLLM_URL || 'http://127.0.0.1:8000/v1/completions';
const TARGET = Number(process.env.TARGET_ESCALATION || 0.2);
const N = Number(process.env.SAMPLE_N || 500);
// The pairs file is a paper artifact: refuse to produce one that cannot name its models.
// ALLOW_TODO_MODELS=1 bypasses this for smoke runs against the mock.
if (process.env.ALLOW_TODO_MODELS !== '1') assertModelsDeclared(['edge', 'judge']);

async function loadPayloads() {
  if (process.env.SAMPLE_STREAM) {
    const Redis = (await import('ioredis')).default;
    const r = new Redis(process.env.REDIS_URL || 'redis://127.0.0.1:6379');
    const rows = await r.xrange(process.env.SAMPLE_STREAM, '-', '+', 'COUNT', N);
    await r.quit();
    return rows.map(([, kv]) => { const o = {}; for (let i = 0; i < kv.length; i += 2) o[kv[i]] = kv[i + 1]; return JSON.parse(o.json); });
  }
  // fallback: <PAYLOADS_ROOT>/{property_audit,telemetry}/*.json  (default payloads/;
  // payloads/synthetic for the published synthetic corpus)
  const root = process.env.PAYLOADS_ROOT ? join(process.cwd(), process.env.PAYLOADS_ROOT) : join(here, '..', 'payloads');
  const out = [];
  for (const d of ['property_audit', 'telemetry']) {
    let files; try { files = readdirSync(join(root, d)); } catch { continue; }
    for (const f of files.filter((x) => x.endsWith('.json')).sort()) out.push(JSON.parse(readFileSync(join(root, d, f), 'utf8')));
  }
  return out;
}

const payloads = await loadPayloads();
const provider = process.env.HEAVY_PROVIDER || JUDGE_MODEL.provider;
const judgeId = process.env.HEAVY_MODEL || (provider === 'openai' ? JUDGE_MODEL.openai_id : JUDGE_MODEL.id);
console.log(`shadow-labeling ${payloads.length} payloads → edge ${EDGE_MODEL.id} @ ${VLLM_URL} + judge ${provider}/${judgeId}`);
if (!process.env.ANTHROPIC_API_KEY && !process.env.OPENAI_API_KEY) {
  console.error('WARNING: no heavy-API key set — adjudication will fail. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.');
}

const HOLDOUT = Number(process.env.HOLDOUT || 0);   // e.g. 0.3 for out-of-sample reporting
// Shadow pass is not latency-bound: give the edge model long enough to finish every object
// (tunnelled GPU host, 8B model ≈ 1–3 s/object). The production deadline stays 800 ms.
const edgeOpts = { deadlineMs: Number(process.env.SHADOW_DEADLINE_MS || 30_000), maxTokens: Number(process.env.SHADOW_MAX_TOKENS || 300) };
const t0 = Date.now();
// SHADOW_TRACE=1 also writes bench/shadow-trace.jsonl (texts + both extractions). Only do this
// for the synthetic corpus: the trace contains payload text.
const trace = process.env.SHADOW_TRACE === '1' ? [] : null;
const pairs = await shadowLabel({ payloads, vllmUrl: VLLM_URL, heavyOpts: { provider, model: judgeId }, edgeOpts, trace,
  concurrency: Number(process.env.SHADOW_CONCURRENCY || 8),
  onProgress: (p) => console.log(`  [${p.i}/${p.n}] labeled=${p.labeled} skipped=${JSON.stringify(p.skipped)}  ${((Date.now() - t0) / 1000).toFixed(0)}s`) });
console.log(`labeled ${pairs.length}/${payloads.length} payloads; skipped ${JSON.stringify(pairs.skipped)}`);
// Line 1 of the artifact is a provenance header (models, stack, sampling design); every
// later line is one pair. Readers must skip lines with kind==='provenance'.
const hasTruth = pairs.length > 0 && pairs.every((p) => typeof p.correct_truth === 'boolean');
const header = provenanceHeader({
  edge_endpoint: VLLM_URL, judge: { provider, id: judgeId },
  sample: { n_requested: N, n_payloads: payloads.length, n_labeled: pairs.length, skipped: pairs.skipped,
            source: process.env.SAMPLE_STREAM || process.env.PAYLOADS_ROOT || 'payloads/',
            ground_truth: hasTruth ? 'constructed (metadata.synthetic_truth), judge scored against it' : 'frontier judge',
            target_escalation: TARGET, holdout: HOLDOUT },
});
writeFileSync(join(here, 'shadow-pairs.jsonl'),
  [JSON.stringify(header), ...pairs.map((p) => JSON.stringify(p))].join('\n') + '\n');

if (trace) writeFileSync(join(here, 'shadow-trace.jsonl'), trace.map((t) => JSON.stringify(t)).join('\n') + '\n');
const pct = (x) => (x == null ? 'n/a' : (x * 100).toFixed(1));
const ci = ([lo, hi]) => (lo == null ? '[n/a]' : `[${pct(lo)}, ${pct(hi)}]`);
function report(title, ps) {
  const c = analyzeCalibration(ps, TARGET, HOLDOUT);
  console.log(`\n─── ${title} ───`);
  if (!c.n) { console.log('no usable pairs — check the edge server, heavy key, and that logprobs are enabled'); return c; }
  console.log(`labeled pairs        : ${c.n}${c.holdout ? `  (fit on train=${c.n_train}, EVALUATED on held-out=${c.n_eval})` : '  (in-sample)'}`);
  console.log(`edge accuracy (raw)  : ${pct(c.baseline_edge_accuracy)}%  95% CI ${ci(c.baseline_ci)}`);
  console.log(`target escalation    : ${(c.target_escalation_rate * 100).toFixed(0)}%   → MAX_SURPRISAL = ${c.threshold.toFixed(3)} nats`);
  console.log(`achieved escalation  : ${pct(c.escalation_rate)}%  95% CI ${ci(c.escalation_ci)}`);
  console.log(`local precision      : ${pct(c.local_precision)}%  95% CI ${ci(c.precision_ci)}`);
  console.log(`edge-error recall    : ${pct(c.error_recall)}%  95% CI ${ci(c.recall_ci)}  (${c.n_errors} errors in eval set)`);
  console.log(`calibrated ECE       : ${c.ece_calibrated.toFixed(3)}`);
  return c;
}
const by = (ps, key, lab) => { const m = {}; for (const p of ps) { const g = (m[p[key]] ??= { n: 0, ok: 0 }); g.n++; if (p[lab]) g.ok++; }
  return Object.entries(m).sort().map(([k, v]) => `${k} ${v.ok}/${v.n} (${pct(v.ok / v.n)}%)`).join('   '); };

if (hasTruth) {
  const jt = pairs.filter((p) => p.judge_truth).length;
  console.log(`\n─── JUDGE FIDELITY (judge vs constructed truth) ───`);
  console.log(`judge agrees with truth : ${jt}/${pairs.length} = ${pct(jt / pairs.length)}%  95% CI ${ci(wilson(jt, pairs.length))}`);
  console.log(`  by kind : ${by(pairs, 'kind', 'judge_truth')}`);
  const agree = pairs.filter((p) => p.correct === p.correct_truth).length;
  console.log(`judge-relative label == truth-relative label : ${agree}/${pairs.length} = ${pct(agree / pairs.length)}%`);
  console.log(`\nedge accuracy vs truth by kind  : ${by(pairs, 'kind', 'correct_truth')}`);
  console.log(`edge accuracy vs truth by style : ${by(pairs, 'style', 'correct_truth')}`);
  report('PRIMARY — labels = constructed truth (report these)', pairs.map((p) => ({ ...p, correct: p.correct_truth })));
  for (const k of ['property_audit', 'telemetry']) report(`per workload: ${k} (truth labels)`, pairs.filter((p) => p.kind === k).map((p) => ({ ...p, correct: p.correct_truth })));
  report('SECONDARY — labels = frontier judge (the deployable, label-free procedure)', pairs);
} else {
  const c = report('CALIBRATION RESULT — labels = frontier judge', pairs);
  if (!c.n) process.exit(1);
}
console.log(`\npairs written to bench/shadow-pairs.jsonl`);
