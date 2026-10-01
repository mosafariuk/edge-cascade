// optimize-guard.mjs — grid search over edge-model generation settings for the guard.
//
// Question: does softening the edge model's output distribution (temperature > 0) or
// teaching it the null / false-unless-stated convention with few-shot examples make its
// errors more detectable by the value-surprisal guard — WITHOUT just manufacturing new
// errors? Each configuration re-queries the served edge model over the synthetic corpus
// (payloads/synthetic, constructed truth in metadata.synthetic_truth) and records, for
// every value token: surprisal, top-k entropy and top-1/top-2 margin. No judge is needed.
//
// The existing shadow-pairs.jsonl cannot be reused: its traces are specific to T=0 /
// zero-shot. This script writes its own trace file; analysis/guard_grid.py turns it into
// the comparison table (selection on the training split, report on the held-out split).
//
//   VLLM_URL=http://127.0.0.1:18000/v1/completions node bench/optimize-guard.mjs
//   TEMPS="0 0.1 0.3 0.5" SHOTS="0 1 2" CONCURRENCY=16 REPEATS=1 node bench/optimize-guard.mjs
//   DRY_RUN=1 node bench/optimize-guard.mjs        # no server: synthetic stream, proves the pipeline
//
// Output: bench/results-zen5-run2/guard-grid.jsonl  (line 1 = provenance + grid definition)
import { readdirSync, readFileSync, writeFileSync, appendFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { SCHEMAS, admit } from '../src/schemas.mjs';
import { JsonPos } from '../src/constrained.mjs';
import { fieldDiff } from '../src/shadow.mjs';
import { EDGE_MODEL, provenanceHeader, assertModelsDeclared } from '../src/models.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const VLLM_URL = process.env.VLLM_URL || 'http://127.0.0.1:18000/v1/completions';
const ROOT = join(process.cwd(), process.env.PAYLOADS_ROOT || 'payloads/synthetic');
const OUT = process.env.OUT || join(here, 'results-zen5-run2', 'guard-grid.jsonl');
const TEMPS = (process.env.TEMPS || '0 0.1 0.3 0.5').split(/\s+/).map(Number);
const SHOTS = (process.env.SHOTS || '0 1 2').split(/\s+/).map(Number);
const REPEATS = Number(process.env.REPEATS || 1);          // >1 only matters for T > 0
const CONC = Number(process.env.CONCURRENCY || 16);
const TOPK = Number(process.env.TOPK || 5);
const SEED = Number(process.env.SEED || 20261003);
const DRY = process.env.DRY_RUN === '1';
const LIMIT = Number(process.env.LIMIT || 0);              // smoke test: first N payloads per kind
if (!DRY) assertModelsDeclared(['edge']);

// ── few-shot examples: hand-written, values chosen to occur nowhere in the corpus ──────
// They demonstrate the two conventions the zero-shot model violates most: `null` for a
// field the text does not state, and a boolean marker `true` ONLY when the text states it.
const SHOT_EXAMPLES = {
  property_audit: [
    { p: { raw_text: 'Annexe C, Harbour Yard. Current rateable value £31,750. A neighbouring unit is assessed at £29,000. No ancillary items.' },
      a: { current_value_gbp: 31750, previous_value_gbp: null, cohort_avg_increase_pct: null, effective_date_raw: null, assets: [] } },
    { p: { raw_text: 'Suite 9, Old Brewery: curr val 27.3k, prev 22,950. Cohort avg incr 6.4%. Eff. 12-Jan. Incl. cycle shelter.' },
      a: { current_value_gbp: 27300, previous_value_gbp: 22950, cohort_avg_increase_pct: 6.4, effective_date_raw: '12-Jan', assets: ['cycle shelter'] } },
  ],
  telemetry: [
    { p: { raw_telemetry: '[01:07:33] Pitch elevated +1.6σ. Frame buffer at 212. Baseline deviation flagged.' },
      a: { acoustic_sigma: 1.6, gaze_deviation_duration_sec: null, gaze_offscreen: false, posture_rigid: false, baseline_nominal: false } },
    { p: { raw_telemetry: '[00:42:10] Gaze off-screen for 5.3s; rigid posture detected. Baseline stress metrics nominal.' },
      a: { acoustic_sigma: null, gaze_deviation_duration_sec: 5.3, gaze_offscreen: true, posture_rigid: true, baseline_nominal: true } },
  ],
};
const turn = (instruction, answer) => `<|im_start|>user\n${instruction}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n${answer ?? ''}`;
function buildPrompt(entry, payload, shots) {
  let p = '';
  for (const ex of SHOT_EXAMPLES[entry.key].slice(0, shots)) p += turn(entry.prompt(ex.p), JSON.stringify(ex.a)) + '<|im_end|>\n';
  return p + turn(entry.prompt(payload));            // open assistant turn; guided_json constrains what follows
}

// ── corpus ───────────────────────────────────────────────────────────────────────────
const payloads = [];
for (const d of ['property_audit', 'telemetry']) {
  const files = readdirSync(join(ROOT, d)).filter((f) => f.endsWith('.json')).sort();
  for (const f of (LIMIT ? files.slice(0, LIMIT) : files)) payloads.push(JSON.parse(readFileSync(join(ROOT, d, f), 'utf8')));
}
if (!payloads.every((p) => p.metadata?.synthetic_truth)) throw new Error('every payload needs metadata.synthetic_truth (use payloads/synthetic)');

// ── one extraction with full per-value-token trace ───────────────────────────────────
const r4 = (x) => Math.round(x * 1e4) / 1e4;
function topkStats(top, lp) {                       // top: {token: logprob} for one position
  const lps = Object.values(top ?? {}).sort((a, b) => b - a);
  if (!lps.length) return { h: 0, m: 0 };
  const ps = lps.map(Math.exp); const rest = Math.max(0, 1 - ps.reduce((a, b) => a + b, 0));
  let h = 0; for (const q of ps) if (q > 0) h -= q * Math.log(q);
  if (rest > 1e-9) h -= rest * Math.log(rest);     // un-listed mass as one bucket (lower bound on H)
  return { h, m: lps.length > 1 ? lps[0] - lps[1] : 20 };   // margin in nats (large = confident)
}
async function extract(payload, entry, cfg, fetchImpl) {
  const body = { model: EDGE_MODEL.id, prompt: buildPrompt(entry, payload, cfg.shots), max_tokens: 300,
    temperature: cfg.T, seed: SEED + cfg.rep, stream: true, logprobs: TOPK,
    guided_json: entry.jsonSchema, guided_decoding_backend: EDGE_MODEL.guided_decoding_backend };
  const ctl = new AbortController(); const dl = setTimeout(() => ctl.abort(), 60_000);
  try {
    const res = await fetchImpl(VLLM_URL, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: ctl.signal });
    if (!res.ok || !res.body) return { status: `http_${res.status}` };
    const dec = new TextDecoder(); const pos = new JsonPos();
    let buf = '', text = ''; const s = [], h = [], m = [], toks = [];
    for await (const chunk of res.body) {
      buf += dec.decode(chunk, { stream: true }); let i;
      while ((i = buf.indexOf('\n\n')) !== -1) {
        const line = buf.slice(0, i).split('\n').find((l) => l.startsWith('data:')); buf = buf.slice(i + 2);
        if (!line) continue; const data = line.slice(5).trim(); if (data === '[DONE]') continue;
        let j; try { j = JSON.parse(data); } catch { continue; }
        const ch = j.choices?.[0]; if (!ch) continue;
        const tok = ch.text ?? ''; text += tok;
        const isValue = pos.feed(tok); const lp = ch.logprobs?.token_logprobs?.[0];
        if (lp != null && isValue) { const st = topkStats(ch.logprobs?.top_logprobs?.[0], lp); s.push(r4(-lp)); h.push(r4(st.h)); m.push(r4(st.m)); toks.push(tok); }
      }
    }
    let parsed; try { parsed = JSON.parse(text); } catch { return { status: 'unparseable', s, h, m, toks }; }
    if (!entry.validate(parsed)) return { status: 'schema_invalid', s, h, m, toks };
    return { status: 'done', parsed, s, h, m, toks };
  } catch (e) { return { status: ctl.signal.aborted ? 'deadline' : 'stream_error', error: String(e).slice(0, 120) }; }
  finally { clearTimeout(dl); }
}

// ── DRY_RUN: a fake server, so the pipeline + analysis can be proven without a GPU ─────
function fakeFetch(_url, opts) {
  const b = JSON.parse(opts.body); const key = b.guided_json.properties.acoustic_sigma ? 'telemetry' : 'property_audit';
  const pl = payloads.find((p) => b.prompt.endsWith(turn(SCHEMAS[key].prompt(p))));
  let h = 2166136261; for (const c of pl.req_id + b.temperature + b.prompt.length) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619) >>> 0; }
  const rnd = () => { h = Math.imul(h ^ (h >>> 15), 2246822507) >>> 0; return h / 2 ** 32; };
  const truth = structuredClone(pl.metadata.synthetic_truth); const wrongKey = rnd() < 0.25 ? Object.keys(truth)[Math.floor(rnd() * Object.keys(truth).length)] : null;
  if (wrongKey) truth[wrongKey] = typeof truth[wrongKey] === 'boolean' ? !truth[wrongKey] : Array.isArray(truth[wrongKey]) ? [] : truth[wrongKey] == null ? 1 : null;
  const parts = JSON.stringify(truth).split(/([{}\[\],:])/).filter(Boolean); const enc = new TextEncoder();
  const wrongVal = wrongKey ? JSON.stringify(truth[wrongKey]) : null; let afterWrongKey = false;
  return Promise.resolve({ ok: true, status: 200, body: (async function* () {
    for (const t of parts) {
      const isWrong = afterWrongKey && wrongVal != null && wrongVal.includes(t) && !'{}[],:'.includes(t);
      if (t === `"${wrongKey}"`) afterWrongKey = true; else if (t === ',') afterWrongKey = false;
      const lp = '{}[],:'.includes(t) ? 0 : -(isWrong && rnd() < 0.6 ? 0.05 + rnd() * 0.3 : rnd() * 0.02) * (1 + b.temperature);
      yield enc.encode(`data: ${JSON.stringify({ choices: [{ text: t, logprobs: { token_logprobs: [lp], top_logprobs: [{ [t]: lp, x: Math.log(Math.max(1e-9, 1 - Math.exp(lp))) }] } }] })}\n\n`);
    }
    yield enc.encode('data: [DONE]\n\n');
  })() });
}

// ── grid ─────────────────────────────────────────────────────────────────────────────
const configs = [];
for (const shots of SHOTS) for (const T of TEMPS) for (let rep = 0; rep < (T > 0 ? REPEATS : 1); rep++)
  configs.push({ id: `T${T}_shot${shots}${T > 0 && REPEATS > 1 ? `_r${rep}` : ''}`, T, shots, rep });
writeFileSync(OUT, JSON.stringify(provenanceHeader({ kind: 'guard-grid', dry_run: DRY, endpoint: DRY ? 'dry-run' : VLLM_URL,
  corpus: ROOT.replace(process.cwd() + '/', ''), n_payloads: payloads.length, topk: TOPK, seed: SEED, configs,
  few_shot_examples: SHOT_EXAMPLES, baseline: 'T0_shot0 (= the configuration behind paper Table VIII)' })) + '\n');
console.log(`${configs.length} configurations × ${payloads.length} payloads → ${OUT}${DRY ? '   [DRY RUN — numbers are meaningless]' : ''}`);

const fetchImpl = DRY ? fakeFetch : fetch;
for (const cfg of configs) {
  const t0 = Date.now(); let next = 0, correct = 0, bad = 0; const lines = [];
  const worker = async () => {
    while (next < payloads.length) {
      const p = payloads[next++]; const entry = admit(p).entry; const truth = p.metadata.synthetic_truth;
      const r = await extract(p, entry, cfg, fetchImpl);
      const diff = r.status === 'done' ? fieldDiff(r.parsed, truth, entry.judgeKeys) : ['__' + r.status];
      const ok = diff.length === 0; if (ok) correct++; if (r.status !== 'done') bad++;
      lines.push(JSON.stringify({ cfg: cfg.id, T: cfg.T, shots: cfg.shots, rep: cfg.rep, req_id: p.req_id, kind: entry.key, style: p.metadata.style,
        status: r.status, correct_truth: ok, diff_edge: diff, s: r.s ?? [], h: r.h ?? [], m: r.m ?? [], toks: r.toks ?? [] }));
    }
  };
  await Promise.all(Array.from({ length: CONC }, worker));
  appendFileSync(OUT, lines.sort().join('\n') + '\n');
  console.log(`  ${cfg.id.padEnd(14)} accuracy ${(100 * correct / payloads.length).toFixed(1)}%  (${correct}/${payloads.length})  non-done ${bad}  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
}
console.log(`done. Next: python3 analysis/guard_grid.py ${OUT.replace(process.cwd() + '/', '')}`);
