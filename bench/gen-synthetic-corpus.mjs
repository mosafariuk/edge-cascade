// gen-synthetic-corpus.mjs — build the synthetic calibration corpus (paper §VI-G).
//
// GROUND TRUTH IS CONSTRUCTED, NOT INFERRED. For every record the field values are
// sampled HERE, in code, from a seeded PRNG; an LLM is used only as a *writer* that
// turns those facts into a terse, noisy field record in one of five styles. The truth
// travels with the payload in `metadata.synthetic_truth` (never shown to any model: the
// extraction prompts read only raw_text / raw_telemetry). The shadow harness can then
// score the edge model against truth it did not get from another LLM, and score the
// frontier judge against the same truth — a direct measurement of judge fidelity.
//
// No personal data: every value is a random draw; nothing is derived from real records.
//
//   OPENAI_API_KEY=… node bench/gen-synthetic-corpus.mjs
//   N_PER_KIND=250 SEED=20261001 WRITER_MODEL=gpt-4o OUT=payloads/synthetic node bench/…
import { mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { SCHEMAS } from '../src/schemas.mjs';

const N = Number(process.env.N_PER_KIND || 250);
const SEED = Number(process.env.SEED || 20261001);
const WRITER = process.env.WRITER_MODEL || 'gpt-4o';
const OUT = process.env.OUT || 'payloads/synthetic';
const BATCH = 10, CONC = 8;

// ── seeded PRNG (mulberry32) ─────────────────────────────────────────────────
let _s = SEED >>> 0;
const rnd = () => { _s |= 0; _s = (_s + 0x6D2B79F5) | 0; let t = Math.imul(_s ^ (_s >>> 15), 1 | _s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const pick = (a) => a[Math.floor(rnd() * a.length)];
const chance = (p) => rnd() < p;
const round = (x, q) => Math.round(x / q) * q;
const dec1 = (x) => Math.round(x * 10) / 10;

const STYLES = ['clean', 'shorthand', 'distractor', 'negation', 'reordered'];
const ASSETS = ['2 parking spaces', '1 loading bay', 'roof terrace', 'basement storage unit', '3 EV charging points',
  'external signage rights', 'bicycle store', 'mezzanine level', '4 parking spaces', 'rear service yard'];
const DATES = ['01-Apr', '1 April 2026', '2026-04-01', '01/04/2026', '15-Oct', '30 September 2025', '2025-10-15'];
const PLACES = ['Unit 4, Ground Floor Retail', '3rd Floor Office Suite, EC1A', 'Warehouse B, Trafford Park', 'Kiosk 2, Station Concourse',
  'First Floor Studio, Northern Quarter', 'Unit 12, Riverside Trade Park', 'Corner Shop, 81 High Street', 'Workshop 7, Canal Mills'];

function propertyTruth() {
  const current = round(Math.exp(Math.log(3000) + rnd() * (Math.log(400000) - Math.log(3000))), pick([50, 100, 500]));
  const previous = chance(0.7) ? Math.max(500, round(current / (1 + (rnd() * 0.7 - 0.1)), 50)) : null;
  const cohort = chance(0.6) ? dec1(2 + rnd() * 23) : null;
  const date = chance(0.5) ? pick(DATES) : null;
  const k = pick([0, 0, 1, 1, 2, 3]); const assets = [];
  while (assets.length < k) { const a = pick(ASSETS); if (!assets.some((x) => x.replace(/^\d+ /, '') === a.replace(/^\d+ /, ''))) assets.push(a); }
  return { current_value_gbp: current, previous_value_gbp: previous === current ? previous + 50 : previous,
    cohort_avg_increase_pct: cohort, effective_date_raw: date, assets };
}
function telemetryTruth() {
  return {
    acoustic_sigma: chance(0.7) ? dec1(0.3 + rnd() * 4.2) : null,
    gaze_deviation_duration_sec: chance(0.6) ? dec1(0.2 + rnd() * 8.8) : null,
    gaze_offscreen: chance(0.4), posture_rigid: chance(0.3), baseline_nominal: chance(0.5),
  };
}

const STYLE_RULES = {
  clean: 'Plain, explicit sentences. Every stated value written in full.',
  shorthand: 'Terse clerk/log shorthand: abbreviations, dropped articles, compact numbers. A money value may be abbreviated ONLY if exact (45000 → "45k", 18200 → "18.2k"); otherwise write it in full.',
  distractor: 'Include exactly one or two extra numbers that clearly belong to something else (a neighbouring unit, a floor area, a frame count, a coordinate, another channel) and cannot be mistaken for a requested field by a careful reader.',
  negation: 'For every field that is null / false / empty, say so explicitly in words (e.g. "no previous assessment on record", "gaze remained on-screen", "no ancillary assets").',
  reordered: 'State the facts in an unusual order, embedding values mid-sentence; do not use labelled key: value pairs.',
};
const PROPERTY_RULES = `Each record is a UK commercial property rating note. Facts:
- current_value_gbp: the CURRENT rateable value in pounds (must be recoverable exactly).
- previous_value_gbp: the PREVIOUS rateable value, or null. If null, do not state any previous value (unless the style says to state its absence).
- cohort_avg_increase_pct: the cohort/baseline average percentage increase, or null. If null, state no cohort percentage.
- effective_date_raw: the effective date string; if given it must appear VERBATIM, character for character. If null, give no effective date.
- assets: ancillary items the property includes; each phrase must appear VERBATIM and UNABBREVIATED, exactly as given, in every style. If empty, mention no ancillary items at all (or say there are none). Never describe the property itself as an asset.
Open with the given premises name. 1–3 sentences. Write every number in digits. Never compute or state the percentage change between the two values. The only percentage in the text is the cohort figure (none if it is null). Any distractor must be something other than a rateable value labelled previous/current, a percentage, or an ancillary item.`;
const TELEMETRY_RULES = `Each record is one line of athlete-monitoring telemetry, beginning with a [hh:mm:ss] timestamp. Facts:
- acoustic_sigma: pitch elevation in σ (write like "+2.4σ"), or null → do not report any pitch figure.
- gaze_deviation_duration_sec: seconds of gaze deviation (write like "4.2s"), or null → report no duration.
- gaze_offscreen: true → state the gaze is off-screen; false → do not say it is off-screen.
- posture_rigid: true → state rigid posture / rigid contraction; false → do not report rigidity.
- baseline_nominal: true → state that baseline stress metrics are nominal; false → do NOT say the baseline is nominal (you may say a baseline deviation is flagged).
Begin with the given timestamp exactly. 1–3 short sentences. Do not add stress markers that are not in the facts. A distractor must never be a σ value, a duration in seconds, or a statement about gaze, posture or baseline.`;


// ── mechanical faithfulness check: can the constructed truth be read off the text? ──
// Independent of every model. A record that fails is regenerated; after RETRIES it falls
// back to a deterministic template, so the published corpus is faithful by verification.
const moneyForms = (n) => { const f = [String(n), n.toLocaleString('en-GB')]; if (n % 10 === 0) f.push(`${+(n / 1000).toFixed(2)}k`); return f; };
const hasAny = (t, forms) => forms.some((f) => t.includes(f.toLowerCase()));
const numForms = (x) => (Number.isInteger(x) ? [String(x), x.toFixed(1)] : [String(x)]);
const ASSET_NOUNS = ['parking', 'loading bay', 'roof terrace', 'storage', 'charging', 'signage', 'bicycle', 'mezzanine', 'service yard'];
export function faithful(kind, truth, text) {
  const t = text.toLowerCase(), bad = [];
  if (kind === 'property_audit') {
    if (!hasAny(t, moneyForms(truth.current_value_gbp))) bad.push('current value not recoverable');
    if (truth.previous_value_gbp != null && !hasAny(t, moneyForms(truth.previous_value_gbp))) bad.push('previous value missing');
    if (truth.previous_value_gbp == null && /(prev\w*|prior|former\w*)[^.;]{0,40}?\d/.test(t)) bad.push('a previous value is stated but truth is null');
    const pctRe = /(\d+(?:\.\d+)?)\s?(%|percent|pct)/g; const pcts = [...t.matchAll(pctRe)].map((m) => m[1]);
    if (truth.cohort_avg_increase_pct != null && !pcts.some((p) => numForms(truth.cohort_avg_increase_pct).includes(p) || Number(p) === truth.cohort_avg_increase_pct)) bad.push('cohort % missing');
    if (truth.cohort_avg_increase_pct == null && pcts.length) bad.push('a percentage is stated but cohort is null');
    if (pcts.length > 1) bad.push('more than one percentage');
    if (truth.effective_date_raw != null && !t.includes(truth.effective_date_raw.toLowerCase())) bad.push('date not verbatim');
    for (const a of truth.assets) if (!t.includes(a.toLowerCase())) bad.push(`asset not verbatim: ${a}`);
    for (const n of ASSET_NOUNS) if (!truth.assets.some((a) => a.toLowerCase().includes(n)) && t.includes(n) && !new RegExp(`(no|zero|without)[^.;]{0,30}${n}`).test(t)) bad.push(`asset noun present but not in truth: ${n}`);
  } else {
    const body = t.replace(/^\[[^\]]*\]/, '');
    const sig = [...body.matchAll(/(\d+(?:\.\d+)?)\s?σ/g)].map((m) => Number(m[1]));
    if (truth.acoustic_sigma != null && !sig.includes(truth.acoustic_sigma)) bad.push('σ value missing');
    if (truth.acoustic_sigma == null && (sig.length || /σ|sigma/.test(body))) bad.push('a σ value is stated but truth is null');
    if (sig.length > 1) bad.push('more than one σ value');
    const dur = [...body.matchAll(/(\d+(?:\.\d+)?)\s?(?:s\b|sec\b|secs\b|seconds?\b)/g)].map((m) => Number(m[1]));
    if (truth.gaze_deviation_duration_sec != null && !dur.includes(truth.gaze_deviation_duration_sec)) bad.push('gaze duration missing');
    if (truth.gaze_deviation_duration_sec == null && dur.length) bad.push('a duration is stated but truth is null');
    if (dur.length > 1) bad.push('more than one duration');
    const flag = (has, neg) => has.test(body) && !neg.test(body);
    const off = flag(/off-?screen/, /(not|no|never|without|n't)[^.;]{0,25}off-?screen/);
    const rig = flag(/rigid/, /(not|no|never|without|absent|n't)[^.;]{0,30}rigid|rigid\w*[^.;]{0,25}(absent|not detected|not present|not observed)/);
    const nom = flag(/nominal/, /(not|non-?|no longer|n't)[^.;]{0,15}nominal/);
    if (off !== truth.gaze_offscreen) bad.push(`gaze_offscreen reads ${off}`);
    if (rig !== truth.posture_rigid) bad.push(`posture_rigid reads ${rig}`);
    if (nom !== truth.baseline_nominal) bad.push(`baseline_nominal reads ${nom}`);
  }
  return bad;
}
function template(kind, it) {
  const x = it.truth;
  if (kind === 'property_audit') return `${it.place}. Current rateable value £${x.current_value_gbp.toLocaleString('en-GB')}.`
    + (x.previous_value_gbp != null ? ` Previous rateable value £${x.previous_value_gbp.toLocaleString('en-GB')}.` : '')
    + (x.cohort_avg_increase_pct != null ? ` Cohort average increase ${x.cohort_avg_increase_pct}%.` : '')
    + (x.effective_date_raw != null ? ` Effective ${x.effective_date_raw}.` : '')
    + (x.assets.length ? ` Property includes ${x.assets.join(', ')}.` : ' No ancillary items.');
  return `${it.ts}`
    + (x.acoustic_sigma != null ? ` Pitch elevated +${x.acoustic_sigma}σ.` : '')
    + (x.gaze_deviation_duration_sec != null ? ` Gaze deviation ${x.gaze_deviation_duration_sec}s.` : '')
    + (x.gaze_offscreen ? ' Gaze off-screen.' : '')
    + (x.posture_rigid ? ' Rigid posture detected.' : '')
    + (x.baseline_nominal ? ' Baseline stress metrics nominal.' : ' Baseline deviation flagged.');
}

async function writeBatch(client, kind, items) {
  const rules = kind === 'property_audit' ? PROPERTY_RULES : TELEMETRY_RULES;
  const lines = items.map((it, i) => `#${i + 1} style=${it.style}${it.place ? ` premises="${it.place}"` : ''}${it.ts ? ` timestamp=${it.ts}` : ''}${it.problems?.length ? ` FIX-PREVIOUS-ATTEMPT(${it.problems.join('; ')})` : ''} facts=${JSON.stringify(it.truth)}`).join('\n');
  const res = await client.chat.completions.create({
    model: WRITER, temperature: 0.9, max_tokens: 2500,
    response_format: { type: 'json_schema', json_schema: { name: 'records', strict: true,
      schema: { type: 'object', additionalProperties: false, required: ['texts'],
        properties: { texts: { type: 'array', items: { type: 'string' } } } } } },
    messages: [
      { role: 'system', content: `You write short, realistic operational records from structured facts. Each text must state exactly the given facts — nothing a reader could mistake for an extra or different value of a requested field. Return one text per item, in order.\n\n${rules}\n\nStyles:\n${Object.entries(STYLE_RULES).map(([k, v]) => `- ${k}: ${v}`).join('\n')}` },
      { role: 'user', content: `Write ${items.length} records.\n${lines}` },
    ],
  });
  const texts = JSON.parse(res.choices[0].message.content).texts;
  if (!Array.isArray(texts) || texts.length !== items.length) throw new Error(`writer returned ${texts?.length} texts for ${items.length} items`);
  return texts;
}

const OpenAI = (await import('openai')).default;
const client = new OpenAI();
rmSync(OUT, { recursive: true, force: true });
const manifest = { kind: 'synthetic-corpus', generated_at: new Date().toISOString(), seed: SEED, writer_model: WRITER,
  n_per_kind: N, styles: STYLES, note: 'truth sampled in code from the seeded PRNG; the writer model only verbalizes it; every text is mechanically verified to state the truth (faithful()), regenerated on failure, templated as last resort', counts: {} };

for (const kind of ['property_audit', 'telemetry']) {
  const items = Array.from({ length: N }, (_, i) => ({
    id: `syn_${kind === 'property_audit' ? 'prop' : 'tele'}_${String(i + 1).padStart(4, '0')}`,
    style: STYLES[i % STYLES.length],
    place: kind === 'property_audit' ? pick(PLACES) : null,
    ts: kind === 'telemetry' ? `[${String(Math.floor(rnd() * 2)).padStart(2, '0')}:${String(Math.floor(rnd() * 60)).padStart(2, '0')}:${String(Math.floor(rnd() * 60)).padStart(2, '0')}]` : null,
    truth: kind === 'property_audit' ? propertyTruth() : telemetryTruth(),
  }));
  for (const it of items) if (!SCHEMAS[kind].validate(it.truth)) throw new Error(`sampled truth violates schema: ${JSON.stringify(SCHEMAS[kind].validate.errors)}`);
  const batches = []; for (let i = 0; i < items.length; i += BATCH) batches.push(items.slice(i, i + BATCH));
  let done = 0;
  const run = async (b) => { for (let a = 0; ; a++) { try { const t = await writeBatch(client, kind, b); b.forEach((it, i) => { it.text = t[i]; }); break; } catch (e) { if (a >= 3) throw e; } } done += b.length; process.stdout.write(`\r${kind}: ${done}/${N}`); };
  for (let i = 0; i < batches.length; i += CONC) await Promise.all(batches.slice(i, i + CONC).map(run));
  process.stdout.write('\n');
  // verify → regenerate failures (feeding the problems back) → template fallback
  const RETRIES = 3; let regen = 0, fallback = 0;
  for (let round = 0; round <= RETRIES; round++) {
    const failing = items.filter((it) => { it.problems = faithful(kind, it.truth, it.text); return it.problems.length; });
    console.log(`${kind}: verification round ${round}: ${failing.length} unfaithful`);
    if (!failing.length) break;
    if (round === RETRIES) { for (const it of failing) { it.text = template(kind, it); it.writer = 'template'; fallback++; const p = faithful(kind, it.truth, it.text); if (p.length) throw new Error(`template unfaithful: ${p} :: ${it.text}`); } break; }
    regen += failing.length;
    const fb = []; for (let i = 0; i < failing.length; i += BATCH) fb.push(failing.slice(i, i + BATCH));
    for (let i = 0; i < fb.length; i += CONC) await Promise.all(fb.slice(i, i + CONC).map(async (b) => { const t = await writeBatch(client, kind, b); b.forEach((it, j) => { it.text = t[j]; }); }));
  }
  manifest.verification ??= {}; manifest.verification[kind] = { regenerated: regen, template_fallback: fallback };
  mkdirSync(join(OUT, kind), { recursive: true });
  for (const it of items) {
    const meta = { synthetic: true, style: it.style, writer: it.writer ?? 'llm', ref_year: 2026, synthetic_truth: it.truth };
    const payload = kind === 'property_audit'
      ? { req_id: it.id, source: 'synthetic', raw_text: it.text, metadata: meta }
      : { req_id: it.id, stream_type: 'synthetic', raw_telemetry: it.text, timestamp: 1790000000, metadata: meta };
    writeFileSync(join(OUT, kind, `${it.id}.json`), JSON.stringify(payload, null, 2) + '\n');
  }
  manifest.counts[kind] = Object.fromEntries(STYLES.map((s) => [s, items.filter((x) => x.style === s).length]));
}
writeFileSync(join(OUT, 'MANIFEST.json'), JSON.stringify(manifest, null, 2) + '\n');
console.log(`wrote ${2 * N} payloads + MANIFEST.json under ${OUT}/`);
