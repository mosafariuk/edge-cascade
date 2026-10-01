// verify-constrained.mjs — proves the constrained-decoding layer:
// schema validation, code-derived fields, JSON value-position tracking, and the
// value-surprisal guard (structural/key tokens ignored) end-to-end via a mock.
import assert from 'node:assert/strict';
import { SCHEMAS, routeSchema } from '../src/schemas.mjs';
import { extractStructured, JsonPos } from '../src/constrained.mjs';

let pass = 0; const ok = (n) => { console.log(`  ✓ ${n}`); pass++; };
const tick = () => new Promise((r) => setImmediate(r));

// ── 1. routing + Ajv validation ──────────────────────────────────────────────
console.log('1. routing + schema validation');
assert.equal(routeSchema({ raw_text: 'x', source: 'reg' }), SCHEMAS.property_audit);
assert.equal(routeSchema({ raw_telemetry: 'y', stream_type: 'audio' }), SCHEMAS.telemetry);
const FULL = { current_value_gbp: 18200, previous_value_gbp: null, cohort_avg_increase_pct: null, effective_date_raw: null, assets: [] };
assert.equal(SCHEMAS.property_audit.validate(FULL), true);
assert.equal(SCHEMAS.property_audit.validate({ current_value_gbp: 18200, assets: [] }), false);   // nullable fields must still be PRESENT
assert.equal(SCHEMAS.property_audit.validate({ assets: [] }), false);         // missing required
assert.equal(SCHEMAS.property_audit.validate({ ...FULL, junk: 1 }), false); // additionalProperties
ok('property/telemetry routing + accept-valid + reject-missing + reject-extra');

// ── 2. code-derived fields (NOT asked of the LLM) ────────────────────────────
console.log('2. derived fields computed in code');
{
  const d = SCHEMAS.property_audit.derive({ current_value_gbp: 18200, previous_value_gbp: 14500, cohort_avg_increase_pct: 12, assets: ['2 parking spaces'] });
  assert.equal(d.increase_pct, 25.5);                                          // (18200-14500)/14500 = 25.5%
  assert.equal(d.discrepancy_flag, true);                                      // 25.5% > 12% cohort
  const d2 = SCHEMAS.property_audit.derive({ current_value_gbp: 15000, previous_value_gbp: 14500, cohort_avg_increase_pct: 12, assets: [] });
  assert.equal(d2.discrepancy_flag, false);                                    // 3.4% < 12%
  assert.equal(SCHEMAS.property_audit.derive({ current_value_gbp: 9, assets: [] }).discrepancy_flag, null); // no prev → unknown
}
{
  const t = SCHEMAS.telemetry.derive({ acoustic_sigma: 2.4, gaze_deviation_duration_sec: 4.2, gaze_offscreen: true, baseline_nominal: false });
  assert.equal(t.marker_count, 3);                                             // σ≥2, gaze≥3s, offscreen
  assert.equal(t.event_classification, 'acoustic_stress_spike');
  assert.equal(t.requires_deep_diagnostic, true);                             // ≥2 markers
  const t2 = SCHEMAS.telemetry.derive({ acoustic_sigma: 0.3, gaze_deviation_duration_sec: 0.1, gaze_offscreen: false, baseline_nominal: true });
  assert.equal(t2.event_classification, 'nominal_baseline');
  assert.equal(t2.requires_deep_diagnostic, false);
}
ok('property discrepancy math + telemetry marker AND — all in code, deterministic');

// ── 3. JsonPos: keys/structure excluded, values included ─────────────────────
console.log('3. JSON value-position machine (grammar-forced keys excluded)');
{
  const p = new JsonPos(); const cls = [];
  for (const t of ['{', '"k"', ':', '123', ',', '"m"', ':', '"ab"', '}']) cls.push([t, p.feed(t)]);
  const valued = cls.filter(([, v]) => v).map(([t]) => t);
  assert.deepEqual(valued, ['123', '"ab"'], `only values counted, got ${JSON.stringify(valued)}`);
  ok(`keys("k","m") + structure excluded; values(123,"ab") counted`);
}

// ── mock vLLM stream: tokenize JSON, per-run logprob; keys forced (~0), values set ──
function jsonTokens(json, valueLogprob) {
  const toks = []; let run = '';
  const flush = () => { if (run) { toks.push({ text: run, lp: valueLogprob }); run = ''; } };
  for (const c of json) {
    if ('{}[],:"'.includes(c)) { flush(); toks.push({ text: c, lp: -0.001 }); }  // structure ~forced
    else run += c;
  }
  flush();
  return toks;
}
function mockFetch(tokens) {
  return async () => ({ ok: true, status: 200, body: (async function* () {
    const enc = new TextEncoder();
    for (const t of tokens) {
      const frame = { choices: [{ text: t.text, logprobs: { token_logprobs: [t.lp], top_logprobs: [{ [t.text]: t.lp }] } }] };
      yield enc.encode(`data: ${JSON.stringify(frame)}\n\n`);
    }
    yield enc.encode('data: [DONE]\n\n');
  })() });
}
const runExtract = (json, valueLp, opts = {}) => extractStructured({
  payload: { raw_text: 'x', source: 's' }, schemaEntry: SCHEMAS.property_audit,
  fetchImpl: mockFetch(jsonTokens(json, valueLp)), ...opts,
});

// ── 4. confident stream → done + validated + derived ─────────────────────────
console.log('4. end-to-end: confident extraction → done + derived');
{
  const json = '{"current_value_gbp":18200,"previous_value_gbp":14500,"cohort_avg_increase_pct":12,"effective_date_raw":null,"assets":["2 parking spaces"]}';
  const r = await runExtract(json, -0.05, { k: 3, maxSurprisal: 3.0 });
  assert.equal(r.status, 'done', `expected done, got ${JSON.stringify(r)}`);
  assert.equal(r.data.discrepancy_flag, true);
  assert.equal(r.data.increase_pct, 25.5);
  assert.deepEqual(r.data.assets, ['2 parking spaces']);
  ok('parsed + Ajv-validated + discrepancy derived (flag=true, +25.5%)');
}

// ── 5. low-confidence VALUES → escalate (keys/structure did NOT mask it) ──────
console.log('5. end-to-end: unsure values → escalate low_value_confidence');
{
  const json = '{"current_value_gbp":18200,"previous_value_gbp":14500,"cohort_avg_increase_pct":12,"effective_date_raw":null,"assets":["x"]}';
  const r = await runExtract(json, -4.0, { k: 3, maxSurprisal: 3.0 });   // value surprisal 4.0 > 3.0
  assert.equal(r.status, 'escalate');
  assert.equal(r.reason, 'low_value_confidence');
  assert.ok(r.meanSurprisal > 3.0);
  ok(`escalated on value surprisal ${r.meanSurprisal.toFixed(2)} (structural ~0 did not dilute)`);
}

// ── 6. structural-heavy but confident → NOT escalated (the whole point) ──────
console.log('6. many structural tokens, confident values → NOT escalated');
{
  // deeply nested/structural JSON; values confident. Crude classifier would have
  // been fooled by all the braces; JsonPos + surprisal is not.
  const json = '{"current_value_gbp":1,"previous_value_gbp":1,"cohort_avg_increase_pct":0,"effective_date_raw":null,"assets":["a","b","c"]}';
  const r = await runExtract(json, -0.05, { k: 3, maxSurprisal: 3.0 });
  assert.equal(r.status, 'done');
  ok('structural tokens ignored; confident values pass');
}

// ── 7. schema-invalid output (missing required) → escalate ───────────────────
console.log('7. guided output missing required field → escalate schema_invalid');
{
  const r = await runExtract('{"assets":[]}', -0.05, { k: 3 });   // no current_value_gbp
  assert.equal(r.status, 'escalate');
  assert.equal(r.reason, 'schema_invalid');
  ok('Ajv belt-and-suspenders caught missing required field → escalate');
}

// ── 8. inbound admission gate (dead-letter malformed queue payloads) ─────────
console.log('8. inbound admission gate');
{
  const { admit } = await import('../src/schemas.mjs');
  assert.equal(admit({ req_id: 'a1', raw_text: 'Unit 4 £18,200', source: 'reg' }).ok, true);
  assert.equal(admit({ req_id: 'f9', raw_telemetry: 'pitch +2.4σ', stream_type: 'audio' }).entry.key, 'telemetry');
  assert.equal(admit({ kind: 'property_audit', req_id: 'k1', raw_text: 'x' }).entry.key, 'property_audit'); // explicit kind
  assert.deepEqual({ ...admit({ raw_text: 'x' }) }.reason, 'inbound_invalid');   // missing req_id
  assert.equal(admit({ nonsense: 1 }).reason, 'unroutable');
  ok('accepts valid property/telemetry, honors kind, rejects missing req_id + unroutable');
}

// ── 9. balanced threshold calibration (rate quantile) ────────────────────────
console.log('9. balanced maxSurprisal calibration');
{
  const { calibrateSurprisalThreshold } = await import('../src/constrained.mjs');
  const samples = Array.from({ length: 100 }, (_, i) => i / 100);   // 0.00..0.99 uniform
  const thr = calibrateSurprisalThreshold(samples, 0.2);            // want ~20% to exceed
  assert.ok(thr >= 0.78 && thr <= 0.82, `80th pctile ~0.80, got ${thr}`);
  const exceed = samples.filter((s) => s > thr).length;
  assert.ok(exceed >= 18 && exceed <= 22, `~20% escalate, got ${exceed}%`);
  ok(`threshold=${thr.toFixed(2)} → ${exceed}% escalate (balanced ~20% target)`);
}

// ── 10. date normalizer ──────────────────────────────────────────────────────
console.log('10. date normalizer (code-side)');
{
  const { normalizeDate } = await import('../src/schemas.mjs');
  assert.deepEqual(normalizeDate('2024-04-01'), { effective_date: '2024-04-01', date_ambiguous: false });
  assert.deepEqual(normalizeDate('01-Apr'), { effective_date: null, date_ambiguous: true });          // month known, year missing
  assert.deepEqual(normalizeDate('01-Apr', 2024), { effective_date: '2024-04-01', date_ambiguous: false }); // refYear resolves it
  assert.equal(normalizeDate('01/04/2024').date_ambiguous, true);                                       // DD/MM vs MM/DD
  // wired into derive():
  const d = SCHEMAS.property_audit.derive({ current_value_gbp: 18200, previous_value_gbp: 14500, cohort_avg_increase_pct: 12, effective_date_raw: '01-Apr', assets: [] });
  assert.equal(d.date_ambiguous, true);
  assert.equal(d.discrepancy_flag, true);   // arithmetic still correct alongside
  ok('ISO parsed, "01-Apr" flagged ambiguous (year missing), refYear resolves, DD/MM flagged');
}

// ── 11. PRODUCTION samples end-to-end derive (the extracted primitives → derived) ──
console.log('11. real production samples → derive()');
{
  const P = SCHEMAS.property_audit, T = SCHEMAS.telemetry;

  // VOA Sample A (voa_audit_mn_092): 25.5% > 12.5% cohort → discrepancy; 01-Apr + ref_year 2026 → ISO
  const a = P.derive({ current_value_gbp: 18200, previous_value_gbp: 14500, cohort_avg_increase_pct: 12.5,
    effective_date_raw: '01-Apr', assets: ['2 parking spaces'] }, { refYear: 2026 });
  assert.equal(a.increase_pct, 25.5);
  assert.equal(a.discrepancy_flag, true);
  assert.equal(a.effective_date, '2026-04-01');
  assert.equal(a.date_ambiguous, false);
  assert.equal(a.value_suspect, false);

  // VOA Sample B (voa_audit_ld_104): "45k" → 45000; "unchanged"/no previous → increase null; empty assets
  const b = P.derive({ current_value_gbp: 45000, assets: [] }, { refYear: 2026 });
  assert.equal(b.increase_pct, null);
  assert.equal(b.discrepancy_flag, null);
  assert.equal(b.value_suspect, false);
  // the "45k → 45" misread would be caught:
  assert.equal(P.derive({ current_value_gbp: 45, assets: [] }).value_suspect, true);

  // Telemetry Sample A (99482): σ2.4 + gaze 4.2s + offscreen → 3 markers, deep diagnostic
  const ta = T.derive({ acoustic_sigma: 2.4, gaze_deviation_duration_sec: 4.2, gaze_offscreen: true, baseline_nominal: true });
  assert.equal(ta.marker_count, 3);
  assert.equal(ta.event_classification, 'acoustic_stress_spike');
  assert.equal(ta.requires_deep_diagnostic, true);

  // Telemetry Sample B (99483): σ3.1 + posture_rigid, gaze center-fixed → 2 markers, deep diagnostic
  const tb = T.derive({ acoustic_sigma: 3.1, gaze_offscreen: false, posture_rigid: true, baseline_nominal: false });
  assert.equal(tb.marker_count, 2);
  assert.equal(tb.requires_deep_diagnostic, true);
  // WITHOUT the posture_rigid primitive this would be 1 → false (the bug the real sample exposed):
  const tbNoPosture = T.derive({ acoustic_sigma: 3.1, gaze_offscreen: false, baseline_nominal: false });
  assert.equal(tbNoPosture.marker_count, 1);
  assert.equal(tbNoPosture.requires_deep_diagnostic, false);
  ok('4 production samples derive correctly; posture_rigid fixes Sample B (1→2 markers → deep diagnostic)');
}

// ── 12. objects with FEWER than k value tokens are still guarded ─────────────
console.log('12. short object (< k value tokens) is judged on the tokens it emitted');
{
  const json = '{"current_value_gbp":18200,"previous_value_gbp":14500,"cohort_avg_increase_pct":12,"effective_date_raw":null,"assets":["x"]}';
  const bad = await runExtract(json, -4.0, { k: 50, maxSurprisal: 3.0 });   // 4 value tokens ≪ k=50
  assert.equal(bad.status, 'escalate', `expected escalate, got ${JSON.stringify(bad)}`);
  assert.equal(bad.reason, 'low_value_confidence');
  assert.ok(bad.valueTokens < 50 && bad.valueTokens > 0);
  const good = await runExtract(json, -0.05, { k: 50, maxSurprisal: 3.0 });
  assert.equal(good.status, 'done');
  const shadow = await runExtract(json, -4.0, { k: 50, maxSurprisal: 3.0, shadow: true });
  assert.equal(shadow.status, 'done', 'shadow mode never aborts');
  assert.ok(shadow.signal > 3.0);
  ok(`short object: unsure → escalate (${bad.valueTokens} value tokens), confident → done, shadow → labelled`);
}

console.log(`\nALL ${pass} CONSTRAINED-DECODING ASSERTIONS PASSED ✅`);
