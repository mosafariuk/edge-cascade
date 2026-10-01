// schemas.mjs — extraction schema registry for constrained decoding.
//
// DESIGN RULE (important): the LLM extracts only PRIMITIVES it can read directly
// off the text. Any field that is arithmetic or boolean LOGIC over those
// primitives is computed in CODE (`derive`), never asked of a 4-bit model.
// Constrained decoding guarantees the output PARSES — it does NOT guarantee the
// model did the math right. So we don't let it do math.
import Ajv from 'ajv';
const ajv = new Ajv({ allErrors: true, coerceTypes: false, strict: false });

// ── Pattern 1: Property / rating audit (ETL) ─────────────────────────────────
// The model reads values off the text; discrepancy_flag is DERIVED in code.
const PropertyExtract = {
  type: 'object', additionalProperties: false,
  properties: {
    current_value_gbp:       { type: 'integer', description: 'current rateable value, £' },
    previous_value_gbp:      { type: ['integer', 'null'], description: 'previous value, £' },
    cohort_avg_increase_pct: { type: ['number', 'null'], description: 'cohort baseline % increase' },
    effective_date_raw:      { type: ['string', 'null'], description: 'effective date, as written' },
    assets:                  { type: 'array', items: { type: 'string' } },
  },
  // EVERY field is required (nullable where the text may not state it). With optional
  // fields the grammar lets the model close the object early; an omission is a structural
  // token, so it is invisible to the value-surprisal guard (measured on the first live run:
  // a stated 12.5% baseline was silently dropped at signal 0.000). Requiring the key forces
  // an explicit value-or-null at a VALUE position, which the guard can score. It also makes
  // the schema legal for strict structured-output modes on the heavy path.
  required: ['current_value_gbp', 'previous_value_gbp', 'cohort_avg_increase_pct', 'effective_date_raw', 'assets'],
};
// Date normalizer (SEED PATTERN — extend as real payloads reveal more formats).
// The model emits `effective_date_raw` verbatim (a string); code resolves it.
// Returns { effective_date: ISO|null, date_ambiguous: bool }.
const MONTHS = { jan: 1, feb: 2, mar: 3, apr: 4, may: 5, jun: 6, jul: 7, aug: 8, sep: 9, oct: 10, nov: 11, dec: 12 };
export function normalizeDate(raw, refYear = null) {
  if (!raw || typeof raw !== 'string') return { effective_date: null, date_ambiguous: false };
  const s = raw.trim();
  let m;
  // ISO: 2024-04-01
  if ((m = s.match(/^(\d{4})-(\d{2})-(\d{2})$/))) return { effective_date: `${m[1]}-${m[2]}-${m[3]}`, date_ambiguous: false };
  // "01-Apr" / "1 April" — no year → resolvable only with refYear; else ambiguous
  if ((m = s.match(/^(\d{1,2})[-\s]([A-Za-z]{3,})/))) {
    const mon = MONTHS[m[2].slice(0, 3).toLowerCase()];
    if (mon && refYear) return { effective_date: `${refYear}-${String(mon).padStart(2, '0')}-${m[1].padStart(2, '0')}`, date_ambiguous: false };
    if (mon) return { effective_date: null, date_ambiguous: true };   // month known, year missing
  }
  // DD/MM/YYYY or MM/DD/YYYY — genuinely ambiguous, flag for escalation
  if (/^\d{1,2}\/\d{1,2}\/\d{2,4}$/.test(s)) return { effective_date: null, date_ambiguous: true };
  return { effective_date: null, date_ambiguous: true };              // unrecognized → flag, don't guess
}

function propertyDerive(x, ctx = {}) {
  const out = { ...x };
  if (x.current_value_gbp != null && x.previous_value_gbp) {
    const inc = ((x.current_value_gbp - x.previous_value_gbp) / x.previous_value_gbp) * 100;
    out.increase_pct = Math.round(inc * 10) / 10;
    out.discrepancy_flag =
      x.cohort_avg_increase_pct != null ? inc > x.cohort_avg_increase_pct : null;
  } else {
    out.increase_pct = null; out.discrepancy_flag = null;
  }
  // date normalization in code (refYear from batch metadata when available)
  Object.assign(out, normalizeDate(x.effective_date_raw, ctx.refYear ?? null));
  // plausibility guard for shorthand-currency misreads ("45k" → 45 drops the ×1000).
  // Rateable values are effectively never < £500; flag for review rather than trust.
  out.value_suspect = x.current_value_gbp != null && x.current_value_gbp < 500;
  return out;
}

// ── Pattern 2: Real-time telemetry (anomaly router) ──────────────────────────
// The model classifies primitives; requires_deep_diagnostic is DERIVED in code.
const TelemetryExtract = {
  type: 'object', additionalProperties: false,
  properties: {
    acoustic_sigma:              { type: ['number', 'null'], description: 'pitch elevation in σ' },
    gaze_deviation_duration_sec: { type: ['number', 'null'] },
    gaze_offscreen:              { type: 'boolean' },
    // posture / biomechanical stress marker — REAL sample B ("rigid contraction")
    // needs this or requires_deep_diagnostic under-counts to 1 and returns false.
    posture_rigid:               { type: 'boolean', description: 'rigid contraction / posture anomaly present' },
    baseline_nominal:            { type: 'boolean' },
  },
  required: ['acoustic_sigma', 'gaze_deviation_duration_sec', 'gaze_offscreen', 'posture_rigid', 'baseline_nominal'],
};
const T_SIGMA = 2.0, T_GAZE = 3.0;
function telemetryDerive(x) {
  // count EACH independent stress marker in code; the "multiple models triggering
  // simultaneously" outcome is the derived AND, never asked of the 4-bit model.
  const markers =
    ((x.acoustic_sigma ?? 0) >= T_SIGMA ? 1 : 0) +
    ((x.gaze_deviation_duration_sec ?? 0) >= T_GAZE ? 1 : 0) +
    (x.gaze_offscreen === true ? 1 : 0) +
    (x.posture_rigid === true ? 1 : 0);
  let event_classification = 'nominal_baseline';
  if ((x.acoustic_sigma ?? 0) >= T_SIGMA) event_classification = 'acoustic_stress_spike';
  else if ((x.gaze_deviation_duration_sec ?? 0) >= T_GAZE || x.gaze_offscreen || x.posture_rigid === true)
    event_classification = 'attention_deviation';
  return { ...x, event_classification, marker_count: markers, requires_deep_diagnostic: markers >= 2 };
}

// ── INBOUND payload contracts (validate the QUEUE message before extraction) ──
// Malformed ingestion is dead-lettered, never fed to the model. additionalProperties
// stays open (upstream may add fields) but the load-bearing fields are required+typed.
const PropertyInbound = {
  type: 'object', additionalProperties: true,
  required: ['req_id', 'raw_text'],
  properties: {
    req_id: { type: 'string' }, source: { type: 'string' },
    raw_text: { type: 'string', minLength: 1 }, metadata: { type: 'object' },
  },
};
const TelemetryInbound = {
  type: 'object', additionalProperties: true,
  required: ['req_id', 'raw_telemetry'],
  properties: {
    req_id: { type: 'string' }, stream_type: { type: 'string' },
    raw_telemetry: { type: 'string', minLength: 1 }, timestamp: { type: 'number' },
  },
};

// ── registry ─────────────────────────────────────────────────────────────────
export const SCHEMAS = {
  property_audit: {
    key: 'property_audit',
    inbound: ajv.compile(PropertyInbound),
    jsonSchema: PropertyExtract,
    validate: ajv.compile(PropertyExtract),
    derive: propertyDerive,
    judgeKeys: ['current_value_gbp', 'previous_value_gbp', 'cohort_avg_increase_pct', 'assets'],
    // In-context examples (hand-written; values occur in no evaluation corpus). They show the
    // two conventions a zero-shot model violates most: `null` for a field the text does not
    // state, and nothing invented in a list. FROZEN: the confirmation run (paper §VI-G) used
    // exactly these; changing them invalidates Table VIII.
    examples: [
      { payload: { raw_text: 'Annexe C, Harbour Yard. Current rateable value £31,750. A neighbouring unit is assessed at £29,000. No ancillary items.' },
        answer: { current_value_gbp: 31750, previous_value_gbp: null, cohort_avg_increase_pct: null, effective_date_raw: null, assets: [] } },
      { payload: { raw_text: 'Suite 9, Old Brewery: curr val 27.3k, prev 22,950. Cohort avg incr 6.4%. Eff. 12-Jan. Incl. cycle shelter.' },
        answer: { current_value_gbp: 27300, previous_value_gbp: 22950, cohort_avg_increase_pct: 6.4, effective_date_raw: '12-Jan', assets: ['cycle shelter'] } },
    ],
    field: 'raw_text',
    prompt: (p) => `Extract the fields as strict JSON. Do not compute or infer; copy values as written. Use null for any field the text does not state. "assets" lists only ancillary items the text says the property includes (e.g. parking spaces), never the property itself; use [] if none.\nTEXT: ${p.raw_text}\nJSON:`,
  },
  telemetry: {
    key: 'telemetry',
    inbound: ajv.compile(TelemetryInbound),
    jsonSchema: TelemetryExtract,
    validate: ajv.compile(TelemetryExtract),
    derive: telemetryDerive,
    judgeKeys: ['acoustic_sigma', 'gaze_deviation_duration_sec', 'gaze_offscreen', 'posture_rigid', 'baseline_nominal'],
    // In-context examples: a boolean marker is `true` ONLY when the telemetry states it.
    // FROZEN — see property_audit.examples.
    examples: [
      { payload: { raw_telemetry: '[01:07:33] Pitch elevated +1.6σ. Frame buffer at 212. Baseline deviation flagged.' },
        answer: { acoustic_sigma: 1.6, gaze_deviation_duration_sec: null, gaze_offscreen: false, posture_rigid: false, baseline_nominal: false } },
      { payload: { raw_telemetry: '[00:42:10] Gaze off-screen for 5.3s; rigid posture detected. Baseline stress metrics nominal.' },
        answer: { acoustic_sigma: null, gaze_deviation_duration_sec: 5.3, gaze_offscreen: true, posture_rigid: true, baseline_nominal: true } },
    ],
    field: 'raw_telemetry',
    prompt: (p) => `Extract the fields as strict JSON. Do not infer; copy observed values. Use null for any numeric field the telemetry does not state; set a boolean marker true only if the telemetry states it, otherwise false.\nTELEMETRY: ${p.raw_telemetry}\nJSON:`,
  },
};

/** Route a raw ingestion payload to its schema. Prefers an explicit `kind`
 *  discriminator; falls back to shape. Returns null if unroutable. */
export function routeSchema(payload) {
  if (payload.kind && SCHEMAS[payload.kind]) return SCHEMAS[payload.kind];
  if (payload.stream_type || payload.raw_telemetry) return SCHEMAS.telemetry;
  if (payload.source || payload.raw_text) return SCHEMAS.property_audit;
  return null;
}

/** Full inbound gate: route + validate envelope. Returns
 *  { entry, ok:true } or { ok:false, reason, errors } for dead-lettering. */
export function admit(payload) {
  const entry = routeSchema(payload);
  if (!entry) return { ok: false, reason: 'unroutable' };
  if (!entry.inbound(payload)) return { ok: false, reason: 'inbound_invalid', errors: entry.inbound.errors, entry };
  return { ok: true, entry };
}
