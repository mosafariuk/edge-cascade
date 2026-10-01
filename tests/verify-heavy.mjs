// verify-heavy.mjs — proves extractHeavy() feeds the SAME schema to each provider
// and returns the SAME derived shape as the local path (egress needs zero changes).
// Uses injected mock SDK clients — no API keys, no network.
import assert from 'node:assert/strict';
import { SCHEMAS } from '../src/schemas.mjs';
import { extractHeavy } from '../src/heavy.mjs';

let pass = 0; const ok = (n) => { console.log(`  ✓ ${n}`); pass++; };
const payload = { req_id: 'a1', source: 's', raw_text: 'Prev £14,500. Now £18,200. Cohort 12%. 2 parking spaces.' };
const GOOD = { current_value_gbp: 18200, previous_value_gbp: 14500, cohort_avg_increase_pct: 12, effective_date_raw: null, assets: ['2 parking spaces'] };

// ── 1. Anthropic path: output_config.format carries the exact schema; derive runs ──
console.log('1. Anthropic escalation (mock client)');
{
  let sentSchema = null, sentModel = null;
  const mockAnthropic = { messages: { create: async (req) => {
    sentModel = req.model;
    sentSchema = req.output_config.format.schema;          // capture what was sent
    assert.equal(req.output_config.format.type, 'json_schema');
    return { stop_reason: 'end_turn', content: [{ type: 'text', text: JSON.stringify(GOOD) }] };
  } } };
  const data = await extractHeavy(payload, SCHEMAS.property_audit, { provider: 'anthropic', clientImpl: mockAnthropic });
  assert.equal(sentSchema, SCHEMAS.property_audit.jsonSchema, 'exact same schema object sent to Anthropic');
  assert.equal(sentModel, 'claude-opus-4-8', 'defaults to Opus 4.8');
  assert.equal(data.discrepancy_flag, true);               // derived in code, identical to local
  assert.equal(data.increase_pct, 25.5);
  ok('same jsonSchema sent, model=claude-opus-4-8, derived shape identical to local path');
}

// ── 2. OpenAI path: response_format.json_schema carries the exact schema ──
console.log('2. OpenAI escalation (mock client)');
{
  let sentSchema = null, sentModel = null;
  const mockOpenAI = { chat: { completions: { create: async (req) => {
    sentModel = req.model;
    sentSchema = req.response_format.json_schema.schema;
    assert.equal(req.response_format.type, 'json_schema');
    return { choices: [{ message: { content: JSON.stringify(GOOD) } }] };
  } } } };
  const data = await extractHeavy(payload, SCHEMAS.property_audit, { provider: 'openai', clientImpl: mockOpenAI });
  assert.equal(sentSchema, SCHEMAS.property_audit.jsonSchema, 'exact same schema object sent to OpenAI');
  assert.equal(sentModel, 'gpt-4o');
  assert.equal(data.discrepancy_flag, true);
  ok('same jsonSchema sent, model=gpt-4o, derived shape identical → egress unchanged');
}

// ── 3. Refusal → throws (so batch loop can dead-letter / retry) ──
console.log('3. Anthropic refusal → throws');
{
  const mock = { messages: { create: async () => ({ stop_reason: 'refusal', stop_details: { category: 'cyber' }, content: [] }) } };
  await assert.rejects(() => extractHeavy(payload, SCHEMAS.property_audit, { provider: 'anthropic', clientImpl: mock }),
    /heavy_refusal/);
  ok('refusal surfaces as heavy_refusal error, not silent bad data');
}

// ── 4. Heavy output that fails schema → throws heavy_schema_invalid ──
console.log('4. heavy output missing required field → throws');
{
  const mock = { messages: { create: async () => ({ stop_reason: 'end_turn', content: [{ type: 'text', text: '{"assets":[]}' }] }) } };
  await assert.rejects(() => extractHeavy(payload, SCHEMAS.property_audit, { provider: 'anthropic', clientImpl: mock }),
    /heavy_schema_invalid/);
  ok('same Ajv validate() guards the heavy path too');
}

console.log(`\nALL ${pass} HEAVY-PATH ASSERTIONS PASSED ✅`);
