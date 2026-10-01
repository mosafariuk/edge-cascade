// heavy.mjs — the escalation path. When the local 4-bit model trips the guard or
// fails validation, the payload goes to a frontier model, forced to emit the
// SAME schemaEntry.jsonSchema. The result runs through the SAME validate + derive,
// so the egress record is byte-identical regardless of which model produced it.
//
// Provider request shapes differ:
//   Anthropic — output_config.format json_schema (NOT OpenAI's response_format)
//   OpenAI    — response_format json_schema
// Both are fed schemaEntry.jsonSchema unchanged.
//
// SDKs are OPTIONAL deps, lazy-imported only on escalation, and injectable for
// tests — the worker runs fine (hash/plumbing modes) without them installed.
'use strict';
import { JUDGE_MODEL } from './models.mjs';

// Model ids come from src/models.mjs so the escalation target and the shadow-labeling
// judge are provably the same model the paper names.
const DEFAULTS = { anthropic: JUDGE_MODEL.id, openai: JUDGE_MODEL.openai_id };

/** Anthropic Messages API — structured output via output_config.format. */
async function callAnthropic(payload, schemaEntry, model, clientImpl) {
  const client = clientImpl || new (await import('@anthropic-ai/sdk')).default();
  const res = await client.messages.create({
    model: model || DEFAULTS.anthropic,
    max_tokens: 1024,
    // native Anthropic structured outputs — guarantees the first text block is
    // valid JSON matching the schema (Opus 4.8 / Sonnet 5 / Haiku 4.5)
    output_config: { format: { type: 'json_schema', schema: schemaEntry.jsonSchema } },
    messages: [{ role: 'user', content: schemaEntry.prompt(payload) }],
  });
  if (res.stop_reason === 'refusal') {
    const e = new Error('heavy_refusal'); e.category = res.stop_details?.category; throw e;
  }
  const text = res.content.find((b) => b.type === 'text')?.text ?? '';
  return JSON.parse(text);
}

/** OpenAI Chat Completions — structured output via response_format json_schema.
 *  The extraction schemas are all-required + additionalProperties:false, which is what
 *  OpenAI `strict:true` demands, so the response is guaranteed schema-valid; the SAME Ajv
 *  validate() below still runs as a belt-and-suspenders check. */
async function callOpenAI(payload, schemaEntry, model, clientImpl) {
  const client = clientImpl || new (await import('openai')).default();
  const res = await client.chat.completions.create({
    model: model || DEFAULTS.openai,
    max_tokens: 1024,
    temperature: 0,
    response_format: {
      type: 'json_schema',
      json_schema: { name: schemaEntry.key, schema: schemaEntry.jsonSchema, strict: true },
    },
    messages: [{ role: 'user', content: schemaEntry.prompt(payload) }],
  });
  return JSON.parse(res.choices[0].message.content ?? '');
}

/**
 * Escalate one payload to the heavy model. Returns the SAME derived shape the
 * local path returns (validate → derive), so egress needs zero changes.
 * Throws on refusal / unparseable / schema-invalid so the batch loop can
 * dead-letter or retry (idempotent egress makes a retry safe).
 */
export async function extractHeavy(payload, schemaEntry, {
  provider = process.env.HEAVY_PROVIDER || JUDGE_MODEL.provider,
  model = process.env.HEAVY_MODEL,
  clientImpl = null,
} = {}) {
  const raw = provider === 'openai'
    ? await callOpenAI(payload, schemaEntry, model, clientImpl)
    : await callAnthropic(payload, schemaEntry, model, clientImpl);
  if (!schemaEntry.validate(raw)) {
    const e = new Error('heavy_schema_invalid'); e.errors = schemaEntry.validate.errors; throw e;
  }
  // same ctx (batch metadata) so the heavy path's derived output matches the local path
  return schemaEntry.derive(raw, { refYear: payload?.metadata?.ref_year ?? null });
}
