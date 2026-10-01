// constrained.mjs — schema-constrained extraction on the local vLLM model.
//
// WHY NOT the entropy guard: under guided_json the grammar masks the vocabulary,
// so entropy at STRUCTURAL positions ({ " : , }) is ~0 regardless of whether the
// model actually knows the answer. Averaging entropy over the first k tokens is
// therefore defeated by constrained decoding. Instead we measure the SURPRISAL
// (-logprob of the emitted token) at VALUE positions only — a token the model was
// forced to emit but found unlikely (low logprob among grammar-valid tokens) is
// the real "I'm guessing" signal. Structural tokens are ignored.
//
// Belt-and-suspenders: even though guided_json should guarantee parseability, we
// JSON.parse + Ajv-validate before egress, and escalate on any failure (covers
// vLLM version quirks, deadline-truncated output, etc.).
//
// GUARD: value-surprisal. bench/vllm-probe.mjs measured POST-MASK logprobs on the served
// stack (Qwen3-8B-AWQ, vLLM 0.9.2: H = 0.049 nats at the first grammar-forced token), so
// entropy at structural/key positions is useless. The guard scores -log p of the emitted
// token at VALUE positions only and decides on the mean over ALL value tokens of the object.
// Measured on held-out synthetic data with constructed truth (analysis/guard_eval.py):
// AUROC 0.81; at s_max = 0.027 nats it escalates 17%, lifts local precision 77% → 84% and
// catches 41% of edge errors. It is a useful filter, NOT a sufficient safety mechanism.
'use strict';
import { EDGE_MODEL, wrapEdgePrompt } from './models.mjs';

// Streaming JSON position tracker: classifies each emitted token as 'value'
// (string-VALUE content, number, bool, null) vs key/structure. This is what lets
// us exclude grammar-FORCED tokens (keys + punctuation, which carry no model
// uncertainty under guided_json) and measure surprisal only where the model had
// genuine choice over the content.
export class JsonPos {
  constructor() { this.stack = []; this.mode = 'value'; this.inStr = false; this.strKey = false; this.esc = false; }
  ctx() { return this.stack[this.stack.length - 1]; }
  feed(s) {
    let sawValue = false;
    for (const c of s) {
      if (this.inStr) {
        if (this.esc) { this.esc = false; if (!this.strKey) sawValue = true; continue; }
        if (c === '\\') { this.esc = true; continue; }
        if (c === '"') { this.inStr = false; continue; }
        if (!this.strKey) sawValue = true;             // string VALUE content
        continue;
      }
      if (c === '"') { this.inStr = true; this.strKey = (this.ctx() === 'obj' && this.mode === 'key'); continue; }
      if (c === '{') { this.stack.push('obj'); this.mode = 'key'; continue; }
      if (c === '[') { this.stack.push('arr'); this.mode = 'value'; continue; }
      if (c === '}' || c === ']') { this.stack.pop(); this.mode = 'value'; continue; }
      if (c === ':') { this.mode = 'value'; continue; }
      if (c === ',') { this.mode = (this.ctx() === 'obj') ? 'key' : 'value'; continue; }
      if (/\s/.test(c)) continue;
      sawValue = true;                                  // bare literal: number/true/false/null
    }
    return sawValue;
  }
}

// Build the request body. `guidedApi` picks the vLLM dialect:
//   'guided_json'      → legacy/native vLLM field (default, widest support)
//   'response_format'  → OpenAI-compatible structured outputs (newer vLLM)
function buildBody({ model, prompt, jsonSchema, maxTokens, guidedApi }) {
  const body = { model, prompt, max_tokens: maxTokens, temperature: 0.0, stream: true, logprobs: 1 };
  if (guidedApi === 'response_format') {
    body.response_format = { type: 'json_schema', json_schema: { name: 'extract', schema: jsonSchema, strict: true } };
  } else {
    body.guided_json = jsonSchema;                     // native vLLM guided decoding
    body.guided_decoding_backend = 'xgrammar';         // fast FSM compile + low per-token cost
  }
  return body;
}

/**
 * Extract structured data with a value-surprisal guard.
 * Returns { status:'done', data } | { status:'escalate', reason, ... }.
 */
export async function extractStructured({
  endpoint = 'http://127.0.0.1:8000/v1/completions',
  model = EDGE_MODEL.id,       // src/models.mjs — the served name vLLM exposes
  payload,
  schemaEntry,                 // from SCHEMAS registry
  // Decision statistic over value-token surprisals:
  //   'mean-all'    (default) mean over EVERY value token, decided when the object completes.
  //   'window-mean' legacy: mean over the first k value tokens, aborting in-stream at k. On
  //                 real data a single multi-digit number consumes the window, so errors in
  //                 later fields are never seen (AUROC ≈ chance on the property workload).
  statistic = 'mean-all',
  k = 6,                       // window size for 'window-mean'
  maxSurprisal = 0.027,        // nats; 20%-target quantile of the mean-all signal on the training split (results-zen5-run2)
  shadow = false,              // shadow-labeling: never abort on surprisal; always return `signal`
  maxTokens = 200,
  deadlineMs = 800,
  guidedApi = 'guided_json',
  promptFormat = EDGE_MODEL.prompt_format,   // src/models.mjs: chat-format wrapper for the edge model
  fetchImpl = fetch,
} = {}) {
  const controller = new AbortController();
  let deadlineHit = false;
  const deadline = setTimeout(() => { deadlineHit = true; controller.abort(); }, deadlineMs);
  if (deadline.unref) deadline.unref();
  const done = (v) => { clearTimeout(deadline); return v; };

  const body = buildBody({
    model, prompt: wrapEdgePrompt(schemaEntry.prompt(payload), promptFormat), jsonSchema: schemaEntry.jsonSchema, maxTokens, guidedApi,
  });

  let res;
  try {
    res = await fetchImpl(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body), signal: controller.signal });
  } catch (e) { return done({ status: 'escalate', reason: 'connect_error', error: String(e) }); }
  if (!res.ok || !res.body) return done({ status: 'escalate', reason: `http_${res.status ?? 'nobody'}` });

  const dec = new TextDecoder();
  let buf = '', text = '';
  const surprisals = [];                               // value-token surprisals
  const valueToks = [];                                // the value tokens themselves (shadow trace)
  const pos = new JsonPos();
  let verdict = null;
  try {
    outer:
    for await (const chunk of res.body) {
      buf += dec.decode(chunk, { stream: true });
      let i;
      while ((i = buf.indexOf('\n\n')) !== -1) {
        const line = buf.slice(0, i).split('\n').find((l) => l.startsWith('data:'));
        buf = buf.slice(i + 2);
        if (!line) continue;
        const data = line.slice(5).trim();
        if (data === '[DONE]') { verdict = { status: 'complete' }; break outer; }
        let j; try { j = JSON.parse(data); } catch { continue; }
        const ch = j.choices?.[0]; if (!ch) continue;
        const tok = ch.text ?? '';
        text += tok;
        const isValue = pos.feed(tok);                 // stateful: value content only
        // POST-MASK: surprisal of the EMITTED token at VALUE positions only
        const lp = ch.logprobs?.token_logprobs?.[0];
        if (lp != null && isValue) {
          surprisals.push(-lp); valueToks.push(tok);
          if (!shadow && statistic === 'window-mean' && surprisals.length === k) {   // shadow never aborts — it labels
            const mean = surprisals.reduce((a, b) => a + b, 0) / k;
            if (mean > maxSurprisal) { verdict = { status: 'escalate', reason: 'low_value_confidence', meanSurprisal: mean }; break outer; }
          }
        }
      }
    }
  } catch (e) {
    if (controller.signal.aborted) return done({ status: 'escalate', reason: deadlineHit ? 'deadline' : 'aborted' });
    return done({ status: 'escalate', reason: 'stream_error', error: String(e) });
  }
  if (verdict?.status === 'escalate') { controller.abort(); return done(verdict); }

  // Short objects: the in-stream check above only fires when the k-th value token
  // arrives, so an object that completes with fewer than k value tokens must be
  // judged here on the tokens it did emit, or it would bypass the guard entirely.
  if (!shadow && surprisals.length > 0 && (statistic === 'mean-all' || surprisals.length < k)) {
    const mean = surprisals.reduce((a, b) => a + b, 0) / surprisals.length;
    if (mean > maxSurprisal) {
      return done({ status: 'escalate', reason: 'low_value_confidence', meanSurprisal: mean, valueTokens: surprisals.length });
    }
  }

  // parse + validate + derive (belt-and-suspenders even with guided_json)
  let parsed;
  try { parsed = JSON.parse(text); }
  catch { return done({ status: 'escalate', reason: 'unparseable_json', raw: text.slice(0, 200) }); }
  if (!schemaEntry.validate(parsed)) {
    return done({ status: 'escalate', reason: 'schema_invalid', errors: schemaEntry.validate.errors });
  }
  // ctx carries batch metadata the code-side normalizers need (e.g. ref_year → date)
  const ctx = { refYear: payload?.metadata?.ref_year ?? null };
  // guard signal = mean surprisal over the first k value tokens (the decision statistic)
  const win = statistic === 'window-mean' ? surprisals.slice(0, k) : surprisals;
  const signal = win.length ? win.reduce((a, b) => a + b, 0) / win.length : null;
  // shadow mode also returns the full per-value-token trace so alternative decision
  // statistics can be evaluated offline without re-querying the model
  return done({ status: 'done', data: schemaEntry.derive(parsed, ctx), signal,
    ...(shadow ? { surprisals: surprisals.slice(), valueTokenTexts: valueToks.slice(), raw: parsed } : {}) });
}

/**
 * BALANCED calibration: choose maxSurprisal so ~`targetEscalationRate` of a
 * representative sample escalates. Run the local model in shadow mode over N
 * real payloads, collect each one's mean value-surprisal, then set the threshold
 * at the (1 - rate) quantile. E.g. rate=0.2 → 80th percentile → ~20% escalate.
 * (If you also have correctness labels, prefer isotonic calibration on
 *  (surprisal, was_correct?) — this rate-only form just hits a throughput target.)
 */
export function calibrateSurprisalThreshold(meanSurprisalSamples, targetEscalationRate = 0.2) {
  if (!meanSurprisalSamples.length) return Infinity;
  const s = meanSurprisalSamples.slice().sort((a, b) => a - b);
  const idx = Math.min(s.length - 1, Math.floor((1 - targetEscalationRate) * s.length));
  return s[idx];
}
