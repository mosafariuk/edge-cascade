// vllm-probe.mjs — settle the one open question: does your vLLM return logprobs
// computed AFTER the guided-decoding grammar mask?
//
// Sends a minimal single-boolean guided_json schema, finds the first STRUCTURAL
// token (the opening `{` — a position where the grammar allows exactly one token),
// and measures its entropy over the returned top_logprobs.
//   entropy ≈ 0  → grammar mask applied to logprobs  → VERDICT: POST-MASK
//   entropy high → raw distribution returned         → VERDICT: PRE-MASK
//
// POST-MASK means the value-surprisal guard in src/constrained.mjs is the correct
// design (entropy is defeated at structure). PRE-MASK means the original entropy
// guard would still work. Either way you stop guessing.
//
// Run against the mock (proves the probe logic, both modes):
//   PORT=8000 node bench/mock-vllm.mjs &
//   MOCK_MASK=post node bench/vllm-probe.mjs      # → POST-MASK
//   MOCK_MASK=pre  node bench/mock-vllm.mjs & ; node bench/vllm-probe.mjs   # → PRE-MASK
// Run against your real server:
//   VLLM_URL=http://<host>:8000/v1/completions node bench/vllm-probe.mjs
import { shannonEntropy } from '../src/cascade-lib.mjs';
import { EDGE_MODEL, provenanceHeader } from '../src/models.mjs';

const URL = process.env.VLLM_URL || 'http://127.0.0.1:8000/v1/completions';
// MODEL env is for the mock only; a real probe must use the declared edge model so the
// verdict is attributable (the paper cites this probe's measured entropy and verdict).
const MODEL = process.env.MODEL || EDGE_MODEL.id;
if (MODEL === 'TODO') { console.error('EDGE_MODEL.id is still TODO in src/models.mjs (or set MODEL=... for the mock)'); process.exit(2); }
const SCHEMA = { type: 'object', additionalProperties: false,
  properties: { flag: { type: 'boolean' } }, required: ['flag'] };

const body = {
  model: MODEL, prompt: 'Return {"flag": true} as JSON.', max_tokens: 20,
  temperature: 0.0, stream: true, logprobs: 20,
  guided_json: SCHEMA, guided_decoding_backend: 'xgrammar',
};

console.log(`probing ${URL} (single-boolean guided_json)…`);
let res;
try {
  res = await fetch(URL, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
} catch (e) { console.error('CANNOT REACH vLLM:', e.message); process.exit(2); }
if (!res.ok || !res.body) { console.error('bad response:', res.status); process.exit(2); }

const dec = new TextDecoder();
let buf = '', structuralEntropy = null, firstStructuralTok = null;
for await (const chunk of res.body) {
  buf += dec.decode(chunk, { stream: true });
  let i;
  while ((i = buf.indexOf('\n\n')) !== -1) {
    const line = buf.slice(0, i).split('\n').find((l) => l.startsWith('data:')); buf = buf.slice(i + 2);
    if (!line) continue;
    const data = line.slice(5).trim();
    if (data === '[DONE]') break;
    let j; try { j = JSON.parse(data); } catch { continue; }
    const ch = j.choices?.[0]; if (!ch) continue;
    const tok = (ch.text ?? '').trim();
    const tl = ch.logprobs?.top_logprobs?.[0];
    // first structural token = a grammar-forced position (`{`)
    if (structuralEntropy === null && /^[{[]/.test(tok) && tl) {
      structuralEntropy = shannonEntropy(tl);
      firstStructuralTok = tok;
      break;
    }
  }
  if (structuralEntropy !== null) break;
}

if (structuralEntropy === null) {
  console.error('no structural token with logprobs seen — is logprobs enabled? is guided_json supported?');
  process.exit(2);
}
const nats = structuralEntropy;
const verdict = nats < 0.1 ? 'POST-MASK' : nats > 0.5 ? 'PRE-MASK' : 'AMBIGUOUS';
console.log(`structural token ${JSON.stringify(firstStructuralTok)}  entropy=${nats.toFixed(4)} nats`);
console.log(`VERDICT: ${verdict}`);
// Self-documenting record: one JSON line with the verdict AND the models/stack that
// produced it. Redirect the probe's stdout into bench/results-zen5/ to retain it.
console.log(JSON.stringify(provenanceHeader({
  kind: 'vllm-probe', endpoint: URL, model_requested: MODEL, structural_token: firstStructuralTok,
  entropy_nats: Number(nats.toFixed(6)), verdict,
})));
console.log(verdict === 'POST-MASK'
  ? '→ value-surprisal guard (src/constrained.mjs) is correct; calibrate MAX_SURPRISAL.'
  : verdict === 'PRE-MASK'
  ? '→ the original entropy guard (src/entropy-guard.mjs) would also work here.'
  : '→ inconclusive; try more structural samples or a longer schema.');
