// models.mjs — the exact models behind the paper's guard-calibration table (§VI-G,
// Table VII) and the embedding benchmarks (§VI-A). Single source of truth: every
// runtime default, probe, and benchmark artifact reads from here, so a number in the
// paper can always be traced to the model that produced it. Change here only.
//
// Fill every TODO from the live stack before running bench/vllm-probe.mjs or
// bench/shadow-run.mjs:
//   curl -s $VLLM/v1/models | jq '.data[].id'     → EDGE_MODEL.id
//   pip show vllm | grep ^Version                  → EDGE_MODEL.vllm
//   nvidia-smi --query-gpu=name --format=csv,noheader → EDGE_MODEL.gpu
//   HF model card / config.json quantization block → EDGE_MODEL.hf_repo, .quant
'use strict';

/** The local ("edge") model served by vLLM with guided_json + logprobs. */
export const EDGE_MODEL = Object.freeze({
  id: 'Qwen/Qwen3-8B-AWQ',       // served model name as vLLM exposes it at /v1/models
  hf_repo: 'Qwen/Qwen3-8B-AWQ',  // Hugging Face path of the weights (official Qwen3 AWQ release)
  revision: '4da05a8edb55c6046cce958586c33b61da07bb79', // HF commit sha of the weights
  quant: 'AWQ 4-bit, group size 128 (official Qwen3-8B-AWQ release)', // cite [awq] in the paper
  vllm: '0.9.2',                 // vllm/vllm-openai:v0.9.2 (Qwen3 support, guided_json + xgrammar)
  gpu: 'NVIDIA GeForce RTX 3090 24 GB (vast.ai, PCIe x16, 21.9 GB/s)',
  guided_decoding_backend: 'xgrammar',
  // Qwen3 is a chat model served through /v1/completions: wrap the instruction in its
  // ChatML turn format with an empty think block (thinking disabled) so the constrained
  // JSON starts immediately. 'raw' sends the instruction unwrapped.
  prompt_format: 'chatml-nothink',
  // In-context examples prepended to every extraction prompt (schemaEntry.examples). The
  // confirmation run chose 2: held-out accuracy 78.3% (0-shot) → 91.5% (1) → 97.0% (2).
  few_shot: 2,
});

/** The frontier model used as ground-truth judge in shadow labeling and as the escalation target. */
export const JUDGE_MODEL = Object.freeze({
  provider: 'openai',            // 'anthropic' | 'openai'  (HEAVY_PROVIDER overrides)
  id: 'claude-opus-4-8',         // used only when provider === 'anthropic' (HEAVY_MODEL overrides)
  openai_id: 'gpt-4o',           // the judge: OpenAI Chat Completions, response_format json_schema
  pinned_on: '2026-10-01',       // date of the shadow-labeling pass; API resolved gpt-4o → gpt-4o-2024-08-06
});

/** The embedding model behind every throughput number (routing stage). Verified values. */
export const EMBED_MODEL = Object.freeze({
  id: 'Xenova/all-MiniLM-L6-v2',
  file: 'onnx/model_quantized.onnx',
  quant: 'INT8 (dynamic)',
  bytes: 22_972_370,
  sha256: 'afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1',
  dim: 384,
  onnxruntime_node: '1.27.0',
});

/** The guard configuration that paper Table VIII was measured with. The worker runs THIS or
 *  refuses to start (see resolveGuardConfig in pipeline-worker.mjs). */
export const GUARD_EVALUATED = Object.freeze({
  shots: 2,                      // in-context examples per schema (schemaEntry.examples)
  maxSurprisal: 0,               // nats — escalate on any measurable hesitation
  statistic: 'mean-all',         // mean surprisal over every value token of the object
  resolution_nats: 1e-4,
  source: 'bench/results-zen5-run2/guard-confirm.txt',
});

/** Wrap an instruction in the edge model's prompt format. */
export function wrapEdgePrompt(instruction, format = EDGE_MODEL.prompt_format) {
  if (format === 'chatml-nothink') return `<|im_start|>user\n${instruction}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
  if (format === 'chatml') return `<|im_start|>user\n${instruction}<|im_end|>\n<|im_start|>assistant\n`;
  return instruction;
}

/** The full edge prompt: `shots` solved example turns from the schema entry, then the open
 *  turn for this payload. Used by the worker AND by the benchmarks, so what is evaluated is
 *  what is deployed. Falls back to a single wrapped instruction for non-chat formats. */
export function buildEdgePrompt(schemaEntry, payload, shots = EDGE_MODEL.few_shot, format = EDGE_MODEL.prompt_format) {
  if (format !== 'chatml-nothink' && format !== 'chatml') return wrapEdgePrompt(schemaEntry.prompt(payload), format);
  let p = '';
  for (const ex of (schemaEntry.examples ?? []).slice(0, shots)) {
    p += wrapEdgePrompt(schemaEntry.prompt(ex.payload), format) + JSON.stringify(ex.answer) + '<|im_end|>\n';
  }
  return p + wrapEdgePrompt(schemaEntry.prompt(payload), format);
}

/** Everything above, plus the fields that vary per run, for embedding in artifact headers. */
export function provenanceHeader(extra = {}) {
  return {
    kind: 'provenance',
    generated_at: new Date().toISOString(),
    edge_model: EDGE_MODEL, judge_model: JUDGE_MODEL, embed_model: EMBED_MODEL,
    node: process.version,
    ...extra,
  };
}

/** Fail loudly if a TODO is still in place when a benchmark that depends on it runs. */
export function assertModelsDeclared(which = ['edge', 'judge']) {
  const missing = [];
  if (which.includes('edge'))  for (const [k, v] of Object.entries(EDGE_MODEL))  if (v === 'TODO') missing.push(`EDGE_MODEL.${k}`);
  if (which.includes('judge')) for (const [k, v] of Object.entries(JUDGE_MODEL)) if (v === 'TODO') missing.push(`JUDGE_MODEL.${k}`);
  if (missing.length) throw new Error(`src/models.mjs still has TODO placeholders: ${missing.join(', ')} — fill them before producing paper artifacts`);
}
