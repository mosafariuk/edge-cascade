// bootstrap-model.mjs — force the MiniLM ONNX weights onto disk BEFORE any native run.
//
// WHY THIS EXISTS (artifact-evaluation prerequisite)
//
//   src/embed-native.mjs drives onnxruntime-node directly and calls findModel(), which
//   only ever *looks for* an already-cached file:
//
//     node_modules/@xenova/transformers/.cache/Xenova/all-MiniLM-L6-v2/onnx/model_quantized.onnx
//     ~/.cache/Xenova/all-MiniLM-L6-v2/onnx/model_quantized.onnx
//     $ONNX_MODEL
//
//   createNativeEmbedder() calls AutoTokenizer.from_pretrained(), which caches the
//   *tokenizer* — but nothing in the native path ever downloads the *model*. On a clean
//   machine the very first worker therefore dies with:
//
//     Error: MiniLM onnx not found; set ONNX_MODEL (run once via @xenova to cache it)
//
//   Verified on a fresh Ubuntu 24.04 bare-metal host: tokenizer.json and
//   tokenizer_config.json were cached, onnx/ was absent, every worker exited non-zero.
//
//   Running the high-level @xenova pipeline once populates onnx/model_quantized.onnx
//   (~23 MB), after which the native path resolves it locally and needs no network.
//
// USAGE
//   node bench/bootstrap-model.mjs          # or: npm run bootstrap-model
//   ONNX_MODEL=/path/to/model.onnx ...      # skip entirely if supplying your own
//
// Exit code 0 = model present and verified. Non-zero = do not proceed to benchmarks.

import { existsSync, statSync } from 'node:fs';
import { homedir } from 'node:os';

const MODEL = process.env.MODEL_ID || 'Xenova/all-MiniLM-L6-v2';
const CANDIDATES = [
  process.env.ONNX_MODEL,
  `node_modules/@xenova/transformers/.cache/${MODEL}/onnx/model_quantized.onnx`,
  `${homedir()}/.cache/${MODEL}/onnx/model_quantized.onnx`,
].filter(Boolean);

const found = () => CANDIDATES.find((p) => existsSync(p));

const pre = found();
if (pre) {
  console.log(`model already cached: ${pre} (${(statSync(pre).size / 1e6).toFixed(1)} MB)`);
  process.exit(0);
}

console.log(`model not cached — fetching ${MODEL} (quantized) via @xenova pipeline...`);

const t = await import('@xenova/transformers');
t.env.allowRemoteModels = true;

let pipe;
try {
  pipe = await t.pipeline('feature-extraction', MODEL, { quantized: true });
} catch (e) {
  console.error(`FAILED to fetch ${MODEL}: ${e.message}`);
  console.error('Network access to huggingface.co is required for this one-time step.');
  console.error('Offline alternative: place model_quantized.onnx anywhere and set ONNX_MODEL.');
  process.exit(1);
}

// Prove the weights actually execute, not merely that a file landed on disk.
const out = await pipe('bootstrap verification sentence', { pooling: 'mean', normalize: true });
const dims = out.data.length;
const norm = Math.sqrt(Array.from(out.data).reduce((a, b) => a + b * b, 0));

const post = found();
if (!post) {
  console.error('FAILED: pipeline ran but no model_quantized.onnx appeared in any candidate path.');
  console.error('Searched:'); CANDIDATES.forEach((p) => console.error(`  ${p}`));
  process.exit(1);
}

console.log(`cached: ${post} (${(statSync(post).size / 1e6).toFixed(1)} MB)`);
console.log(`verified: dims=${dims} l2norm=${norm.toFixed(4)}`);

if (dims !== 384) { console.error(`FAILED: expected 384 dims, got ${dims}`); process.exit(1); }
if (Math.abs(norm - 1) > 1e-3) { console.error(`FAILED: not L2-normalized (${norm})`); process.exit(1); }

console.log('bootstrap OK — native onnxruntime-node path can now resolve the model offline.');
