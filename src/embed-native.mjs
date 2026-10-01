// embed-native.mjs — explicit onnxruntime-node (native C++) embedder for
// all-MiniLM-L6-v2, with STRICT thread pinning at the C++ session level.
// Tokenization + mean-pool + L2-normalize done here (ort-node runs only the graph).
import ort from 'onnxruntime-node';
import { AutoTokenizer, env } from '@xenova/transformers';
import { existsSync } from 'node:fs';
import { homedir } from 'node:os';

function findModel() {
  const cands = [
    process.env.ONNX_MODEL,
    'node_modules/@xenova/transformers/.cache/Xenova/all-MiniLM-L6-v2/onnx/model_quantized.onnx',
    `${homedir()}/.cache/Xenova/all-MiniLM-L6-v2/onnx/model_quantized.onnx`,
  ].filter(Boolean);
  for (const p of cands) if (existsSync(p)) return p;
  throw new Error('MiniLM onnx not found; set ONNX_MODEL (run once via @xenova to cache it)');
}

export async function createNativeEmbedder({ model = 'Xenova/all-MiniLM-L6-v2',
  intraOpNumThreads = 1,
  // Intra-op pool spin-wait. ORT's default is to spin in userspace before parking a
  // worker thread. `false` sets session.intra_op.allow_spinning=0 through
  // SessionOptions.extra — which the shipped onnxruntime-node 1.27 binding forwards to
  // the runtime despite its .d.ts still saying "WebAssembly only" (verified: the
  // session-options trace prints the key). null = leave the runtime default.
  allowSpinning = null,
} = {}) {
  env.allowRemoteModels = true;                       // tokenizer.json may fetch once
  const tokenizer = await AutoTokenizer.from_pretrained(model);
  const modelPath = findModel();
  const opts = {
    intraOpNumThreads,                                // <-- strict C++ pinning
    interOpNumThreads: 1,
    executionMode: 'sequential',
    graphOptimizationLevel: 'all',
  };
  if (allowSpinning !== null) {
    opts.extra = { session: { intra_op: { allow_spinning: allowSpinning ? '1' : '0' } } };
  }
  const session = await ort.InferenceSession.create(modelPath, opts);
  const inNames = new Set(session.inputNames);
  const outName = session.outputNames[0];

  async function embed(texts) {
    const enc = await tokenizer(texts, { padding: true, truncation: true });
    const dims = enc.input_ids.dims;                  // [B, S]
    const [B, S] = dims;
    const big = (t) => BigInt64Array.from(t.data, (x) => BigInt(x));
    const feeds = {
      input_ids: new ort.Tensor('int64', big(enc.input_ids), dims),
      attention_mask: new ort.Tensor('int64', big(enc.attention_mask), dims),
    };
    if (inNames.has('token_type_ids')) {
      const tti = enc.token_type_ids ? big(enc.token_type_ids) : new BigInt64Array(B * S);
      feeds.token_type_ids = new ort.Tensor('int64', tti, dims);
    }
    for (const k of Object.keys(feeds)) if (!inNames.has(k)) delete feeds[k];

    const out = await session.run(feeds);
    const hid = out[outName];                         // [B, S, H]
    const H = hid.dims[2];
    const data = hid.data;                            // Float32Array
    const mask = enc.attention_mask.data;
    const result = [];
    for (let b = 0; b < B; b++) {                     // masked mean-pool + normalize
      const vec = new Float32Array(H);
      let msum = 0;
      for (let s = 0; s < S; s++) {
        const m = Number(mask[b * S + s]); if (!m) continue;
        msum += m; const base = (b * S + s) * H;
        for (let h = 0; h < H; h++) vec[h] += data[base + h] * m;
      }
      const inv = msum > 0 ? 1 / msum : 0; let norm = 0;
      for (let h = 0; h < H; h++) { vec[h] *= inv; norm += vec[h] * vec[h]; }
      norm = Math.sqrt(norm) || 1;
      for (let h = 0; h < H; h++) vec[h] /= norm;
      result.push(Array.from(vec));
    }
    return result;
  }
  return { embed, inputNames: session.inputNames, outputNames: session.outputNames, modelPath, allowSpinning };
}
