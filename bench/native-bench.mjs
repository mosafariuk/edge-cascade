// native_bench.mjs — correctness + latency of the native embedder, and the
// decisive test: intraOpNumThreads=1 vs default (all cores).
import { createNativeEmbedder } from './embed-native.mjs';

const THREADS = Number(process.env.ORT_INTRA_OP || 1);
console.log(`onnxruntime-node, intraOpNumThreads=${THREADS === 0 ? 'default(all cores)' : THREADS}`);
const { embed, inputNames, outputNames, modelPath } = await createNativeEmbedder({ intraOpNumThreads: THREADS });
console.log('model:', modelPath.split('/').slice(-3).join('/'));
console.log('inputs:', inputNames.join(','), '| output:', outputNames[0]);

const text = 'classify this incoming record for the routing gateway';
// correctness sanity: normalized 384-dim vector
const [v] = await embed([text]);
const norm = Math.sqrt(v.reduce((a, x) => a + x * x, 0));
console.log(`dim=${v.length}  L2norm=${norm.toFixed(4)} (expect 1.000)  v[0..3]=${v.slice(0,3).map(x=>x.toFixed(3))}`);

for (let i = 0; i < 20; i++) await embed([text]);                 // warmup
let n = 200, t = performance.now();
for (let i = 0; i < n; i++) await embed([text]);
const dSingle = (performance.now() - t) / n;
console.log(`\nSINGLE embed: ${dSingle.toFixed(2)} ms  => ${(1000/dSingle).toFixed(0)}/s/core`);
for (const b of [8, 16, 32]) {
  const arr = Array.from({ length: b }, () => text);
  const t2 = performance.now();
  for (let i = 0; i < 20; i++) await embed(arr);
  const ms = (performance.now() - t2) / 20;
  console.log(`BATCH b=${String(b).padStart(2)}: ${ms.toFixed(2)} ms/batch => ${(b/ms*1000).toFixed(0)}/s/core, ${(ms/b).toFixed(2)} ms/item`);
}
// hold briefly so an external `ps` sampler can read this process's CPU%
if (process.env.HOLD) { const end = Date.now() + Number(process.env.HOLD) * 1000;
  while (Date.now() < end) await embed([text]); }
