// mock-vllm.mjs — minimal, dependency-free stand-in for a vLLM OpenAI-compatible
// /v1/completions endpoint. Streams SSE frames with `logprobs.top_logprobs` so the
// entropy guard can be exercised for real.
//
//   * prompt containing "[HARD]"  -> near-UNIFORM top-k  (H≈ln(k) high) -> guard aborts
//   * otherwise                    -> SHARP distribution  (H≈0 low)      -> completes
//   * TOKEN_MS  simulates per-token generation latency (default 4ms)
//   * respects client disconnect (AbortController from the guard) -> stops work
import http from 'node:http';

const PORT     = Number(process.env.PORT || 8000);
const TOKEN_MS = Number(process.env.TOKEN_MS || 4);
const K        = Number(process.env.LOGPROBS_K || 20);
const ln = Math.log;

function dist(high, step) {
  const toks = Array.from({ length: K }, (_, i) => `t${i}`);
  const d = {};
  if (high) { const p = 1 / K; for (const t of toks) d[t] = ln(p); return d; }   // H≈ln(K)
  // low entropy: dominant mass on a UNIQUE token per step (avoids false repetition)
  d[`w${step}`] = ln(0.95);
  for (let i = 1; i < K; i++) d[toks[i]] = ln(0.05 / (K - 1));
  return d;
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

http.createServer((req, res) => {
  if (req.url === '/health') { res.writeHead(200); return res.end('ok'); }
  if (req.method !== 'POST' || !req.url.startsWith('/v1/completions')) { res.writeHead(404); return res.end(); }

  let body = '';
  req.on('data', (c) => (body += c));
  req.on('end', async () => {
    let j = {}; try { j = JSON.parse(body); } catch {}
    const prompt = j.prompt || '';
    const maxTok = j.max_tokens || 64;
    const high = /\[HARD\]/.test(prompt);
    res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' });
    res.flushHeaders();                                        // push headers immediately
    let closed = false;
    res.on('close', () => { closed = true; });                 // client (guard) aborted

    // ── guided_json mode: emit a schema-shaped JSON object token by token ──
    // MOCK_MASK=post (default) → structural tokens are grammar-forced (peaked
    // top_logprobs, entropy≈0). MOCK_MASK=pre → structural tokens carry a natural
    // spread (entropy≈ln K). Lets vllm-probe.mjs exercise both verdicts.
    if (j.guided_json || j.response_format) {
      const mask = process.env.MOCK_MASK || 'post';
      const toks = ['{', '"flag"', ':', 'true', '}'];  // structural + value
      const peaked = (t) => ({ [t]: ln(0.999) });
      const spread = () => Object.fromEntries(
        Array.from({ length: K }, (_, i) => [`c${i}`, ln(1 / K)]));  // natural, H≈lnK
      for (let i = 0; i < toks.length && !closed; i++) {
        const t = toks[i];
        const structural = /^[{}[\]:,"]/.test(t);
        const tl = structural && mask === 'post' ? peaked(t)
                 : structural && mask === 'pre' ? spread()
                 : peaked(t);                              // values always peaked
        const chunk = { choices: [{ text: t,
          logprobs: { tokens: [t], token_logprobs: [Object.values(tl)[0]], top_logprobs: [tl] }, finish_reason: null }] };
        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
        await sleep(TOKEN_MS);
      }
      if (!closed) { res.write('data: [DONE]\n\n'); res.end(); }
      return;
    }

    for (let i = 0; i < maxTok && !closed; i++) {
      const d = dist(high, i);
      const top = Object.keys(d)[0];
      const chunk = { choices: [{ text: (i ? ' ' : '') + top,
        logprobs: { tokens: [top], token_logprobs: [d[top]], top_logprobs: [d] }, finish_reason: null }] };
      res.write(`data: ${JSON.stringify(chunk)}\n\n`);         // small frames: no backpressure await needed
      await sleep(TOKEN_MS);
    }
    if (!closed) { res.write('data: [DONE]\n\n'); res.end(); }
  });
}).listen(PORT, () => console.log(`[mock-vllm] SSE /v1/completions on :${PORT} (TOKEN_MS=${TOKEN_MS}, K=${K})`));
