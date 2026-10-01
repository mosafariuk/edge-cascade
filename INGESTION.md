# Ingestion Contract & Constrained Extraction (locked)

Decisions: **both schemas + shape router**, **vLLM `guided_json` (native, xgrammar)**,
**balanced** escalation (~20% to the heavy model).

## 1. Queue payload contract (what upstream must XADD / publish)

Common envelope + one discriminated body. An explicit `kind` is preferred; shape is the
fallback. Malformed payloads are **dead-lettered**, never fed to the model (`admit()`).

**Property / rating audit** (`kind: "property_audit"`):
```json
{
  "kind": "property_audit",            // optional but recommended
  "req_id": "audit_88291",             // REQUIRED, string
  "source": "public_registry",
  "raw_text": "Unit 4 … £14,500 … £18,200 … 12% baseline … 2 parking spaces.",  // REQUIRED, non-empty
  "metadata": { "batch_id": "manchester_q3" }
}
```
**Real-time telemetry** (`kind: "telemetry"`):
```json
{
  "kind": "telemetry",
  "req_id": "frame_99482",             // REQUIRED, string
  "stream_type": "audio_cv_sync",
  "raw_telemetry": "[00:14:22] pitch +2.4σ; gaze X:-45 Y:12 4.2s; baseline nominal.", // REQUIRED
  "timestamp": 1718819400
}
```
Reject reasons (dead-letter, no model call): `unroutable` (no kind/shape match),
`inbound_invalid` (missing/typed-wrong `req_id`/`raw_text`/`raw_telemetry`).

## 2. What the model extracts vs. what CODE derives

The 4-bit model extracts **only primitives it can read off the text**. Every arithmetic /
logical field is computed deterministically in `derive()` — constrained decoding guarantees
the output *parses*, not that the model's *math* is right.

| Schema | LLM extracts (guided_json) | CODE derives |
|---|---|---|
| `property_audit` | `current_value_gbp`, `previous_value_gbp?`, `cohort_avg_increase_pct?`, `effective_date_raw?`, `assets[]` | `increase_pct`, `discrepancy_flag` (= increase > cohort), `effective_date`+`date_ambiguous` (normalized from raw + batch `ref_year`), `value_suspect` (= £<500, catches `45k→45` misread) |
| `telemetry` | `acoustic_sigma?`, `gaze_deviation_duration_sec?`, `gaze_offscreen`, `posture_rigid`, `baseline_nominal` | `event_classification`, `marker_count`, `requires_deep_diagnostic` (= ≥2 markers) |

**Hardened against real samples:** `posture_rigid` was added after production Sample B
(`+3.1σ` + "rigid contraction" + "multiple models triggering") — without it `marker_count`
under-counts to 1 and `requires_deep_diagnostic` returns a wrong `false`. Date normalization
uses batch `metadata.ref_year` to resolve partial dates like `01-Apr` → `2026-04-01`.

## 3. The vLLM request (native guided_json)
```jsonc
{
  "model": "<your-awq-4bit>",
  "prompt": "<schemaEntry.prompt(payload)>",
  "max_tokens": 200,
  "temperature": 0.0,                  // deterministic ETL
  "stream": true,
  "logprobs": 1,                       // needed for the value-surprisal guard
  "guided_json": <extraction JSON Schema>,
  "guided_decoding_backend": "xgrammar" // fast FSM compile + low per-token cost
}
```
**Pre-warm each schema at startup** (first request per schema compiles the grammar; xgrammar
is fast, outlines can be slow). **Verify once** that your vLLM returns *post-mask* logprobs
(entropy ~0 at a structural position) — that's the assumption the value-surprisal guard needs.

## 4. Guard under constrained decoding (value-surprisal — POST-MASK measured)
`bench/vllm-probe.mjs` against the served stack (Qwen/Qwen3-8B-AWQ, vLLM 0.9.2, xgrammar,
RTX 3090) returned **POST-MASK**: entropy at the first grammar-forced token is 0.049 nats, so a
windowed-entropy guard is defeated. The guard measures **surprisal (−logprob) at VALUE
positions only** (a `JsonPos` state machine excludes keys + structure) and escalates when the
**mean over all value tokens of the completed object** exceeds `MAX_SURPRISAL`.

Two rules learned from the first run against a real model (neither is visible with the mock):

- **Every extraction field is required and nullable.** With optional fields the grammar lets
  the model close the object early; the omission is a structural token and the guard sees
  nothing (a stated "12.5% baseline" was dropped at signal 0.000).
- **Do not window the statistic.** The first design averaged the first 6 value tokens; one
  5-digit number fills that window and later fields are never scored (AUROC 0.48 on the
  property workload = chance). `statistic: 'window-mean'` is kept only for comparison.

Belt-and-suspenders: `JSON.parse` + Ajv validate + `derive`; any failure also escalates.

## 5. Calibrated threshold (measured, held-out)
Stack and corpus: `src/models.mjs`; `payloads/synthetic/` (500 records, constructed truth,
mechanically verified); raw pairs `bench/results-zen5-run2/shadow-pairs.jsonl`; every number
below is printed by `python3 analysis/guard_eval.py`.

**`MAX_SURPRISAL = 0.027` nats** (default in code; 20%-target quantile on the 350-record
training split). On the 150 held-out records (34 edge errors):

| metric | value (Wilson 95%) |
|---|---|
| AUROC, mean over all value tokens | 0.805 (window-mean: 0.680; 0.478 on property) |
| escalation rate | 17.3% [12.1, 24.2] |
| local precision | 83.9% [76.4, 89.3] (no guard: 77.3%) |
| edge-error recall | 41.2% [26.4, 57.8] |
| edge accuracy, all 500 | 74.2% (property 84.0%, telemetry 64.4%) |
| judge (GPT-4o) vs constructed truth | 96.4% [94.4, 97.7] |

Recall reaches 68% at 29% escalation and 85% at 53%. **The guard is a filter, not a safety
case**: most edge errors are confident (a hallucinated boolean at 0.1–0.2 nats).
Earlier figures in this file (threshold 1.845, 20.8% / 98.2% / 91.8%) came from a run whose
data were not retained and could not be reproduced; they are withdrawn.

Re-fit on a sliding window as traffic drifts: `PAYLOADS_ROOT=… HOLDOUT=0.3 npm run shadow`
(needs a served edge model and a judge API key), then `python3 analysis/guard_eval.py`.

## 6. Running it
```bash
EXTRACT=1 EMBED_MODE=real EGRESS=redis MAX_SURPRISAL=<calibrated> \
  VLLM_URL=http://127.0.0.1:8000/v1/completions node src/pipeline-worker.mjs
```
Egress record: `{ data: <derived structured object>, meta: { path: 'local'|'escalated'|'dead_letter', kind }, id, t_ingest }`.
`extractHeavy()` is a stub — wire your heavy-model provider SDK (same schema, guided_json).
