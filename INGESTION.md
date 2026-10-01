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

## 4. Correctness: prompt conventions first, guard second (POST-MASK measured)
`bench/vllm-probe.mjs` against the served stack (Qwen/Qwen3-8B-AWQ, vLLM 0.9.2, xgrammar,
RTX 3090) returned **POST-MASK**: entropy at the first grammar-forced token is 0.049 nats, so a
windowed-entropy guard is defeated.

**First-order lever — the prompt.** Every extraction prompt carries two solved in-context
examples per schema (`SCHEMAS[*].examples`, built by `buildEdgePrompt`, `EDGE_MODEL.few_shot = 2`).
They teach the two conventions a zero-shot model breaks: `null` for a field the text does not
state, and a boolean marker `true` only when the text states it. The examples are FROZEN —
the confirmation run used exactly these.

**Second-order filter — the guard.** Surprisal (−logprob) is measured at VALUE positions only
(a `JsonPos` state machine excludes keys + structure) at 1e-4 nat resolution; the object
escalates when the **mean over all value tokens** exceeds `MAX_SURPRISAL`.

Rules learned against a real model (none is visible with the mock):
- **Every extraction field is required and nullable** — an omitted optional field is a
  structural token; the guard sees nothing (a stated "12.5% baseline" dropped at signal 0).
- **Do not window the statistic** — a first-6-token mean is at chance (AUROC 0.50) on the
  property workload: one multi-digit number fills the window.
- **Temperature does not help** — T 0→0.5 changes neither accuracy nor the reported logprobs.

Belt-and-suspenders: `JSON.parse` + Ajv validate + `derive`; any failure also escalates.

## 5. Measured operating point (pre-registered, held-out)
Corpus `payloads/synthetic-confirm/` (2,000 records, constructed truth, mechanically verified);
traces `bench/results-zen5-run2/guard-confirm.jsonl`; every number below is printed by
`python3 analysis/guard_confirm.py` (committed before the data were collected).

| held-out n=600, T=0 | 0-shot | 1-shot | **2-shot (deployed)** |
|---|---|---|---|
| edge accuracy | 78.3% | 91.5% | **97.0%** [95.3, 98.1] |
| errors | 130 | 51 | **18** (all `assets`) |
| AUROC of the guard signal | 0.794 | 0.896 | 0.851 |
| fitted `MAX_SURPRISAL` (20% target) | 0.021 | 0 | **0** |
| escalated | 21.0% | 22.2% | **13.7%** |
| error recall | 49.2% | 88.2% | 77.8% [54.8, 91.0] |
| silent errors, guard / no guard | 11.0% / 21.7% | 1.0% / 8.5% | **0.7% / 3.0%** |

**`MAX_SURPRISAL = 0`** (default): with the conventions in context 87% of extractions have no
measurable surprisal, so the guard is "escalate on any hesitation". Use 0.021 if you deploy
zero-shot. Judge (GPT-4o) vs constructed truth on the exploratory corpus: 96.4%.

Earlier figures in this file (threshold 1.845, 20.8% / 98.2% / 91.8%) were never reproducible
and are withdrawn. The guard is a filter, not a safety case: the surviving errors are
confident list-item mismatches that no confidence check can see.

## 6. Running it
```bash
# EXTRACT=1 and the evaluated guard (2-shot prompt, s_max=0) are the defaults; the worker
# refuses to start with a different MAX_SURPRISAL unless ALLOW_GUARD_OVERRIDE=1.
EMBED_MODE=real EGRESS=redis \
  VLLM_URL=http://127.0.0.1:8000/v1/completions node src/pipeline-worker.mjs
```
Egress record: `{ data: <derived structured object>, meta: { path: 'local'|'escalated'|'dead_letter', kind }, id, t_ingest }`.
`extractHeavy()` is a stub — wire your heavy-model provider SDK (same schema, guided_json).
