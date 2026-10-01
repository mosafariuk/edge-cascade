# edge-cascade

Reference implementation and benchmark artifact for **"Edge-Network Semantic Cascading:
Resolving CPU Oversubscription and Scaling Constraints in High-Throughput LLM Ingestion
Pipelines"** (`semantic-cascading.tex`).

An **LLM cascade** as an ingestion microservice: a CPU-resident routing stage embeds every
record with a small ONNX model; each admitted payload is extracted by a small local model
under `guided_json`, and a **value-surprisal guard** escalates the ones the local model is
not confident about to a frontier model. Both paths run the same schema validation and
code-side derivation, so the egress record is identical whichever model produced it.

> **Terminology.** This is **LLM cascading / model routing** — a query-level, *lossy* cost
> optimization — not "speculative decoding" (token-level, lossless). See paper §I-A.

## What the paper measures, and what this repo contains

| Paper | Artifact |
|---|---|
| §III, §VI-B–F: ONNX Runtime oversubscription, pinned scaling, L3/chiplet placement, latency vs load | `bench/scaling-sweep.sh`, `bench/results-zen5/*.csv`, `CONCLUSIONS.md` |
| §III-C3, Table II-a: spin-wait demonstrated by intervention (`allow_spinning=0`) | `bench/spin-proof.sh`, `bench/results-zen5-run2/spin-proof.csv` |
| §III-C3, Table II-b: intra-op vs `taskset` 2×2 ablation | `bench/ablation-2x2.sh`, `bench/results-zen5-run2/ablation-2x2.csv` |
| §III-C3, Table II-c: pool-size sweep (non-spinning thread-count loss) | `bench/pool-sweep.sh`, `bench/results-zen5-run2/pool-sweep.csv` |
| §V-A: `guided_json` masks logprobs (POST-MASK probe) | `bench/vllm-probe.mjs` |
| §V-B: JsonPos value-surprisal guard | `src/constrained.mjs` |
| §V-B/C, §VI-G, Table VIII: pre-registered confirmation — few-shot prompt as first-order lever, guard as second-order filter | `payloads/synthetic-confirm/`, `bench/optimize-guard.mjs`, `analysis/guard_confirm.py`, `bench/results-zen5-run2/guard-confirm.{jsonl,txt}` |
| §VI-G exploratory results: statistic choice, temperature grid, judge fidelity | `payloads/synthetic/`, `bench/shadow-run.mjs`, `analysis/guard_eval.py`, `analysis/guard_grid.py`, `bench/results-zen5-run2/{shadow-pairs,guard-grid}.jsonl` |
| §V-D: deterministic derivation | `src/schemas.mjs`, `payloads/` |
| §IV: adaptive batching, ack-after-commit, PEL recovery, backoff | `src/pipeline-worker.mjs`, `src/egress.mjs`, `src/cascade-lib.mjs` |

**Scope of the throughput numbers.** Every req/s figure in the paper is the throughput of the
embedding-and-routing stage (`ROUTE_TAU=2`, short synthetic payloads, no LLM in the loop). The
LLM stages are exercised by the guard/calibration tooling, not by the scaling sweep.

## Layout
```
src/
  models.mjs           # single source of truth for the edge, judge and embedding models (fill the TODOs)
  pipeline-worker.mjs  # worker: Redis Streams source, embed, router, guard/escalate, egress, PEL recovery
  embed-native.mjs     # onnxruntime-node embedder: intraOpNumThreads pinning, allow_spinning control
  cascade-lib.mjs      # RateEstimator (λ EWMA), adaptiveBatchSize, shannonEntropy  [pure]
  constrained.mjs      # guided_json extraction + JsonPos value-surprisal guard
  entropy-guard.mjs    # unconstrained path: windowed-entropy / repetition / deadline guard
  schemas.mjs          # payload schemas, admission gate, code-side derive()
  heavy.mjs            # frontier escalation (Anthropic / OpenAI), same schema → validate → derive
  egress.mjs           # EgressBuffer: micro-batch, backpressure, jittered backoff; pgvector / Redis sinks
  shadow.mjs, calibration.mjs   # shadow labelling, PAVA isotonic map, ECE, Wilson CIs
tests/
  verify.mjs             #  7 assertions — batching, rate estimator, entropy guard
  verify-egress.mjs      # 10 assertions — buffer, ack-after-commit, sink failure, backoff, bounded drain, vector-dim check
  verify-constrained.mjs # 13 assertions — schemas, JsonPos, value-surprisal guard, few-shot prompt builder
  verify-heavy.mjs       #  4 assertions — escalation path
  verify-shadow.mjs      #  6 assertions — shadow labelling, PAVA, Wilson CIs, held-out split
  verify-worker.mjs      #  6 assertions — router reachability, stable ids, per-message failure isolation, PEL recovery, enforced guard config
bench/
  scaling-sweep.sh       # pinned/unpinned worker-count sweep, n trials, t-based CIs
  spin-proof.sh          # unpinned + allow_spinning=0: direct test of the spin-wait hypothesis
  ablation-2x2.sh        # intra-op × taskset factorial at W=8,14
  pool-sweep.sh          # intraOpNumThreads 1..32 at W=8, spin off, no affinity
  load-gen.mjs           # open-loop generator + TTA percentiles
  mock-vllm.mjs          # dependency-free SSE /v1/completions with tunable entropy and mask mode
  vllm-probe.mjs         # POST-MASK / PRE-MASK verdict, self-documenting
  gen-synthetic-corpus.mjs  # synthetic corpus: truth sampled in code, LLM writes the text, mechanical faithfulness check
  shadow-run.mjs         # edge + judge over a corpus → shadow-pairs.jsonl (provenance header, traces, truth labels)
  optimize-guard.mjs     # runs prompt/temperature arms over a corpus, full per-token traces (DRY_RUN=1 without a GPU)
  GPU-RUNBOOK.md         # serving the edge model on a rented GPU for the guard runs
  audit-sample.mjs, audit-score.mjs   # human-audit subset + judge-fidelity scoring
  bootstrap-model.mjs    # downloads the MiniLM ONNX weights (required before EMBED_MODE=real)
  docker-compose.yml     # redis + postgres/pgvector + mock-vllm (+ an unused rabbitmq service)
  init.sql               # pgvector schema, vector(384)
  results-zen5/          # run 1 (2026-07-20): scaling, placement, latency CSVs, provenance, perf captures, CONCLUSIONS.md
  results-zen5-run2/     # run 2 (2026-09-30): spin-proof + 2x2 ablation CSVs, per-trial logs, PROVENANCE.txt
analysis/
  queue_sim.py           # M/D/c queue + batching simulation (illustrative; M1-era parameters)
  calibrate.py           # isotonic calibration demo on toy data
  guard_confirm.py       # PRE-REGISTERED analysis behind paper Table VIII (confirmation corpus)
  guard_eval.py          # exploratory corpus: statistic comparison, judge fidelity
  guard_grid.py          # exploratory temperature × few-shot grid
```

## Quick start (local workers; ~5 minutes)
```bash
npm ci                                  # NOT npm install — the lockfile pins onnxruntime-node 1.27.0
npm test                                # 46 assertions across 6 suites + load-gen selftest
npm run bootstrap-model                 # caches all-MiniLM-L6-v2 INT8 (22.97 MB) for the native embedder

# infrastructure: broker, pgvector (schema from bench/init.sql), mock vLLM
docker compose -f bench/docker-compose.yml up -d redis postgres mock-vllm

# one worker against the mock: local-first router → mock vLLM → entropy guard → escalate stub → pgvector
# (the evaluated cascade — EXTRACT=1, the default — needs a real served edge model; see INGESTION.md §6)
REDIS_URL=redis://127.0.0.1:6379 PG_URL=postgres://bench:bench@127.0.0.1:5432/cascade \
VLLM_URL=http://127.0.0.1:8000/v1/completions EMBED_MODE=real EGRESS=pgvector EXTRACT=0 \
WORKER_SLOT=0 node src/pipeline-worker.mjs &          # EXTRACT=0: the mock server cannot serve the real schemas

# offered load (open-loop; minimum rate is 50 req/s)
REDIS_URL=redis://127.0.0.1:6379 RATE_START=100 RATE_END=2000 DURATION=30 node bench/load-gen.mjs
```
The worker refuses to start if `documents.embedding` is not `vector(384)` (the dimension the
embedder produces), so a schema mismatch is a boot error, not a silent requeue loop. A sink
outage is retried with jittered exponential backoff (100 ms → 10 s) and logged once per
attempt; `SIGTERM` drains for up to `EGRESS_DRAIN_TIMEOUT_MS` and exits 1 if records remain
undelivered (they were never acked and will be redelivered).

`bench/Dockerfile.worker` is **not** aligned with this flow (it floats `^1.14.0` and does not
run `bootstrap-model`); run workers locally until it is fixed.

## Env knobs
| Var | Default | Meaning |
|---|---|---|
| `EMBED_MODE` | `real` | `real` = native onnxruntime-node (384-d); `hash` = 8-d deterministic stub for plumbing tests (use `EGRESS=redis`) |
| `ORT_INTRA_OP` | 1 | `intraOpNumThreads`. **Keep at 1** per worker; `0` = runtime default (pool sized to all logical CPUs — the pathological arm) |
| `ORT_ALLOW_SPINNING` | unset | `0`/`1` → `session.intra_op.allow_spinning` via `SessionOptions.extra`; unset = runtime default |
| `ROUTER` | `local-first` | `local-first`: score 1, every payload tries the local model and the guard decides; `centroid`: `1 − max cos` to hard-example centroids in `ROUTER_CENTROIDS` |
| `ROUTE_TAU` | 0.75 | attempt local iff calibrated router score ≥ τ; `>1` disables the local path (bench mode) |
| `EXTRACT` | 1 | constrained extraction with the evaluated guard (2-shot prompt, `guided_json`, value-surprisal). `0` = legacy unconstrained path, for the mock server and the throughput benchmarks only |
| `MAX_SURPRISAL` | 0 | guard threshold (nats) on the mean surprisal over all value tokens; 0 = escalate on any measurable hesitation. **The worker refuses to start with any other value** unless `ALLOW_GUARD_OVERRIDE=1` (then logged as not the evaluated configuration) |
| `EDGE_DEADLINE_MS` | 5000 | abort and escalate an extraction slower than this; must exceed the serving stack's object latency (1.1–1.9 s measured) |
| `EGRESS` | `redis` | `pgvector` (durable, idempotent upsert) or `redis` (results stream, for TTA measurement) |
| `B_MAX` / `W_MAX_MS` | 32 / 19 | batch-size cap and `BLOCK` window (Eq. 3 of the paper is a cap, not a collector) |
| `EGRESS_BACKOFF_BASE_MS` / `_MAX_MS` / `EGRESS_DRAIN_TIMEOUT_MS` | 100 / 10000 / 30000 | sink retry backoff and shutdown drain bound |
| `WORKER_SLOT` / `WORKER_NAME` | pid | stable consumer-group name (restart recovers own pending entries) |
| `RECLAIM_MS` / `RECLAIM_MIN_IDLE_MS` / `MAX_DELIVERIES` | 30000 / 60000 / 5 | PEL sweep interval, idle threshold to claim a dead consumer's entries, deliveries before dead-letter to `ingest:dead` |
| `UV_THREADPOOL_SIZE` | 4 (bench) / 8 (`npm run worker`) | libuv pool; must be set before Node starts; do **not** set to the core count per worker |
| `HEAVY_PROVIDER` / `HEAVY_MODEL` | from `src/models.mjs` | escalation target / shadow-labeling judge |

## Delivery semantics
At-least-once, ack-after-commit: a source entry is `XACK`ed only after its output is
committed. Entries of a crashed consumer stay in the group's pending list; every worker
sweeps it on boot and every `RECLAIM_MS`, reclaims entries idle ≥ `RECLAIM_MIN_IDLE_MS`, and
dead-letters entries that have reached `MAX_DELIVERIES`. A routing failure on one record
produces a `dead_letter` output for that record only. Record ids are `req_id → id → uuid →
src-<entry id>`, stable across redelivery, so the pgvector `ON CONFLICT (id)` upsert absorbs
duplicates. The Redis results sink is **not** idempotent. The RabbitMQ adapters in
`pipeline-worker.mjs` / `egress.mjs` are unwired and untested; the compose file's `rabbitmq`
service is unused.

## Reproducing the paper's numbers
1. **Scaling / placement / latency (Tables IV–VII):** on a quiesced bare-metal Linux host, see
   `bench/results-zen5/CONCLUSIONS.md` for the exact commands and interventions; raw rows are in
   `bench/results-zen5/` (per-trial CSVs, `latency-lambda.csv`, `PROVENANCE.txt`, profiling logs and the two `perf` captures).
2. **Mechanism, ablation, pool sweep (Table II):** `./bench/spin-proof.sh`, `./bench/ablation-2x2.sh`
   and `./bench/pool-sweep.sh` (~1 h, ~50 min, ~35 min; n=5 × 60 s). Run-2 results and verdict printouts are in `bench/results-zen5-run2/`
   (`run-all.log` holds the full console output).
3. **Guard (Table VIII):** `python3 analysis/guard_confirm.py` reproduces every reported number
   from the committed `bench/results-zen5-run2/guard-confirm.jsonl` (no model needed). To
   regenerate the traces: serve `Qwen/Qwen3-8B-AWQ` with vLLM 0.9.2 (`bench/GPU-RUNBOOK.md`),
   then `VLLM_URL=… TEMPS="0" SHOTS="0 1 2" PAYLOADS_ROOT=payloads/synthetic-confirm
   OUT=bench/results-zen5-run2/guard-confirm.jsonl node bench/optimize-guard.mjs`.
   Corpora: `payloads/synthetic-confirm/` (2,000, seed 20261004 — the reported one),
   `payloads/synthetic/` (500, exploratory).
