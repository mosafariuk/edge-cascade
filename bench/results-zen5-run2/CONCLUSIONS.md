# Run 2 — spin-wait intervention and 2×2 pinning ablation (2026-09-30)

**Host:** bare-metal AMD Ryzen 9 9950X (same model/provider as run 1), Ubuntu 24.04.5, kernel
6.8.0, Node v24.21.0, onnxruntime-node 1.27.0, model sha256 `afdb6f1a…` (identical to run 1).
Quiesced by the run-1 procedure (RAID resync frozen, apt timers masked, host redis disabled,
perf_event_paranoid=1, infra pinned to core 15). Full record: `PROVENANCE.txt`.
Design: n=5 × 60 s at 50,000 req/s offered, EMBED_MODE=real, ROUTE_TAU=2, EGRESS=none.
Console transcript: `run-all.log`. Per-trial rows: `spin-proof.csv`, `ablation-2x2.csv`.

## 1. Reference arm reproduces run 1

| W | run 2 pinned | run 1 pinned (Table IV) | Δ |
|---|---|---|---|
| 1 | 2,237 ± 14 | 2,285 ± 14 | −2.1% |
| 8 | 13,937 ± 71 | 14,124 ± 54 | −1.3% |
| 14 | 21,853 ± 76 | 21,803 ± 193 | +0.2% |

Unpinned default also reproduces (4,170 / 1,492 / 1,485 vs 4,123 / 1,514 / 1,381).

## 2. Spin-wait test (paper Table II-a) — mechanism DEMONSTRATED

`session.intra_op.allow_spinning=0` via `SessionOptions.extra`, nothing else changed
(no taskset, intra-op pool at the runtime default of 32 threads):

| W | A spin on | B spin off | C pinned | B/A | share of A→C recovered |
|---|---|---|---|---|---|
| 1 | 4,170 ± 361 | **4,759 ± 123** | 2,237 ± 14 | 1.14× | (B > C) |
| 8 | 1,492 ± 99 | **11,537 ± 398** | 13,937 ± 71 | 7.73× | **80.7%** |
| 14 | 1,485 ± 133 | **13,494 ± 466** | 21,853 ± 76 | 9.09× | **59.0%** |

- Spin-wait is the dominant loss. The 66%-of-cycles hot site in run 1's profile is, by
  elimination, the pool's spin loop.
- Residual gap grows with W (19% → 41%): consistent with a second loss ∝ thread count (32W).
- At W=1, spin-off unpinned is the fastest configuration measured (keeps multi-core GEMM,
  sheds spin).
- Arm D (pinned + spin off) ≡ C: once the pool is one thread, spinning is irrelevant.

## 3. 2×2 ablation (paper Table II-b) — the remedy is pool size, not affinity

| | taskset OFF | taskset ON |
|---|---|---|
| **W=8** intra-op default | 1,473 ± 95 | 1,375 ± 20 |
| **W=8** intra-op 1 | **14,240 ± 87** | 13,870 ± 75 |
| **W=14** intra-op default | 1,533 ± 39 | 1,365 ± 18 |
| **W=14** intra-op 1 | **21,672 ± 67** | 21,786 ± 137 |

Share of the (4)−(1) gap: intra-op alone **103.0% / 99.4%**; taskset alone **−0.8% / −0.8%**;
interaction −2.2% / +1.4%.

- Confining 32 spinning threads to one core (cell 2) does not help: they still starve the one
  thread doing work.
- `taskset` on top of a single-thread pool changes throughput by −2.6% (W=8) / +0.5% (W=14):
  noise. It is a determinism control, not a performance one.
- **Run 1's `CONCLUSIONS.md` §4 ("taskset is the only available mitigation") is withdrawn.**

## 4. What this changes in the paper

- §III-C2 "userspace spin-wait" → demonstrated by intervention (was "best-supported").
- §III-C3 rewritten around Table II; the "unreachable from Node.js" claim is withdrawn.
- Threat 2 (two-variable treatment) and Threat 3 (mechanism attribution) resolved/narrowed.
- Contributions 1–2 and the abstract updated with 81%/59% and 103%/99%/−0.8%.

## 5. Open

- Residual thread-count loss is inferred from cell 3, not measured separately: a pool-size
  sweep (1, 2, 4, 8, 16, 32 threads) at fixed W=8 with spin off would quantify it.
- Boost clocks were not disabled (same as run 1).

## 6. Pool-size sweep (paper Table II-c) — the non-spinning thread-count loss, measured

W=8, `allow_spinning=0`, no affinity, `intraOpNumThreads` ∈ {1,2,4,8,16,32}; n=5 × 60 s.
Rows: `pool-sweep.csv`; transcript: `pool-sweep.log`.

| intra-op | fleet compute threads | req/s | vs peak |
|---|---|---|---|
| 1 | 8 | 14,203 ± 87 | 88.1% |
| **2** | **16 (= physical cores)** | **16,117 ± 442** | **100%** |
| 4 | 32 | 15,221 ± 209 | 94.4% |
| 8 | 64 | 16,029 ± 461 | 99.5% |
| 16 | 128 | 14,785 ± 259 | 91.7% |
| 32 (default) | 256 | 13,012 ± 181 | 80.7% |

- The optimum is pool × workers ≈ physical cores, not "one thread". At W=8 a 2-thread pool
  beats the pinned configuration (13,870–13,937) by ~15%.
- With spinning off, oversubscription costs little: ≤19% at 16× (the runtime default),
  versus the 89% collapse the same pool produces when spinning (arm A). Spin-wait is close
  to the entire mechanism at W=8.
- **Caveat:** pool=32/spin-off was measured twice in this run: arm B 11,537 ± 398
  (22:1x UTC) and this sweep 13,012 ± 181 (00:3x UTC). Non-overlapping CIs → ~12%
  run-to-run drift in the large non-spinning pool; single-thread/pinned arms did not drift
  (C vs cell 4 vs pool=1: 13,937 / 13,870 / 14,203). Report both; do not cherry-pick.
  Frequency was not logged (threat 4); that is the first suspect.
- §5 "Open" item on the thread-count loss is now closed at W=8; W=14 remains inferred.

## 7. Guard evaluation on a served model (paper §VI-G, Table VIII) — 2026-10-01

Stack: Qwen/Qwen3-8B-AWQ (rev `4da05a8e…`), vLLM 0.9.2, xgrammar, RTX 3090 (vast.ai, PCIe 4.0
×16 — `gpu-host.txt`); judge GPT-4o (`gpt-4o-2024-08-06`, strict JSON schema).
Files: `vllm-probe.jsonl`, `shadow-pairs.jsonl` (final corpus), `shadow-trace.jsonl`,
`guard-eval.txt` (= `python3 analysis/guard_eval.py`).

- **Probe:** POST-MASK, H = 0.049 nats at the first grammar-forced token.
- **Corpus:** 500 synthetic records, truth sampled in code (seed 20261002), texts written by
  GPT-4o, mechanically verified: 97/500 first drafts unfaithful → regenerated; 2 templated.
- **Judge fidelity vs constructed truth:** 96.4% [94.4, 97.7].
- **Edge accuracy:** 74.2% (property 84.0%, telemetry 64.4%). Errors: hallucinated/mis-set
  booleans (87), list items (40), stated numbers nulled (22).
- **Guard, held-out n=150 (34 errors):** AUROC 0.805 (mean over all value tokens) vs 0.680
  (first-6 window; 0.478 on property). At the 20% target, s_max = 0.027 nats → 17.3% escalated,
  precision 77.3% → 83.9% [76.4, 89.3], recall 41.2% [26.4, 57.8]. 30% target: 89.6% / 67.6%.
- **What changed because of this run:** required-nullable schemas (omissions were invisible),
  mean-all statistic (window was at chance), ChatML prompt for the edge model, strict judge.
- **Withdrawn:** threshold 1.845 and the 20.8% / 98.2% / 91.8% operating point of earlier drafts.

## 8. Guard optimization grid: temperature × few-shot (2026-10-01, NOT yet in the paper)

Second rented host (RTX 3090, PCIe 4.0 ×16), same model/stack as §7.
Files: `guard-grid.jsonl` (6,000 traces), `guard-grid.log`, `guard-grid.txt`
(= `python3 analysis/guard_grid.py`). 12 configs × 500 synthetic payloads, truth-labelled.

- **Baseline reproduces Table VIII exactly** (`T0_shot0`: 371/500, held-out AUROC 0.803,
  17.3% escalated, precision 83.9%, recall 41.2%).
- **Temperature does not help.** Zero-shot, T 0→0.5: accuracy 74.2→73.0%, held-out silent
  error 13.3% → 13.3%, AUROC 0.803→0.846 (within noise). Reported logprobs on
  token-identical records move by ≤0.0013 nats.
- **Few-shot prompting is the first-order lever — on accuracy, not on detectability.**
  Accuracy 74.2% → 91.6% (1-shot) → 96.2% (2-shot); telemetry 64.4 → 98.8 → 100%,
  property 84.0 → 84.4 → 92.4%. Errors 129 → 42 → 19, and the survivors are all `assets`
  list mismatches.
- **Guard on the 1-shot model:** ~22% escalated, held-out silent error 1.3% vs 6.7% without
  a guard; 8–9 of 10–11 held-out errors caught (CI 52–95%).
- **Pre-registered selection** (lowest train silent-error @20%): `T0.1_shot1` / max
  surprisal. Held-out silent error 13.3% → 1.3%, Δ = −12.0 pts, 95% [−18.0, −6.7];
  ΔAUROC +0.04 [−0.12, +0.19] (not significant — the gain is accuracy, not detection).
- **Not yet claimable:** the few-shot examples were written after this corpus's error
  anatomy was known; the held-out split has now been used twice; 2-shot leaves 2 held-out
  errors, so its guard figures are meaningless. Needs a fresh, larger corpus
  (≥2,000 records → ~80 errors at 96% accuracy) before Table VIII changes.

## 9. Confirmation run (paper Table VIII) — 2026-10-01

Pre-registered: `analysis/guard_confirm.py` committed before the data (commit "Pre-register the
guard confirmation run"). Fresh corpus `payloads/synthetic-confirm` (2,000 records, seed
20261004; 349 first drafts regenerated, 2 templated). rented host, RTX 3090 PCIe 4.0 ×16,
Qwen3-8B-AWQ, vLLM 0.9.2 (`gpu-host-confirm.txt`). Files: `guard-confirm.jsonl` (6,000 traces),
`guard-confirm.log`, `guard-confirm.txt`.

Held-out n=600 (train 1,400), T=0, statistic = mean surprisal over all value tokens:

| | 0-shot | 1-shot | 2-shot |
|---|---|---|---|
| edge accuracy | 78.3% [74.9, 81.4] | 91.5% [89.0, 93.5] | **97.0% [95.3, 98.1]** |
| errors | 130 | 51 | 18 |
| AUROC | 0.794 | 0.896 | 0.851 |
| s_max (20% target, train) | 0.0208 | 0 | 0 |
| escalated | 21.0% | 22.2% | **13.7%** |
| local precision | 86.1% | 98.7% | 99.2% |
| error recall | 49.2% [40.8, 57.7] | 88.2% [76.6, 94.5] | 77.8% [54.8, 91.0] |
| silent errors, guard / none | 11.0% / 21.7% | 1.0% / 8.5% | **0.7% / 3.0%** |

- **H1 confirmed** (few-shot → accuracy): +13.2 [9.7, 16.8] and +18.7 [15.3, 22.0] pts;
  2-shot vs 1-shot +5.5 [3.7, 7.3].
- **H2 confirmed** (guard → fewer silent errors) in every arm: −10.7, −7.5, −2.3 pts, all
  intervals exclude zero. With the guard, 2-shot vs 1-shot is not distinguishable (−0.3).
- **Decision (pre-registered): deploy 2-shot**, `MAX_SURPRISAL = 0` at 1e-4 nat resolution.
- Zero-shot arm agrees with the exploratory Table (AUROC 0.794 vs 0.805; recall 49.2% vs 41.2%,
  overlapping intervals).
- Residual errors at 2-shot are all `assets` list mismatches; the 4 the guard misses have zero
  surprisal. Windowed statistic: AUROC 0.50 on property in every arm.
- Code now deploys what was evaluated: `SCHEMAS[*].examples`, `buildEdgePrompt`,
  `EDGE_MODEL.few_shot = 2`; prompt verified byte-identical to the one the run sent.
