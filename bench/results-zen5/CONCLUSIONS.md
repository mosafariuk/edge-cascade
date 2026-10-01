# Benchmark Conclusions — AMD Ryzen 9 9950X (Zen 5), bare metal

**Host:** Cherry Servers hourly bare metal, Stockholm · AMD Ryzen 9 9950X, 16C/32T,
186 GiB RAM · Ubuntu 24.04.4 LTS · kernel 6.17.0-23-generic (EEVDF) · **not virtualized**
(`systemd-detect-virt` = none, zero hypervisor CPU flags)

**Workload:** `all-MiniLM-L6-v2` INT8 via native `onnxruntime-node` **1.27.0**, Redis 7.0.15
broker, Node.js v24.18.0. All runs `EMBED_MODE=real` unless stated.

**Statistical basis:** n=5 trials per point (n=3 for the CCD matrix), 60 s steady-state per
trial, open-loop saturation. Reported as mean with 95% CI.

> **Harness validity.** A plumbing-only sweep (`EMBED_MODE=hash`) established a harness
> ceiling of **~90,000–107,000 req/s**. Peak real-mode throughput measured was **21,803 req/s**
> — a **4.1× margin**. Load-generator CPU never exceeded **24%** and Redis never exceeded
> **18.4%** in any run. **No data point was harness-limited**, and this is recorded per-row in
> the `loadgen_cpu_pct` / `redis_cpu_pct` CSV columns rather than asserted.

---

## 1. The L3 capacity discontinuity

Scaling efficiency is **not monotonic**. It decays as workers fill one chiplet's L3 domain,
then **recovers superlinearly** when workers spill onto the second chiplet.

| Step | Gain | Efficiency | Core placement |
|---|---|---|---|
| 1→2 | 1.956× | 97.8% | CCD0 |
| 2→4 | 1.928× | 96.4% | CCD0 |
| 4→6 | 1.407× | 93.8% | CCD0 |
| 6→8 | 1.165× | 87.4% | CCD0 saturating |
| **8→10** | **1.265×** | **101.2%** | **crosses to CCD1** |
| 10→12 | 1.150× | 95.8% | both |
| 12→14 | 1.061× | 91.0% | both |

Per-core throughput *rises* at the boundary (1,765 → 1,786 req/s). Overall 1→14 efficiency is
**68.2%** — so an unqualified "near-linear scaling" claim is false at scale.

**Topology:** CCD0 = cores 0–7, CCD1 = cores 8–15; 64 MiB L3 in **2 instances** (32 MiB each).

### Causal attribution (CCD isolation matrix, n=3)

| Run | Cores | L3 domains | Mean req/s |
|---|---|---|---|
| C1 | 0–3 | 1 (CCD0, infra on CCD1) | 8,656 |
| C2 | 8–11 | 1 (CCD1, **shares CCD with infra**) | 8,346 |
| A | 0–7 | **1** (packed) | 14,183 |
| B | 0–3 + 8–11 | **2** (split) | **16,556** |

- **L3 capacity is the dominant term.** A vs B: identical worker count, identical core count,
  differing only in chiplet placement → **+16.7% throughput**. Against two independent 4-worker
  groups (2 × C1 = 17,312): packed achieves **81.9%** of ideal, split achieves **95.6%**.
- **Fabric distance is minor, and negative in sign.** C1 vs C2: placing workers on the *same*
  chiplet as the Redis/load-gen infra core was **3.6% slower**, not faster — the L3 those infra
  processes consume outweighs any fabric-proximity benefit.
- **Mechanism confirmed independently.** `/proc/PID/maps` shows **no `model_quantized.onnx`
  mapping** — weights are *not* file-mapped, so each worker holds a **private** ~23 MB copy
  (`Private_Dirty` 232 MB/worker). Eight workers on one 32 MiB L3 therefore contend for a
  ~184 MB weight working set. Three independent methods — throughput discontinuity, OS memory
  telemetry, and the placement A/B — agree.

**Engineering consequence:** naive ascending core assignment packs workers onto one chiplet and
forfeits **16.7%** of throughput at zero hardware cost. The paper's pinned-scaling table (Table I)
was measured under this default policy and therefore *understates* achievable performance.

**Validity note:** C1 (8,656) and A (14,183) reproduce the main sweep's W=4 (8,618 ±8) and W=8
(14,124 ±38) to within 0.4%, confirming the `CORE_LIST` path is equivalent to the default.

---

## 2. The unpinned pathology

`ORT_INTRA_OP=0` + no `taskset` (the out-of-the-box default) does not merely scale poorly — it
**scales negatively**, then converges toward a floor independent of fleet size.

| W | Pinned (req/s) | Unpinned (req/s) | Pinned advantage |
|---|---|---|---|
| 1 | 2,285 ±10 | **4,123 ±218** | **0.55× — unpinned WINS by 1.80×** |
| 2 | 4,470 ±11 | 1,981 ±12 | 2.26× |
| 4 | 8,618 ±8 | 1,721 ±36 | 5.01× |
| 6 | 12,123 ±29 | 1,566 ±6 | 7.74× |
| 8 | 14,124 ±38 | 1,514 ±35 | 9.33× |
| 10 | 17,862 ±10 | 1,417 ±47 | 12.60× |
| 12 | 20,544 ±104 | 1,397 ±11 | 14.71× |
| 14 | **21,803 ±136** | **1,381 ±9** | **15.79×** |

**The crossover sits between W=1 and W=2.** A single unpinned worker beats a pinned one by
**1.80×** because it commandeers all 16 cores. Adding one more worker **halves** absolute
throughput (4,123 → 1,981) while the pinned fleet nearly doubles.

**Pinned throughput grows with the core budget; unpinned throughput converges toward a
constant (~1,380 req/s).** The gap therefore widens as the fleet scales. The decline decelerates
rather than flattening exactly — a soft floor, not a hard asymptote.

> **This is the central practitioner warning:** the configuration that wins every
> single-process microbenchmark by 1.80× is the one that loses by 15.79× in production. Fleet
> configuration cannot be selected on `W=1` measurements.

`W=14` unpinned (**~588 threads on 16 cores**) completed every trial and cleared every readiness
gate. The pathology is pure throughput collapse, **not** initialization failure.

---

## 3. Mechanism: context-switch churn is disqualified

Single-variable A/B at W=8 (only `PIN_CORES`/`ORT_INTRA_OP` changed):

| Metric | Pinned | Unpinned |
|---|---|---|
| Context switches/s | 5,819 | 13,286 |
| Throughput | 14,124 req/s | 1,514 req/s |
| **Ctx per request** | **0.41** | **8.78** (21.3×) |
| Threads per worker | 11 | **42** |
| Thread census | **8 R / 80 S** (9% runnable) | **238 R / 98 S** (71% runnable) |
| CPU per request | **0.57 core-ms** | **10.46 core-ms** (**18.3×**) |
| Cycles in `libonnxruntime.so.1` | 86.4% (84.7% self) | **99.0%** (98.7% self) |
| Cycles in `node` / `libc` | 9.5% / 1.5% | 0.8% / 0.2% |
| Hottest single offset | **3.60%** | **65.99%** |

### Why context-switch churn cannot be the cause

Unpinned does 2.28× more switches per second. But the **absolute magnitude is far too small**:

| Assumed cost/switch | Share of the 16-core machine |
|---|---|
| 2 µs (typical) | 0.17% |
| 10 µs | 0.83% |
| 50 µs (generous, incl. cache effects) | **4.15%** |

A 9.33× collapse means **~89% of throughput vanished**. Context switching accounts for **at
most 4.15%** under an implausibly pessimistic assumption. **It is not the mechanism, and the
claim must be removed from the manuscript.**

### What the evidence does support: userspace spin-wait contention

1. **18.3× more CPU per request** for identical work.
2. **71% of threads in `R` state** (unpinned) vs **9%** (pinned) — unpinned ORT threads do
   **not** sleep. This resolves the load-average-vs-context-switch paradox: threads burning
   cycles in userspace never enter the scheduler, so load climbs while `ctxt` stays flat.
3. **99% of all system cycles inside `libonnxruntime.so.1`**, while `node` and `libc` — the
   real pipeline work — drop to 0.8% and 0.2%. The fleet has effectively stopped doing work.
4. **A regime-specific hot function.** Under oversubscription a *single* offset consumes
   **65.99%** of cycles (plus the same per-process offset at ~9% in each of 8 workers). Under
   pinning the busiest offset is **3.60%**, with cycles spread across many functions. **A GEMM
   microkernel would be hot in both regimes** — the same model doing the same math. A function
   that only dominates when 336 threads contend for 16 cores is regime-specific by construction.
5. ORT's spin controls exist in the shipped runtime: `session.intra_op.allow_spinning`,
   `session.intra_op.spin_duration_us`, `session.force_spinning_stop`.

**Limitation — stated explicitly.** The hot function is **not identified by name**. The shipped
`libonnxruntime.so.1` exports only **9 dynamic symbols**, with no `.symtab` and no
`.debug_info`. Symbol-level confirmation requires a debug build of ONNX Runtime and is **future
work**. Spin-wait is the best-supported explanation, **not a proven one**.

---

## 4. ~~`taskset` is the only available mitigation~~ — WITHDRAWN 2026-09-30

> **Superseded by run 2** (`../results-zen5-run2/CONCLUSIONS.md`). The Node.js binding *does*
> forward `SessionOptions.extra` to the runtime; disabling `allow_spinning` alone recovers 81%
> of the loss at W=8, and the 2×2 ablation shows `taskset` contributes −0.8%. The remedy is
> `intraOpNumThreads=1`; affinity is optional. The text below is kept for history only.


`session.intra_op.allow_spinning` exists in the runtime but is **unreachable from Node.js by
every available route**:

| Route | Status |
|---|---|
| `SessionOptions.extra` (config entries) | **Unavailable** — documented *"available only in WebAssembly backend. Will support Node.js binding and react-native later"* |
| `ORT_LOAD_CONFIG_FROM_MODEL` (model-embedded `ort_config`) | **Insufficient** — parser accepts only `enable_profiling`, `execution_mode`, `graph_optimization_level`, `inter_op_num_threads`, `intra_op_num_threads`. **No `session_config_entries`.** |
| `SessionOptions.externalData` | Exposed to Node, but requires supplying data as a **buffer/typed array**, so each process holds private bytes — no cross-process page sharing |
| `session.intra_op_thread_affinities` | Exists in the runtime, **not exposed** by the binding |

**Consequence:** OS-level `taskset` pinning is not a workaround of convenience — it is the
**only architecturally available mitigation** for Node.js deployments. The runtime possesses
both the spin control and native thread-affinity support; the language binding suppresses both.

---

## 5. Reproducibility notes

- **USE `npm ci`, NEVER `npm install`.** `package.json` declares `onnxruntime-node ^1.14.0`,
  but the version actually measured was **1.27.0** (pinned correctly in `package-lock.json`).
  `npm install` honours the caret and may resolve a newer ORT with different threading
  defaults, silently invalidating every number here. `npm ci` installs the locked tree exactly.
- **Two ORT versions coexist in the dependency tree.** Top-level `onnxruntime-node` is
  **1.27.0** — this is what `src/embed-native.mjs` imports and what every benchmark number was
  produced by. `@xenova/transformers` 2.17.2 carries its own nested `onnxruntime-node` **1.14.0**,
  used only by `bench/bootstrap-model.mjs` when downloading the weights. So the model is fetched
  and validated under 1.14.0 and executed under 1.27.0. Functionally fine, but do not assume a
  single ORT version is in play when interpreting `perf` captures or thread counts.
- **Model integrity:** `model_quantized.onnx`, 22,972,370 bytes,
  sha256 `afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1`.
  Verify before comparing against these results — a silently updated upstream artifact would
  invalidate any comparison.
- **Air-gap verified empirically.** Under `iptables -A OUTPUT -j DROP` (HuggingFace HTTP 000,
  DNS blocked), the benchmark ran to completion at **4,499 req/s** vs the networked baseline of
  **4,470 ±11** — within 0.6%. After `bootstrap-model`, the suite requires **no external
  network**, and network isolation has no measurable effect on results.
- **Artifact bootstrap:** the native path resolves the model **locally only**; on a clean
  machine every worker dies with `MiniLM onnx not found`. `npm run bootstrap-model` is a
  **mandatory prerequisite**. Both branches (download and cached) were executed on a simulated
  clean machine.
- **Prerequisite order:** `npm install` → `npm run bootstrap-model` (**only step needing
  network**) → `npm test` → `./bench/scaling-sweep.sh` (offline).

### Interventions applied to achieve an idle host — must be disclosed

1. **RAID1 initial resync frozen** (`echo frozen > /sys/block/md0/md/sync_action`) — `md0_resync`
   sat in `D` state with ~62 min remaining and would have contended during runs.
2. **`apt-daily.timer` and `apt-daily-upgrade.timer` masked** — prevents mid-run `apt` activity.
3. **Host `redis-server` systemd unit disabled**; the harness starts its own on `:6390` pinned
   to the infra core.
4. **`kernel.perf_event_paranoid` set to 1** (was 4) for profiling.
5. **Load generator and Redis pinned to core 15** via `taskset`.

"Benchmarked on an idle host" is only honest alongside this list.

---

## 6. Latency vs offered load (Pollaczek–Khinchine), W=8 pinned, n=2

Capacity μ = 14,124 req/s (W=8 pinned). Offered load λ held **constant** per run
(`RATE_START = RATE_END`) at sub-capacity rates, so the queue does not diverge and
time-to-answer percentiles are meaningful — unlike the saturation runs.

| λ (nominal) | ρ nominal | achieved | ρ **realised** | p50 (ms) | p95 (ms) | **p99 (ms)** |
|---|---|---|---|---|---|---|
| 1,400 | 0.10 | 1,295 | 0.09 | 15 | 27 | **28** |
| 2,800 | 0.20 | 2,580 | 0.18 | 16 | 28 | **29** |
| 5,600 | 0.40 | 5,112 | 0.36 | 18 | 31 | **34** |
| 8,500 | 0.60 | 7,720 | 0.55 | 20 | 35 | **59** |
| 11,300 | 0.80 | 10,180 | 0.72 | 25 | 41 | **129** |
| 12,700 | 0.90 | 11,405 | 0.81 | 30 | 47 | **164** |
| 13,400 | 0.95 | 11,914 | 0.84 | 33 | 58 | **185** |
| 14,000 | 0.99 | 12,466 | **0.88** | 37 | 63 | **200** |

**The tail diverges far ahead of the median** — the paper's core queueing claim, now measured
on the same hardware as the scaling table (Table I):

- **p50 grows 2.4×** (15 → 37 ms) across the whole sweep
- **p99 grows 7.0×** (28 → 200 ms)
- The p99 knee appears around **ρ ≈ 0.55** (34 → 59 ms), while p50 is still nearly flat

**Caveat — realised ρ, not nominal.** Achieved throughput falls short of nominal λ at the top
end (12,466 vs 14,000 offered) because the load generator is **timer-bound**, not
CPU-bound: `perTick = rate/50` every 20 ms, and each iteration costs pipeline-exec time plus
the sleep. True utilisation therefore topped out at **ρ ≈ 0.88**, not 0.99. The divergence is
real and correctly ordered; the nominal column must not be reported as realised utilisation.
Reaching ρ → 1.0 would require a multi-process or busy-poll load generator.

**Do not reuse the saturation runs' percentiles for any latency claim.** At `RATE=50000`
against ~14k capacity, p99 was 3,900–6,500 ms — that is queue backlog, not service latency.

---

### Harness caveats

- **Open-loop saturation runs measure throughput correctly and latency not at all.** At
  `RATE=50000` against ~14k capacity, p99 is dominated by queue backlog. Use the λ-sweep
  (§6) for latency claims — never the saturation runs' percentiles.
- **The load generator is round-trip-bound, not CPU-bound.** `perTick = rate/50` every 20 ms
  via `setTimeout`; each iteration costs pipeline-exec time *plus* the sleep. Raising `RATE`
  raises `perTick`, which is the actual lever. At the ~100k plateau, load-gen CPU was only 40%.
- **Plumbing mode shows mild negative scaling** (106k → 92k, −14% from W=2→14) purely from
  per-worker overhead with zero embedding work. This term still exists in real runs, so measured
  efficiency is very slightly depressed — smaller in practice, since real workers spend most
  time computing rather than hammering Redis.
- **Process-matching hazard:** any `pgrep -f` / `pkill -f` pattern inside a script whose own
  command line contains the target string **will match itself**. The `[s]` bracket trick does
  not protect against this. Bit this harness three times (killed an SSH session; hung a wait
  loop). **Use PID files.**
