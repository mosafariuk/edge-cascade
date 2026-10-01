#!/usr/bin/env python3
"""
Discrete-event simulation of a local-embedding classifier under load.

Models the router's embedding stage as an M/D/c queue:
  - Poisson arrivals at rate lambda (req/s)
  - c parallel service channels (CPU cores available to ONNX Runtime)
  - Deterministic service time D per embedding (CPU-bound matmul)

Then compares against DYNAMIC BATCHING: a collector waits up to W seconds
(or until B items) then submits one batch. Batch service time is modeled as
  S(b) = fixed_overhead + b * per_item_marginal
which captures the empirical fact that batching amortizes fixed kernel-launch /
tokenization / memory-setup cost, so marginal per-item cost falls.

Pure stdlib. Deterministic (seeded). No hidden dependencies.
"""
import heapq, random, statistics

# ---- measured-ish primitives for all-MiniLM-L6-v2 INT8 ONNX on one core ----
# These are the ONLY inputs you must replace with a real benchmark on YOUR box.
# Defaults are conservative estimates for a short (<=32 token) sentence.
D_SINGLE      = 0.006      # 6.0 ms to embed one sentence, single-threaded, no batch
BATCH_FIXED   = 0.0020     # 2.0 ms fixed per-batch overhead (launch/setup)
BATCH_MARGIN  = 0.0018     # 1.8 ms marginal per extra item inside a batch
CORES         = 8          # Apple M1 performance+efficiency; ONNX intra-op threads

def simulate_mdc(lam, D, c, n=200_000, seed=1):
    """M/D/c: n arrivals, Poisson(lam), c servers, deterministic service D."""
    rnd = random.Random(seed)
    # server free-at times
    servers = [0.0]*c
    heapq.heapify(servers)
    t = 0.0
    sojourns = []
    for _ in range(n):
        t += rnd.expovariate(lam)            # next arrival
        free = heapq.heappop(servers)        # earliest-free server
        start = max(t, free)                 # queue wait if server busy
        finish = start + D
        heapq.heappush(servers, finish)
        sojourns.append(finish - t)          # total time in system
    sojourns.sort()
    return {
        "lambda": lam,
        "rho": lam*D/c,
        "mean_ms": 1000*statistics.fmean(sojourns),
        "p50_ms": 1000*sojourns[len(sojourns)//2],
        "p99_ms": 1000*sojourns[int(len(sojourns)*0.99)],
    }

def simulate_batched(lam, c, W, B, n=200_000, seed=1):
    """
    Dynamic batching in front of c batch-workers.
    Collector emits a batch when it holds B items OR W seconds elapsed since the
    batch's first item. Each batch runs on the earliest-free worker.
    Reports sojourn = (batch finish) - (item arrival), including collection wait.
    """
    rnd = random.Random(seed)
    workers = [0.0]*c
    heapq.heapify(workers)
    sojourns = []
    # generate arrival times
    t = 0.0
    arrivals = []
    for _ in range(n):
        t += rnd.expovariate(lam)
        arrivals.append(t)
    i = 0
    N = len(arrivals)
    while i < N:
        batch_start_arr = arrivals[i]
        # close batch at B items or when W elapses from first item
        j = i
        deadline = batch_start_arr + W
        while j < N and (j - i) < B and arrivals[j] <= deadline:
            j += 1
        batch = arrivals[i:j]
        b = len(batch)
        emit_time = min(batch_start_arr + W, batch[-1] if b >= B else batch_start_arr + W)
        emit_time = max(emit_time, batch[-1])   # cannot emit before last item arrives
        free = heapq.heappop(workers)
        start = max(emit_time, free)
        svc = BATCH_FIXED + b*BATCH_MARGIN
        finish = start + svc
        heapq.heappush(workers, finish)
        for a in batch:
            sojourns.append(finish - a)
        i = j
    sojourns.sort()
    eff_mu = c / (BATCH_FIXED/B + BATCH_MARGIN)   # max throughput at full batches
    return {
        "lambda": lam, "W_ms": W*1000, "B": B,
        "mean_ms": 1000*statistics.fmean(sojourns),
        "p99_ms": 1000*sojourns[int(len(sojourns)*0.99)],
        "capacity_rps": eff_mu,
    }

print("="*72)
print("A1. NO BATCHING  (M/D/c, D=%.1fms/embed, c=%d cores)" % (D_SINGLE*1000, CORES))
print("    Per-core service rate mu = %.1f req/s ; system capacity = %.0f req/s"
      % (1/D_SINGLE, CORES/D_SINGLE))
print("="*72)
print(f"{'arrival(req/s)':>14} {'rho':>6} {'mean(ms)':>9} {'p50(ms)':>8} {'p99(ms)':>8}")
for lam in [10, 100, 500, 1000, 1200, 1300]:
    if lam*D_SINGLE/CORES >= 1.0:
        print(f"{lam:>14} {lam*D_SINGLE/CORES:>6.3f}   UNSTABLE (rho>=1): queue grows without bound")
        continue
    r = simulate_mdc(lam, D_SINGLE, CORES)
    print(f"{lam:>14} {r['rho']:>6.3f} {r['mean_ms']:>9.2f} {r['p50_ms']:>8.2f} {r['p99_ms']:>8.2f}")

print()
print("="*72)
print("A2. DYNAMIC BATCHING at 1000 req/s (sweep batch size B, window W=5ms)")
print("="*72)
print(f"{'B':>4} {'W(ms)':>6} {'mean(ms)':>9} {'p99(ms)':>8} {'capacity(req/s)':>15}")
for B in [1, 4, 8, 16, 32, 64]:
    r = simulate_batched(1000, CORES, W=0.005, B=B)
    print(f"{B:>4} {r['W_ms']:>6.1f} {r['mean_ms']:>9.2f} {r['p99_ms']:>8.2f} {r['capacity_rps']:>15.0f}")

print()
print("="*72)
print("A3. OPTIMAL WINDOW at 1000 req/s, B=32 (sweep W)")
print("="*72)
print(f"{'W(ms)':>6} {'mean(ms)':>9} {'p99(ms)':>8}")
for W in [0.001, 0.002, 0.005, 0.010, 0.020, 0.050]:
    r = simulate_batched(1000, CORES, W=W, B=32)
    print(f"{W*1000:>6.1f} {r['mean_ms']:>9.2f} {r['p99_ms']:>8.2f}")
