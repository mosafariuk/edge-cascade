#!/usr/bin/env bash
# pool-sweep.sh — residual thread-count loss at fixed W, spin-wait OFF, no affinity.
#
# Table II-a showed that disabling the pool's spin-wait recovers 81% (W=8) / 59% (W=14) of
# the oversubscription loss; Table II-b showed a one-thread pool recovers all of it. What is
# left between the two is a loss that grows with the number of pool threads even when they
# do not spin. This sweep measures it directly: W=8 workers, allow_spinning=0, no taskset,
# intraOpNumThreads ∈ {1, 2, 4, 8, 16, 32}. Thread count per worker = intra-op + ~10
# (libuv/V8/GC); 32 is the runtime default on this 32-logical-CPU host.
#
# Same trial design as Table II (n=5 × 60 s at 50k req/s). ~35 min.
#
#   ./bench/pool-sweep.sh
#   POOLS="1 4 16 32" TRIALS=3 ./bench/pool-sweep.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

POOLS="${POOLS:-1 2 4 8 16 32}"
WORKERS="${WORKERS:-8}"
TRIALS="${TRIALS:-5}"
DURATION="${DURATION:-60}"
RATE="${RATE:-50000}"
OUT="${OUT:-bench/results-zen5-run2/pool-sweep.csv}"
LOGDIR="${LOGDIR:-bench/results-zen5-run2/pool-sweep-logs}"
export TRIALS DURATION RATE OUT LOGDIR ALLOW_CONTENDED=1 EMBED_MODE=real

for P in $POOLS; do
  echo; echo "═══════ pool intra-op=$P (spin off, no affinity) ═══════"
  LABEL="pool-$P" ORT_INTRA_OP="$P" PIN_CORES=0 ORT_ALLOW_SPINNING=0 WORKERS="$WORKERS" \
    ./bench/scaling-sweep.sh
done

echo; echo "════════ pool-size sweep ($OUT) ════════"
python3 - "$OUT" <<'PY'
import csv, sys, math, statistics as st
T = {2: 4.303, 4: 2.776}
rows = list(csv.DictReader(open(sys.argv[1])))
by = {}
for r in rows: by.setdefault((int(r['ort_intra_op']), int(r['workers'])), []).append(float(r['achieved_rps']))
for w in sorted({w for _, w in by}):
    ref = st.mean(by[(1, w)]) if (1, w) in by else None
    print(f"W={w}   intra-op   mean req/s   ±t95   vs intra-op=1")
    for p in sorted(p for p, ww in by if ww == w):
        v = by[(p, w)]; m = st.mean(v); sd = st.stdev(v) if len(v) > 1 else 0
        ci = T.get(len(v) - 1, 1.96) * sd / math.sqrt(len(v)) if len(v) > 1 else float('nan')
        rel = f"{100*m/ref:6.1f}%" if ref else "   —"
        print(f"        {p:>7}   {m:10.1f}   {ci:5.0f}   {rel}   (threads/worker ≈ {p+10}, fleet ≈ {w*(p+10)})")
print("\nReading: a monotone decline from intra-op=1 with spin OFF is the pure thread-count loss;")
print("compare intra-op=32 here against Table II-a arm B (same config) and arm A (spin on).")
PY
