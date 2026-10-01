#!/usr/bin/env bash
# ablation-2x2.sh — untangle the two variables inside "pinned" (paper Threat 2).
#
# The benchmarked "pinned" arm changed TWO things at once:
#   intra-op = 1   (session-level: ORT gets one compute thread per worker)
#   taskset        (OS-level: the process is confined to one physical core)
# The 15.8× at W=14 is the joint effect. This script runs the full factorial so the
# paper can say how much each contributes:
#
#            │ taskset OFF            │ taskset ON
#   ─────────┼────────────────────────┼─────────────────────────
#   intra=0  │ (1) runtime default    │ (2) OS confines the pool
#            │     = "unpinned" arm   │     (pool still spawns 32 threads on 1 core)
#   intra=1  │ (3) single thread,     │ (4) = "pinned" arm
#            │     free to migrate    │
#
# Reading:  (3)≈(4) ⇒ intra-op=1 alone is the fix; taskset adds nothing but determinism.
#           (2)≈(4) ⇒ confinement alone is enough (the pool's threads are harmless once
#                     they share one core) — spin-wait on OTHER cores was the loss.
#           (2)≪(4) and (3)≪(4) ⇒ both are needed; report the interaction.
#
# Same trial design as Table III (n=5 × 60 s at 50k req/s). Cells (1) and (4) are
# re-measured rather than copied so all four share one session of the host.
# Expected wall time: 4 cells × 2 W × 5 trials × ~75 s ≈ 50 min.
#
#   ./bench/ablation-2x2.sh                  # W="8 14", n=5
#   WORKERS="8" TRIALS=3 ./bench/ablation-2x2.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

WORKERS="${WORKERS:-8 14}"
TRIALS="${TRIALS:-5}"
DURATION="${DURATION:-60}"
RATE="${RATE:-50000}"
OUT="${OUT:-bench/results-zen5/ablation-2x2.csv}"
LOGDIR="${LOGDIR:-bench/ablation-logs}"
export TRIALS DURATION RATE OUT LOGDIR ALLOW_CONTENDED=1 EMBED_MODE=real

run_cell() {  # label  ORT_INTRA_OP  PIN_CORES
  echo; echo "═══════ cell $1 ═══════"
  LABEL="$1" ORT_INTRA_OP="$2" PIN_CORES="$3" WORKERS="$WORKERS" ./bench/scaling-sweep.sh
}
run_cell "1:intra0-taskset0" 0 0
run_cell "2:intra0-taskset1" 0 1
run_cell "3:intra1-taskset0" 1 0
run_cell "4:intra1-taskset1" 1 1

echo; echo "════════ 2×2 table ($OUT) ════════"
python3 - "$OUT" <<'PY'
import csv, sys, math, statistics as st
T = {2: 4.303, 4: 2.776}
rows = list(csv.DictReader(open(sys.argv[1])))
by = {}
for r in rows: by.setdefault((r['label'], int(r['workers'])), []).append(float(r['achieved_rps']))
def cell(lab, w):
    v = by.get((lab, w));
    if not v: return None
    m = st.mean(v); sd = st.stdev(v) if len(v) > 1 else 0
    return m, T.get(len(v) - 1, 1.96) * sd / math.sqrt(len(v)) if len(v) > 1 else float('nan'), len(v)
for w in sorted({w for _, w in by}):
    print(f"\nW={w}                 taskset OFF              taskset ON")
    for intra, (l0, l1) in {0: ('1:intra0-taskset0', '2:intra0-taskset1'), 1: ('3:intra1-taskset0', '4:intra1-taskset1')}.items():
        c0, c1 = cell(l0, w), cell(l1, w)
        f = lambda c: f"{c[0]:9.0f} ±{c[1]:5.0f} (n={c[2]})" if c else "      —"
        print(f"  intra-op={intra}   {f(c0)}   {f(c1)}")
    c1, c2, c3, c4 = (cell(l, w) for l in ('1:intra0-taskset0', '2:intra0-taskset1', '3:intra1-taskset0', '4:intra1-taskset1'))
    if all((c1, c2, c3, c4)):
        gap = c4[0] - c1[0]
        print(f"  share of the (4)−(1) gap recovered by:  intra-op alone (3): {100*(c3[0]-c1[0])/gap:5.1f}%   "
              f"taskset alone (2): {100*(c2[0]-c1[0])/gap:5.1f}%   interaction: {100*(c4[0]-c2[0]-c3[0]+c1[0])/gap:+5.1f}%")
PY
