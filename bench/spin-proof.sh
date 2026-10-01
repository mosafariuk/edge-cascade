#!/usr/bin/env bash
# spin-proof.sh — direct test of the userspace spin-wait hypothesis (paper §III-C2/C3).
#
# The unpinned pathology (ORT sizes its intra-op pool to every logical CPU, the fleet
# oversubscribes the host, throughput COLLAPSES as workers are added) is attributed to
# pool threads spinning in userspace. If that is the mechanism, then disabling spinning
# — session.intra_op.allow_spinning=0, reachable via SessionOptions.extra in
# onnxruntime-node 1.27 — while leaving EVERYTHING ELSE in the pathological arm
# unchanged (no taskset, pool sized to all logical CPUs) must recover a large share of
# the lost throughput. If it does not, the hypothesis is wrong.
#
# Three arms at the same W, same host, same trial design (n=5 × 60 s, 50k req/s):
#   A  unpinned, spin default   (the pathological arm, Table III unpinned column)
#   B  unpinned, spin OFF       (the test)
#   C  pinned,   spin default   (the mitigation the paper benchmarked, reference)
# Optional:
#   D  pinned,   spin OFF       (does spin-off cost anything once pinned?)
#
# Run on the SAME quiesced host as the Zen 5 sweep (see PROVENANCE.txt interventions).
# Expected wall time: 3 arms × 3 W × 5 trials × ~75 s ≈ 1 h.
#
#   ./bench/spin-proof.sh                    # W="1 8 14", n=5
#   WORKERS="8 14" TRIALS=3 ./bench/spin-proof.sh
#   WITH_D=1 ./bench/spin-proof.sh           # add arm D
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

WORKERS="${WORKERS:-1 8 14}"
TRIALS="${TRIALS:-5}"
DURATION="${DURATION:-60}"
RATE="${RATE:-50000}"
OUT="${OUT:-bench/results-zen5/spin-proof.csv}"
LOGDIR="${LOGDIR:-bench/spin-proof-logs}"
WITH_D="${WITH_D:-0}"
export TRIALS DURATION RATE OUT LOGDIR ALLOW_CONTENDED=1 EMBED_MODE=real

# Sanity check BEFORE spending an hour: prove the extra option actually reaches the
# runtime on this exact build (the .d.ts says WebAssembly-only; the binary disagrees).
node --input-type=module - <<'JS' || { echo "FATAL: allow_spinning not honoured by this onnxruntime-node build"; exit 1; }
import { createNativeEmbedder } from './src/embed-native.mjs';
const ne = await createNativeEmbedder({ intraOpNumThreads: 1, allowSpinning: false });
if (ne.allowSpinning !== false) process.exit(1);
console.log('preflight: session created with session.intra_op.allow_spinning=0 →', ne.modelPath);
JS

run_arm() {  # label  ORT_INTRA_OP  PIN_CORES  ORT_ALLOW_SPINNING
  echo; echo "═══════ arm $1 ═══════"
  LABEL="$1" ORT_INTRA_OP="$2" PIN_CORES="$3" ORT_ALLOW_SPINNING="$4" WORKERS="$WORKERS" \
    ./bench/scaling-sweep.sh
}
run_arm "A:unpinned-spinON"  0 0 ""
run_arm "B:unpinned-spinOFF" 0 0 0
run_arm "C:pinned-spinON"    1 1 ""
(( WITH_D == 1 )) && run_arm "D:pinned-spinOFF" 1 1 0

echo; echo "════════ spin-proof verdict ($OUT) ════════"
python3 - "$OUT" <<'PY'
import csv, sys, statistics as st
rows = list(csv.DictReader(open(sys.argv[1])))
by = {}
for r in rows: by.setdefault((r['label'], int(r['workers'])), []).append(float(r['achieved_rps']))
Ws = sorted({w for _, w in by})
for w in Ws:
    a = by.get(('A:unpinned-spinON', w)); b = by.get(('B:unpinned-spinOFF', w)); c = by.get(('C:pinned-spinON', w))
    if not (a and b and c): continue
    A, B, C = st.mean(a), st.mean(b), st.mean(c)
    rec = (B - A) / (C - A) * 100 if C != A else float('nan')
    print(f"W={w:2d}  A unpinned/spinON={A:9.1f}  B unpinned/spinOFF={B:9.1f}  C pinned={C:9.1f}  "
          f"B/A={B/A:5.2f}x   recovery of the pinned gap: {rec:5.1f}%")
print("\nReading: recovery ≳ 50% at W=8/14 ⇒ spin-wait is the dominant mechanism (promote §III-C2 to a result).")
print("         recovery ≲ 10%          ⇒ spin-wait is NOT the mechanism; the 66% hot site is something else — demote.")
print("         in between              ⇒ spin is one of several losses; report the split.")
PY
