#!/usr/bin/env bash
# scaling-sweep.sh — worker-count scaling sweep with repeated trials and error bars.
#
# Produces the per-worker-count throughput curve that Table II needs, including the
# saturation knee past the physical core budget.
#
# METHODOLOGY NOTES (read these; they affect whether the numbers mean anything)
#
#   1. FIXED-RATE, NOT RAMPED.  load-gen.mjs reports `done/wall` — a cumulative mean
#      over the whole run.  Under the default linear ramp (RATE_START≪RATE_END) that
#      average includes the under-loaded early phase and is biased LOW relative to
#      steady-state mu_sys.  We therefore drive RATE_START == RATE_END == RATE, well
#      above capacity, so the system is saturated for the entire window and the
#      cumulative mean converges to steady state.  Residual bias: worker warmup and
#      the 3s drain tail are still included, so longer DURATION => less bias.
#      DURATION=60 keeps that bias under ~5%.
#
#   2. CORE BUDGET IS REAL.  N pinned workers need N physical cores, PLUS a core for
#      the load generator and one for Redis.  On a 6-core host a genuinely isolated
#      sweep tops out at 4 workers.  Beyond that the infra lands on SMT siblings and
#      contends — the script marks those rows CONTENDED and refuses to run them
#      unless ALLOW_CONTENDED=1.  Do not report contended rows as clean data points.
#
#   3. OVERSUBSCRIPTION ROWS ARE THE POINT.  W > physical cores is contended BY
#      DESIGN — that is where scaling breaks and where the paper's thesis is tested.
#      Run those deliberately, label them as such.
#
# USAGE
#   chmod +x bench/scaling-sweep.sh
#   ./bench/scaling-sweep.sh                       # default sweep
#   WORKERS="1 2 3 4" TRIALS=5 ./bench/scaling-sweep.sh
#   WORKERS="8 12" ALLOW_CONTENDED=1 ./bench/scaling-sweep.sh   # saturation knee
#   ORT_INTRA_OP=0 WORKERS="2 4" ./bench/scaling-sweep.sh       # unpinned A/B
#
# OUTPUT
#   bench/sweep-results.csv   raw per-trial rows
#   stdout                    per-W mean, sample stdev, 95% CI half-width
#
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
cd "$ROOT"

# ── configuration ────────────────────────────────────────────────────────────
WORKERS="${WORKERS:-1 2 3 4}"
TRIALS="${TRIALS:-5}"
DURATION="${DURATION:-60}"
RATE="${RATE:-4000}"                 # fixed offered load; must exceed capacity
REDIS_PORT="${REDIS_PORT:-6390}"     # dedicated instance, never the host's live one
REDIS_URL="redis://127.0.0.1:${REDIS_PORT}"
ORT_INTRA_OP="${ORT_INTRA_OP:-1}"
EMBED_MODE="${EMBED_MODE:-real}"
ROUTE_TAU="${ROUTE_TAU:-2}"
HARD_FRAC="${HARD_FRAC:-0.0}"        # 0 = pure embed workload, no escalation path
ALLOW_CONTENDED="${ALLOW_CONTENDED:-0}"
# PIN_CORES=1 (default): each worker is taskset'd to one physical core.
# PIN_CORES=0: no taskset at all — required to reproduce the PATHOLOGICAL arm, where
#   ORT_INTRA_OP=0 lets ORT spawn one intra-op thread per core in EVERY worker and the
#   fleet oversubscribes the machine.  With taskset applied, an "unpinned" ORT is still
#   confined to a single core by the OS, so the A/B would be meaningless.
PIN_CORES="${PIN_CORES:-1}"
# ORT_ALLOW_SPINNING: "" (runtime default), 0 or 1 → session.intra_op.allow_spinning.
#   0 with PIN_CORES=0 ORT_INTRA_OP=0 is the direct test of the spin-wait hypothesis.
ORT_ALLOW_SPINNING="${ORT_ALLOW_SPINNING:-}"
# LABEL: free-text arm name written to every row (e.g. "intra1+taskset"), so a 2x2
#   ablation file is self-describing instead of relying on column combinations.
LABEL="${LABEL:-}"
OUT="${OUT:-$HERE/sweep-results.csv}"
LOGDIR="${LOGDIR:-$HERE/sweep-logs}"
# CORE_LIST: explicit worker-core placement, e.g. "0 1 2 3 8 9 10 11".
#   Needed for CCD/L3 isolation experiments on chiplet CPUs, where WHICH cores you use
#   changes L3 domain and cross-fabric distance to the infra core.  Default = physical
#   cores in ascending order, which on a 9950X packs W<=8 entirely onto CCD0.
# INFRA_CORE_OVERRIDE: place load-gen + redis on a specific core (default = last physical).
CORE_LIST="${CORE_LIST:-}"

command -v taskset >/dev/null || { echo "FATAL: taskset not found (Linux only)"; exit 1; }
command -v node    >/dev/null || { echo "FATAL: node not found"; exit 1; }

# ── topology: one entry per PHYSICAL core (first sibling of each sibling group) ──
mapfile -t PHYS < <(
  for d in /sys/devices/system/cpu/cpu[0-9]*; do
    sib="$d/topology/thread_siblings_list"
    [[ -r "$sib" ]] || continue
    cut -d, -f1 < "$sib" | cut -d- -f1
  done | sort -n -u
)
NPHYS=${#PHYS[@]}
(( NPHYS > 0 )) || { echo "FATAL: could not read CPU topology from /sys"; exit 1; }

# infra gets the LAST physical core; workers take from the front
INFRA_CORE="${INFRA_CORE_OVERRIDE:-${PHYS[$((NPHYS - 1))]}}"
USABLE=$((NPHYS - 1))

# explicit worker-core placement overrides the default ascending order
if [[ -n "$CORE_LIST" ]]; then
  read -r -a WORKER_CORES <<< "$CORE_LIST"
  USABLE=${#WORKER_CORES[@]}
  echo "CORE_LIST override: workers pinned to [${WORKER_CORES[*]}]"
else
  WORKER_CORES=("${PHYS[@]}")
fi

echo "topology: ${NPHYS} physical cores [${PHYS[*]}]"
echo "infra core (load-gen + redis): ${INFRA_CORE}   usable worker cores: ${USABLE}"
echo "config: TRIALS=$TRIALS DURATION=${DURATION}s RATE=${RATE}/s ORT_INTRA_OP=$ORT_INTRA_OP PIN_CORES=$PIN_CORES ALLOW_SPINNING=${ORT_ALLOW_SPINNING:-default} EMBED_MODE=$EMBED_MODE${LABEL:+ LABEL=$LABEL}"
echo

mkdir -p "$LOGDIR"
[[ -f "$OUT" ]] || echo "workers,trial,achieved_rps,mean_ms,p50_ms,p95_ms,p99_ms,contended,ort_intra_op,pin_cores,embed_mode,rate,duration,loadgen_cpu_pct,redis_cpu_pct,allow_spinning,label,core_list" > "$OUT"

cleanup_workers() {
  [[ -n "${WORKER_PIDS:-}" ]] && kill $WORKER_PIDS 2>/dev/null
  wait $WORKER_PIDS 2>/dev/null
  WORKER_PIDS=""
}
trap 'echo; echo "interrupted — cleaning up"; cleanup_workers; exit 130' INT TERM

# ── redis: dedicated instance on its own port, pinned to the infra core ───────
if ! redis-cli -p "$REDIS_PORT" ping >/dev/null 2>&1; then
  command -v redis-server >/dev/null || { echo "FATAL: redis-server not found and nothing on :$REDIS_PORT"; exit 1; }
  echo "starting dedicated redis on :$REDIS_PORT (core $INFRA_CORE)"
  taskset -c "$INFRA_CORE" redis-server --port "$REDIS_PORT" --save '' --appendonly no \
    >"$LOGDIR/redis.log" 2>&1 &
  REDIS_PID=$!
  for _ in $(seq 1 50); do redis-cli -p "$REDIS_PORT" ping >/dev/null 2>&1 && break; sleep 0.2; done
  redis-cli -p "$REDIS_PORT" ping >/dev/null 2>&1 || { echo "FATAL: redis failed to start"; exit 1; }
else
  echo "reusing redis already listening on :$REDIS_PORT"
fi

# ── sweep ────────────────────────────────────────────────────────────────────
for W in $WORKERS; do
  CONTENDED=0
  if (( W > USABLE )); then
    CONTENDED=1
    if (( ALLOW_CONTENDED != 1 )); then
      echo "SKIP W=$W — needs $W cores but only $USABLE are free of infra."
      echo "     This row would be co-location-contaminated.  Re-run with ALLOW_CONTENDED=1"
      echo "     if you intend it as a deliberate oversubscription data point."
      echo
      continue
    fi
    echo "W=$W  *** CONTENDED *** (exceeds $USABLE isolated cores — oversubscription row)"
  fi

  for T in $(seq 1 "$TRIALS"); do
    printf 'W=%d trial %d/%d ... ' "$W" "$T" "$TRIALS"

    redis-cli -p "$REDIS_PORT" flushall >/dev/null 2>&1

    WORKER_PIDS=""
    for ((i = 0; i < W; i++)); do
      CORE="${WORKER_CORES[$((i % ${#WORKER_CORES[@]}))]}"
      if (( PIN_CORES == 1 )); then PINCMD=(taskset -c "$CORE"); else PINCMD=(); fi
      UV_THREADPOOL_SIZE=4 \
      ORT_INTRA_OP="$ORT_INTRA_OP" \
      ORT_ALLOW_SPINNING="$ORT_ALLOW_SPINNING" \
      EMBED_MODE="$EMBED_MODE" \
      ROUTE_TAU="$ROUTE_TAU" \
      EXTRACT=0 \
      REDIS_URL="$REDIS_URL" \
      EGRESS="none" \
      "${PINCMD[@]}" node src/pipeline-worker.mjs \
        >"$LOGDIR/w${W}_t${T}_worker${i}.log" 2>&1 &
      WORKER_PIDS="$WORKER_PIDS $!"
    done

    # Wait until every worker has actually JOINED THE CONSUMER GROUP.
    # Do NOT grep worker logs for 'intraOpNumThreads=' — that line is only emitted on
    # the native-embedder path, so it never appears under EMBED_MODE=hash and the check
    # silently times out on every trial.  Consumer-group membership is mode-agnostic and
    # proves the worker is genuinely consuming, not merely alive.
    # Readiness = every worker has OPENED ITS REDIS CONNECTIONS.
    #
    # Do NOT wait on XINFO GROUPS consumer count: on Redis 7.x a consumer is only
    # registered once it actually READS AN ENTRY, and the stream is empty until the
    # load generator starts — which this gate would be blocking.  Measured on this
    # harness: each pipeline-worker opens exactly 3 connections (reader, writer,
    # metricsRedis), so W workers => 3W client connections (+1 for the polling
    # redis-cli itself).  Threshold 3W is reached only when the last worker is
    # essentially up, and cannot be satisfied by W-1 workers (3(W-1)+1 = 3W-2).
    # Second condition matters: workers open their Redis connections BEFORE loading the
    # ONNX session, so connection count alone would mark them ready mid-model-load and
    # put session-init time inside the measurement window.  Under EMBED_MODE=real the
    # native embedder logs 'intraOpNumThreads=', so require that too.  (Under hash mode
    # that line never appears, which is what broke the original gate.)
    NEED=$(( 3 * W ))
    READY=0; CLIENTS=0; LOADED=0
    for _ in $(seq 1 240); do
      CLIENTS=$(redis-cli -p "$REDIS_PORT" --raw CLIENT LIST 2>/dev/null | wc -l)
      if [[ "$EMBED_MODE" == "real" ]]; then
        LOADED=$(grep -l 'intraOpNumThreads=' "$LOGDIR"/w${W}_t${T}_worker*.log 2>/dev/null | wc -l)
      else
        LOADED=$W
      fi
      (( CLIENTS >= NEED && LOADED >= W )) && { READY=1; break; }
      sleep 0.5
    done
    if (( READY != 1 )); then
      echo "FAILED ($CLIENTS/$NEED redis connections; $LOADED/$W ONNX sessions loaded;"
      echo "        see $LOGDIR/w${W}_t${T}_worker*.log)"
      cleanup_workers
      continue
    fi

    sleep 2   # settle

    LG="$LOGDIR/w${W}_t${T}_loadgen.log"

    # ── harness-saturation instrumentation ────────────────────────────────────
    # If the load generator or the broker pegs its pinned core, the fleet is
    # STARVING FOR INPUT and the throughput number is a harness artifact, not a
    # scaling result.  Sample both so that is provable rather than assumed.
    HZ="$(getconf CLK_TCK 2>/dev/null || echo 100)"
    RPID="$(redis-cli -p "$REDIS_PORT" info server 2>/dev/null | awk -F: '/process_id/{gsub(/\r/,"");print $2}')"
    r_start=0; [[ -r /proc/$RPID/stat ]] && r_start=$(awk '{print $14+$15}' /proc/$RPID/stat)
    t_start=$(date +%s)

    REDIS_URL="$REDIS_URL" \
    RATE_START="$RATE" RATE_END="$RATE" \
    DURATION="$DURATION" HARD_FRAC="$HARD_FRAC" \
    /usr/bin/time -f "LOADGEN_CPU_PCT=%P" -o "$LG.time" \
      taskset -c "$INFRA_CORE" node bench/load-gen.mjs >"$LG" 2>&1

    t_end=$(date +%s); wall=$(( t_end - t_start )); (( wall < 1 )) && wall=1
    r_end=0; [[ -r /proc/$RPID/stat ]] && r_end=$(awk '{print $14+$15}' /proc/$RPID/stat)
    REDIS_CPU=$(awk -v d="$((r_end - r_start))" -v hz="$HZ" -v w="$wall" \
                    'BEGIN{ if (hz>0 && w>0) printf "%.1f", (d/hz)/w*100; else print "" }')
    LG_CPU=$(grep -oE 'LOADGEN_CPU_PCT=[0-9]+' "$LG.time" 2>/dev/null | cut -d= -f2)
    LG_CPU="${LG_CPU:-}"

    # final report block is the last occurrence of each metric
    ACH=$(grep -oE 'achieved≈[0-9]+' "$LG" | tail -1 | grep -oE '[0-9]+')
    MEAN=$(grep -oE 'mean=[0-9.]+' "$LG" | tail -1 | cut -d= -f2)
    P50=$(grep -oE 'p50=[0-9]+'  "$LG" | tail -1 | cut -d= -f2)
    P95=$(grep -oE 'p95=[0-9]+'  "$LG" | tail -1 | cut -d= -f2)
    P99=$(grep -oE 'p99=[0-9]+'  "$LG" | tail -1 | cut -d= -f2)

    cleanup_workers
    sleep 1

    if [[ -z "$ACH" ]]; then
      echo "FAILED (no throughput parsed; see $LG)"
      continue
    fi
    WARN=""
    awk -v l="${LG_CPU:-0}" -v r="${REDIS_CPU:-0}" 'BEGIN{exit !(l>90 || r>90)}' \
      && WARN="   ⚠️ HARNESS SATURATED (loadgen=${LG_CPU:-?}% redis=${REDIS_CPU:-?}%)"
    echo "${ACH} req/s   p99=${P99:-?}ms   lg=${LG_CPU:-?}% redis=${REDIS_CPU:-?}%${WARN}"
    USED_CORES=""; if (( PIN_CORES == 1 )); then USED_CORES="${WORKER_CORES[*]:0:$W}"; fi
    echo "$W,$T,$ACH,${MEAN:-},${P50:-},${P95:-},${P99:-},$CONTENDED,$ORT_INTRA_OP,$PIN_CORES,$EMBED_MODE,$RATE,$DURATION,${LG_CPU:-},${REDIS_CPU:-},${ORT_ALLOW_SPINNING:-default},${LABEL},${USED_CORES// /;}" >> "$OUT"
  done
  echo
done

# Wait for our redis to actually exit: a back-to-back invocation (spin-proof.sh,
# ablation-2x2.sh) would otherwise see `redis-cli ping` succeed on the dying instance,
# "reuse" it, and have every worker fail with ECONNREFUSED a second later.
if [[ -n "${REDIS_PID:-}" ]]; then
  kill "$REDIS_PID" 2>/dev/null; wait "$REDIS_PID" 2>/dev/null
  for _ in $(seq 1 50); do redis-cli -p "$REDIS_PORT" ping >/dev/null 2>&1 || break; sleep 0.1; done
fi

# ── summary: mean, sample stdev, 95% CI half-width (Student-t across trials) ──
echo "════════ SUMMARY ($OUT) ════════"
python3 - "$OUT" <<'PY'
import csv, sys, math, statistics as st
# two-sided 95% t critical values by degrees of freedom (n-1); z for large n
T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
rows = list(csv.DictReader(open(sys.argv[1])))
by = {}
for r in rows:
    key = (r.get('label') or '', int(r['workers']), r['ort_intra_op'], r.get('pin_cores', ''), r.get('allow_spinning', 'default'), r['contended'])
    by.setdefault(key, []).append(float(r['achieved_rps']))
print(f"{'label':<18} {'W':>3} {'intraOp':>7} {'pin':>3} {'spin':>7} {'n':>3} {'mean rps':>10} {'stdev':>8} {'95% CI ±':>9}  note")
prev = {}
for (lab, w, ort, pin, spin, cont), v in sorted(by.items()):
    m = st.mean(v)
    sd = st.stdev(v) if len(v) > 1 else 0.0
    ci = T.get(len(v) - 1, 1.96) * sd / math.sqrt(len(v)) if len(v) > 1 else float('nan')
    note = 'CONTENDED' if cont == '1' else ''
    if (lab, w // 2, ort, pin, spin, cont) in prev and w % 2 == 0:
        base = prev[(lab, w // 2, ort, pin, spin, cont)]
        note = (note + f' scaling {w//2}->{w}: {m/base:.2f}x ({100*m/base/2:.1f}%)').strip()
    prev[(lab, w, ort, pin, spin, cont)] = m
    print(f"{lab:<18} {w:>3} {ort:>7} {pin:>3} {spin:>7} {len(v):>3} {m:>10.1f} {sd:>8.1f} {ci:>9.1f}  {note}")
print("\nReport mean ± CI in the paper. A single point estimate per W will be challenged.")
PY
