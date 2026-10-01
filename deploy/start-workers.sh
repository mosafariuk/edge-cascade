#!/usr/bin/env bash
# start-workers.sh — launch a taskset-pinned worker fleet, one worker per PHYSICAL
# core, each with onnxruntime intraOpNumThreads=1. taskset stops the OS scheduler
# from migrating a worker off its core (kills the last of the co-location tax).
#
#   REDIS_PASSWORD=... ./deploy/start-workers.sh [NUM_WORKERS]
# Stop with:  ./deploy/start-workers.sh stop
set -euo pipefail
cd "$(dirname "$0")/.."

PIDFILE=".worker-pids"
if [ "${1:-}" = "stop" ]; then
  [ -f "$PIDFILE" ] && xargs -r kill < "$PIDFILE" 2>/dev/null || true
  rm -f "$PIDFILE"; echo "workers stopped"; exit 0
fi

REDIS_PASSWORD="${REDIS_PASSWORD:?set REDIS_PASSWORD}"
RESERVE="${RESERVE:-2}"                       # cores left for Redis/PG/OS
PHYS="$(lscpu -p=CORE | grep -vc '^#')"       # count of logical CPUs
# one logical CPU id per physical core (dedupe the CORE column)
mapfile -t CORE_CPUS < <(lscpu -p=CPU,CORE | grep -v '^#' | sort -t, -k2 -u -n | cut -d, -f1)
PHYS_CORES="${#CORE_CPUS[@]}"
NUM_WORKERS="${1:-$(( PHYS_CORES - RESERVE ))}"
[ "$NUM_WORKERS" -ge 1 ] || { echo "not enough cores"; exit 1; }

export REDIS_URL="redis://:${REDIS_PASSWORD}@127.0.0.1:6379"
export EMBED_MODE=real ORT_INTRA_OP=1 UV_THREADPOOL_SIZE="${UV_THREADPOOL_SIZE:-2}"
export EGRESS="${EGRESS:-pgvector}" PG_URL="${PG_URL:-postgres://bench:bench@127.0.0.1:5432/cascade}"
export VLLM_URL="${VLLM_URL:-http://127.0.0.1:8000/v1/completions}"
# Cascade defaults: ROUTE_TAU=0.75 with the local-first router tries the edge model on every
# admitted payload; EXTRACT=1 runs the evaluated guard configuration (2-shot prompt, s_max=0).
# For the embed-bound capacity benchmark use:  ROUTE_TAU=2 EXTRACT=0 ./deploy/start-workers.sh
export B_MAX="${B_MAX:-32}" W_MAX_MS="${W_MAX_MS:-19}" ROUTE_TAU="${ROUTE_TAU:-0.75}" EXTRACT="${EXTRACT:-1}"

echo "physical cores=$PHYS_CORES  logical=$PHYS  reserve=$RESERVE  workers=$NUM_WORKERS  intraOp=1"
: > "$PIDFILE"
for ((i=0; i<NUM_WORKERS; i++)); do
  cpu="${CORE_CPUS[$i]}"
  # WORKER_SLOT gives the consumer a name that survives restarts (PEL recovery).
  WORKER_SLOT="$i" taskset -c "$cpu" node src/pipeline-worker.mjs > "logs-worker-$i.log" 2>&1 &
  echo $! >> "$PIDFILE"
  echo "  worker $i -> pinned to CPU $cpu (pid $!)"
done
echo "fleet up. tail: tail -f logs-worker-*.log | stop: ./deploy/start-workers.sh stop"
