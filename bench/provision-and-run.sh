#!/usr/bin/env bash
# provision-and-run.sh — provision a CLEAN bare-metal box, run the scaling sweep, pull results back.
#
# RUNS LOCALLY.  Drives the remote host over ssh/rsync.  Does not need to be copied anywhere.
#
#   ./bench/provision-and-run.sh <user@host> [ssh-key]
#
#   TARGET=root@1.2.3.4 KEY=~/.ssh/id_ed25519 ./bench/provision-and-run.sh
#   WORKERS="1 2 3 4 5" TRIALS=5 ./bench/provision-and-run.sh root@1.2.3.4 ~/.ssh/ax41.pem
#
# WHAT IT DOES
#   1. Refuses to touch production hosts listed in the PROD_HOST_IP environment variable.
#   2. Asserts the target is BARE METAL — aborts on any hypervisor signal.  A CPU-pinning
#      benchmark on virtualized topology is methodologically void: taskset pins to vCPUs,
#      thread_siblings_list is a hypervisor fiction, and steal time is indistinguishable
#      from the contention being measured.
#   3. Asserts the target is IDLE (no co-resident web/db/broker services, low load).
#   4. Installs Node.js 24 (NodeSource), Redis 7, build-essential, python3.
#   5. rsyncs the repo (excluding node_modules/artifacts), npm install.
#   6. Runs bench/scaling-sweep.sh with whatever WORKERS/TRIALS/etc you set.
#   7. scps sweep-results.csv + logs back into bench/sweep-<host>-<stamp>/.
#
# It is idempotent: re-running skips installs that are already satisfied.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

TARGET="${1:-${TARGET:-}}"
KEY="${2:-${KEY:-}}"
[[ -n "$TARGET" ]] || { echo "usage: $0 <user@host> [ssh-key]"; exit 1; }

# ── HARD SAFETY RAIL ─────────────────────────────────────────────────────────
# Hosts that carry live services.  This script saturates every core; running it
# against one of them would degrade production.  The addresses are NOT kept in the
# repository: supply them through the environment (space-separated IPs or hostnames),
# e.g. in your shell profile:
#     export PROD_HOST_IP="203.0.113.10 prod-db.internal"
read -r -a PROD_BLOCKLIST <<< "${PROD_HOST_IP:-}"
if (( ${#PROD_BLOCKLIST[@]} == 0 )); then
  echo "WARNING: PROD_HOST_IP is not set — no production blocklist is active."
  echo "         Double-check that $TARGET is a disposable benchmark host."
fi
for bad in "${PROD_BLOCKLIST[@]}"; do
  if [[ "$TARGET" == *"$bad"* ]]; then
    echo "REFUSING: $TARGET matches a host in PROD_HOST_IP (production, live services)."
    echo "This script saturates all cores and would degrade them."
    exit 1
  fi
done

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)
[[ -n "$KEY" ]] && SSH_OPTS+=(-i "$KEY")
RSH="ssh ${SSH_OPTS[*]}"
sshx() { ssh "${SSH_OPTS[@]}" "$TARGET" "$@"; }

REMOTE_DIR="${REMOTE_DIR:-/opt/edge-cascade-bench}"
STAMP="$(date +%Y%m%d-%H%M%S)"
HOSTTAG="$(echo "$TARGET" | tr -c 'A-Za-z0-9._-' '_')"
LOCAL_OUT="$HERE/sweep-${HOSTTAG}-${STAMP}"

echo "════ target: $TARGET   remote dir: $REMOTE_DIR ════"

# ── 1. reachability ──────────────────────────────────────────────────────────
sshx 'echo ok' >/dev/null 2>&1 || { echo "FATAL: cannot ssh to $TARGET"; exit 1; }

# ── 2 & 3. bare-metal + idleness assertions (remote, read-only) ──────────────
echo "── verifying host is bare metal and idle ──"
PRECHECK=$(sshx 'bash -s' <<'REMOTE'
set -u
fail=0

# NOTE: systemd-detect-virt EXITS NON-ZERO on bare metal while still printing "none".
# Do not use `|| echo ...` here — it appends a second line and breaks the comparison.
# Likewise `grep -c` prints "0" and exits 1 when there are no matches.
virt="$(systemd-detect-virt 2>/dev/null)"
[[ -z "$virt" ]] && virt="unknown"
hyp_flag="$(grep -c '\bhypervisor\b' /proc/cpuinfo 2>/dev/null)"
[[ -z "$hyp_flag" ]] && hyp_flag="0"
hyp_vendor="$(lscpu 2>/dev/null | grep -i 'Hypervisor vendor' || true)"

echo "virt=$virt"
echo "cpuinfo_hypervisor_flag_count=$hyp_flag"
echo "lscpu_hypervisor_vendor=${hyp_vendor:-none}"

if [[ "$virt" != "none" ]]; then echo "VERDICT_VIRT=FAIL(systemd-detect-virt=$virt)"; fail=1
elif [[ "$hyp_flag" != "0" ]]; then echo "VERDICT_VIRT=FAIL(hypervisor cpu flag present)"; fail=1
elif [[ -n "$hyp_vendor" ]]; then echo "VERDICT_VIRT=FAIL(lscpu reports hypervisor vendor)"; fail=1
else echo "VERDICT_VIRT=PASS"; fi

echo "model=$(lscpu | grep '^Model name' | sed 's/.*: *//')"
echo "phys_cores=$(lscpu | awk -F: '/^Core\(s\) per socket/{gsub(/ /,"",$2);print $2}')"
echo "threads_per_core=$(lscpu | awk -F: '/^Thread\(s\) per core/{gsub(/ /,"",$2);print $2}')"

load1="$(cut -d' ' -f1 /proc/loadavg)"
echo "load1=$load1"
awk -v l="$load1" 'BEGIN{ if (l+0 > 1.0) print "VERDICT_LOAD=WARN(load1="l")"; else print "VERDICT_LOAD=PASS" }'

busy=""
for svc in nginx apache2 mariadbd mysqld postgres redis-server rabbitmq-server celery php-fpm node next-server; do
  pgrep -x "$svc" >/dev/null 2>&1 && busy="$busy $svc"
done
if [[ -n "$busy" ]]; then echo "VERDICT_CLEAN=WARN(co-resident:$busy)"; else echo "VERDICT_CLEAN=PASS"; fi
exit $fail
REMOTE
)
echo "$PRECHECK" | sed 's/^/    /'

if grep -q 'VERDICT_VIRT=FAIL' <<<"$PRECHECK"; then
  echo
  echo "ABORT: target is virtualized."
  echo "  taskset would pin to vCPUs, not physical cores; thread_siblings_list is a"
  echo "  hypervisor fiction; steal time is indistinguishable from real contention."
  echo "  This benchmark's conclusions would be void.  Use bare metal."
  exit 1
fi
if grep -q 'VERDICT_CLEAN=WARN' <<<"$PRECHECK"; then
  echo
  echo "WARNING: co-resident services detected on the target."
  echo "  The whole point of this box is an unloaded host.  Continue anyway? [y/N]"
  read -r ans; [[ "$ans" == "y" || "$ans" == "Y" ]] || exit 1
fi

# ── 4. dependency chain ──────────────────────────────────────────────────────
echo "── installing dependencies (idempotent) ──"
sshx 'bash -s' <<'REMOTE'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

. /etc/os-release
echo "os: $PRETTY_NAME"
[[ "$ID" == "ubuntu" ]] || echo "  (note: tuned for Ubuntu 24.04; proceeding on $ID)"

apt-get update -qq
apt-get install -y -qq build-essential python3 redis-server util-linux curl ca-certificates gnupg rsync time >/dev/null

if ! command -v node >/dev/null 2>&1 || [[ "$(node -v | cut -c2- | cut -d. -f1)" -lt 24 ]]; then
  echo "installing Node.js 24 via NodeSource"
  mkdir -p /etc/apt/keyrings
  curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
    | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
  echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_24.x nodistro main" \
    > /etc/apt/sources.list.d/nodesource.list
  apt-get update -qq && apt-get install -y -qq nodejs >/dev/null
fi

# host redis must not squat :6390 or compete; sweep starts its own pinned instance
systemctl disable --now redis-server >/dev/null 2>&1 || true

echo "node:  $(node -v)"
echo "redis: $(redis-server --version | grep -oE 'v=[0-9.]+' || echo unknown)"
echo "taskset: $(command -v taskset || echo MISSING)"
REMOTE
[[ $? -eq 0 ]] || { echo "FATAL: dependency install failed"; exit 1; }

# ── 5. sync repo + npm install ───────────────────────────────────────────────
echo "── syncing repo → $REMOTE_DIR ──"
sshx "mkdir -p '$REMOTE_DIR'"
rsync -az --delete -e "$RSH" \
  --exclude 'node_modules/' --exclude '.git/' --exclude '*.pdf' \
  --exclude 'bench/sweep-logs/' --exclude 'bench/sweep-results.csv' --exclude 'bench/sweep-*/' \
  "$ROOT/" "$TARGET:$REMOTE_DIR/"

echo "── npm install ──"
sshx "cd '$REMOTE_DIR' && npm install --no-audit --no-fund 2>&1 | tail -5"

# ── 6. run the sweep ─────────────────────────────────────────────────────────
echo "── running sweep ──"
sshx "cd '$REMOTE_DIR' && chmod +x bench/scaling-sweep.sh && \
  WORKERS='${WORKERS:-1 2 3 4}' \
  TRIALS='${TRIALS:-5}' \
  DURATION='${DURATION:-60}' \
  RATE='${RATE:-4000}' \
  EMBED_MODE='${EMBED_MODE:-real}' \
  ORT_INTRA_OP='${ORT_INTRA_OP:-1}' \
  ALLOW_CONTENDED='${ALLOW_CONTENDED:-0}' \
  ./bench/scaling-sweep.sh"
SWEEP_RC=$?

# ── 7. retrieve results ──────────────────────────────────────────────────────
echo "── retrieving results → $LOCAL_OUT ──"
mkdir -p "$LOCAL_OUT"
rsync -az -e "$RSH" \
  "$TARGET:$REMOTE_DIR/bench/sweep-results.csv" "$LOCAL_OUT/" 2>/dev/null \
  || echo "  (no sweep-results.csv — sweep may have failed)"
rsync -az -e "$RSH" \
  "$TARGET:$REMOTE_DIR/bench/sweep-logs/" "$LOCAL_OUT/sweep-logs/" 2>/dev/null \
  || echo "  (no sweep-logs)"

# provenance: exactly what hardware produced these numbers
sshx 'echo "=== HOST PROVENANCE ==="; date -u; systemd-detect-virt; lscpu; echo; uname -a; \
      echo; node -v; redis-server --version; echo; cat /proc/loadavg' \
  > "$LOCAL_OUT/provenance.txt" 2>&1

echo
echo "════ done (sweep rc=$SWEEP_RC) ════"
echo "results:    $LOCAL_OUT/sweep-results.csv"
echo "provenance: $LOCAL_OUT/provenance.txt   <- keep this; it is the artifact-evaluation evidence"
echo
echo "Retain sweep-results.csv AND provenance.txt in the repo this time."
echo "The previous Ryzen run was torn down and left no artifact; that is why Table II"
echo "currently has no reproducible backing."
