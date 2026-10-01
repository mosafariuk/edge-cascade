#!/usr/bin/env bash
# deploy.sh — provision the edge-cascade gateway on a bare-metal Ubuntu 24.04 host.
# Installs Docker + Node 20 (glibc → native onnxruntime-node works), restores deps,
# warms the ONNX model cache, and boots Redis + Postgres(pgvector) + mock-vLLM.
# Idempotent: safe to re-run. Workers are launched separately by start-workers.sh.
#
# Usage (from the directory containing edge-cascade.tar.gz OR inside an unpacked repo):
#   sudo REDIS_PASSWORD='<strong-secret>' REDIS_BIND='10.0.0.5' ./deploy.sh
set -euo pipefail

# ── config (override via env) ────────────────────────────────────────────────
TARBALL="${TARBALL:-edge-cascade.tar.gz}"
APP_DIR="${APP_DIR:-$PWD/edge-cascade}"
REDIS_PASSWORD="${REDIS_PASSWORD:?set REDIS_PASSWORD to a strong secret}"
REDIS_BIND="${REDIS_BIND:-127.0.0.1}"   # host interface to expose Redis on (VPC IP for remote load-gen)
PG_BIND="${PG_BIND:-127.0.0.1}"
NODE_MAJOR="${NODE_MAJOR:-20}"

log() { printf '\033[1;32m[deploy]\033[0m %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# ── 1. system packages ───────────────────────────────────────────────────────
log "apt prerequisites"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ca-certificates curl gnupg build-essential util-linux jq >/dev/null

# ── 2. Docker (official repo) ─────────────────────────────────────────────────
if ! have docker; then
  log "installing Docker Engine + compose plugin"
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin >/dev/null
fi
systemctl enable --now docker >/dev/null 2>&1 || true

# ── 3. Node.js 20 (NodeSource, glibc build for onnxruntime-node) ──────────────
if ! have node || [ "$(node -p 'process.versions.node.split(".")[0]')" -lt "$NODE_MAJOR" ]; then
  log "installing Node.js ${NODE_MAJOR}.x"
  curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash - >/dev/null
  apt-get install -y -qq nodejs >/dev/null
fi
log "node $(node -v) / npm $(npm -v) / docker $(docker --version | awk '{print $3}' | tr -d ,)"

# ── 4. unpack app ─────────────────────────────────────────────────────────────
if [ ! -d "$APP_DIR" ]; then
  [ -f "$TARBALL" ] || { echo "no $TARBALL and no $APP_DIR"; exit 1; }
  log "unpacking $TARBALL"
  tar -xzf "$TARBALL"
fi
cd "$APP_DIR"

# ── 5. production deps (pulls onnxruntime-node glibc prebuilt binary) ─────────
log "npm ci --omit=dev"
if [ -f package-lock.json ]; then npm ci --omit=dev --no-audit --no-fund; else npm install --omit=dev --no-audit --no-fund; fi
node -e "require('onnxruntime-node'); console.log('  onnxruntime-node loads OK (glibc native binding)')"

# ── 6. warm the model cache ONCE (so workers don't each download) ─────────────
log "warming all-MiniLM-L6-v2 cache"
node -e "import('@xenova/transformers').then(async m => { await m.pipeline('feature-extraction','Xenova/all-MiniLM-L6-v2',{quantized:true}); console.log('  model cached'); })"

# ── 7. boot infrastructure (Redis + Postgres/pgvector + mock-vLLM) ────────────
log "booting infra via docker compose (secure bind)"
export REDIS_PASSWORD REDIS_BIND PG_BIND
docker compose -f bench/docker-compose.yml -f deploy/compose.server.yml up -d redis postgres mock-vllm
sleep 3
docker compose -f bench/docker-compose.yml -f deploy/compose.server.yml ps

CORES="$(nproc)"
log "DONE. host has ${CORES} logical CPUs."
cat <<EOF

Next:
  1. Open the firewall to your load-gen box only:
       sudo ufw allow from <LOADGEN_IP> to any port 6379 proto tcp
  2. Launch the pinned worker fleet (see deploy/DEPLOY.md for core-count tuning):
       REDIS_PASSWORD='$REDIS_PASSWORD' ./deploy/start-workers.sh
  3. From the EXTERNAL machine, run the distributed load test:
       SERVER_IP=$REDIS_BIND REDIS_PASSWORD='<same>' ./deploy/loadgen-remote.sh
EOF
