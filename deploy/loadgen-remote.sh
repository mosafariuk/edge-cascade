#!/usr/bin/env bash
# loadgen-remote.sh — run on the EXTERNAL load-gen box (never on the worker host).
# Floods the server's Redis 'ingest' stream and reads back 'results' for true TTA.
#
#   SERVER_IP=10.0.0.5 REDIS_PASSWORD='<secret>' ./deploy/loadgen-remote.sh
# Requires: node 20 + this repo + `npm install ioredis` on the external box.
set -euo pipefail
cd "$(dirname "$0")/.."

SERVER_IP="${SERVER_IP:?set SERVER_IP to the server VPC or private IP}"
REDIS_PASSWORD="${REDIS_PASSWORD:?set REDIS_PASSWORD}"
export REDIS_URL="redis://:${REDIS_PASSWORD}@${SERVER_IP}:6379"

# connectivity preflight (heredoc avoids shell-quoting pitfalls; reads REDIS_URL from env)
node <<'JS'
const R = require('ioredis');
const r = new R(process.env.REDIS_URL, { lazyConnect: true, connectTimeout: 4000 });
r.connect()
  .then(() => r.ping())
  .then((x) => { console.log('redis reachable:', x); return r.quit(); })
  .catch((e) => { console.error('CANNOT REACH REDIS:', e.message); process.exit(1); });
JS

export RATE_START="${RATE_START:-100}" RATE_END="${RATE_END:-8000}" \
       DURATION="${DURATION:-60}" HARD_FRAC="${HARD_FRAC:-0.2}"
echo "flooding ${SERVER_IP}: ramp ${RATE_START} -> ${RATE_END} req/s over ${DURATION}s"
node bench/load-gen.mjs
