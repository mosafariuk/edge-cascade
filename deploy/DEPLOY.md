# Bare-Metal Deployment Protocol (Ubuntu 24.04, AMD Ryzen)

Goal: publication-grade compute-bound metrics with **zero co-location tax** — workers on
the server, load generator on a **separate** machine.

## Topology
```
   [ load-gen box ]  --XADD ingest / XREAD results-->  [ server: Ryzen, Ubuntu 24.04 ]
   node + ioredis        (over VPC, auth+firewall)       Redis + Postgres (docker)
                                                         N pinned workers (bare metal, taskset)
                                                         mock-vLLM (or real vLLM)
```
Workers run **directly on the host** (not in Docker) so `taskset` pins each to one physical
core with no container-scheduler interference. Infra (Redis/PG) stays in Docker.

## Step 1 — provision the server
```bash
scp edge-cascade.tar.gz user@server:/opt/ && ssh user@server
cd /opt && tar xzf edge-cascade.tar.gz
sudo REDIS_PASSWORD='CHANGE-ME-strong' REDIS_BIND='10.0.0.5' ./edge-cascade/deploy/deploy.sh
#   REDIS_BIND = the server's VPC/private IP the load-gen will connect to
```
`deploy.sh` installs Docker + Node 20 (glibc → `onnxruntime-node` native binary loads),
`npm ci`, warms the model cache, and boots Redis/Postgres/mock-vLLM with auth.

## Step 2 — launch the pinned worker fleet
```bash
cd /opt/edge-cascade
REDIS_PASSWORD='CHANGE-ME-strong' ./deploy/start-workers.sh          # auto-sizes to cores-RESERVE
#   ROUTE_TAU=2 (default) = pure embed-bound run for μ_sys (local path off, every payload
#   hits the heavy stub). For the full cascade set ROUTE_TAU=0.75 with the default
#   ROUTER=local-first (score is 1, so every payload is tried locally and the
#   value-surprisal guard decides escalation), and point VLLM_URL at a real server.
```
Each worker: `taskset -c <core> node src/pipeline-worker.mjs` with `ORT_INTRA_OP=1`.

### Core-count tuning (the important table)
`worker_count = physical_cores − RESERVE`. Pin one worker per **physical** core; reserve a
few for Redis/PG/OS. `intraOpNumThreads=1` is invariant — it's what makes scaling linear.

| Physical cores | Workers | RESERVE | `ORT_INTRA_OP` | `UV_THREADPOOL_SIZE` | `B_MAX` |
|---|---|---|---|---|---|
| 8  (M1 baseline) | 6  | 2 | 1 | 2 | 32 |
| **16** (Ryzen)   | **14** | 2 | **1** | 2 | 32 |
| **32** (Ryzen)   | **28–30** | 2–4 | **1** | 2 | 32–48 |

Notes:
- **`UV_THREADPOOL_SIZE` is per-process, not a core count.** Each worker owns one core, so
  2 is plenty (tokenizer I/O + the single ORT async op). Do **not** set it to the machine's
  core count per worker — that's the mistake the whole thread-pinning finding warns against.
- **SMT/hyperthreading:** `start-workers.sh` pins to one logical CPU **per physical core**
  (via `lscpu -p`). Physical-core pinning is the clean baseline. You *may* test doubling to
  SMT siblings for +15–30% aggregate INT8 throughput — measure, don't assume.
- **Larger `B_MAX` at 32 cores:** more workers pulling raises per-worker λ_obs; a slightly
  bigger batch cap improves matmul amortization. Sweep 32→48 and keep the p99-respecting value.

### Projected μ_sys (label as projection until measured)
From the M1 measurement of **~420 embed-req/s per pinned core** (840 req/s ÷ 2 workers):

| Server | Workers | Linear projection @420/core | Reality check |
|---|---|---|---|
| 16-core Ryzen | 14 | ~5,900 req/s | Ryzen x86 AVX2 (or AVX-512 VNNI for INT8) may run **faster** per core |
| 32-core Ryzen | 28 | ~11,800 req/s | watch for memory-bandwidth ceiling before core ceiling |

These are extrapolations from Apple-M1 per-core throughput; **x86 INT8 with VNNI typically
exceeds it**, but memory bandwidth can bind before cores do at 28+ workers. Record the real
curve with Step 4 and replace this table.

## Step 3 — network & security (external load-gen)
An **unauthenticated, internet-exposed Redis is remote code execution.** Do exactly one of:

**A. VPC-private + auth + firewall (production):**
```bash
# deploy.sh already set requirepass + protected-mode and published Redis on REDIS_BIND only.
sudo ufw allow from <LOADGEN_IP> to any port 6379 proto tcp   # allowlist the load-gen box only
sudo ufw enable
# load-gen connects with:  redis://:<password>@<REDIS_BIND>:6379
```
Never publish on a public IP; bind to the VPC/private interface (`REDIS_BIND`) and restrict
by security-group / `ufw` to the load-gen's IP.

**B. SSH tunnel (simplest, nothing exposed):**
```bash
# on the load-gen box:
ssh -N -L 6379:127.0.0.1:6379 user@server &     # forwards local 6379 -> server Redis
SERVER_IP=127.0.0.1 REDIS_PASSWORD='...' ./deploy/loadgen-remote.sh
```
Keep `REDIS_BIND=127.0.0.1` on the server; the tunnel needs no open firewall port.

## Step 4 — run the distributed load test (from the load-gen box)
```bash
tar xzf edge-cascade.tar.gz && cd edge-cascade && npm install ioredis
SERVER_IP=10.0.0.5 REDIS_PASSWORD='CHANGE-ME-strong' \
  RATE_START=100 RATE_END=8000 DURATION=60 HARD_FRAC=0.2 \
  ./deploy/loadgen-remote.sh
```
It preflights Redis reachability, then ramps and prints TTA mean/p50/p95/p99 + backpressure
events. Watch the server in parallel:
```bash
# on the server:
htop                                   # confirm each worker pins ~100% on its own core
watch -n1 'redis-cli -a "$REDIS_PASSWORD" XLEN ingest'   # backlog = you passed μ_sys
```
The knee (where p99 goes vertical / ingest backlog grows without bound) is μ_sys.

## Step 5 — teardown
```bash
./deploy/start-workers.sh stop
docker compose -f bench/docker-compose.yml -f deploy/compose.server.yml down -v
```
