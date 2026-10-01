#!/usr/bin/env bash
cd /opt/edge-cascade
echo "=== START $(date -u) ==="
WITH_D=1 OUT=bench/results-zen5-run2/spin-proof.csv LOGDIR=bench/results-zen5-run2/spin-proof-logs ./bench/spin-proof.sh
echo "=== spin-proof done $(date -u) ==="
OUT=bench/results-zen5-run2/ablation-2x2.csv LOGDIR=bench/results-zen5-run2/ablation-logs ./bench/ablation-2x2.sh
echo "=== ALL DONE $(date -u) ==="
