#!/usr/bin/env bash
# SplatLab GPU autoscaler — runs on the CONTROL PLANE (not a GPU box).
#
# Watches how many training workflows need a GPU and rents/reaps ephemeral
# Vast.ai boxes to match (see server/vast/pool.py). Keep it running alongside the
# API + control worker.
#
# Required:
#   VAST_API_KEY              your Vast.ai API key (Account → API)
#
# The addresses below must be what a *remote Vast box* can reach — NOT localhost.
# Set them to your control plane's public host / mesh-net address; the pool
# forwards them onto every box it rents.
#   TEMPORAL_ADDRESS          e.g. tcp://your-host:7233  -> host:7233
#   REDIS_URL                 e.g. redis://:pw@your-host:6379
#   SPLATLAB_STORAGE=s3       + SPLATLAB_S3_ENDPOINT / bucket / AWS_* creds
#
# Common tunables (see server/config.py for all): VAST_GPU_NAME, VAST_MAX_PRICE,
# SPLATLAB_POOL_MAX_BOXES, SPLATLAB_POOL_IDLE_TIMEOUT, VAST_IMAGE, VAST_REPO_URL.
#
#   bash deploy/run_pool.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

[ -d .venv ] && . .venv/bin/activate

if [ -z "${VAST_API_KEY:-}" ]; then
  echo "!! VAST_API_KEY is not set — the pool cannot rent GPUs." >&2
  exit 1
fi

exec python -m server.vast.pool
