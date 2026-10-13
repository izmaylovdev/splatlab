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
# The pool talks to Temporal/Redis locally (TEMPORAL_ADDRESS / REDIS_URL). Boxes
# reach them over an SSH link the pool opens to each box (SPLATLAB_BOX_LINK=ssh,
# the default: key SPLATLAB_BOX_SSH_KEY, default ~/.ssh/vast_ai; MinIO address to
# forward: SPLATLAB_BOX_FORWARD_S3). With SPLATLAB_BOX_LINK=relay the pool instead
# FORWARDS these box-facing addresses — the GCP relay's public ports, written into
# deploy/.env by `deploy/gcp_tunnel_up.sh`:
#   BOX_TEMPORAL_ADDRESS      e.g. 203.0.113.7:17233
#   BOX_REDIS_URL             e.g. redis://:PASS@203.0.113.7:16379
#   BOX_S3_ENDPOINT           e.g. http://203.0.113.7:19000   (+ SPLATLAB_S3_* / AWS_* creds)
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
