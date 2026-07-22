#!/usr/bin/env bash
# SplatLab GPU autoscaler — runs on the CONTROL PLANE (not a GPU box).
#
# Watches how many training workflows need a GPU and rents/reaps ephemeral
# Vast.ai boxes to match (see server/vast/pool.py). Keep it running alongside the
# API + control worker.
#
# Required:
#   VAST_API_KEY              your Vast.ai API key (Account → API)
#   TS_AUTHKEY                Tailscale auth key — each rented box joins the tailnet
#
# The pool talks to Temporal/Redis locally (TEMPORAL_ADDRESS / REDIS_URL), but
# FORWARDS the box-facing addresses below to each rented box. Set these to your
# Mac's tailnet address (`tailscale ip -4`) so a remote box can dial back in:
#   BOX_TEMPORAL_ADDRESS      e.g. 100.x.y.z:7233
#   BOX_REDIS_URL             e.g. redis://100.x.y.z:6379
#   BOX_S3_ENDPOINT           e.g. http://100.x.y.z:9000   (+ SPLATLAB_S3_* / AWS_* creds)
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
