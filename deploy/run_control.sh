#!/usr/bin/env bash
# SplatLab control worker — runs on the CONTROL PLANE (no GPU needed).
#
# Runs SplatTrainingWorkflow on the control queue. This is what keeps a run's
# orchestration alive while GPU boxes come and go; it dispatches the COLMAP /
# training activities to the GPU queue that pool-rented boxes poll.
#
#   bash deploy/run_control.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

[ -d .venv ] && . .venv/bin/activate

export SPLATLAB_WORKER_ROLE=control
export TEMPORAL_ADDRESS="${TEMPORAL_ADDRESS:-localhost:7233}"
export REDIS_URL="${REDIS_URL:-redis://localhost:6379}"
export SPLATLAB_STORAGE="${SPLATLAB_STORAGE:-s3}"

exec python -m server.worker.main
