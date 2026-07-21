#!/usr/bin/env bash
# Launch the SplatLab API service (control plane). No GPU needed.
#
# Talks to Temporal + Redis (and, on the S3 backend, MinIO/S3) and serves the
# REST API, the WebSocket telemetry, and — if the client/ dir is present — the
# static UI. Point the addresses below at your control plane.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

. .venv/bin/activate

export TEMPORAL_ADDRESS="${TEMPORAL_ADDRESS:-localhost:7233}"
export REDIS_URL="${REDIS_URL:-redis://localhost:6379}"
export SPLATLAB_STORAGE="${SPLATLAB_STORAGE:-local}"   # set to 's3' for remote worker

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
exec python -m server.api.app --host "$HOST" --port "$PORT"
