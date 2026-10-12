#!/usr/bin/env bash
# SplatLab SFM worker — runs COLMAP natively on the control-plane Mac.
#
# Polls the SFM queue (splat-sfm) and runs the COLMAP activity only. It runs on
# the host, not in Docker: pycolmap ships no linux-arm64 wheel, and COLMAP wants
# all of the Mac's cores and memory rather than Docker's VM slice. It talks to the
# Docker stack (Temporal, Redis, MinIO) through their published localhost ports.
#
#   bash deploy/run_sfm_worker.sh
#
# Keep it running (and the Mac awake: caffeinate -dims) while SFM jobs are
# queued — without it, SFM runs wait in the queue. One job at a time by default;
# SPLATLAB_SFM_CONCURRENCY raises that.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

. .venv/bin/activate

ENV_FILE=deploy/.env
[ -f "$ENV_FILE" ] || { echo "!! $ENV_FILE not found — cp deploy/.env.example deploy/.env" >&2; exit 1; }
# Read single values from .env without exporting the whole file.
getenv() { grep -E "^$1=" "$ENV_FILE" | tail -n1 | cut -d= -f2- ; }

# COLMAP parallelises internally; keep BLAS from oversubscribing on top of it.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

export SPLATLAB_WORKER_ROLE=sfm
export PYTHONUNBUFFERED=1   # progress reaches the log as it happens
export TEMPORAL_ADDRESS="${TEMPORAL_ADDRESS:-localhost:7233}"
export REDIS_URL="${REDIS_URL:-redis://:$(getenv REDIS_PASSWORD)@localhost:6379}"
export SPLATLAB_STORAGE=s3
export SPLATLAB_S3_ENDPOINT="${SPLATLAB_S3_ENDPOINT:-http://localhost:9000}"
export SPLATLAB_S3_PUBLIC_ENDPOINT="${SPLATLAB_S3_PUBLIC_ENDPOINT:-http://localhost:9000}"
export SPLATLAB_S3_BUCKET="${SPLATLAB_S3_BUCKET:-$(getenv SPLATLAB_S3_BUCKET)}"
export SPLATLAB_S3_BUCKET="${SPLATLAB_S3_BUCKET:-splatlab}"
export SPLATLAB_S3_REGION="${SPLATLAB_S3_REGION:-$(getenv SPLATLAB_S3_REGION)}"
export SPLATLAB_S3_REGION="${SPLATLAB_S3_REGION:-us-east-1}"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-$(getenv MINIO_ROOT_USER)}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-$(getenv MINIO_ROOT_PASSWORD)}"
# Local scratch for photos + COLMAP work dirs (results are uploaded to MinIO).
export SPLATLAB_DATA_DIR="${SPLATLAB_DATA_DIR:-$PWD/data/sfm-worker}"

exec python -m server.worker.main
