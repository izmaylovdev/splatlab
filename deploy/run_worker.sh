#!/usr/bin/env bash
# Launch the SplatLab Temporal worker (real GPU mode) with a sane threading env.
#
# This is the GPU box. It connects OUT to the control plane (Temporal + Redis)
# and, on the S3 backend, to MinIO/S3 — set those addresses via the env below.
#
# On very-high-core hosts, OpenBLAS/OpenMP each spawn one thread per core and
# nest that inside COLMAP's own worker pool; the oversubscription corrupts the
# heap during pycolmap matching ("BLAS: Bad memory unallocation!" -> SIGABRT).
# Pinning the BLAS/OMP pools to one thread lets COLMAP own the parallelism (see
# _colmap_threads in server/shared/colmap_runner.py). GPU training is unaffected.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Prefer a repo-local .venv (slim-image path); the prebaked image instead ships
# its venv on PATH (/opt/venv) and has no .venv here, so activation is optional.
[ -f .venv/bin/activate ] && . .venv/bin/activate

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"

# Avoid nested BLAS/OMP thread pools (see note above).
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

# Control-plane addresses (override for a remote control plane / tunnels).
export TEMPORAL_ADDRESS="${TEMPORAL_ADDRESS:-localhost:7233}"
export REDIS_URL="${REDIS_URL:-redis://localhost:6379}"
export SPLATLAB_STORAGE="${SPLATLAB_STORAGE:-local}"   # set to 's3' for remote

exec python -m server.worker.main
