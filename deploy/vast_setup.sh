#!/usr/bin/env bash
# SplatLab — vast.ai (Linux/CUDA) worker bootstrap.
#
# Run this ON the vast.ai GPU instance, from the repo root, after the code is on
# the box (git clone or scp). It creates a venv, installs the server deps + CUDA
# PyTorch + gsplat, verifies the GPU, and (unless --no-run) starts the Temporal
# WORKER, which connects out to the control plane (Temporal + Redis, and MinIO/S3
# on the s3 backend).
#
#   bash deploy/vast_setup.sh                # set up, verify, then run the worker
#   bash deploy/vast_setup.sh --no-run       # set up + verify only
#
# The worker connects OUT — point it at your control plane and reach those from
# the box directly or via SSH reverse tunnels, e.g.:
#   ssh -p <SSH_PORT> -N -R 7233:localhost:7233 -R 6379:localhost:6379 root@<HOST>
# then run with TEMPORAL_ADDRESS=localhost:7233 REDIS_URL=redis://localhost:6379.
#
# Tunables (env vars):
#   TEMPORAL_ADDRESS  control-plane Temporal gRPC   (default localhost:7233)
#   REDIS_URL         control-plane Redis           (default redis://localhost:6379)
#   SPLATLAB_STORAGE  local | s3                    (default local)
#   PY             python interpreter             (default python3)
#   TORCH_VERSION  torch version                  (default 2.4.1)
#   TV_VERSION     torchvision version            (default 0.19.1)
#   CUDA_TAG       torch/gsplat CUDA wheel tag     (default cu121)
#   GSPLAT_WHEEL   1 = install gsplat from the prebuilt wheel index instead of compiling
set -euo pipefail

RUN_SERVER=1
for arg in "$@"; do
  case "$arg" in
    --no-run) RUN_SERVER=0 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
PY="${PY:-python3}"
TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
TV_VERSION="${TV_VERSION:-0.19.1}"
CUDA_TAG="${CUDA_TAG:-cu121}"
GSPLAT_WHEEL="${GSPLAT_WHEEL:-0}"

# Move to the repo root (parent of this script's deploy/ dir) so relative paths work.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "==> SplatLab vast.ai bootstrap  (torch $TORCH_VERSION+$CUDA_TAG, port $PORT)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "!! nvidia-smi not found — this doesn't look like a GPU instance." >&2
  echo "   Real training needs an NVIDIA GPU. Continuing anyway (setup only)." >&2
else
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi

echo "==> Creating venv (.venv)"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

echo "==> Installing server requirements"
pip install -r requirements.txt

echo "==> Installing CUDA PyTorch ($TORCH_VERSION / $TV_VERSION, $CUDA_TAG)"
pip install "torch==$TORCH_VERSION" "torchvision==$TV_VERSION" \
  --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

if [ "$GSPLAT_WHEEL" = "1" ]; then
  # Prebuilt wheel — skips the on-import CUDA compile. Index is torch/cuda specific;
  # the pt24cu121 tag matches the default torch 2.4.x + cu121 above.
  echo "==> Installing gsplat from prebuilt wheel index"
  pip install gsplat --index-url "https://docs.gsplat.studio/whl/pt24${CUDA_TAG}"
else
  # Compile against the image's CUDA toolkit (works out of the box on vast CUDA images).
  echo "==> Installing gsplat (compiles CUDA kernels on first import)"
  pip install gsplat
fi

echo "==> Environment check"
python -m server.shared.check

export TEMPORAL_ADDRESS="${TEMPORAL_ADDRESS:-localhost:7233}"
export REDIS_URL="${REDIS_URL:-redis://localhost:6379}"
export SPLATLAB_STORAGE="${SPLATLAB_STORAGE:-local}"

if [ "$RUN_SERVER" = "1" ]; then
  echo
  echo "==> Starting SplatLab worker (Temporal=$TEMPORAL_ADDRESS, storage=$SPLATLAB_STORAGE)"
  echo "    Tip: run this inside 'tmux' so it survives disconnects."
  exec bash deploy/run_worker.sh
else
  echo
  echo "==> Setup complete. To start the worker:"
  echo "    source .venv/bin/activate"
  echo "    bash deploy/run_worker.sh"
fi
