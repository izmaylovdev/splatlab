#!/usr/bin/env bash
# SplatLab — vast.ai (Linux/CUDA) bootstrap.
#
# Run this ON the vast.ai instance, from the repo root, after the code is on the box
# (git clone or scp — see README / deploy notes). It creates a venv, installs the
# server + CUDA PyTorch + gsplat, verifies the GPU, and (unless --no-run) starts the
# web server bound to 0.0.0.0 so you can reach the UI from your laptop.
#
#   bash deploy/vast_setup.sh                # set up, verify, then run the server
#   bash deploy/vast_setup.sh --no-run       # set up + verify only
#   PORT=8080 bash deploy/vast_setup.sh      # override the port (default 8000)
#
# Reach the UI from your laptop either via the vast "Open Ports" mapping for $PORT,
# or an SSH tunnel:  ssh -p <SSH_PORT> -N -L 8000:localhost:8000 root@<HOST>
#
# Tunables (env vars):
#   PORT           web server port                (default 8000)
#   HOST           bind address                   (default 0.0.0.0 — needed for remote access)
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
python -m backend.check

if [ "$RUN_SERVER" = "1" ]; then
  echo
  echo "==> Starting SplatLab on http://$HOST:$PORT"
  echo "    (reach it via the vast Open-Ports mapping for $PORT, or an SSH -L tunnel)"
  echo "    Tip: run this inside 'tmux' so it survives disconnects."
  exec python -m backend.main --host "$HOST" --port "$PORT"
else
  echo
  echo "==> Setup complete. To start the server:"
  echo "    source .venv/bin/activate"
  echo "    python -m backend.main --host $HOST --port $PORT"
fi
