#!/usr/bin/env bash
# SplatLab — vast.ai (Linux/CUDA) worker bootstrap.
#
# Run this ON the vast.ai GPU instance, from the repo root, after the code is on
# the box (git clone or scp). It joins the Tailscale tailnet (if TS_AUTHKEY is
# set), creates a venv, installs the server deps + CUDA PyTorch + gsplat, verifies
# the GPU, and (unless --no-run) starts the Temporal WORKER, which connects out to
# the control plane (Temporal + Redis, and MinIO/S3 on the s3 backend).
#
#   bash deploy/vast_setup.sh                # set up, verify, then run the worker
#   bash deploy/vast_setup.sh --no-run       # set up + verify only
#
# The worker connects OUT to the control plane. When the control plane is a laptop
# behind NAT (the default SplatLab flow), the box reaches it over Tailscale: set
# TS_AUTHKEY and point TEMPORAL_ADDRESS / REDIS_URL / SPLATLAB_S3_ENDPOINT at the
# Mac's tailnet address. The GPU pool forwards all of these automatically.
#
# Tunables (env vars):
#   TS_AUTHKEY        Tailscale auth key — joins the tailnet (unset = skip)
#   TS_EXTRA_ARGS     extra args passed to `tailscale up`
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

# ---- Tailscale --------------------------------------------------------------
# Join the tailnet so the worker can dial the (NAT'd, laptop-local) control plane
# at its stable tailnet address. Skipped when TS_AUTHKEY is unset (e.g. a box on
# the same network as the control plane, or a public-host deploy).
if [ -n "${TS_AUTHKEY:-}" ]; then
  echo "==> Joining Tailscale tailnet"
  if ! command -v tailscale >/dev/null 2>&1; then
    curl -fsSL https://tailscale.com/install.sh | sh
  fi
  # Vast instances are containers without systemd; run tailscaled ourselves.
  # The runtime + state dirs don't exist on a fresh box — tailscaled won't create
  # its socket's parent, so make them first (this was the boot-hang bug).
  mkdir -p /var/run/tailscale /var/lib/tailscale /dev/net
  # Ensure the TUN device exists (needs CAP_MKNOD, present on Vast instances).
  [ -c /dev/net/tun ] || mknod /dev/net/tun c 10 200 || true
  if ! pgrep -x tailscaled >/dev/null 2>&1; then
    tailscaled --state=/var/lib/tailscale/tailscaled.state \
               --socket=/var/run/tailscale/tailscaled.sock >/var/log/tailscaled.log 2>&1 &
  fi
  # Wait for the daemon socket to come up (fixed sleeps race on a cold box).
  for _ in $(seq 1 30); do [ -S /var/run/tailscale/tailscaled.sock ] && break; sleep 1; done
  # Don't let a tailnet failure abort the whole bootstrap silently — surface it.
  if ! tailscale up --authkey="$TS_AUTHKEY" --hostname="splatlab-${SPLATLAB_BOX_ID:-box}" \
                    --accept-routes ${TS_EXTRA_ARGS:-}; then
    echo "!! tailscale up failed — tailscaled log:" >&2
    tail -n 30 /var/log/tailscaled.log >&2 || true
    exit 1
  fi
  echo "    tailnet IP: $(tailscale ip -4 2>/dev/null || echo '?')"
fi

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
