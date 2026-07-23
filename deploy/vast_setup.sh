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
# The worker connects OUT to the control plane. When the control plane is a laptop
# behind NAT (the default SplatLab flow), the box reaches it over ngrok: the Mac
# runs deploy/ngrok_up.sh and TEMPORAL_ADDRESS / REDIS_URL / SPLATLAB_S3_ENDPOINT
# point at the ngrok public addresses. The GPU pool forwards all of these
# automatically — nothing tunnel-related needs installing on the box.
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

# emit_stage <name>: best-effort boot progress to the control-plane Redis. The
# first one that lands proves the box reached the Mac over ngrok with the right
# creds — the visibility we were missing when boxes hung in "booting".
# No-op until the venv (with redis) is active; failures never abort the boot.
emit_stage() {
  [ -n "${REDIS_URL:-}" ] || return 0
  command -v python >/dev/null 2>&1 || return 0
  SPLATLAB_STAGE="$1" python - <<'PY' 2>/dev/null || true
import os, time
try:
    import redis
    r = redis.Redis.from_url(os.environ["REDIS_URL"], socket_connect_timeout=5, socket_timeout=5)
    bid = os.environ.get("SPLATLAB_BOX_ID", "box")
    key = f"splatlab:box:{bid}:boot"
    r.hset(key, mapping={"stage": os.environ["SPLATLAB_STAGE"], "ts": int(time.time())})
    r.expire(key, 3600)
    print(f"[boot] stage -> {os.environ['SPLATLAB_STAGE']}")
except Exception:
    pass
PY
}

echo "==> SplatLab vast.ai bootstrap  (torch $TORCH_VERSION+$CUDA_TAG, port $PORT)"

# ---- Prebaked-image fast path ----------------------------------------------
# The prebaked box image (deploy/Dockerfile.box) ships a venv at /opt/venv with
# torch + gsplat + pycolmap + the server requirements already installed, and sets
# SPLATLAB_PREBAKED=1. Skip venv creation and ALL installs — just use that venv
# and start the worker. This turns a 30-45min cold boot into ~1min after pull.
if [ "${SPLATLAB_PREBAKED:-0}" = "1" ]; then
  export PATH="/opt/venv/bin:$PATH"
  if python -c 'import torch, gsplat, pycolmap' >/dev/null 2>&1; then
    echo "==> Prebaked image — deps already installed, skipping setup"
    command -v nvidia-smi >/dev/null 2>&1 && \
      nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
    emit_stage deps-installed
    emit_stage torch-installed
    emit_stage gsplat-installed
    export TEMPORAL_ADDRESS="${TEMPORAL_ADDRESS:-localhost:7233}"
    export REDIS_URL="${REDIS_URL:-redis://localhost:6379}"
    export SPLATLAB_STORAGE="${SPLATLAB_STORAGE:-local}"
    if [ "$RUN_SERVER" = "1" ]; then
      echo "==> Starting SplatLab worker (Temporal=$TEMPORAL_ADDRESS, storage=$SPLATLAB_STORAGE)"
      emit_stage worker-starting
      exec bash deploy/run_worker.sh
    fi
    echo "==> Setup complete (prebaked, --no-run)."
    exit 0
  fi
  echo "!! SPLATLAB_PREBAKED=1 but torch/gsplat/pycolmap not importable — full setup" >&2
fi

# ---- Base tools -------------------------------------------------------------
# The slim CUDA runtime base has no python/git/curl. Install them (plus the
# system libs pycolmap/opencv need at runtime) before anything else. On a fat
# image that already has them this is a fast no-op.
#
# NOTE: some images ship the `python3` binary but NOT the venv module (ensurepip),
# so checking for the command alone isn't enough — `python3 -m venv` then fails
# with "ensurepip is not available". Probe ensurepip explicitly and install when
# any piece is missing.
if ! command -v python3 >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1 \
   || ! command -v curl >/dev/null 2>&1 \
   || ! python3 -c 'import ensurepip' >/dev/null 2>&1; then
  echo "==> Installing base tools (python3 + venv, git, curl, runtime libs)"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git curl ca-certificates \
    libgl1 libgomp1 libglib2.0-0
fi

# Connectivity: the box reaches the (NAT'd, laptop-local) control plane over ngrok.
# There is nothing to set up on the box — TEMPORAL_ADDRESS / REDIS_URL /
# SPLATLAB_S3_ENDPOINT already point at the ngrok public addresses (forwarded by
# the pool), and the worker just dials them OUT. The first emit_stage below is the
# proof the box reached the Mac.

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
# First Redis contact: if this lands, ngrok + creds + reachability all work.
emit_stage deps-installed

echo "==> Installing CUDA PyTorch ($TORCH_VERSION / $TV_VERSION, $CUDA_TAG)"
pip install "torch==$TORCH_VERSION" "torchvision==$TV_VERSION" \
  --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
emit_stage torch-installed

if [ "$GSPLAT_WHEEL" = "1" ]; then
  # Prebuilt wheel — skips the on-import CUDA compile. Index is torch/cuda specific;
  # the pt24cu121 tag matches the default torch 2.4.x + cu121 above. That index has
  # ONLY gsplat, so install gsplat's deps from PyPI first, then the wheel --no-deps.
  echo "==> Installing gsplat from prebuilt wheel index"
  # gsplat lazily imports some deps (e.g. `packaging`) only at render time, so list
  # them explicitly — --no-deps + the gsplat-only index won't pull them otherwise.
  pip install ninja jaxtyping rich packaging typing_extensions
  pip install gsplat --no-deps --index-url "https://docs.gsplat.studio/whl/pt24${CUDA_TAG}"
else
  # Compile against the image's CUDA toolkit (works out of the box on vast CUDA images).
  echo "==> Installing gsplat (compiles CUDA kernels on first import)"
  pip install gsplat
fi

emit_stage gsplat-installed

echo "==> Environment check"
python -m server.shared.check

export TEMPORAL_ADDRESS="${TEMPORAL_ADDRESS:-localhost:7233}"
export REDIS_URL="${REDIS_URL:-redis://localhost:6379}"
export SPLATLAB_STORAGE="${SPLATLAB_STORAGE:-local}"

if [ "$RUN_SERVER" = "1" ]; then
  echo
  echo "==> Starting SplatLab worker (Temporal=$TEMPORAL_ADDRESS, storage=$SPLATLAB_STORAGE)"
  echo "    Tip: run this inside 'tmux' so it survives disconnects."
  emit_stage worker-starting
  exec bash deploy/run_worker.sh
else
  echo
  echo "==> Setup complete. To start the worker:"
  echo "    source .venv/bin/activate"
  echo "    bash deploy/run_worker.sh"
fi
