# SplatLab control-plane image — API, control worker, and GPU pool.
#
# One image, three roles (selected by the compose `command`). It deliberately
# carries NO torch / gsplat / CUDA: the control plane never runs training — that
# lives on the (GPU) worker, which is provisioned separately (Vast.ai box). So
# this image stays small and builds on any arch, including the dev Mac.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps: pycolmap wheels + imageio-ffmpeg's bundled binary need libGL/glib
# at runtime (the API's /video frame extraction imports imageio-ffmpeg).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for layer caching. The control-plane requirements
# are a CPU-only subset (no torch / gsplat / pycolmap — those are GPU-box only).
COPY deploy/requirements-controlplane.txt ./requirements-controlplane.txt
RUN pip install --upgrade pip && pip install -r requirements-controlplane.txt

# App code. The API also serves the static client if client/ is present.
COPY server ./server
COPY client ./client

# Drop privileges. Give the app a home + a writable scratch dir (S3 backend keeps
# a local mirror under SPLATLAB_DATA_DIR).
RUN useradd --create-home --uid 10001 splat \
    && mkdir -p /app/data \
    && chown -R splat:splat /app
USER splat

EXPOSE 8000

# Default role: the API service. Overridden per-service in docker-compose.
CMD ["python", "-m", "server.api.app", "--host", "0.0.0.0", "--port", "8000"]
