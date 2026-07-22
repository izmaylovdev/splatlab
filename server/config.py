"""Central configuration, read from the environment.

Every process (API, worker) imports this. Nothing here does I/O — it only
resolves settings so the rest of the code can stay declarative.
"""
from __future__ import annotations

import os

# Repo root = parent of the `server/` package.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None and v != "" else default


# ---- Temporal ---------------------------------------------------------------
TEMPORAL_ADDRESS = _env("TEMPORAL_ADDRESS", "localhost:7233")
TEMPORAL_NAMESPACE = _env("TEMPORAL_NAMESPACE", "default")
# GPU_TASK_QUEUE: the queue the (ephemeral, pool-rented) GPU boxes poll — COLMAP
# and training activities run here. CONTROL_TASK_QUEUE: the always-on control
# plane that runs the workflow itself (and thus survives every GPU box being
# reaped). TASK_QUEUE stays as an alias of GPU_TASK_QUEUE for backwards compat
# (SPLATLAB_TASK_QUEUE still points a worker at the GPU queue).
GPU_TASK_QUEUE = _env("SPLATLAB_TASK_QUEUE", "splat-gpu")
CONTROL_TASK_QUEUE = _env("SPLATLAB_CONTROL_QUEUE", "splat-control")
TASK_QUEUE = GPU_TASK_QUEUE  # legacy alias

# ---- Redis (live telemetry bus) --------------------------------------------
REDIS_URL = _env("REDIS_URL", "redis://localhost:6379")
# How many progress events to retain per project, and how many to replay on
# WebSocket connect (matches the old in-memory ProjectManager behaviour).
HISTORY_MAXLEN = int(_env("SPLATLAB_HISTORY_MAXLEN", "3000"))
HISTORY_HELLO = int(_env("SPLATLAB_HISTORY_HELLO", "1500"))
# Expire a project's live keys this many seconds after it goes idle. The
# durable copy of snapshots lives in storage; progress history is ephemeral.
HISTORY_TTL_SECONDS = int(_env("SPLATLAB_HISTORY_TTL", str(6 * 3600)))

# ---- Storage ----------------------------------------------------------------
# "local" — API and worker share a filesystem (dev / co-located).
# "s3"    — object storage (MinIO/S3) bridges a remote GPU worker and the API.
STORAGE_BACKEND = _env("SPLATLAB_STORAGE", "local").lower()
DATA_DIR = _env("SPLATLAB_DATA_DIR", os.path.join(ROOT, "data"))

# S3 / MinIO (only used when STORAGE_BACKEND == "s3")
S3_BUCKET = _env("SPLATLAB_S3_BUCKET", "splatlab")
S3_ENDPOINT_URL = os.environ.get("SPLATLAB_S3_ENDPOINT") or None
# Endpoint the *browser* can reach for presigned GET URLs. Falls back to the
# internal endpoint when they're the same host.
S3_PUBLIC_ENDPOINT = os.environ.get("SPLATLAB_S3_PUBLIC_ENDPOINT") or S3_ENDPOINT_URL
S3_REGION = _env("SPLATLAB_S3_REGION", "us-east-1")
S3_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("SPLATLAB_S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("SPLATLAB_S3_SECRET_KEY")
S3_PRESIGN_TTL = int(_env("SPLATLAB_S3_PRESIGN_TTL", "3600"))

WORKFLOW_ID_PREFIX = "train-"


def workflow_id(project_id: str) -> str:
    return f"{WORKFLOW_ID_PREFIX}{project_id}"


# ---- Vast.ai GPU pool (autoscaler) -----------------------------------------
# The pool (server.vast.pool) rents/reaps GPU boxes on demand so the control
# plane never has to keep an idle GPU running. All settings are optional; the
# pool refuses to start without VAST_API_KEY.
VAST_API_KEY = os.environ.get("VAST_API_KEY") or None
VAST_API_BASE = _env("VAST_API_BASE", "https://console.vast.ai/api/v0")

# Offer search filters. VAST_GPU_NAME is matched against the API's raw gpu_name,
# which is SPACED (e.g. "RTX 5090", "RTX 4090" — not "RTX_5090"); empty = any GPU
# meeting the other bars.
VAST_GPU_NAME = os.environ.get("VAST_GPU_NAME") or ""
VAST_MIN_GPU_RAM_GB = float(_env("VAST_MIN_GPU_RAM_GB", "16"))
VAST_MIN_RELIABILITY = float(_env("VAST_MIN_RELIABILITY", "0.98"))
VAST_MAX_PRICE = float(_env("VAST_MAX_PRICE", "0.60"))   # $/hr ceiling per box
VAST_NUM_GPUS = int(_env("VAST_NUM_GPUS", "1"))
VAST_DISK_GB = int(_env("VAST_DISK_GB", "40"))
# Docker image the box boots into. Must have CUDA + git; vast_setup.sh installs
# the rest on first boot. Override with a prebaked image to cut cold start.
VAST_IMAGE = _env("VAST_IMAGE", "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime")
# Repo the onstart script clones onto a fresh box (branch optional).
VAST_REPO_URL = _env("VAST_REPO_URL", "https://github.com/izamylovdev/splatlab.git")
VAST_REPO_BRANCH = _env("VAST_REPO_BRANCH", "main")
# Label stamped on every instance we create — the pool only ever touches boxes
# carrying this label, so it can never destroy someone else's instance.
VAST_LABEL = _env("VAST_LABEL", "splatlab-pool")

# Scaling knobs.
POOL_MAX_BOXES = int(_env("SPLATLAB_POOL_MAX_BOXES", "3"))
POOL_JOBS_PER_BOX = int(_env("SPLATLAB_POOL_JOBS_PER_BOX", "2"))  # match worker MAX_ACTIVITIES
POOL_IDLE_TIMEOUT = int(_env("SPLATLAB_POOL_IDLE_TIMEOUT", "600"))     # reap an idle box after Ns
POOL_MAX_LIFETIME = int(_env("SPLATLAB_POOL_MAX_LIFETIME", str(6 * 3600)))  # hard cap per box
POOL_RECONCILE_INTERVAL = int(_env("SPLATLAB_POOL_INTERVAL", "30"))    # reconcile loop period
# A box whose registry heartbeat is older than this is treated as dead and reaped
# even if it never reported idle (covers a crashed worker / hung box).
POOL_HEARTBEAT_DEADLINE = int(_env("SPLATLAB_POOL_HEARTBEAT_DEADLINE", "120"))
# How long to allow a freshly-rented box to boot + register before we treat it as
# a failed launch and destroy it.
POOL_BOOT_DEADLINE = int(_env("SPLATLAB_POOL_BOOT_DEADLINE", str(20 * 60)))
# Kill switch: when set truthy, the pool drains the whole fleet to zero and rents
# nothing (for emergencies / cost freezes).
POOL_PAUSED = _env("SPLATLAB_POOL_PAUSED", "").lower() in ("1", "true", "yes")
