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
TASK_QUEUE = _env("SPLATLAB_TASK_QUEUE", "splat-gpu")

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
