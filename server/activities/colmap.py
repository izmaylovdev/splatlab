"""Activity: photos -> camera poses (COLMAP), wrapping shared.colmap_runner.

Idempotent: the undistorted model is fingerprinted against the photo set, so a
re-run with unchanged photos short-circuits (this also survives a worker crash
after COLMAP but before training — Temporal won't re-run a completed activity,
but if it ever does, this makes it cheap and safe). Cancellation is coarse:
COLMAP runs as a subprocess that can't be interrupted mid-stage, so a stop takes
effect at the next stage boundary / on completion.
"""
from __future__ import annotations

import os
import threading

from temporalio import activity

from ..shared.storage import get_storage
from ..workflows.types import ColmapArgs, ColmapResult
from .emit import make_redis_emitter
from .runner import run_with_heartbeat

FINGERPRINT = ".photos_fingerprint"


def _photos_fingerprint(photos_dir: str) -> str:
    parts = []
    for name in sorted(os.listdir(photos_dir)):
        p = os.path.join(photos_dir, name)
        if os.path.isfile(p):
            parts.append(f"{name}:{os.path.getsize(p)}")
    return str(hash("|".join(parts)))


def _undistorted_valid(undist: str) -> bool:
    model = os.path.join(undist, "sparse")
    if os.path.isdir(os.path.join(model, "0")):
        model = os.path.join(model, "0")
    return (os.path.exists(os.path.join(model, "cameras.bin"))
            or os.path.exists(os.path.join(model, "cameras.txt")))


@activity.defn
async def run_colmap_activity(args: ColmapArgs) -> ColmapResult:
    from ..shared.colmap_runner import run_colmap

    storage = get_storage()
    pid = args.project_id
    emit = make_redis_emitter(pid, args.run_id)
    stop = threading.Event()

    def blocking() -> str:
        photos = storage.ensure_photos_local(pid)          # download from S3 if remote
        work = os.path.join(storage.project_dir(pid), "colmap")
        undist = os.path.join(work, "undistorted")
        fp = _photos_fingerprint(photos)
        fp_path = os.path.join(work, FINGERPRINT)

        if _undistorted_valid(undist) and _read(fp_path) == fp:
            emit({"type": "status", "stage": "features",
                  "detail": "reusing cached camera poses"})
            emit({"type": "status", "stage": "done", "detail": "poses ready (cached)"})
            return undist

        result = run_colmap(
            photos, work,
            progress=lambda s, d: emit({"type": "status", "stage": s, "detail": d}),
            matcher=args.matcher,
        )
        os.makedirs(work, exist_ok=True)
        _write(fp_path, fp)
        return result

    # wait_for_flush=False: can't cleanly interrupt COLMAP's subprocess.
    try:
        undist = await run_with_heartbeat(blocking, stop, interval=30.0, wait_for_flush=False)
    except Exception as e:  # noqa: BLE001  (CancelledError is BaseException — passes through)
        emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
        raise

    # Publish the undistorted dataset so a *different* box can train from it
    # (no-op on the local backend). Return a portable project-relative handle.
    rel = os.path.relpath(undist, storage.project_dir(pid)).replace(os.sep, "/")
    storage.upload_dir(pid, rel)
    return ColmapResult(undistorted_rel=rel)


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _write(path: str, val: str) -> None:
    with open(path, "w") as f:
        f.write(val)
