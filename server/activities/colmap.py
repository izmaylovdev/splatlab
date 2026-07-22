"""Activity: photos -> camera poses (COLMAP), wrapping shared.colmap_runner.

Idempotent: the undistorted model is fingerprinted against the photo set, so a
re-run with unchanged photos short-circuits (this also survives a worker crash
after COLMAP but before training — Temporal won't re-run a completed activity,
but if it ever does, this makes it cheap and safe). Cancellation is coarse:
COLMAP runs as a subprocess that can't be interrupted mid-stage, so a stop takes
effect at the next stage boundary / on completion.

On success it also publishes a *reviewable* sparse point cloud: the SfM points
exported as a splat ``.ply`` (colmap/points.ply) plus a tiny sfm.json sidecar,
and emits an ``sfm`` event carrying the point-cloud URL + counts. The two-phase
workflow's SFM run ends here, letting the user inspect the reconstruction before
committing a GPU to training.
"""
from __future__ import annotations

import json
import os
import threading

from temporalio import activity

from ..shared.constants import SFM_META_REL, SFM_POINTS_REL
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


def _sparse_model_dir(undist: str) -> str:
    """The PINHOLE sparse model dir inside an undistorted dataset (sparse or sparse/0)."""
    model = os.path.join(undist, "sparse")
    if os.path.isdir(os.path.join(model, "0")):
        model = os.path.join(model, "0")
    return model


def _undistorted_valid(undist: str) -> bool:
    model = _sparse_model_dir(undist)
    return (os.path.exists(os.path.join(model, "cameras.bin"))
            or os.path.exists(os.path.join(model, "cameras.txt")))


def _export_points(undist: str, out_ply: str) -> tuple[int, int]:
    """Read the undistorted sparse model and write the reviewable point-cloud
    .ply. Returns (num_points, num_registered_images)."""
    from ..shared.colmap_io import read_model
    from ..shared.splat_export import export_points_ply

    _cameras, images, (xyz, rgb) = read_model(_sparse_model_dir(undist))
    os.makedirs(os.path.dirname(out_ply), exist_ok=True)
    export_points_ply(out_ply, xyz, rgb)
    return int(len(xyz)), int(len(images))


@activity.defn
async def run_colmap_activity(args: ColmapArgs) -> ColmapResult:
    from ..shared.colmap_runner import run_colmap

    storage = get_storage()
    pid = args.project_id
    emit = make_redis_emitter(pid, args.run_id)
    stop = threading.Event()

    def blocking() -> tuple[str, int, int]:
        photos = storage.ensure_photos_local(pid)          # download from S3 if remote
        work = os.path.join(storage.project_dir(pid), "colmap")
        undist = os.path.join(work, "undistorted")
        fp = _photos_fingerprint(photos)
        fp_path = os.path.join(work, FINGERPRINT)

        if _undistorted_valid(undist) and _read(fp_path) == fp:
            emit({"type": "status", "stage": "features",
                  "detail": "reusing cached camera poses"})
        else:
            run_colmap(
                photos, work,
                progress=lambda s, d: emit({"type": "status", "stage": s, "detail": d}),
                matcher=args.matcher,
            )
            os.makedirs(work, exist_ok=True)
            _write(fp_path, fp)

        # Publish the reviewable sparse cloud (fresh or cached path alike).
        emit({"type": "status", "stage": "review", "detail": "exporting sparse points"})
        n_points, n_images = _export_points(undist, os.path.join(work, "points.ply"))
        with open(os.path.join(work, "sfm.json"), "w") as f:
            json.dump({"num_points": n_points, "num_images": n_images}, f)
        return undist, n_points, n_images

    # wait_for_flush=False: can't cleanly interrupt COLMAP's subprocess.
    try:
        undist, n_points, n_images = await run_with_heartbeat(
            blocking, stop, interval=30.0, wait_for_flush=False)
    except Exception as e:  # noqa: BLE001  (CancelledError is BaseException — passes through)
        emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
        raise

    # Publish the undistorted dataset + the reviewable point cloud so a *different*
    # box can train from it and the browser can review it (no-op on local backend).
    rel = os.path.relpath(undist, storage.project_dir(pid)).replace(os.sep, "/")
    storage.upload_dir(pid, rel)
    storage.upload_artifact(pid, SFM_POINTS_REL)
    storage.upload_artifact(pid, SFM_META_REL)

    points_uri = storage.public_url(pid, SFM_POINTS_REL)
    emit({"type": "sfm", "url": points_uri,
          "num_points": n_points, "num_images": n_images})
    return ColmapResult(undistorted_rel=rel, points_rel=SFM_POINTS_REL,
                        points_uri=points_uri, num_points=n_points, num_images=n_images)


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _write(path: str, val: str) -> None:
    with open(path, "w") as f:
        f.write(val)
