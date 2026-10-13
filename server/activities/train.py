"""Activity: undistorted dataset -> trained 3DGS, wrapping shared.trainer.train.

Reopens the dataset from the URI produced by run_colmap_activity (never receives
the bytes). Live progress/frames/snapshots stream out through the Redis emitter;
this activity's return value is only coarse bookkeeping for the workflow.

Cancellation is cooperative and prompt: setting the `stop` event makes the
trainer break its step loop, write a final snapshot, and emit `done`, so a /stop
keeps the partially-trained result (matches the old UX).

A fresh run first clears the project's snapshots; a resume run instead pulls the
latest one down and continues training from its step.
"""
from __future__ import annotations

import threading

from temporalio import activity

from ..shared.storage import get_storage, snapshot_step
from ..workflows.types import TrainArgs, TrainResult
from .emit import make_redis_emitter
from .runner import run_with_heartbeat


@activity.defn
async def train_activity(args: TrainArgs) -> TrainResult:
    from ..shared.dataset import load_colmap_dataset
    from ..shared.train_common import TrainConfig
    from ..shared.trainer import train as real_train

    storage = get_storage()
    pid = args.project_id
    cfg = TrainConfig.from_dict(args.config)
    emit = make_redis_emitter(pid, args.run_id)
    stop = threading.Event()

    def blocking() -> None:
        # On the pool, COLMAP may have run on another box — pull its uploaded
        # undistorted dataset down locally first (no-op on the local backend).
        undist = storage.ensure_dir_local(pid, args.undistorted_rel)
        ds = load_colmap_dataset(undist, cfg.max_image_side)
        resume_ply, resume_step = None, 0
        resume = bool(args.config.get("resume"))
        snaps = storage.list_snapshots(pid) if resume else []
        if snaps:
            resume_ply = storage.ensure_file_local(pid, snaps[-1])
            resume_step = snapshot_step(snaps[-1])
        elif not resume:
            storage.clear_snapshots(pid)
        real_train(ds, storage.project_dir(pid), cfg, emit, stop,
                   resume_ply=resume_ply, resume_step=resume_step,
                   resume_prev_max=args.config.get("resume_prev_max"))

    try:
        await run_with_heartbeat(blocking, stop, interval=10.0, wait_for_flush=True)
    except Exception as e:  # noqa: BLE001  (CancelledError is BaseException — passes through)
        emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
        raise

    snaps = storage.list_snapshots(pid)
    if not snaps:
        return TrainResult(final_step=0, snapshot_uri=None)
    return TrainResult(final_step=snapshot_step(snaps[-1]),
                       snapshot_uri=storage.public_url(pid, snaps[-1]))
