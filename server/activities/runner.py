"""Bridge synchronous, long-running CPU/GPU work into an async Temporal activity.

`shared.trainer.train` and `shared.colmap_runner.run_colmap` are blocking and
carry no heartbeat logic of their own. Running them directly in an async activity
would starve the event loop, so no `activity.heartbeat()` would fire and the
heartbeat-timeout would kill a perfectly healthy multi-hour job.

`run_with_heartbeat` runs the blocking function in a thread executor and beats on
a fixed wall-clock cadence (never tied to training steps, which can stall for
seconds during densification). On cancellation it sets the caller's
`threading.Event` — bridging Temporal cancel -> the existing `stop.is_set()`
checks inside the trainer.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Callable, TypeVar

from temporalio import activity

T = TypeVar("T")


async def run_with_heartbeat(
    blocking: Callable[[], T],
    stop: threading.Event,
    *,
    interval: float = 10.0,
    wait_for_flush: bool = True,
) -> T:
    """Run `blocking` off-loop, heartbeating every `interval` seconds.

    On cancellation the `stop` event is set. If `wait_for_flush` is True we then
    wait for the blocking call to notice the stop, flush its final snapshot and
    finish (used for training, which cooperatively checks `stop` every step)
    before re-raising. If False we re-raise immediately and abandon the thread —
    used for COLMAP, whose subprocess can't be interrupted mid-stage.

    Either way the CancelledError is re-raised so Temporal records the activity
    as cancelled and the workflow can report the run as `stopped` (rather than
    silently `done`). The final snapshot has already gone out over Redis.
    """
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(None, blocking)
    try:
        while True:
            done, _ = await asyncio.wait({fut}, timeout=interval)
            if done:
                return fut.result()
            activity.heartbeat()
    except asyncio.CancelledError:
        stop.set()
        if wait_for_flush:
            try:
                await asyncio.shield(fut)   # let training flush its final .ply
            except Exception:
                pass
        raise
