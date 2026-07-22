"""SplatTrainingWorkflow — durable orchestration of one pipeline *phase*.

Split into two phases (selected by TrainParams.phase), one workflow run each,
sharing the per-project id so they run sequentially and the id is reusable:

  "sfm"    run COLMAP, publish the reviewable sparse point cloud, then COMPLETE.
           The GPU box is released while the user reviews the reconstruction —
           the pool only counts *open* workflows as demand, so review time costs
           no GPU. Poses-ready is durably recorded by the points.ply artifact.

  "train"  assume poses are ready (a prior sfm run uploaded colmap/undistorted)
           and run training only, re-materializing that dataset on whatever box
           picks the job up.

Thin and deterministic: it only sequences activities, tracks coarse status/stage
for the query API, and reacts to a `stop` signal. All filesystem / CUDA /
subprocess / Redis side effects live in the activities. Config is passed in as
data (never read from disk here) so replay stays deterministic.
"""
from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError
from temporalio.exceptions import CancelledError as TemporalCancelledError
from temporalio.workflow import ActivityCancellationType

with workflow.unsafe.imports_passed_through():
    from .. import config
    from ..shared.constants import UNDISTORTED_REL
    from ..activities.colmap import run_colmap_activity
    from ..activities.train import train_activity
    from .types import ColmapArgs, TrainArgs, TrainParams

# Expensive, side-effectful activities: no auto-retry (a transient blip must not
# silently burn hours of GPU / stack OOM). Failures surface to the user instead.
_NO_RETRY = RetryPolicy(maximum_attempts=1)

# The workflow runs on the always-on control plane; the GPU work is dispatched to
# the pool-managed GPU queue, where an ephemeral Vast.ai box picks it up. Reading
# the queue name at import time is deterministic (it's fixed config, not I/O), so
# it's safe inside the workflow.
_GPU_QUEUE = config.GPU_TASK_QUEUE

# A GPU box may not exist the instant a run is queued — the autoscaler needs a
# reconcile cycle plus cold-boot time to bring one online. Give the activity a
# generous schedule-to-start window so it waits in the queue for a box instead of
# failing; start-to-close still bounds the actual run once a box picks it up.
_SCHEDULE_TO_START = timedelta(minutes=30)


@workflow.defn
class SplatTrainingWorkflow:
    def __init__(self) -> None:
        self._status = "queued"
        self._stage = {"stage": "queued", "detail": ""}
        self._config: dict = {}
        self._phase = "sfm"
        self._sfm: dict = {}          # {points_uri, num_points, num_images} once ready
        self._task: asyncio.Task | None = None

    @workflow.run
    async def run(self, p: TrainParams) -> dict:
        self._config = p.config
        self._phase = p.phase
        try:
            if p.phase == "sfm":
                return await self._run_sfm(p)
            return await self._run_train(p)
        except (asyncio.CancelledError, TemporalCancelledError):
            # workflow-level cancel, or the stop signal cancelling the handle
            return self._stopped()
        except ActivityError as e:
            # A cancelled activity surfaces as ActivityError(cause=CancelledError):
            # that's a user stop, not a failure. Anything else is a real error.
            if isinstance(e.cause, TemporalCancelledError):
                return self._stopped()
            self._status = "error"
            self._stage = {"stage": "error", "detail": str(e)}
            raise

    async def _run_sfm(self, p: TrainParams) -> dict:
        self._status = "processing_poses"
        self._stage = {"stage": "features", "detail": "starting COLMAP"}
        self._task = workflow.start_activity(
            run_colmap_activity,
            ColmapArgs(p.project_id, p.config.get("colmap_matcher", "auto"), p.run_id),
            task_queue=_GPU_QUEUE,
            schedule_to_start_timeout=_SCHEDULE_TO_START,
            start_to_close_timeout=timedelta(hours=3),
            heartbeat_timeout=timedelta(minutes=5),
            cancellation_type=ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
            retry_policy=_NO_RETRY,
        )
        colmap = await self._task

        # Done: record poses-ready state and complete, releasing the GPU box.
        self._sfm = {"points_uri": colmap.points_uri,
                     "num_points": colmap.num_points,
                     "num_images": colmap.num_images}
        self._status = "poses_ready"
        self._stage = {"stage": "review",
                       "detail": f"{colmap.num_points} points from "
                                 f"{colmap.num_images} images — review, then train"}
        return {"status": "poses_ready", **self._sfm}

    async def _run_train(self, p: TrainParams) -> dict:
        self._status = "training"
        self._stage = {"stage": "init", "detail": "initializing"}
        self._task = workflow.start_activity(
            train_activity,
            # Poses were produced by a prior sfm run and uploaded to this fixed
            # project-relative path; the activity re-materializes it locally.
            TrainArgs(p.project_id, p.config, UNDISTORTED_REL, p.run_id),
            task_queue=_GPU_QUEUE,
            schedule_to_start_timeout=_SCHEDULE_TO_START,
            start_to_close_timeout=timedelta(hours=12),
            heartbeat_timeout=timedelta(seconds=60),
            cancellation_type=ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
            retry_policy=_NO_RETRY,
        )
        result = await self._task

        self._status = "done"
        self._stage = {"stage": "done", "detail": ""}
        return {"status": "done", "final_step": result.final_step,
                "snapshot_uri": result.snapshot_uri}

    def _stopped(self) -> dict:
        # The activity flushed a final snapshot and emitted `done` over Redis
        # before returning; here we just record the coarse durable status.
        self._status = "stopped"
        self._stage = {"stage": "stopped", "detail": "stopped by user"}
        return {"status": "stopped"}

    @workflow.signal
    def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()

    @workflow.query
    def state(self) -> dict:
        return {"status": self._status, "stage": self._stage,
                "config": self._config, "phase": self._phase, "sfm": self._sfm}
