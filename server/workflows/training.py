"""SplatTrainingWorkflow — durable orchestration of one training run.

Thin and deterministic: it only sequences activities, tracks coarse
status/stage for the query API, and reacts to a `stop` signal. All filesystem /
CUDA / subprocess / Redis side effects live in the activities. Config is passed
in as data (never read from disk here) so replay stays deterministic.

Durability: because COLMAP and training are *separate* activities, a worker
crash after COLMAP resumes straight into training on a fresh worker — Temporal
does not re-run a completed activity. This replaces the old behaviour where an
in-flight job was silently demoted to "stopped" on any restart.
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
    from ..activities.colmap import run_colmap_activity
    from ..activities.train import train_activity
    from .types import ColmapArgs, TrainArgs, TrainParams

# Expensive, side-effectful activities: no auto-retry (a transient blip must not
# silently burn hours of GPU / stack OOM). Failures surface to the user instead.
_NO_RETRY = RetryPolicy(maximum_attempts=1)


@workflow.defn
class SplatTrainingWorkflow:
    def __init__(self) -> None:
        self._status = "queued"
        self._stage = {"stage": "queued", "detail": ""}
        self._config: dict = {}
        self._task: asyncio.Task | None = None

    @workflow.run
    async def run(self, p: TrainParams) -> dict:
        self._config = p.config
        try:
            # ---- poses ------------------------------------------------------
            self._status = "processing_poses"
            self._stage = {"stage": "features", "detail": "starting COLMAP"}
            self._task = workflow.start_activity(
                run_colmap_activity,
                ColmapArgs(p.project_id, p.config.get("colmap_matcher", "auto"), p.run_id),
                start_to_close_timeout=timedelta(hours=3),
                heartbeat_timeout=timedelta(minutes=5),
                cancellation_type=ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
                retry_policy=_NO_RETRY,
            )
            colmap = await self._task

            # ---- training ---------------------------------------------------
            self._status = "training"
            self._stage = {"stage": "init", "detail": "initializing"}
            self._task = workflow.start_activity(
                train_activity,
                TrainArgs(p.project_id, p.config, colmap.undistorted_uri, p.run_id),
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
        return {"status": self._status, "stage": self._stage, "config": self._config}
