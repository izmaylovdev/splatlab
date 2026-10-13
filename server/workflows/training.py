"""SplatTrainingWorkflow — durable orchestration of one pipeline *phase*.

Split into two phases (selected by TrainParams.phase), one workflow run each,
sharing the per-project id so they run sequentially and the id is reusable:

  "pipeline"  run COLMAP, publish the reviewable sparse point cloud, then wait
              for the user's `approve` signal (or continue at once when
              config["auto_train"]), then train — one run end to end. COLMAP
              runs on the control-plane machine (SFM queue), never a GPU box,
              and the pool only counts GPU-queue work as demand, so neither SFM
              nor an open review wait costs GPU. Poses-ready is durably
              recorded by the points.ply artifact.

  "train"     assume poses are ready (an earlier run uploaded colmap/undistorted)
              and run training only, re-materializing that dataset on whatever
              box picks the job up. Used to re-train with new settings.

  "sfm"       legacy, from before "pipeline" existed: COLMAP, then the same
              review -> train flow. Runs that completed right after COLMAP
              replay through it unchanged (patch "legacy-sfm-joins-pipeline").

Thin and deterministic: it only sequences activities, tracks coarse status/stage
for the query API, and reacts to a `stop` signal. All filesystem / CUDA /
subprocess / Redis side effects live in the activities. Config is passed in as
data (never read from disk here) so replay stays deterministic.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
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
_SFM_QUEUE = config.SFM_TASK_QUEUE

# SFM runs serially on one machine, so a big batch queues for hours behind
# itself; the GPU-box boot window above doesn't apply. Wide enough for a batch.
_SFM_SCHEDULE_TO_START = timedelta(hours=12)

# A GPU box may not exist the instant a run is queued — the autoscaler needs a
# reconcile cycle plus cold-boot time to bring one online. Give the activity a
# generous schedule-to-start window so it waits in the queue for a box instead of
# failing; start-to-close still bounds the actual run once a box picks it up.
#
# Cold boot on a fresh box today = image pull (up to the pool boot deadline) PLUS
# an on-box torch/gsplat download (~800MB torch, slow on some hosts ≈ 25min). A
# reaped-and-re-rented slow host can stack these, so the window has to cover a
# couple of boot attempts. A prebaked image (torch/gsplat/pycolmap preinstalled)
# would cut boot to a few minutes and let this drop back to ~15min.
_SCHEDULE_TO_START = timedelta(minutes=60)

# How long Temporal waits for the next activity heartbeat before declaring the
# activity dead. The activity itself beats every 10s (activities/runner.py), so
# this is pure slack for the network in between — and that network is a single
# tunnel from the Mac (via the GCP relay) carrying Temporal, Redis AND the bulk
# MinIO traffic. A
# training run pulls the undistorted dataset down and pushes .ply snapshots back
# through that same uplink; while a multi-hundred-MB transfer saturates it the
# heartbeat RPC can be starved for minutes even though the box is perfectly
# healthy. At 60s that killed a 29-minute-in room run (15k steps, 76 images).
#
# Raising this does NOT slow down /stop: cancellation rides the heartbeat
# *response*, throttled at SPLATLAB_HEARTBEAT_THROTTLE (5s), independent of this
# timeout. The only cost is that a box which dies hard takes this long to notice,
# which start_to_close still bounds.
_HEARTBEAT_TIMEOUT = timedelta(minutes=10)

# How long a pipeline run waits for the user to approve training after SFM. An
# open wait costs nothing (no activity is scheduled), but it shouldn't hang
# forever: past this the run completes as poses-ready, and a later Train starts
# a training-only run from the stored poses.
_REVIEW_WINDOW = timedelta(days=7)


@workflow.defn
class SplatTrainingWorkflow:
    def __init__(self) -> None:
        self._status = "queued"
        self._stage = {"stage": "queued", "detail": ""}
        self._config: dict = {}
        self._phase = "sfm"
        self._sfm: dict = {}          # {points_uri, num_points, num_images} once ready
        self._task: asyncio.Task | None = None
        self._awaiting_approval = False
        self._approved_config: dict | None = None   # set by the approve signal
        self._stop_requested = False

    @workflow.run
    async def run(self, p: TrainParams) -> dict:
        self._config = p.config
        self._phase = p.phase
        try:
            if p.phase == "pipeline":
                return await self._run_pipeline(p)
            if p.phase == "sfm":
                return await self._run_legacy_sfm(p)
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
        # COLMAP is CPU-only, so it runs on the control-plane machine (SFM queue),
        # not a rented GPU box. Runs that scheduled it before this change replay
        # with patched() == False and keep their original GPU-queue command.
        on_sfm_queue = workflow.patched("sfm-on-control-plane")
        self._task = workflow.start_activity(
            run_colmap_activity,
            ColmapArgs(p.project_id, p.config.get("colmap_matcher", "auto"), p.run_id),
            task_queue=_SFM_QUEUE if on_sfm_queue else _GPU_QUEUE,
            schedule_to_start_timeout=(_SFM_SCHEDULE_TO_START if on_sfm_queue
                                       else _SCHEDULE_TO_START),
            start_to_close_timeout=timedelta(hours=3),
            heartbeat_timeout=_HEARTBEAT_TIMEOUT,
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

    async def _run_legacy_sfm(self, p: TrainParams) -> dict:
        # "sfm" runs queued before "pipeline" existed: same COLMAP command, so
        # runs still waiting on it can join the review -> train flow when it
        # finishes. Runs that already completed replay with patched() == False.
        colmap_done = await self._run_sfm(p)
        if not workflow.patched("legacy-sfm-joins-pipeline"):
            return colmap_done
        return await self._review_then_train(p)

    async def _run_pipeline(self, p: TrainParams) -> dict:
        await self._run_sfm(p)
        return await self._review_then_train(p)

    async def _review_then_train(self, p: TrainParams) -> dict:
        if self._stop_requested:          # stop landed as COLMAP finished
            return self._end_review("closed without training")
        if not p.config.get("auto_train"):
            self._awaiting_approval = True
            self._stage = {"stage": "review",
                           "detail": f"{self._sfm['num_points']} points from "
                                     f"{self._sfm['num_images']} images — review, "
                                     "then approve training"}
            try:
                await workflow.wait_condition(
                    lambda: self._approved_config is not None or self._stop_requested,
                    timeout=_REVIEW_WINDOW)
            except asyncio.TimeoutError:
                return self._end_review("review window expired — Train starts a new run")
            finally:
                self._awaiting_approval = False
            if self._stop_requested:
                return self._end_review("closed without training")
            self._config = {**self._config, **self._approved_config}
        return await self._run_train(replace(p, config=self._config))

    def _end_review(self, detail: str) -> dict:
        # Nothing was running, so this isn't a "stopped" run: the poses are made
        # and stay usable. Report the resting poses-ready state.
        self._status = "poses_ready"
        self._stage = {"stage": "review", "detail": detail}
        return {"status": "poses_ready", **self._sfm}

    async def _run_train(self, p: TrainParams) -> dict:
        self._status = "training"
        self._stage = {"stage": "init", "detail": "initializing"}
        self._task = workflow.start_activity(
            train_activity,
            # Poses were produced by a prior sfm run and uploaded to this fixed
            # project-relative path; the activity re-materializes it locally.
            # `resume` rides in the config dict, not as a TrainArgs field: boxes
            # booted on older code reject unknown TrainArgs fields outright.
            TrainArgs(p.project_id, {**p.config, "resume": True} if p.resume
                      else p.config, UNDISTORTED_REL, p.run_id),
            task_queue=_GPU_QUEUE,
            schedule_to_start_timeout=_SCHEDULE_TO_START,
            start_to_close_timeout=timedelta(hours=12),
            heartbeat_timeout=_HEARTBEAT_TIMEOUT,
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
        self._stop_requested = True       # also ends a pipeline's review wait
        if self._task is not None and not self._task.done():
            self._task.cancel()

    @workflow.signal
    def approve(self, config: dict | None = None) -> None:
        """Start training after review, optionally with updated settings."""
        if self._awaiting_approval and self._approved_config is None:
            self._approved_config = config or {}

    @workflow.query
    def state(self) -> dict:
        return {"status": self._status, "stage": self._stage,
                "config": self._config, "phase": self._phase, "sfm": self._sfm,
                "awaiting_approval": self._awaiting_approval}
