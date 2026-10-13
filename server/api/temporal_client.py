"""Thin async wrapper around the Temporal client used by the API service.

Workflows are addressed by *string* type/query names so the API never imports
worker/activity code (keeping it GPU- and CUDA-free). One workflow per project,
id ``train-<pid>``.
"""
from __future__ import annotations

from typing import Any

from temporalio.client import Client, WorkflowHandle
from temporalio.api.enums.v1 import PendingActivityState
from temporalio.common import WorkflowIDConflictPolicy
from temporalio.service import RPCError, RPCStatusCode

try:  # moved from temporalio.client to temporalio.exceptions in newer SDKs
    from temporalio.exceptions import WorkflowAlreadyStartedError
except ImportError:  # pragma: no cover - older temporalio
    from temporalio.client import WorkflowAlreadyStartedError  # type: ignore

from .. import config
from ..workflows.types import TrainParams

WORKFLOW_TYPE = "SplatTrainingWorkflow"

_client: Client | None = None


async def get_client() -> Client:
    global _client
    if _client is None:
        _client = await Client.connect(
            config.TEMPORAL_ADDRESS, namespace=config.TEMPORAL_NAMESPACE)
    return _client


def _handle(client: Client, pid: str) -> WorkflowHandle:
    return client.get_workflow_handle(config.workflow_id(pid))


class AlreadyRunning(Exception):
    """A training workflow for this project is already running."""


async def start_run(pid: str, cfg: dict[str, Any], run_id: str, phase: str,
                    resume: bool = False) -> None:
    """Start a run for a project: "pipeline" (SFM -> review -> train) or "train"
    (training only, from stored poses; ``resume`` continues from the latest
    snapshot instead of starting over).

    Runs share the per-project workflow id, so the FAIL conflict policy rejects
    a second *concurrent* run while letting the id be reused once a run closes.
    """
    client = await get_client()
    try:
        await client.start_workflow(
            WORKFLOW_TYPE,
            TrainParams(project_id=pid, config=cfg, run_id=run_id, phase=phase,
                        resume=resume),
            id=config.workflow_id(pid),
            # Workflow runs on the always-on control plane; it dispatches the
            # GPU activities to the pool-managed GPU queue itself.
            task_queue=config.CONTROL_TASK_QUEUE,
            id_conflict_policy=WorkflowIDConflictPolicy.FAIL,
        )
    except WorkflowAlreadyStartedError as e:
        raise AlreadyRunning(str(e)) from e


async def stop_training(pid: str) -> None:
    """Signal the workflow to stop (cancels the running activity gracefully).

    Best-effort: if there's no running workflow, or Temporal is unreachable,
    there's nothing to stop — return quietly (also lets delete proceed).
    """
    try:
        client = await get_client()
        await _handle(client, pid).signal("stop")
    except RPCError as e:
        if e.status == RPCStatusCode.NOT_FOUND:
            return
        raise
    except Exception:
        return


async def approve(pid: str, cfg: dict[str, Any]) -> None:
    """Tell a pipeline run waiting on review to go ahead and train with cfg."""
    client = await get_client()
    await _handle(client, pid).signal("approve", cfg)


async def close_review(pid: str, timeout: float = 15.0) -> None:
    """End a pipeline run that is waiting on review and wait until it has
    closed, so the project's workflow id is free for a new run."""
    import asyncio

    client = await get_client()
    handle = _handle(client, pid)
    await handle.signal("stop")
    try:
        await asyncio.wait_for(handle.result(), timeout)
    except Exception:  # noqa: BLE001  (closed with any outcome is fine)
        pass


_TERMINAL = {
    "COMPLETED": "done",
    "FAILED": "error",
    "CANCELED": "stopped",
    "TERMINATED": "error",
    "TIMED_OUT": "error",
}


async def query_state(pid: str) -> dict[str, Any] | None:
    """Current {status, stage, config} from the workflow, or None.

    Read-only and never raises — callers fall back to stored meta status. Returns
    None when there is no workflow OR Temporal itself is unreachable, so the API
    keeps serving (degraded) while the control plane is down.
    """
    try:
        client = await get_client()
    except Exception:
        return None
    handle = _handle(client, pid)
    try:
        return await handle.query("state")
    except RPCError as e:
        if e.status == RPCStatusCode.NOT_FOUND:
            return None
        # Workflow exists but isn't queryable (failed/terminated). Derive coarse
        # status from its execution description so a failed run isn't reported stale.
        try:
            desc = await handle.describe()
            name = getattr(desc.status, "name", str(desc.status))
            return {"status": _TERMINAL.get(name, "error"), "stage": None, "config": None}
        except Exception:
            return None
    except Exception:
        return None


async def waiting_activity(pid: str) -> dict[str, Any] | None:
    """The open run's activity if it is queued but not yet picked up by a
    worker — {"activity": "colmap"|"train", "since": epoch seconds} — else None
    (running, idle, no run, or Temporal unreachable). Lets the UI tell a project
    waiting for the SFM worker / a GPU box from one actually running."""
    try:
        client = await get_client()
        desc = await _handle(client, pid).describe()
    except Exception:
        return None
    for a in desc.raw_description.pending_activities:
        if a.state == PendingActivityState.PENDING_ACTIVITY_STATE_SCHEDULED:
            name = a.activity_type.name
            return {"activity": "colmap" if "colmap" in name else "train",
                    "since": a.scheduled_time.seconds}
    return None
