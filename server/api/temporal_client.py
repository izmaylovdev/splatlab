"""Thin async wrapper around the Temporal client used by the API service.

Workflows are addressed by *string* type/query names so the API never imports
worker/activity code (keeping it GPU- and CUDA-free). One workflow per project,
id ``train-<pid>``.
"""
from __future__ import annotations

from typing import Any

from temporalio.client import Client, WorkflowHandle
from temporalio.common import WorkflowIDConflictPolicy
from temporalio.service import RPCError, RPCStatusCode

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


async def start_training(pid: str, cfg: dict[str, Any], run_id: str) -> None:
    from temporalio.client import WorkflowAlreadyStartedError

    client = await get_client()
    try:
        await client.start_workflow(
            WORKFLOW_TYPE,
            TrainParams(project_id=pid, config=cfg, run_id=run_id),
            id=config.workflow_id(pid),
            # Workflow runs on the always-on control plane; it dispatches the
            # GPU activities to the pool-managed GPU queue itself.
            task_queue=config.CONTROL_TASK_QUEUE,
            # Reject a second concurrent run for the same project; allow a fresh
            # run once the previous one has closed.
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
