"""SplatLab Temporal worker(s).

Two roles, selected by SPLATLAB_WORKER_ROLE:

  gpu      Polls the GPU queue (config.GPU_TASK_QUEUE) and runs the COLMAP /
           training activities. This is the *only* process that needs a GPU,
           COLMAP, gsplat, and scratch disk. On a pool-rented box it also reports
           liveness + in-flight load to the Redis box registry (via SPLATLAB_BOX_ID)
           so the autoscaler can tell when the box is idle or dead.

  control  Polls the control queue (config.CONTROL_TASK_QUEUE) and runs
           SplatTrainingWorkflow only. Lives on the always-on control plane so the
           workflow survives every GPU box being reaped; it dispatches the GPU
           activities to the GPU queue itself. No GPU needed.

  both     One process runs both of the above (co-located dev / single box). This
           is the default, so existing single-box deployments keep working.

Run:  python -m server.worker.main            # role from SPLATLAB_WORKER_ROLE (default: both)
      SPLATLAB_WORKER_ROLE=gpu python -m server.worker.main
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from datetime import timedelta

# How often the SDK actually flushes heartbeat RPCs to the server. This also
# bounds how fast a /stop (activity cancellation) is delivered to a running
# activity — the default (~30s) makes stops feel unresponsive. A few seconds
# keeps stop latency low at a negligible RPC cost.
HEARTBEAT_THROTTLE = timedelta(seconds=int(os.environ.get("SPLATLAB_HEARTBEAT_THROTTLE", "5")))

import redis
from temporalio.client import Client
from temporalio.worker import (ActivityInboundInterceptor, ExecuteActivityInput,
                               Interceptor, Worker)

from .. import config
from ..activities.colmap import run_colmap_activity
from ..activities.train import train_activity
from ..vast import registry
from ..workflows.training import SplatTrainingWorkflow

# GPU memory — not Temporal defaults — is the real concurrency limit. One or two
# concurrent trainings per box; override with SPLATLAB_MAX_ACTIVITIES.
MAX_ACTIVITIES = int(os.environ.get("SPLATLAB_MAX_ACTIVITIES", "2"))

ROLE = os.environ.get("SPLATLAB_WORKER_ROLE", "both").lower()
# Set by the pool onstart on a rented box; enables registry reporting. Absent on
# a manually-run / co-located worker (registry reporting is then simply skipped).
BOX_ID = os.environ.get("SPLATLAB_BOX_ID") or None
BOX_HEARTBEAT_INTERVAL = int(os.environ.get("SPLATLAB_BOX_HEARTBEAT", "20"))


# ---- box registry reporting (pool-rented boxes only) ------------------------
class _BoxTracker(Interceptor):
    """Reports this box's liveness + in-flight activity count to Redis so the
    autoscaler can idle-reap it safely. Counts activities via an interceptor and
    heartbeats on a background thread."""

    def __init__(self, r: "redis.Redis", box_id: str) -> None:
        self._r = r
        self._box_id = box_id
        self._lock = threading.Lock()
        self._count = 0
        registry.register_ready(r, box_id)
        threading.Thread(target=self._beat, daemon=True).start()

    def _beat(self) -> None:
        while True:
            try:
                with self._lock:
                    n = self._count
                registry.heartbeat(self._r, self._box_id, n)
            except Exception:
                pass  # transient Redis blip: next tick retries
            time.sleep(BOX_HEARTBEAT_INTERVAL)

    def _delta(self, d: int) -> None:
        with self._lock:
            self._count += d
        try:
            registry.touch_active(self._r, self._box_id)
        except Exception:
            pass

    def intercept_activity(
            self, next: ActivityInboundInterceptor) -> ActivityInboundInterceptor:
        outer = self

        class _Inbound(ActivityInboundInterceptor):
            async def execute_activity(self, input: ExecuteActivityInput):
                outer._delta(+1)
                try:
                    return await super().execute_activity(input)
                finally:
                    outer._delta(-1)

        return _Inbound(next)


def _gpu_worker(client: Client) -> Worker:
    interceptors: list[Interceptor] = []
    if BOX_ID:
        r = redis.Redis.from_url(config.REDIS_URL)
        interceptors.append(_BoxTracker(r, BOX_ID))
    return Worker(
        client,
        task_queue=config.GPU_TASK_QUEUE,
        activities=[run_colmap_activity, train_activity],
        interceptors=interceptors,
        max_concurrent_activities=MAX_ACTIVITIES,
        # Deliver activity cancellation (/stop) promptly, not on the ~30s default.
        default_heartbeat_throttle_interval=HEARTBEAT_THROTTLE,
        max_heartbeat_throttle_interval=HEARTBEAT_THROTTLE,
        # Training won't stop within Temporal's default 10s. Give cancellation
        # time to flush a final snapshot; real handoff relies on the
        # heartbeat-timeout if the box dies hard.
        graceful_shutdown_timeout=timedelta(minutes=5),
    )


def _control_worker(client: Client) -> Worker:
    return Worker(
        client,
        task_queue=config.CONTROL_TASK_QUEUE,
        workflows=[SplatTrainingWorkflow],
    )


async def main() -> None:
    client = await Client.connect(
        config.TEMPORAL_ADDRESS, namespace=config.TEMPORAL_NAMESPACE)

    workers: list[Worker] = []
    if ROLE in ("gpu", "both"):
        workers.append(_gpu_worker(client))
    if ROLE in ("control", "both"):
        workers.append(_control_worker(client))
    if not workers:
        raise SystemExit(f"unknown SPLATLAB_WORKER_ROLE={ROLE!r} (want gpu|control|both)")

    box = f" box={BOX_ID}" if BOX_ID else ""
    print(f"SplatLab worker(s) role={ROLE} → {config.TEMPORAL_ADDRESS} "
          f"gpu_queue={config.GPU_TASK_QUEUE} control_queue={config.CONTROL_TASK_QUEUE} "
          f"storage={config.STORAGE_BACKEND} max_activities={MAX_ACTIVITIES}{box}")
    await asyncio.gather(*(w.run() for w in workers))


if __name__ == "__main__":
    asyncio.run(main())
