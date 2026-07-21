"""SplatLab Temporal worker — runs on the GPU box.

Polls the `splat-gpu` task queue and executes SplatTrainingWorkflow plus the
COLMAP / training activities. This process is the *only* thing that needs a GPU,
COLMAP, gsplat, and (on the S3 backend) local scratch disk.

Run:  python -m server.worker.main
"""
from __future__ import annotations

import asyncio
import os
from datetime import timedelta

# How often the SDK actually flushes heartbeat RPCs to the server. This also
# bounds how fast a /stop (activity cancellation) is delivered to a running
# activity — the default (~30s) makes stops feel unresponsive. A few seconds
# keeps stop latency low at a negligible RPC cost.
HEARTBEAT_THROTTLE = timedelta(seconds=int(os.environ.get("SPLATLAB_HEARTBEAT_THROTTLE", "5")))

from temporalio.client import Client
from temporalio.worker import Worker

from .. import config
from ..activities.colmap import run_colmap_activity
from ..activities.train import train_activity
from ..workflows.training import SplatTrainingWorkflow

# GPU memory — not Temporal defaults — is the real concurrency limit. One or two
# concurrent trainings per box; override with SPLATLAB_MAX_ACTIVITIES.
MAX_ACTIVITIES = int(os.environ.get("SPLATLAB_MAX_ACTIVITIES", "2"))


async def main() -> None:
    client = await Client.connect(
        config.TEMPORAL_ADDRESS, namespace=config.TEMPORAL_NAMESPACE)

    worker = Worker(
        client,
        task_queue=config.TASK_QUEUE,
        workflows=[SplatTrainingWorkflow],
        activities=[run_colmap_activity, train_activity],
        max_concurrent_activities=MAX_ACTIVITIES,
        # Deliver activity cancellation (/stop) promptly, not on the ~30s default.
        default_heartbeat_throttle_interval=HEARTBEAT_THROTTLE,
        max_heartbeat_throttle_interval=HEARTBEAT_THROTTLE,
        # Training won't stop within Temporal's default 10s. Give cancellation
        # time to flush a final snapshot; real handoff relies on the
        # heartbeat-timeout if the box dies hard.
        graceful_shutdown_timeout=timedelta(minutes=5),
    )
    print(f"SplatLab worker → {config.TEMPORAL_ADDRESS} "
          f"queue={config.TASK_QUEUE} storage={config.STORAGE_BACKEND} "
          f"max_activities={MAX_ACTIVITIES}")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
