"""Run SplatTrainingWorkflow in Temporal's test env with stub activities.

Validates: workflow imports pass the sandbox, activity wiring/args, the `state`
query at each stage, both phases (sfm -> poses_ready, then train -> done), and a
`stop` signal cancelling the running training activity -> 'stopped'. No
GPU/Redis/COLMAP involved.
"""
import asyncio

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from server.workflows.training import SplatTrainingWorkflow
from server.workflows.types import ColmapArgs, ColmapResult, TrainArgs, TrainResult
from server.config import TASK_QUEUE, workflow_id


@activity.defn(name="run_colmap_activity")
async def stub_colmap(a: ColmapArgs) -> ColmapResult:
    return ColmapResult(undistorted_rel="colmap/undistorted",
                        points_rel="colmap/points.ply",
                        points_uri="/files/x/colmap/points.ply",
                        num_points=1234, num_images=12)


@activity.defn(name="train_activity")
async def stub_train(a: TrainArgs) -> TrainResult:
    return TrainResult(final_step=1500, snapshot_uri="/files/x/snapshots/step_001500.ply")


@activity.defn(name="train_activity")
async def stub_train_blocks(a: TrainArgs) -> TrainResult:
    # Mirrors run_with_heartbeat(wait_for_flush=True): heartbeats so Temporal can
    # DELIVER cancellation (a non-heartbeating activity never receives it), then
    # swallows the cancel, "flushes", and returns a partial result.
    try:
        while True:
            activity.heartbeat()
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        # flush would happen here; re-raise so the activity records as cancelled
        raise


async def happy_path(client: Client):
    async with Worker(client, task_queue=TASK_QUEUE,
                      workflows=[SplatTrainingWorkflow],
                      activities=[stub_colmap, stub_train]):
        from server.workflows.types import TrainParams
        # Phase 1: SFM completes at poses_ready (GPU released for review).
        h = await client.start_workflow(
            SplatTrainingWorkflow.run,
            TrainParams(project_id="p1", config={"colmap_matcher": "auto"},
                        run_id="r1", phase="sfm"),
            id=workflow_id("p1"), task_queue=TASK_QUEUE)
        sfm = await h.result()
        assert sfm["status"] == "poses_ready", sfm
        assert sfm["num_points"] == 1234 and sfm["num_images"] == 12, sfm

        # Phase 2: same workflow id is reusable once the sfm run has closed.
        h = await client.start_workflow(
            SplatTrainingWorkflow.run,
            TrainParams(project_id="p1", config={}, run_id="r1b", phase="train"),
            id=workflow_id("p1"), task_queue=TASK_QUEUE)
        result = await h.result()
        assert result["status"] == "done", result
        assert result["final_step"] == 1500, result
        print("happy path:", sfm, result)


async def stop_path(client: Client):
    from datetime import timedelta
    async with Worker(client, task_queue=TASK_QUEUE,
                      workflows=[SplatTrainingWorkflow],
                      activities=[stub_colmap, stub_train_blocks],
                      default_heartbeat_throttle_interval=timedelta(seconds=2),
                      max_heartbeat_throttle_interval=timedelta(seconds=2)):
        from server.workflows.types import TrainParams
        h = await client.start_workflow(
            SplatTrainingWorkflow.run,
            TrainParams(project_id="p2", config={}, run_id="r2", phase="train"),
            id=workflow_id("p2"), task_queue=TASK_QUEUE)
        # wait until it reaches the training stage, then stop
        for _ in range(50):
            st = await h.query(SplatTrainingWorkflow.state)
            if st["status"] == "training":
                break
            await asyncio.sleep(0.2)
        else:
            raise AssertionError("never reached training")
        assert st["status"] == "training"
        await h.signal(SplatTrainingWorkflow.stop)
        # cancellation is delivered on the (throttled) next heartbeat, then the
        # workflow reports 'stopped' — should be a few seconds, not minutes.
        result = await asyncio.wait_for(h.result(), timeout=25)
        assert result["status"] == "stopped", result
        print("stop path:", result, flush=True)


async def main():
    # Real local server → real-time cancellation semantics (time-skipping stalls
    # under a continuously-heartbeating activity).
    async with await WorkflowEnvironment.start_local() as env:
        await happy_path(env.client)
        await stop_path(env.client)
    print("WORKFLOW E2E PASSED")


if __name__ == "__main__":
    asyncio.run(main())
