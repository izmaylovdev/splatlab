"""Run SplatTrainingWorkflow in Temporal's test env with stub activities.

Validates: workflow imports pass the sandbox, activity wiring/args, the `state`
query at each stage, the pipeline run (sfm -> review -> approve -> train, with
settings changed at approval; stop during review; auto_train), the legacy split
phases (sfm -> poses_ready, then train -> done), and a `stop` signal cancelling
the running training activity -> 'stopped'. No GPU/Redis/COLMAP involved.

    .venv/bin/python -m tests.test_workflow_e2e
"""
import asyncio

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from server.workflows.training import SplatTrainingWorkflow
from server.workflows.types import ColmapArgs, ColmapResult, TrainArgs, TrainResult
from server.config import SFM_TASK_QUEUE, TASK_QUEUE, workflow_id
from server.workflows.types import TrainParams


@activity.defn(name="run_colmap_activity")
async def stub_colmap(a: ColmapArgs) -> ColmapResult:
    return ColmapResult(undistorted_rel="colmap/undistorted",
                        points_rel="colmap/points.ply",
                        points_uri="/files/x/colmap/points.ply",
                        num_points=1234, num_images=12)


@activity.defn(name="train_activity")
async def stub_train(a: TrainArgs) -> TrainResult:
    # final_step echoes the config the activity received, so tests can check
    # which settings reached training
    return TrainResult(final_step=a.config.get("max_steps", 1500),
                       snapshot_uri="/files/x/snapshots/step_001500.ply")


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


def _sfm_worker(client: Client) -> Worker:
    # COLMAP is dispatched to the SFM queue (control-plane machine), not the GPU one
    return Worker(client, task_queue=SFM_TASK_QUEUE, activities=[stub_colmap])


async def _until(h, pred, what: str) -> dict:
    for _ in range(100):
        st = await h.query(SplatTrainingWorkflow.state)
        if pred(st):
            return st
        await asyncio.sleep(0.1)
    raise AssertionError(f"never reached {what}: {st}")


async def pipeline_paths(client: Client):
    async with _sfm_worker(client), Worker(
            client, task_queue=TASK_QUEUE, workflows=[SplatTrainingWorkflow],
            activities=[stub_train]):
        # Review then approve, changing a setting at approval time.
        h = await client.start_workflow(
            SplatTrainingWorkflow.run,
            TrainParams(project_id="p3", config={"max_steps": 1000}, run_id="r3",
                        phase="pipeline"),
            id=workflow_id("p3"), task_queue=TASK_QUEUE)
        st = await _until(h, lambda s: s["awaiting_approval"], "review wait")
        assert st["status"] == "poses_ready" and st["sfm"]["num_images"] == 12, st
        await h.signal(SplatTrainingWorkflow.approve, {"max_steps": 2000})
        result = await asyncio.wait_for(h.result(), timeout=20)
        assert result == {"status": "done", "final_step": 2000,
                          "snapshot_uri": "/files/x/snapshots/step_001500.ply"}, result

        # Stop during review: closes with the poses kept, nothing trained.
        h = await client.start_workflow(
            SplatTrainingWorkflow.run,
            TrainParams(project_id="p4", config={}, run_id="r4", phase="pipeline"),
            id=workflow_id("p4"), task_queue=TASK_QUEUE)
        await _until(h, lambda s: s["awaiting_approval"], "review wait")
        await h.signal(SplatTrainingWorkflow.stop)
        result = await asyncio.wait_for(h.result(), timeout=20)
        assert result["status"] == "poses_ready" and result["num_points"] == 1234, result

        # auto_train: straight from SFM into training, no approval needed.
        h = await client.start_workflow(
            SplatTrainingWorkflow.run,
            TrainParams(project_id="p5", config={"auto_train": True, "max_steps": 700},
                        run_id="r5", phase="pipeline"),
            id=workflow_id("p5"), task_queue=TASK_QUEUE)
        result = await asyncio.wait_for(h.result(), timeout=20)
        assert result["status"] == "done" and result["final_step"] == 700, result
        print("pipeline paths: ok", flush=True)


async def happy_path(client: Client):
    async with _sfm_worker(client), Worker(
            client, task_queue=TASK_QUEUE, workflows=[SplatTrainingWorkflow],
            activities=[stub_train]):
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
        await pipeline_paths(env.client)
        await happy_path(env.client)
        await stop_path(env.client)
    print("WORKFLOW E2E PASSED")


if __name__ == "__main__":
    asyncio.run(main())
