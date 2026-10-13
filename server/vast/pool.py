"""SplatLab GPU autoscaler — rents/reaps Vast.ai boxes to match demand.

Run on the control plane:  python -m server.vast.pool

Every `POOL_RECONCILE_INTERVAL` seconds it computes:

    demand   = GPU-queue work: training activities waiting for a box, plus
               activities running on boxes (COLMAP runs off-box, never counts)
               (open SplatTrainingWorkflow executions, via Temporal visibility)
    desired  = min(POOL_MAX_BOXES, ceil(demand / POOL_JOBS_PER_BOX))

and reconciles the live fleet toward `desired`:

    * reap dead boxes      — stale heartbeat, boot timeout, or past max lifetime
    * reap orphans         — Vast instances tagged ours but not in the registry
    * scale down           — destroy the longest-idle boxes past POOL_IDLE_TIMEOUT
                             while the fleet is larger than desired (warm pool: a
                             just-finished box lingers for reuse until then)
    * scale up             — rent the cheapest offers meeting the filters

Safety: it only ever touches Vast instances carrying VAST_LABEL, honours a
POOL_PAUSED kill-switch (drain to zero), and caps spend via POOL_MAX_BOXES +
VAST_MAX_PRICE + POOL_MAX_LIFETIME. Teardown does not depend on any single
workflow — a crashed worker is reaped by the stale-heartbeat rule, and a leaked
instance by the orphan rule, so a paid GPU is never left running unmanaged.
"""
from __future__ import annotations

import asyncio
import math
import time
import uuid
from urllib.parse import urlsplit

import redis

from temporalio.api.enums.v1 import PendingActivityState

from .. import config
from . import links, registry
from .client import VastClient, VastError

WORKFLOW_TYPE = "SplatTrainingWorkflow"


def _log(msg: str) -> None:
    print(f"[pool {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- demand -----------------------------------------------------------------
# Activities dispatched to the GPU queue. COLMAP is not one of them: it runs on the
# control-plane machine (config.SFM_TASK_QUEUE), so an open SFM-phase workflow
# must not rent a box.
_GPU_ACTIVITIES = frozenset({"train_activity"})


async def _gpu_demand(temporal, boxes: list[registry.Box]) -> int:
    """GPU work right now = GPU activities still waiting for a box + activities
    already running on boxes (the boxes' own reported in-flight counts).

    Waiting work is read from each open workflow's pending activity rather than
    the task queue's backlog stat: that stat still counts tasks of workflows that
    were terminated or timed out until a poller discards them, which would rent
    boxes for work that no longer exists. Falls back to the open-workflow count
    (over-rents, never under-rents) if Temporal can't be described.
    """
    query = f"ExecutionStatus='Running' AND WorkflowType='{WORKFLOW_TYPE}'"
    try:
        waiting = 0
        async for wf in temporal.list_workflows(query):
            desc = await temporal.get_workflow_handle(wf.id, run_id=wf.run_id).describe()
            for pa in desc.raw_description.pending_activities:
                if (pa.activity_type.name in _GPU_ACTIVITIES and pa.state
                        == PendingActivityState.PENDING_ACTIVITY_STATE_SCHEDULED):
                    waiting += 1
    except Exception as e:
        _log(f"WARN could not read pending activities: {e!r}; using open-workflow count")
        return await _open_workflow_count(temporal)
    return waiting + sum(b.active_count for b in boxes)


async def _open_workflow_count(temporal) -> int:
    """How many training workflows are open (Running) right now."""
    query = f"ExecutionStatus='Running' AND WorkflowType='{WORKFLOW_TYPE}'"
    try:
        resp = await temporal.count_workflows(query)
        return int(resp.count)
    except Exception:
        # Older SDKs / server: fall back to iterating the visibility list.
        try:
            n = 0
            async for _ in temporal.list_workflows(query):
                n += 1
            return n
        except Exception as e:
            _log(f"WARN could not read demand from Temporal: {e}; assuming 0")
            return 0


# ---- box bootstrap ----------------------------------------------------------
# A rented box can't dial the (NAT'd, laptop-local) control plane directly. Two
# ways around it (config.BOX_LINK):
#   ssh    the pool SSHes OUT to the box and reverse-forwards Temporal/Redis/MinIO
#          onto the box's localhost (server/vast/links.py); the box uses localhost.
#   relay  the Mac runs `deploy/gcp_tunnel_up.sh`, which reverse-forwards them to
#          a GCP relay VM's public ports and writes those into deploy/.env as
#          BOX_*; the box dials them OUT.
# Either way the addresses forwarded to the box differ from the ones the pool
# itself uses (in-compose service names).
#
# Env forwarded verbatim (creds + bootstrap tunables — same value for pool & box).
_FORWARD_ENV = (
    "TEMPORAL_NAMESPACE",
    "SPLATLAB_S3_BUCKET", "SPLATLAB_S3_REGION",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
    "SPLATLAB_S3_ACCESS_KEY", "SPLATLAB_S3_SECRET_KEY",
    # bootstrap tunables consumed by deploy/vast_setup.sh
    "TORCH_VERSION", "TV_VERSION", "CUDA_TAG", "GSPLAT_WHEEL",
)


def _box_env(box_id: str) -> dict[str, str]:
    import os
    env = {k: os.environ[k] for k in _FORWARD_ENV if os.environ.get(k)}

    if config.BOX_LINK == "ssh":
        # The pool's SSH links put the services on the box's own loopback.
        r = urlsplit(config.REDIS_URL)
        auth = r.netloc.rsplit("@", 1)[0] + "@" if "@" in r.netloc else ""
        env["TEMPORAL_ADDRESS"] = f"localhost:{links.BOX_TEMPORAL_PORT}"
        env["REDIS_URL"] = f"{r.scheme}://{auth}localhost:{links.BOX_REDIS_PORT}{r.path}"
        env["SPLATLAB_S3_ENDPOINT"] = f"http://localhost:{links.BOX_S3_PORT}"
        env["SPLATLAB_WAIT_FOR_LINK"] = "1"   # vast_setup.sh waits for the tunnel
    else:
        # Relay: the explicit BOX_* addresses, else the pool's own value.
        def _box_addr(box_key: str, own: str) -> str:
            return os.environ.get(box_key) or own

        env["TEMPORAL_ADDRESS"] = _box_addr("BOX_TEMPORAL_ADDRESS", config.TEMPORAL_ADDRESS)
        env["REDIS_URL"] = _box_addr("BOX_REDIS_URL", config.REDIS_URL)
        s3_endpoint = (os.environ.get("BOX_S3_ENDPOINT")
                       or os.environ.get("SPLATLAB_S3_ENDPOINT"))
        if s3_endpoint:
            env["SPLATLAB_S3_ENDPOINT"] = s3_endpoint

    # Non-negotiable: the box runs GPU activities only (the control plane owns the
    # workflow), polls the GPU queue, and reports under its box id.
    env["SPLATLAB_WORKER_ROLE"] = "gpu"
    env["SPLATLAB_TASK_QUEUE"] = config.GPU_TASK_QUEUE
    env["SPLATLAB_BOX_ID"] = box_id
    env["SPLATLAB_STORAGE"] = "s3"            # ephemeral box shares no FS with API
    env.setdefault("GSPLAT_WHEEL", "1")       # prebuilt wheel: skips the slow CUDA compile on boot
    # The prebaked image (deploy/Dockerfile.box) deliberately does NOT put its venv
    # on the global PATH (that breaks Vast's onstart), and Vast may not propagate the
    # image's own ENV into onstart — so signal the prebaked fast path via the box env
    # too. vast_setup.sh falls back to a full install if the baked deps aren't there.
    if "splatlab-box" in config.VAST_IMAGE:
        env["SPLATLAB_PREBAKED"] = "1"
    return env


def _onstart(box_id: str) -> str:
    """Shell run on a fresh box: clone the repo and start the worker.

    Secrets/addresses arrive via the instance env (see `_box_env`); only
    non-secret repo coordinates are inlined here.
    """
    return (
        "set -e\n"
        "export DEBIAN_FRONTEND=noninteractive\n"
        "command -v git >/dev/null 2>&1 || (apt-get update -y && apt-get install -y git)\n"
        "cd /workspace 2>/dev/null || cd /root\n"
        "rm -rf splatlab\n"
        f"git clone --depth 1 --branch {config.VAST_REPO_BRANCH} "
        f"{config.VAST_REPO_URL} splatlab\n"
        "cd splatlab\n"
        "bash deploy/vast_setup.sh\n"
    )


# ---- the pool ---------------------------------------------------------------
class Pool:
    def __init__(self, temporal, vast: VastClient, r) -> None:
        self.temporal = temporal
        self.vast = vast
        self.r = r
        self.links = links.BoxLinks(vast) if config.BOX_LINK == "ssh" else None

    async def _destroy(self, box: registry.Box, reason: str) -> None:
        _log(f"destroy {box.box_id} (instance {box.instance_id}) — {reason}")
        if self.links:
            self.links.drop(box.box_id)
        if box.instance_id:
            try:
                await self.vast.destroy_instance(box.instance_id)
            except VastError as e:
                _log(f"  destroy warned (treating as gone): {e}")
        registry.remove_box(self.r, box.box_id)

    async def _rent(self, n: int) -> None:
        try:
            offers = await self.vast.search_offers(
                gpu_name=config.VAST_GPU_NAME,
                min_gpu_ram_gb=config.VAST_MIN_GPU_RAM_GB,
                min_reliability=config.VAST_MIN_RELIABILITY,
                max_price=config.VAST_MAX_PRICE,
                num_gpus=config.VAST_NUM_GPUS,
                min_inet_mbps=config.VAST_MIN_INET_MBPS,
                verification=config.VAST_VERIFICATION,
                limit=max(20, n * 5),
            )
        except VastError as e:
            _log(f"offer search failed: {e}")
            return
        if not offers:
            _log("no offers match the filters (check VAST_GPU_NAME / VAST_MAX_PRICE)")
            return

        rented = 0
        for offer in offers:
            if rented >= n:
                break
            box_id = uuid.uuid4().hex[:12]
            label = f"{config.VAST_LABEL}:{box_id}"
            try:
                instance_id = await self.vast.create_instance(
                    int(offer["id"]),
                    image=config.VAST_IMAGE,
                    disk_gb=config.VAST_DISK_GB,
                    onstart=_onstart(box_id),
                    label=label,
                    env=_box_env(box_id),
                )
            except VastError as e:
                _log(f"rent offer {offer.get('id')} failed, trying next: {e}")
                continue
            registry.put_booting(
                self.r, box_id, instance_id,
                offer_id=int(offer["id"]),
                dph=float(offer.get("dph_total", 0) or 0),
                gpu_name=str(offer.get("gpu_name", "")),
            )
            _log(f"rented {box_id} (instance {instance_id}) "
                 f"{offer.get('gpu_name')} @ ${offer.get('dph_total')}/hr")
            rented += 1
        if rented < n:
            _log(f"wanted {n} box(es), rented {rented}")

    async def _reap_orphans(self, known_ids: set[str]) -> None:
        """Destroy Vast instances tagged ours but absent from the registry
        (leaked by a pool crash between create and registry write)."""
        prefix = f"{config.VAST_LABEL}:"
        try:
            instances = await self.vast.list_instances()
        except VastError as e:
            _log(f"orphan scan skipped (list failed): {e}")
            return
        for inst in instances:
            label = str(inst.get("label") or "")
            if not label.startswith(prefix):
                continue
            box_id = label[len(prefix):]
            if box_id and box_id not in known_ids:
                iid = inst.get("id")
                _log(f"orphan instance {iid} (label {label}) — destroying")
                try:
                    await self.vast.destroy_instance(int(iid))
                except VastError as e:
                    _log(f"  orphan destroy warned: {e}")

    async def reconcile(self) -> None:
        now = int(time.time())
        boxes = registry.list_boxes(self.r)
        known_ids = {b.box_id for b in boxes}

        # 1. Reap dead / expired boxes (also covers a crashed worker).
        survivors: list[registry.Box] = []
        for b in boxes:
            reason = None
            if b.age(now) > config.POOL_MAX_LIFETIME:
                reason = "max-lifetime"
            elif b.status == "booting" and b.age(now) > config.POOL_BOOT_DEADLINE:
                reason = "boot-timeout"
            elif b.status == "ready" and b.heartbeat_age(now) > config.POOL_HEARTBEAT_DEADLINE:
                reason = "stale-heartbeat"
            if reason:
                await self._destroy(b, reason)
            else:
                survivors.append(b)

        # Keep every live box's SSH link up (booting boxes need it to report in).
        if self.links:
            self.links.sync(survivors)

        # 2. Reap orphans not tracked in the registry at all.
        await self._reap_orphans(known_ids)

        # 3. Compute demand -> desired.
        demand = await _gpu_demand(self.temporal, survivors)
        if config.POOL_PAUSED:
            desired = 0
        else:
            desired = min(config.POOL_MAX_BOXES,
                          math.ceil(demand / max(1, config.POOL_JOBS_PER_BOX)))

        alive = len(survivors)   # booting + ready, all still healthy
        _log(f"demand={demand} desired={desired} fleet={alive} "
             f"(ready={sum(b.status == 'ready' for b in survivors)}, "
             f"booting={sum(b.status == 'booting' for b in survivors)})")

        # 4. Scale down: destroy the longest-idle ready boxes while we're over
        #    desired (warm pool keeps the rest). Normally a box must have been idle
        #    past POOL_IDLE_TIMEOUT; when PAUSED we drain every idle box at once
        #    (a busy box is still never killed — it's reaped once its job ends).
        if alive > desired:
            min_idle = 0 if config.POOL_PAUSED else config.POOL_IDLE_TIMEOUT
            idle = [b for b in survivors
                    if b.status == "ready" and b.active_count == 0
                    and b.idle_for(now) >= min_idle]
            idle.sort(key=lambda b: b.idle_for(now), reverse=True)
            for b in idle[:alive - desired]:
                await self._destroy(b, "paused-drain" if config.POOL_PAUSED
                                    else f"idle {b.idle_for(now)}s")
                alive -= 1

        # 5. Scale up: rent the shortfall.
        if desired > alive:
            await self._rent(desired - alive)

    async def run(self) -> None:
        _log(f"autoscaler up — queue={config.GPU_TASK_QUEUE} "
             f"max_boxes={config.POOL_MAX_BOXES} jobs/box={config.POOL_JOBS_PER_BOX} "
             f"idle_timeout={config.POOL_IDLE_TIMEOUT}s price<=${config.VAST_MAX_PRICE}/hr "
             f"link={config.BOX_LINK}"
             + ("  [PAUSED — draining to zero]" if config.POOL_PAUSED else ""))
        while True:
            try:
                await self.reconcile()
            except Exception as e:  # never let one bad loop kill the pool
                _log(f"reconcile error: {e!r}")
            await asyncio.sleep(config.POOL_RECONCILE_INTERVAL)


async def main() -> None:
    from temporalio.client import Client

    if not config.VAST_API_KEY:
        raise SystemExit("VAST_API_KEY is not set — the pool cannot rent GPUs.")

    temporal = await Client.connect(
        config.TEMPORAL_ADDRESS, namespace=config.TEMPORAL_NAMESPACE)
    r = redis.Redis.from_url(config.REDIS_URL)
    async with VastClient() as vast:
        await Pool(temporal, vast, r).run()


if __name__ == "__main__":
    asyncio.run(main())
