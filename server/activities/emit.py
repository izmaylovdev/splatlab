"""The `emit` callback, reimplemented over Redis + storage.

`shared.trainer.train` / `shared.colmap_runner.run_colmap` are untouched — they
still call `emit({...})` / `progress(stage, detail)` exactly as before. Only the
closure behind them changed: instead of pushing to an in-process asyncio.Queue it
(a) uploads finished artifacts to storage and rewrites their local path to a
public URL, then (b) publishes to the project's Redis channel (live fan-out) and
appends `progress` events to a capped Redis Stream (reconnect backlog).
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable

import redis

from .. import config
from ..shared.events import channel, frame_to_wire, history_key
from ..shared.storage import get_storage

EmitFn = Callable[[dict[str, Any]], None]


def make_redis_emitter(pid: str, run_id: str) -> EmitFn:
    r = redis.Redis.from_url(config.REDIS_URL)
    storage = get_storage()
    proj_dir = storage.project_dir(pid)

    def emit(msg: dict[str, Any]) -> None:
        t = msg.get("type")
        if t == "frame":
            wire = frame_to_wire(msg)
        elif t in ("snapshot", "done") and "path" in msg:
            rel = os.path.relpath(msg["path"], proj_dir).replace(os.sep, "/")
            storage.upload_artifact(pid, rel)               # no-op on local disk
            wire = {k: v for k, v in msg.items() if k != "path"}
            wire["url"] = storage.public_url(pid, rel)
        else:
            wire = msg

        payload = json.dumps(wire)
        pipe = r.pipeline()
        pipe.publish(channel(pid), payload)
        if t == "progress":
            pipe.xadd(history_key(pid), {"data": payload},
                      maxlen=config.HISTORY_MAXLEN, approximate=True)
            pipe.expire(history_key(pid), config.HISTORY_TTL_SECONDS)
        pipe.execute()

    return emit
