"""Redis view of the GPU boxes the pool manages.

Two writers, one schema:

- The **pool** records a box the moment it rents it (`put_booting`), stamping the
  Vast instance id / offer / price, and removes it when it destroys it.
- The **GPU worker** running on that box marks itself `ready`, then heartbeats its
  liveness and in-flight activity count (`heartbeat` / `touch_active`).

The pool reads the merged picture to decide idle-reaping and to detect dead
boxes (stale heartbeat). Keeping this in Redis — which every box already reaches
for telemetry — avoids giving the pool SSH access to the boxes.

Key layout:
    splat:pool:boxes            SET of box ids we manage
    splat:pool:box:<box_id>     HASH of the fields below

`box_id` is a uuid the pool generates *before* renting (also used as the Vast
label), so the worker can report under it via the SPLATLAB_BOX_ID env var without
knowing its own Vast instance id.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

BOXES_SET = "splat:pool:boxes"


def box_key(box_id: str) -> str:
    return f"splat:pool:box:{box_id}"


def _now() -> int:
    return int(time.time())


@dataclass
class Box:
    box_id: str
    instance_id: int | None
    status: str            # "booting" | "ready"
    rented_at: int
    ready_at: int
    heartbeat: int
    last_active: int
    active_count: int
    dph: float
    gpu_name: str

    def idle_for(self, now: int | None = None) -> int:
        """Seconds this box has had zero in-flight activities (0 if busy)."""
        if self.active_count > 0:
            return 0
        return max(0, (now or _now()) - max(self.last_active, self.ready_at))

    def heartbeat_age(self, now: int | None = None) -> int:
        return (now or _now()) - self.heartbeat

    def age(self, now: int | None = None) -> int:
        return (now or _now()) - self.rented_at


def parse_box(box_id: str, h: dict[Any, Any]) -> Box:
    """Build a Box from a raw Redis hash (bytes-or-str keys tolerated)."""
    d = {(k.decode() if isinstance(k, bytes) else k):
         (v.decode() if isinstance(v, bytes) else v) for k, v in h.items()}

    def _i(key: str, default: int = 0) -> int:
        try:
            return int(d.get(key, default))
        except (TypeError, ValueError):
            return default

    inst = d.get("instance_id")
    return Box(
        box_id=box_id,
        instance_id=int(inst) if inst not in (None, "") else None,
        status=d.get("status", "booting"),
        rented_at=_i("rented_at"),
        ready_at=_i("ready_at"),
        heartbeat=_i("heartbeat"),
        last_active=_i("last_active"),
        active_count=_i("active_count"),
        dph=float(d.get("dph", 0) or 0),
        gpu_name=d.get("gpu_name", ""),
    )


# ---- pool side (rents/reaps; may be called from async via run_in_executor) --
def put_booting(r: Any, box_id: str, instance_id: int, *,
                offer_id: int, dph: float, gpu_name: str) -> None:
    now = _now()
    pipe = r.pipeline()
    pipe.sadd(BOXES_SET, box_id)
    pipe.hset(box_key(box_id), mapping={
        "box_id": box_id, "instance_id": instance_id, "status": "booting",
        "rented_at": now, "ready_at": 0, "heartbeat": 0, "last_active": now,
        "active_count": 0, "dph": dph, "gpu_name": gpu_name, "offer_id": offer_id,
    })
    pipe.execute()


def remove_box(r: Any, box_id: str) -> None:
    pipe = r.pipeline()
    pipe.srem(BOXES_SET, box_id)
    pipe.delete(box_key(box_id))
    pipe.execute()


def list_boxes(r: Any) -> list[Box]:
    ids = [b.decode() if isinstance(b, bytes) else b for b in r.smembers(BOXES_SET)]
    out: list[Box] = []
    for box_id in ids:
        h = r.hgetall(box_key(box_id))
        if h:
            out.append(parse_box(box_id, h))
        else:
            # dangling set member with no hash — clean it up
            r.srem(BOXES_SET, box_id)
    return out


# ---- worker side (the GPU box reports on itself) ---------------------------
def register_ready(r: Any, box_id: str) -> None:
    """Mark this box ready to take work. Also joins the set in case the box was
    started outside the pool (manual boot) — harmless if already present."""
    now = _now()
    pipe = r.pipeline()
    pipe.sadd(BOXES_SET, box_id)
    pipe.hset(box_key(box_id), mapping={
        "box_id": box_id, "status": "ready", "ready_at": now,
        "heartbeat": now, "last_active": now, "active_count": 0,
    })
    pipe.execute()


def heartbeat(r: Any, box_id: str, active_count: int) -> None:
    now = _now()
    mapping: dict[str, Any] = {"heartbeat": now, "active_count": active_count}
    if active_count > 0:
        mapping["last_active"] = now
    r.hset(box_key(box_id), mapping=mapping)


def touch_active(r: Any, box_id: str) -> None:
    """Bump last_active — call at each activity start and finish so a box that
    just went idle is timestamped from the moment its last job ended."""
    r.hset(box_key(box_id), "last_active", _now())
