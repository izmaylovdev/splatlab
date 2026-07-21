"""Redis event-bus naming + the wire-message transforms.

The live telemetry protocol is unchanged from the old in-process event bus:
server -> browser JSON messages of type
    hello | status | progress | frame | snapshot | done | error

Trainers/COLMAP still call ``emit({...})`` with the exact same dicts they always
did (see shared.train_common.EmitFn). Only the transport changed: instead of an
in-process asyncio.Queue, messages go onto a per-project Redis pub/sub channel
(live fan-out) and a capped Redis Stream (``progress`` backlog for reconnects).
"""
from __future__ import annotations

import base64
from typing import Any


def channel(pid: str) -> str:
    """Pub/sub channel a project's live events are published to."""
    return f"splat:events:{pid}"


def history_key(pid: str) -> str:
    """Capped Redis Stream holding recent ``progress`` events for replay."""
    return f"splat:history:{pid}"


def frame_to_wire(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw ``frame`` event (jpeg bytes) to its wire form (base64).

    This is exactly what the old ProjectManager._publish did. Non-frame messages
    pass through untouched.
    """
    if msg.get("type") == "frame" and "jpeg" in msg:
        return {"type": "frame", "step": msg["step"],
                "jpeg_b64": base64.b64encode(msg["jpeg"]).decode()}
    return msg
