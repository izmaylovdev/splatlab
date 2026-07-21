"""WebSocket telemetry: Redis (pub/sub + backlog stream) -> browser.

The wire protocol is unchanged from the monolith. On connect we send a `hello`
(project snapshot + recent progress backlog), then forward every live event
published to the project's Redis channel. Reconnects replay the backlog from a
capped Redis Stream, so a dropped WebSocket no longer loses the gap.
"""
from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis
from fastapi import WebSocket, WebSocketDisconnect

from .. import config
from ..shared.events import channel, history_key


async def _backlog(r: aioredis.Redis, pid: str) -> list[dict[str, Any]]:
    # newest-first, capped; reverse to chronological for the client
    entries = await r.xrevrange(history_key(pid), count=config.HISTORY_HELLO)
    out = []
    for _id, fields in reversed(entries):
        data = fields.get("data") or fields.get(b"data")
        if data is None:
            continue
        if isinstance(data, bytes):
            data = data.decode()
        try:
            out.append(json.loads(data))
        except json.JSONDecodeError:
            pass
    return out


async def serve(socket: WebSocket, pid: str, project: dict[str, Any]) -> None:
    await socket.accept()
    r = aioredis.from_url(config.REDIS_URL, decode_responses=True)
    pubsub = r.pubsub()
    try:
        await socket.send_json({
            "type": "hello",
            "project": project,
            "history": await _backlog(r, pid),
            "latest_snapshot": project.get("latest_snapshot"),
        })
        await pubsub.subscribe(channel(pid))
        async for message in pubsub.listen():
            if message.get("type") != "message":
                continue
            data = message["data"]
            if isinstance(data, bytes):
                data = data.decode()
            await socket.send_text(data)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        try:
            await pubsub.aclose()
        except Exception:
            pass
        try:
            await r.aclose()
        except Exception:
            pass
