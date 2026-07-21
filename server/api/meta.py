"""Project metadata — the durable record the API owns (name/created/config).

Live status/stage come from the Temporal workflow, not from here; this is only
the slow-changing record, persisted through the storage backend so the API stays
stateless and works even when the worker lives on another machine.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from ..shared.train_common import TrainConfig


def new_project_meta(name: str) -> dict[str, Any]:
    pid = uuid.uuid4().hex[:12]
    return {
        "id": pid,
        "name": (name or f"project-{pid[:6]}")[:80],
        "created": time.time(),
        "status": "new",              # fallback only; live status is the workflow's
        "config": TrainConfig().to_dict(),
    }


def merge_config(meta: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    base = meta.get("config", {})
    if overrides:
        base = {**base, **overrides}
    return TrainConfig.from_dict(base).to_dict()
