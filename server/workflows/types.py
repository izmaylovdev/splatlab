"""Serializable payloads passed between the workflow and its activities.

Plain dataclasses — Temporal's default data converter (de)serializes them as
JSON. Never put large binary/dataset blobs here: pass URIs/handles instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainParams:
    """Input to SplatTrainingWorkflow."""
    project_id: str
    config: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""


@dataclass
class ColmapArgs:
    project_id: str
    matcher: str
    run_id: str


@dataclass
class ColmapResult:
    # Local path (on the GPU worker) to the undistorted dataset dir. The dataset
    # can be many GB, so we pass this handle, never the bytes.
    undistorted_uri: str


@dataclass
class TrainArgs:
    project_id: str
    config: dict[str, Any]
    undistorted_uri: str
    run_id: str


@dataclass
class TrainResult:
    final_step: int
    snapshot_uri: str | None
