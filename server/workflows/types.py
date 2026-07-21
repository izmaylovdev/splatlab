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
    # Project-*relative* path (e.g. "colmap/undistorted") to the undistorted
    # dataset dir. Relative — not an absolute box-local path — because on the pool
    # the training activity may run on a *different* box than COLMAP did; the
    # producer uploads the subtree (upload_dir) and the consumer re-materializes
    # it (ensure_dir_local). The dataset can be many GB, so we pass this handle,
    # never the bytes.
    undistorted_rel: str


@dataclass
class TrainArgs:
    project_id: str
    config: dict[str, Any]
    undistorted_rel: str
    run_id: str


@dataclass
class TrainResult:
    final_step: int
    snapshot_uri: str | None
