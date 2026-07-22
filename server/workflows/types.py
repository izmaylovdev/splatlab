"""Serializable payloads passed between the workflow and its activities.

Plain dataclasses — Temporal's default data converter (de)serializes them as
JSON. Never put large binary/dataset blobs here: pass URIs/handles instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainParams:
    """Input to SplatTrainingWorkflow.

    ``phase`` selects which half of the pipeline this run executes:
      "sfm"   — run COLMAP, publish the reviewable sparse point cloud, then
                COMPLETE (releasing the GPU box while the user reviews).
      "train" — assume poses are ready (from a prior sfm run) and run training.
    Same workflow id per project, so the two phases run sequentially, never at
    once, and the id is reusable once a phase completes.
    """
    project_id: str
    config: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    phase: str = "sfm"


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
    # Reviewable sparse point cloud (a splat .ply) + coarse stats for the UI.
    points_rel: str = ""
    points_uri: str | None = None
    num_points: int = 0
    num_images: int = 0


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
