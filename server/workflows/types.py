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

    ``phase`` selects what this run executes:
      "pipeline" — COLMAP, then (unless config["auto_train"]) wait for the user
                   to approve the reviewed points, then train. One run end to end.
      "train"    — assume poses are ready (from an earlier run) and train only;
                   used to re-train with new settings without redoing SFM.
      "sfm"      — legacy, from before "pipeline" existed: COLMAP, then the
                   same review -> train flow (runs that completed after COLMAP
                   alone replay unchanged).
    Same workflow id per project, so runs never overlap and the id is reusable
    once a run completes.
    """
    project_id: str
    config: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    phase: str = "sfm"
    resume: bool = False      # "train" only: continue from the latest snapshot


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
