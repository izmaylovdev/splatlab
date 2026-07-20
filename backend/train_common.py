"""Shared config + event interface for real and mock trainers."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable

# Trainers emit JSON-serializable dicts:
#   {"type": "status",   "stage": str, "detail": str}
#   {"type": "progress", "step", "max_steps", "loss", "psnr", "num_gaussians", "sh_degree"}
#   {"type": "frame",    "step", "jpeg": <bytes>}          (server base64-encodes)
#   {"type": "snapshot", "step", "path": str}              (relative URL path of .ply)
#   {"type": "done",     "step", "path": str}
#   {"type": "error",    "message": str}
EmitFn = Callable[[dict[str, Any]], None]


@dataclass
class TrainConfig:
    max_steps: int = 15000
    sh_degree: int = 3
    sh_degree_interval: int = 1000     # raise active SH degree every N steps
    ssim_lambda: float = 0.2
    init_opacity: float = 0.1
    max_image_side: int = 1600
    # cadence
    frame_every: int = 20              # live-view render every N steps
    snapshot_every: int = 500          # .ply snapshot every N steps
    # densification (gsplat DefaultStrategy)
    refine_start: int = 500
    refine_stop: int = 12000
    refine_every: int = 100
    grow_grad2d: float = 0.0002
    # camera pose stage
    colmap_matcher: str = "auto"       # auto | exhaustive | sequential

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
