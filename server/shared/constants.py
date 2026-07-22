"""Shared constants used by both the API service and the worker."""
from __future__ import annotations

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp")
VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".wmv", ".mpg", ".mpeg")

# ---- pipeline artifact layout (project-relative) ---------------------------
# The SFM (COLMAP) phase writes these; the train phase and API read them. Their
# presence in storage is the durable, stateless signal that "poses are ready to
# review" (points.ply) — independent of any live workflow.
UNDISTORTED_REL = "colmap/undistorted"   # PINHOLE dataset the trainer consumes
SFM_POINTS_REL = "colmap/points.ply"     # sparse SfM cloud, as a viewable splat .ply
SFM_META_REL = "colmap/sfm.json"         # {"num_points": int, "num_images": int}
