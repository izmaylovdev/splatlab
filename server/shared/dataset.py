"""COLMAP undistorted output → training dataset (numpy; torch conversion in trainer)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image as PILImage

from .colmap_io import qvec2rotmat, read_model


@dataclass
class SplatDataset:
    image_paths: list[str]
    # world-to-camera 4x4 matrices, OpenCV convention (x right, y down, z forward)
    viewmats: np.ndarray          # [N, 4, 4] float32
    Ks: np.ndarray                # [N, 3, 3] float32 (per image, scaled)
    sizes: np.ndarray             # [N, 2] int  (width, height)
    points: np.ndarray            # [P, 3] float32 sparse SfM points
    points_rgb: np.ndarray        # [P, 3] uint8
    scene_scale: float = 1.0
    scene_center: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))

    def __len__(self) -> int:
        return len(self.image_paths)


def load_colmap_dataset(undistorted_dir: str, max_side: int = 1600) -> SplatDataset:
    """Load an undistorted COLMAP dataset. Images larger than max_side are
    flagged for downscale (the trainer resizes on load; K is pre-scaled here)."""
    sparse_dir = os.path.join(undistorted_dir, "sparse")
    if os.path.isdir(os.path.join(sparse_dir, "0")):
        sparse_dir = os.path.join(sparse_dir, "0")
    image_dir = os.path.join(undistorted_dir, "images")

    cameras, images, (points, points_rgb) = read_model(sparse_dir)

    paths, viewmats, Ks, sizes = [], [], [], []
    for img in sorted(images.values(), key=lambda i: i.name):
        path = os.path.join(image_dir, img.name)
        if not os.path.exists(path):
            continue
        cam = cameras[img.camera_id]
        if cam.model == "SIMPLE_PINHOLE":
            f, cx, cy = cam.params[:3]
            fx = fy = f
        elif cam.model == "PINHOLE":
            fx, fy, cx, cy = cam.params[:4]
        else:
            raise ValueError(
                f"Camera model {cam.model} after undistortion — expected PINHOLE"
            )
        scale = min(1.0, max_side / max(cam.width, cam.height))
        w, h = round(cam.width * scale), round(cam.height * scale)
        # exact scale factors after rounding
        sx, sy = w / cam.width, h / cam.height
        K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]],
                     dtype=np.float32)

        R = qvec2rotmat(img.qvec)
        t = img.tvec
        vm = np.eye(4, dtype=np.float32)
        vm[:3, :3] = R
        vm[:3, 3] = t
        paths.append(path)
        viewmats.append(vm)
        Ks.append(K)
        sizes.append((w, h))

    if len(paths) < 3:
        raise RuntimeError("Too few registered images to train")

    viewmats = np.stack(viewmats)
    cam_centers = -np.einsum("nij,nj->ni", viewmats[:, :3, :3].transpose(0, 2, 1),
                             viewmats[:, :3, 3])
    center = cam_centers.mean(0)
    scene_scale = float(np.linalg.norm(cam_centers - center, axis=1).max()) * 1.1

    return SplatDataset(
        image_paths=paths,
        viewmats=viewmats,
        Ks=np.stack(Ks),
        sizes=np.array(sizes, dtype=np.int64),
        points=points.astype(np.float32),
        points_rgb=points_rgb,
        scene_scale=max(scene_scale, 1e-6),
        scene_center=center.astype(np.float32),
    )


def load_image(path: str, size: tuple[int, int]) -> np.ndarray:
    """Load an image as float32 [H, W, 3] in [0, 1], resized to (w, h)."""
    img = PILImage.open(path).convert("RGB")
    if img.size != tuple(size):
        img = img.resize(tuple(size), PILImage.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0
