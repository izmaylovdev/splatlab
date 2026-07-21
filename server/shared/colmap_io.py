"""Minimal reader for COLMAP sparse reconstruction output (binary or text).

Trimmed adaptation of COLMAP's official scripts/python/read_write_model.py
(BSD licensed). Only what training needs: cameras, images (poses), 3D points.
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray  # model-specific


@dataclass
class Image:
    id: int
    qvec: np.ndarray  # rotation world->cam, wxyz
    tvec: np.ndarray  # translation world->cam
    camera_id: int
    name: str


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
MODEL_NAME_TO_ID = {name: (mid, n) for mid, (name, n) in CAMERA_MODELS.items()}


def qvec2rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def _read(fid, n, fmt):
    return struct.unpack("<" + fmt, fid.read(n))


def read_cameras_binary(path: str) -> dict[int, Camera]:
    cams: dict[int, Camera] = {}
    with open(path, "rb") as f:
        (num,) = _read(f, 8, "Q")
        for _ in range(num):
            cid, model_id, w, h = _read(f, 24, "iiQQ")
            name, n_params = CAMERA_MODELS[model_id]
            params = np.array(_read(f, 8 * n_params, "d" * n_params))
            cams[cid] = Camera(cid, name, int(w), int(h), params)
    return cams


def read_images_binary(path: str) -> dict[int, Image]:
    images: dict[int, Image] = {}
    with open(path, "rb") as f:
        (num,) = _read(f, 8, "Q")
        for _ in range(num):
            iid = _read(f, 4, "i")[0]
            qvec = np.array(_read(f, 32, "dddd"))
            tvec = np.array(_read(f, 24, "ddd"))
            cam_id = _read(f, 4, "i")[0]
            name = b""
            c = f.read(1)
            while c != b"\x00":
                name += c
                c = f.read(1)
            (n_pts,) = _read(f, 8, "Q")
            f.read(24 * n_pts)  # skip 2D points (x, y, point3D_id)
            images[iid] = Image(iid, qvec, tvec, cam_id, name.decode("utf-8"))
    return images


def read_points3d_binary(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (xyz [N,3] float64, rgb [N,3] uint8)."""
    xyzs, rgbs = [], []
    with open(path, "rb") as f:
        (num,) = _read(f, 8, "Q")
        for _ in range(num):
            _read(f, 8, "Q")  # point id
            xyz = _read(f, 24, "ddd")
            rgb = _read(f, 3, "BBB")
            _read(f, 8, "d")  # error
            (track_len,) = _read(f, 8, "Q")
            f.read(8 * track_len)  # skip track (image_id, point2D_idx)
            xyzs.append(xyz)
            rgbs.append(rgb)
    return np.array(xyzs, dtype=np.float64), np.array(rgbs, dtype=np.uint8)


def _detect(sparse_dir: str, stem: str) -> str:
    for ext in (".bin", ".txt"):
        p = os.path.join(sparse_dir, stem + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{stem}.bin/.txt not found in {sparse_dir}")


def read_cameras_text(path: str) -> dict[int, Camera]:
    cams: dict[int, Camera] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            el = line.split()
            cams[int(el[0])] = Camera(
                int(el[0]), el[1], int(el[2]), int(el[3]),
                np.array([float(x) for x in el[4:]]),
            )
    return cams


def read_images_text(path: str) -> dict[int, Image]:
    images: dict[int, Image] = {}
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    for meta in lines[::2]:  # every other line is the 2D point list
        el = meta.split()
        images[int(el[0])] = Image(
            int(el[0]),
            np.array([float(x) for x in el[1:5]]),
            np.array([float(x) for x in el[5:8]]),
            int(el[8]),
            el[9],
        )
    return images


def read_points3d_text(path: str) -> tuple[np.ndarray, np.ndarray]:
    xyzs, rgbs = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            el = line.split()
            xyzs.append([float(x) for x in el[1:4]])
            rgbs.append([int(x) for x in el[4:7]])
    return np.array(xyzs, dtype=np.float64), np.array(rgbs, dtype=np.uint8)


def read_model(sparse_dir: str):
    """Read a COLMAP sparse model directory (auto-detects .bin / .txt)."""
    cam_path = _detect(sparse_dir, "cameras")
    img_path = _detect(sparse_dir, "images")
    pts_path = _detect(sparse_dir, "points3D")
    if cam_path.endswith(".bin"):
        return (
            read_cameras_binary(cam_path),
            read_images_binary(img_path),
            read_points3d_binary(pts_path),
        )
    return (
        read_cameras_text(cam_path),
        read_images_text(img_path),
        read_points3d_text(pts_path),
    )
