"""Photos → camera poses.

Runs COLMAP (native CLI if on PATH, else pycolmap) on an image folder:
feature extraction → matching → incremental mapping → undistortion to a
PINHOLE model. Emits progress through a callback so the UI can show stages.

Output layout (under <project>/colmap/):
    database.db
    sparse/0/            raw reconstruction
    undistorted/
        images/          undistorted images used for training
        sparse/          PINHOLE sparse model used for training
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Callable

ProgressCb = Callable[[str, str], None]  # (stage, detail)


def _has_colmap_cli() -> bool:
    return shutil.which("colmap") is not None


def _run(cmd: list[str], progress: ProgressCb, stage: str) -> None:
    progress(stage, " ".join(cmd[:2]))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if line:
            tail.append(line)
            tail = tail[-30:]
            if len(tail) % 10 == 0:
                progress(stage, line[-160:])
    if proc.wait() != 0:
        raise RuntimeError(f"COLMAP failed at {stage}:\n" + "\n".join(tail[-15:]))


def run_colmap(
    image_dir: str,
    work_dir: str,
    progress: ProgressCb = lambda *_: None,
    matcher: str = "auto",
    use_gpu: bool = True,
) -> str:
    """Returns the path to the undistorted dataset dir (images/ + sparse/)."""
    os.makedirs(work_dir, exist_ok=True)
    db = os.path.join(work_dir, "database.db")
    sparse = os.path.join(work_dir, "sparse")
    undist = os.path.join(work_dir, "undistorted")
    for p in (db,):
        if os.path.exists(p):
            os.remove(p)
    for p in (sparse, undist):
        shutil.rmtree(p, ignore_errors=True)
    os.makedirs(sparse, exist_ok=True)

    n_images = len([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"))
    ])
    if n_images < 3:
        raise RuntimeError(f"Need at least 3 photos, found {n_images}")
    if matcher == "auto":
        matcher = "exhaustive" if n_images <= 250 else "sequential"

    if _has_colmap_cli():
        _run_cli(image_dir, db, sparse, undist, matcher, use_gpu, progress)
    else:
        _run_pycolmap(image_dir, db, sparse, undist, matcher, progress)

    model_dir = os.path.join(undist, "sparse")
    # colmap image_undistorter puts the model directly in sparse/ (no /0)
    if os.path.isdir(os.path.join(model_dir, "0")):
        model_dir = os.path.join(model_dir, "0")
    if not os.path.exists(os.path.join(model_dir, "cameras.bin")) and \
       not os.path.exists(os.path.join(model_dir, "cameras.txt")):
        raise RuntimeError("COLMAP produced no reconstruction — check photo overlap/quality")
    progress("done", f"poses ready ({n_images} images)")
    return undist


def _run_cli(image_dir, db, sparse, undist, matcher, use_gpu, progress: ProgressCb):
    gpu = "1" if use_gpu else "0"
    _run([
        "colmap", "feature_extractor",
        "--database_path", db, "--image_path", image_dir,
        "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", gpu,
    ], progress, "features")
    matcher_cmd = "exhaustive_matcher" if matcher == "exhaustive" else "sequential_matcher"
    _run(["colmap", matcher_cmd, "--database_path", db,
          "--SiftMatching.use_gpu", gpu], progress, "matching")
    _run([
        "colmap", "mapper",
        "--database_path", db, "--image_path", image_dir,
        "--output_path", sparse,
    ], progress, "mapping")
    _run([
        "colmap", "image_undistorter",
        "--image_path", image_dir,
        "--input_path", os.path.join(sparse, "0"),
        "--output_path", undist,
        "--output_type", "COLMAP",
    ], progress, "undistort")


def _run_pycolmap(image_dir, db, sparse, undist, matcher, progress: ProgressCb):
    import pycolmap  # imported lazily so mock mode never needs it

    progress("features", "extracting SIFT features (pycolmap)")
    try:
        # newer pycolmap (>=0.6 / 4.x): camera_model lives in reader_options
        reader_options = pycolmap.ImageReaderOptions()
        reader_options.camera_model = "OPENCV"
        pycolmap.extract_features(
            db, image_dir,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader_options,
        )
    except TypeError:
        # older pycolmap accepted camera_model as a direct kwarg
        pycolmap.extract_features(
            db, image_dir,
            camera_model="OPENCV",
            camera_mode=pycolmap.CameraMode.SINGLE,
        )
    progress("matching", f"{matcher} matching")
    if matcher == "exhaustive":
        pycolmap.match_exhaustive(db)
    else:
        try:
            pycolmap.match_sequential(db)
        except AttributeError:
            pycolmap.match_exhaustive(db)
    progress("mapping", "incremental mapping (this is the slow part)")
    maps = pycolmap.incremental_mapping(db, image_dir, sparse)
    if not maps:
        raise RuntimeError("Mapping failed — no reconstruction. Check photo overlap.")
    # keep the largest reconstruction
    best_idx = max(maps, key=lambda k: maps[k].num_reg_images())
    progress("undistort", "undistorting images")
    pycolmap.undistort_images(undist, os.path.join(sparse, str(best_idx)), image_dir)
