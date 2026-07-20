"""GPU-free mock trainer: same event interface as trainer.train, but fakes a
converging reconstruction so the whole UI can be developed/demoed anywhere.

It ignores the photos' content and "converges" a noisy point cloud onto a
colorful torus-knot, emitting progress, live JPEG frames rendered on CPU, and
real .ply snapshots (so the WebGL splat viewer shows something real).
"""
from __future__ import annotations

import io
import math
import os
import threading
import time

import numpy as np
from PIL import Image

from .splat_export import export_ply, rgb_to_sh0
from .train_common import EmitFn, TrainConfig


def _torus_knot(n: int, rng: np.random.Generator):
    t = rng.uniform(0, 2 * np.pi, n)
    p, q, r_tube = 2, 3, 0.35
    r = 1.0 + 0.5 * np.cos(q * t)
    base = np.stack([r * np.cos(p * t), r * np.sin(p * t), 0.5 * np.sin(q * t)], 1)
    # thicken into a tube
    offs = rng.normal(0, r_tube * 0.35, (n, 3))
    pts = base + offs
    hue = (t / (2 * np.pi) + 0.5 * rng.normal(0, 0.02, n)) % 1.0
    cols = np.stack([
        0.5 + 0.5 * np.cos(2 * np.pi * (hue + s)) for s in (0.0, 1 / 3, 2 / 3)
    ], 1)
    return pts.astype(np.float32), np.clip(cols, 0, 1).astype(np.float32)


def _render_points(pts, cols, angle: float, w=480, h=360) -> np.ndarray:
    """Tiny CPU point-splat renderer: rotate, project, z-sort, paint 3x3 dots."""
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)
    tilt = math.radians(-20)
    ct, st = math.cos(tilt), math.sin(tilt)
    Rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]], np.float32)
    p = pts @ (Rx @ R).T
    p[:, 2] += 4.0                                    # push in front of camera
    f = 0.9 * h
    x = (p[:, 0] / p[:, 2] * f + w / 2).astype(np.int32)
    y = (p[:, 1] / p[:, 2] * f + h / 2).astype(np.int32)
    ok = (p[:, 2] > 0.1) & (x >= 1) & (x < w - 1) & (y >= 1) & (y < h - 1)
    x, y, z, cc = x[ok], y[ok], p[ok, 2], cols[ok]
    order = np.argsort(-z)                            # far → near
    img = np.full((h, w, 3), 18, np.float32)
    xs, ys, cs = x[order], y[order], cc[order] * 255
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            img[ys + dy, xs + dx] = cs
    return img.astype(np.uint8)


def train(
    dataset_dir: str,          # unused (kept for signature parity)
    out_dir: str,
    cfg: TrainConfig,
    emit: EmitFn,
    stop: threading.Event,
) -> None:
    rng = np.random.default_rng(7)
    max_steps = cfg.max_steps

    emit({"type": "status", "stage": "init", "detail": "mock mode — no GPU used"})

    n0 = 4000
    target_pts, target_cols = _torus_knot(30000, rng)
    idx = rng.choice(30000, n0, replace=False)

    t0 = time.time()
    step = 0
    for step in range(1, max_steps + 1):
        if stop.is_set():
            break
        prog = step / max_steps
        # densify: grow the active set over time
        n_now = int(n0 + (len(target_pts) - n0) * min(1.0, prog * 1.6))
        if n_now > len(idx):
            extra = rng.choice(30000, n_now - len(idx), replace=False)
            idx = np.concatenate([idx, extra])
        noise_sigma = 1.2 * math.exp(-4.5 * prog)
        pts = target_pts[idx] + rng.normal(0, noise_sigma, (len(idx), 3)).astype(np.float32)
        cols = target_cols[idx] * (0.35 + 0.65 * min(1.0, prog * 2)) \
            + 0.4 * (1 - min(1.0, prog * 2))

        loss = 0.5 * math.exp(-5 * prog) + 0.02 + rng.normal(0, 0.004)
        psnr = 12 + 18 * (1 - math.exp(-4 * prog)) + rng.normal(0, 0.15)

        if step % 5 == 0 or step == 1:
            emit({"type": "progress", "step": step, "max_steps": max_steps,
                  "loss": max(0.0, float(loss)), "psnr": float(psnr),
                  "num_gaussians": int(len(idx)),
                  "sh_degree": min(step // cfg.sh_degree_interval, cfg.sh_degree)})

        if step % cfg.frame_every == 0 or step == 1:
            frame = _render_points(pts, np.clip(cols, 0, 1),
                                   angle=time.time() - t0)
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, "JPEG", quality=80)
            emit({"type": "frame", "step": step, "jpeg": buf.getvalue()})

        if step % cfg.snapshot_every == 0 or step == max_steps:
            path = _save_ply(out_dir, step, pts, np.clip(cols, 0, 1),
                             noise_sigma, rng)
            emit({"type": "snapshot", "step": step, "path": path})

        time.sleep(0.03)  # ~30 steps/sec so the demo feels like training

    final = _save_ply(out_dir, step, target_pts[idx], target_cols[idx], 0.01, rng)
    emit({"type": "done", "step": step, "path": final})


def _save_ply(out_dir, step, pts, cols, sigma, rng) -> str:
    n = len(pts)
    path = os.path.join(out_dir, "snapshots", f"step_{step:06d}.ply")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scales = np.full((n, 3), 0.02 + sigma * 0.05, np.float32) \
        * rng.uniform(0.6, 1.4, (n, 3)).astype(np.float32)
    quats = rng.normal(size=(n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    export_ply(
        path,
        pts.astype(np.float32),
        scales,
        quats,
        np.full(n, 0.85, np.float32),
        rgb_to_sh0(cols.astype(np.float32)),
    )
    return path
