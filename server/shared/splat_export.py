"""Export Gaussians to the standard 3DGS .ply layout (Inria convention),
readable by GaussianSplats3D, SuperSplat, most splat viewers."""
from __future__ import annotations

import numpy as np

SH_C0 = 0.28209479177387814


def export_ply(
    path: str,
    means: np.ndarray,       # [N,3]
    scales: np.ndarray,      # [N,3]  (linear, not log)
    quats: np.ndarray,       # [N,4]  wxyz
    opacities: np.ndarray,   # [N]    (linear 0..1)
    sh0: np.ndarray,         # [N,3]  DC SH coefficients
    shN: np.ndarray | None = None,  # [N,K,3] higher-order SH (optional)
) -> None:
    n = means.shape[0]
    props = ["x", "y", "z", "nx", "ny", "nz",
             "f_dc_0", "f_dc_1", "f_dc_2"]
    n_rest = 0 if shN is None else shN.shape[1] * 3
    props += [f"f_rest_{i}" for i in range(n_rest)]
    props += ["opacity", "scale_0", "scale_1", "scale_2",
              "rot_0", "rot_1", "rot_2", "rot_3"]

    header = "\n".join(
        ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
        + [f"property float {p}" for p in props]
        + ["end_header", ""]
    )

    eps = 1e-9
    data = np.zeros((n, len(props)), dtype=np.float32)
    data[:, 0:3] = means
    # normals left zero
    data[:, 6:9] = sh0
    col = 9
    if shN is not None:
        # Inria layout: f_rest is [K coeffs of R..., K of G..., K of B...]
        data[:, col:col + n_rest] = shN.transpose(0, 2, 1).reshape(n, n_rest)
        col += n_rest
    # Clamp opacity away from 0/1 *in float64*: 1 - 1e-9 rounds to exactly 1.0
    # in float32, so a naive clip still leaves opacity==1 -> logit == +inf, and
    # those inf values pollute the .ply and break strict web viewers
    # (GaussianSplats3D poisons its GPU sort to NaN -> nothing renders). A 1e-6
    # margin keeps the logit finite (|logit| <= ~13.8) and is visually identical.
    op = np.clip(opacities.astype(np.float64), 1e-6, 1.0 - 1e-6)
    data[:, col] = np.log(op / (1 - op)).astype(np.float32)     # stored as logit
    data[:, col + 1:col + 4] = np.log(np.maximum(scales, eps))  # stored as log
    data[:, col + 4:col + 8] = quats

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def rgb_to_sh0(rgb: np.ndarray) -> np.ndarray:
    """[0,1] RGB → DC SH coefficient."""
    return (rgb - 0.5) / SH_C0


def export_points_ply(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write a sparse SfM point cloud as a 3DGS ``.ply`` for *review*.

    Each SfM point becomes a tiny, opaque, isotropic Gaussian coloured by its
    SfM RGB, so the same GaussianSplats3D viewer that shows training snapshots
    can display the raw reconstruction the user is asked to sign off on. The
    point size is a small fraction of the scene extent so dense scenes stay
    crisp and sparse ones stay visible.

    xyz: [N,3] float, rgb: [N,3] uint8 (COLMAP points3D output).
    """
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb, dtype=np.float32).reshape(-1, 3) / 255.0
    n = xyz.shape[0]
    if n == 0:
        raise ValueError("no SfM points to export")

    # Point radius: ~0.15% of the bounding-box diagonal (robust to scene scale),
    # with a tiny floor so a degenerate cloud still renders.
    diag = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
    radius = max(diag * 0.0015, 1e-4)

    scales = np.full((n, 3), radius, dtype=np.float32)
    quats = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1))  # identity
    opacities = np.full((n,), 0.9, dtype=np.float32)
    sh0 = rgb_to_sh0(rgb)

    export_ply(path, means=xyz, scales=scales, quats=quats,
               opacities=opacities, sh0=sh0, shN=None)
