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


def _median_nn_spacing(xyz: np.ndarray, samples: int = 1000, chunk: int = 64) -> float:
    """Median distance from a point to its nearest neighbour, estimated on a
    random sample (brute force in chunks: memory ~chunk*N floats, no scipy)."""
    n = xyz.shape[0]
    if n < 2:
        return 0.0
    rng = np.random.default_rng(0)
    pick = xyz[rng.choice(n, min(samples, n), replace=False)]
    sq = (xyz * xyz).sum(1)
    nn = []
    for i in range(0, len(pick), chunk):
        p = pick[i:i + chunk]
        d2 = (p * p).sum(1)[:, None] + sq[None, :] - 2.0 * p @ xyz.T
        # column 0 of the partition is the point itself (distance ~0)
        nn.append(np.partition(d2, 1, axis=1)[:, 1])
    return float(np.sqrt(np.maximum(np.median(np.concatenate(nn)), 0.0)))


def export_points_ply(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write a sparse SfM point cloud as a 3DGS ``.ply`` for *review*.

    Each SfM point becomes a tiny, opaque, isotropic Gaussian coloured by its
    SfM RGB, so the same GaussianSplats3D viewer that shows training snapshots
    can display the raw reconstruction the user is asked to sign off on. The
    point size follows the typical spacing between neighbouring points, so
    points read as distinct dots at any scene scale instead of merging into fog.

    xyz: [N,3] float, rgb: [N,3] uint8 (COLMAP points3D output).
    """
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb, dtype=np.float32).reshape(-1, 3) / 255.0
    n = xyz.shape[0]
    if n == 0:
        raise ValueError("no SfM points to export")

    # Point radius: half the median nearest-neighbour spacing, so neighbours
    # just touch. Capped at ~0.15% of the bounding-box diagonal (the old rule,
    # fine for compact indoor captures) and floored so a degenerate cloud still
    # renders. The box spans the 1st-99th percentiles: outdoor/aerial clouds
    # have far-off points (horizon, sky mismatches) that would otherwise
    # inflate it ~10x. Box-only sizing made aerial points ~7x their spacing —
    # one blur that hid the scene.
    lo, hi = np.percentile(xyz, [1, 99], axis=0)
    diag = float(np.linalg.norm(hi - lo))
    radius = max(min(0.5 * _median_nn_spacing(xyz), diag * 0.0015), 1e-4)

    scales = np.full((n, 3), radius, dtype=np.float32)
    quats = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1))  # identity
    opacities = np.full((n,), 0.9, dtype=np.float32)
    sh0 = rgb_to_sh0(rgb)

    export_ply(path, means=xyz, scales=scales, quats=quats,
               opacities=opacities, sh0=sh0, shN=None)
