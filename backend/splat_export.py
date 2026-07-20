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
    op = np.clip(opacities, eps, 1 - eps)
    data[:, col] = np.log(op / (1 - op))          # stored as logit
    data[:, col + 1:col + 4] = np.log(np.maximum(scales, eps))  # stored as log
    data[:, col + 4:col + 8] = quats

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def rgb_to_sh0(rgb: np.ndarray) -> np.ndarray:
    """[0,1] RGB → DC SH coefficient."""
    return (rgb - 0.5) / SH_C0
