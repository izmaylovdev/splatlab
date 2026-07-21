"""Real 3D Gaussian Splatting trainer built on gsplat (CUDA required).

Structure follows gsplat's simple_trainer: Adam per parameter group +
DefaultStrategy for densification/pruning.
"""
from __future__ import annotations

import io
import math
import os
import threading

import numpy as np

from .dataset import SplatDataset, load_image
from .splat_export import export_ply
from .train_common import EmitFn, TrainConfig


def train(
    dataset: SplatDataset,
    out_dir: str,
    cfg: TrainConfig,
    emit: EmitFn,
    stop: threading.Event,
) -> None:
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not available. Real training needs an NVIDIA GPU — "
            "run the server with --mock to demo the UI."
        )
    device = "cuda"
    torch.manual_seed(42)

    emit({"type": "status", "stage": "init", "detail": "initializing gaussians"})

    # ---- init gaussians from SfM points -------------------------------------
    pts = torch.from_numpy(dataset.points).float()
    rgb = torch.from_numpy(dataset.points_rgb).float() / 255.0
    n = pts.shape[0]
    if n < 100:
        raise RuntimeError(f"Only {n} SfM points — reconstruction too sparse")

    # scale init: mean distance to 3 nearest neighbours (chunked knn on GPU)
    pts_gpu = pts.to(device)
    dists = []
    for i in range(0, n, 4096):
        d = torch.cdist(pts_gpu[i:i + 4096], pts_gpu)          # [c, N]
        knn = d.topk(4, largest=False).values[:, 1:]           # skip self
        dists.append(knn.mean(1))
    avg_d = torch.cat(dists).clamp_min(1e-7).cpu()

    sh0 = ((rgb - 0.5) / 0.28209479177387814).unsqueeze(1)          # [N,1,3]
    n_rest = (cfg.sh_degree + 1) ** 2 - 1

    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(pts),
        "scales": torch.nn.Parameter(torch.log(avg_d)[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(
            torch.cat([torch.ones(n, 1), torch.zeros(n, 3)], 1)),
        "opacities": torch.nn.Parameter(
            torch.logit(torch.full((n,), cfg.init_opacity))),
        "sh0": torch.nn.Parameter(sh0),
        "shN": torch.nn.Parameter(torch.zeros(n, n_rest, 3)),
    }).to(device)

    scene_scale = dataset.scene_scale
    lrs = {
        "means": 1.6e-4 * scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20,
    }
    optimizers = {
        k: torch.optim.Adam([{"params": [params[k]], "lr": lrs[k], "name": k}],
                            eps=1e-15)
        for k in params
    }
    means_sched = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / cfg.max_steps))

    strategy = DefaultStrategy(
        refine_start_iter=cfg.refine_start,
        refine_stop_iter=cfg.refine_stop,
        refine_every=cfg.refine_every,
        grow_grad2d=cfg.grow_grad2d,
        verbose=False,
    )
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    viewmats_all = torch.from_numpy(dataset.viewmats).float().to(device)
    Ks_all = torch.from_numpy(dataset.Ks).float().to(device)

    # simple in-memory image cache (small scenes) with lazy load
    cache: dict[int, torch.Tensor] = {}

    def gt_image(i: int) -> torch.Tensor:
        if i not in cache:
            w, h = int(dataset.sizes[i][0]), int(dataset.sizes[i][1])
            arr = load_image(dataset.image_paths[i], (w, h))
            t = torch.from_numpy(arr)
            # keep at most ~2GB in RAM; beyond that, don't cache
            if len(cache) * t.numel() * 4 < 2e9:
                cache[i] = t
            else:
                return t.to(device)
        return cache[i].to(device)

    # ---- ssim ---------------------------------------------------------------
    def _gauss_kernel(size=11, sigma=1.5):
        c = torch.arange(size).float() - size // 2
        g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).to(device)
        return (g[:, None] @ g[None, :])[None, None].repeat(3, 1, 1, 1)

    _win = _gauss_kernel()

    def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a, b: [1,3,H,W] in [0,1]
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        mu_a = F.conv2d(a, _win, padding=5, groups=3)
        mu_b = F.conv2d(b, _win, padding=5, groups=3)
        mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
        s_a = F.conv2d(a * a, _win, padding=5, groups=3) - mu_a2
        s_b = F.conv2d(b * b, _win, padding=5, groups=3) - mu_b2
        s_ab = F.conv2d(a * b, _win, padding=5, groups=3) - mu_ab
        return (((2 * mu_ab + c1) * (2 * s_ab + c2)) /
                ((mu_a2 + mu_b2 + c1) * (s_a + s_b + c2))).mean()

    def render(cam_i: int, sh_degree: int, scale: float = 1.0):
        K = Ks_all[cam_i].clone()
        w, h = int(dataset.sizes[cam_i][0]), int(dataset.sizes[cam_i][1])
        if scale != 1.0:
            K = K * scale
            K[2, 2] = 1.0
            w, h = max(8, int(w * scale)), max(8, int(h * scale))
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        return rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors,
            viewmats=viewmats_all[cam_i][None],
            Ks=K[None],
            width=w,
            height=h,
            sh_degree=sh_degree,
            packed=False,
        )

    def save_snapshot(step: int) -> str:
        with torch.no_grad():
            path = os.path.join(out_dir, "snapshots", f"step_{step:06d}.ply")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            export_ply(
                path,
                params["means"].detach().cpu().numpy(),
                torch.exp(params["scales"]).detach().cpu().numpy(),
                torch.nn.functional.normalize(params["quats"], dim=1)
                    .detach().cpu().numpy(),
                torch.sigmoid(params["opacities"]).detach().cpu().numpy(),
                params["sh0"].detach().cpu().numpy()[:, 0, :],
                params["shN"].detach().cpu().numpy(),
            )
        return path

    # ---- loop ---------------------------------------------------------------
    emit({"type": "status", "stage": "training", "detail": f"{n} initial gaussians"})
    perm = np.random.permutation(len(dataset))
    ptr = 0
    for step in range(1, cfg.max_steps + 1):
        if stop.is_set():
            break
        if ptr >= len(perm):
            perm, ptr = np.random.permutation(len(dataset)), 0
        cam_i = int(perm[ptr]); ptr += 1

        active_sh = min(step // cfg.sh_degree_interval, cfg.sh_degree)
        renders, alphas, info = render(cam_i, active_sh)
        pred = renders[0].clamp(0, 1)                       # [H,W,3]
        gt = gt_image(cam_i)                                # [H,W,3]

        l1 = (pred - gt).abs().mean()
        p4 = pred.permute(2, 0, 1)[None]
        g4 = gt.permute(2, 0, 1)[None]
        loss = (1 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * (1 - ssim(p4, g4))

        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        loss.backward()
        strategy.step_post_backward(params, optimizers, strategy_state, step, info,
                                    packed=False)
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_sched.step()

        if step % 10 == 0 or step == 1:
            with torch.no_grad():
                mse = ((pred - gt) ** 2).mean().item()
                psnr = -10 * math.log10(max(mse, 1e-10))
            emit({"type": "progress", "step": step, "max_steps": cfg.max_steps,
                  "loss": float(loss.item()), "psnr": psnr,
                  "num_gaussians": int(params["means"].shape[0]),
                  "sh_degree": active_sh})

        if step % cfg.frame_every == 0 or step == 1:
            with torch.no_grad():
                view_i = step // cfg.frame_every % len(dataset)
                r, _, _ = render(view_i, active_sh, scale=0.35)
                img = (r[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                buf = io.BytesIO()
                Image.fromarray(img).save(buf, "JPEG", quality=80)
                emit({"type": "frame", "step": step, "jpeg": buf.getvalue()})

        if step % cfg.snapshot_every == 0:
            emit({"type": "snapshot", "step": step, "path": save_snapshot(step)})

    final_step = min(step, cfg.max_steps)
    path = save_snapshot(final_step)
    import torch as _t
    _t.save({k: v.detach().cpu() for k, v in params.items()},
            os.path.join(out_dir, "checkpoint.pt"))
    emit({"type": "done", "step": final_step, "path": path})
