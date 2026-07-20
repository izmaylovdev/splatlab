# SplatLab — train 3D Gaussian Splats with a live web UI

Upload a folder of photos → camera poses are computed with COLMAP → a 3D Gaussian
Splatting model is trained with [gsplat](https://github.com/nerfstudio-project/gsplat)
while you watch it converge in the browser:

- **Live render stream** — the trainer renders a held-out / orbiting camera every few
  iterations and streams JPEG frames over a WebSocket.
- **Interactive 3D viewer** — periodic `.ply` snapshots of the model are loaded into an
  in-browser WebGL Gaussian-splat renderer you can orbit/zoom while training runs.
- **Loss & Gaussian-count charts** updating in real time.
- **Mock mode** — develop / demo the UI on any machine with no GPU.

```
splatlab/
├── backend/
│   ├── main.py            FastAPI app: REST + WebSocket + static frontend
│   ├── projects.py        project store & job manager (threads, event bus)
│   ├── colmap_runner.py   photos → COLMAP poses (pycolmap or colmap CLI)
│   ├── colmap_io.py       parser for COLMAP binary/text sparse models
│   ├── dataset.py         COLMAP model → training tensors
│   ├── trainer.py         real gsplat training loop (CUDA)
│   ├── mock_trainer.py    GPU-free fake trainer (same event interface)
│   └── splat_export.py    export Gaussians → standard 3DGS .ply
├── frontend/
│   ├── index.html         single-file UI (charts, live view, controls)
│   └── vendor/            three.js + GaussianSplats3D (vendored — works offline)
├── requirements.txt
└── data/                  created at runtime: one folder per project
```

## Setup

Python 3.10–3.12 recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### Real training (NVIDIA GPU)

1. Install CUDA-enabled PyTorch (pick your CUDA version on https://pytorch.org):

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

2. Install gsplat. On Windows the easy path is a prebuilt wheel
   (see https://docs.gsplat.studio/main/installation.html); building from source
   requires the CUDA toolkit + Visual Studio Build Tools. On Linux/WSL:

   ```bash
   pip install gsplat
   ```

   > **Windows tip:** everything also works great under WSL2 (Ubuntu), which is often
   > the least painful way to get gsplat + CUDA compiling.

3. COLMAP: `pycolmap` (installed via requirements.txt) is used by default.
   If you prefer the native COLMAP CLI (often faster, can use the GPU for SIFT),
   install it from https://colmap.github.io and make sure `colmap` is on your PATH —
   it will be picked automatically.

### Check your setup

```bash
python -m backend.check
```

Prints what was found: CUDA device, gsplat, pycolmap / colmap CLI.

## Run

```bash
python -m backend.main            # real mode (needs GPU for training)
python -m backend.main --mock     # UI demo mode, no GPU needed
```

Open http://localhost:8000

## Workflow

1. **New project** → give it a name.
2. **Upload photos** — pick a folder (30–300 photos of a scene/object with good
   overlap; avoid motion blur; keep exposure fixed if possible).
3. **Start training** — poses are computed first (this can take minutes),
   then training starts. Watch the live view, loss chart, and the interactive
   3D viewer (updates every snapshot interval).
4. **Export** — download the latest `.ply` (standard 3DGS format, opens in
   SuperSplat, Polycam viewer, gsplat viewers, etc.).

## Notes

- Training parameters (iterations, snapshot cadence, SH degree, densification)
  are in the "Settings" panel per project; defaults are sane for ~100-photo scenes.
- Everything the server writes lives under `data/<project-id>/`:
  `photos/`, `colmap/`, `snapshots/`, `checkpoint.pt`.
- The WebSocket protocol (for building your own client) is documented at the top
  of `backend/projects.py`.
