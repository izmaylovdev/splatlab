# SplatLab — train 3D Gaussian Splats with a live web UI

Upload a folder of photos → camera poses are computed with COLMAP → a 3D Gaussian
Splatting model is trained with [gsplat](https://github.com/nerfstudio-project/gsplat)
while you watch it converge in the browser:

- **Live render stream** — the trainer renders a held-out / orbiting camera every few
  iterations and streams JPEG frames over a WebSocket.
- **Interactive 3D viewer** — periodic `.ply` snapshots of the model are loaded into an
  in-browser WebGL Gaussian-splat renderer you can orbit/zoom while training runs.
- **Loss & Gaussian-count charts** updating in real time.
- **Durable jobs** — the COLMAP→train→export pipeline runs as a [Temporal](https://temporal.io)
  workflow on a **separate GPU worker**, so a control-plane restart never loses an
  in-flight training run, and the GPU box can live anywhere (e.g. vast.ai).

## Architecture

Three tiers, one repo. The **client** is a static app; the **API service** is the
control plane (no GPU) that starts/cancels/queries Temporal workflows and serves
telemetry; the **worker** (GPU box) runs the actual pipeline as a workflow.

```
client (static)  ──REST+WS──►  API service  ──►  Temporal  ──►  worker (GPU)
                                    ▲  subscribe        dispatch      │ emit
                                    └──────────────  Redis  ◄─────────┘
                                snapshots/photos ──►  S3 / MinIO
```

```
splatlab/
├── client/               static UI (was frontend/); config.js sets the API/WS base
│   ├── index.html        single-file UI (charts, live view, controls)
│   └── vendor/           three.js + GaussianSplats3D (vendored — works offline)
├── server/
│   ├── config.py         env-driven settings (Temporal, Redis, storage)
│   ├── api/              FastAPI control plane: REST + WebSocket + /files
│   ├── worker/           Temporal worker entrypoint (runs on the GPU box)
│   ├── workflows/        SplatTrainingWorkflow (durable orchestration)
│   ├── activities/       run_colmap / train activities + Redis emitter
│   └── shared/           trainer, colmap_runner, dataset, storage, events, …
├── deploy/
│   ├── docker-compose.yml  control plane: Temporal + Postgres + Redis + MinIO
│   ├── run_api.sh          launch the API service
│   ├── run_worker.sh       launch the GPU worker
│   └── vast_setup.sh       one-shot GPU-box (vast.ai) bootstrap → worker
├── requirements.txt
└── data/                 local scratch / artifact store (local backend)
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
python -m server.shared.check
```

Prints what was found: CUDA device, gsplat, pycolmap / colmap CLI.

## Run

**1. Control plane** (Temporal + Redis; MinIO too if using the S3 backend):

```bash
docker compose -f deploy/docker-compose.yml up -d
# dev shortcut: `temporal server start-dev` + a local redis instead
```

**2. Worker** (the GPU box — connects out to the control plane):

```bash
bash deploy/run_worker.sh          # or, on a fresh vast.ai box: bash deploy/vast_setup.sh
```

**3. API service** (control plane; also serves the client in co-located dev):

```bash
bash deploy/run_api.sh             # → http://localhost:8000
```

Open http://localhost:8000. Storage defaults to the local disk backend
(`SPLATLAB_STORAGE=local`, API + worker sharing `data/`); set `SPLATLAB_STORAGE=s3`
with the `SPLATLAB_S3_*` env vars to run the worker on a **remote** GPU box that
shares artifacts with the API through MinIO/S3. To host the client separately,
serve `client/` from any static host and edit `client/config.js` to point
`SPLATLAB_API_BASE` / `SPLATLAB_WS_BASE` at the API's origin.

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
- Artifacts live under `data/<project-id>/` on the local backend
  (`photos/`, `colmap/`, `snapshots/`, `checkpoint.pt`); on the S3 backend the
  same layout is keyed under `<project-id>/` in the bucket.
- The live WebSocket protocol (for building your own client) is documented at the
  top of `server/shared/events.py`; the durable job state lives in Temporal and is
  visible in the Temporal UI (http://localhost:8233).
