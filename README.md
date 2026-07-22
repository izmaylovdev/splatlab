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

### Local, all-in-Docker (recommended)

Brings up the **entire control plane** — Temporal (+Postgres), Redis, MinIO
("dev S3"), the API, and the control worker — in one command:

```bash
cp deploy/.env.example deploy/.env         # tweak creds/tunables if you like
docker compose -f deploy/docker-compose.yml up -d --build
```

Open:
- **UI** → http://localhost:8000
- **Temporal UI** → http://localhost:8233
- **MinIO console** → http://localhost:9001  (`splatlab` / `splatlab-secret`)

The stack runs on the **S3 backend** against the bundled MinIO. **GPU training
happens on rented Vast.ai boxes, not in this stack**: the control worker runs the
workflow and dispatches COLMAP/train activities to the `splat-gpu` task queue,
which auto-rented GPU boxes poll. Turn that on with the `pool` profile below —
without it, a started run waits in the queue while everything up to that point
(projects, photo/video upload, the UI, telemetry) works locally. See
[`deploy/Dockerfile`](deploy/Dockerfile) (control-plane image, no CUDA) and the
header of [`deploy/docker-compose.yml`](deploy/docker-compose.yml).

Tear down with `docker compose -f deploy/docker-compose.yml down` (add `-v` to
also wipe the Postgres/Redis/MinIO volumes).

### From source (individual processes)

Prefer running the processes directly (e.g. to attach a real GPU worker)? Bring
up just the infra, then launch the app processes from your venv:

**1. Infra** (Temporal + Redis + MinIO):

```bash
docker compose -f deploy/docker-compose.yml up -d postgresql temporal temporal-ui redis minio
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

### Automatic GPU rental (Vast.ai pool over Tailscale)

This is the intended flow: develop locally on your Mac, and let the control plane
**rent a GPU box on demand** for the COLMAP/training steps, then release it when
idle. The rented box reaches your laptop-local Temporal / Redis / MinIO over a
**Tailscale** tailnet, so a run started from your Mac actually completes.

**One-time setup:**

1. **Tailscale on your Mac** — install the app (`brew install --cask tailscale`,
   or from tailscale.com) and sign in. Note your Mac's tailnet IP:

   ```bash
   tailscale ip -4        # e.g. 100.101.102.103
   ```

2. **An auth key** — Tailscale admin console → *Settings → Keys* → generate a
   key (reusable + ephemeral recommended, so reaped boxes drop off the tailnet).

3. **Fill in `deploy/.env`:**

   ```ini
   VAST_API_KEY=…            # Vast.ai: Account → API
   SPLATLAB_HOST=100.101.102.103   # your Mac's tailnet IP from step 1
   TAILSCALE_AUTHKEY=tskey-auth-…  # from step 2
   VAST_MAX_PRICE=0.60       # $/hr ceiling per box (optional)
   ```

**Run it** — bring up the stack with the `pool` profile:

```bash
docker compose -f deploy/docker-compose.yml --profile pool up -d --build
```

Now start a run from the UI at http://localhost:8000. The pool rents a GPU box,
the box joins your tailnet and dials back into your Mac, runs COLMAP/training
(streaming live telemetry to the browser), and is reaped once idle.

The **pool** ([`server/vast/pool.py`](server/vast/pool.py)) watches how many
training workflows need a GPU and keeps the fleet sized to match — renting the
cheapest offer that meets your filters, keeping a just-used box **warm** for reuse,
and reaping it after `SPLATLAB_POOL_IDLE_TIMEOUT`. Boxes only ever run the GPU
activities; the **control worker** runs the workflow itself so a run survives every
box being reaped. Teardown never depends on a single workflow: a crashed box is
reaped by stale-heartbeat, a leaked instance by the orphan sweep, and every box by
`SPLATLAB_POOL_MAX_LIFETIME` — so a paid GPU is never left running unmanaged.

Key knobs live in [`server/config.py`](server/config.py): `VAST_GPU_NAME`,
`VAST_MAX_PRICE`, `SPLATLAB_POOL_MAX_BOXES`, `SPLATLAB_POOL_IDLE_TIMEOUT`,
`VAST_IMAGE`, `VAST_REPO_URL`. `SPLATLAB_POOL_PAUSED=1` drains the whole fleet to
zero. Running from source instead of Docker? Use
[`deploy/run_pool.sh`](deploy/run_pool.sh) — it takes the same `VAST_*` /
`TS_AUTHKEY` / `BOX_*` env described in its header.

## Workflow

1. **New project** → give it a name.
2. **Upload photos** — pick a folder (30–300 photos of a scene/object with good
   overlap; avoid motion blur; keep exposure fixed if possible).
3. **Run SFM** — COLMAP computes camera poses and a sparse point cloud (this can
   take minutes). The pipeline runs as two separate steps, so this **releases the
   GPU** when it finishes rather than rolling straight into training.
4. **Review the points** — the sparse SfM cloud loads into the 3D viewer (orbit /
   zoom it), with its point + registered-image counts. If the reconstruction
   looks wrong (too few points, wrong shape), fix the photos and **Re-run SFM**
   before spending GPU time on training.
5. **Train** — once the points look right, start training. Watch the live view,
   loss chart, and the interactive 3D viewer (updates every snapshot interval).
6. **Export** — download the latest `.ply` (standard 3DGS format, opens in
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
