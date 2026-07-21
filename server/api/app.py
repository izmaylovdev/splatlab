"""SplatLab API service (control plane).

Runs no training itself: it translates HTTP/WebSocket <-> Temporal workflow calls
and the Redis telemetry bus, and serves artifacts through the storage backend.
The REST routes and WebSocket message shapes are unchanged from the monolith, so
the decoupled client only needs a configurable API/WS base URL.

Run:  python -m server.api.app [--host 0.0.0.0] [--port 8000]
"""
from __future__ import annotations

import argparse
import os
import tempfile
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, JSONResponse, RedirectResponse,
                               Response)
from fastapi.staticfiles import StaticFiles

from .. import config
from ..shared.constants import IMAGE_EXTS, VIDEO_EXTS
from ..shared.storage import get_storage
from . import meta as metamod
from . import temporal_client as tc
from . import ws as wsmod

CLIENT_DIR = os.path.join(config.ROOT, "client")

app = FastAPI(title="SplatLab API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=False,
)
storage = get_storage()

# busy states during which the photo set must not change
BUSY = {"queued", "processing_poses", "training"}


# ---- helpers ----------------------------------------------------------------
def _meta_or_404(pid: str) -> dict:
    m = storage.read_meta(pid)
    if m is None:
        raise HTTPException(404, "project not found")
    return m


async def _view(pid: str, meta: dict | None = None) -> dict:
    meta = meta or _meta_or_404(pid)
    live = await tc.query_state(pid)
    status = live["status"] if live else meta.get("status", "new")
    stage = live["stage"] if live else None
    cfg = (live.get("config") if live and live.get("config") else meta.get("config", {}))
    snaps = storage.list_snapshots(pid)
    return {
        "id": pid,
        "name": meta.get("name", ""),
        "created": meta.get("created", 0),
        "status": status,
        "error": None,
        "stage": stage,
        "num_photos": storage.num_photos(pid),
        "config": cfg,
        "latest_snapshot": storage.public_url(pid, snaps[-1]) if snaps else None,
        "snapshots": [storage.public_url(pid, s) for s in snaps],
    }


async def _is_busy(pid: str) -> bool:
    live = await tc.query_state(pid)
    return bool(live and live.get("status") in BUSY)


def _unique_photo_name(pid: str, name: str) -> str:
    dest = name
    i = 1
    while storage.photo_exists(pid, dest):
        stem, ext = os.path.splitext(name)
        dest = f"{stem}_{i}{ext}"
        i += 1
    return dest


# ---- projects ---------------------------------------------------------------
@app.get("/api/projects")
async def list_projects():
    metas = [storage.read_meta(pid) for pid in storage.list_project_ids()]
    metas = [m for m in metas if m]
    metas.sort(key=lambda m: -m.get("created", 0))
    return [await _view(m["id"], m) for m in metas]


@app.post("/api/projects")
async def create_project(body: dict):
    meta = metamod.new_project_meta(str(body.get("name", "")).strip())
    storage.write_meta(meta["id"], meta)
    return await _view(meta["id"], meta)


@app.get("/api/projects/{pid}")
async def get_project(pid: str):
    return await _view(pid)


@app.delete("/api/projects/{pid}")
async def delete_project(pid: str):
    _meta_or_404(pid)
    await tc.stop_training(pid)
    storage.delete_project(pid)
    return {"ok": True}


# ---- photos -----------------------------------------------------------------
@app.post("/api/projects/{pid}/photos")
async def upload_photos(pid: str, files: list[UploadFile]):
    _meta_or_404(pid)
    if await _is_busy(pid):
        raise HTTPException(409, "stop training before changing photos")
    saved = skipped = 0
    for f in files:
        name = os.path.basename(f.filename or "")
        if not name.lower().endswith(IMAGE_EXTS):
            skipped += 1
            continue
        storage.save_photo(pid, _unique_photo_name(pid, name), f.file)
        saved += 1
    return {"saved": saved, "skipped": skipped, "total": storage.num_photos(pid)}


@app.post("/api/projects/{pid}/video")
async def upload_video(pid: str, file: UploadFile, frames: int = 150):
    """Extract evenly-spaced frames from an uploaded video into the photo set."""
    from ..shared.video_frames import extract_frames

    _meta_or_404(pid)
    if await _is_busy(pid):
        raise HTTPException(409, "stop training before changing photos")
    name = os.path.basename(file.filename or "")
    if not name.lower().endswith(VIDEO_EXTS):
        raise HTTPException(400, "unsupported video format")

    ext = os.path.splitext(name)[1] or ".mp4"
    fd, tmp_vid = tempfile.mkstemp(suffix=ext)
    tmp_dir = tempfile.mkdtemp(prefix="splat_frames_")
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := await file.read(1 << 20):
                out.write(chunk)
        stem = os.path.splitext(name)[0] or "video"
        prefix = "".join(c if c.isalnum() else "_" for c in stem)[:40] or "video"
        try:
            extract_frames(tmp_vid, tmp_dir, target_frames=frames, prefix=prefix)
        except RuntimeError as e:
            raise HTTPException(422, str(e)) from e
        saved = 0
        for fname in sorted(os.listdir(tmp_dir)):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            with open(os.path.join(tmp_dir, fname), "rb") as fh:
                storage.save_photo(pid, _unique_photo_name(pid, fname), fh)
            saved += 1
    finally:
        _rm(tmp_vid)
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return {"saved": saved, "total": storage.num_photos(pid)}


@app.delete("/api/projects/{pid}/photos")
async def clear_photos(pid: str):
    _meta_or_404(pid)
    if await _is_busy(pid):
        raise HTTPException(409, "stop training first")
    storage.clear_photos(pid)
    return {"ok": True}


# ---- training control -------------------------------------------------------
@app.post("/api/projects/{pid}/train")
async def start_training(pid: str, body: dict | None = None):
    meta = _meta_or_404(pid)
    if storage.num_photos(pid) < 3:
        raise HTTPException(409, "Upload at least 3 photos first")
    cfg = metamod.merge_config(meta, (body or {}).get("config"))
    meta["config"] = cfg
    storage.write_meta(pid, meta)
    try:
        await tc.start_training(pid, cfg, run_id=uuid.uuid4().hex)
    except tc.AlreadyRunning as e:
        raise HTTPException(409, "Training already running for this project") from e
    return await _view(pid, meta)


@app.post("/api/projects/{pid}/stop")
async def stop_training(pid: str):
    _meta_or_404(pid)
    await tc.stop_training(pid)
    return {"ok": True}


# ---- files (snapshots, exports) --------------------------------------------
@app.get("/files/{pid}/{path:path}")
def project_file(pid: str, path: str):
    local = storage.local_artifact_path(pid, path)
    if local:
        return FileResponse(local)
    # remote backend: browser should already use presigned URLs; redirect anyway
    return RedirectResponse(storage.public_url(pid, path))


@app.get("/api/projects/{pid}/export")
async def export_latest(pid: str):
    meta = _meta_or_404(pid)
    snaps = storage.list_snapshots(pid)
    if not snaps:
        raise HTTPException(404, "no snapshot yet")
    rel = snaps[-1]
    fname = f"{meta.get('name') or pid}.ply"
    local = storage.local_artifact_path(pid, rel)
    if local:
        return FileResponse(local, filename=fname, media_type="application/octet-stream")
    return RedirectResponse(storage.public_url(pid, rel, download_name=fname))


# ---- websocket --------------------------------------------------------------
@app.websocket("/ws/{pid}")
async def ws(pid: str, socket: WebSocket):
    meta = storage.read_meta(pid)
    if meta is None:
        await socket.close(code=4004)
        return
    await wsmod.serve(socket, pid, await _view(pid, meta))


# ---- health + client config -------------------------------------------------
@app.get("/api/health")
def health():
    return JSONResponse({"ok": True, "storage": config.STORAGE_BACKEND,
                         "task_queue": config.TASK_QUEUE})


@app.get("/config.js")
def client_config():
    # Injected into the API-served client. A separately hosted client ships its
    # own config.js pointing SPLATLAB_API_BASE/WS_BASE at this API's origin.
    return Response(
        "window.SPLATLAB_API_BASE = window.SPLATLAB_API_BASE || '';\n",
        media_type="application/javascript",
    )


# ---- optional: serve the client in dev/co-located deploys -------------------
# Mounted LAST so explicit API/WS/file routes above take precedence. Serves
# index.html at "/" and client assets (incl. ./vendor/*) at their paths.
if os.path.isdir(CLIENT_DIR):
    app.mount("/", StaticFiles(directory=CLIENT_DIR, html=True), name="client")


def _rm(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    import uvicorn
    print(f"SplatLab API → http://{args.host}:{args.port} "
          f"(storage={config.STORAGE_BACKEND}, temporal={config.TEMPORAL_ADDRESS})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
