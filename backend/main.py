"""SplatLab server.  Run:  python -m backend.main [--mock] [--port 8000]"""
from __future__ import annotations

import argparse
import asyncio
import os

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .projects import DATA_DIR, IMAGE_EXTS, VIDEO_EXTS, ProjectManager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(ROOT, "frontend")

app = FastAPI(title="SplatLab")
manager: ProjectManager  # set in main()


@app.on_event("startup")
async def _startup() -> None:
    manager.loop = asyncio.get_running_loop()


# ---- projects ---------------------------------------------------------------
@app.get("/api/projects")
def list_projects():
    return [p.to_dict() for p in
            sorted(manager.projects.values(), key=lambda p: -p.created)]


@app.post("/api/projects")
async def create_project(body: dict):
    p = manager.create(str(body.get("name", "")).strip()[:80])
    return p.to_dict()


def _get(pid: str):
    p = manager.projects.get(pid)
    if p is None:
        raise HTTPException(404, "project not found")
    return p


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    return _get(pid).to_dict()


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    _get(pid)
    manager.delete(pid)
    return {"ok": True}


# ---- photos -----------------------------------------------------------------
@app.post("/api/projects/{pid}/photos")
async def upload_photos(pid: str, files: list[UploadFile]):
    p = _get(pid)
    if p.status in ("training", "processing_poses"):
        raise HTTPException(409, "stop training before changing photos")
    os.makedirs(p.photos_dir, exist_ok=True)
    saved = skipped = 0
    for f in files:
        name = os.path.basename(f.filename or "")
        if not name.lower().endswith(IMAGE_EXTS):
            skipped += 1
            continue
        # flatten any folder structure; avoid collisions
        dest = os.path.join(p.photos_dir, name)
        i = 1
        while os.path.exists(dest):
            stem, ext = os.path.splitext(name)
            dest = os.path.join(p.photos_dir, f"{stem}_{i}{ext}")
            i += 1
        with open(dest, "wb") as out:
            while chunk := await f.read(1 << 20):
                out.write(chunk)
        saved += 1
    # photo set changed → previous poses are stale
    if saved:
        import shutil
        shutil.rmtree(os.path.join(p.dir, "colmap"), ignore_errors=True)
    return {"saved": saved, "skipped": skipped, "total": p.num_photos()}


@app.post("/api/projects/{pid}/video")
async def upload_video(pid: str, file: UploadFile, frames: int = 150):
    """Extract evenly-spaced frames from a video into the project's photo set."""
    p = _get(pid)
    if p.status in ("training", "processing_poses"):
        raise HTTPException(409, "stop training before changing photos")
    name = os.path.basename(file.filename or "")
    if not name.lower().endswith(VIDEO_EXTS):
        raise HTTPException(400, "unsupported video format")
    os.makedirs(p.photos_dir, exist_ok=True)

    # stream the upload to a temp file, then hand it to ffmpeg
    import tempfile

    from .video_frames import extract_frames

    ext = os.path.splitext(name)[1] or ".mp4"
    fd, tmp = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := await file.read(1 << 20):
                out.write(chunk)
        stem = os.path.splitext(name)[0] or "video"
        prefix = "".join(c if c.isalnum() else "_" for c in stem)[:40] or "video"
        try:
            saved = extract_frames(tmp, p.photos_dir, target_frames=frames,
                                   prefix=prefix)
        except RuntimeError as e:
            raise HTTPException(422, str(e)) from e
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass

    if saved:
        import shutil
        shutil.rmtree(os.path.join(p.dir, "colmap"), ignore_errors=True)
    return {"saved": saved, "total": p.num_photos()}


@app.delete("/api/projects/{pid}/photos")
def clear_photos(pid: str):
    p = _get(pid)
    if p.status in ("training", "processing_poses"):
        raise HTTPException(409, "stop training first")
    import shutil
    shutil.rmtree(p.photos_dir, ignore_errors=True)
    shutil.rmtree(os.path.join(p.dir, "colmap"), ignore_errors=True)
    os.makedirs(p.photos_dir, exist_ok=True)
    return {"ok": True}


# ---- training control -------------------------------------------------------
@app.post("/api/projects/{pid}/train")
def start_training(pid: str, body: dict | None = None):
    p = _get(pid)
    try:
        manager.start_training(pid, (body or {}).get("config"))
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return p.to_dict()


@app.post("/api/projects/{pid}/stop")
def stop_training(pid: str):
    _get(pid)
    manager.stop_training(pid)
    return {"ok": True}


# ---- files (snapshots, exports) --------------------------------------------
@app.get("/files/{pid}/{path:path}")
def project_file(pid: str, path: str):
    p = _get(pid)
    full = os.path.realpath(os.path.join(p.dir, path))
    if not full.startswith(os.path.realpath(p.dir) + os.sep):
        raise HTTPException(403, "forbidden")
    if not os.path.isfile(full):
        raise HTTPException(404, "file not found")
    return FileResponse(full)


@app.get("/api/projects/{pid}/export")
def export_latest(pid: str):
    p = _get(pid)
    snaps = p.snapshots()
    if not snaps:
        raise HTTPException(404, "no snapshot yet")
    full = os.path.join(p.dir, "snapshots", snaps[-1])
    return FileResponse(full, filename=f"{p.name or p.id}.ply",
                        media_type="application/octet-stream")


# ---- websocket --------------------------------------------------------------
@app.websocket("/ws/{pid}")
async def ws(pid: str, socket: WebSocket):
    p = manager.projects.get(pid)
    if p is None:
        await socket.close(code=4004)
        return
    await socket.accept()
    q = manager.subscribe(pid)
    try:
        await socket.send_json({
            "type": "hello",
            "project": p.to_dict(),
            "history": p.history[-1500:],
            "latest_snapshot": (f"/files/{pid}/{p.latest_snapshot}"
                                if p.latest_snapshot else None),
        })
        while True:
            msg = await q.get()
            await socket.send_json(msg)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        manager.unsubscribe(pid, q)


# ---- frontend ---------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND, "index.html"))


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/api/health")
def health():
    return JSONResponse({"ok": True, "mock": manager.mock, "data_dir": DATA_DIR})


def main() -> None:
    global manager
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true",
                    help="fake trainer, no GPU/COLMAP needed (UI demo)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    manager = ProjectManager(mock=args.mock)

    import uvicorn
    print(f"SplatLab {'[MOCK MODE] ' if args.mock else ''}→ http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
