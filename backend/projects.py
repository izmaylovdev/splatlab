"""Project store, background training jobs, and the WebSocket event bus.

WebSocket protocol (ws://host/ws/{project_id}) — server → client JSON messages:

  {"type": "hello", "project": {...}, "history": [progress...],
   "latest_snapshot": "/files/<id>/snapshots/step_000500.ply" | null}
  {"type": "status",   "stage": "features|matching|mapping|undistort|init|training|done",
                       "detail": str}
  {"type": "progress", "step", "max_steps", "loss", "psnr", "num_gaussians", "sh_degree"}
  {"type": "frame",    "step", "jpeg_b64": str}
  {"type": "snapshot", "step", "url": str}          # .ply ready to load in viewer
  {"type": "done",     "step", "url": str}
  {"type": "error",    "message": str}

No client → server messages are required (control is via REST).
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import threading
import time
import uuid
from typing import Any

from .train_common import TrainConfig

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp")
VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".wmv", ".mpg", ".mpeg")


class Project:
    def __init__(self, pid: str, name: str, created: float | None = None):
        self.id = pid
        self.name = name
        self.created = created or time.time()
        self.status = "new"          # new|processing_poses|training|done|stopped|error
        self.deleted = False
        self.error: str | None = None
        self.config = TrainConfig()
        # live state
        self.history: list[dict[str, Any]] = []      # recent progress events
        self.latest_snapshot: str | None = None      # relative path under project dir
        self.stage: dict[str, str] | None = None

    @property
    def dir(self) -> str:
        return os.path.join(DATA_DIR, self.id)

    @property
    def photos_dir(self) -> str:
        return os.path.join(self.dir, "photos")

    def num_photos(self) -> int:
        if not os.path.isdir(self.photos_dir):
            return 0
        return len([f for f in os.listdir(self.photos_dir)
                    if f.lower().endswith(IMAGE_EXTS)])

    def snapshots(self) -> list[str]:
        d = os.path.join(self.dir, "snapshots")
        if not os.path.isdir(d):
            return []
        return sorted(f for f in os.listdir(d) if f.endswith(".ply"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "created": self.created,
            "status": self.status, "error": self.error,
            "num_photos": self.num_photos(),
            "config": self.config.to_dict(),
            "stage": self.stage,
            "latest_snapshot": (f"/files/{self.id}/{self.latest_snapshot}"
                                if self.latest_snapshot else None),
            "snapshots": [f"/files/{self.id}/snapshots/{s}" for s in self.snapshots()],
        }

    def save_meta(self) -> None:
        if self.deleted:
            return
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump({"id": self.id, "name": self.name, "created": self.created,
                       "status": self.status, "config": self.config.to_dict()}, f)


class ProjectManager:
    def __init__(self, mock: bool = False):
        self.mock = mock
        self.projects: dict[str, Project] = {}
        self.jobs: dict[str, threading.Thread] = {}
        self.stops: dict[str, threading.Event] = {}
        self.subscribers: dict[str, set[asyncio.Queue]] = {}
        self.loop: asyncio.AbstractEventLoop | None = None
        os.makedirs(DATA_DIR, exist_ok=True)
        self._load()

    # ---- persistence --------------------------------------------------------
    def _load(self) -> None:
        for pid in os.listdir(DATA_DIR):
            meta_path = os.path.join(DATA_DIR, pid, "meta.json")
            if not os.path.exists(meta_path):
                continue
            try:
                with open(meta_path) as f:
                    m = json.load(f)
                p = Project(m["id"], m["name"], m.get("created"))
                p.config = TrainConfig.from_dict(m.get("config", {}))
                p.status = m.get("status", "new")
                if p.status in ("training", "processing_poses"):
                    p.status = "stopped"     # server restarted mid-run
                snaps = p.snapshots()
                if snaps:
                    p.latest_snapshot = f"snapshots/{snaps[-1]}"
                self.projects[p.id] = p
            except Exception:
                continue

    # ---- crud ---------------------------------------------------------------
    def create(self, name: str) -> Project:
        pid = uuid.uuid4().hex[:12]
        p = Project(pid, name or f"project-{pid[:6]}")
        os.makedirs(p.photos_dir, exist_ok=True)
        self.projects[pid] = p
        p.save_meta()
        return p

    def delete(self, pid: str) -> None:
        self.stop_training(pid)
        p = self.projects.pop(pid, None)
        if p:
            p.deleted = True
            shutil.rmtree(p.dir, ignore_errors=True)

    # ---- events -------------------------------------------------------------
    def subscribe(self, pid: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self.subscribers.setdefault(pid, set()).add(q)
        return q

    def unsubscribe(self, pid: str, q: asyncio.Queue) -> None:
        self.subscribers.get(pid, set()).discard(q)

    def _publish(self, pid: str, msg: dict[str, Any]) -> None:
        """Called from trainer threads — hop onto the event loop."""
        p = self.projects.get(pid)
        if p is not None:
            t = msg.get("type")
            if t == "progress":
                p.history.append(msg)
                if len(p.history) > 3000:
                    del p.history[: len(p.history) - 3000]
            elif t == "status":
                p.stage = {"stage": msg["stage"], "detail": msg.get("detail", "")}
            elif t in ("snapshot", "done"):
                if "path" in msg:
                    rel = os.path.relpath(msg["path"], p.dir).replace(os.sep, "/")
                    p.latest_snapshot = rel
                    msg = {**msg, "url": f"/files/{pid}/{rel}"}
                    msg.pop("path", None)
            if t == "frame":
                msg = {"type": "frame", "step": msg["step"],
                       "jpeg_b64": base64.b64encode(msg["jpeg"]).decode()}
            elif t == "error":
                p.status = "error"
                p.error = msg.get("message")
                p.save_meta()
            if t == "done":
                p.status = "done"
                p.save_meta()

        if self.loop is None:
            return
        queues = list(self.subscribers.get(pid, ()))

        def _push():
            for q in queues:
                if q.full():
                    try:
                        q.get_nowait()      # drop oldest (frames) rather than block
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait(msg)

        self.loop.call_soon_threadsafe(_push)

    # ---- training jobs ------------------------------------------------------
    def start_training(self, pid: str, overrides: dict[str, Any] | None = None) -> None:
        p = self.projects[pid]
        if pid in self.jobs and self.jobs[pid].is_alive():
            raise RuntimeError("Training already running for this project")
        if p.num_photos() < 3 and not self.mock:
            raise RuntimeError("Upload at least 3 photos first")
        if overrides:
            p.config = TrainConfig.from_dict({**p.config.to_dict(), **overrides})
        p.history.clear()
        p.error = None
        p.save_meta()
        stop = threading.Event()
        self.stops[pid] = stop
        th = threading.Thread(target=self._job, args=(p, stop), daemon=True)
        self.jobs[pid] = th
        th.start()

    def stop_training(self, pid: str) -> None:
        if pid in self.stops:
            self.stops[pid].set()

    def _job(self, p: Project, stop: threading.Event) -> None:
        emit = lambda msg: self._publish(p.id, msg)  # noqa: E731
        try:
            if self.mock:
                p.status = "processing_poses"
                for stage in ("features", "matching", "mapping", "undistort"):
                    emit({"type": "status", "stage": stage, "detail": "mock"})
                    time.sleep(0.4)
                    if stop.is_set():
                        p.status = "stopped"
                        return
                p.status = "training"
                from .mock_trainer import train as mock_train
                mock_train(p.dir, p.dir, p.config, emit, stop)
            else:
                undist = os.path.join(p.dir, "colmap", "undistorted")
                if not os.path.isdir(undist):
                    p.status = "processing_poses"
                    from .colmap_runner import run_colmap
                    undist = run_colmap(
                        p.photos_dir,
                        os.path.join(p.dir, "colmap"),
                        progress=lambda s, d: emit(
                            {"type": "status", "stage": s, "detail": d}),
                        matcher=p.config.colmap_matcher,
                    )
                if stop.is_set():
                    p.status = "stopped"
                    return
                p.status = "training"
                from .dataset import load_colmap_dataset
                from .trainer import train as real_train
                ds = load_colmap_dataset(undist, p.config.max_image_side)
                real_train(ds, p.dir, p.config, emit, stop)
            if stop.is_set() and p.status != "done":
                p.status = "stopped"
        except Exception as e:  # noqa: BLE001
            emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            p.save_meta()
