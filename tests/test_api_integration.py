"""Integration smoke of the real API routes + storage, and the emit telemetry
contract. No live Temporal/Redis: TEMPORAL_ADDRESS points at a dead port (proves
graceful degradation to stored meta status) and the emit path uses fakeredis.
"""
import io
import os
import tempfile

# Must be set BEFORE importing server.config / server.api.app.
os.environ["SPLATLAB_STORAGE"] = "local"
os.environ["SPLATLAB_DATA_DIR"] = tempfile.mkdtemp(prefix="splat_api_test_")
os.environ["TEMPORAL_ADDRESS"] = "127.0.0.1:1"   # unreachable → query_state degrades

from fastapi.testclient import TestClient  # noqa: E402

from server.api.app import app  # noqa: E402


def test_api_crud_and_files():
    c = TestClient(app)

    # health + client config
    h = c.get("/api/health").json()
    assert h["ok"] and h["storage"] == "local", h
    assert "SPLATLAB_API_BASE" in c.get("/config.js").text

    # client index served at /
    assert "SplatLab" in c.get("/").text

    # create → list → get  (Temporal down → falls back to meta status 'new')
    p = c.post("/api/projects", json={"name": "demo"}).json()
    pid = p["id"]
    assert p["status"] == "new" and p["num_photos"] == 0, p
    assert any(x["id"] == pid for x in c.get("/api/projects").json())
    assert c.get(f"/api/projects/{pid}").json()["name"] == "demo"

    # upload two "photos"
    files = [("files", ("a.jpg", io.BytesIO(b"x" * 32), "image/jpeg")),
             ("files", ("b.png", io.BytesIO(b"y" * 32), "image/png")),
             ("files", ("notes.txt", io.BytesIO(b"z"), "text/plain"))]
    r = c.post(f"/api/projects/{pid}/photos", files=files).json()
    assert r == {"saved": 2, "skipped": 1, "total": 2}, r
    assert c.get(f"/api/projects/{pid}").json()["num_photos"] == 2

    # train guard: fewer than 3 photos → 409 (no Temporal needed to reject)
    tr = c.post(f"/api/projects/{pid}/train", json={})
    assert tr.status_code == 409, (tr.status_code, tr.text)

    # fake a snapshot artifact and fetch it through /files + export
    from server.shared.storage import get_storage
    st = get_storage()
    snapdir = os.path.join(st.project_dir(pid), "snapshots")
    os.makedirs(snapdir, exist_ok=True)
    with open(os.path.join(snapdir, "step_000500.ply"), "wb") as f:
        f.write(b"PLYDATA")
    view = c.get(f"/api/projects/{pid}").json()
    assert view["latest_snapshot"] == f"/files/{pid}/snapshots/step_000500.ply", view
    assert view["snapshots"] == [view["latest_snapshot"]]
    got = c.get(view["latest_snapshot"])
    assert got.status_code == 200 and got.content == b"PLYDATA"
    exp = c.get(f"/api/projects/{pid}/export")
    assert exp.status_code == 200 and exp.content == b"PLYDATA", exp.status_code

    # path-traversal guard on /files
    assert c.get(f"/files/{pid}/../../etc/passwd").status_code in (403, 404)

    # clear photos + delete project
    assert c.delete(f"/api/projects/{pid}/photos").json() == {"ok": True}
    assert c.get(f"/api/projects/{pid}").json()["num_photos"] == 0
    assert c.delete(f"/api/projects/{pid}").json() == {"ok": True}
    assert c.get(f"/api/projects/{pid}").status_code == 404
    print("API CRUD + files smoke PASSED")


def test_emit_telemetry_contract():
    import json

    import server.activities.emit as emitmod
    from server.shared.events import channel, history_key
    from server.shared.storage import get_storage

    # Recording double for the sync redis client: capture every pipeline op.
    class Rec:
        def __init__(self):
            self.published = []      # (channel, payload)
            self.xadds = []          # (key, payload)
        def pipeline(self):
            return self
        def publish(self, ch, payload):
            self.published.append((ch, payload))
        def xadd(self, key, fields, **kw):
            self.xadds.append((key, fields["data"]))
        def expire(self, *a, **k):
            pass
        def execute(self):
            pass

    rec = Rec()
    emitmod.redis.Redis.from_url = staticmethod(lambda *a, **k: rec)

    st = get_storage()
    pid = "telem1"
    os.makedirs(os.path.join(st.project_dir(pid), "snapshots"), exist_ok=True)
    snap = os.path.join(st.project_dir(pid), "snapshots", "step_000500.ply")
    with open(snap, "wb") as f:
        f.write(b"P")

    emit = emitmod.make_redis_emitter(pid, "run1")
    emit({"type": "progress", "step": 10, "loss": 0.5})       # published + backlog
    emit({"type": "frame", "step": 20, "jpeg": b"\xff\xd8jpeg"})  # base64, published only
    emit({"type": "snapshot", "step": 500, "path": snap})     # path→url, published only

    pub = [json.loads(p) for ch, p in rec.published]
    assert all(ch == channel(pid) for ch, _ in rec.published)
    assert [m["type"] for m in pub] == ["progress", "frame", "snapshot"], pub

    frame = pub[1]
    assert frame["jpeg_b64"] and "jpeg" not in frame
    snapmsg = pub[2]
    assert snapmsg["url"] == f"/files/{pid}/snapshots/step_000500.ply", snapmsg
    assert "path" not in snapmsg

    # only the progress event went to the capped backlog stream
    assert len(rec.xadds) == 1 and rec.xadds[0][0] == history_key(pid)
    assert json.loads(rec.xadds[0][1])["type"] == "progress"
    print("emit telemetry contract PASSED")


if __name__ == "__main__":
    test_api_crud_and_files()
    test_emit_telemetry_contract()
    print("ALL API/TELEMETRY TESTS PASSED")
