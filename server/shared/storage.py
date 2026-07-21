"""Artifact + metadata storage, abstracted over two backends.

Both the worker (which *writes* .ply snapshots and reads photos) and the API
(which *serves* artifacts and lists projects) go through this interface, so the
two processes can live on different machines.

  LocalDiskStorage  co-located API + worker sharing a filesystem (dev default).
  S3Storage         MinIO/S3 bridges a remote GPU worker and a separate API.

Layout (keys/relpaths are identical across backends):
    <pid>/meta.json
    <pid>/photos/<name>
    <pid>/snapshots/step_NNNNNN.ply
    <pid>/checkpoint.pt

The worker always does heavy I/O on *local* disk under ``project_dir(pid)`` and
then calls ``upload_artifact`` / ``ensure_photos_local`` to sync with the remote
store (no-ops on the local backend).
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Any, BinaryIO

from .. import config
from .constants import IMAGE_EXTS

SNAP_PREFIX = "snapshots/"


class Storage:
    """Interface. See module docstring for the key layout."""

    # ---- project metadata ---------------------------------------------------
    def read_meta(self, pid: str) -> dict[str, Any] | None: ...
    def write_meta(self, pid: str, meta: dict[str, Any]) -> None: ...
    def list_project_ids(self) -> list[str]: ...
    def delete_project(self, pid: str) -> None: ...

    # ---- photos (API writes; worker reads) ----------------------------------
    def save_photo(self, pid: str, name: str, src: BinaryIO) -> None: ...
    def photo_exists(self, pid: str, name: str) -> bool: ...
    def list_photos(self, pid: str) -> list[str]: ...
    def clear_photos(self, pid: str) -> None: ...
    def ensure_photos_local(self, pid: str) -> str:
        """Worker-side: guarantee photos exist under a local dir; return it."""
        ...

    def num_photos(self, pid: str) -> int:
        return len(self.list_photos(pid))

    # ---- artifacts (worker writes locally, then uploads; API serves) --------
    def project_dir(self, pid: str) -> str:
        """Local working directory for a project on this machine."""
        return os.path.join(config.DATA_DIR, pid)

    def upload_artifact(self, pid: str, rel: str) -> None:
        """Push a locally-written file (rel to project_dir) to the store."""
        ...

    def upload_dir(self, pid: str, rel: str) -> None:
        """Push a locally-written *directory* subtree (rel to project_dir) to the
        store. Used to hand an intermediate dataset (COLMAP's undistorted output)
        from the box that produced it to whichever box consumes it next.
        No-op on the local backend."""
        ...

    def ensure_dir_local(self, pid: str, rel: str) -> str:
        """Worker-side inverse of ``upload_dir``: guarantee the subtree exists
        under a local dir; return its absolute path. On the local backend this is
        just the existing shared path."""
        ...

    def list_snapshots(self, pid: str) -> list[str]:
        """Sorted rel paths like 'snapshots/step_000500.ply'."""
        ...

    def public_url(self, pid: str, rel: str, download_name: str | None = None) -> str:
        """URL the browser can load the artifact from."""
        ...

    def local_artifact_path(self, pid: str, rel: str) -> str | None:
        """A local filesystem path the API can FileResponse, or None (remote)."""
        ...


# ---------------------------------------------------------------------------
class LocalDiskStorage(Storage):
    def __init__(self, data_dir: str = config.DATA_DIR):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def _pdir(self, pid: str) -> str:
        return os.path.join(self.data_dir, pid)

    def _photos(self, pid: str) -> str:
        return os.path.join(self._pdir(pid), "photos")

    # metadata
    def read_meta(self, pid: str) -> dict[str, Any] | None:
        path = os.path.join(self._pdir(pid), "meta.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None

    def write_meta(self, pid: str, meta: dict[str, Any]) -> None:
        os.makedirs(self._pdir(pid), exist_ok=True)
        path = os.path.join(self._pdir(pid), "meta.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(meta, f)
        os.replace(tmp, path)

    def list_project_ids(self) -> list[str]:
        out = []
        for pid in os.listdir(self.data_dir):
            if os.path.exists(os.path.join(self.data_dir, pid, "meta.json")):
                out.append(pid)
        return out

    def delete_project(self, pid: str) -> None:
        shutil.rmtree(self._pdir(pid), ignore_errors=True)

    # photos
    def save_photo(self, pid: str, name: str, src: BinaryIO) -> None:
        d = self._photos(pid)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, name), "wb") as out:
            shutil.copyfileobj(src, out)

    def photo_exists(self, pid: str, name: str) -> bool:
        return os.path.exists(os.path.join(self._photos(pid), name))

    def list_photos(self, pid: str) -> list[str]:
        d = self._photos(pid)
        if not os.path.isdir(d):
            return []
        return sorted(f for f in os.listdir(d) if f.lower().endswith(IMAGE_EXTS))

    def clear_photos(self, pid: str) -> None:
        shutil.rmtree(self._photos(pid), ignore_errors=True)
        os.makedirs(self._photos(pid), exist_ok=True)

    def ensure_photos_local(self, pid: str) -> str:
        return self._photos(pid)

    # artifacts
    def upload_artifact(self, pid: str, rel: str) -> None:
        return  # already on the shared disk

    def upload_dir(self, pid: str, rel: str) -> None:
        return  # already on the shared disk

    def ensure_dir_local(self, pid: str, rel: str) -> str:
        return os.path.join(self._pdir(pid), rel)

    def list_snapshots(self, pid: str) -> list[str]:
        d = os.path.join(self._pdir(pid), "snapshots")
        if not os.path.isdir(d):
            return []
        return [SNAP_PREFIX + f for f in sorted(os.listdir(d)) if f.endswith(".ply")]

    def public_url(self, pid: str, rel: str, download_name: str | None = None) -> str:
        return f"/files/{pid}/{rel}"

    def local_artifact_path(self, pid: str, rel: str) -> str | None:
        full = os.path.realpath(os.path.join(self._pdir(pid), rel))
        root = os.path.realpath(self._pdir(pid))
        if not full.startswith(root + os.sep):
            return None
        return full if os.path.isfile(full) else None


# ---------------------------------------------------------------------------
class S3Storage(Storage):
    """MinIO/S3-backed. Worker keeps a local scratch mirror under DATA_DIR."""

    def __init__(self):
        import boto3  # lazy: API/worker only import when this backend is chosen
        from botocore.client import Config as BotoConfig

        self.bucket = config.S3_BUCKET
        self._boto = boto3
        self._client = boto3.client(
            "s3",
            endpoint_url=config.S3_ENDPOINT_URL,
            region_name=config.S3_REGION,
            aws_access_key_id=config.S3_ACCESS_KEY,
            aws_secret_access_key=config.S3_SECRET_KEY,
            config=BotoConfig(signature_version="s3v4"),
        )
        # A second client bound to the browser-reachable endpoint, used only to
        # mint presigned URLs (the internal endpoint may be unreachable outside).
        self._public = boto3.client(
            "s3",
            endpoint_url=config.S3_PUBLIC_ENDPOINT,
            region_name=config.S3_REGION,
            aws_access_key_id=config.S3_ACCESS_KEY,
            aws_secret_access_key=config.S3_SECRET_KEY,
            config=BotoConfig(signature_version="s3v4"),
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except Exception:
            try:
                self._client.create_bucket(Bucket=self.bucket)
            except Exception:
                pass

    def _key(self, pid: str, rel: str) -> str:
        return f"{pid}/{rel}"

    def _list(self, prefix: str) -> list[str]:
        keys, token = [], None
        while True:
            kw = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            resp = self._client.list_objects_v2(**kw)
            for obj in resp.get("Contents", []):
                keys.append(obj["Key"])
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return keys

    # metadata
    def read_meta(self, pid: str) -> dict[str, Any] | None:
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=self._key(pid, "meta.json"))
            return json.loads(resp["Body"].read())
        except self._client.exceptions.NoSuchKey:
            return None
        except Exception:
            return None

    def write_meta(self, pid: str, meta: dict[str, Any]) -> None:
        self._client.put_object(
            Bucket=self.bucket, Key=self._key(pid, "meta.json"),
            Body=json.dumps(meta).encode(), ContentType="application/json",
        )

    def list_project_ids(self) -> list[str]:
        resp = self._client.list_objects_v2(Bucket=self.bucket, Delimiter="/")
        pids = []
        for cp in resp.get("CommonPrefixes", []):
            pid = cp["Prefix"].rstrip("/")
            if self.read_meta(pid) is not None:
                pids.append(pid)
        return pids

    def delete_project(self, pid: str) -> None:
        keys = self._list(f"{pid}/")
        for i in range(0, len(keys), 1000):
            batch = [{"Key": k} for k in keys[i:i + 1000]]
            self._client.delete_objects(Bucket=self.bucket, Delete={"Objects": batch})
        shutil.rmtree(self.project_dir(pid), ignore_errors=True)

    # photos
    def save_photo(self, pid: str, name: str, src: BinaryIO) -> None:
        self._client.upload_fileobj(src, self.bucket, self._key(pid, f"photos/{name}"))

    def photo_exists(self, pid: str, name: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=self._key(pid, f"photos/{name}"))
            return True
        except Exception:
            return False

    def list_photos(self, pid: str) -> list[str]:
        prefix = self._key(pid, "photos/")
        names = [k[len(prefix):] for k in self._list(prefix)]
        return sorted(n for n in names if n and n.lower().endswith(IMAGE_EXTS))

    def clear_photos(self, pid: str) -> None:
        keys = self._list(self._key(pid, "photos/"))
        for i in range(0, len(keys), 1000):
            batch = [{"Key": k} for k in keys[i:i + 1000]]
            self._client.delete_objects(Bucket=self.bucket, Delete={"Objects": batch})

    def ensure_photos_local(self, pid: str) -> str:
        dest = os.path.join(self.project_dir(pid), "photos")
        os.makedirs(dest, exist_ok=True)
        for name in self.list_photos(pid):
            local = os.path.join(dest, name)
            if not os.path.exists(local):
                self._client.download_file(self.bucket, self._key(pid, f"photos/{name}"), local)
        return dest

    # artifacts
    def upload_artifact(self, pid: str, rel: str) -> None:
        local = os.path.join(self.project_dir(pid), rel)
        self._client.upload_file(local, self.bucket, self._key(pid, rel))

    def upload_dir(self, pid: str, rel: str) -> None:
        base = os.path.join(self.project_dir(pid), rel)
        if not os.path.isdir(base):
            return
        for root, _, files in os.walk(base):
            for name in files:
                local = os.path.join(root, name)
                sub = os.path.relpath(local, self.project_dir(pid)).replace(os.sep, "/")
                self._client.upload_file(local, self.bucket, self._key(pid, sub))

    def ensure_dir_local(self, pid: str, rel: str) -> str:
        dest = os.path.join(self.project_dir(pid), rel)
        prefix = self._key(pid, rel.rstrip("/") + "/")
        for key in self._list(prefix):
            local = os.path.join(self.project_dir(pid), key[len(pid) + 1:])
            if not os.path.exists(local):
                os.makedirs(os.path.dirname(local), exist_ok=True)
                self._client.download_file(self.bucket, key, local)
        return dest

    def list_snapshots(self, pid: str) -> list[str]:
        prefix = self._key(pid, SNAP_PREFIX)
        rels = [SNAP_PREFIX + k[len(prefix):] for k in self._list(prefix)]
        return sorted(r for r in rels if r.endswith(".ply"))

    def public_url(self, pid: str, rel: str, download_name: str | None = None) -> str:
        params = {"Bucket": self.bucket, "Key": self._key(pid, rel)}
        if download_name:
            params["ResponseContentDisposition"] = f'attachment; filename="{download_name}"'
        return self._public.generate_presigned_url(
            "get_object", Params=params, ExpiresIn=config.S3_PRESIGN_TTL)

    def local_artifact_path(self, pid: str, rel: str) -> str | None:
        return None  # browser loads presigned URLs directly


_storage: Storage | None = None


def get_storage() -> Storage:
    """Process-wide storage singleton, chosen by SPLATLAB_STORAGE."""
    global _storage
    if _storage is None:
        _storage = S3Storage() if config.STORAGE_BACKEND == "s3" else LocalDiskStorage()
    return _storage
