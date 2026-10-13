"""SSH links from the control plane to rented boxes (SPLATLAB_BOX_LINK=ssh).

The Mac sits behind carrier-grade NAT, so a box can't dial in — but the Mac can
dial OUT to a box's sshd. For every box the pool keeps reverse port forwards
open, so on the box

    localhost:7233 -> Temporal    localhost:6379 -> Redis    localhost:9000 -> MinIO

lead back here, and the worker simply uses localhost addresses. The forwards bind
on the box's loopback only: nothing is exposed to the internet.

Two SSH sessions per box: one for Temporal + Redis, one for MinIO, so a bulk
snapshot upload can't queue control traffic behind it inside a single TCP
connection (the reason the relay tunnel was split per service). Each session
reconnects on its own; a box that never comes up is reaped by the pool's boot
deadline as usual.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import time
from typing import Any
from urllib.parse import urlsplit

from .. import config
from .client import VastClient, VastError

# Ports the box's worker connects to on its own localhost (see pool._box_env).
BOX_TEMPORAL_PORT = 7233
BOX_REDIS_PORT = 6379
BOX_S3_PORT = 9000

_RETRY_DELAY = 5          # seconds between reconnect attempts
_UP_AFTER = 8             # an ssh still running after this long counts as "up"


def _log(msg: str) -> None:
    print(f"[link {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _host_port(addr: str, default_port: int) -> str:
    """'host:port' of a URL or bare address (as the pool reaches the service)."""
    if "://" in addr:
        u = urlsplit(addr)
        return f"{u.hostname}:{u.port or default_port}"
    return addr if ":" in addr else f"{addr}:{default_port}"


def _sessions() -> dict[str, list[str]]:
    """Session name -> its `-R` forward specs (box loopback port -> service)."""
    return {
        "control": [
            f"127.0.0.1:{BOX_TEMPORAL_PORT}:{_host_port(config.TEMPORAL_ADDRESS, 7233)}",
            f"127.0.0.1:{BOX_REDIS_PORT}:{_host_port(config.REDIS_URL, 6379)}",
        ],
        "data": [f"127.0.0.1:{BOX_S3_PORT}:{_host_port(config.BOX_FORWARD_S3, 9000)}"],
    }


def ssh_endpoint(inst: dict[str, Any]) -> tuple[str, int, str] | None:
    """(host, port, kind) to reach a running instance's sshd: its public IP when
    the host maps port 22 directly, else Vast's ssh proxy. None until running."""
    if inst.get("actual_status") != "running":
        return None
    ip = inst.get("public_ipaddr")
    mapped = (inst.get("ports") or {}).get("22/tcp") or []
    if ip and mapped and mapped[0].get("HostPort"):
        return str(ip).strip(), int(mapped[0]["HostPort"]), "direct"
    if inst.get("ssh_host") and inst.get("ssh_port"):
        return str(inst["ssh_host"]), int(inst["ssh_port"]), "proxy"
    return None


def public_key() -> str:
    """The OpenSSH public half of BOX_SSH_KEY (attached to every box)."""
    pub = config.BOX_SSH_KEY + ".pub"
    if os.path.exists(pub):
        return open(pub).read().strip()
    return subprocess.run(["ssh-keygen", "-y", "-f", config.BOX_SSH_KEY],
                          check=True, capture_output=True, text=True).stdout.strip()


class BoxLinks:
    """Keeps the SSH sessions of every live box running; the pool calls `sync`
    each reconcile and `drop` when it destroys a box."""

    def __init__(self, vast: VastClient) -> None:
        self.vast = vast
        self._tasks: dict[str, asyncio.Task] = {}
        self._pubkey = public_key()
        self._known_hosts = tempfile.mkdtemp(prefix="splatlab-known-hosts-")

    def sync(self, boxes: list[Any]) -> None:
        """Start links for new boxes; stop links of boxes that are gone."""
        live = {b.box_id: b for b in boxes}
        for box_id in list(self._tasks):
            if box_id not in live:
                self.drop(box_id)
        for box_id, b in live.items():
            t = self._tasks.get(box_id)
            if t is None or t.done():
                self._tasks[box_id] = asyncio.create_task(self._run(box_id, b.instance_id))

    def drop(self, box_id: str) -> None:
        t = self._tasks.pop(box_id, None)
        if t:
            t.cancel()

    async def _run(self, box_id: str, instance_id: int) -> None:
        # The key can only be attached once the instance exists; retry until it sticks.
        while True:
            try:
                await self.vast.attach_ssh_key(instance_id, self._pubkey)
                break
            except VastError as e:
                _log(f"{box_id}: attach ssh key failed, retrying: {e}")
                await asyncio.sleep(15)
        await asyncio.gather(*(self._session(box_id, instance_id, name, fwds)
                               for name, fwds in _sessions().items()))

    async def _session(self, box_id: str, instance_id: int, name: str,
                       forwards: list[str]) -> None:
        fails = 0                            # consecutive failed attempts
        while True:
            try:
                inst = await self.vast.get_instance(instance_id)
            except VastError as e:
                _log(f"{box_id}/{name}: instance lookup failed: {e}")
                inst = None
            ep = ssh_endpoint(inst or {})
            if not ep:                       # still loading / no ssh yet
                await asyncio.sleep(10)
                continue
            host, port, kind = ep
            started = time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                "ssh", "-N", "-T",
                "-i", config.BOX_SSH_KEY, "-o", "IdentitiesOnly=yes",
                "-o", "BatchMode=yes",
                # Boxes are ephemeral: trust a box's host key on first contact.
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"UserKnownHostsFile={self._known_hosts}/{box_id}",
                "-o", "ConnectTimeout=20",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
                "-o", "ExitOnForwardFailure=yes",
                *[a for f in forwards for a in ("-R", f)],
                "-p", str(port), f"{config.BOX_SSH_USER}@{host}",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                try:
                    await asyncio.wait_for(asyncio.shield(proc.wait()), _UP_AFTER)
                except asyncio.TimeoutError:
                    _log(f"{box_id}/{name}: up via {kind} {host}:{port}")
                    fails = 0
                    await proc.wait()
            finally:
                if proc.returncode is None:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), 5)
                    except asyncio.TimeoutError:
                        proc.kill()
            err = (await proc.stderr.read()).decode(errors="replace").strip()
            last = err.splitlines()[-1] if err else f"exit {proc.returncode}"
            lived = time.monotonic() - started
            fails = 0 if lived >= _UP_AFTER else fails + 1
            # sshd may still be starting on a fresh box: report the first failure
            # and then only every 10th, not every few seconds.
            if fails <= 1 or fails % 10 == 0:
                _log(f"{box_id}/{name}: {'down after %.0fs' % lived if not fails else 'connect failed'}"
                     f" via {kind} ({last[:160]}) — retrying")
            await asyncio.sleep(_RETRY_DELAY)
