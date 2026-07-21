"""Async wrapper over the Vast.ai REST API (https://console.vast.ai/api/v0).

Only the handful of calls the pool needs: search rentable offers, create an
instance from an offer, list/get our instances, destroy an instance. Auth is a
bearer token (VAST_API_KEY). Everything returns plain dicts/lists straight from
the API — the pool layer decides policy.

Vast quirks this hides:
- Offer search is `PUT /bundles/` with a JSON filter under `{"q": {...}}` whose
  leaf values are `{"<op>": value}` (e.g. `{"gte": 16}`); `order` is a list of
  `[field, "asc"|"desc"]`.
- Renting is `PUT /asks/{offer_id}/` and returns `{"success", "new_contract"}`
  where `new_contract` is the new instance id.
- List returns `{"instances": [...]}`; a single get returns `{"instances": {...}}`.
"""
from __future__ import annotations

from typing import Any

import httpx

from .. import config


class VastError(RuntimeError):
    """A Vast API call failed (non-2xx, or success=false in the body)."""


class VastClient:
    def __init__(self, api_key: str | None = None, base: str | None = None,
                 timeout: float = 30.0) -> None:
        key = api_key or config.VAST_API_KEY
        if not key:
            raise VastError("VAST_API_KEY is not set")
        self._base = (base or config.VAST_API_BASE).rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {key}",
                     "Accept": "application/json"},
        )

    async def __aenter__(self) -> "VastClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str,
                       json: dict | None = None) -> dict[str, Any]:
        url = f"{self._base}/{path.lstrip('/')}"
        try:
            resp = await self._client.request(method, url, json=json)
        except httpx.HTTPError as e:
            raise VastError(f"{method} {path}: {e}") from e
        if resp.status_code >= 400:
            raise VastError(f"{method} {path}: HTTP {resp.status_code}: {resp.text[:400]}")
        try:
            body = resp.json()
        except ValueError:
            return {}
        # Many endpoints report logical failure with success=false + 200 OK.
        if isinstance(body, dict) and body.get("success") is False:
            raise VastError(f"{method} {path}: {body.get('msg') or body}")
        return body

    # ---- offers -------------------------------------------------------------
    async def search_offers(self, *, gpu_name: str = "", min_gpu_ram_gb: float = 0,
                            min_reliability: float = 0, max_price: float = 0,
                            num_gpus: int = 1, limit: int = 20) -> list[dict[str, Any]]:
        """Return rentable on-demand offers matching the filter, cheapest first.

        Prices are `dph_total` ($/hr including the machine). GPU RAM in the API is
        `gpu_ram` (MB per GPU).
        """
        q: dict[str, Any] = {
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "type": "on-demand",
            "num_gpus": {"eq": num_gpus},
            "order": [["dph_total", "asc"]],
            "limit": limit,
        }
        if gpu_name:
            q["gpu_name"] = {"eq": gpu_name}
        if min_gpu_ram_gb:
            q["gpu_ram"] = {"gte": min_gpu_ram_gb * 1024}
        if min_reliability:
            q["reliability2"] = {"gte": min_reliability}
        if max_price:
            q["dph_total"] = {"lte": max_price}
        body = await self._request("PUT", "/bundles/", json={"q": q})
        return body.get("offers", []) if isinstance(body, dict) else []

    # ---- instances ----------------------------------------------------------
    async def create_instance(self, offer_id: int, *, image: str, disk_gb: int,
                              onstart: str, label: str,
                              env: dict[str, str] | None = None) -> int:
        """Rent `offer_id`; return the new instance id.

        `onstart` is the shell run on boot (we use it to bootstrap the worker).
        """
        payload: dict[str, Any] = {
            "client_id": "me",
            "image": image,
            "disk": disk_gb,
            "label": label,
            "runtype": "ssh",
            "onstart": onstart,
        }
        if env:
            payload["env"] = env
        body = await self._request("PUT", f"/asks/{offer_id}/", json=payload)
        new_id = body.get("new_contract")
        if not new_id:
            raise VastError(f"create_instance: no new_contract in response: {body}")
        return int(new_id)

    async def list_instances(self) -> list[dict[str, Any]]:
        body = await self._request("GET", "/instances/")
        insts = body.get("instances", []) if isinstance(body, dict) else []
        return insts if isinstance(insts, list) else []

    async def get_instance(self, instance_id: int) -> dict[str, Any] | None:
        body = await self._request("GET", f"/instances/{instance_id}/")
        inst = body.get("instances") if isinstance(body, dict) else None
        return inst if isinstance(inst, dict) else None

    async def destroy_instance(self, instance_id: int) -> None:
        await self._request("DELETE", f"/instances/{instance_id}/")
