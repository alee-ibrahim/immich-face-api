"""Read one photo's preview from Immich's own API (admin "Label subjects" only).

Needs ``FACE_API_IMMICH_API_URL`` (in-cluster URL of immich-server) and ``FACE_API_IMMICH_API_KEY`` (an API key of
the library owner with only the ``asset.view`` permission). The preview is the image Immich detected the faces on;
it is kept in memory while it is analysed or passed to a signed-in curator, and never written anywhere.
"""

from __future__ import annotations

import re

import httpx

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
MAX_PREVIEW_BYTES = 25 * 1024 * 1024


class ImmichError(RuntimeError):
    pass


class ImmichAssets:
    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0):
        self.client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout, headers={"x-api-key": api_key})

    async def close(self) -> None:
        await self.client.aclose()

    async def preview(self, asset_id: str) -> tuple[bytes, str] | None:
        """(bytes, content type) of the asset's preview, or None if Immich does not have it (deleted, offline, not
        the key owner's). Raises ImmichError when Immich is unreachable or answers with an error."""
        if not _UUID.fullmatch(asset_id):
            return None
        try:
            async with self.client.stream("GET", f"/api/assets/{asset_id}/thumbnail", params={"size": "preview"}) as r:
                if r.status_code in (400, 403, 404):
                    return None
                if r.status_code != 200:
                    raise ImmichError(f"Immich answered {r.status_code}")
                chunks, total = [], 0
                async for chunk in r.aiter_bytes():
                    total += len(chunk)
                    if total > MAX_PREVIEW_BYTES:
                        raise ImmichError("preview too large")
                    chunks.append(chunk)
                return b"".join(chunks), r.headers.get("content-type", "application/octet-stream").split(";")[0]
        except httpx.HTTPError as exc:
            raise ImmichError(f"Immich unreachable: {type(exc).__name__}") from exc
