"""Request log and face crops: what each identify request asked and which faces were missed, for curators.

Written to the store (the service's own PostgreSQL) **after** the response, by a background writer fed through a
bounded in-memory queue: identify never waits for the store, and if the store is down or slow the entries are
dropped (counted in ``face_api_request_log_dropped_total``), exactly like usage counts keep the API independent of it.

What is kept (see README "Request log and face crops"):

* per request: time, app, status / error code, latency, image bytes / size / format, counts of faces, matched,
  unknown, ambiguous and near-threshold faces;
* per face (at most ``log_max_faces``): box, detection score, kind, the match (person id, name, distance) and the
  nearest named people **also beyond the threshold**, so curators see near-misses;
* with ``store_crops`` = ``unknown`` or ``all``: a small JPEG crop of the face (``crop_size`` px, a new image without
  metadata) and its embedding (float16) for "find similar in Immich". Never the uploaded photo.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import metrics as m

log = logging.getLogger(__name__)

KINDS = ("unknown", "ambiguous", "near", "matched")
CROP_MARGIN = 0.25  # extra context around the detector's box, per side, as a fraction of the box size


def kind_of(entry: dict, max_distance: float, near_margin: float) -> str:
    """unknown (no named person within max_distance), ambiguous (close runner-up), near (match close to the
    threshold), or matched."""
    if not entry.get("name"):
        return "unknown"
    if entry.get("ambiguous"):
        return "ambiguous"
    if entry.get("distance") is not None and entry["distance"] >= max_distance - near_margin:
        return "near"
    return "matched"


def wants_crop(mode: str, kind: str) -> bool:
    return mode == "all" or (mode == "unknown" and kind != "matched")


def make_crop(image: Image.Image, box: dict, size: int, quality: int = 80) -> bytes:
    """A JPEG of one face: the detector's box (pixels of ``image``) plus a margin, at most ``size`` px on the long
    edge. Pasted into a new image, so nothing of the upload's metadata can be carried along."""
    x1, y1, x2, y2 = (float(box[k]) for k in ("x1", "y1", "x2", "y2"))
    mx, my = (x2 - x1) * CROP_MARGIN, (y2 - y1) * CROP_MARGIN
    left = min(max(0, int(x1 - mx)), image.width - 1)
    top = min(max(0, int(y1 - my)), image.height - 1)
    right = max(left + 1, min(image.width, int(x2 + mx + 0.5)))
    bottom = max(top + 1, min(image.height, int(y2 + my + 0.5)))
    face = image.crop((left, top, right, bottom))
    face.thumbnail((size, size), Image.Resampling.LANCZOS)
    clean = Image.new("RGB", face.size)
    clean.paste(face)
    out = io.BytesIO()
    clean.save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()


def embedding_bytes(embedding: str) -> bytes:
    """ML's JSON text embedding -> float16 bytes (1 KiB for 512 dimensions)."""
    return np.array(embedding.strip("[]").split(","), dtype=np.float32).astype("<f2").tobytes()


def embedding_text(data: bytes) -> str:
    """float16 bytes -> pgvector text '[...]'."""
    v = np.frombuffer(data, dtype="<f2").astype(np.float32)
    return "[" + ",".join(f"{x:.6g}" for x in v) + "]"


@dataclass
class FaceRecord:
    idx: int
    kind: str
    box: dict
    detection_score: float
    person_id: str | None
    name: str | None
    distance: float | None
    candidates: list[dict]
    crop: bytes | None = None
    embedding: bytes | None = None


@dataclass
class RequestRecord:
    at: float  # unix time
    rid: str
    client: str
    endpoint: str
    status: int
    error_code: str | None
    latency_ms: int
    image_bytes: int | None = None
    image_w: int | None = None
    image_h: int | None = None
    image_format: str | None = None
    faces: int = 0
    matched: int = 0
    unknown: int = 0
    ambiguous: int = 0
    near: int = 0
    face_records: list[FaceRecord] = field(default_factory=list)


class RequestLogWriter:
    """Bounded queue + one background task that inserts records into the store. Never blocks the caller."""

    def __init__(self, store, maxsize: int = 500):
        self.store = store
        self.queue: asyncio.Queue[RequestRecord] = asyncio.Queue(maxsize=maxsize)
        self._last_warning = 0.0

    def submit(self, rec: RequestRecord) -> bool:
        try:
            self.queue.put_nowait(rec)
            return True
        except asyncio.QueueFull:
            m.REQUEST_LOG_DROPPED.labels("queue_full").inc()
            return False

    async def loop(self) -> None:
        while True:
            rec = await self.queue.get()
            batch = [rec]
            while len(batch) < 50 and not self.queue.empty():
                batch.append(self.queue.get_nowait())
            try:
                await self.store.ensure_migrated()
                await self.store.insert_requests(batch)
                m.REQUEST_LOG_WRITTEN.inc(len(batch))
            except Exception as exc:  # noqa: BLE001 - best effort: the API keeps answering
                m.REQUEST_LOG_DROPPED.labels("store_error").inc(len(batch))
                now = time.monotonic()
                if now - self._last_warning > 60:
                    self._last_warning = now
                    log.warning("request log write failed", extra={"fields": {"error": type(exc).__name__}})

    async def drain(self, timeout: float = 5.0) -> None:
        """Write what is queued (shutdown)."""
        batch = []
        while not self.queue.empty():
            batch.append(self.queue.get_nowait())
        if batch:
            await asyncio.wait_for(self.store.insert_requests(batch), timeout)
