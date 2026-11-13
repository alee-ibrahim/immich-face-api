"""In-process cache of the owner's named-face embeddings, searched with numpy.

Why: the exact SQL search reads every named face's 512-d embedding out of PostgreSQL for every detected face. With
~55,000 named faces that took about 4.4 s per face (embeddings are stored out of line), so a group photo ran into
the statement timeout. Holding the embeddings in memory makes a search a single matrix product (milliseconds).

What is cached: for every named, not hidden person of the owner, at most ``per_person`` faces (default 100), chosen
by a stable pseudo-random order of the face id, so the sample spreads over the person's photos and does not change
between refreshes. Most people have fewer faces than that and are cached completely. Stored as float16 (1 KiB per
face), compared in float32.

Decision ("nearest face wins") is unchanged: the k nearest cached faces, grouped by person, best distance first.
Trade-off: for people with more faces than the cap, the nearest face may be one that is not in the sample; the
sample of up to 100 faces of the same person nearly always contains one about as close.

Refresh: a cheap signature (count of named faces and the newest change time of people and faces) is checked every
``refresh_seconds``; the cache is rebuilt only when it changed. If a refresh fails, the old cache keeps serving.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from dataclasses import dataclass

import numpy as np

from .matching import Neighbour

log = logging.getLogger(__name__)

DIM = 512


def vector_from_pg(value) -> np.ndarray:
    """A pgvector value: binary send format (int16 dim, int16 unused, float4 big-endian ...) or text '[1,2,...]'."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        b = bytes(value)
        dim = struct.unpack_from(">H", b, 0)[0]
        return np.frombuffer(b, dtype=">f4", count=dim, offset=4).astype(np.float32)
    if isinstance(value, str):
        return np.array(value.strip("[]").split(","), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


@dataclass
class Snapshot:
    vectors: np.ndarray  # (n, 512) float16, L2-normalised
    person_idx: np.ndarray  # (n,) int32 index into person_ids/names
    person_ids: list[str]
    names: list[str]
    signature: tuple
    loaded_at: float

    @property
    def faces(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def nbytes(self) -> int:
        return int(self.vectors.nbytes + self.person_idx.nbytes)


def build(rows: list[tuple[str, str, object]], signature: tuple = ()) -> Snapshot:
    """rows: (person_id, name, embedding) -> Snapshot."""
    ids: dict[str, int] = {}
    names: list[str] = []
    person_idx = np.empty(len(rows), dtype=np.int32)
    vectors = np.empty((len(rows), DIM), dtype=np.float32)
    for i, (pid, name, emb) in enumerate(rows):
        if pid not in ids:
            ids[pid] = len(names)
            names.append(name or "")
        person_idx[i] = ids[pid]
        vectors[i] = vector_from_pg(emb)
    vectors = _normalise(vectors).astype(np.float16) if len(rows) else np.empty((0, DIM), dtype=np.float16)
    return Snapshot(vectors, person_idx, list(ids), names, signature, time.time())


def search(snap: Snapshot, queries: list[str], k: int, chunk: int = 65536) -> list[list[Neighbour]]:
    """For each query embedding (ML's JSON text), the k nearest cached faces as Neighbours (cosine distance)."""
    if not queries:
        return []
    if snap.faces == 0:
        return [[] for _ in queries]
    q = _normalise(np.stack([vector_from_pg(x) for x in queries]))  # (m, 512) float32
    if q.shape[1] != snap.vectors.shape[1]:
        raise ValueError(f"embedding size {q.shape[1]} != cached {snap.vectors.shape[1]} (different face model?)")
    best_d = np.full((len(queries), 0), np.inf, dtype=np.float32)
    best_i = np.empty((len(queries), 0), dtype=np.int64)
    for start in range(0, snap.faces, chunk):
        block = snap.vectors[start : start + chunk].astype(np.float32)
        d = 1.0 - q @ block.T  # (m, c) cosine distance
        kk = min(k, d.shape[1])
        part = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        best_d = np.concatenate([best_d, np.take_along_axis(d, part, axis=1)], axis=1)
        best_i = np.concatenate([best_i, part + start], axis=1)
        if best_d.shape[1] > k:
            keep = np.argpartition(best_d, k - 1, axis=1)[:, :k]
            best_d = np.take_along_axis(best_d, keep, axis=1)
            best_i = np.take_along_axis(best_i, keep, axis=1)
    out = []
    for row_d, row_i in zip(best_d, best_i, strict=True):
        order = np.argsort(row_d)
        out.append(
            [
                Neighbour(
                    person_id=snap.person_ids[snap.person_idx[i]],
                    name=snap.names[snap.person_idx[i]],
                    distance=float(max(0.0, row_d[j])),
                )
                for j in order
                for i in (int(row_i[j]),)
            ]
        )
    return out


class FaceCache:
    """Holds the current Snapshot; ``refresh`` rebuilds it from the database when the signature changed."""

    def __init__(self, db, per_person: int, refresh_seconds: int, knn: int):
        self.db = db
        self.per_person = per_person
        self.refresh_seconds = refresh_seconds
        self.knn = knn
        self.snap: Snapshot | None = None
        self._lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return self.snap is not None

    async def refresh(self, force: bool = False) -> bool:
        async with self._lock:
            sig = await self.db.named_signature()
            if not force and self.snap is not None and sig == self.snap.signature:
                return False
            t0 = time.perf_counter()
            rows = await self.db.named_faces(self.per_person)
            snap = await asyncio.to_thread(build, rows, sig)
            self.snap = snap
            log.info(
                "face cache loaded",
                extra={
                    "fields": {
                        "faces": snap.faces,
                        "people": len(snap.person_ids),
                        "mib": round(snap.nbytes / 2**20, 1),
                        "seconds": round(time.perf_counter() - t0, 2),
                    }
                },
            )
            return True

    async def loop(self) -> None:
        while True:
            try:
                await self.refresh()
            except Exception as exc:  # noqa: BLE001 - keep serving the old snapshot
                log.warning("face cache refresh failed", extra={"fields": {"error": type(exc).__name__}})
            await asyncio.sleep(self.refresh_seconds if self.snap is not None else 15)

    async def nearest(self, queries: list[str]) -> list[list[Neighbour]]:
        snap = self.snap
        if snap is None:
            raise CacheNotReady()
        return await asyncio.to_thread(search, snap, queries, self.knn)


class CacheNotReady(RuntimeError):
    pass
