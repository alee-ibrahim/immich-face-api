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

Memory (bounded and predictable): the faces per person are counted first (no embeddings), the float16 matrix is
allocated once at that size, and the embeddings are streamed in batches straight into it (a batch is normalised in
float32 and dropped). While a rebuild runs the old snapshot keeps serving, so the peak is about two matrices
(old + new, ~1 KiB per cached face each) plus one batch; the old one is released on the swap. ``max_faces`` caps
the total: when the named people have more faces than that, the per-person cap is lowered until the total fits
(every person keeps up to the same number of faces). ``idle_seconds`` > 0 makes the cache lazy (the admin UI):
loaded on first use, dropped after that long without use.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import struct
import time
from dataclasses import dataclass, field

import numpy as np

from . import metrics as m
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
    centroids: np.ndarray | None = None  # (people, 512) float32, L2-normalised mean of each person's cached faces
    index: dict[str, int] = field(default_factory=dict)  # person_id -> row of person_ids/names/centroids

    @property
    def faces(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def nbytes(self) -> int:
        c = self.centroids.nbytes if self.centroids is not None else 0
        return int(self.vectors.nbytes + self.person_idx.nbytes + c)


class Builder:
    """Builds a Snapshot from batches of (person_id, name, embedding) rows without holding the rows: each batch is
    decoded and normalised in float32, written into a float16 matrix allocated once for ``capacity`` faces, and
    added to the per-person sums for the centroids. If more rows arrive than expected (faces named between the count
    and the load), the matrix grows by a quarter."""

    def __init__(self, capacity: int = 0, people: int = 0):
        self.vectors = np.empty((max(capacity, 0), DIM), dtype=np.float16)
        self.person_idx = np.empty(max(capacity, 0), dtype=np.int32)
        self.sums = np.zeros((max(people, 0), DIM), dtype=np.float32)
        self.ids: dict[str, int] = {}
        self.names: list[str] = []
        self.n = 0

    def _grow(self, need: int) -> None:
        cap = max(need, int(self.vectors.shape[0] * 1.25) + 1024)
        vectors = np.empty((cap, DIM), dtype=np.float16)
        vectors[: self.n] = self.vectors[: self.n]
        person_idx = np.empty(cap, dtype=np.int32)
        person_idx[: self.n] = self.person_idx[: self.n]
        self.vectors, self.person_idx = vectors, person_idx

    def add(self, rows) -> None:
        if not rows:
            return
        b = len(rows)
        if self.n + b > self.vectors.shape[0]:
            self._grow(self.n + b)
        block = np.empty((b, DIM), dtype=np.float32)
        idx = np.empty(b, dtype=np.int32)
        for i, (pid, name, emb) in enumerate(rows):
            j = self.ids.get(pid)
            if j is None:
                j = self.ids[pid] = len(self.names)
                self.names.append(name or "")
            idx[i] = j
            block[i] = vector_from_pg(emb)
        block = _normalise(block)
        if len(self.names) > self.sums.shape[0]:
            sums = np.zeros((max(len(self.names), int(self.sums.shape[0] * 1.25) + 64), DIM), dtype=np.float32)
            sums[: self.sums.shape[0]] = self.sums
            self.sums = sums
        np.add.at(self.sums, idx, block)
        self.vectors[self.n : self.n + b] = block
        self.person_idx[self.n : self.n + b] = idx
        self.n += b

    def finish(self, signature: tuple = ()) -> Snapshot:
        n, people = self.n, len(self.names)
        # spare rows (estimate a little high): keep a view when few, so the trim never needs a second matrix
        spare = self.vectors.shape[0] - n
        vectors = self.vectors[:n] if spare <= max(1024, n // 50) else self.vectors[:n].copy()
        person_idx = self.person_idx[:n].copy()
        centroids = _normalise(self.sums[:people]) if people else np.empty((0, DIM), dtype=np.float32)
        self.vectors = self.person_idx = self.sums = None  # type: ignore[assignment]
        return Snapshot(vectors, person_idx, list(self.ids), self.names, signature, time.time(), centroids, self.ids)


def build(rows: list[tuple[str, str, object]], signature: tuple = ()) -> Snapshot:
    """rows: (person_id, name, embedding) -> Snapshot."""
    b = Builder(len(rows))
    b.add(rows)
    return b.finish(signature)


def per_person_cap(counts: list[int], per_person: int, max_faces: int) -> int:
    """The largest per-person cap <= per_person for which the cached faces (sum of min(count, cap)) fit max_faces
    (0 = no total cap). At least 1."""

    def total(cap: int) -> int:
        return sum(min(c, cap) for c in counts)

    if max_faces <= 0 or total(per_person) <= max_faces:
        return per_person
    lo, hi = 1, per_person
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if total(mid) <= max_faces:
            lo = mid
        else:
            hi = mid - 1
    return lo


def same_person_rule(snap: Snapshot | None, max_distance: float):
    """``decide``'s same_person for a snapshot: two people (already known to share a name) are one person split over
    several clusters when their centroids are within ``max_distance`` (cosine). 0 or no snapshot: never.

    Why centroids: a split person's clusters are the same face, so their mean embeddings are close (measured on one
    archive: most same-name split clusters within 0.5); namesakes are different faces (typically 0.8-1.1 apart).
    Between the two, the faces may be one person or two lookalikes; keeping them apart (ambiguous) is the safe side."""
    if snap is None or snap.centroids is None or max_distance <= 0:
        return None
    index = snap.index or {p: i for i, p in enumerate(snap.person_ids)}

    def same(a: str, b: str) -> bool:
        ia, ib = index.get(a), index.get(b)
        if ia is None or ib is None:
            return False
        return float(1.0 - snap.centroids[ia] @ snap.centroids[ib]) <= max_distance

    return same


def search(snap: Snapshot, queries: list[str], k: int, chunk: int = 16384) -> list[list[Neighbour]]:
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


def person_distances(snap: Snapshot, queries: list[str], person_ids: list[str]) -> list[dict[str, float]]:
    """For each query embedding: {person_id: min cosine distance to that person's cached faces}, for the given people
    that are in the cache (named, not hidden, with at least one face)."""
    rows = [snap.index.get(p) if snap.index else None for p in person_ids]
    wanted = [(p, r) for p, r in zip(person_ids, rows, strict=True) if r is not None]
    out: list[dict[str, float]] = [{} for _ in queries]
    if not queries or not wanted or snap.faces == 0:
        return out
    q = _normalise(np.stack([vector_from_pg(x) for x in queries]))
    for pid, row in wanted:
        block = snap.vectors[snap.person_idx == row].astype(np.float32)
        if block.shape[0] == 0:
            continue
        d = 1.0 - q @ block.T  # (m, faces of this person)
        best = d.min(axis=1)
        for i, v in enumerate(best):
            out[i][pid] = float(max(0.0, v))
    return out


class FaceCache:
    """Holds the current Snapshot; ``refresh`` rebuilds it from the database when the signature changed."""

    def __init__(
        self,
        db,
        per_person: int,
        refresh_seconds: int,
        knn: int,
        max_faces: int = 0,
        idle_seconds: int = 0,
        batch: int = 2048,
    ):
        self.db = db
        self.per_person = per_person
        self.refresh_seconds = refresh_seconds
        self.knn = knn
        self.max_faces = max_faces
        self.idle_seconds = idle_seconds  # > 0: lazy (load on first use, drop when idle)
        self.batch = batch
        self.snap: Snapshot | None = None
        self.last_used = 0.0
        self.effective_per_person = per_person
        self._lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return self.snap is not None

    @property
    def lazy(self) -> bool:
        return self.idle_seconds > 0

    async def _load(self, sig: tuple) -> Snapshot:
        counts = await self.db.named_face_counts()
        cap = per_person_cap(counts, self.per_person, self.max_faces)
        if cap < self.per_person:
            log.warning(
                "face cache capped",
                extra={"fields": {"max_faces": self.max_faces, "per_person": cap, "people": len(counts)}},
            )
        builder = Builder(sum(min(c, cap) for c in counts), len(counts))
        async with contextlib.aclosing(self.db.iter_named_faces(cap, self.batch)) as batches:
            async for rows in batches:  # one batch of rows in memory at a time
                await asyncio.to_thread(builder.add, rows)
        self.effective_per_person = cap
        return builder.finish(sig)

    async def refresh(self, force: bool = False) -> bool:
        async with self._lock:
            sig = await self.db.named_signature()
            if not force and self.snap is not None and sig == self.snap.signature:
                return False
            t0 = time.perf_counter()
            try:
                snap = await self._load(sig)
            except Exception:
                m.FACE_CACHE_REFRESH_ERRORS.inc()
                raise
            self.snap = snap  # swap; the old snapshot is released once running searches finish with it
            seconds = time.perf_counter() - t0
            _observe(snap, seconds, self.effective_per_person)
            log.info(
                "face cache loaded",
                extra={
                    "fields": {
                        "faces": snap.faces,
                        "people": len(snap.person_ids),
                        "per_person": self.effective_per_person,
                        "mib": round(snap.nbytes / 2**20, 1),
                        "seconds": round(seconds, 2),
                    }
                },
            )
            return True

    def unload(self) -> None:
        if self.snap is not None:
            self.snap = None
            _observe(None, None, self.effective_per_person)
            log.info("face cache unloaded (idle)", extra={"fields": {"idle_seconds": self.idle_seconds}})

    async def use(self) -> bool:
        """Mark the cache used; a lazy cache that is not loaded loads now. True when a snapshot is ready."""
        self.last_used = time.monotonic()
        if self.snap is None and self.lazy:
            try:
                await self.refresh()
            except Exception as exc:  # noqa: BLE001 - the caller falls back to the database search
                log.warning("face cache load failed", extra={"fields": {"error": type(exc).__name__}})
        return self.snap is not None

    async def loop(self) -> None:
        while True:
            if self.lazy and self.snap is not None and time.monotonic() - self.last_used > self.idle_seconds:
                self.unload()
            if not self.lazy or self.snap is not None:
                try:
                    await self.refresh()
                except Exception as exc:  # noqa: BLE001 - keep serving the old snapshot
                    log.warning("face cache refresh failed", extra={"fields": {"error": type(exc).__name__}})
            if self.lazy:
                await asyncio.sleep(min(self.refresh_seconds, self.idle_seconds))
            else:
                await asyncio.sleep(self.refresh_seconds if self.snap is not None else 15)

    async def nearest(self, queries: list[str]) -> list[list[Neighbour]]:
        snap = self.snap
        if snap is None:
            raise CacheNotReady()
        return await asyncio.to_thread(search, snap, queries, self.knn)

    async def distances(self, queries: list[str], person_ids: list[str]) -> list[dict[str, float]]:
        snap = self.snap
        if snap is None:
            raise CacheNotReady()
        return await asyncio.to_thread(person_distances, snap, queries, person_ids)


def _observe(snap: Snapshot | None, seconds: float | None, per_person: int) -> None:
    m.FACE_CACHE_FACES.set(snap.faces if snap else 0)
    m.FACE_CACHE_PEOPLE.set(len(snap.person_ids) if snap else 0)
    m.FACE_CACHE_BYTES.set(snap.nbytes if snap else 0)
    m.FACE_CACHE_PER_PERSON.set(per_person)
    if snap is not None:
        m.FACE_CACHE_LOADED.set(snap.loaded_at)
    if seconds is not None:
        m.FACE_CACHE_REFRESH.observe(seconds)


class CacheNotReady(RuntimeError):
    pass
