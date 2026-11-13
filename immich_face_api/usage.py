"""Usage aggregation: counts per hour, client and endpoint, plus a latency histogram. Nothing per request is kept,
and nothing that identifies a person: no images, embeddings, names, IPs or keys."""

from __future__ import annotations

import bisect
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime

# Upper bounds (ms) of the latency buckets; one more bucket counts everything above the last bound.
LATENCY_BOUNDS_MS = (50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 8000, 13000, 20000, 30000, 60000)
N_BUCKETS = len(LATENCY_BOUNDS_MS) + 1


def hour_of(ts: datetime) -> datetime:
    return ts.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


@dataclass
class Counts:
    requests: int = 0
    client_errors: int = 0
    server_errors: int = 0
    faces: int = 0
    matched: int = 0
    ambiguous: int = 0
    latency_ms_sum: int = 0
    latency_hist: list[int] = field(default_factory=lambda: [0] * N_BUCKETS)

    def add(self, other: Counts) -> None:
        for name in ("requests", "client_errors", "server_errors", "faces", "matched", "ambiguous", "latency_ms_sum"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.latency_hist = [a + b for a, b in zip(self.latency_hist, other.latency_hist, strict=True)]


class UsageBuffer:
    """Thread-safe in-memory aggregate, drained by the store's flush loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[tuple[datetime, str, str], Counts] = {}

    def record(
        self,
        client: str,
        endpoint: str,
        status: int,
        ms: float,
        faces: int = 0,
        matched: int = 0,
        ambiguous: int = 0,
        now: datetime | None = None,
    ) -> None:
        key = (hour_of(now or datetime.now(UTC)), client[:64], endpoint[:64])
        with self._lock:
            c = self._data.setdefault(key, Counts())
            c.requests += 1
            c.client_errors += int(400 <= status < 500)
            c.server_errors += int(status >= 500)
            c.faces += faces
            c.matched += matched
            c.ambiguous += ambiguous
            c.latency_ms_sum += int(ms)
            c.latency_hist[bisect.bisect_left(LATENCY_BOUNDS_MS, ms)] += 1

    def drain(self) -> dict[tuple[datetime, str, str], Counts]:
        with self._lock:
            data, self._data = self._data, {}
        return data

    def restore(self, data: dict[tuple[datetime, str, str], Counts]) -> None:
        """Put back what could not be written (database down), merged with what came in meanwhile."""
        with self._lock:
            for key, counts in data.items():
                self._data.setdefault(key, Counts()).add(counts)


def percentile(hist: list[int], q: float) -> float | None:
    """Approximate percentile (ms) from bucket counts, interpolating linearly inside the bucket."""
    total = sum(hist)
    if not total:
        return None
    rank = q * total
    seen = 0
    for i, n in enumerate(hist):
        if n and seen + n >= rank:
            lo = LATENCY_BOUNDS_MS[i - 1] if i > 0 else 0
            hi = LATENCY_BOUNDS_MS[i] if i < len(LATENCY_BOUNDS_MS) else LATENCY_BOUNDS_MS[-1] * 2
            return lo + (hi - lo) * ((rank - seen) / n)
        seen += n
    return float(LATENCY_BOUNDS_MS[-1])
