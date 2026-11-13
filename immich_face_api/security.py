"""API-key check (hashed keys, constant time) and a per-client token-bucket rate limiter."""

from __future__ import annotations

import hashlib
import hmac
import math
import secrets
import threading
import time


def hash_key(key: str) -> str:
    """SHA-256 hex of an API key. Keys are long random strings, so a fast hash is enough (no dictionary attack)."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def new_key() -> str:
    return secrets.token_urlsafe(36)


class ApiKeys:
    def __init__(self, hashed: dict[str, bytes]):
        self._hashed = dict(hashed)

    def client_for(self, authorization: str | None, x_api_key: str | None = None) -> str | None:
        """Client name for the presented key (``Authorization: Bearer <key>`` or ``X-API-Key: <key>``), else None.

        Every stored digest is compared, with hmac.compare_digest, whatever the outcome, so the time taken does
        not tell which client (if any) almost matched.
        """
        token = token_from_headers(authorization, x_api_key)
        if token is None:
            return None
        presented = hashlib.sha256(token.encode("utf-8")).digest()
        found = None
        for name, digest in self._hashed.items():
            if hmac.compare_digest(presented, digest) and found is None:
                found = name
        return found


def token_from_headers(authorization: str | None, x_api_key: str | None) -> str | None:
    """The key from ``X-API-Key`` (wins if present) or ``Authorization: Bearer``; None if missing or oversized."""
    if x_api_key is not None:
        token = x_api_key.strip()
    elif authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer":
            return None
        token = token.strip()
    else:
        return None
    if not token or len(token) > 512:
        return None
    return token


class RateLimiter:
    """Token bucket per client: ``per_minute`` sustained, ``burst`` at once. In-memory, per replica."""

    def __init__(self, per_minute: int, burst: int, clock=time.monotonic):
        self.rate = per_minute / 60.0
        self.burst = float(burst)
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def take(self, client: str, per_minute: int | None = None) -> tuple[bool, int]:
        """(allowed, retry_after_seconds). ``per_minute`` overrides the default rate for this client (store keys);
        the burst then scales with it (at most the rate per minute)."""
        rate = per_minute / 60.0 if per_minute else self.rate
        burst = min(self.burst, float(per_minute)) if per_minute else self.burst
        if per_minute and per_minute > 60 * self.rate:
            burst = max(self.burst, per_minute / 6.0)
        now = self._clock()
        with self._lock:
            tokens, last = self._buckets.get(client, (burst, now))
            tokens = min(burst, tokens + (now - last) * rate)
            if tokens >= 1.0:
                self._buckets[client] = (tokens - 1.0, now)
                return True, 0
            self._buckets[client] = (tokens, now)
            return False, max(1, math.ceil((1.0 - tokens) / rate))
