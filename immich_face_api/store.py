"""The service's own small PostgreSQL ("store"): API keys created in the admin UI and hourly usage aggregates.

It is optional (``FACE_API_STORE_URL``). It is never Immich's database; Immich stays read-only. Keys look like
``ifa_<prefix>_<secret>``: the prefix is stored in clear (so the UI can show which key is which and lookups use an
index), the whole key only as a SHA-256 hash, and it is shown once when it is created.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib.resources import files

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from .usage import N_BUCKETS, Counts, UsageBuffer

log = logging.getLogger(__name__)

KEY_PREFIX = "ifa"


def new_key() -> tuple[str, str, str]:
    """(key, prefix, sha256 hex). The key is shown once and never stored."""
    prefix = secrets.token_hex(4)
    key = f"{KEY_PREFIX}_{prefix}_{secrets.token_urlsafe(32)}"
    return key, prefix, hashlib.sha256(key.encode()).hexdigest()


def split_key(token: str) -> str | None:
    """The prefix of a well-formed store key, else None (then no database lookup is made at all)."""
    parts = token.split("_", 2)
    if len(parts) != 3 or parts[0] != KEY_PREFIX or len(parts[1]) != 8 or len(parts[2]) < 32:
        return None
    if not all(ch in "0123456789abcdef" for ch in parts[1]):
        return None
    return parts[1]


@dataclass(frozen=True)
class StoreKey:
    id: str
    app: str
    secret_hash: str
    rate_limit_per_minute: int | None
    expires_at: datetime | None
    revoked: bool

    def valid(self, token: str, now: datetime) -> bool:
        ok = hmac.compare_digest(self.secret_hash, hashlib.sha256(token.encode()).hexdigest())
        return ok and not self.revoked and (self.expires_at is None or self.expires_at > now)


class Store:
    def __init__(self, url: str, password: str | None, pool_size: int = 4, key_cache_seconds: int = 30):
        kwargs = {"autocommit": True, "row_factory": dict_row}
        if password:
            kwargs["password"] = password
        self.pool = AsyncConnectionPool(url, min_size=1, max_size=pool_size, open=False, timeout=10, kwargs=kwargs)
        self.buffer = UsageBuffer()
        self._cache: dict[str, tuple[float, StoreKey | None]] = {}
        self._cache_ttl = key_cache_seconds
        self._used: set[str] = set()
        self._last_purge: float | None = None
        self._migrated = False

    # ---------------------------------------------------------------- lifecycle

    async def open(self, migrate: bool = True, wait: bool = True) -> None:
        """wait=False (the API): start even if the store is down; migrations are retried by flush_loop."""
        self._migrated = not migrate
        if wait:
            await self.pool.open(wait=True, timeout=30)
            await self.ensure_migrated()
        else:
            await self.pool.open(wait=False)
            with contextlib.suppress(Exception):
                await self.ensure_migrated()

    async def ensure_migrated(self) -> None:
        if not self._migrated:
            await self.migrate()
            self._migrated = True

    async def close(self) -> None:
        await self.pool.close()

    async def migrate(self) -> None:
        """Apply migrations/NNN_*.sql once each, in order, under an advisory lock (several pods may start at once)."""
        names = sorted(f.name for f in files(__package__).joinpath("migrations").iterdir() if f.name.endswith(".sql"))
        async with self.pool.connection() as conn:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock(724113)")
                await conn.execute(
                    "CREATE TABLE IF NOT EXISTS schema_migrations "
                    "(name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
                )
                done = {r["name"] for r in await (await conn.execute("SELECT name FROM schema_migrations")).fetchall()}
                for name in names:
                    if name in done:
                        continue
                    sql = files(__package__).joinpath("migrations", name).read_text(encoding="utf-8")
                    await conn.execute(sql)
                    await conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (name,))
                    log.info("store migration applied", extra={"fields": {"migration": name}})

    async def ping(self) -> bool:
        try:
            async with self.pool.connection() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception:  # noqa: BLE001
            return False

    # ---------------------------------------------------------------- keys (API side)

    async def lookup(self, prefix: str) -> StoreKey | None:
        """Key by prefix, cached for a few seconds (also misses), so a revoke takes effect within the TTL."""
        now = time.monotonic()
        hit = self._cache.get(prefix)
        if hit and now - hit[0] < self._cache_ttl:
            return hit[1]
        async with self.pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT id::text, app, secret_hash, rate_limit_per_minute, expires_at, revoked_at IS NOT NULL AS revoked "
                    "FROM api_keys WHERE prefix = %s",
                    (prefix,),
                )
            ).fetchone()
        key = StoreKey(**row) if row else None
        if len(self._cache) > 10_000:
            self._cache.clear()
        self._cache[prefix] = (now, key)
        return key

    def mark_used(self, key_id: str) -> None:
        self._used.add(key_id)

    # ---------------------------------------------------------------- flushing

    async def flush(self) -> None:
        data = self.buffer.drain()
        used, self._used = self._used, set()
        if not data and not used:
            return
        try:
            async with self.pool.connection() as conn:
                async with conn.transaction():
                    for (hour, client, endpoint), c in data.items():
                        await conn.execute(
                            """
                            INSERT INTO usage_hourly AS u (hour, client, endpoint, requests, client_errors,
                                server_errors, faces, matched, ambiguous, latency_ms_sum, latency_hist)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (hour, client, endpoint) DO UPDATE SET
                              requests = u.requests + EXCLUDED.requests,
                              client_errors = u.client_errors + EXCLUDED.client_errors,
                              server_errors = u.server_errors + EXCLUDED.server_errors,
                              faces = u.faces + EXCLUDED.faces,
                              matched = u.matched + EXCLUDED.matched,
                              ambiguous = u.ambiguous + EXCLUDED.ambiguous,
                              latency_ms_sum = u.latency_ms_sum + EXCLUDED.latency_ms_sum,
                              latency_hist = (SELECT array_agg(a + b ORDER BY i)
                                              FROM unnest(u.latency_hist, EXCLUDED.latency_hist)
                                                   WITH ORDINALITY AS t(a, b, i))
                            """,
                            (
                                hour,
                                client,
                                endpoint,
                                c.requests,
                                c.client_errors,
                                c.server_errors,
                                c.faces,
                                c.matched,
                                c.ambiguous,
                                c.latency_ms_sum,
                                c.latency_hist,
                            ),
                        )
                    if used:
                        await conn.execute(
                            "UPDATE api_keys SET last_used_at = now() WHERE id = ANY(%s::uuid[])", (list(used),)
                        )
        except (TimeoutError, psycopg.Error, OSError):
            self.buffer.restore(data)
            self._used |= used
            raise

    async def purge(self, retention_days: int) -> None:
        if self._last_purge is not None and time.monotonic() - self._last_purge < 6 * 3600:
            return
        async with self.pool.connection() as conn:
            await conn.execute(
                "DELETE FROM usage_hourly WHERE hour < now() - make_interval(days => %s)", (retention_days,)
            )
        self._last_purge = time.monotonic()

    async def flush_loop(self, retention_days: int, interval: float = 30.0) -> None:
        while True:
            await asyncio.sleep(interval)
            try:
                await self.ensure_migrated()
                await self.flush()
                await self.purge(retention_days)
            except Exception as exc:  # noqa: BLE001 - keep counting in memory, try again later
                log.warning("store flush failed", extra={"fields": {"error": type(exc).__name__}})

    # ---------------------------------------------------------------- keys (admin side)

    async def create_key(
        self,
        *,
        app: str,
        owner: str,
        label: str,
        rate_limit_per_minute: int | None,
        expires_at: datetime | None,
        actor: str,
    ) -> str:
        key, prefix, hashed = new_key()
        async with self.pool.connection() as conn:
            async with conn.transaction():
                row = await (
                    await conn.execute(
                        "INSERT INTO api_keys (app, owner, label, prefix, secret_hash, rate_limit_per_minute, "
                        "expires_at, created_by) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id::text",
                        (app, owner, label, prefix, hashed, rate_limit_per_minute, expires_at, actor),
                    )
                ).fetchone()
                await self._audit(conn, actor, "key.create", {"id": row["id"], "app": app, "prefix": prefix})
        return key

    async def revoke_key(self, key_id: str, actor: str) -> bool:
        async with self.pool.connection() as conn:
            async with conn.transaction():
                row = await (
                    await conn.execute(
                        "UPDATE api_keys SET revoked_at = now(), revoked_by = %s "
                        "WHERE id = %s AND revoked_at IS NULL RETURNING app, prefix",
                        (actor, key_id),
                    )
                ).fetchone()
                if row:
                    await self._audit(conn, actor, "key.revoke", {"id": key_id, **row})
        if row:
            self._cache.pop(row["prefix"], None)
        return bool(row)

    async def list_keys(self) -> list[dict]:
        async with self.pool.connection() as conn:
            return await (
                await conn.execute(
                    """
                    SELECT k.id::text, k.app, k.owner, k.label, k.prefix, k.rate_limit_per_minute, k.expires_at,
                           k.created_at, k.created_by, k.last_used_at, k.revoked_at, k.revoked_by,
                           coalesce((SELECT sum(requests) FROM usage_hourly u
                                     WHERE u.client = k.app AND u.hour > now() - interval '1 day'), 0) AS app_requests_24h
                    FROM api_keys k
                    ORDER BY k.revoked_at NULLS FIRST, k.created_at DESC
                    """
                )
            ).fetchall()

    async def audit(self, limit: int = 50) -> list[dict]:
        async with self.pool.connection() as conn:
            return await (
                await conn.execute(
                    "SELECT at, actor, action, detail FROM audit_log ORDER BY id DESC LIMIT %s", (limit,)
                )
            ).fetchall()

    async def _audit(self, conn, actor: str, action: str, detail: dict) -> None:
        await conn.execute(
            "INSERT INTO audit_log (actor, action, detail) VALUES (%s, %s, %s::jsonb)",
            (actor, action, json.dumps(detail, default=str)),
        )

    # ---------------------------------------------------------------- analytics

    async def usage(
        self, since: datetime, client: str | None = None, endpoint: str | None = "/v1/identify"
    ) -> list[dict]:
        """Raw hourly rows (one per hour, client, endpoint) since a time, optionally for one client."""
        async with self.pool.connection() as conn:
            return await (
                await conn.execute(
                    "SELECT hour, client, endpoint, requests, client_errors, server_errors, faces, matched, ambiguous, "
                    "latency_ms_sum, latency_hist FROM usage_hourly "
                    "WHERE hour >= %(since)s AND (%(c)s::text IS NULL OR client = %(c)s) "
                    "AND (%(e)s::text IS NULL OR endpoint = %(e)s) ORDER BY hour",
                    {"since": since, "c": client, "e": endpoint},
                )
            ).fetchall()

    async def usage_by_client(self, since: datetime) -> list[dict]:
        async with self.pool.connection() as conn:
            return await (
                await conn.execute(
                    """
                    SELECT client, sum(requests)::bigint requests, sum(client_errors)::bigint client_errors,
                           sum(server_errors)::bigint server_errors, sum(faces)::bigint faces,
                           sum(matched)::bigint matched, max(hour) last_hour
                    FROM usage_hourly WHERE hour >= %s AND endpoint = '/v1/identify'
                    GROUP BY client ORDER BY 2 DESC
                    """,
                    (since,),
                )
            ).fetchall()

    async def clients(self) -> list[str]:
        async with self.pool.connection() as conn:
            rows = await (
                await conn.execute(
                    "SELECT DISTINCT client FROM usage_hourly UNION SELECT DISTINCT app FROM api_keys ORDER BY 1"
                )
            ).fetchall()
        return [r["client"] for r in rows]


def _counts(r: dict) -> Counts:
    return Counts(
        requests=r["requests"],
        client_errors=r["client_errors"],
        server_errors=r["server_errors"],
        faces=r["faces"],
        matched=r["matched"],
        ambiguous=r["ambiguous"],
        latency_ms_sum=r["latency_ms_sum"],
        latency_hist=list(r["latency_hist"] or [0] * N_BUCKETS),
    )


def totals(rows: list[dict]) -> Counts:
    out = Counts()
    for r in rows:
        out.add(_counts(r))
    return out


def series(rows: list[dict], start: datetime, bucket: str) -> list[tuple[datetime, Counts]]:
    """Rows summed per hour or per day, with empty buckets filled in, from start to now."""
    step = timedelta(hours=1) if bucket == "hour" else timedelta(days=1)

    def floor(ts: datetime) -> datetime:
        ts = ts.astimezone(UTC)
        return (
            ts.replace(minute=0, second=0, microsecond=0)
            if bucket == "hour"
            else ts.replace(hour=0, minute=0, second=0, microsecond=0)
        )

    acc: dict[datetime, Counts] = {}
    for r in rows:
        acc.setdefault(floor(r["hour"]), Counts()).add(_counts(r))
    out, t, end = [], floor(start), floor(datetime.now(UTC))
    while t <= end:
        out.append((t, acc.get(t, Counts())))
        t += step
    return out


def since_for(range_name: str) -> tuple[datetime, str]:
    """('24h'|'7d'|'30d'|'90d') -> (start, bucket 'hour'|'day')."""
    now = datetime.now(UTC)
    return {
        "24h": (now - timedelta(hours=24), "hour"),
        "7d": (now - timedelta(days=7), "day"),
        "30d": (now - timedelta(days=30), "day"),
        "90d": (now - timedelta(days=90), "day"),
    }.get(range_name, (now - timedelta(days=7), "day"))
