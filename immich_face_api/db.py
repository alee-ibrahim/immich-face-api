"""Read-only access to Immich's PostgreSQL database.

Only these relations are read: ``person``, ``asset_face``, ``face_search``, ``version_history`` (and ``"user"``
columns ``id``/``email`` only when the owner is configured by e-mail). The queries are written for the Immich
schema of the pinned version (``EXPECTED_IMMICH_VERSION``); ``check_schema`` verifies the columns and types this
module relies on before the service reports ready.

Search is **exact**: the owner's named faces are materialised first and then ordered by cosine distance. Immich's
``face_index`` (VectorChord ``vchordrq``) is an approximate index searched before the WHERE filter, so with a
selective filter ("named people of one owner") an index scan can return fewer than k rows or none at all. An
exact scan of the named faces costs about 20 ms per 1,000 faces per query face on a small node, which is fine up
to some tens of thousands of named faces; beyond that, see README "Scaling".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import psycopg
from psycopg_pool import AsyncConnectionPool

from .config import Settings
from .matching import Neighbour

log = logging.getLogger(__name__)

# (table, column) -> type as printed by format_type(); the columns this service reads
EXPECTED_COLUMNS: dict[tuple[str, str], str] = {
    ("person", "personGroupId"): "uuid",
    ("person", "ownerId"): "uuid",
    ("person", "name"): "character varying",
    ("person", "isHidden"): "boolean",
    ("person", "updatedAt"): "timestamp with time zone",
    ("asset_face", "id"): "uuid",
    ("asset_face", "personGroupId"): "uuid",
    ("asset_face", "ownerId"): "uuid",
    ("asset_face", "deletedAt"): "timestamp with time zone",
    ("asset_face", "isVisible"): "boolean",
    ("asset_face", "assetDeletedAt"): "timestamp with time zone",
    ("asset_face", "assetVisibility"): "asset_visibility_enum",
    ("asset_face", "updatedAt"): "timestamp with time zone",
    ("face_search", "faceId"): "uuid",
    ("face_search", "embedding"): "vector(512)",
    ("version_history", "version"): "character varying",
    ("version_history", "createdAt"): "timestamp with time zone",
}
READ_TABLES = ("person", "asset_face", "face_search", "version_history")

# Faces that count: not deleted, visible, on a live asset that is not in the owner's locked folder.
_FACE_FILTER = """
  af."deletedAt" IS NULL AND af."isVisible" AND af."assetDeletedAt" IS NULL
  AND af."assetVisibility" IN ('timeline', 'archive')
"""

_KNN_SQL = (
    """
WITH cand AS MATERIALIZED (
  SELECT p."personGroupId"::text AS person_id, p.name, fs.embedding
  FROM person p
  JOIN asset_face af ON af."personGroupId" = p."personGroupId" AND af."ownerId" = p."ownerId"
  JOIN face_search fs ON fs."faceId" = af.id
  WHERE p."ownerId" = %(owner)s AND NOT p."isHidden" AND {name_filter} AND"""
    + _FACE_FILTER
    + """
),
q(idx, e) AS (SELECT * FROM unnest(%(idx)s::int[], %(emb)s::text[]::vector[]))
SELECT q.idx, r.person_id, r.name, r.distance
FROM q CROSS JOIN LATERAL (
  SELECT cand.person_id, cand.name, (cand.embedding <=> q.e)::float8 AS distance
  FROM cand ORDER BY cand.embedding <=> q.e LIMIT %(k)s
) r
ORDER BY q.idx, r.distance
"""
)

_PEOPLE_SQL = (
    """
SELECT p."personGroupId"::text AS person_id, p.name, count(af.id) AS faces
FROM person p
LEFT JOIN asset_face af ON af."personGroupId" = p."personGroupId" AND af."ownerId" = p."ownerId" AND"""
    + _FACE_FILTER
    + """
WHERE p."ownerId" = %(owner)s AND p.name <> '' AND NOT p."isHidden"
GROUP BY p."personGroupId", p.name
ORDER BY p.name, p."personGroupId"
"""
)


_SIGNATURE_SQL = (
    """
SELECT count(*), max(p."updatedAt"), max(af."updatedAt")
FROM person p
JOIN asset_face af ON af."personGroupId" = p."personGroupId" AND af."ownerId" = p."ownerId"
WHERE p."ownerId" = %(owner)s AND p.name <> '' AND NOT p."isHidden" AND"""
    + _FACE_FILTER
)

_NAMED_FACES_SQL = (
    """
SELECT person_id, name, embedding FROM (
  SELECT p."personGroupId"::text AS person_id, p.name, fs.embedding,
         row_number() OVER (PARTITION BY p."personGroupId" ORDER BY md5(af.id::text)) AS rn
  FROM person p
  JOIN asset_face af ON af."personGroupId" = p."personGroupId" AND af."ownerId" = p."ownerId"
  JOIN face_search fs ON fs."faceId" = af.id
  WHERE p."ownerId" = %(owner)s AND p.name <> '' AND NOT p."isHidden" AND"""
    + _FACE_FILTER
    + """
) x WHERE rn <= %(cap)s
"""
)


class SchemaMismatch(RuntimeError):
    pass


@dataclass
class DbStatus:
    immich_version: str | None = None
    owner_id: str | None = None


class Database:
    def __init__(self, settings: Settings):
        self.s = settings
        conninfo = psycopg.conninfo.make_conninfo(
            host=settings.db_host,
            port=settings.db_port,
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            sslmode=settings.db_sslmode,
            sslrootcert=settings.db_sslrootcert,
            application_name="immich-face-api",
            connect_timeout=5,
            # every transaction read-only, and no runaway query
            options="-c default_transaction_read_only=on -c statement_timeout=15000",
        )
        self.pool = AsyncConnectionPool(
            conninfo, min_size=1, max_size=settings.db_pool_size, open=False, timeout=10, kwargs={"autocommit": True}
        )
        self.status = DbStatus(owner_id=settings.owner_id)

    async def open(self) -> None:
        await self.pool.open(wait=False)

    async def close(self) -> None:
        await self.pool.close()

    async def check(self) -> DbStatus:
        """Version pin + schema shape + privileges + owner. Raises SchemaMismatch on any difference."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT c.relname, a.attname, format_type(a.atttypid, a.atttypmod)
                    FROM pg_attribute a
                    JOIN pg_class c ON c.oid = a.attrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = current_schema() AND c.relname = ANY(%s)
                      AND a.attnum > 0 AND NOT a.attisdropped
                    """,
                    (list(READ_TABLES),),
                )
                found = {(t, c): typ for t, c, typ in await cur.fetchall()}
                problems = [
                    f"{t}.{c}: expected {typ}, found {found.get((t, c), 'missing')}"
                    for (t, c), typ in EXPECTED_COLUMNS.items()
                    if found.get((t, c)) != typ
                ]
                if problems:
                    raise SchemaMismatch("Immich schema differs: " + "; ".join(problems))

                for table in READ_TABLES:
                    await cur.execute("SELECT has_table_privilege(%s, 'SELECT')", (table,))
                    if not (await cur.fetchone())[0]:
                        raise SchemaMismatch(f"no SELECT privilege on {table}")

                await cur.execute('SELECT version FROM version_history ORDER BY "createdAt" DESC LIMIT 1')
                row = await cur.fetchone()
                version = row[0].lstrip("v") if row else None
                if version != self.s.expected_immich_version:
                    raise SchemaMismatch(
                        f"Immich version {version or 'unknown'} != pinned {self.s.expected_immich_version}"
                    )
                self.status.immich_version = version

                if not self.s.owner_id:
                    await cur.execute(
                        'SELECT id::text FROM "user" WHERE lower(email) = lower(%s)', (self.s.owner_email,)
                    )
                    row = await cur.fetchone()
                    if not row:
                        raise SchemaMismatch("owner account not found")
                    self.status.owner_id = row[0]
        return self.status

    async def nearest(self, embeddings: list[str], *, include_unnamed: bool = False) -> list[list[Neighbour]]:
        """For each query embedding (Immich ML's JSON text form), its k nearest faces of the owner's people."""
        if not embeddings:
            return []
        name_filter = "TRUE" if include_unnamed else "p.name <> ''"
        sql = _KNN_SQL.format(name_filter=name_filter)
        params = {
            "owner": self.status.owner_id,
            "idx": list(range(len(embeddings))),
            "emb": embeddings,
            "k": self.s.knn,
        }
        out: list[list[Neighbour]] = [[] for _ in embeddings]
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                for idx, person_id, name, distance in await cur.fetchall():
                    out[idx].append(Neighbour(person_id=person_id, name=name or "", distance=float(distance)))
        return out

    async def named_signature(self) -> tuple:
        """Cheap change detector for the face cache: count of named faces and the newest change of people and faces."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SIGNATURE_SQL, {"owner": self.status.owner_id})
                return tuple(await cur.fetchone())

    async def named_faces(self, per_person: int) -> list[tuple]:
        """(person_id, name, embedding bytes) for up to per_person faces of each named, not hidden person, chosen by a
        stable pseudo-random order of the face id. Binary transfer (pgvector send format), longer statement timeout."""
        async with self.pool.connection() as conn:
            async with conn.transaction():
                await conn.execute("SET LOCAL statement_timeout = '180s'")
                async with conn.cursor(binary=True) as cur:
                    await cur.execute(_NAMED_FACES_SQL, {"owner": self.status.owner_id, "cap": per_person})
                    return await cur.fetchall()

    async def people(self) -> list[dict]:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_PEOPLE_SQL, {"owner": self.status.owner_id})
                return [
                    {"person_id": pid, "name": name, "face_count": int(faces)}
                    for pid, name, faces in await cur.fetchall()
                ]
