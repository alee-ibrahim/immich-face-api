"""Immich access for the labeler: reads from the database (read-only session, a narrow role is enough), writes ONLY
through the HTTP API with the library owner's API key.

Database privileges needed (SELECT): person, asset_face, face_search, version_history, and on asset the columns
id, "ownerId", "deletedAt", "originalPath" (only when folder rules are configured).

Immich 3.4 keeps one ``person`` row per user and person group (``personGroupId``); the API's person id is the group
id. A user with a "write" share who renames a person also renames the owner's row, so "the owner's row has no name"
and "no row of the group has a name" are both checked before anything is written.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
import numpy as np
import psycopg

from .match import Cluster, normalise, parse_vector
from .settings import LabelerSettings

log = logging.getLogger(__name__)

EXPECTED_COLUMNS = {
    ("person", "personGroupId"),
    ("person", "ownerId"),
    ("person", "name"),
    ("person", "isHidden"),
    ("asset_face", "id"),
    ("asset_face", "assetId"),
    ("asset_face", "personGroupId"),
    ("asset_face", "ownerId"),
    ("asset_face", "deletedAt"),
    ("asset_face", "isVisible"),
    ("asset_face", "assetDeletedAt"),
    ("asset_face", "assetVisibility"),
    ("asset_face", "boundingBoxX1"),
    ("asset_face", "boundingBoxY1"),
    ("asset_face", "boundingBoxX2"),
    ("asset_face", "boundingBoxY2"),
    ("face_search", "faceId"),
    ("face_search", "embedding"),
    ("asset", "id"),
    ("asset", "ownerId"),
    ("asset", "deletedAt"),
    ("asset", "originalPath"),
}

_FACE_FILTER = """af."deletedAt" IS NULL AND af."isVisible" AND af."assetDeletedAt" IS NULL
  AND af."assetVisibility" IN ('timeline', 'archive')"""

_CLUSTERS_SQL = f"""
WITH c AS (
  SELECT af."personGroupId" AS g, count(*) AS n, avg(fs.embedding) AS centroid
  FROM asset_face af JOIN face_search fs ON fs."faceId" = af.id
  WHERE af."ownerId" = %(owner)s AND af."personGroupId" IS NOT NULL AND {_FACE_FILTER}
  GROUP BY af."personGroupId"
)
SELECT c.g::text, c.n, c.centroid::text, p.name, p."isHidden",
       EXISTS (SELECT 1 FROM person o WHERE o."personGroupId" = c.g AND o.name <> '') AS any_named
FROM c JOIN person p ON p."personGroupId" = c.g AND p."ownerId" = %(owner)s
"""

_FOLDER_FACES_SQL = f"""
SELECT af.id::text, af."assetId"::text, a."originalPath", af."personGroupId"::text,
       greatest(0, (af."boundingBoxX2" - af."boundingBoxX1")) * greatest(0, (af."boundingBoxY2" - af."boundingBoxY1")),
       fs.embedding::text
FROM asset a
JOIN asset_face af ON af."assetId" = a.id
JOIN face_search fs ON fs."faceId" = af.id
WHERE a."ownerId" = %(owner)s AND a."deletedAt" IS NULL AND starts_with(a."originalPath", %(root)s) AND {_FACE_FILTER}
"""

_SAMPLES_SQL = f"""
SELECT g, emb FROM (
  SELECT af."personGroupId"::text AS g, fs.embedding::text AS emb,
         row_number() OVER (PARTITION BY af."personGroupId" ORDER BY md5(af.id::text)) AS rn
  FROM asset_face af JOIN face_search fs ON fs."faceId" = af.id
  WHERE af."ownerId" = %(owner)s AND af."personGroupId" = ANY(%(ids)s::uuid[]) AND {_FACE_FILTER}
) x WHERE rn <= %(k)s
"""


class ImmichError(RuntimeError):
    pass


@dataclass
class FolderFace:
    face_id: str
    asset_id: str
    path: str
    person_id: str | None
    area: int
    embedding: np.ndarray


def _conninfo(s: LabelerSettings, dbname: str, host: str, user: str, password: str | None, app: str) -> str:
    return psycopg.conninfo.make_conninfo(
        host=host,
        port=s.db_port,
        dbname=dbname,
        user=user,
        password=password,
        sslmode=s.db_sslmode,
        sslrootcert=s.db_sslrootcert,
        application_name=app,
        connect_timeout=10,
    )


class ImmichDB:
    def __init__(self, s: LabelerSettings):
        self.s = s
        self.conn = psycopg.connect(
            _conninfo(s, s.db_name, s.db_host, s.db_user, s.db_password, "immich-people-labeler"),
            autocommit=True,
            options="-c default_transaction_read_only=on -c statement_timeout=600000",
        )

    def close(self) -> None:
        self.conn.close()

    def check(self, need_paths: bool) -> str:
        tables = sorted({t for t, _ in EXPECTED_COLUMNS})
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT table_name, column_name FROM information_schema.column_privileges
                   WHERE table_schema = current_schema() AND table_name = ANY(%s) AND privilege_type = 'SELECT'
                   AND grantee IN (current_user, 'PUBLIC')""",
                (tables,),
            )
            have = set(cur.fetchall())
            want = {(t, c) for t, c in EXPECTED_COLUMNS if need_paths or t != "asset"}
            missing = sorted(f"{t}.{c}" for t, c in want - have)
            if missing:
                raise ImmichError("schema changed or SELECT not granted on: " + ", ".join(missing))
            cur.execute('SELECT version FROM version_history ORDER BY "createdAt" DESC LIMIT 1')
            row = cur.fetchone()
        version = (row[0] if row else "").lstrip("v")
        if self.s.expected_immich_version and version != self.s.expected_immich_version:
            raise ImmichError(f"Immich {version or 'unknown'} != pinned {self.s.expected_immich_version}")
        return version

    def clusters(self, owner: str) -> list[Cluster]:
        out = []
        with self.conn.cursor() as cur:
            cur.execute(_CLUSTERS_SQL, {"owner": owner})
            for pid, n, centroid, name, hidden, any_named in cur:
                # a name on any copy of the person (a curator's) counts as named
                shown = name or ("(named by another user)" if any_named else "")
                out.append(Cluster(pid, int(n), normalise(parse_vector(centroid)), shown, bool(hidden)))
        return out

    def folder_faces(self, owner: str, root: str) -> list[FolderFace]:
        out = []
        with self.conn.cursor() as cur:
            cur.execute(_FOLDER_FACES_SQL, {"owner": owner, "root": root})
            for fid, aid, path, pid, area, emb in cur:
                out.append(FolderFace(fid, aid, path, pid, int(area or 0), normalise(parse_vector(emb))))
        return out

    def samples(self, owner: str, person_ids: list[str], k: int) -> dict[str, np.ndarray]:
        rows: dict[str, list[np.ndarray]] = {}
        for i in range(0, len(person_ids), 500):
            with self.conn.cursor() as cur:
                cur.execute(_SAMPLES_SQL, {"owner": owner, "ids": person_ids[i : i + 500], "k": k})
                for g, emb in cur:
                    rows.setdefault(g, []).append(parse_vector(emb))
        return {g: normalise(np.stack(v)) for g, v in rows.items()}

    def group_names(self, person_id: str) -> list[tuple[str, str]]:
        with self.conn.cursor() as cur:
            cur.execute('SELECT "ownerId"::text, name FROM person WHERE "personGroupId" = %s', (person_id,))
            return list(cur.fetchall())


class ImmichAPI:
    def __init__(self, s: LabelerSettings):
        self.client = httpx.Client(
            base_url=s.immich_url,
            headers={"x-api-key": s.api_key, "accept": "application/json"},
            timeout=s.http_timeout,
        )

    def close(self) -> None:
        self.client.close()

    def _get(self, path: str) -> httpx.Response:
        r = self.client.get(path)
        if r.status_code != 200:
            raise ImmichError(f"GET {path}: HTTP {r.status_code}")
        return r

    def version(self) -> str:
        v = self._get("/api/server/version").json()
        return f"{v['major']}.{v['minor']}.{v['patch']}"

    def me(self) -> dict:
        me = self._get("/api/users/me").json()
        if not isinstance(me, dict) or "id" not in me or "email" not in me:
            raise ImmichError("unexpected /api/users/me answer")
        return me

    def person(self, person_id: str) -> dict:
        p = self._get(f"/api/people/{person_id}").json()
        if not isinstance(p, dict) or "name" not in p:
            raise ImmichError("unexpected person answer")
        return p

    def thumbnail(self, person_id: str) -> bytes | None:
        r = self.client.get(f"/api/people/{person_id}/thumbnail", headers={"accept": "image/*"})
        return r.content if r.status_code == 200 else None

    def set_name(self, person_id: str, name: str) -> dict:
        r = self.client.put(f"/api/people/{person_id}", json={"name": name})
        if r.status_code != 200:
            raise ImmichError(f"PUT /api/people/{person_id}: HTTP {r.status_code}")
        return r.json()
