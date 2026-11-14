"""People change log: renames, merges, splits (faces moved), deletions and new people, as an ordered event feed.

Why a snapshot: Immich 3.4 records only *deletions* in its audit tables (``person_audit``, ``person_group_audit``,
``asset_face_audit``: which row was deleted, and when). A merge is "move the faces, then delete the source person",
a split is "move some faces to another person", a rename is an UPDATE of ``person.name``; none of these leaves a trace
of where faces went or what a person was called before. So the API keeps its own lightweight snapshot in the store
(``people_snapshot``: the owner's named, not hidden people with their names; ``face_snapshot``: which of them each
counted face belongs to; ids only) and turns the differences between the snapshot and Immich into events.

How a check works (one process at a time: an advisory lock in the store; every ``changes_interval`` seconds):

1. In one REPEATABLE READ transaction on Immich: the owner's people (id, name, hidden), and the faces that changed
   since the last check (``asset_face.updateId`` is a time-ordered UUIDv7 that Immich's trigger renews on every update,
   with an index; re-read with an overlap of ``changes_overlap`` seconds so late commits are not missed). Every
   ``changes_full_hours`` hours, instead, all faces of the named people (a full reconcile).
2. In one transaction on the store: compare with the snapshot, write events, update the snapshot.

Events (``people_changes``), in this order within one check:

* ``created``     a person appears in the named set (named for the first time, or un-hidden);
* ``renamed``     its name changed (``old_name`` -> ``name``);
* ``faces_moved`` faces that belonged to a named person now belong to another person (``to_person_id``; NULL =
                  unassigned); ``faces`` = how many of the tracked faces moved in this window;
* ``merged``      a named person no longer exists and (most of) its moved faces went to one person: the merge target;
* ``deleted``     a named person no longer exists and its faces went nowhere (``reason`` deleted), or it was hidden
                  (``hidden``) or its name was cleared (``unnamed``).

The snapshot is first built ("seeded") without events; changes before that are unknown (``started_at``).
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from . import metrics as m

log = logging.getLogger(__name__)

LOCK_ID = 724114
TYPES = ("created", "renamed", "faces_moved", "merged", "deleted")

# Faces that count, as in db._FACE_FILTER (kept in step with it)
_COUNTS = """(af."deletedAt" IS NULL AND af."isVisible" AND af."assetDeletedAt" IS NULL
  AND af."assetVisibility" IN ('timeline', 'archive'))"""

PERSONS_SQL = 'SELECT "personGroupId"::text, name, "isHidden" FROM person WHERE "ownerId" = %(owner)s'

# all counted faces of the owner's named, not hidden people (optionally only some people); COPY TO STDOUT
_TRACKED_FACES = (
    """
SELECT af.id, af."personGroupId"
FROM person p
JOIN asset_face af ON af."personGroupId" = p."personGroupId" AND af."ownerId" = p."ownerId"
WHERE p."ownerId" = {owner} AND p.name <> '' AND NOT p."isHidden" AND {people} AND """
    + _COUNTS
)

# faces of the owner updated since a UUIDv7 bound (index on "updateId"); COPY TO STDOUT
_CHANGED_FACES = (
    """
SELECT af.id, af."personGroupId", """
    + _COUNTS
    + """
FROM asset_face af
WHERE af."updateId" > {after}::uuid AND af."ownerId" = {owner}
"""
)

_FACES_BY_ID = (
    """
SELECT af.id::text, af."personGroupId"::text, """
    + _COUNTS
    + """
FROM asset_face af WHERE af.id = ANY(%(ids)s::uuid[]) AND af."ownerId" = %(owner)s
"""
)


def uuid7_floor(ts: datetime) -> str:
    """The smallest UUIDv7 of the millisecond ``ts``: Immich's "updateId" > this = updated at or after ``ts``."""
    h = f"{int(ts.timestamp() * 1000):012x}"
    return f"{h[:8]}-{h[8:12]}-0000-0000-000000000000"


def _lit(value: str) -> str:
    """A UUID as an SQL literal (validated, so it can be put into a COPY statement, which takes no parameters)."""
    import uuid

    return "'" + str(uuid.UUID(value)) + "'"


@dataclass
class Event:
    type: str
    person_id: str
    name: str | None = None
    to_person_id: str | None = None
    old_name: str | None = None
    to_name: str | None = None
    faces: int | None = None
    reason: str | None = None


def diff_events(
    prev: dict[str, str],
    persons: dict[str, tuple[str, bool]],
    moves: list[tuple[str, str | None, int]],
) -> list[Event]:
    """Events from the previous named set ``prev`` {id: name}, all of the owner's people now ``persons``
    {id: (name, hidden)} and the face moves seen in this window ``moves`` [(from_id, to_id or None, count)]."""
    tracked = {p: n for p, (n, hidden) in persons.items() if n and not hidden}
    gone = sorted(p for p in prev if p not in persons)
    dropped = sorted(p for p in prev if p in persons and p not in tracked)
    events: list[Event] = []
    for p in sorted(p for p in tracked if p not in prev):
        events.append(Event("created", p, name=tracked[p]))
    for p in sorted(p for p in tracked if p in prev and prev[p] != tracked[p]):
        events.append(Event("renamed", p, name=tracked[p], old_name=prev[p]))
    merges: dict[str, tuple[int, str]] = {}
    for g in gone:
        dests = [(n, to) for frm, to, n in moves if frm == g and to]
        if dests:
            merges[g] = max(dests, key=lambda d: (d[0], d[1]))
    for frm, to, n in sorted(moves, key=lambda mv: (mv[0], mv[1] or "", mv[2])):
        if frm not in prev or (frm in merges and merges[frm][1] == to):
            continue
        events.append(
            Event("faces_moved", frm, name=prev[frm], to_person_id=to, to_name=tracked.get(to) if to else None, faces=n)
        )
    for g, (n, to) in merges.items():
        events.append(Event("merged", g, name=prev[g], to_person_id=to, to_name=tracked.get(to), faces=n))
    for g in gone:
        if g not in merges:
            events.append(Event("deleted", g, name=prev[g], reason="deleted"))
    for d in dropped:
        events.append(Event("deleted", d, name=prev[d], reason="hidden" if persons[d][1] else "unnamed"))
    return events


class ChangeTracker:
    """Runs the checks (see the module docstring). ``db``: Database (Immich, read-only); ``store``: Store."""

    def __init__(
        self,
        db,
        store,
        *,
        interval: int = 300,
        overlap: int = 600,
        full_hours: int = 24,
        retention_days: int = 365,
        ready=lambda: True,
    ):
        self.db, self.store, self.ready = db, store, ready
        self.interval, self.overlap, self.full_hours, self.retention_days = (
            interval,
            overlap,
            full_hours,
            retention_days,
        )

    async def loop(self) -> None:
        await asyncio.sleep(random.uniform(5, 20))  # noqa: S311 - jitter, not security
        while True:
            try:
                # only while the version pin and schema check pass: after an Immich upgrade, stop rather than diff
                # against a database whose meaning may have changed
                if self.db.status.owner_id and self.ready():
                    await self.check()
            except Exception as exc:  # noqa: BLE001 - try again next time; the old events stay valid
                m.CHANGES_ERRORS.inc()
                log.warning("people change check failed", extra={"fields": {"error": f"{type(exc).__name__}: {exc}"}})
                await self._record_error(f"{type(exc).__name__}: {exc}"[:500])
            await asyncio.sleep(self.interval)

    async def _record_error(self, message: str) -> None:
        try:
            async with self.store.pool.connection() as conn:
                await conn.execute(
                    "UPDATE people_tracker SET last_error = %s, last_error_at = now() WHERE id = 1", (message,)
                )
        except Exception:  # noqa: BLE001, S110 - the store may be the problem
            pass

    async def check(self, full: bool | None = None) -> dict | None:
        """One check. None when another process holds the lock. ``full``: force (True) or skip (False) a full
        reconcile; default: when the last one is older than ``full_hours``."""
        await self.store.ensure_migrated()
        t0 = time.perf_counter()
        async with self.store.pool.connection() as s:
            async with s.transaction():
                got = await (await s.execute("SELECT pg_try_advisory_xact_lock(%s) AS ok", (LOCK_ID,))).fetchone()
                if not got["ok"]:
                    return None
                st = await (await s.execute("SELECT * FROM people_tracker WHERE id = 1")).fetchone()
                owner = self.db.status.owner_id
                seed = st is None or str(st["owner_id"]) != owner
                if not seed and full is None:
                    full = st["last_full_at"] < datetime.now(st["last_full_at"].tzinfo) - timedelta(
                        hours=self.full_hours
                    )
                result = await self._run(s, st, owner, seed=seed, full=bool(full))
                await self._purge(s)
                ms = int((time.perf_counter() - t0) * 1000)
                await s.execute(
                    "UPDATE people_tracker SET last_cycle_ms = %s, last_error = NULL, last_error_at = NULL WHERE id = 1",
                    (ms,),
                )
        m.CHANGES_CHECKS.labels(result["mode"]).inc()
        for e in result["events"]:
            m.CHANGES_EVENTS.labels(e.type).inc()
        m.CHANGES_LAST_CHECK.set(time.time())
        log.info(
            "people change check",
            extra={
                "fields": {
                    "mode": result["mode"],
                    "changed_faces": result["changed"],
                    "events": len(result["events"]),
                    "ms": ms,
                }
            },
        )
        return {**result, "ms": ms}

    async def _run(self, s, st, owner: str, *, seed: bool, full: bool) -> dict:
        async with self.db.pool.connection() as i:
            async with i.transaction():
                await i.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
                await i.execute("SELECT set_config('statement_timeout', '600s', true)")
                immich_now = (await (await i.execute("SELECT now()")).fetchone())[0]
                rows = await (await i.execute(PERSONS_SQL, {"owner": owner})).fetchall()
                persons = {pid: (name or "", bool(hidden)) for pid, name, hidden in rows}
                tracked = {p: n for p, (n, hidden) in persons.items() if n and not hidden}

                await s.execute("CREATE TEMP TABLE trk (person_id uuid PRIMARY KEY, name text NOT NULL) ON COMMIT DROP")
                async with s.cursor() as cur:
                    async with cur.copy("COPY trk (person_id, name) FROM STDIN") as cp:
                        for p, n in tracked.items():
                            await cp.write_row((p, n))

                if seed:
                    return await self._seed(s, i, st, owner, immich_now)

                prev = {
                    str(r["person_id"]): r["name"]
                    for r in await (await s.execute("SELECT person_id, name FROM people_snapshot")).fetchall()
                }
                await s.execute(
                    "CREATE TEMP TABLE chg (face_id uuid PRIMARY KEY, person_id uuid, counts boolean NOT NULL DEFAULT TRUE) "
                    "ON COMMIT DROP"
                )
                if full:
                    await self._pipe(
                        i, s, _TRACKED_FACES.format(owner=_lit(owner), people="TRUE"), "chg (face_id, person_id)"
                    )
                    await s.execute("ANALYZE chg")
                    await self._lookup_missing(i, s, owner)
                else:
                    after = uuid7_floor(st["checked_at"] - timedelta(seconds=self.overlap))
                    await self._pipe(
                        i,
                        s,
                        _CHANGED_FACES.format(after=_lit(after), owner=_lit(owner)),
                        "chg (face_id, person_id, counts)",
                    )
                    new_people = [p for p in tracked if p not in prev]
                    if new_people:
                        await s.execute("CREATE TEMP TABLE newf (face_id uuid, person_id uuid) ON COMMIT DROP")
                        people = 'p."personGroupId" IN (' + ",".join(_lit(p) for p in new_people) + ")"
                        await self._pipe(i, s, _TRACKED_FACES.format(owner=_lit(owner), people=people), "newf")
        # Immich transaction closed; the rest is store-only
        await s.execute("ANALYZE chg")
        changed = (await (await s.execute("SELECT count(*) AS n FROM chg")).fetchone())["n"]
        moves = [
            (r["from_id"], r["to_id"], r["n"])
            for r in await (
                await s.execute(
                    """
                    SELECT s.person_id::text AS from_id, c.person_id::text AS to_id, count(*)::int AS n
                    FROM chg c JOIN face_snapshot s ON s.face_id = c.face_id
                    WHERE c.counts AND c.person_id IS DISTINCT FROM s.person_id
                    GROUP BY 1, 2
                    """
                )
            ).fetchall()
        ]
        events = diff_events(prev, persons, moves)

        # update the face snapshot
        await s.execute(
            """
            DELETE FROM face_snapshot f USING chg c WHERE f.face_id = c.face_id
              AND (NOT c.counts OR c.person_id IS NULL OR NOT EXISTS (SELECT 1 FROM trk WHERE trk.person_id = c.person_id))
            """
        )
        await s.execute(
            """
            INSERT INTO face_snapshot (face_id, person_id)
            SELECT c.face_id, c.person_id FROM chg c
            JOIN trk ON trk.person_id = c.person_id
            LEFT JOIN face_snapshot f ON f.face_id = c.face_id
            WHERE c.counts AND (f.face_id IS NULL OR f.person_id <> c.person_id)  -- only real changes (no row locks)
            ON CONFLICT (face_id) DO UPDATE SET person_id = EXCLUDED.person_id
            """
        )
        if not full and any(p not in prev for p in tracked):
            await s.execute(
                """
                INSERT INTO face_snapshot (face_id, person_id)
                SELECT n.face_id, n.person_id FROM newf n LEFT JOIN face_snapshot f ON f.face_id = n.face_id
                WHERE f.face_id IS NULL OR f.person_id <> n.person_id
                ON CONFLICT (face_id) DO UPDATE SET person_id = EXCLUDED.person_id
                """
            )
        untracked = [p for p in prev if p not in tracked]
        if untracked:
            await s.execute("DELETE FROM face_snapshot WHERE person_id = ANY(%s::uuid[])", (untracked,))
        if full:  # anything left over from people that left the named set while a check was missed
            await s.execute(
                "DELETE FROM face_snapshot f WHERE NOT EXISTS (SELECT 1 FROM trk WHERE trk.person_id = f.person_id)"
            )
        await self._update_people(s)

        if events:
            async with s.cursor() as cur:
                await cur.executemany(
                    """
                    INSERT INTO people_changes (at, window_start, type, person_id, to_person_id, name, old_name,
                                                to_name, faces, reason)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            immich_now,
                            st["checked_at"],
                            e.type,
                            e.person_id,
                            e.to_person_id,
                            e.name,
                            e.old_name,
                            e.to_name,
                            e.faces,
                            e.reason,
                        )
                        for e in events
                    ],
                )
        await s.execute(
            "UPDATE people_tracker SET checked_at = %s, last_changed_faces = %s, last_events = %s"
            + (", last_full_at = now()" if full else "")
            + " WHERE id = 1",
            (immich_now, changed, len(events)),
        )
        return {"mode": "full" if full else "incremental", "changed": changed, "events": events}

    async def _seed(self, s, i, st, owner: str, immich_now) -> dict:
        """First check, or the owner changed: build the snapshot without events. Old events (another owner's people)
        are dropped and old cursors then ask clients to resync."""
        await s.execute("TRUNCATE face_snapshot, people_snapshot")
        await self._pipe(
            i, s, _TRACKED_FACES.format(owner=_lit(owner), people="TRUE"), "face_snapshot (face_id, person_id)"
        )
        if st is not None:
            await s.execute(
                "UPDATE people_tracker SET purged_through = greatest(purged_through, "
                "(SELECT coalesce(max(id), 0) FROM people_changes)) WHERE id = 1"
            )
            await s.execute("DELETE FROM people_changes")
        await s.execute(
            """
            INSERT INTO people_tracker (id, owner_id, started_at, checked_at, last_full_at)
            VALUES (1, %(o)s, %(t)s, %(t)s, now())
            ON CONFLICT (id) DO UPDATE SET owner_id = %(o)s, started_at = %(t)s, checked_at = %(t)s,
              last_full_at = now(), last_changed_faces = NULL, last_events = 0
            """,
            {"o": owner, "t": immich_now},
        )
        await self._update_people(s)
        n = (await (await s.execute("SELECT count(*) AS n FROM face_snapshot")).fetchone())["n"]
        log.info("people change log seeded", extra={"fields": {"faces": n}})
        return {"mode": "seed", "changed": n, "events": []}

    async def _update_people(self, s) -> None:
        await s.execute(
            "DELETE FROM people_snapshot p WHERE NOT EXISTS (SELECT 1 FROM trk WHERE trk.person_id = p.person_id)"
        )
        await s.execute(
            """
            INSERT INTO people_snapshot (person_id, name) SELECT person_id, name FROM trk
            ON CONFLICT (person_id) DO UPDATE SET name = EXCLUDED.name, updated_at = now()
              WHERE people_snapshot.name <> EXCLUDED.name
            """
        )
        await s.execute(
            """
            WITH c AS (SELECT person_id, count(*)::int AS n FROM face_snapshot GROUP BY person_id)
            UPDATE people_snapshot p SET faces = coalesce(c.n, 0)
            FROM people_snapshot p2 LEFT JOIN c ON c.person_id = p2.person_id
            WHERE p.person_id = p2.person_id AND p.faces <> coalesce(c.n, 0)
            """
        )

    @staticmethod
    async def _pipe(i, s, select_sql: str, target: str) -> None:
        """Stream a SELECT on Immich into a store table with COPY on both sides (no rows in Python memory)."""
        async with i.cursor() as icur, s.cursor() as scur:
            async with icur.copy(f"COPY ({select_sql}) TO STDOUT") as src:
                async with scur.copy(f"COPY {target} FROM STDIN") as dst:
                    async for data in src:
                        await dst.write(data)

    @staticmethod
    async def _lookup_missing(i, s, owner: str, batch: int = 10_000) -> None:
        """Full reconcile: faces still in the snapshot but no longer among the named people's faces. Look up where
        they are now (moved to an unnamed person, unassigned, trashed or deleted)."""
        missing = [
            str(r["face_id"])
            for r in await (
                await s.execute(
                    "SELECT face_id FROM face_snapshot f WHERE NOT EXISTS (SELECT 1 FROM chg c WHERE c.face_id = f.face_id)"
                )
            ).fetchall()
        ]
        for k in range(0, len(missing), batch):
            ids = missing[k : k + batch]
            found = {
                fid: (pid, counts)
                for fid, pid, counts in await (await i.execute(_FACES_BY_ID, {"ids": ids, "owner": owner})).fetchall()
            }
            async with s.cursor() as cur:
                async with cur.copy("COPY chg (face_id, person_id, counts) FROM STDIN") as cp:
                    for fid in ids:
                        pid, counts = found.get(fid, (None, False))
                        await cp.write_row((fid, pid, counts))

    async def _purge(self, s) -> None:
        row = await (
            await s.execute(
                "WITH d AS (DELETE FROM people_changes WHERE at < now() - make_interval(days => %s) RETURNING id) "
                "SELECT max(id) AS m FROM d",
                (self.retention_days,),
            )
        ).fetchone()
        if row and row["m"]:
            await s.execute(
                "UPDATE people_tracker SET purged_through = greatest(purged_through, %s) WHERE id = 1", (row["m"],)
            )


# ------------------------------------------------------------------------------------------- reading (API and admin)

CURSOR_PREFIX = "pc_"


def cursor_of(event_id: int) -> str:
    return f"{CURSOR_PREFIX}{event_id}"


def parse_since(value: str | None) -> tuple[int | None, datetime | None]:
    """``since`` = a cursor (``pc_<n>``) or an ISO-8601 timestamp with a time zone -> (after_id, after_time).
    Raises ValueError."""
    if not value:
        return 0, None
    if value.startswith(CURSOR_PREFIX) and value[len(CURSOR_PREFIX) :].isdigit() and len(value) < 24:
        return int(value[len(CURSOR_PREFIX) :]), None
    # a "+" in an unencoded query string arrives as a space
    ts = datetime.fromisoformat(value.strip().replace(" ", "+").replace("Z", "+00:00").replace("z", "+00:00"))
    if ts.tzinfo is None:
        raise ValueError("timestamp needs a time zone, e.g. 2026-01-31T00:00:00Z")
    return None, ts


def event_json(r: dict) -> dict:
    """One people_changes row in the public shape."""
    t = r["type"]
    pid = str(r["person_id"])
    to = str(r["to_person_id"]) if r["to_person_id"] else None
    base = {"cursor": cursor_of(r["id"]), "type": t, "at": _iso(r["at"]), "after": _iso(r["window_start"])}
    if t == "created":
        return {**base, "person_id": pid, "name": r["name"]}
    if t == "renamed":
        return {**base, "person_id": pid, "old_name": r["old_name"], "new_name": r["name"]}
    if t == "faces_moved":
        return {
            **base,
            "from_person_id": pid,
            "from_name": r["name"],
            "to_person_id": to,
            "to_name": r["to_name"],
            "count": r["faces"],
        }
    if t == "merged":
        return {
            **base,
            "from_person_id": pid,
            "from_name": r["name"],
            "into_person_id": to,
            "into_name": r["to_name"],
            "faces": r["faces"],
        }
    return {**base, "person_id": pid, "name": r["name"], "reason": r["reason"]}


def _iso(ts: datetime | None) -> str | None:
    return ts.astimezone(UTC).isoformat().replace("+00:00", "Z") if ts else None


async def tracker_status(store) -> dict | None:
    async with store.pool.connection() as conn:
        row = await (await conn.execute("SELECT * FROM people_tracker WHERE id = 1")).fetchone()
        if not row:
            return None
        extra = await (
            await conn.execute(
                "SELECT (SELECT count(*) FROM people_snapshot) AS people, "
                "(SELECT count(*) FROM face_snapshot) AS faces, "
                "(SELECT coalesce(max(id), 0) FROM people_changes) AS last_id, "
                "(SELECT count(*) FROM people_changes) AS events"
            )
        ).fetchone()
    return {**row, **extra}


async def latest_cursor(store) -> str | None:
    """The cursor of the newest event (or the purge mark), for a client that has just re-synced fully."""
    async with store.pool.connection() as conn:
        row = await (
            await conn.execute(
                "SELECT greatest((SELECT coalesce(max(id), 0) FROM people_changes), "
                "(SELECT purged_through FROM people_tracker WHERE id = 1)) AS id, "
                "EXISTS (SELECT 1 FROM people_tracker) AS ready"
            )
        ).fetchone()
    return cursor_of(row["id"] or 0) if row["ready"] else None


async def list_changes(
    store,
    *,
    after_id: int | None = None,
    after_time: datetime | None = None,
    limit: int = 500,
    types: tuple[str, ...] = TYPES,
    newest_first: bool = False,
    person_id: str | None = None,
    before_id: int | None = None,
) -> list[dict]:
    async with store.pool.connection() as conn:
        return await (
            await conn.execute(
                f"""
                SELECT * FROM people_changes
                WHERE (%(after_id)s::bigint IS NULL OR id > %(after_id)s)
                  AND (%(after_time)s::timestamptz IS NULL OR at > %(after_time)s)
                  AND type = ANY(%(types)s)
                  AND (%(pid)s::uuid IS NULL OR person_id = %(pid)s::uuid OR to_person_id = %(pid)s::uuid)
                  AND (%(before_id)s::bigint IS NULL OR id < %(before_id)s)
                ORDER BY id {"DESC" if newest_first else "ASC"} LIMIT %(limit)s
                """,
                {
                    "after_id": after_id,
                    "after_time": after_time,
                    "types": list(types),
                    "limit": limit,
                    "pid": person_id,
                    "before_id": before_id,
                },
            )
        ).fetchall()


async def fates(store, person_ids: list[str]) -> dict[str, dict]:
    """The last merged / deleted event of each person id (what happened to a person that no longer exists)."""
    if not person_ids:
        return {}
    async with store.pool.connection() as conn:
        rows = await (
            await conn.execute(
                """
                SELECT DISTINCT ON (person_id) * FROM people_changes
                WHERE person_id = ANY(%s::uuid[]) AND type IN ('merged', 'deleted')
                ORDER BY person_id, id DESC
                """,
                (person_ids,),
            )
        ).fetchall()
    return {str(r["person_id"]): r for r in rows}


async def resolve(db, store, person_ids: list[str], max_hops: int = 10) -> dict[str, dict]:
    """Where each person id stands now: current (named, visible), hidden, unnamed, merged (followed to the person it
    was merged into, also over several merges), deleted or unknown (never seen by the change log)."""
    out: dict[str, dict] = {}
    pending = {p: p for p in dict.fromkeys(person_ids)}  # original id -> id to look up
    chain: dict[str, list[str]] = {p: [] for p in pending}
    for _ in range(max_hops + 1):
        if not pending:
            break
        found = await db.persons_by_id(sorted(set(pending.values())))
        gone = sorted({cur for cur in pending.values() if cur not in found})
        fate = await fates(store, gone) if store is not None else {}
        nxt: dict[str, str] = {}
        for orig, cur in pending.items():
            if cur in found:
                name, hidden = found[cur]
                status = "hidden" if hidden else ("current" if name else "unnamed")
                out[orig] = {"status": status, "current_person_id": cur, "name": name or None}
            elif cur in fate and fate[cur]["type"] == "merged" and fate[cur]["to_person_id"]:
                target = str(fate[cur]["to_person_id"])
                if target in chain[orig] or target == orig:
                    out[orig] = {"status": "deleted", "current_person_id": None, "name": None}
                else:
                    chain[orig].append(target)
                    nxt[orig] = target
            elif cur in fate:
                out[orig] = {"status": "deleted", "current_person_id": None, "name": fate[cur]["name"]}
            else:
                out[orig] = {"status": "unknown", "current_person_id": None, "name": None}
        pending = nxt
    for orig in pending:  # too many hops
        out[orig] = {"status": "unknown", "current_person_id": None, "name": None}
    for orig, info in out.items():
        if chain[orig] and info["status"] != "unknown":
            info["merged_via"] = chain[orig]
            if info["status"] in ("current", "hidden", "unnamed"):
                info["target_status"] = info["status"]
                info["status"] = "merged"
    return out
