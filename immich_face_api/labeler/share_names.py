"""Copy the library owner's person names into the still-empty copies of the users the owner shares people with.

Immich 3.4 keeps one ``person`` row (and name) per user and person group. When a person is shared for the first time,
Immich's own trigger (``person_user_after_insert``) copies the sharer's name into the new copy; afterwards nothing
copies it again, so a name the owner sets later (or that one user sets) never reaches the other users' copies. This
command does what that trigger does, again: for every person the owner shares, a copy that is still EMPTY gets the
owner's name. A copy with any name is never touched; the empty check is part of the UPDATE itself.

Database role needed on Immich (column grants; examples/labeler/roles.sql): SELECT person ("ownerId",
"personGroupId", name), person_user ("personGroupId", "sharedById", "sharedWithId"), version_history (version,
"createdAt"); UPDATE person (name). Every copy written is in the audit table (target "copy") first; ``undo`` puts the
empty name back only where the copy still has the name this command wrote.
"""

from __future__ import annotations

import logging
from collections import Counter

import psycopg

from .immich import ImmichError, _conninfo
from .settings import LabelerSettings

log = logging.getLogger(__name__)

NEEDED = {
    ("person", "ownerId", "SELECT"),
    ("person", "personGroupId", "SELECT"),
    ("person", "name", "SELECT"),
    ("person", "name", "UPDATE"),
    ("person_user", "personGroupId", "SELECT"),
    ("person_user", "sharedById", "SELECT"),
    ("person_user", "sharedWithId", "SELECT"),
    ("version_history", "version", "SELECT"),
    ("version_history", "createdAt", "SELECT"),
}

CANDIDATES_SQL = """
SELECT c."personGroupId"::text, c."ownerId"::text, o.name
FROM person o
JOIN person_user pu ON pu."personGroupId" = o."personGroupId" AND pu."sharedById" = o."ownerId"
JOIN person c ON c."personGroupId" = o."personGroupId" AND c."ownerId" = pu."sharedWithId"
WHERE o."ownerId" = %(owner)s AND o.name <> '' AND c.name = ''
ORDER BY c."personGroupId", c."ownerId"
"""

COPY_SQL = """
UPDATE person c SET name = o.name
FROM person o
WHERE c."ownerId" = %(user)s AND c."personGroupId" = %(group)s AND c.name = ''
  AND o."ownerId" = %(owner)s AND o."personGroupId" = %(group)s AND o.name = %(name)s
RETURNING c.name
"""

COPIES_WITH_OWNER_SQL = """
SELECT c."personGroupId"::text, c."ownerId"::text, c.name, coalesce(o.name, '')
FROM person c LEFT JOIN person o ON o."personGroupId" = c."personGroupId" AND o."ownerId" = %(owner)s
WHERE c."ownerId" <> %(owner)s AND c.name <> ''
ORDER BY 1, 2
"""

UNDO_SQL = """
UPDATE person SET name = %(old)s
WHERE "ownerId" = %(user)s AND "personGroupId" = %(group)s AND name = %(new)s
RETURNING 1
"""


class NameWriter:
    """A read-write connection to Immich used ONLY for the copies (the labeler's other reads stay read-only)."""

    def __init__(self, s: LabelerSettings):
        self.s = s
        self.conn = psycopg.connect(
            _conninfo(s, s.db_name, s.db_host, s.db_user, s.db_password, "immich-people-labeler-share-names"),
            autocommit=True,
            options="-c statement_timeout=120000",
        )

    def close(self) -> None:
        self.conn.close()

    def check(self) -> str:
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT table_name, column_name, privilege_type FROM information_schema.column_privileges
                   WHERE table_schema = current_schema() AND grantee IN (current_user, 'PUBLIC')
                   AND table_name IN ('person', 'person_user', 'version_history')"""
            )
            have = set(cur.fetchall())
            missing = sorted(f"{t}.{c}:{p}" for t, c, p in NEEDED - have)
            if missing:
                raise ImmichError("schema changed or not granted: " + ", ".join(missing))
            cur.execute('SELECT version FROM version_history ORDER BY "createdAt" DESC LIMIT 1')
            row = cur.fetchone()
        version = (row[0] if row else "").lstrip("v")
        if self.s.expected_immich_version and version != self.s.expected_immich_version:
            raise ImmichError(f"Immich {version or 'unknown'} != pinned {self.s.expected_immich_version}")
        return version

    def candidates(self, owner: str) -> list[tuple[str, str, str]]:
        return self.conn.execute(CANDIDATES_SQL, {"owner": owner}).fetchall()

    def copy(self, owner: str, user: str, group: str, name: str) -> bool:
        return bool(
            self.conn.execute(COPY_SQL, {"owner": owner, "user": user, "group": group, "name": name}).fetchall()
        )

    def copies_with_owner(self, owner: str) -> list[tuple[str, str, str, str]]:
        """(group, user, copy name, owner name) of every named copy of another user (tidy-names --copies)."""
        return self.conn.execute(COPIES_WITH_OWNER_SQL, {"owner": owner}).fetchall()

    def rename(self, user: str, group: str, old: str, new: str) -> bool:
        """Rename one user's copy, only if it still has ``old`` (tidy-names --copies)."""
        return self.revert(user, group, old, new)

    def revert(self, user: str, group: str, new: str, old: str) -> bool:
        return bool(self.conn.execute(UNDO_SQL, {"user": user, "group": group, "new": new, "old": old}).fetchall())


def share_names(writer: NameWriter, store, run_id: str, owner: str, dry_run: bool, limit: int) -> Counter:
    c: Counter = Counter()
    rows = writer.candidates(owner)
    c["candidates"] = len(rows)
    for group, user, name in rows:
        if c["copied"] + c["would_copy"] >= limit:
            c["over_cap"] += 1
            continue
        if dry_run:
            c["would_copy"] += 1
            continue
        aid = store.copy_pending(run_id, group, user, name)
        if writer.copy(owner, user, group, name):
            store.audit_set(aid, "applied")
            c["copied"] += 1
        else:
            store.audit_set(aid, "skipped", "copy no longer empty, or the owner's name changed")
            c["skipped"] += 1
    return c


def undo_copies(writer: NameWriter, store, undo_run: str, rows: list[dict]) -> Counter:
    c: Counter = Counter()
    for row in rows:
        if writer.revert(row["owner_id"], row["person_id"], row["new_name"], row["old_name"]):
            store.audit_undone(row["id"], undo_run)
            c["undone"] += 1
        else:
            c["changed_since"] += 1
    return c
