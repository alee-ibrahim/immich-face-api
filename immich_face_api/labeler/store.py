"""The labeler's own state, in a PostgreSQL database of its own (never Immich's tables): runs and their reports,
proposals, the audit log of every name written (the basis of undo), and caches for web pages, portraits and their
embeddings. Tables are created when missing."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime

import psycopg
from psycopg.types.json import Jsonb

from .immich import _conninfo
from .settings import LabelerSettings

SCHEMA = """
CREATE TABLE IF NOT EXISTS labeler_run (
  run_id text PRIMARY KEY,
  command text NOT NULL,
  started_at timestamptz NOT NULL DEFAULT now(),
  finished_at timestamptz,
  status text NOT NULL DEFAULT 'running',
  stats jsonb NOT NULL DEFAULT '{}',
  report_html_gz bytea,
  proposals_csv_gz bytea
);
CREATE TABLE IF NOT EXISTS labeler_proposal (
  run_id text NOT NULL REFERENCES labeler_run(run_id) ON DELETE CASCADE,
  person_id uuid NOT NULL,
  band text NOT NULL,
  score real NOT NULL,
  name text NOT NULL,
  alt_names jsonb NOT NULL DEFAULT '{}',
  identity_key text NOT NULL,
  distance real NOT NULL,
  margin real NOT NULL,
  faces int NOT NULL,
  evidence jsonb NOT NULL DEFAULT '[]',
  PRIMARY KEY (run_id, person_id)
);
CREATE TABLE IF NOT EXISTS labeler_audit (
  id bigserial PRIMARY KEY,
  run_id text NOT NULL,
  source_run_id text,
  person_id uuid NOT NULL,
  owner_id uuid NOT NULL,
  old_name text NOT NULL,
  new_name text NOT NULL,
  band text,
  score real,
  evidence jsonb NOT NULL DEFAULT '[]',
  status text NOT NULL,
  note text,
  created_at timestamptz NOT NULL DEFAULT now(),
  applied_at timestamptz,
  undone_at timestamptz,
  undo_run_id text
);
CREATE INDEX IF NOT EXISTS labeler_audit_person ON labeler_audit (person_id);
CREATE INDEX IF NOT EXISTS labeler_audit_run ON labeler_audit (run_id);
CREATE TABLE IF NOT EXISTS labeler_http_cache (
  url text PRIMARY KEY,
  fetched_at timestamptz NOT NULL,
  status int NOT NULL,
  etag text,
  last_modified text,
  content_type text,
  body bytea
);
CREATE TABLE IF NOT EXISTS labeler_embedding_cache (
  image_sha256 text NOT NULL,
  settings text NOT NULL,
  faces jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (image_sha256, settings)
);
"""


def new_run_id() -> str:
    import secrets

    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(2)


class Store:
    def __init__(self, s: LabelerSettings):
        self.s = s
        self.conn = psycopg.connect(
            _conninfo(
                s,
                s.state_db_name,
                s.state_db_host or s.db_host,
                s.state_db_user or s.db_user,
                s.state_db_password if s.state_db_password is not None else s.db_password,
                "immich-people-labeler-state",
            ),
            autocommit=True,
        )
        with self.conn.cursor() as cur:
            cur.execute(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # --- runs
    def start_run(self, run_id: str, command: str) -> None:
        self.conn.execute("INSERT INTO labeler_run (run_id, command) VALUES (%s, %s)", (run_id, command))

    def finish_run(self, run_id: str, status: str, stats: dict, html: str | None = None, csv: str | None = None):
        self.conn.execute(
            """UPDATE labeler_run SET finished_at = now(), status = %s, stats = %s,
               report_html_gz = coalesce(%s, report_html_gz), proposals_csv_gz = coalesce(%s, proposals_csv_gz)
               WHERE run_id = %s""",
            (
                status,
                Jsonb(stats),
                gzip.compress(html.encode()) if html else None,
                gzip.compress(csv.encode()) if csv else None,
                run_id,
            ),
        )
        # keep the last N report blobs; proposals and audit stay
        self.conn.execute(
            """UPDATE labeler_run SET report_html_gz = NULL, proposals_csv_gz = NULL
               WHERE report_html_gz IS NOT NULL AND run_id NOT IN (
                 SELECT run_id FROM labeler_run WHERE report_html_gz IS NOT NULL
                 ORDER BY started_at DESC LIMIT %s)""",
            (self.s.keep_reports,),
        )

    def resolve_run(self, run_id: str) -> str:
        if run_id == "latest":
            row = self.conn.execute(
                "SELECT run_id FROM labeler_run WHERE command = 'run' AND status = 'ok' ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            if not row:
                raise LookupError("no finished run yet")
            return row[0]
        if not self.conn.execute("SELECT 1 FROM labeler_run WHERE run_id = %s", (run_id,)).fetchone():
            raise LookupError(f"unknown run {run_id}")
        return run_id

    def report(self, run_id: str) -> tuple[str | None, str | None]:
        row = self.conn.execute(
            "SELECT report_html_gz, proposals_csv_gz FROM labeler_run WHERE run_id = %s", (run_id,)
        ).fetchone()
        if not row:
            raise LookupError(f"unknown run {run_id}")
        return tuple(gzip.decompress(bytes(b)).decode() if b else None for b in row)

    def recent_runs(self, limit: int = 10) -> list[tuple]:
        return self.conn.execute(
            "SELECT run_id, command, status, started_at, finished_at, stats FROM labeler_run "
            "ORDER BY started_at DESC LIMIT %s",
            (limit,),
        ).fetchall()

    # --- proposals
    def save_proposals(self, run_id: str, rows: list[dict]) -> None:
        with self.conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO labeler_proposal (run_id, person_id, band, score, name, alt_names, identity_key,
                   distance, margin, faces, evidence)
                   VALUES (%(run_id)s, %(person_id)s, %(band)s, %(score)s, %(name)s, %(alt_names)s,
                   %(identity_key)s, %(distance)s, %(margin)s, %(faces)s, %(evidence)s)""",
                [
                    {**r, "run_id": run_id, "alt_names": Jsonb(r["alt_names"]), "evidence": Jsonb(r["evidence"])}
                    for r in rows
                ],
            )

    def proposals(self, run_id: str, bands: list[str]) -> list[dict]:
        cur = self.conn.execute(
            """SELECT person_id::text, band, score, name, alt_names, evidence FROM labeler_proposal
               WHERE run_id = %s AND band = ANY(%s) ORDER BY score DESC, person_id""",
            (run_id, bands),
        )
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]

    # --- audit
    def audit_pending(self, run_id: str, source_run: str | None, p: dict, owner: str, old: str) -> int:
        row = self.conn.execute(
            """INSERT INTO labeler_audit (run_id, source_run_id, person_id, owner_id, old_name, new_name, band, score,
               evidence, status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending') RETURNING id""",
            (
                run_id,
                source_run,
                p["person_id"],
                owner,
                old,
                p["name"],
                p.get("band"),
                p.get("score"),
                Jsonb(p.get("evidence", [])),
            ),
        ).fetchone()
        return row[0]

    def audit_skip(self, run_id: str, source_run: str | None, p: dict, owner: str, old: str, note: str) -> None:
        self.conn.execute(
            """INSERT INTO labeler_audit (run_id, source_run_id, person_id, owner_id, old_name, new_name, band, score,
               evidence, status, note) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'skipped', %s)""",
            (
                run_id,
                source_run,
                p["person_id"],
                owner,
                old,
                p["name"],
                p.get("band"),
                p.get("score"),
                Jsonb(p.get("evidence", [])),
                note,
            ),
        )

    def audit_set(self, audit_id: int, status: str, note: str | None = None) -> None:
        col = {"applied": "applied_at", "undone": "undone_at"}.get(status)
        stamp = f", {col} = now()" if col else ""
        self.conn.execute(
            f"UPDATE labeler_audit SET status = %s, note = %s{stamp} WHERE id = %s", (status, note, audit_id)
        )

    def audit_undone(self, audit_id: int, undo_run: str) -> None:
        self.conn.execute(
            "UPDATE labeler_audit SET status = 'undone', undone_at = now(), undo_run_id = %s WHERE id = %s",
            (undo_run, audit_id),
        )

    def applied(self, run_id: str | None, person_ids: list[str] | None) -> list[dict]:
        q = "SELECT id, run_id, person_id::text, old_name, new_name FROM labeler_audit WHERE status = 'applied'"
        args: list = []
        if run_id:
            q += " AND run_id = %s"
            args.append(run_id)
        if person_ids:
            q += " AND person_id = ANY(%s::uuid[])"
            args.append(person_ids)
        cur = self.conn.execute(q + " ORDER BY id DESC", args)
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]

    def ever_applied(self, person_id: str) -> bool:
        return bool(
            self.conn.execute(
                "SELECT 1 FROM labeler_audit WHERE person_id = %s AND status IN ('applied', 'undone') LIMIT 1",
                (person_id,),
            ).fetchone()
        )

    # --- caches
    def http_get(self, url: str) -> dict | None:
        cur = self.conn.execute(
            "SELECT fetched_at, status, etag, last_modified, content_type, body FROM labeler_http_cache WHERE url = %s",
            (url,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return dict(zip([d.name for d in cur.description], row, strict=True))

    def http_put(self, url: str, status: int, etag, last_modified, content_type, body: bytes | None) -> None:
        self.conn.execute(
            """INSERT INTO labeler_http_cache (url, fetched_at, status, etag, last_modified, content_type, body)
               VALUES (%s, now(), %s, %s, %s, %s, %s)
               ON CONFLICT (url) DO UPDATE SET fetched_at = now(), status = excluded.status, etag = excluded.etag,
               last_modified = excluded.last_modified, content_type = excluded.content_type, body = excluded.body""",
            (url, status, etag, last_modified, content_type, body),
        )

    def http_touch(self, url: str) -> None:
        self.conn.execute("UPDATE labeler_http_cache SET fetched_at = now() WHERE url = %s", (url,))

    def embedding_get(self, sha: str, settings: str) -> list | None:
        row = self.conn.execute(
            "SELECT faces FROM labeler_embedding_cache WHERE image_sha256 = %s AND settings = %s", (sha, settings)
        ).fetchone()
        return row[0] if row else None

    def embedding_put(self, sha: str, settings: str, faces: list) -> None:
        self.conn.execute(
            """INSERT INTO labeler_embedding_cache (image_sha256, settings, faces) VALUES (%s, %s, %s)
               ON CONFLICT DO NOTHING""",
            (sha, settings, Jsonb(faces)),
        )


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)
