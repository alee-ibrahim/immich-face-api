"""People change log (/v1/people/changes, /v1/people/resolve), /v1/verify and the admin Changes page.

Unit tests need nothing; the tracker tests run against a real PostgreSQL (FACE_API_TEST_STORE_URL): the store in that
database, and a tiny stand-in for Immich's ``person`` / ``asset_face`` tables in a second database next to it. No
real faces or people."""

import base64
import dataclasses
import os
import uuid
from datetime import UTC, datetime, timedelta

import httpx
import psycopg
import pytest

from immich_face_api import changes as ch
from immich_face_api.admin import create_admin_app
from immich_face_api.app import create_app, iou, parse_box, person_ids
from immich_face_api.config import Settings
from immich_face_api.db import Database
from immich_face_api.matching import Neighbour
from immich_face_api.ml import DetectedFace
from immich_face_api.security import RateLimiter, hash_key
from immich_face_api.store import Store

from .conftest import KEY, synthetic_jpeg
from .test_app import FakeDB, FakeML
from .test_store_admin import ADMIN_TOKEN, sign_in

URL = os.environ.get("FACE_API_TEST_STORE_URL")
needs_db = pytest.mark.skipif(not URL, reason="FACE_API_TEST_STORE_URL not set")
AUTH = {"Authorization": f"Bearer {KEY}"}

OWNER = "00000000-0000-4000-8000-000000000001"
OTHER_OWNER = "00000000-0000-4000-8000-000000000002"
A, B, C, D = (f"aaaaaaaa-0000-4000-8000-00000000000{i}" for i in (1, 2, 3, 4))


# ---------------------------------------------------------------------------------------------- unit


def test_diff_events_rename_split_merge_delete():
    prev = {A: "Alice", B: "Bob", D: "Dan", "e": "Eve", "f": "Fay"}
    persons = {
        A: ("Alice Example", False),  # renamed
        B: ("Bob", False),
        C: ("Carol", False),  # created (named now)
        "e": ("Eve", True),  # hidden
        "f": ("", False),  # name cleared
        # D gone: merged into B
    }
    moves = [(A, B, 2), (D, B, 5), (D, None, 1), (A, "unnamed-x", 3)]
    ev = ch.diff_events(prev, persons, moves)
    got = [(e.type, e.person_id, e.to_person_id, e.faces, e.reason) for e in ev]
    assert got == [
        ("created", C, None, None, None),
        ("renamed", A, None, None, None),
        ("faces_moved", A, B, 2, None),
        ("faces_moved", A, "unnamed-x", 3, None),
        ("faces_moved", D, None, 1, None),
        ("merged", D, B, 5, None),
        ("deleted", "e", None, None, "hidden"),
        ("deleted", "f", None, None, "unnamed"),
    ]
    renamed = ev[1]
    assert (renamed.old_name, renamed.name) == ("Alice", "Alice Example")
    assert ev[3].to_name is None and ev[2].to_name == "Bob"


def test_diff_events_delete_without_target():
    ev = ch.diff_events({A: "Alice"}, {}, [(A, None, 4)])
    assert [(e.type, e.reason) for e in ev] == [("faces_moved", None), ("deleted", "deleted")]
    assert ch.diff_events({A: "Alice"}, {A: ("Alice", False)}, []) == []


def test_cursor_and_since_parsing():
    assert ch.parse_since(None) == (0, None)
    assert ch.parse_since("pc_42") == (42, None)
    after_id, ts = ch.parse_since("2026-01-31T10:00:00Z")
    assert after_id is None and ts == datetime(2026, 1, 31, 10, tzinfo=UTC)
    assert ch.parse_since("2026-01-31T10:00:00 00:00")[1] == ts  # "+" decoded as a space
    for bad in ("2026-01-31T10:00:00", "yesterday", "pc_x"):
        with pytest.raises(ValueError):
            ch.parse_since(bad)


def test_uuid7_floor_orders_by_time():
    t = datetime(2026, 1, 31, 10, tzinfo=UTC)
    lo, hi = ch.uuid7_floor(t), ch.uuid7_floor(t + timedelta(milliseconds=1))
    assert uuid.UUID(lo).bytes < uuid.UUID(hi).bytes
    ms = int(t.timestamp() * 1000)
    v7 = uuid.UUID(f"{ms:012x}"[:8] + "-" + f"{ms:012x}"[8:] + "-7abc-8def-0123456789ab")
    assert uuid.UUID(lo).bytes <= v7.bytes < uuid.UUID(hi).bytes


def test_event_json_shapes():
    base = {"id": 7, "at": datetime(2026, 1, 1, tzinfo=UTC), "window_start": None, "old_name": None, "reason": None}
    row = {**base, "type": "merged", "person_id": A, "to_person_id": B, "name": "Dan", "to_name": "Bob", "faces": 5}
    assert ch.event_json(row) == {
        "cursor": "pc_7",
        "type": "merged",
        "at": "2026-01-01T00:00:00Z",
        "after": None,
        "from_person_id": A,
        "from_name": "Dan",
        "into_person_id": B,
        "into_name": "Bob",
        "faces": 5,
    }
    row = {
        **base,
        "type": "faces_moved",
        "person_id": A,
        "to_person_id": None,
        "name": "A",
        "to_name": None,
        "faces": 2,
    }
    assert ch.event_json(row)["to_person_id"] is None and ch.event_json(row)["count"] == 2


def test_verify_helpers():
    assert parse_box("0.1,0.2,0.3,0.4") == {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}
    assert parse_box('{"x":0.1,"y":0.2,"w":0.3,"h":0.4}') == parse_box({"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4})
    assert parse_box("") is None
    box = {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}
    assert iou(box, box) == pytest.approx(1.0)
    assert iou(box, {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}) == 0.0
    assert person_ids([f"{A},{B}", A.upper()], 5) == [A, B]
    from immich_face_api.app import HttpError

    for bad in (["nope"], [], [A, B, C]):
        with pytest.raises(HttpError):
            person_ids(bad, 2)
    for bad in ("1,2,3", "0.1,0.1,0,0.1", "x"):
        with pytest.raises(HttpError):
            parse_box(bad)


def test_rate_limiter_cost():
    now = [0.0]
    rl = RateLimiter(60, 5, clock=lambda: now[0])
    assert rl.take("c", cost=3) == (True, 0)
    ok, retry = rl.take("c", cost=3)  # 2 left
    assert not ok and retry == 1
    assert rl.take("c", cost=50)[0] is False  # capped at the burst (5): needs a full bucket
    now[0] += 5
    assert rl.take("c", cost=50) == (True, 0)


# ---------------------------------------------------------------------------------------------- verify (fakes)

EMB_R, EMB_L = "[0.1]", "[0.2]"  # the fakes below answer by embedding
VFACES = [
    DetectedFace(box={"x1": 600, "y1": 100, "x2": 700, "y2": 220}, score=0.91, embedding=EMB_R),  # right
    DetectedFace(box={"x1": 100, "y1": 100, "x2": 200, "y2": 220}, score=0.88, embedding=EMB_L),  # left
]


class VerifyDB(FakeDB):
    """Two faces (left, right after sorting); P1 is the left face, P2 is close to the right one but P3 is closer."""

    P1, P2, P3, HIDDEN = (f"bbbbbbbb-0000-4000-8000-00000000000{i}" for i in (1, 2, 3, 4))

    def __init__(self):
        super().__init__([])
        self.people_rows = {self.P1: ("Alice", False), self.P2: ("Bob", False), self.P3: ("Carl", False)}
        self.people_rows[self.HIDDEN] = ("Hidden", True)

    async def nearest(self, embeddings, include_unnamed=False):
        rows = {
            EMB_R: [Neighbour(self.P3, "Carl", 0.30), Neighbour(self.P2, "Bob", 0.40)],
            EMB_L: [Neighbour(self.P1, "Alice", 0.2)],
        }
        return [rows[e] for e in embeddings]

    async def persons_by_id(self, ids):
        return {p: self.people_rows[p] for p in ids if p in self.people_rows}

    async def claim_distances(self, embeddings, ids, per_person):
        table = {EMB_R: {self.P1: 0.9, self.P2: 0.40, self.P3: 0.30}, EMB_L: {self.P1: 0.2, self.P2: 0.8, self.P3: 0.7}}
        return [{p: d for p, d in table[e].items() if p in ids} for e in embeddings]


def verify_app(settings, **kw):
    app = create_app(dataclasses.replace(settings, face_cache=False, **kw))
    st = app.state.face
    st.ml, st.db, st.ready = FakeML(VFACES), VerifyDB(), True
    return app, httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


@pytest.mark.asyncio
async def test_verify_claims(settings):
    app, c = verify_app(settings)
    db = app.state.face.db
    unknown = "cccccccc-0000-4000-8000-000000000009"
    async with c:
        r = await c.post(
            "/v1/verify",
            headers=AUTH,
            files={"image": ("x.jpg", synthetic_jpeg(800, 600), "image/jpeg")},
            data={"person_id": [db.P1, f"{db.P2},{db.HIDDEN}", unknown]},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        by = {cl["person_id"]: cl for cl in body["claims"]}
        assert [cl["person_id"] for cl in body["claims"]] == [db.P1, db.P2, db.HIDDEN, unknown]
        assert by[db.P1]["verdict"] == "match" and by[db.P1]["face"] == 0 and by[db.P1]["distance"] == 0.2
        assert body["faces"][0]["person_id"] == db.P1  # left face first
        assert by[db.P2]["verdict"] == "match_not_best" and by[db.P2]["best_person_id"] == db.P3
        assert by[db.P2]["within_threshold"] is True and by[db.P2]["face"] == 1
        assert by[db.HIDDEN]["verdict"] == "person_gone" and by[db.HIDDEN]["status"] == "hidden"
        assert by[unknown]["verdict"] == "person_gone" and by[unknown]["status"] == "unknown"

        # a box selects the face: P1 claimed at the right face -> beyond the threshold
        r = await c.post(
            f"/v1/verify?person_id={db.P1}&box=0.75,0.1667,0.125,0.2", headers=AUTH, content=synthetic_jpeg(800, 600)
        )
        (cl,) = r.json()["claims"]
        assert (cl["verdict"], cl["face"], cl["within_threshold"]) == ("no_match", 1, False)
        # a box where there is no face
        r = await c.post(f"/v1/verify?person_id={db.P1}&box=0.4,0.8,0.05,0.05", headers=AUTH, content=synthetic_jpeg())
        assert r.json()["claims"][0]["verdict"] == "no_face"

        assert (await c.post("/v1/verify", headers=AUTH, content=synthetic_jpeg())).status_code == 400  # no claim
        r = await c.post(f"/v1/verify?person_id={db.P1}", content=synthetic_jpeg())
        assert r.status_code == 401


@pytest.mark.asyncio
async def test_verify_batch(settings):
    app, c = verify_app(settings, verify_batch_max=3)
    db = app.state.face.db
    img = base64.b64encode(synthetic_jpeg(800, 600)).decode()
    async with c:
        items = [
            {"ref": "photo-1", "image": img, "person_ids": [db.P1]},
            {"ref": "photo-2", "image": "%%%", "person_ids": [db.P1]},
            {
                "ref": "photo-3",
                "image": img,
                "person_ids": [db.P2],
                "box": {"x": 0.75, "y": 0.1667, "w": 0.125, "h": 0.2},
            },
        ]
        r = await c.post("/v1/verify/batch", headers=AUTH, json={"items": items})
        assert r.status_code == 200, r.text
        out = r.json()["items"]
        assert [i["ref"] for i in out] == ["photo-1", "photo-2", "photo-3"]
        assert out[0]["claims"][0]["verdict"] == "match"
        assert out[1]["error"]["code"] == "bad_request"
        assert out[2]["claims"][0]["verdict"] == "match_not_best"
        r = await c.post("/v1/verify/batch", headers=AUTH, json={"items": items * 2})
        assert r.status_code == 400
        assert (await c.post("/v1/verify/batch", headers=AUTH, content=b"nope")).status_code == 400


@pytest.mark.asyncio
async def test_changes_endpoints_need_the_store(settings):
    app, c = verify_app(settings)
    async with c:
        assert (await c.get("/v1/people/changes", headers=AUTH)).status_code == 404
        r = await c.post("/v1/people/resolve", headers=AUTH, json={"person_ids": [VerifyDB.P1, VerifyDB.HIDDEN]})
        assert r.status_code == 200
        assert [p["status"] for p in r.json()["people"]] == ["current", "hidden"]
        assert r.json()["change_log"] is False
        assert "changes_cursor" not in (await c.get("/v1/people", headers=AUTH)).json()


# ---------------------------------------------------------------------------------------------- tracker (real PG)

FAKE_IMMICH = """
CREATE FUNCTION fake_uuid7() RETURNS uuid LANGUAGE sql AS $$
  SELECT (substr(h, 1, 8) || '-' || substr(h, 9, 4) || '-7' || substr(r, 1, 3) || '-8' || substr(r, 4, 3) || '-'
          || substr(r, 7, 12))::uuid
  FROM (SELECT lpad(to_hex((extract(epoch FROM clock_timestamp()) * 1000)::bigint), 12, '0') AS h,
               md5(random()::text) AS r) x
$$;
CREATE TABLE person ("personGroupId" uuid NOT NULL, "ownerId" uuid NOT NULL, name varchar NOT NULL DEFAULT '',
                     "isHidden" boolean NOT NULL DEFAULT false, PRIMARY KEY ("ownerId", "personGroupId"));
CREATE TABLE asset_face (id uuid PRIMARY KEY, "personGroupId" uuid, "ownerId" uuid NOT NULL, "deletedAt" timestamptz,
                         "isVisible" boolean NOT NULL DEFAULT true, "assetDeletedAt" timestamptz,
                         "assetVisibility" text NOT NULL DEFAULT 'timeline', "updateId" uuid NOT NULL DEFAULT fake_uuid7());
CREATE INDEX ON asset_face ("updateId");
-- like Immich: a new updateId on every update, except when only the asset state (trash, visibility) changes
CREATE FUNCTION touch() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN NEW."updateId" := fake_uuid7(); RETURN NEW; END $$;
CREATE TRIGGER touch BEFORE UPDATE ON asset_face FOR EACH ROW
  WHEN (OLD."assetDeletedAt" IS NOT DISTINCT FROM NEW."assetDeletedAt"
        AND OLD."assetVisibility" IS NOT DISTINCT FROM NEW."assetVisibility") EXECUTE FUNCTION touch();
"""


def face_id(n: int) -> str:
    return f"ffffffff-0000-4000-8000-{n:012d}"


@pytest.fixture
def immich():
    """(admin conninfo of the fake Immich database, Settings pointing at it)."""
    info = psycopg.conninfo.conninfo_to_dict(URL)
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute("DROP DATABASE IF EXISTS faceapi_fake_immich")
        conn.execute("CREATE DATABASE faceapi_fake_immich")
    fake = psycopg.conninfo.make_conninfo(URL, dbname="faceapi_fake_immich")
    with psycopg.connect(fake, autocommit=True) as conn:
        conn.execute(FAKE_IMMICH)
        conn.execute(
            "INSERT INTO person VALUES (%s,%s,'Alice',false), (%s,%s,'Bob',false), (%s,%s,'',false), (%s,%s,'Dan',false),"
            " (%s,%s,'Alice copy',false)",
            (A, OWNER, B, OWNER, C, OWNER, D, OWNER, A, OTHER_OWNER),
        )
        faces = [(A, 1, 5), (B, 6, 3), (C, 9, 2), (D, 11, 2)]
        for pid, start, n in faces:
            for k in range(start, start + n):
                conn.execute(
                    'INSERT INTO asset_face (id, "personGroupId", "ownerId") VALUES (%s,%s,%s)',
                    (face_id(k), pid, OWNER),
                )
    settings = Settings(
        ml_url="http://ml.test",
        db_host=info.get("host", "localhost"),
        db_port=int(info.get("port", 5432)),
        db_name="faceapi_fake_immich",
        db_user=info.get("user", "postgres"),
        db_password=info.get("password", ""),
        db_sslmode="disable",
        owner_id=OWNER,
        api_keys={"website": bytes.fromhex(hash_key(KEY))},
        metrics_port=0,
        store_url=URL,
        face_cache=False,
        rate_limit_per_minute=600,
        rate_limit_burst=50,
        admin_token=ADMIN_TOKEN,
        session_secret="s" * 40,
        session_https_only=False,
    )
    yield fake, settings


@pytest.fixture
def clean_store():
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute("DROP SCHEMA public CASCADE")
        conn.execute("CREATE SCHEMA public")
    yield URL


def run(fake, sql, *params):
    with psycopg.connect(fake, autocommit=True) as conn:
        conn.execute(sql, params or None)


async def tracker_for(settings):
    db = Database(settings)
    await db.open()
    await db.pool.wait()
    store = Store(settings.store_url, None)
    await store.open(migrate=True)
    return db, store, ch.ChangeTracker(db, store, interval=300, overlap=60, full_hours=24)


async def snapshot(store):
    async with store.pool.connection() as conn:
        rows = await (await conn.execute("SELECT face_id::text, person_id::text FROM face_snapshot")).fetchall()
    return {r["face_id"]: r["person_id"] for r in rows}


@needs_db
@pytest.mark.asyncio
async def test_tracker_seed_incremental_full(immich, clean_store):
    fake, settings = immich
    db, store, tr = await tracker_for(settings)
    try:
        r = await tr.check()
        assert r["mode"] == "seed" and r["events"] == []
        snap = await snapshot(store)
        assert len(snap) == 10 and snap[face_id(1)] == A and face_id(9) not in snap  # C is unnamed

        assert (await tr.check())["events"] == []  # nothing changed

        # rename A, split 2 faces A -> B, name C, merge D into B (Immich: move faces, then delete the person)
        run(fake, 'UPDATE person SET name = \'Alice Example\' WHERE "personGroupId" = %s AND "ownerId" = %s', A, OWNER)
        run(fake, 'UPDATE asset_face SET "personGroupId" = %s WHERE id IN (%s, %s)', B, face_id(1), face_id(2))
        run(fake, "UPDATE person SET name = 'Carol' WHERE \"personGroupId\" = %s", C)
        run(fake, 'UPDATE asset_face SET "personGroupId" = %s WHERE "personGroupId" = %s', B, D)
        run(fake, 'DELETE FROM person WHERE "personGroupId" = %s', D)
        r = await tr.check(full=False)
        got = [(e.type, e.person_id, e.to_person_id, e.faces) for e in r["events"]]
        assert got == [
            ("created", C, None, None),
            ("renamed", A, None, None),
            ("faces_moved", A, B, 2),
            ("merged", D, B, 2),
        ]
        snap = await snapshot(store)
        assert snap[face_id(1)] == B and snap[face_id(9)] == C and snap[face_id(11)] == B and len(snap) == 12

        # unassign one of A's faces; hide B
        run(fake, 'UPDATE asset_face SET "personGroupId" = NULL WHERE id = %s', face_id(3))
        run(fake, 'UPDATE person SET "isHidden" = true WHERE "personGroupId" = %s', B)
        r = await tr.check(full=False)
        assert [(e.type, e.person_id, e.to_person_id, e.reason) for e in r["events"]] == [
            ("faces_moved", A, None, None),
            ("deleted", B, None, "hidden"),
        ]
        assert set((await snapshot(store)).values()) == {A, C}

        # changes Immich makes without a new updateId (asset trashed) or that an incremental check missed: the full
        # reconcile drops the trashed face silently and still reports the move
        run(fake, 'UPDATE asset_face SET "assetDeletedAt" = now() WHERE id = %s', face_id(4))
        assert (await tr.check(full=False))["events"] == []
        with psycopg.connect(fake, autocommit=True) as conn:
            conn.execute("ALTER TABLE asset_face DISABLE TRIGGER touch")
            conn.execute(  # a change with an old updateId: outside the overlap window
                'UPDATE asset_face SET "personGroupId" = %s, "updateId" = %s WHERE id = %s',
                (C, "00000000-0000-7000-8000-000000000000", face_id(5)),
            )
            conn.execute("ALTER TABLE asset_face ENABLE TRIGGER touch")
        assert (await tr.check(full=False))["events"] == []
        r = await tr.check(full=True)
        assert r["mode"] == "full"
        assert [(e.type, e.person_id, e.to_person_id, e.faces) for e in r["events"]] == [("faces_moved", A, C, 1)]
        snap = await snapshot(store)
        assert face_id(4) not in snap and snap[face_id(5)] == C

        st = await ch.tracker_status(store)
        assert st["people"] == 2 and st["events"] == 7 and st["last_error"] is None
        async with store.pool.connection() as conn:
            faces = {
                str(r["person_id"]): r["faces"]
                for r in await (await conn.execute("SELECT person_id, faces FROM people_snapshot")).fetchall()
            }
        assert faces == {A: 0, C: 3}

        # resolve: D merged into B (now hidden), B hidden, A current, unknown id
        unknown = "cccccccc-0000-4000-8000-000000000009"
        res = await ch.resolve(db, store, [D, B, A, unknown])
        assert res[D]["status"] == "merged" and res[D]["current_person_id"] == B and res[D]["target_status"] == "hidden"
        assert res[B]["status"] == "hidden" and res[A] == {
            "status": "current",
            "current_person_id": A,
            "name": "Alice Example",
        }
        assert res[unknown]["status"] == "unknown"

        # another process holding the lock: skipped
        async with store.pool.connection() as conn:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock(%s)", (ch.LOCK_ID,))
                assert await tr.check() is None
    finally:
        await store.close()
        await db.close()


@needs_db
@pytest.mark.asyncio
async def test_changes_api_and_admin(immich, clean_store):
    fake, settings = immich
    app = create_app(settings)
    st = app.state.face
    st.ml, st.ready = FakeML([]), True
    await st.db.open()
    await st.db.pool.wait()
    st.store = st.tracker.store = Store(settings.store_url, None)
    await st.store.open(migrate=True)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            r = await c.get("/v1/people/changes", headers=AUTH)
            assert r.status_code == 503  # not seeded yet
            await st.tracker.check()
            t0 = datetime.now(UTC)
            people = (await c.get("/v1/people", headers=AUTH)).json()
            start = people["changes_cursor"]
            assert start == "pc_0"

            run(fake, "UPDATE person SET name = 'Bobby' WHERE \"personGroupId\" = %s", B)
            run(fake, 'UPDATE asset_face SET "personGroupId" = %s WHERE "personGroupId" = %s', A, D)
            run(fake, 'DELETE FROM person WHERE "personGroupId" = %s', D)
            await st.tracker.check(full=False)

            r = await c.get("/v1/people/changes", headers=AUTH, params={"since": start})
            body = r.json()
            assert r.status_code == 200 and body["resync_required"] is False and body["has_more"] is False
            assert [e["type"] for e in body["changes"]] == ["renamed", "merged"]
            assert body["changes"][0] == {
                "cursor": body["changes"][0]["cursor"],
                "type": "renamed",
                "at": body["changes"][0]["at"],
                "after": body["changes"][0]["after"],
                "person_id": B,
                "old_name": "Bob",
                "new_name": "Bobby",
            }
            merged = body["changes"][1]
            assert (merged["from_person_id"], merged["into_person_id"], merged["faces"]) == (D, A, 2)
            nxt = body["next_cursor"]
            assert nxt == merged["cursor"]

            r = await c.get("/v1/people/changes", headers=AUTH, params={"since": nxt})
            assert r.json()["changes"] == [] and r.json()["next_cursor"] == nxt
            r = await c.get("/v1/people/changes", headers=AUTH, params={"since": start, "limit": 1})
            assert r.json()["has_more"] is True and len(r.json()["changes"]) == 1
            r = await c.get("/v1/people/changes", headers=AUTH, params={"since": t0.isoformat()})
            assert len(r.json()["changes"]) == 2 and r.json()["resync_required"] is False
            r = await c.get("/v1/people/changes", headers=AUTH, params={"since": "2020-01-01T00:00:00Z"})
            assert r.json()["resync_required"] is True  # before tracking started
            assert (await c.get("/v1/people/changes", headers=AUTH, params={"since": "pc_999"})).status_code == 400
            assert (await c.get("/v1/people/changes", headers=AUTH, params={"since": "soon"})).status_code == 400
            assert (await c.get("/v1/people/changes")).status_code == 401

            # retention purged the start: an old cursor must re-sync
            async with st.store.pool.connection() as conn:
                await conn.execute("UPDATE people_changes SET at = at - interval '400 days' WHERE type = 'renamed'")
            await st.tracker.check(full=False)
            r = await c.get("/v1/people/changes", headers=AUTH, params={"since": start})
            assert r.json()["resync_required"] is True and [e["type"] for e in r.json()["changes"]] == ["merged"]

            r = await c.post("/v1/people/resolve", headers=AUTH, json={"person_ids": [D, B]})
            got = r.json()["people"]
            assert got[0]["status"] == "merged" and got[0]["current_person_id"] == A and got[0]["merged_via"] == [A]
            assert got[1] == {"person_id": B, "status": "current", "current_person_id": B, "name": "Bobby"}
            assert (await c.post("/v1/people/resolve", headers=AUTH, json={"ids": []})).status_code == 400

        admin = create_admin_app(settings)
        ast = admin.state.face
        ast.ml, ast.ready = FakeML([]), True
        ast.db, ast.store = st.db, st.store
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=admin), base_url="http://a") as c:
            await sign_in(c)
            r = await c.get("/changes")
            assert r.status_code == 200 and "merged" in r.text and "Dan" in r.text and "named people tracked" in r.text
            r = await c.get("/changes", params={"type": "renamed"})
            assert "No changes match" in r.text  # purged
            r = await c.get("/changes", params={"person": D.upper()})
            assert "Dan" in r.text
            r = await c.get("/")
            assert "Recent people changes" in r.text and "pc_" in r.text
    finally:
        await st.store.close()
        await st.db.close()


@needs_db
@pytest.mark.asyncio
async def test_tracker_owner_change_reseeds(immich, clean_store):
    fake, settings = immich
    db, store, tr = await tracker_for(settings)
    try:
        await tr.check()
        run(fake, "UPDATE person SET name = 'Bobby' WHERE \"personGroupId\" = %s", B)
        await tr.check(full=False)
        db.status.owner_id = OTHER_OWNER
        r = await tr.check()
        assert r["mode"] == "seed"
        st = await ch.tracker_status(store)
        assert st["events"] == 0 and st["purged_through"] >= 1 and str(st["owner_id"]) == OTHER_OWNER
        assert await ch.latest_cursor(store) == f"pc_{st['purged_through']}"
    finally:
        await store.close()
        await db.close()
