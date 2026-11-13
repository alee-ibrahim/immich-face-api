"""Request log and face crops: unit tests (no database), and store / admin tests against a real PostgreSQL
(FACE_API_TEST_STORE_URL). Faces come from the fake ML server; images are synthetic, no real person."""

import asyncio
import dataclasses
import io
import os
from datetime import UTC, datetime, timedelta

import httpx
import numpy as np
import psycopg
import pytest
from PIL import Image

from immich_face_api import metrics as m
from immich_face_api.admin import create_admin_app
from immich_face_api.app import create_app
from immich_face_api.config import ConfigError, Settings
from immich_face_api.ml import DetectedFace
from immich_face_api.requestlog import (
    FaceRecord,
    RequestLogWriter,
    RequestRecord,
    embedding_bytes,
    embedding_text,
    kind_of,
    make_crop,
    wants_crop,
)
from immich_face_api.store import Store

from .conftest import KEY, synthetic_jpeg
from .test_app import NEIGHBOURS, FakeDB, FakeML
from .test_store_admin import ADMIN_TOKEN, sign_in

URL = os.environ.get("FACE_API_TEST_STORE_URL")
needs_db = pytest.mark.skipif(not URL, reason="FACE_API_TEST_STORE_URL not set")
AUTH = {"Authorization": f"Bearer {KEY}"}

EMB = "[" + ",".join(["0.05"] * 512) + "]"
# three faces in an 800x600 image: matched (0.31), unknown (0.62 > 0.5), near the threshold (0.47)
FACES3 = [
    DetectedFace(box={"x1": 600, "y1": 100, "x2": 700, "y2": 220}, score=0.91, embedding=EMB),
    DetectedFace(box={"x1": 100, "y1": 100, "x2": 200, "y2": 220}, score=0.88, embedding=EMB),
    DetectedFace(box={"x1": 350, "y1": 300, "x2": 420, "y2": 390}, score=0.75, embedding=EMB),
]
NEIGHBOURS3 = [*NEIGHBOURS, [type(NEIGHBOURS[0][0])("p4", "Dan Example", 0.47)]]


class FakeImmich(FakeDB):
    def __init__(self, neighbours):
        super().__init__(neighbours)
        self.similar_calls = []

    async def similar_people(self, embedding, *, probes=0, scan=200, limit=12):
        self.similar_calls.append(embedding)
        return [
            {"person_id": "u-1", "name": None, "distance": 0.41, "faces": 7},
            {"person_id": "p3", "name": "Carol Example", "distance": 0.55, "faces": 2},
        ]


# ---------------------------------------------------------------------------------------------- unit


def test_kinds_and_modes():
    assert kind_of({"name": None}, 0.5, 0.05) == "unknown"
    assert kind_of({"name": "A", "ambiguous": True, "distance": 0.2}, 0.5, 0.05) == "ambiguous"
    assert kind_of({"name": "A", "ambiguous": False, "distance": 0.46}, 0.5, 0.05) == "near"
    assert kind_of({"name": "A", "ambiguous": False, "distance": 0.30}, 0.5, 0.05) == "matched"
    assert not wants_crop("none", "unknown")
    assert wants_crop("unknown", "unknown") and wants_crop("unknown", "near") and wants_crop("unknown", "ambiguous")
    assert not wants_crop("unknown", "matched")
    assert wants_crop("all", "matched")


def test_crop_is_small_clean_jpeg():
    img = Image.new("RGB", (1440, 960), (120, 80, 40))
    data = make_crop(img, {"x1": 100, "y1": 100, "x2": 500, "y2": 600}, 160)
    out = Image.open(io.BytesIO(data))
    assert out.format == "JPEG" and max(out.size) <= 160
    assert not out.getexif() and "exif" not in out.info and "icc_profile" not in out.info
    # boxes partly or wholly outside the image never fail
    for box in ({"x1": -50, "y1": -50, "x2": 30, "y2": 30}, {"x1": 2000, "y1": 2000, "x2": 2100, "y2": 2100}):
        assert Image.open(io.BytesIO(make_crop(img, box, 160))).size[0] >= 1


def test_embedding_roundtrip():
    raw = "[" + ",".join(str(x) for x in np.linspace(-0.2, 0.2, 512)) + "]"
    b = embedding_bytes(raw)
    assert len(b) == 1024
    back = np.array(embedding_text(b).strip("[]").split(","), dtype=np.float32)
    assert np.allclose(back, np.linspace(-0.2, 0.2, 512), atol=1e-3)


def test_config(monkeypatch):
    monkeypatch.setenv("FACE_API_ML_URL", "http://ml")
    monkeypatch.setenv("FACE_API_OWNER_ID", "00000000-0000-4000-8000-000000000001")
    monkeypatch.setenv("FACE_API_STORE_URL", "postgresql://x@y/z")
    s = Settings.from_env()
    assert s.request_log and s.store_crops == "none" and s.request_log_retention_days == 90
    assert s.crop_retention_days == 30 and s.log_max_faces == 50
    monkeypatch.setenv("FACE_API_STORE_CROPS", "ALL")
    monkeypatch.setenv("FACE_API_REQUEST_LOG", "false")
    s = Settings.from_env()
    assert s.store_crops == "all" and not s.request_log
    monkeypatch.setenv("FACE_API_STORE_CROPS", "faces")
    with pytest.raises(ConfigError):
        Settings.from_env()


def api(settings, mode="none", faces=FACES3, neighbours=NEIGHBOURS3, **kw):
    s = dataclasses.replace(settings, store_crops=mode, **kw)
    app = create_app(s)
    st = app.state.face
    st.ml, st.db, st.ready = FakeML(faces), FakeImmich(neighbours), True
    return app, httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api")


class MemoryStore:
    """Collects what the writer would insert."""

    def __init__(self, fail=False):
        self.records, self.fail = [], fail

    async def ensure_migrated(self):
        pass

    async def insert_requests(self, records):
        if self.fail:
            raise OSError("store down")
        self.records.extend(records)


@pytest.mark.parametrize(
    ("mode", "cropped"),
    [("none", []), ("unknown", ["unknown", "near"]), ("all", ["unknown", "near", "matched"])],
)
@pytest.mark.asyncio
async def test_identify_logs_faces_per_crop_mode(settings, mode, cropped):
    app, c = api(settings, mode)
    st = app.state.face
    st.writer = RequestLogWriter(MemoryStore())
    async with c:
        r = await c.post("/v1/identify", headers=AUTH, content=synthetic_jpeg(800, 600))
        assert r.status_code == 200 and "_log" not in r.json()
    rec = st.writer.queue.get_nowait()
    assert (rec.client, rec.status, rec.faces, rec.matched, rec.unknown, rec.near) == ("website", 200, 3, 2, 1, 1)
    assert rec.image_w == 800 and rec.image_bytes > 0 and rec.image_format == "JPEG"
    kinds = {f.idx: f.kind for f in rec.face_records}
    assert kinds == {0: "unknown", 1: "near", 2: "matched"}  # left to right, like the response
    assert sorted(f.kind for f in rec.face_records if f.crop) == sorted(cropped)
    assert all((f.crop is None) == (f.embedding is None) for f in rec.face_records)
    unknown = rec.face_records[0]
    assert unknown.person_id is None and unknown.candidates[0]["name"] == "Carol Example"  # near-miss beyond 0.5
    assert unknown.candidates[0]["distance"] == 0.62


@pytest.mark.asyncio
async def test_log_is_bounded_and_errors_are_logged(settings):
    many = [DetectedFace(box={"x1": i, "y1": 0, "x2": i + 5, "y2": 5}, score=0.9, embedding=EMB) for i in range(30)]
    app, c = api(settings, "all", faces=many, neighbours=[[] for _ in many], log_max_faces=10)
    st = app.state.face
    st.writer = RequestLogWriter(MemoryStore())
    async with c:
        assert (await c.post("/v1/identify", headers=AUTH, content=synthetic_jpeg(200, 200))).status_code == 200
        assert (await c.post("/v1/identify", headers=AUTH, content=b"not an image")).status_code == 415
        assert (await c.post("/v1/identify", content=b"x")).status_code == 401  # unknown client: not logged
    ok, bad = st.writer.queue.get_nowait(), st.writer.queue.get_nowait()
    assert st.writer.queue.empty()
    assert ok.faces == 30 and len(ok.face_records) == 10 and sum(1 for f in ok.face_records if f.crop) == 10
    assert (bad.status, bad.error_code, bad.faces, bad.face_records) == (415, "invalid_image", 0, [])


@pytest.mark.asyncio
async def test_store_down_does_not_slow_identify(settings):
    app, c = api(settings, "all")
    st = app.state.face
    st.writer = RequestLogWriter(MemoryStore(fail=True), maxsize=2)
    before = m.REQUEST_LOG_DROPPED.labels("store_error")._value.get()
    full_before = m.REQUEST_LOG_DROPPED.labels("queue_full")._value.get()
    async with c:
        codes = [(await c.post("/v1/identify", headers=AUTH, content=synthetic_jpeg())).status_code for _ in range(3)]
    assert codes == [200, 200, 200]  # third entry dropped: queue full, identify unaffected
    assert m.REQUEST_LOG_DROPPED.labels("queue_full")._value.get() == full_before + 1
    task = asyncio.create_task(st.writer.loop())
    await asyncio.sleep(0.05)
    task.cancel()
    assert m.REQUEST_LOG_DROPPED.labels("store_error")._value.get() == before + 2


@pytest.mark.asyncio
async def test_try_it_and_no_writer_log_nothing(settings):
    app, c = api(settings, "all")
    assert app.state.face.writer is None  # no store configured: nothing to log to
    async with c:
        assert (await c.post("/v1/identify", headers=AUTH, content=synthetic_jpeg())).status_code == 200


@pytest.mark.asyncio
async def test_api_port_serves_no_admin_routes(settings):
    app, c = api(settings, "all")
    async with c:
        for path in ("/requests", "/requests/1", "/missed", "/crops/1.jpg", "/faces/1/similar", "/keys", "/", "/login"):
            for headers in ({}, AUTH):
                r = await c.get(path, headers=headers)
                assert r.status_code in (404, 405), path
                assert "image/jpeg" not in r.headers.get("content-type", "")
        assert (await c.post("/privacy/purge", headers=AUTH, data={"scope": "all"})).status_code in (404, 405)
    admin_paths = {"/requests", "/missed", "/crops/{name}", "/privacy/purge"}
    assert not admin_paths & {getattr(r, "path", "") for r in app.routes}


# ---------------------------------------------------------------------------------------------- store


@pytest.fixture
def clean_db():
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute("DROP SCHEMA public CASCADE")
        conn.execute("CREATE SCHEMA public")
    yield URL


async def opened(url):
    st = Store(url, None, key_cache_seconds=0)
    await st.open(migrate=True)
    return st


def record(client="web", status=200, at=None, faces=None, **kw):
    faces = faces if faces is not None else []
    return RequestRecord(
        at=(at or datetime.now(UTC)).timestamp(),
        rid="rid-1",
        client=client,
        endpoint="/v1/identify",
        status=status,
        error_code=None if status < 400 else "busy",
        latency_ms=420,
        image_bytes=12345,
        image_w=800,
        image_h=600,
        image_format="JPEG",
        faces=len(faces),
        matched=sum(1 for f in faces if f.name),
        unknown=sum(1 for f in faces if not f.name),
        face_records=faces,
        **kw,
    )


def face(idx, kind, name=None, crop=b"\xff\xd8jpeg", emb=b"\0" * 1024):
    return FaceRecord(
        idx=idx,
        kind=kind,
        box={"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1},
        detection_score=0.9,
        person_id="p1" if name else None,
        name=name,
        distance=0.3 if name else None,
        candidates=[{"person_id": "p1", "name": "Alice Example", "distance": 0.61, "votes": 1}],
        crop=crop,
        embedding=emb if crop else None,
    )


@needs_db
@pytest.mark.asyncio
async def test_store_log_filters_retention_and_purge(clean_db):
    st = await opened(clean_db)
    try:
        await st.migrate()  # idempotent
        old = datetime.now(UTC) - timedelta(days=40)
        await st.insert_requests(
            [
                record("web", at=old, faces=[face(0, "unknown")]),
                record("web", faces=[face(0, "matched", "Alice Example"), face(1, "unknown")]),
                record("kiosk", status=503),
            ]
        )
        rows = await st.list_requests()  # newest first
        assert [r["client"] for r in rows] == ["kiosk", "web", "web"]
        assert rows[1]["names"] == ["Alice Example"] and rows[1]["crops"] == 2
        assert [r["client"] for r in await st.list_requests(client="kiosk")] == ["kiosk"]
        assert len(await st.list_requests(unknowns=True)) == 2
        assert [r["status"] for r in await st.list_requests(status="error")] == [503]
        assert [r["status"] for r in await st.list_requests(status="503")] == [503]
        assert len(await st.list_requests(since=datetime.now(UTC) - timedelta(days=1))) == 2
        assert len(await st.list_requests(before=rows[0]["id"])) == 2
        assert len(await st.list_requests(until=datetime.now(UTC) - timedelta(days=1))) == 1

        req, faces = await st.get_request(rows[1]["id"])
        assert req["image_w"] == 800 and [f["kind"] for f in faces] == ["matched", "unknown"]
        assert faces[1]["candidates"][0]["distance"] == 0.61 and faces[1]["has_crop"]
        missed = await st.missed_faces()
        assert [f["kind"] for f in missed] == ["unknown", "unknown"]  # matched faces are not "missed"
        assert len(await st.missed_faces(since=datetime.now(UTC) - timedelta(days=1))) == 1
        assert await st.crop(missed[0]["id"]) == b"\xff\xd8jpeg"
        assert (await st.log_stats())["crops"] == 3

        # retention: crops older than 30 days lose crop and embedding, requests older than 90 days go entirely
        out = await st.purge(400, request_days=90, crop_days=30, force=True)
        assert out["crops"] == 1 and out["requests"] == 0
        assert await st.crop(missed[1]["id"]) is None
        assert (await st.face(missed[1]["id"]))["embedding"] is None
        async with st.pool.connection() as conn:
            await conn.execute(
                "UPDATE request_log SET at = at - interval '60 days' WHERE at < now() - interval '1 day'"
            )
        out = await st.purge(400, request_days=90, crop_days=30, force=True)
        assert out["requests"] == 1 and len(await st.list_requests()) == 2
        assert len(await st.missed_faces()) == 1  # faces went with their request

        # purge-all: crops only, then everything; both audited
        assert (await st.purge_all("crops", "tester"))["crops"] == 2
        assert (await st.log_stats())["crops"] == 0 and len(await st.list_requests()) == 2
        assert (await st.purge_all("all", "tester"))["requests"] == 2
        assert await st.list_requests() == [] and await st.missed_faces() == []
        audit = await st.audit(10, "log.")
        assert [a["detail"]["scope"] for a in audit] == ["all", "crops"] and audit[0]["actor"] == "tester"
        assert await st.audit(10, "key.") == []
        with pytest.raises(ValueError):
            await st.purge_all("everything", "tester")
    finally:
        await st.close()


@needs_db
@pytest.mark.asyncio
async def test_api_writes_log_to_store(settings, clean_db):
    s = dataclasses.replace(settings, store_url=clean_db, store_crops="unknown")
    app = create_app(s)
    st = app.state.face
    st.ml, st.db, st.ready = FakeML(FACES3), FakeImmich(NEIGHBOURS3), True
    st.store = await opened(clean_db)
    st.writer = RequestLogWriter(st.store)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            assert (await c.post("/v1/identify", headers=AUTH, content=synthetic_jpeg(800, 600))).status_code == 200
        await st.writer.drain()
        (row,) = await st.store.list_requests()
        assert (row["client"], row["faces"], row["unknown"], row["near"], row["crops"]) == ("website", 3, 1, 1, 2)
        _, faces = await st.store.get_request(row["id"])
        data = await st.store.crop(faces[0]["id"])
        assert Image.open(io.BytesIO(data)).format == "JPEG"
        assert (await st.store.face(faces[2]["id"]))["embedding"] is None  # matched: no crop in "unknown" mode
    finally:
        await st.store.close()


# ---------------------------------------------------------------------------------------------- admin


@needs_db
@pytest.mark.asyncio
async def test_admin_requests_missed_crops_similar_purge(settings, clean_db):
    s = dataclasses.replace(
        settings,
        store_url=clean_db,
        admin_token=ADMIN_TOKEN,
        session_secret="s" * 40,
        session_https_only=False,
        store_crops="all",
        immich_url="https://photos.example.org",
    )
    app = create_admin_app(s)
    state = app.state.face
    state.ml, state.db, state.ready = FakeML(FACES3), FakeImmich(NEIGHBOURS3), True
    state.store = await opened(clean_db)
    crop = make_crop(Image.new("RGB", (300, 300), (90, 90, 90)), {"x1": 50, "y1": 50, "x2": 150, "y2": 150}, 160)
    await state.store.insert_requests(
        [record("website", faces=[face(0, "unknown", crop=crop, emb=embedding_bytes(EMB)), face(1, "matched", "Ann")])]
    )
    (req,) = await state.store.list_requests()
    _, faces = await state.store.get_request(req["id"])
    fid = faces[0]["id"]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://admin.test") as c:
        # everything behind sign-in, crops included
        for path in ("/requests", f"/requests/{req['id']}", "/missed", f"/crops/{fid}.jpg", f"/faces/{fid}/similar"):
            r = await c.get(path)
            assert r.status_code == 303 and r.headers["location"] == "/login", path
        r = await c.post("/privacy/purge", data={"csrf": "x", "scope": "all", "confirm": "DELETE"})
        assert r.status_code == 303 and r.headers["location"] == "/login"
        assert len(await state.store.list_requests()) == 1

        csrf = await sign_in(c)
        page = await c.get("/requests?client=website&unknowns=1")
        assert page.status_code == 200 and "Ann" in page.text and "website" in page.text
        assert "Face crops: <b>all</b>" in page.text
        assert (await c.get("/requests?status=error")).text.count("/requests/") == 0
        detail = await c.get(f"/requests/{req['id']}")
        assert f'src="/crops/{fid}.jpg"' in detail.text and "0.610" in detail.text
        assert "https://photos.example.org/people/p1" in detail.text
        assert (await c.get("/requests/999999")).status_code == 404
        missed = await c.get("/missed")
        assert f"/faces/{fid}/similar" in missed.text and "Ann" not in missed.text  # matched faces not listed
        assert "near threshold" in (await c.get("/missed?kind=near")).text  # filter renders

        img = await c.get(f"/crops/{fid}.jpg")
        assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg" and img.content == crop
        assert "no-store" in img.headers["cache-control"]
        assert (await c.get("/crops/999999.jpg")).status_code == 404
        assert (await c.get("/crops/../app.py")).status_code == 404

        sim = await c.get(f"/faces/{fid}/similar")
        assert sim.status_code == 200 and "https://photos.example.org/people/u-1" in sim.text
        assert "Carol Example" in sim.text and state.db.similar_calls
        assert "Find similar" in sim.text

        # dashboard additions
        state.store.buffer.record("website", "/v1/identify", 200, 300, faces=4, matched=1)
        await state.store.flush()
        dash = await c.get("/?range=24h")
        assert "Missed-face rate" in dash.text and "Top apps by unknown faces" in dash.text and "75.0%" in dash.text

        # purge: needs csrf and the typed confirmation; audited
        r = await c.post("/privacy/purge", data={"csrf": csrf, "scope": "all", "confirm": "yes"})
        assert r.headers["location"].endswith("purged=unconfirmed") and len(await state.store.list_requests()) == 1
        r = await c.post("/privacy/purge", data={"csrf": "bad", "scope": "all", "confirm": "DELETE"})
        assert r.headers["location"].endswith("purged=expired") and len(await state.store.list_requests()) == 1
        r = await c.post("/privacy/purge", data={"csrf": csrf, "scope": "crops", "confirm": "DELETE"})
        assert r.headers["location"].endswith("purged=crops") and await state.store.crop(fid) is None
        r = await c.post("/privacy/purge", data={"csrf": csrf, "scope": "all", "confirm": "DELETE"})
        assert r.headers["location"].endswith("purged=all") and await state.store.list_requests() == []
        page = await c.get("/requests?purged=all")
        assert "log.purge_all (all)" in page.text
        assert "admin-token" in page.text  # actor recorded
        keys = await c.get("/keys")
        assert "log.purge_all" not in keys.text  # key audit list shows key changes only
    await state.store.close()
