"""Store and admin UI against a real PostgreSQL (FACE_API_TEST_STORE_URL, e.g. a throwaway container; CI starts one).
Immich's database and ML server are faked; no real faces."""

import dataclasses
import os
import re
from datetime import UTC, datetime, timedelta

import httpx
import psycopg
import pytest

from immich_face_api.admin import create_admin_app
from immich_face_api.app import create_app
from immich_face_api.store import Store

from .conftest import KEY, synthetic_jpeg
from .test_app import FACES, NEIGHBOURS, FakeDB, FakeML

URL = os.environ.get("FACE_API_TEST_STORE_URL")
pytestmark = [pytest.mark.asyncio, pytest.mark.skipif(not URL, reason="FACE_API_TEST_STORE_URL not set")]
ADMIN_TOKEN = "admin-token-" + "t" * 40


@pytest.fixture
def clean_db():
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute("DROP SCHEMA public CASCADE")
        conn.execute("CREATE SCHEMA public")
    yield URL


@pytest.fixture
def store_settings(settings, clean_db):
    return dataclasses.replace(
        settings,
        store_url=clean_db,
        admin_token=ADMIN_TOKEN,
        session_secret="s" * 40,
        session_https_only=False,
        key_cache_seconds=0,
    )


async def opened_store(url):
    st = Store(url, None, key_cache_seconds=0)
    await st.open(migrate=True)
    return st


async def test_store_keys_usage_and_purge(clean_db):
    st = await opened_store(clean_db)
    try:
        await st.migrate()  # idempotent
        key = await st.create_key(
            app="web", owner="team", label="prod", rate_limit_per_minute=5, expires_at=None, actor="tester"
        )
        prefix = key.split("_")[1]
        k = await st.lookup(prefix)
        now = datetime.now(UTC)
        assert k.app == "web" and k.rate_limit_per_minute == 5 and k.valid(key, now)
        assert not k.valid(key + "x", now)
        rows = await st.list_keys()
        assert rows[0]["prefix"] == prefix and "secret_hash" not in rows[0]
        assert await st.revoke_key(rows[0]["id"], "tester")
        assert not (await st.lookup(prefix)).valid(key, now)
        expired = await st.create_key(
            app="old", owner="", label="", rate_limit_per_minute=None, expires_at=now - timedelta(days=1), actor="t"
        )
        assert not (await st.lookup(expired.split("_")[1])).valid(expired, now)
        assert await st.lookup("deadbeef") is None

        st.buffer.record("web", "/v1/identify", 200, 300, faces=4, matched=3)
        st.buffer.record("web", "/v1/identify", 200, 900, faces=2, matched=0)
        await st.flush()
        st.buffer.record("web", "/v1/identify", 500, 50)
        await st.flush()
        rows = await st.usage(now - timedelta(hours=2), "web")
        assert len(rows) == 1 and rows[0]["requests"] == 3 and rows[0]["faces"] == 6 and rows[0]["server_errors"] == 1
        assert sum(rows[0]["latency_hist"]) == 3
        assert (await st.usage_by_client(now - timedelta(hours=2)))[0]["client"] == "web"
        actions = [a["action"] for a in await st.audit()]
        assert actions.count("key.create") == 2 and "key.revoke" in actions
        async with st.pool.connection() as conn:
            await conn.execute("UPDATE usage_hourly SET hour = hour - interval '500 days'")
        await st.purge(400)
        assert await st.usage(now - timedelta(days=1000), "web") == []
    finally:
        await st.close()


async def test_api_accepts_store_keys_and_records_usage(store_settings):
    st = await opened_store(store_settings.store_url)
    key = await st.create_key(app="kiosk", owner="", label="", rate_limit_per_minute=2, expires_at=None, actor="t")
    app = create_app(store_settings)
    state = app.state.face
    state.ml, state.db, state.ready = FakeML(FACES), FakeDB(NEIGHBOURS), True
    state.store = st
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        img = synthetic_jpeg(200, 200)
        h = {"Authorization": f"Bearer {key}"}
        codes = [(await c.post("/v1/identify", headers=h, content=img)).status_code for _ in range(3)]
        assert codes == [200, 200, 429]  # per-key limit 2/min
        bad = key[:-2] + ("aa" if not key.endswith("aa") else "bb")
        assert (
            await c.post("/v1/identify", headers={"Authorization": f"Bearer {bad}"}, content=img)
        ).status_code == 401
    await st.flush()
    rows = await st.usage(datetime.now(UTC) - timedelta(hours=1), "kiosk")
    assert rows[0]["requests"] == 3 and rows[0]["faces"] == 4 and rows[0]["client_errors"] == 1
    assert (await st.list_keys())[0]["last_used_at"] is not None
    await st.close()


async def admin_client(store_settings):
    app = create_admin_app(store_settings)
    state = app.state.face
    state.ml, state.db, state.ready = FakeML(FACES), FakeDB(NEIGHBOURS), True
    state.store = await opened_store(store_settings.store_url)
    return app, httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://admin.test")


async def sign_in(c):
    r = await c.get("/login")
    csrf = re.search(r'name="csrf" value="([^"]+)"', r.text).group(1)
    r = await c.post("/login", data={"token": ADMIN_TOKEN, "csrf": csrf})
    assert r.status_code == 303
    return csrf


async def test_admin_flow(store_settings):
    app, c = await admin_client(store_settings)
    async with c:
        assert (await c.get("/")).status_code == 303  # not signed in -> /login
        r = await c.get("/login")
        assert "Admin token" in r.text and "default-src 'self'" in r.headers["content-security-policy"]
        csrf = re.search(r'name="csrf" value="([^"]+)"', r.text).group(1)
        assert (await c.post("/login", data={"token": "wrong", "csrf": csrf})).status_code == 401
        csrf = await sign_in(c)

        # create: key shown once, not stored in the cookie, never shown again
        r = await c.post(
            "/keys",
            data={"csrf": csrf, "app_name": "webportal", "owner": "web team", "rate": "60", "expires_days": "365"},
        )
        assert r.status_code == 200 and "shown only this once" in r.text
        key = re.search(r'<span class="token">(ifa_[^<]+)</span>', r.text).group(1)
        assert key not in str(c.cookies.jar)
        page = await c.get("/keys")
        assert key not in page.text and f"ifa_{key.split('_')[1]}_…" in page.text and "webportal" in page.text
        assert "environment" in page.text  # env keys from settings are listed read-only

        # validation, csrf
        assert (await c.post("/keys", data={"csrf": csrf, "app_name": "bad name!"})).status_code == 422
        assert (await c.post("/keys", data={"csrf": csrf, "app_name": "webportal2", "rate": "0"})).status_code == 422
        assert (await c.post("/keys", data={"csrf": "nope", "app_name": "x"})).status_code == 403
        assert (await c.post("/keys", data={"csrf": csrf, "app_name": "webportal", "owner": "x"})).status_code == 200

        # usage pages render with data
        app.state.face.store.buffer.record("webportal", "/v1/identify", 200, 400, faces=5, matched=4)
        await app.state.face.store.flush()
        for path in ("/", "/?range=24h", "/usage?client=webportal&range=7d", "/usage", "/try"):
            r = await c.get(path)
            assert r.status_code == 200, path
        assert "80.0%" in (await c.get("/?range=24h")).text  # 4 of 5 faces matched
        assert "<svg" in (await c.get("/")).text

        # try it: csrf header required, result has boxes and names
        img = synthetic_jpeg(400, 300)
        assert (await c.post("/try", files={"image": ("x.jpg", img, "image/jpeg")})).status_code == 403
        r = await c.post("/try", files={"image": ("x.jpg", img, "image/jpeg")}, headers={"X-CSRF-Token": csrf})
        assert r.status_code == 200 and r.json()["faces"][1]["name"] == "Alice Example" and "_stats" not in r.json()

        # revoke
        key_id = re.search(r'action="/keys/([0-9a-f-]{36})/revoke"', (await c.get("/keys")).text).group(1)
        assert (await c.post(f"/keys/{key_id}/revoke", data={"csrf": csrf})).status_code == 303
        assert "revoked" in (await c.get("/keys")).text

        # static allowlist, sign out
        assert (await c.get("/static/admin.css")).status_code == 200
        assert (await c.get("/static/../app.py")).status_code == 404
        assert (await c.post("/logout", data={"csrf": csrf})).status_code == 303
        assert (await c.get("/keys")).status_code == 303
    await app.state.face.store.close()


async def test_admin_refuses_env_client_names(store_settings):
    app, c = await admin_client(store_settings)
    async with c:
        csrf = await sign_in(c)
        r = await c.post("/keys", data={"csrf": csrf, "app_name": "website"})  # "website" is an env client in conftest
        assert r.status_code == 422 and "environment" in r.text
    await app.state.face.store.close()


async def test_api_starts_and_serves_env_keys_without_store(settings):
    """A store outage must not take the API down: env keys work, store keys get 503, usage stays in memory."""
    s = dataclasses.replace(settings, store_url="postgresql://nobody@127.0.0.1:9/none?connect_timeout=1")
    app = create_app(s)
    state = app.state.face
    state.ml, state.db, state.ready = FakeML(FACES), FakeDB(NEIGHBOURS), True
    await state.store.open(migrate=True, wait=False)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            img = synthetic_jpeg(200, 200)
            r = await c.post("/v1/identify", headers={"Authorization": f"Bearer {KEY}"}, content=img)
            assert r.status_code == 200
            r = await c.post("/v1/identify", headers={"Authorization": "Bearer ifa_0123abcd_" + "x" * 43}, content=img)
            assert r.status_code == 503
        assert state.store.buffer.drain()  # counted in memory, waiting for the store
    finally:
        await state.store.close()


async def test_admin_docs(store_settings):
    app, c = await admin_client(store_settings)
    async with c:
        assert (await c.get("/docs")).status_code == 303  # behind sign-in
        await sign_in(c)
        r = await c.get("/docs")
        assert r.status_code == 200
        assert "http://admin.test/v1/identify" in r.text  # default: the request's own host
        assert 'data-lang="php"' in r.text and "X-API-Key" in r.text and 'src="/static/docs.js"' in r.text
        assert not re.search(r"<script(?![^>]*\bsrc=)", r.text)  # no inline script (strict CSP)
        assert "&lt;secret&gt;" in r.text  # code is escaped
        assert 'id="verify"' in r.text and 'id="changes"' in r.text and "/v1/people/changes?since=" in r.text
        ref = await c.get("/docs/reference")
        assert ref.status_code == 200
        for path in ("/v1/identify", "/v1/verify", "/v1/verify/batch", "/v1/people", "/v1/people/changes", "/healthz"):
            assert path in ref.text
        assert "box.x" in ref.text and "candidates[].votes" in ref.text
        spec = await c.get("/docs/openapi.json")
        assert spec.status_code == 200 and "attachment" in spec.headers["content-disposition"]
        assert spec.json()["servers"] == [{"url": "http://admin.test"}]
        assert (await c.get("/static/docs.js")).status_code == 200
    await app.state.face.store.close()

    custom = dataclasses.replace(store_settings, docs_base_url="https://faces.example.org")
    app, c = await admin_client(custom)
    async with c:
        await sign_in(c)
        assert "https://faces.example.org/v1/identify" in (await c.get("/docs")).text
    await app.state.face.store.close()
