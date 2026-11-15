"""Admin "Label subjects" against a real PostgreSQL store (FACE_API_TEST_STORE_URL): roles, sampling from a fake
Immich, uploads, saving labels, and calibrate reading them back. ML and Immich's database are faked."""

import asyncio
import dataclasses
import json
import re

import httpx
import pytest
from fastapi import Request

from immich_face_api import calibrate as cal
from immich_face_api import labels as lb
from immich_face_api.admin import create_admin_app
from immich_face_api.immich_api import ImmichAssets

from .test_changes import VFACES, VerifyDB
from .test_store_admin import URL, clean_db, opened_store, sign_in, store_settings  # noqa: F401 - fixtures
from .test_subject_api import photo

pytestmark = [pytest.mark.asyncio, pytest.mark.skipif(not URL, reason="FACE_API_TEST_STORE_URL not set")]
ASSET = "aaaaaaaa-0000-4000-8000-000000000001"


class FakeML:
    async def detect(self, image, landmarks=False):
        return VFACES

    async def ping(self):
        return True

    async def close(self):
        pass


class SampleDB(VerifyDB):
    def __init__(self, assets):
        super().__init__()
        self.assets = assets

    async def sample_assets(self, limit=60, named=False):
        self.named = named
        return [(a, 2) for a in self.assets]


async def client(settings, monkeypatch, assets=(ASSET,), previews=None):
    previews = previews if previews is not None else {ASSET: photo()}

    async def preview(self, asset_id):
        data = previews.get(asset_id)
        return (data, "image/png") if data else None

    monkeypatch.setattr(ImmichAssets, "preview", preview)
    app = create_admin_app(settings)
    st = app.state.face
    st.ml, st.db, st.ready = FakeML(), SampleDB(list(assets)), True
    st.store = await opened_store(settings.store_url)

    async def as_user(request: Request):  # test-only sign-in with a role (OIDC is not available here)
        data = await request.json()
        request.session["user"] = data
        return {"ok": True}

    app.add_api_route("/_test_login", as_user, methods=["POST"])
    return app, httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://admin.test")


def csrf_of(html: str) -> str:
    return re.search(r'name="csrf" value="([^"]+)"', html).group(1)


async def test_label_from_immich_and_calibrate(store_settings, monkeypatch, tmp_path):  # noqa: F811
    settings = dataclasses.replace(store_settings, immich_api_url="http://immich.test", immich_api_key="k" * 40)
    app, c = await client(settings, monkeypatch)
    async with c:
        await sign_in(c)
        r = await c.get("/subjects")
        assert r.status_code == 200 and "Label a random photo" in r.text
        r = await c.get("/subjects/next?named=1")
        assert r.status_code == 303
        loc = r.headers["location"]
        sid = int(re.match(r"/subjects/(\d+)", loc).group(1))
        assert app.state.face.db.named is True
        page = await c.get(loc)
        assert page.status_code == 200 and f'src="/subjects/{sid}/image"' in page.text
        assert "model:" not in page.text  # no prediction shown before labelling
        img = await c.get(f"/subjects/{sid}/image")
        assert img.status_code == 200 and img.headers["cache-control"].startswith("no-store")
        assert img.content == photo()

        # one main at most
        csrf = csrf_of(page.text)
        bad = await c.post(
            f"/subjects/{sid}", data={"csrf": csrf, "role_0": "main", "role_1": "main", "action": "save"}
        )
        assert "msg=invalid" in bad.headers["location"]
        r = await c.post(f"/subjects/{sid}", data={"csrf": csrf, "role_0": "main", "action": "next", "named": "1"})
        assert r.status_code == 303 and r.headers["location"] == "/subjects/next?named=1"
        sample = await lb.get_sample(app.state.face.store, sid)
        assert sample["labels"] == ["main", "background"] and sample["labeled_by"] == "admin-token"
        assert sample["asset_id"] == ASSET and sample["faces"][0]["signals"]["sharpness"] > 0

        # the asset is labelled: the next draw finds nothing new
        r = await c.get("/subjects/next")
        assert r.headers["location"] == "/subjects?msg=no-sample"
        stats = await lb.stats(app.state.face.store)
        assert stats["photos"] == 1 and stats["faces"]["main"] == 1 and stats["agreement"] == 1.0

        # bad csrf
        r = await c.post(f"/subjects/{sid}", data={"csrf": "x", "role_0": "co_subject"})
        assert r.headers["location"] == "/subjects?msg=expired"

    # calibrate reads the store (export)
    monkeypatch.setenv("FACE_API_STORE_URL", settings.store_url)
    out = tmp_path / "photos.jsonl"
    assert await asyncio.to_thread(cal.main, ["--export", str(out)]) == 0  # the CLI runs its own loop
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert rows[0]["labels"] == ["main", "background"] and len(rows[0]["signals"]) == 2
    assert await asyncio.to_thread(cal.main, []) == 0  # report on the current model


async def test_upload_and_skip(store_settings, monkeypatch):  # noqa: F811
    app, c = await client(store_settings, monkeypatch)  # no Immich API configured
    async with c:
        csrf = await sign_in(c)
        r = await c.get("/subjects")
        assert "Label a random photo" not in r.text
        assert (await c.get("/subjects/next")).headers["location"] == "/subjects?msg=no-immich"
        r = await c.post("/subjects/upload", files={"image": ("x.png", photo(), "image/png")})
        assert r.status_code == 403  # csrf header missing
        r = await c.post(
            "/subjects/upload", files={"image": ("x.png", photo(), "image/png")}, headers={"X-CSRF-Token": csrf}
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert len(data["faces"]) == 2 and set(data["faces"][0]) == {"box"}  # boxes only, no prediction
        sid = data["id"]
        assert (await c.get(f"/subjects/{sid}/image")).status_code == 404  # uploads are never served back
        r = await c.post(f"/subjects/{sid}", data={"csrf": csrf, "action": "skip"})
        assert r.headers["location"] == "/subjects?msg=saved"
        sample = await lb.get_sample(app.state.face.store, sid)
        assert sample["skipped"] and sample["labels"] is None
        assert await lb.training_photos(app.state.face.store) == []


async def test_curator_role(store_settings, monkeypatch):  # noqa: F811
    settings = dataclasses.replace(store_settings, immich_api_url="http://immich.test", immich_api_key="k" * 40)
    app, c = await client(settings, monkeypatch)
    async with c:
        await c.post("/_test_login", json={"name": "Cura", "id": "cura@example.org", "role": "curator"})
        for path in ("/", "/keys", "/requests", "/missed", "/try", "/docs", "/changes"):
            r = await c.get(path)
            assert r.status_code == 303 and r.headers["location"] == "/subjects", path
        r = await c.get("/subjects")
        assert r.status_code == 200 and 'href="/keys"' not in r.text and "Label subjects" in r.text
        loc = (await c.get("/subjects/next")).headers["location"]
        sid = int(re.match(r"/subjects/(\d+)", loc).group(1))
        assert (await c.get(f"/subjects/{sid}/image")).status_code == 200

        # another curator cannot see this sample or its image
        await c.post("/_test_login", json={"name": "Other", "id": "other@example.org", "role": "curator"})
        assert (await c.get(f"/subjects/{sid}")).status_code == 404
        assert (await c.get(f"/subjects/{sid}/image")).status_code == 404
        # an admin can
        await c.post("/_test_login", json={"name": "Admin", "id": "admin@example.org", "role": "admin"})
        assert (await c.get(f"/subjects/{sid}/image")).status_code == 200
        # signed out: nothing
        c.cookies.clear()
        assert (await c.get(f"/subjects/{sid}/image")).status_code == 303


async def test_role_claims(store_settings):  # noqa: F811
    from immich_face_api.admin import role_from_claims

    s = dataclasses.replace(
        store_settings, oidc_admin_claim="roles", oidc_admin_values=("admin",), oidc_curator_values=("curator",)
    )
    assert role_from_claims(s, {"roles": ["admin", "curator"]}) == "admin"
    assert role_from_claims(s, {"roles": ["curator"]}) == "curator"
    assert role_from_claims(s, {"roles": ["other"]}) is None
    assert role_from_claims(s, {}) is None
    assert role_from_claims(store_settings, {}) == "admin"  # no claim configured: the IdP decides who gets in
