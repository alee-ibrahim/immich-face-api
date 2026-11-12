"""API tests with a fake ML server and a fake database (no network, no real faces)."""

import dataclasses

import httpx
import pytest

from immich_face_api.app import create_app
from immich_face_api.matching import Neighbour
from immich_face_api.ml import DetectedFace

from .conftest import KEY, OTHER_KEY, synthetic, synthetic_jpeg

pytestmark = pytest.mark.asyncio

AUTH = {"Authorization": f"Bearer {KEY}"}


class FakeML:
    def __init__(self, faces):
        self.faces = faces
        self.calls = []

    async def detect(self, jpeg):
        self.calls.append(jpeg)
        return self.faces

    async def ping(self):
        return True

    async def close(self):
        pass


class FakeDB:
    def __init__(self, neighbours):
        self.neighbours = neighbours
        self.include_unnamed = None
        self.status = type("S", (), {"immich_version": "3.4.1", "owner_id": "o"})()

    async def nearest(self, embeddings, include_unnamed=False):
        self.include_unnamed = include_unnamed
        return self.neighbours[: len(embeddings)]

    async def people(self):
        return [{"person_id": "p1", "name": "Alice Example", "face_count": 12}]


FACES = [
    DetectedFace(box={"x1": 600, "y1": 100, "x2": 700, "y2": 220}, score=0.91, embedding="[0.1]"),
    DetectedFace(box={"x1": 100, "y1": 100, "x2": 200, "y2": 220}, score=0.88, embedding="[0.2]"),
]
NEIGHBOURS = [
    [
        Neighbour("p1", "Alice Example", 0.31),
        Neighbour("p1", "Alice Example", 0.35),
        Neighbour("p2", "Bob Example", 0.44),
    ],
    [Neighbour("p3", "Carol Example", 0.62)],  # beyond max_distance -> unknown
]


def make(settings, faces=FACES, neighbours=NEIGHBOURS, ready=True, **overrides):
    if overrides:
        settings = dataclasses.replace(settings, **overrides)
    app = create_app(settings)
    st = app.state.face
    st.ml, st.db, st.ready = FakeML(faces), FakeDB(neighbours), ready
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    return app, client


async def test_identify_multipart(settings):
    app, c = make(settings)
    r = await c.post("/v1/identify", headers=AUTH, files={"image": ("x.jpg", synthetic_jpeg(800, 600), "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image"] == {"w": 800, "h": 600}
    left, right = body["faces"]  # sorted left to right
    assert right["name"] == "Alice Example" and right["person_id"] == "p1"
    assert right["distance"] == 0.31 and right["confidence"] == 0.69
    assert right["box"] == {"x": 0.75, "y": 0.16667, "w": 0.125, "h": 0.2}
    assert "candidates" not in right
    assert left["name"] is None and left["person_id"] is None
    assert r.headers["cache-control"] == "no-store"
    assert "access-control-allow-origin" not in r.headers


async def test_identify_raw_and_candidates(settings):
    app, c = make(settings)
    r = await c.post(
        "/v1/identify?candidates=3", headers={**AUTH, "Content-Type": "image/png"}, content=synthetic("PNG", 800, 600)
    )
    assert r.status_code == 200
    right = r.json()["faces"][1]
    assert [x["person_id"] for x in right["candidates"]] == ["p1", "p2"]
    assert right["candidates"][0]["votes"] == 2


async def test_auth(settings):
    app, c = make(settings)
    img = synthetic_jpeg()
    assert (await c.post("/v1/identify", content=img)).status_code == 401
    r = await c.post("/v1/identify", content=img, headers={"Authorization": "Bearer wrong-" + "z" * 40})
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert (
        await c.post("/v1/identify", content=img, headers={"Authorization": f"Bearer {OTHER_KEY}"})
    ).status_code == 200
    assert (await c.get("/v1/people")).status_code == 401
    assert app.state.face.ml.calls.__len__() == 1


async def test_bad_inputs(settings):
    app, c = make(settings, max_upload_bytes=50_000)
    assert (await c.post("/v1/identify", headers=AUTH, content=b"not an image at all")).status_code == 415
    assert (await c.post("/v1/identify", headers=AUTH, content=b"\0" * 200_000)).status_code == 413
    r = await c.post("/v1/identify", headers=AUTH, files={"other": ("x.jpg", synthetic_jpeg(), "image/jpeg")})
    assert r.status_code == 400
    assert (await c.post("/v1/identify?candidates=99", headers=AUTH, content=synthetic_jpeg())).status_code == 400
    assert (await c.post("/v1/identify?candidates=abc", headers=AUTH, content=synthetic_jpeg())).status_code == 400
    assert app.state.face.ml.calls == []


async def test_test_mode_gate(settings):
    app, c = make(settings)
    r = await c.post("/v1/identify?include_unnamed=1", headers=AUTH, content=synthetic_jpeg())
    assert r.status_code == 400
    app, c = make(settings, test_mode=True)
    r = await c.post("/v1/identify?include_unnamed=1", headers=AUTH, content=synthetic_jpeg())
    assert r.status_code == 200 and r.json()["test_mode"] is True
    assert app.state.face.db.include_unnamed is True


async def test_rate_limit(settings):
    app, c = make(settings, rate_limit_burst=2, rate_limit_per_minute=1)
    img = synthetic_jpeg(200, 200)
    codes = [(await c.post("/v1/identify", headers=AUTH, content=img)).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    r = await c.post("/v1/identify", headers=AUTH, content=img)
    assert int(r.headers["retry-after"]) >= 1
    # another client has its own bucket
    assert (
        await c.post("/v1/identify", headers={"Authorization": f"Bearer {OTHER_KEY}"}, content=img)
    ).status_code == 200


async def test_not_ready(settings):
    app, c = make(settings, ready=False)
    assert (await c.post("/v1/identify", headers=AUTH, content=synthetic_jpeg())).status_code == 503
    assert (await c.get("/readyz")).status_code == 503
    assert (await c.get("/healthz")).status_code == 200


async def test_people(settings):
    app, c = make(settings)
    r = await c.get("/v1/people", headers=AUTH)
    assert r.status_code == 200 and r.json()["count"] == 1
    app, c = make(settings, enable_people=False)
    assert (await c.get("/v1/people", headers=AUTH)).status_code == 404


async def test_no_cors_no_docs(settings):
    app, c = make(settings)
    r = await c.options(
        "/v1/identify", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}
    )
    assert r.status_code == 405 and "access-control-allow-origin" not in r.headers
    for path in ("/docs", "/redoc", "/metrics"):
        assert (await c.get(path)).status_code == 404


async def test_no_faces(settings):
    app, c = make(settings, faces=[], neighbours=[])
    r = await c.post("/v1/identify", headers=AUTH, content=synthetic_jpeg())
    assert r.status_code == 200 and r.json()["faces"] == []


async def test_openapi_served_and_complete(settings):
    app, c = make(settings)
    r = await c.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    assert spec["openapi"].startswith("3.1")
    documented = {(path, method.upper()) for path, ops in spec["paths"].items() for method in ops}
    served = {(route.path, m) for route in app.routes for m in getattr(route, "methods", set()) if m != "HEAD"}
    assert served == documented
    app, c = make(settings, enable_openapi=False)
    assert (await c.get("/openapi.json")).status_code == 404
