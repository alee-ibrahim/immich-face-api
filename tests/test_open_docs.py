"""Open (login-free) docs served by the API: only with FACE_API_OPEN_DOCS, static, nothing from the admin."""

import dataclasses
import re

import httpx
import pytest

from immich_face_api.app import create_app

pytestmark = pytest.mark.asyncio

DOC_PATHS = ("/docs", "/docs/integration", "/docs/static/base.css", "/docs/static/docs.css", "/docs/static/docs.js")
# admin pages, admin-only links and settings that must never show up in the open docs
ADMIN_STRINGS = (
    'href="/keys"',
    'href="/requests"',
    'href="/missed"',
    'href="/subjects"',
    'href="/try"',
    'href="/changes"',
    'href="/usage"',
    "/docs/reference",
    "/docs/openapi.json",
    "Apps &amp; keys",
    "Sign out",
    "csrf",
    "purge",
    "Missed faces",
    "include_unnamed",
    "/readyz",
)


def client(settings, **overrides):
    app = create_app(dataclasses.replace(settings, **overrides))
    app.state.face.ready = True
    return app, httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://faces.test")


async def test_off_by_default(settings):
    assert settings.open_docs is False
    app, c = client(settings)
    async with c:
        for path in DOC_PATHS:
            assert (await c.get(path)).status_code == 404, path
        spec = (await c.get("/openapi.json")).json()
        assert spec["servers"] == [{"url": "https://faces.example.org"}]  # unchanged without the flag
    assert not any(getattr(r, "path", "").startswith("/docs") for r in app.routes)


async def test_pages_without_key(settings):
    app, c = client(
        settings, open_docs=True, docs_base_url="https://faces-api.example.net", store_url="postgresql://u@store.test/s"
    )
    async with c:
        for path in DOC_PATHS:
            r = await c.get(path)
            assert r.status_code == 200, path
            assert "x-request-id" in r.headers
            if path.endswith(".js"):
                assert r.headers["content-type"].startswith("text/javascript")
            elif path.endswith(".css"):
                assert r.headers["content-type"].startswith("text/css")
        ref = await c.get("/docs")
        guide = await c.get("/docs/integration")
    for r in (ref, guide):
        assert r.headers["content-type"].startswith("text/html")
        csp = r.headers["content-security-policy"]
        assert "default-src 'none'" in csp and "script-src 'self'" in csp and "frame-ancestors 'none'" in csp
        assert "unsafe-inline" not in csp
        assert r.headers["cache-control"] == "public, max-age=300"
        assert not re.search(r"<script(?![^>]*\bsrc=)", r.text)  # no inline script
        assert not re.search(r"\sstyle=", r.text) and "<style" not in r.text  # no inline style
        assert not re.search(r"""(src|href)=["']?(https?:)?//""", r.text)  # nothing loaded or linked from elsewhere
        assert "https://faces-api.example.net" in r.text
        for s in ADMIN_STRINGS:
            assert s not in r.text, s
    text = ref.text
    for path in (
        "/v1/identify",
        "/v1/verify",
        "/v1/verify/batch",
        "/v1/people",
        "/v1/people/changes",
        "/v1/people/resolve",
        "/healthz",
    ):
        assert path in text, path
    for word in (
        "X-API-Key",
        "Bearer",
        "prominence",
        "photo_type",
        "subject_signals",
        "is_main_subject",
        "co_subject",
        "Retry-After",
        "401",
        "413",
        "415",
        "429",
        "503",
        "operator of this service",
    ):
        assert word in text, word
    for lang in ('data-lang="curl"', 'data-lang="php"', 'data-lang="python"', 'data-lang="node"'):
        assert lang in text
    assert "&lt;your key&gt;" in text  # code is escaped
    assert "box.x" in text and "candidates[].votes" in text  # generated field reference
    g = guide.text
    for word in (
        "member",
        "one member, many person ids",
        "authoritative",
        "/v1/people/changes",
        "merged",
        "faces_moved",
        "renamed",
        "deleted",
        "/v1/verify",
        "co_subject",
        "Server side only",
        "Privacy",
    ):
        assert word.lower() in g.lower(), word


async def test_static_allow_list(settings):
    app, c = client(settings, open_docs=True)
    async with c:
        for name in ("admin.css", "try.js", "confirm.js", "subjects.js", "..%2Fopenapi.json", "open-docs.css", "x"):
            assert (await c.get(f"/docs/static/{name}")).status_code == 404, name
        assert (await c.get("/static/docs.js")).status_code == 404  # the admin's static path is not served


async def test_api_still_needs_key(settings):
    app, c = client(settings, open_docs=True)
    async with c:
        assert (await c.get("/v1/people")).status_code == 401
        assert (await c.post("/v1/identify", content=b"x")).status_code == 401


async def test_openapi_server_and_base_url(settings):
    app, c = client(settings, open_docs=True)
    async with c:
        assert (await c.get("/openapi.json")).json()["servers"] == [{"url": "http://faces.test"}]
        r = await c.get("/docs", headers={"x-forwarded-proto": "https"})
        assert "https://faces.test/v1/identify" in r.text
        bad = await c.get("/docs", headers={"host": 'evil.test"><script>'})
        assert bad.status_code in (200, 400)
        if bad.status_code == 200:
            assert "<script>" not in bad.text.replace('<script src="/docs/static/docs.js"></script>', "")
            assert "https://faces.example.org" in bad.text
    app, c = client(settings, open_docs=True, enable_openapi=False)
    async with c:
        assert (await c.get("/openapi.json")).status_code == 404
        assert 'href="/openapi.json"' not in (await c.get("/docs")).text


async def test_sections_follow_settings(settings):
    app, c = client(settings, open_docs=True, enable_people=False, subject_scoring="off")
    async with c:
        text = (await c.get("/docs")).text
        guide = (await c.get("/docs/integration")).text
    assert "/v1/people/changes" not in text and 'id="people"' not in text
    assert "prominence" not in text.split('id="fields"')[0]  # the generated field reference lists optional fields
    assert 'id="sync"' not in guide and 'id="profiles"' not in guide


async def test_openapi_documents_every_api_route(settings):
    """The docs pages are not API routes: the OpenAPI document still matches what the API serves."""
    app, c = client(settings, open_docs=True)
    async with c:
        spec = (await c.get("/openapi.json")).json()
    documented = {(path, method.upper()) for path, ops in spec["paths"].items() for method in ops}
    served = {
        (route.path, m)
        for route in app.routes
        for m in getattr(route, "methods", set())
        if m != "HEAD" and not route.path.startswith("/docs")
    }
    assert served == documented
