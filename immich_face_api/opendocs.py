"""Open (login-free), read-only docs served by the API process itself (``FACE_API_OPEN_DOCS=true``, off by default).

``GET /docs`` (the API reference), ``GET /docs/integration`` (the integration guide) and ``GET /docs/static/<name>``
(the stylesheet and the tab/copy script), next to ``GET /openapi.json``. No key needed, nothing is read from the
databases, and nothing from the admin UI is linked or shown: no admin paths, no key management, no request log. The
pages are server-rendered with a strict Content-Security-Policy (no inline script or style, nothing from elsewhere).
The base URL in the examples is ``FACE_API_DOCS_BASE_URL`` (default: the request's own host).

Whether the pages are reachable from the internet or only from an internal network is the gateway's choice: route
the exact paths ``/docs``, ``/docs/integration`` and ``/openapi.json`` and the prefix ``/docs/static/`` to the API
on the hosts where developers should read them.
"""

from __future__ import annotations

import re
from importlib.resources import files
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from . import __version__
from .apidocs import reference
from .openapi import OPENAPI_SPEC

if TYPE_CHECKING:
    from .config import Settings

# the API paths a client can call (the reference lists only these; /readyz and /openapi.json are not API calls)
PUBLIC_PATHS = (
    "/v1/identify",
    "/v1/verify",
    "/v1/verify/batch",
    "/v1/people",
    "/v1/people/changes",
    "/v1/people/resolve",
    "/healthz",
)
# served name -> (file in static/, media type); everything else under /docs/static/ is 404
STATIC = {
    "base.css": ("admin.css", "text/css"),
    "docs.css": ("open-docs.css", "text/css"),
    "docs.js": ("docs.js", "text/javascript"),
}
CSP = (
    "default-src 'none'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'none'; "
    "font-src 'none'; frame-ancestors 'none'; form-action 'none'; base-uri 'none'"
)
HTML_HEADERS = {
    "Content-Security-Policy": CSP,
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Cache-Control": "public, max-age=300",
}

templates = Jinja2Templates(directory=str(files(__package__).joinpath("templates")))


def _inline_code(text: str) -> Markup:
    """Escape, then turn `x` into <code>x</code> (OpenAPI descriptions)."""
    parts = str(escape(text or "")).split("`")
    return Markup("".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts)))  # noqa: S704 - escaped


def _codetext(body) -> Markup:
    """A code block captured with {% set %} (template text, not yet escaped) -> escaped, outer newlines dropped."""
    return escape(str(body).strip("\r\n"))


templates.env.filters.update(inline_code=_inline_code, codetext=_codetext)


HOST = re.compile(r"^[A-Za-z0-9.-]{1,253}(:[0-9]{1,5})?$")


def base_url(settings: Settings, request: Request) -> str:
    """FACE_API_DOCS_BASE_URL, else the request's own scheme and host (only if they look like one: they come from
    the client)."""
    if settings.docs_base_url:
        return settings.docs_base_url
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.headers.get("host") or request.url.netloc
    if proto not in ("http", "https") or not HOST.match(host):
        return "https://faces.example.org"
    return f"{proto}://{host}"


def spec_for(settings: Settings, request: Request) -> dict:
    """The bundled OpenAPI document with this deployment's base URL as its server (for ``GET /openapi.json``)."""
    if not settings.open_docs:
        return OPENAPI_SPEC
    return {**OPENAPI_SPEC, "servers": [{"url": base_url(settings, request)}]}


def limits(settings: Settings, subject_version: str) -> dict:
    changes = settings.enable_people and bool(settings.store_url) and settings.changes_interval > 0
    return {
        "max_mb": round(settings.max_upload_bytes / (1024 * 1024), 1),
        "max_mp": round(settings.max_pixels / 1_000_000),
        "rate": settings.rate_limit_per_minute,
        "burst": settings.rate_limit_burst,
        "ml_timeout": int(settings.ml_timeout),
        "queue_timeout": int(settings.ml_queue_timeout),
        "max_candidates": settings.max_candidates,
        "max_distance": settings.max_distance,
        "model": settings.face_model,
        "immich": settings.expected_immich_version,
        "max_edge": settings.max_edge,
        "people": settings.enable_people,
        "verify_claims": settings.verify_max_claims,
        "verify_batch": settings.verify_batch_max,
        "changes": changes,
        "changes_minutes": max(1, settings.changes_interval // 60),
        "changes_days": settings.changes_retention_days,
        "subject": settings.subject_scoring,
        "subject_version": subject_version,
        "openapi": settings.enable_openapi,
        # what the service keeps (for the privacy notes); no settings names, no admin paths
        "request_log": bool(settings.store_url and settings.request_log),
        "request_days": settings.request_log_retention_days,
        "crops": settings.store_crops if settings.store_url and settings.request_log else "none",
        "crop_days": settings.crop_retention_days,
    }


def register(app: FastAPI, settings: Settings) -> None:
    """Add the open docs routes to the API app; nothing is added unless ``settings.open_docs``."""
    if not settings.open_docs:
        return
    ops = [op for op in reference() if op["path"] in PUBLIC_PATHS]
    if not settings.test_mode:  # a parameter that only answers 400 here: leave it out
        for op in ops:
            op["parameters"] = [p for p in op["parameters"] if p["name"] != "include_unnamed"]

    def render(request: Request, name: str, **ctx) -> HTMLResponse:
        state = request.app.state.face
        lim = limits(settings, state.subject_model.version)
        if not lim["people"]:
            hidden = {"/v1/people", "/v1/people/changes", "/v1/people/resolve"}
        elif not lim["changes"]:
            hidden = {"/v1/people/changes"}
        else:
            hidden = set()
        return templates.TemplateResponse(
            request,
            name,
            {
                "base": base_url(settings, request),
                "lim": lim,
                "ops": [op for op in ops if op["path"] not in hidden],
                "spec": OPENAPI_SPEC,
                "version": __version__,
                **ctx,
            },
            headers=HTML_HEADERS,
        )

    @app.get("/docs", include_in_schema=False)
    async def open_docs(request: Request):
        return render(request, "open_docs.html", page="reference")

    @app.get("/docs/integration", include_in_schema=False)
    async def open_integration(request: Request):
        return render(request, "open_integration.html", page="integration")

    @app.get("/docs/static/{name}", include_in_schema=False)
    async def open_static(name: str):
        entry = STATIC.get(name)
        if entry is None:
            return Response(status_code=404)
        filename, media = entry
        data = files(__package__).joinpath("static", filename).read_bytes()
        return Response(data, media_type=media, headers={"Cache-Control": "public, max-age=3600"})
