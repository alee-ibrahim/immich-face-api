"""Optional admin UI (``python -m immich_face_api admin``, port FACE_API_ADMIN_PORT, default 8081).

Server-rendered (Jinja2), no JavaScript build and nothing from a CDN. Pages: dashboard (usage over time), apps and
API keys (create - the key is shown once - and revoke), usage per app, and "Try it" (upload a photo, see boxes and
names; nothing is stored or counted). Sign-in: any OpenID Connect provider (FACE_API_OIDC_*), optionally limited to a
claim value (FACE_API_OIDC_ADMIN_CLAIM / _VALUES); without OIDC, a bootstrap admin token (local use). Every form and
the try-it call carry a per-session CSRF token. Run it on its own port and never route it publicly.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import logging
import re
import secrets
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from importlib.resources import files
from typing import Annotated

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from . import __version__
from .apidocs import reference
from .app import HttpError, State, extract_image, read_body
from .charts import bar_chart, line_chart
from .config import ConfigError, Settings
from .openapi import OPENAPI_SPEC
from .store import series, since_for, totals
from .usage import percentile

log = logging.getLogger("immich_face_api.admin")
templates = Jinja2Templates(directory=str(files(__package__).joinpath("templates")))
STATIC = {
    "admin.css": "text/css",
    "try.js": "text/javascript",
    "confirm.js": "text/javascript",
    "docs.js": "text/javascript",
}
APP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class NotSignedIn(Exception):
    pass


def signed_in_user(request: Request) -> dict:
    u = request.session.get("user")
    if not u:
        raise NotSignedIn()
    return u


User = Annotated[dict, Depends(signed_in_user)]


def ago(ts) -> str:
    if ts is None:
        return "—"
    secs = int((datetime.now(UTC) - ts).total_seconds())
    if secs < 0:
        return "in " + until(ts)
    for unit, n in (("d", 86400), ("h", 3600), ("m", 60)):
        if secs >= n:
            return f"{secs // n}{unit} ago"
    return f"{secs}s ago"


def until(ts) -> str:
    if ts is None:
        return "never"
    secs = int((ts - datetime.now(UTC)).total_seconds())
    if secs <= 0:
        return "expired"
    return f"{secs // 86400}d" if secs >= 86400 else f"{secs // 3600}h"


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "—"


def ms(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v / 1000:.1f} s" if v >= 1000 else f"{v:.0f} ms"


templates.env.filters.update(ago=ago, until=until, ms=ms)
templates.env.globals.update(pct=pct)


def summary(rows: list[dict]) -> dict:
    t = totals(rows)
    return {
        "requests": t.requests,
        "faces": t.faces,
        "matched": t.matched,
        "unknown": t.faces - t.matched,
        "ambiguous": t.ambiguous,
        "server_errors": t.server_errors,
        "client_errors": t.client_errors,
        "p50": percentile(t.latency_hist, 0.5),
        "p95": percentile(t.latency_hist, 0.95),
    }


def charts(rows: list[dict], start: datetime, bucket: str) -> dict:
    pts = series(rows, start, bucket)
    fmt = "%H:00" if bucket == "hour" else "%d %b"
    labels = [t.strftime(fmt) for t, _ in pts]
    req = [c.requests for _, c in pts]
    p95 = [percentile(c.latency_hist, 0.95) for _, c in pts]
    match = [(100 * c.matched / c.faces) if c.faces else None for _, c in pts]
    table = [
        {
            "t": t,
            "label": t.strftime("%Y-%m-%d %H:00" if bucket == "hour" else "%Y-%m-%d"),
            "requests": c.requests,
            "faces": c.faces,
            "matched": c.matched,
            "errors": c.server_errors,
            "p50": percentile(c.latency_hist, 0.5),
            "p95": percentile(c.latency_hist, 0.95),
        }
        for t, c in reversed(pts)
        if c.requests
    ]
    return {
        "requests": bar_chart(labels, req, unit="requests", title="Requests"),
        "p95": line_chart(labels, p95, unit="ms", title="Latency p95", fmt=lambda v: ms(v)),
        "match": line_chart(labels, match, unit="%", title="Match rate", fmt=lambda v: f"{v:.0f}%", y_max=100),
        "table": table,
    }


def create_admin_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    if not settings.store_url:
        raise ConfigError("the admin UI needs FACE_API_STORE_URL (its own PostgreSQL)")
    if not settings.oidc_enabled and not settings.admin_token:
        raise ConfigError("the admin UI needs FACE_API_OIDC_* or FACE_API_ADMIN_TOKEN")
    session_secret = settings.session_secret or (
        hashlib.sha256(("session:" + settings.admin_token).encode()).hexdigest() if settings.admin_token else ""
    )
    if len(session_secret) < 32:
        raise ConfigError("FACE_API_SESSION_SECRET must be at least 32 characters")

    oauth = None
    if settings.oidc_enabled:
        from authlib.integrations.starlette_client import OAuth

        oauth = OAuth()
        oauth.register(
            "idp",
            client_id=settings.oidc_client_id,
            client_secret=settings.oidc_client_secret,
            server_metadata_url=settings.oidc_issuer + "/.well-known/openid-configuration",
            client_kwargs={"scope": settings.oidc_scopes},
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state: State = app.state.face
        await state.db.open()
        await state.store.open(migrate=True)
        task = asyncio.create_task(state.readiness_loop())
        log.info("admin started", extra={"fields": {"version": __version__, "oidc": settings.oidc_enabled}})
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            await state.ml.close()
            await state.db.close()
            await state.store.close()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.face = State(settings)
    app.add_middleware(
        SessionMiddleware,
        secret_key=session_secret,
        session_cookie="face_admin",
        max_age=8 * 3600,
        same_site="lax",
        https_only=settings.session_https_only,
    )

    @app.middleware("http")
    async def headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; "
            "connect-src 'self'; frame-ancestors 'none'; form-action 'self'; base-uri 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.exception_handler(NotSignedIn)
    async def not_signed_in(request: Request, _exc: NotSignedIn):
        return RedirectResponse("/login", 303)

    # ------------------------------------------------------------------ helpers

    def csrf_token(request: Request) -> str:
        tok = request.session.get("csrf")
        if not tok:
            tok = request.session["csrf"] = secrets.token_urlsafe(24)
        return tok

    def check_csrf(request: Request, token: str | None) -> None:
        if not hmac.compare_digest(token or "", request.session.get("csrf", "")):
            raise HttpError(403, "csrf", "form expired; reload the page")

    def render(request: Request, name: str, status: int = 200, **ctx) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            name,
            {
                "user": request.session.get("user"),
                "csrf": csrf_token(request),
                "oidc": settings.oidc_enabled,
                "version": __version__,
                "ready": app.state.face.ready,
                **ctx,
            },
            status_code=status,
        )

    def callback_url(request: Request) -> str:
        return f"{settings.public_url}/auth/callback" if settings.public_url else str(request.url_for("callback"))

    def is_admin(claims: dict) -> bool:
        if not settings.oidc_admin_claim:
            return True  # who may sign in is decided by the identity provider (e.g. assignment required)
        value = claims.get(settings.oidc_admin_claim)
        values = set(value) if isinstance(value, list) else {value} if value else set()
        return bool(values & set(settings.oidc_admin_values))

    # ------------------------------------------------------------------ static, health

    @app.get("/static/{name}")
    async def static(name: str):
        if name not in STATIC:
            return Response(status_code=404)
        data = files(__package__).joinpath("static", name).read_bytes()
        return Response(data, media_type=STATIC[name], headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        st = app.state.face
        ok = st.ready and await st.store.ping()
        return JSONResponse({"status": "ready" if ok else "not ready"}, status_code=200 if ok else 503)

    # ------------------------------------------------------------------ sign-in

    @app.get("/login")
    async def login(request: Request, switch: bool = False):
        if oauth:
            extra = {"prompt": "select_account"} if switch else {}
            return await oauth.idp.authorize_redirect(request, callback_url(request), **extra)
        return render(request, "login.html")

    @app.post("/login")
    async def login_token(request: Request, token: Annotated[str, Form()], csrf: Annotated[str, Form()]):
        try:
            check_csrf(request, csrf)
        except HttpError:
            return render(request, "login.html", 403, error="The form expired; try again.")
        if oauth or not settings.admin_token or not hmac.compare_digest(token, settings.admin_token):
            return render(request, "login.html", 401, error="That is not the admin token.")
        request.session["user"] = {"name": "Admin (token)", "id": "admin-token"}
        return RedirectResponse("/", 303)

    @app.get("/auth/callback", name="callback")
    async def callback(request: Request):
        if not oauth:
            return RedirectResponse("/login", 303)
        try:
            token = await oauth.idp.authorize_access_token(request)
        except Exception:  # noqa: BLE001 - state mismatch, user cancelled, provider error
            log.warning("oidc callback failed")
            return render(request, "login.html", 400, error="Sign-in failed. Try again.")
        claims = token.get("userinfo") or {}
        who = claims.get("email") or claims.get("preferred_username") or claims.get("sub") or "unknown"
        if not is_admin(claims):
            log.warning("admin sign-in refused", extra={"fields": {"user": who}})
            return render(request, "login.html", 403, error=f"{who} is not an admin of this service.")
        request.session.clear()
        request.session["user"] = {"name": claims.get("name") or who, "id": who}
        log.info("admin signed in", extra={"fields": {"user": who}})
        return RedirectResponse("/", 303)

    @app.post("/logout")
    async def logout(request: Request, csrf: Annotated[str, Form()]):
        if hmac.compare_digest(csrf or "", request.session.get("csrf", "")):
            request.session.clear()
        return RedirectResponse("/login" if not oauth else "/signed-out", 303)

    @app.get("/signed-out")
    async def signed_out(request: Request):
        return render(request, "signed_out.html")

    # ------------------------------------------------------------------ dashboard & usage

    async def usage_page(request: Request, u: dict, client: str | None, range_: str, template: str):
        st: State = app.state.face
        start, bucket = since_for(range_)
        rows = await st.store.usage(start, client)
        by_client = await st.store.usage_by_client(start) if client is None else []
        return render(
            request,
            template,
            range=range_ if range_ in ("24h", "7d", "30d", "90d") else "7d",
            client=client,
            clients=await st.store.clients(),
            s=summary(rows),
            c=charts(rows, start, bucket),
            by_client=by_client,
            bucket=bucket,
        )

    @app.get("/")
    async def dashboard(request: Request, u: User, range: str = "7d"):  # noqa: A002
        return await usage_page(request, u, None, range, "dashboard.html")

    @app.get("/usage")
    async def usage(request: Request, u: User, client: str = "", range: str = "30d"):  # noqa: A002
        return await usage_page(request, u, client or None, range, "usage.html")

    # ------------------------------------------------------------------ keys

    async def keys_page(request: Request, status: int = 200, **ctx):
        st: State = app.state.face
        return render(
            request,
            "keys.html",
            status,
            keys=await st.store.list_keys(),
            env_clients=sorted(settings.api_keys),
            audit=await st.store.audit(20),
            default_rate=settings.rate_limit_per_minute,
            **ctx,
        )

    @app.get("/keys")
    async def keys(request: Request, u: User):
        return await keys_page(request)

    @app.post("/keys")
    async def create_key(
        request: Request,
        u: User,
        csrf: Annotated[str, Form()],
        app_name: Annotated[str, Form()],
        owner: Annotated[str, Form()] = "",
        label: Annotated[str, Form()] = "",
        rate: Annotated[str, Form()] = "",
        expires_days: Annotated[str, Form()] = "",
    ):
        try:
            check_csrf(request, csrf)
        except HttpError:
            return await keys_page(request, 403, error="The form expired; reload the page.")
        app_name = app_name.strip()
        if not APP_NAME.fullmatch(app_name):
            return await keys_page(request, 422, error="App name: letters, digits, '.', '_' or '-', up to 64.")
        if app_name in settings.api_keys:
            return await keys_page(
                request, 422, error=f"'{app_name}' is configured by the environment; choose another name."
            )
        try:
            rate_v = int(rate) if rate.strip() else None
            days = int(expires_days) if expires_days.strip() else None
            if (rate_v is not None and not 1 <= rate_v <= 100_000) or (days is not None and not 1 <= days <= 3650):
                raise ValueError
        except ValueError:
            return await keys_page(request, 422, error="Rate limit 1-100000 per minute; expiry 1-3650 days.")
        key = await app.state.face.store.create_key(
            app=app_name,
            owner=owner.strip()[:200],
            label=label.strip()[:200],
            rate_limit_per_minute=rate_v,
            expires_at=datetime.now(UTC) + timedelta(days=days) if days else None,
            actor=u["id"],
        )
        log.info("api key created", extra={"fields": {"app": app_name, "by": u["id"]}})
        # Rendered directly (never stored in the session cookie or anywhere else); this page is not cached.
        return await keys_page(request, created={"app": app_name, "key": key})

    @app.post("/keys/{key_id}/revoke")
    async def revoke(request: Request, u: User, key_id: str, csrf: Annotated[str, Form()]):
        try:
            check_csrf(request, csrf)
        except HttpError:
            return await keys_page(request, 403, error="The form expired; reload the page.")
        if not re.fullmatch(r"[0-9a-f-]{36}", key_id):
            return await keys_page(request, 404, error="No such key.")
        ok = await app.state.face.store.revoke_key(key_id, u["id"])
        log.info("api key revoked", extra={"fields": {"id": key_id, "by": u["id"], "ok": ok}})
        return RedirectResponse("/keys", 303)

    # ------------------------------------------------------------------ docs

    def docs_base(request: Request) -> str:
        if settings.docs_base_url:
            return settings.docs_base_url
        proto = request.headers.get("x-forwarded-proto") or request.url.scheme
        return f"{proto}://{request.headers.get('host') or request.url.netloc}"

    def limits() -> dict:
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
        }

    @app.get("/docs")
    async def docs(request: Request, u: User):
        return render(request, "docs.html", base=docs_base(request), lim=limits())

    @app.get("/docs/reference")
    async def docs_reference(request: Request, u: User):
        return render(request, "docs_reference.html", base=docs_base(request), ops=reference(), spec=OPENAPI_SPEC)

    @app.get("/docs/openapi.json")
    async def docs_openapi(request: Request, u: User):
        spec = dict(OPENAPI_SPEC)
        spec["servers"] = [{"url": docs_base(request)}]
        return JSONResponse(spec, headers={"Content-Disposition": 'attachment; filename="openapi.json"'})

    # ------------------------------------------------------------------ try it

    @app.get("/try")
    async def try_page(request: Request, u: User):
        return render(request, "try.html", max_mb=settings.max_upload_bytes // (1024 * 1024))

    @app.post("/try")
    async def try_identify(request: Request, u: User):
        try:
            check_csrf(request, request.headers.get("x-csrf-token"))
            st: State = app.state.face
            if not st.ready:
                raise HttpError(503, "not_ready", "the service is not ready: " + st.reason)
            body = await read_body(request, settings.max_upload_bytes)
            raw = extract_image(request.headers.get("content-type", ""), body)
            del body
            result = await st.identify(raw, candidates=3, rid="admin-try")
            result.pop("_stats", None)
            return JSONResponse(result)
        except HttpError as exc:
            return JSONResponse({"error": {"code": exc.code, "message": exc.message}}, status_code=exc.status)

    return app


def main() -> None:  # pragma: no cover - process entry point
    import uvicorn

    from .app import setup_logging, setup_sentry

    setup_logging()
    settings = Settings.from_env()
    setup_sentry(settings)
    uvicorn.run(
        create_admin_app(settings),
        host="0.0.0.0",  # noqa: S104 - in a container
        port=settings.admin_port,
        access_log=False,
        server_header=False,
        proxy_headers=False,  # set FACE_API_PUBLIC_URL behind a proxy (OIDC redirect URI)
        log_config=None,
    )
