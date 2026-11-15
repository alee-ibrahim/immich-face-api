"""Optional admin UI (``python -m immich_face_api admin``, port FACE_API_ADMIN_PORT, default 8081).

Server-rendered (Jinja2), no JavaScript build and nothing from a CDN. Pages: dashboard (usage over time), apps and
API keys (create - the key is shown once - and revoke), usage per app, the request log ("Requests") and "Missed faces"
(unknown, ambiguous and near-threshold faces with their crops, if crops are stored), "Try it" (upload a photo, see
boxes, names and subject roles; nothing is stored or counted) and "Label subjects" (curators mark the main subject of
sample photos for calibration; only labels and measured signals are stored). Face crops and Immich previews are served
only by these signed-in routes, never cached. Sign-in: any OpenID Connect provider (FACE_API_OIDC_*), optionally
limited by a role claim (FACE_API_OIDC_ADMIN_CLAIM): FACE_API_OIDC_ADMIN_VALUES give the admin role,
FACE_API_OIDC_CURATOR_VALUES the curator role (only "Label subjects"); without OIDC, a bootstrap admin token (local
use). Every form and the try-it call carry a per-session CSRF token. Run it on its own port and never route it publicly.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import logging
import re
import secrets
import time
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, timedelta
from importlib.resources import files
from typing import Annotated

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape
from starlette.middleware.sessions import SessionMiddleware

from . import __version__
from . import changes as ch
from . import labels as lb
from .apidocs import reference
from .app import HttpError, State, extract_image, read_body
from .charts import bar_chart, line_chart
from .config import ConfigError, Settings
from .immich_api import ImmichAssets, ImmichError
from .matching import normalise_box
from .openapi import OPENAPI_SPEC
from .requestlog import embedding_text
from .store import series, since_for, totals
from .usage import percentile

log = logging.getLogger("immich_face_api.admin")
templates = Jinja2Templates(directory=str(files(__package__).joinpath("templates")))
STATIC = {
    "admin.css": "text/css",
    "try.js": "text/javascript",
    "confirm.js": "text/javascript",
    "docs.js": "text/javascript",
    "subjects.js": "text/javascript",
}
APP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class NotSignedIn(Exception):
    pass


class NotAllowed(Exception):
    """Signed in, but as a curator on an admin page."""


def signed_in_user(request: Request) -> dict:
    """An admin (every page)."""
    u = request.session.get("user")
    if not u:
        raise NotSignedIn()
    if u.get("role", "admin") != "admin":
        raise NotAllowed()
    return u


def signed_in_curator(request: Request) -> dict:
    """An admin or a curator ("Label subjects" only)."""
    u = request.session.get("user")
    if not u:
        raise NotSignedIn()
    return u


def role_from_claims(settings: Settings, claims: dict) -> str | None:
    """'admin', 'curator' or None (no access), from the role claim (FACE_API_OIDC_ADMIN_CLAIM, e.g. Entra app roles
    in "roles"); without a claim configured everyone the identity provider lets in is an admin (e.g. Entra
    "assignment required")."""
    if not settings.oidc_admin_claim:
        return "admin"
    value = claims.get(settings.oidc_admin_claim)
    values = set(value) if isinstance(value, list) else {value} if value else set()
    if values & set(settings.oidc_admin_values):
        return "admin"
    if values & set(settings.oidc_curator_values):
        return "curator"
    return None


User = Annotated[dict, Depends(signed_in_user)]
Curator = Annotated[dict, Depends(signed_in_curator)]


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


def inline_code(text: str) -> Markup:
    """Escape, then turn `x` into <code>x</code> (OpenAPI descriptions)."""
    parts = str(escape(text or "")).split("`")
    return Markup("".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts)))  # noqa: S704 - parts are escaped


templates.env.filters.update(ago=ago, until=until, ms=ms, inline_code=inline_code)
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
    missed = [(100 * (c.faces - c.matched) / c.faces) if c.faces else None for _, c in pts]
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
        "missed": line_chart(labels, missed, unit="%", title="Missed-face rate", fmt=lambda v: f"{v:.0f}%", y_max=100),
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
        cache_task = asyncio.create_task(state.faces.loop()) if state.faces is not None else None
        log.info(
            "admin started",
            extra={
                "fields": {
                    "version": __version__,
                    "oidc": settings.oidc_enabled,
                    "subject_model": state.subject_model.version,
                    "label_from_immich": assets is not None,
                }
            },
        )
        try:
            yield
        finally:
            for t in (task, cache_task):
                if t is not None:
                    t.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await t
            await state.ml.close()
            await state.db.close()
            await state.store.close()
            if assets is not None:
                await assets.close()

    assets = (
        ImmichAssets(settings.immich_api_url, settings.immich_api_key)
        if settings.immich_api_url and settings.immich_api_key
        else None
    )
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.face = State(settings)
    if app.state.face.faces is not None:
        # the admin needs the face cache only for "Try it": load it on first use, drop it when idle
        app.state.face.faces.idle_seconds = settings.admin_face_cache_idle_seconds
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

    @app.exception_handler(NotAllowed)
    async def not_allowed(request: Request, _exc: NotAllowed):
        return RedirectResponse("/subjects", 303)

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
                "person_url": person_url,
                **ctx,
            },
            status_code=status,
        )

    def callback_url(request: Request) -> str:
        return f"{settings.public_url}/auth/callback" if settings.public_url else str(request.url_for("callback"))

    def role_of(claims: dict) -> str | None:
        return role_from_claims(settings, claims)

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
        request.session["user"] = {"name": "Admin (token)", "id": "admin-token", "role": "admin"}
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
        role = role_of(claims)
        if role is None:
            log.warning("admin sign-in refused", extra={"fields": {"user": who}})
            return render(request, "login.html", 403, error=f"{who} is not an admin or curator of this service.")
        request.session.clear()
        request.session["user"] = {"name": claims.get("name") or who, "id": who, "role": role}
        log.info("admin signed in", extra={"fields": {"user": who, "role": role}})
        return RedirectResponse("/" if role == "admin" else "/subjects", 303)

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
        for a in by_client:
            a["unknown"] = a["faces"] - a["matched"]
        top_unknown = sorted((a for a in by_client if a["unknown"]), key=lambda a: -a["unknown"])[:10]
        recent, tracker = [], None
        if template == "dashboard.html" and settings.changes_interval:
            with contextlib.suppress(Exception):
                recent = await ch.list_changes(st.store, limit=8, newest_first=True)
                tracker = await ch.tracker_status(st.store)
        return render(
            request,
            template,
            recent_changes=recent,
            tracker=tracker,
            person_url=person_url,
            range=range_ if range_ in ("24h", "7d", "30d", "90d") else "7d",
            client=client,
            clients=await st.store.clients(),
            s=summary(rows),
            c=charts(rows, start, bucket),
            by_client=by_client,
            top_unknown=top_unknown,
            bucket=bucket,
            log_on=settings.request_log,
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
            audit=await st.store.audit(20, "key."),
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

    # ------------------------------------------------------------------ request log, missed faces, crops

    def day(value: str) -> datetime | None:
        try:
            return datetime.combine(date.fromisoformat(value), datetime.min.time(), UTC) if value else None
        except ValueError:
            return None

    def int_or_none(value: str) -> int | None:
        return int(value) if value.isdigit() and len(value) < 19 else None

    def privacy() -> dict:
        return {
            "request_log": settings.request_log,
            "request_days": settings.request_log_retention_days,
            "crops": settings.store_crops,
            "crop_days": settings.crop_retention_days,
            "crop_size": settings.crop_size,
            "max_faces": settings.log_max_faces,
            "near_margin": settings.near_margin,
            "max_distance": settings.max_distance,
        }

    def person_url(person_id: str | None) -> str:
        return f"{settings.immich_url}/people/{person_id}" if settings.immich_url and person_id else ""

    @app.get("/requests")
    async def requests_page(
        request: Request,
        u: User,
        client: str = "",
        since: str = "",
        until: str = "",
        unknowns: str = "",
        status: str = "",
        before: str = "",
        purged: str = "",
    ):
        st: State = app.state.face
        status = status if status in ("ok", "error") or re.fullmatch(r"[1-5]\d\d", status) else ""
        to = day(until)
        rows = await st.store.list_requests(
            client=client or None,
            since=day(since),
            until=to + timedelta(days=1) if to else None,
            unknowns=unknowns == "1",
            status=status or None,
            before=int_or_none(before),
            limit=50,
        )
        return render(
            request,
            "requests.html",
            rows=rows,
            clients=await st.store.log_clients(),
            f={"client": client, "since": since, "until": until, "unknowns": unknowns == "1", "status": status},
            next_before=rows[-1]["id"] if len(rows) == 50 else None,
            stats=await st.store.log_stats(),
            audit=await st.store.audit(10, "log."),
            p=privacy(),
            purged=purged,
        )

    @app.get("/requests/{request_id}")
    async def request_detail(request: Request, u: User, request_id: str):
        rid = int_or_none(request_id)
        req, faces = await app.state.face.store.get_request(rid) if rid else (None, [])
        if not req:
            return render(request, "request.html", 404, req=None, faces=[], p=privacy())
        return render(request, "request.html", req=req, faces=faces, p=privacy())

    @app.get("/missed")
    async def missed_page(
        request: Request,
        u: User,
        client: str = "",
        since: str = "",
        until: str = "",
        kind: str = "",
        before: str = "",
    ):
        st: State = app.state.face
        kinds = (kind,) if kind in ("unknown", "ambiguous", "near") else ("unknown", "ambiguous", "near")
        to = day(until)
        rows = await st.store.missed_faces(
            client=client or None,
            since=day(since),
            until=to + timedelta(days=1) if to else None,
            kinds=kinds,
            before=int_or_none(before),
            limit=60,
        )
        return render(
            request,
            "missed.html",
            rows=rows,
            clients=await st.store.log_clients(),
            f={"client": client, "since": since, "until": until, "kind": kind if len(kinds) == 1 else ""},
            next_before=rows[-1]["id"] if len(rows) == 60 else None,
            p=privacy(),
        )

    @app.get("/changes")
    async def changes_page(request: Request, u: User, type: str = "", person: str = "", before: str = ""):  # noqa: A002
        st: State = app.state.face
        types = (type,) if type in ch.TYPES else ch.TYPES
        person = person.strip().lower()
        pid = person if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", person) else None
        rows = await ch.list_changes(
            st.store, types=types, person_id=pid, before_id=int_or_none(before), limit=100, newest_first=True
        )
        return render(
            request,
            "changes.html",
            rows=rows,
            t=await ch.tracker_status(st.store),
            enabled=settings.changes_interval > 0,
            interval=settings.changes_interval,
            full_hours=settings.changes_full_hours,
            retention=settings.changes_retention_days,
            types=ch.TYPES,
            f={"type": type if len(types) == 1 else "", "person": pid or ""},
            next_before=rows[-1]["id"] if len(rows) == 100 else None,
            person_url=person_url,
        )

    @app.get("/crops/{name}")
    async def crop(u: User, name: str):
        """A face crop: signed-in admins only, never cached, never on the API port."""
        found = re.fullmatch(r"(\d{1,18})\.jpg", name)
        data = await app.state.face.store.crop(int(found.group(1))) if found else None
        if data is None:
            return Response(status_code=404, headers={"Cache-Control": "no-store"})
        return Response(
            data,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store, private, max-age=0",
                "Pragma": "no-cache",
                "Content-Disposition": "inline",
                "Cross-Origin-Resource-Policy": "same-origin",
            },
        )

    @app.get("/faces/{face_id}/similar")
    async def similar(request: Request, u: User, face_id: str):
        """Nearest Immich people (named or not) for a logged face, from its stored embedding."""
        st: State = app.state.face
        fid = int_or_none(face_id)
        face = await st.store.face(fid) if fid else None
        if not face:
            return render(request, "similar.html", 404, face=None, unnamed=[], named=[], error="No such face.")
        people, error = [], ""
        if face["embedding"] is None:
            error = "This face has no stored embedding (crops are off, or the crop has expired)."
        elif not st.ready:
            error = "The Immich database is not reachable right now: " + st.reason
        else:
            try:
                people = await st.db.similar_people(
                    embedding_text(bytes(face["embedding"])), probes=settings.similar_probes
                )
            except Exception as exc:  # noqa: BLE001 - shown to the admin, not raised
                log.warning("similar search failed", extra={"fields": {"error": type(exc).__name__}})
                error = "The search in Immich failed or timed out (the database may be busy); try again later."
        face.pop("embedding", None)
        return render(
            request,
            "similar.html",
            face=face,
            unnamed=[x for x in people if not x["name"]],
            named=[x for x in people if x["name"]],
            error=error,
            immich=settings.immich_url,
            p=privacy(),
        )

    @app.post("/privacy/purge")
    async def purge(
        request: Request,
        u: User,
        csrf: Annotated[str, Form()],
        scope: Annotated[str, Form()] = "",
        confirm: Annotated[str, Form()] = "",
    ):
        try:
            check_csrf(request, csrf)
        except HttpError:
            return RedirectResponse("/requests?purged=expired", 303)
        if scope not in ("all", "crops") or confirm.strip() != "DELETE":
            return RedirectResponse("/requests?purged=unconfirmed", 303)
        await app.state.face.store.purge_all(scope, u["id"])
        return RedirectResponse(f"/requests?purged={scope}", 303)

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
            "verify_claims": settings.verify_max_claims,
            "verify_batch": settings.verify_batch_max,
            "changes": settings.enable_people and bool(settings.store_url) and settings.changes_interval > 0,
            "changes_minutes": max(1, settings.changes_interval // 60),
            "changes_days": settings.changes_retention_days,
            "subject": settings.subject_scoring,
            "subject_version": app.state.face.subject_model.version,
            "landmarks": settings.subject_landmarks,
        }

    @app.get("/docs")
    async def docs(request: Request, u: User):
        return render(request, "docs.html", base=docs_base(request), lim=limits(), p=privacy())

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
            result = await st.identify(raw, candidates=3, rid="admin-try", subject=settings.subject_scoring != "off")
            result.pop("_stats", None)
            return JSONResponse(result)
        except HttpError as exc:
            return JSONResponse({"error": {"code": exc.code, "message": exc.message}}, status_code=exc.status)

    # ------------------------------------------------------------------ label subjects (calibration)

    def sample_id(value: str) -> int | None:
        return int(value) if value.isdigit() and len(value) < 19 else None

    def can_see(u: dict, sample: dict) -> bool:
        return u.get("role", "admin") == "admin" or sample["created_by"] == u["id"]

    async def analyse_photo(raw: bytes, rid: str) -> tuple[dict, list[dict]]:
        """Detect + score one photo the way the API does; (image info, faces sorted left to right)."""
        st: State = app.state.face
        img, faces = await st._detect(raw, rid, landmarks=settings.subject_landmarks)
        faces = sorted(faces, key=lambda f: (f.box["x1"], f.box["y1"]))
        ps = await st.subjects(img, faces)
        out = [
            {
                "box": normalise_box(f.box, img.sent_width, img.sent_height),
                "prominence": fs.prominence,
                "role": fs.role,
                "signals": fs.signals,
            }
            for f, fs in zip(faces, ps.faces, strict=True)
        ]
        return {"w": img.sent_width, "h": img.sent_height, "photo_type": ps.photo_type}, out

    def model_info() -> dict:
        m = app.state.face.subject_model
        return {
            "version": m.version,
            "description": m.description,
            "scoring": settings.subject_scoring,
            "landmarks": settings.subject_landmarks,
            "file": bool(settings.subject_model_file or settings.subject_model_json),
        }

    @app.get("/subjects")
    async def subjects_page(request: Request, u: Curator, msg: str = ""):
        st: State = app.state.face
        admin = u.get("role", "admin") == "admin"
        return render(
            request,
            "subjects.html",
            stats=await lb.stats(st.store),
            recent=await lb.recent(st.store, None if admin else u["id"]),
            model=model_info(),
            immich=assets is not None,
            admin=admin,
            msg=msg,
            max_mb=settings.max_upload_bytes // (1024 * 1024),
        )

    @app.get("/subjects/next")
    async def subjects_next(request: Request, u: Curator, named: str = ""):
        """A new sample photo from Immich: a random photo with faces that nobody has labelled yet."""
        st: State = app.state.face
        if assets is None:
            return RedirectResponse("/subjects?msg=no-immich", 303)
        if not st.ready:
            return RedirectResponse("/subjects?msg=not-ready", 303)
        rng = secrets.SystemRandom()
        try:
            for _ in range(4):
                cands = await st.db.sample_assets(60, named=named == "1")
                seen = await lb.labelled_assets(st.store, [a for a, _ in cands])
                pool = [(a, n) for a, n in cands if a not in seen]
                while pool:
                    pick = rng.choices(pool, weights=[1.0 / max(1, n) for _, n in pool])[0]
                    pool.remove(pick)
                    got = await assets.preview(pick[0])
                    if got is None:
                        continue
                    info, faces = await analyse_photo(got[0], "admin-label")
                    del got
                    if not faces:
                        continue
                    sid = await lb.create_sample(
                        st.store,
                        source="immich",
                        asset_id=pick[0],
                        created_by=u["id"],
                        image_w=info["w"],
                        image_h=info["h"],
                        model_version=st.subject_model.version,
                        photo_type=info["photo_type"],
                        faces=faces,
                    )
                    return RedirectResponse(f"/subjects/{sid}?named={1 if named == '1' else 0}", 303)
        except (ImmichError, HttpError) as exc:
            log.warning("label sample failed", extra={"fields": {"error": str(exc)}})
            return RedirectResponse("/subjects?msg=immich-error", 303)
        return RedirectResponse("/subjects?msg=no-sample", 303)

    @app.post("/subjects/upload")
    async def subjects_upload(request: Request, u: Curator):
        """A photo from the curator's computer: analysed in memory, only the boxes and signals are stored."""
        try:
            check_csrf(request, request.headers.get("x-csrf-token"))
            st: State = app.state.face
            if not st.ready:
                raise HttpError(503, "not_ready", "the service is not ready: " + st.reason)
            body = await read_body(request, settings.max_upload_bytes)
            raw = extract_image(request.headers.get("content-type", ""), body)
            del body
            info, faces = await analyse_photo(raw, "admin-label")
            del raw
            if not faces:
                return JSONResponse({"id": None, "faces": []})
            sid = await lb.create_sample(
                st.store,
                source="upload",
                asset_id=None,
                created_by=u["id"],
                image_w=info["w"],
                image_h=info["h"],
                model_version=st.subject_model.version,
                photo_type=info["photo_type"],
                faces=faces,
            )
            return JSONResponse({"id": sid, "faces": [{"box": f["box"]} for f in faces]})
        except HttpError as exc:
            return JSONResponse({"error": {"code": exc.code, "message": exc.message}}, status_code=exc.status)

    @app.get("/subjects/{sid}")
    async def subject_sample(request: Request, u: Curator, sid: str, named: str = ""):
        n = sample_id(sid)
        sample = await lb.get_sample(app.state.face.store, n) if n else None
        if not sample or not can_see(u, sample):
            return render(request, "subject_label.html", 404, sample=None)
        fresh = time.time() - sample["created_at"].timestamp() < lb.PENDING_HOURS * 3600
        return render(
            request,
            "subject_label.html",
            sample=sample,
            labels=sample["labels"] or ["background"] * len(sample["faces"]),
            show_image=sample["source"] == "immich" and fresh and assets is not None,
            named=named == "1",
            admin=u.get("role", "admin") == "admin",
        )

    @app.get("/subjects/{sid}/image")
    async def subject_image(u: Curator, sid: str):
        """The Immich preview of a sample being labelled: signed-in, the sample's own curator (or an admin), within
        a day of drawing it; fetched from Immich on each request and never cached."""
        n = sample_id(sid)
        sample = await lb.get_sample(app.state.face.store, n) if n else None
        nothing = Response(status_code=404, headers={"Cache-Control": "no-store"})
        if not sample or not can_see(u, sample) or sample["source"] != "immich" or assets is None:
            return nothing
        if time.time() - sample["created_at"].timestamp() > lb.PENDING_HOURS * 3600:
            return nothing
        try:
            got = await assets.preview(sample["asset_id"])
        except ImmichError:
            return Response(status_code=502, headers={"Cache-Control": "no-store"})
        if got is None:
            return nothing
        data, ctype = got
        if ctype not in ("image/jpeg", "image/webp", "image/png"):
            return nothing
        return Response(
            data,
            media_type=ctype,
            headers={
                "Cache-Control": "no-store, private, max-age=0",
                "Pragma": "no-cache",
                "Content-Disposition": "inline",
                "Cross-Origin-Resource-Policy": "same-origin",
            },
        )

    @app.post("/subjects/{sid}")
    async def subject_save(request: Request, u: Curator, sid: str):
        form = await request.form()
        try:
            check_csrf(request, str(form.get("csrf") or ""))
        except HttpError:
            return RedirectResponse("/subjects?msg=expired", 303)
        n = sample_id(sid)
        st: State = app.state.face
        sample = await lb.get_sample(st.store, n) if n else None
        if not sample or not can_see(u, sample):
            return RedirectResponse("/subjects?msg=missing", 303)
        action = str(form.get("action") or "save")
        if action == "skip":
            await lb.save_labels(st.store, n, None, u["id"], skipped=True)
        else:
            labels = [str(form.get(f"role_{i}") or "background") for i in range(len(sample["faces"]))]
            if any(x not in ("main", "co_subject", "background") for x in labels) or labels.count("main") > 1:
                return RedirectResponse(f"/subjects/{n}?msg=invalid", 303)
            await lb.save_labels(st.store, n, labels, u["id"])
        if action in ("skip", "next") and sample["source"] == "immich":
            return RedirectResponse(f"/subjects/next?named={1 if form.get('named') == '1' else 0}", 303)
        return RedirectResponse("/subjects?msg=saved", 303)

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
