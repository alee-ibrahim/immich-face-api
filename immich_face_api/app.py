"""HTTP API: POST /v1/identify, GET /v1/people, GET /healthz, GET /readyz."""

from __future__ import annotations

import asyncio
import contextlib
import email.parser
import email.policy
import json
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import psycopg
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import start_http_server
from psycopg_pool import PoolTimeout

from . import __version__
from . import metrics as m
from .config import Settings
from .db import Database, SchemaMismatch
from .facecache import FaceCache, same_person_rule
from .imaging import ImageRejected, prepare
from .matching import decide, normalise_box
from .ml import MLClient, MLError
from .openapi import OPENAPI_SPEC
from .requestlog import FaceRecord, RequestLogWriter, RequestRecord, embedding_bytes, kind_of, make_crop, wants_crop
from .scrub import scrub_breadcrumb, scrub_event
from .security import ApiKeys, RateLimiter, token_from_headers
from .store import Store, split_key

log = logging.getLogger("immich_face_api")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}) or {})
        if record.exc_info and record.levelno >= logging.ERROR:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
    for noisy in ("httpx", "httpcore", "psycopg.pool", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def setup_sentry(settings: Settings) -> None:
    if not settings.sentry_dsn:
        return
    import sentry_sdk

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.environment,
        release=f"immich-face-api@{__version__}",
        send_default_pii=False,
        include_local_variables=False,
        max_request_body_size="never",
        traces_sample_rate=0.0,
        before_send=scrub_event,
        before_breadcrumb=scrub_breadcrumb,
    )
    sentry_sdk.set_tag("component", "immich-face-api")


class HttpError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict | None = None):
        self.status, self.code, self.message, self.headers = status, code, message, headers or {}


def error_response(exc: HttpError) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": exc.code, "message": exc.message}}, status_code=exc.status, headers=exc.headers
    )


async def read_body(request: Request, limit: int) -> bytes:
    """Whole body into memory, refusing more than ``limit`` bytes (+ multipart overhead) without reading it all."""
    hard = limit + 64 * 1024
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > hard:
        raise HttpError(413, "too_large", "request body too large")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > hard:
            raise HttpError(413, "too_large", "request body too large")
        chunks.append(chunk)
    return b"".join(chunks)


def extract_image(content_type: str, body: bytes) -> bytes:
    """The image bytes from a multipart/form-data body (field "image") or a raw body."""
    ctype = content_type.split(";", 1)[0].strip().lower()
    if ctype != "multipart/form-data":
        return body
    if "boundary=" not in content_type:
        raise HttpError(400, "bad_request", "multipart body without boundary")
    head = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("latin-1")
    msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(head + body)
    if not msg.is_multipart():
        raise HttpError(400, "bad_request", "malformed multipart body")
    for part in msg.iter_parts():
        if part.get_param("name", header="content-disposition") == "image":
            data = part.get_payload(decode=True)
            if isinstance(data, bytes):
                return data
    raise HttpError(400, "bad_request", 'multipart field "image" is missing')


def db_failure(exc: Exception, rid: str, message: str) -> HttpError:
    """A database problem as an HTTP error: timeouts and a busy pool are temporary (503 + Retry-After, logged as a
    warning); anything else is unexpected (502, reported to Sentry/GlitchTip, scrubbed)."""
    m.DB_ERRORS.inc()
    if isinstance(exc, psycopg.errors.QueryCanceled | psycopg.errors.LockNotAvailable | PoolTimeout):
        log.warning("db busy", extra={"fields": {"rid": rid, "error": type(exc).__name__}})
        return HttpError(
            503, "busy", "the photo archive database is busy, retry in a few seconds", {"Retry-After": "10"}
        )
    log.error("db error", exc_info=True, extra={"fields": {"rid": rid, "error": type(exc).__name__}})
    return HttpError(502, "upstream", message)


class State:
    def __init__(self, settings: Settings):
        self.s = settings
        self.keys = ApiKeys(settings.api_keys)
        self.limiter = RateLimiter(settings.rate_limit_per_minute, settings.rate_limit_burst)
        self.db = Database(settings)
        self.ml = MLClient(settings.ml_url, settings.face_model, settings.min_score, settings.ml_timeout)
        self.ml_slots = asyncio.Semaphore(settings.max_concurrent_ml)
        self.faces = (
            FaceCache(self.db, settings.face_cache_per_person, settings.face_cache_refresh_seconds, settings.knn)
            if settings.face_cache
            else None
        )
        self.store = (
            Store(settings.store_url, settings.store_password, settings.store_pool_size, settings.key_cache_seconds)
            if settings.store_url
            else None
        )
        # request log: only for the API process (the admin's "Try it" is never logged)
        self.writer: RequestLogWriter | None = None
        self.ready = False
        self.reason = "starting"

    async def refresh_readiness(self) -> None:
        try:
            status = await self.db.check()
            if not await self.ml.ping():
                raise RuntimeError("machine-learning server not reachable")
            if self.store and not await self.store.ping():
                # not a readiness failure: env keys keep working and usage stays buffered in memory
                log.warning("store database not reachable (UI-made keys cannot be checked)")
        except SchemaMismatch as exc:
            self._set(False, str(exc), error=True)
        except Exception as exc:  # noqa: BLE001
            self._set(False, f"dependency check failed: {type(exc).__name__}: {exc}", error=False)
        else:
            self._set(True, f"ok (Immich {status.immich_version})")

    def _set(self, ready: bool, reason: str, error: bool = False) -> None:
        if ready != self.ready or reason != self.reason:
            (log.error if error else log.warning if not ready else log.info)(
                "readiness changed", extra={"fields": {"ready": ready, "reason": reason}}
            )
        self.ready, self.reason = ready, reason
        m.READY.set(1 if ready else 0)

    async def identify(
        self,
        raw: bytes,
        *,
        candidates: int = 0,
        include_unnamed: bool = False,
        rid: str = "",
        log_faces: bool = False,
    ) -> dict:
        """The whole pipeline for one image (validate, detect, search, decide). Raises HttpError.

        log_faces: also return "_log" (per-face records for the request log, with crops per FACE_API_STORE_CROPS)."""
        s = self.s
        state = self
        t0 = time.perf_counter()
        try:
            img = await asyncio.to_thread(
                prepare, raw, max_bytes=s.max_upload_bytes, max_pixels=s.max_pixels, max_edge=s.max_edge
            )
        except ImageRejected as exc:
            m.REJECTED.labels("image").inc()
            raise HttpError(exc.status, "invalid_image", str(exc)) from exc
        finally:
            del raw
        m.STAGE.labels("decode").observe(time.perf_counter() - t0)

        try:
            await asyncio.wait_for(state.ml_slots.acquire(), timeout=s.ml_queue_timeout)
        except TimeoutError as exc:
            m.REJECTED.labels("busy").inc()
            raise HttpError(503, "busy", "too many images in progress, retry later", {"Retry-After": "10"}) from exc
        t1 = time.perf_counter()
        m.INFLIGHT.inc()
        try:
            faces = await state.ml.detect(img.encoded)
        except MLError as exc:
            m.ML_ERRORS.inc()
            log.warning("ml error", extra={"fields": {"rid": rid, "error": str(exc)}})
            raise HttpError(502, "upstream", "face detection failed") from exc
        finally:
            m.INFLIGHT.dec()
            state.ml_slots.release()
        m.STAGE.labels("ml").observe(time.perf_counter() - t1)

        t2 = time.perf_counter()
        embeddings = [f.embedding for f in faces]
        try:
            same_person = None
            if state.faces is not None and state.faces.ready and not include_unnamed:
                neighbours = await state.faces.nearest(embeddings)  # in memory, milliseconds
                same_person = same_person_rule(state.faces.snap, s.same_name_max_distance)
            else:  # cache not loaded yet, or test mode: exact search in the database
                neighbours = await state.db.nearest(embeddings, include_unnamed=include_unnamed)
        except Exception as exc:  # noqa: BLE001
            raise db_failure(exc, rid, "face search failed") from exc
        m.STAGE.labels("db").observe(time.perf_counter() - t2)

        out_faces, matched, ambiguous, logged = [], 0, 0, []
        for face, rows in zip(faces, neighbours, strict=True):
            d = decide(
                rows,
                max_distance=s.max_distance,
                max_candidates=s.max_candidates,
                ambiguity_margin=s.ambiguity_margin,
                same_person=same_person,
            )
            best = d.best
            entry = {
                "box": normalise_box(face.box, img.sent_width, img.sent_height),
                "detection_score": round(face.score, 4),
                "person_id": best.person_id if best else None,
                "name": best.name if best else None,
                "confidence": best.confidence if best else None,
                "distance": best.distance if best else None,
                "ambiguous": d.ambiguous,
            }
            if candidates:
                entry["candidates"] = [
                    {
                        "person_id": c.person_id,
                        "name": c.name,
                        "confidence": c.confidence,
                        "distance": c.distance,
                        "votes": c.votes,
                    }
                    for c in d.candidates[:candidates]
                ]
            if best and best.name:
                matched += 1
                ambiguous += int(d.ambiguous)
            out_faces.append(entry)
            if log_faces:
                logged.append((entry, face, rows))
        out_faces.sort(key=lambda f: (f["box"]["x"], f["box"]["y"]))
        log_records = await self._face_records(logged, img) if log_faces else None
        width, height, source_format = img.width, img.height, img.source_format
        del img

        m.FACES_PER_REQUEST.observe(len(faces))
        m.FACES.inc(len(faces))
        m.FACES_MATCHED.inc(matched)
        m.FACES_AMBIGUOUS.inc(ambiguous)
        return {
            "faces": out_faces,
            "image": {"w": width, "h": height},
            "model": s.face_model,
            "max_distance": s.max_distance,
            "immich_version": state.db.status.immich_version,
            **({"test_mode": True} if include_unnamed else {}),
            "_stats": {"faces": len(faces), "matched": matched, "ambiguous": ambiguous, "format": source_format},
            **({"_log": log_records} if log_records is not None else {}),
        }

    async def _face_records(self, logged: list, img) -> list[FaceRecord]:
        """Request-log records for up to log_max_faces faces (in response order), with crops if configured."""
        s = self.s
        logged = sorted(logged, key=lambda t: (t[0]["box"]["x"], t[0]["box"]["y"]))[: s.log_max_faces]
        records = []
        for i, (entry, _face, rows) in enumerate(logged):
            near = decide(rows, max_distance=float("inf"), max_candidates=3, ambiguity_margin=0.0)
            records.append(
                FaceRecord(
                    idx=i,
                    kind=kind_of(entry, s.max_distance, s.near_margin),
                    box=entry["box"],
                    detection_score=entry["detection_score"],
                    person_id=entry["person_id"],
                    name=entry["name"],
                    distance=entry["distance"],
                    candidates=[
                        {"person_id": c.person_id, "name": c.name, "distance": c.distance, "votes": c.votes}
                        for c in near.candidates
                    ],
                )
            )
        todo = [(r, face) for r, (_, face, _) in zip(records, logged, strict=True) if wants_crop(s.store_crops, r.kind)]
        if todo and img.image is not None:

            def cut() -> None:
                for r, face in todo:
                    r.crop = make_crop(img.image, face.box, s.crop_size)
                    r.embedding = embedding_bytes(face.embedding)

            await asyncio.to_thread(cut)
        return records

    async def readiness_loop(self) -> None:
        while True:
            await self.refresh_readiness()
            await asyncio.sleep(self.s.readiness_interval if self.ready else 10)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state: State = app.state.face
        await state.db.open()
        tasks = [asyncio.create_task(state.readiness_loop())]
        if state.faces is not None:
            tasks.append(asyncio.create_task(state.faces.loop()))
        if state.store:
            await state.store.open(migrate=True, wait=False)
            tasks.append(
                asyncio.create_task(
                    state.store.flush_loop(
                        settings.usage_retention_days,
                        request_days=settings.request_log_retention_days,
                        crop_days=settings.crop_retention_days,
                    )
                )
            )
            if settings.request_log:
                state.writer = RequestLogWriter(state.store)
                tasks.append(asyncio.create_task(state.writer.loop()))
        log.info(
            "started",
            extra={
                "fields": {
                    "version": __version__,
                    "model": settings.face_model,
                    "min_score": settings.min_score,
                    "max_distance": settings.max_distance,
                    "pinned_immich": settings.expected_immich_version,
                    "clients": sorted(settings.api_keys),
                    "store": bool(settings.store_url),
                    "request_log": bool(settings.store_url and settings.request_log),
                    "store_crops": settings.store_crops if settings.store_url and settings.request_log else "none",
                    "test_mode": settings.test_mode,
                }
            },
        )
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            if state.store:
                with contextlib.suppress(Exception):
                    await state.store.flush()
                if state.writer:
                    with contextlib.suppress(Exception):
                        await state.writer.drain()
                await state.store.close()
            await state.ml.close()
            await state.db.close()

    app = FastAPI(
        title="immich-face-api",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.face = State(settings)

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        started = time.perf_counter()
        request.state.rid = request.headers.get("x-request-id", "")[:64] or uuid.uuid4().hex
        request.state.client = "-"
        request.state.fields = {}
        request.state.log = None
        try:
            response = await call_next(request)
        except Exception:  # pragma: no cover - FastAPI turns most errors into responses already
            log.exception("unhandled error", extra={"fields": {"rid": request.state.rid}})
            response = JSONResponse({"error": {"code": "internal", "message": "internal error"}}, status_code=500)
        elapsed = time.perf_counter() - started
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "other")
        m.REQUESTS.labels(endpoint, str(response.status_code), request.state.client).inc()
        store = request.app.state.face.store
        if store and request.state.client != "-" and endpoint.startswith("/v1/"):
            f = request.state.fields
            store.buffer.record(
                request.state.client,
                endpoint,
                response.status_code,
                elapsed * 1000,
                faces=f.get("faces", 0),
                matched=f.get("matched", 0),
                ambiguous=f.get("ambiguous", 0),
            )
        writer = request.app.state.face.writer
        if writer and request.state.client != "-" and endpoint == "/v1/identify":
            writer.submit(request_record(request, endpoint, response.status_code, elapsed))
        m.LATENCY.labels(endpoint).observe(elapsed)
        response.headers["X-Request-Id"] = request.state.rid
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        if endpoint not in ("/healthz", "/readyz"):
            log.info(
                "request",
                extra={
                    "fields": {
                        "rid": request.state.rid,
                        "method": request.method,
                        "path": endpoint,
                        "status": response.status_code,
                        "ms": round(elapsed * 1000, 1),
                        "client": request.state.client,
                        "ip": request.headers.get("cf-connecting-ip") or "-",
                        **request.state.fields,
                    }
                },
            )
        return response

    async def authenticate(request: Request) -> str:
        """API key first (env keys, then store keys), then the rate limit, then readiness. Runs before the body is
        read and before any ML or Immich call."""
        state: State = request.app.state.face
        header, x_key = request.headers.get("authorization"), request.headers.get("x-api-key")
        client = state.keys.client_for(header, x_key)
        per_minute = None
        token = token_from_headers(header, x_key)
        if client is None and state.store and token:
            prefix = split_key(token)
            if prefix:
                try:
                    key = await state.store.lookup(prefix)
                except Exception:  # noqa: BLE001 - store down: env keys still work, store keys cannot be checked
                    log.warning("store lookup failed")
                    raise HttpError(503, "not_ready", "service not ready", {"Retry-After": "30"}) from None
                if key and key.valid(token, datetime.now(UTC)):
                    client, per_minute = key.app, key.rate_limit_per_minute
                    state.store.mark_used(key.id)
        if client is None:
            m.REJECTED.labels("auth").inc()
            raise HttpError(401, "unauthorized", "missing or invalid API key", {"WWW-Authenticate": "Bearer"})
        request.state.client = client
        allowed, retry_after = state.limiter.take(client, per_minute)
        if not allowed:
            m.REJECTED.labels("rate_limit").inc()
            raise HttpError(429, "rate_limited", "too many requests", {"Retry-After": str(retry_after)})
        if not state.ready:
            raise HttpError(503, "not_ready", "service not ready", {"Retry-After": "30"})
        return client

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/openapi.json")
    async def openapi_spec(request: Request):
        if not request.app.state.face.s.enable_openapi:
            return JSONResponse({"error": {"code": "not_found", "message": "not found"}}, status_code=404)
        return JSONResponse(OPENAPI_SPEC)

    @app.get("/readyz")
    async def readyz(request: Request):
        state: State = request.app.state.face
        if state.ready:
            return {"status": "ready"}
        return JSONResponse({"status": "not ready"}, status_code=503)

    @app.post("/v1/identify")
    async def identify(request: Request):
        try:
            await authenticate(request)
            state: State = request.app.state.face
            s = state.s
            candidates = _int_param(request, "candidates", 0, 0, s.max_candidates)
            include_unnamed = request.query_params.get("include_unnamed") in ("1", "true")
            if include_unnamed and not s.test_mode:
                raise HttpError(400, "bad_request", "include_unnamed is only available in test mode")

            body = await read_body(request, s.max_upload_bytes)
            raw = extract_image(request.headers.get("content-type", ""), body)
            del body
            request.state.log = {"image_bytes": len(raw)}
            result = await state.identify(
                raw,
                candidates=candidates,
                include_unnamed=include_unnamed,
                rid=request.state.rid,
                log_faces=state.writer is not None,
            )
            del raw
            stats = result.pop("_stats")
            request.state.fields = stats
            request.state.log = {
                **(request.state.log or {}),
                "image": result["image"],
                "format": stats["format"],
                "faces": result.pop("_log", None) or [],
            }
            return result
        except HttpError as exc:
            request.state.log = {**(request.state.log or {}), "error_code": exc.code}
            return error_response(exc)

    @app.get("/v1/people")
    async def people(request: Request):
        try:
            await authenticate(request)
            state: State = request.app.state.face
            if not state.s.enable_people:
                raise HttpError(404, "not_found", "not found")
            try:
                rows = await state.db.people()
            except Exception as exc:  # noqa: BLE001
                raise db_failure(exc, request.state.rid, "people lookup failed") from exc
            request.state.fields = {"people": len(rows)}
            return {"people": rows, "count": len(rows)}
        except HttpError as exc:
            return error_response(exc)

    return app


def request_record(request: Request, endpoint: str, status: int, elapsed: float) -> RequestRecord:
    """The request-log entry of one identify call, from what the handler left in request.state."""
    info = request.state.log or {}
    faces = info.get("faces") or []
    stats = request.state.fields or {}
    image = info.get("image") or {}
    kinds = [f.kind for f in faces]
    total = int(stats.get("faces", 0))
    matched = int(stats.get("matched", 0))
    return RequestRecord(
        at=time.time() - elapsed,
        rid=request.state.rid,
        client=request.state.client,
        endpoint=endpoint,
        status=status,
        error_code=info.get("error_code"),
        latency_ms=int(elapsed * 1000),
        image_bytes=info.get("image_bytes"),
        image_w=image.get("w"),
        image_h=image.get("h"),
        image_format=info.get("format"),
        faces=total,
        matched=matched,
        unknown=total - matched,
        ambiguous=int(stats.get("ambiguous", 0)),
        near=kinds.count("near"),
        face_records=faces,
    )


def _int_param(request: Request, name: str, default: int, lo: int, hi: int) -> int:
    raw = request.query_params.get(name)
    if raw is None:
        return default
    if not raw.isdigit() or not lo <= int(raw) <= hi:
        raise HttpError(400, "bad_request", f"{name} must be an integer between {lo} and {hi}")
    return int(raw)


def main() -> None:  # pragma: no cover - process entry point
    import uvicorn

    setup_logging()
    settings = Settings.from_env()
    setup_sentry(settings)
    if settings.metrics_port:
        start_http_server(settings.metrics_port)
    uvicorn.run(
        create_app(settings),
        host="0.0.0.0",  # noqa: S104 - in a container
        port=8080,
        access_log=False,
        server_header=False,
        proxy_headers=False,
        log_config=None,
        timeout_keep_alive=30,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
