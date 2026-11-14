"""Settings, all from environment variables (prefix FACE_API_). See README "Configuration"."""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

PREFIX = "FACE_API_"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ConfigError(ValueError):
    pass


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(PREFIX + name)
    if value is None or value == "":
        return default
    return value


def _secret(name: str) -> str | None:
    """A value from FACE_API_<NAME>_FILE (preferred, e.g. a mounted Secret) or FACE_API_<NAME>."""
    path = _env(name + "_FILE")
    if path:
        return Path(path).read_text(encoding="utf-8").strip()
    return _env(name)


def _bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    if value.lower() in ("1", "true", "yes", "on"):
        return True
    if value.lower() in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{PREFIX}{name}: expected true/false, got {value!r}")


def _int(name: str, default: int, lo: int, hi: int) -> int:
    raw = _env(name)
    try:
        value = int(raw) if raw is not None else default
    except ValueError as exc:
        raise ConfigError(f"{PREFIX}{name}: not an integer") from exc
    if not lo <= value <= hi:
        raise ConfigError(f"{PREFIX}{name}: must be between {lo} and {hi}")
    return value


def _float(name: str, default: float, lo: float, hi: float) -> float:
    raw = _env(name)
    try:
        value = float(raw) if raw is not None else default
    except ValueError as exc:
        raise ConfigError(f"{PREFIX}{name}: not a number") from exc
    if not lo <= value <= hi:
        raise ConfigError(f"{PREFIX}{name}: must be between {lo} and {hi}")
    return value


def _choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = (_env(name, default) or default).strip().lower()
    if value not in allowed:
        raise ConfigError(f"{PREFIX}{name}: expected one of {', '.join(allowed)}")
    return value


def parse_api_keys(text: str) -> dict[str, bytes]:
    """{"client-name": "<sha256 hex of the key>", ...} -> {name: digest bytes}. Plain keys are refused."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError("API keys: not valid JSON") from exc
    if not isinstance(data, dict) or not data:
        raise ConfigError("API keys: expected a non-empty JSON object {client: sha256-hex}")
    keys: dict[str, bytes] = {}
    for name, digest in data.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", name):
            raise ConfigError("API keys: client names must match [A-Za-z0-9_.-]{1,64}")
        if not isinstance(digest, str) or not _HEX64.fullmatch(digest.lower()):
            raise ConfigError(f"API keys: client {name!r} must map to a SHA-256 hex digest, not a plain key")
        keys[name] = bytes.fromhex(digest.lower())
    return keys


@dataclass(frozen=True)
class Settings:
    # Immich machine-learning server and the face model Immich itself uses (must equal the Immich config)
    ml_url: str
    face_model: str = "buffalo_l"
    min_score: float = 0.7
    max_distance: float = 0.5
    # longest edge sent to ML; Immich detects faces on its preview (default 1440)
    max_edge: int = 1440

    # Immich database (read-only role)
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "immich"
    db_user: str = "immich_face_api"
    db_password: str = field(default="", repr=False)
    db_sslmode: str = "verify-full"
    db_sslrootcert: str | None = None
    db_pool_size: int = 4

    # whose people count: the library owner (uuid) or, if no id is given, the owner's e-mail
    owner_id: str | None = None
    owner_email: str | None = None
    expected_immich_version: str = "3.4.1"

    # matching
    knn: int = 20
    face_cache: bool = True
    face_cache_per_person: int = 100
    face_cache_refresh_seconds: int = 300
    max_candidates: int = 5
    ambiguity_margin: float = 0.03
    # same-name candidates whose centroids are this close are one (split) person, not an ambiguity; 0 = off
    same_name_max_distance: float = 0.6

    # input limits
    max_upload_bytes: int = 15 * 1024 * 1024
    max_pixels: int = 50_000_000

    # protection
    api_keys: dict[str, bytes] = field(default_factory=dict, repr=False)
    rate_limit_per_minute: int = 30
    rate_limit_burst: int = 10
    max_concurrent_ml: int = 2
    ml_queue_timeout: float = 20.0
    ml_timeout: float = 60.0

    enable_people: bool = True
    enable_openapi: bool = True
    test_mode: bool = False
    metrics_port: int = 9090
    readiness_interval: int = 60
    sentry_dsn: str | None = field(default=None, repr=False)
    environment: str = "production"

    # optional store (own PostgreSQL, never Immich's): API keys made in the admin UI + hourly usage aggregates
    store_url: str | None = field(default=None, repr=False)
    store_password: str | None = field(default=None, repr=False)
    store_pool_size: int = 4
    usage_retention_days: int = 400
    key_cache_seconds: int = 30

    # optional admin UI (python -m immich_face_api admin), its own port
    admin_port: int = 8081
    public_url: str = ""
    session_secret: str = field(default="", repr=False)
    session_https_only: bool = True
    oidc_issuer: str = ""
    oidc_client_id: str = ""
    oidc_client_secret: str = field(default="", repr=False)
    oidc_scopes: str = "openid email profile"
    oidc_admin_claim: str = ""
    oidc_admin_values: tuple[str, ...] = ()
    admin_token: str = field(default="", repr=False)
    docs_base_url: str = ""  # API base URL shown in the admin docs; default: the request's own host

    # request log in the store (needs FACE_API_STORE_URL): per identify request what was asked and what was found
    # (app, image size, faces, matched names, distances, status), no images. See README "Request log and face crops".
    request_log: bool = True
    request_log_retention_days: int = 90
    # face crops for curators: "none" (default), "unknown" (unknown, ambiguous and near-threshold faces) or "all".
    # Crops are biometric data; the embedding is kept with the crop (same retention) for "find similar in Immich".
    store_crops: str = "none"
    crop_retention_days: int = 30
    crop_size: int = 160
    log_max_faces: int = 50  # faces logged (and cropped) per request at most
    near_margin: float = 0.05  # a match within this of max_distance counts as "near the threshold"
    # people change log (needs the store): GET /v1/people/changes, /v1/people/resolve. 0 = off
    changes_interval: int = 300  # seconds between checks
    changes_overlap: int = 600  # re-read faces changed this long before the last check (late commits)
    changes_full_hours: int = 24  # full reconcile this often
    changes_retention_days: int = 365
    # POST /v1/verify
    verify_max_claims: int = 10
    verify_batch_max: int = 10

    immich_url: str = ""  # Immich web URL for person links in the admin UI, e.g. https://photos.example.org
    similar_probes: int = 32  # vchordrq.probes for "find similar in Immich" (0: the database default)

    @property
    def oidc_enabled(self) -> bool:
        return bool(self.oidc_issuer and self.oidc_client_id and self.oidc_client_secret)

    @classmethod
    def from_env(cls) -> Settings:
        ml_url = _env("ML_URL")
        if not ml_url:
            raise ConfigError(f"{PREFIX}ML_URL is required")
        owner_id = _env("OWNER_ID")
        owner_email = _env("OWNER_EMAIL")
        if not owner_id and not owner_email:
            raise ConfigError(f"{PREFIX}OWNER_ID or {PREFIX}OWNER_EMAIL is required")
        if owner_id:
            try:
                owner_id = str(uuid.UUID(owner_id))
            except ValueError as exc:
                raise ConfigError(f"{PREFIX}OWNER_ID must be a UUID") from exc
        keys_text = _secret("API_KEYS")
        store_url = _secret("STORE_URL")
        if not keys_text and not store_url:
            raise ConfigError(f"{PREFIX}API_KEYS_FILE (or {PREFIX}API_KEYS) or {PREFIX}STORE_URL is required")
        issuer = _env("OIDC_ISSUER", "")
        if not issuer and _env("OIDC_TENANT_ID"):  # Microsoft Entra shortcut
            issuer = f"https://login.microsoftonline.com/{_env('OIDC_TENANT_ID')}/v2.0"
        admin_values = tuple(v.strip() for v in (_env("OIDC_ADMIN_VALUES", "") or "").split(",") if v.strip())
        sslmode = _env("DB_SSLMODE", "verify-full")
        if sslmode not in ("disable", "require", "verify-ca", "verify-full"):
            raise ConfigError(f"{PREFIX}DB_SSLMODE: unsupported value")
        return cls(
            ml_url=ml_url.rstrip("/"),
            face_model=_env("FACE_MODEL", "buffalo_l"),
            min_score=_float("MIN_SCORE", 0.7, 0.1, 1.0),
            max_distance=_float("MAX_DISTANCE", 0.5, 0.1, 2.0),
            max_edge=_int("MAX_EDGE", 1440, 256, 8192),
            db_host=_env("DB_HOST", "localhost"),
            db_port=_int("DB_PORT", 5432, 1, 65535),
            db_name=_env("DB_NAME", "immich"),
            db_user=_env("DB_USER", "immich_face_api"),
            db_password=_secret("DB_PASSWORD") or "",
            db_sslmode=sslmode,
            db_sslrootcert=_env("DB_SSLROOTCERT"),
            db_pool_size=_int("DB_POOL_SIZE", 4, 1, 32),
            owner_id=owner_id,
            owner_email=owner_email,
            expected_immich_version=_env("EXPECTED_IMMICH_VERSION", "3.4.1").lstrip("v"),
            knn=_int("KNN", 20, 1, 200),
            face_cache=_bool("FACE_CACHE", True),
            face_cache_per_person=_int("FACE_CACHE_PER_PERSON", 100, 1, 100_000),
            face_cache_refresh_seconds=_int("FACE_CACHE_REFRESH_SECONDS", 300, 10, 86_400),
            max_candidates=_int("MAX_CANDIDATES", 5, 1, 20),
            ambiguity_margin=_float("AMBIGUITY_MARGIN", 0.03, 0.0, 0.5),
            same_name_max_distance=_float("SAME_NAME_MAX_DISTANCE", 0.6, 0.0, 2.0),
            max_upload_bytes=_int("MAX_UPLOAD_BYTES", 15 * 1024 * 1024, 1024, 100 * 1024 * 1024),
            max_pixels=_int("MAX_PIXELS", 50_000_000, 10_000, 200_000_000),
            api_keys=parse_api_keys(keys_text) if keys_text else {},
            rate_limit_per_minute=_int("RATE_LIMIT_PER_MINUTE", 30, 1, 100_000),
            rate_limit_burst=_int("RATE_LIMIT_BURST", 10, 1, 100_000),
            max_concurrent_ml=_int("MAX_CONCURRENT_ML", 2, 1, 64),
            ml_queue_timeout=_float("ML_QUEUE_TIMEOUT", 20.0, 0.1, 600.0),
            ml_timeout=_float("ML_TIMEOUT", 60.0, 1.0, 600.0),
            enable_people=_bool("ENABLE_PEOPLE", True),
            enable_openapi=_bool("ENABLE_OPENAPI", True),
            test_mode=_bool("TEST_MODE", False),
            metrics_port=_int("METRICS_PORT", 9090, 0, 65535),
            readiness_interval=_int("READINESS_INTERVAL", 60, 5, 3600),
            sentry_dsn=_secret("SENTRY_DSN"),
            environment=_env("ENVIRONMENT", "production"),
            store_url=store_url,
            store_password=_secret("STORE_PASSWORD"),
            store_pool_size=_int("STORE_POOL_SIZE", 4, 1, 32),
            usage_retention_days=_int("USAGE_RETENTION_DAYS", 400, 1, 3650),
            key_cache_seconds=_int("KEY_CACHE_SECONDS", 30, 0, 600),
            admin_port=_int("ADMIN_PORT", 8081, 1, 65535),
            public_url=(_env("PUBLIC_URL", "") or "").rstrip("/"),
            session_secret=_secret("SESSION_SECRET") or "",
            session_https_only=_bool("SESSION_HTTPS_ONLY", True),
            oidc_issuer=issuer.rstrip("/"),
            oidc_client_id=_secret("OIDC_CLIENT_ID") or "",
            oidc_client_secret=_secret("OIDC_CLIENT_SECRET") or "",
            oidc_scopes=_env("OIDC_SCOPES", "openid email profile"),
            oidc_admin_claim=_env("OIDC_ADMIN_CLAIM", "") or "",
            oidc_admin_values=admin_values,
            admin_token=_secret("ADMIN_TOKEN") or "",
            docs_base_url=(_env("DOCS_BASE_URL", "") or "").rstrip("/"),
            request_log=_bool("REQUEST_LOG", True),
            request_log_retention_days=_int("REQUEST_LOG_RETENTION_DAYS", 90, 1, 3650),
            store_crops=_choice("STORE_CROPS", "none", ("none", "unknown", "all")),
            crop_retention_days=_int("CROP_RETENTION_DAYS", 30, 1, 3650),
            crop_size=_int("CROP_SIZE", 160, 64, 512),
            log_max_faces=_int("LOG_MAX_FACES", 50, 1, 500),
            near_margin=_float("NEAR_MARGIN", 0.05, 0.0, 1.0),
            changes_interval=_int("CHANGES_INTERVAL", 300, 0, 86_400),
            changes_overlap=_int("CHANGES_OVERLAP", 600, 0, 86_400),
            changes_full_hours=_int("CHANGES_FULL_HOURS", 24, 1, 24 * 30),
            changes_retention_days=_int("CHANGES_RETENTION_DAYS", 365, 7, 3650),
            verify_max_claims=_int("VERIFY_MAX_CLAIMS", 10, 1, 100),
            verify_batch_max=_int("VERIFY_BATCH_MAX", 10, 1, 100),
            immich_url=(_env("IMMICH_URL", "") or "").rstrip("/"),
            similar_probes=_int("SIMILAR_PROBES", 32, 0, 100_000),
        )
