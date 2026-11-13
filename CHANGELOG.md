# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- The image sent to the ML server is now lossless PNG instead of a JPEG at quality 92. Re-encoding moved the
  detector's score by up to about 0.006, which dropped faces that sit right at `minScore` (a real frame scored 0.7007
  from the original and 0.6994 after the re-encode, so it was missed at 0.7). Images are still never upscaled.

### Added
- Optional admin UI (`python -m immich_face_api admin`, port 8081): dashboard and per-app usage (requests, faces,
  match and unknown rate, latency p50/p95, errors over time), apps and API keys (shown once, owner, label, rate limit
  per key, expiry, revoke, last used, audit list), and a "Try it" page. OIDC sign-in with an optional admin claim
  filter, or a bootstrap token locally. Server-rendered, strict CSP, CSRF tokens.
- Admin "Docs" section: quick start, authentication, endpoints with curl/PHP/Python/Node examples (tabs, copy),
  limits, errors, best practices, an API reference rendered from the bundled OpenAPI document, `openapi.json` download
  (`FACE_API_DOCS_BASE_URL`).
- `X-API-Key` header accepted as an alternative to `Authorization: Bearer`.
- Optional own PostgreSQL store (`FACE_API_STORE_URL`): UI-made keys (`ifa_<prefix>_<secret>`, SHA-256) accepted by
  the API next to the environment keys, and hourly usage aggregates (counts only), with retention.
- Tests proving the API key is checked before the body is read and before any ML/DB call.

### Changed
- Docs: the API key is the authentication; a proxy allowlist or access token is optional hardening.

## [0.1.0] - not yet tagged

First version, tested against **Immich 3.4.1**.

### Added
- `POST /v1/identify`: faces of a photo matched to the named, not hidden people of one Immich user, with normalised
  boxes, distance, confidence, an `ambiguous` flag and optional ranked candidates.
- `GET /v1/people`, `GET /healthz`, `GET /readyz` (Immich version, table shapes, privileges, owner, ML reachability),
  `GET /openapi.json` (static OpenAPI 3.1 document, `FACE_API_ENABLE_OPENAPI`).
- Same face model and `minScore` request as Immich; exact cosine search over the owner's named faces; "nearest face
  wins" decision.
- Hashed per-client API keys (constant-time compare), per-client rate limit, bounded ML concurrency, size and pixel
  limits, no CORS.
- In-memory image handling: magic-byte check, JPEG (including camera MPO files), PNG, WebP, HEIC/HEIF, EXIF
  orientation, metadata stripped, downscaled to Immich's preview size.
- Prometheus metrics on a separate port; optional Sentry/GlitchTip error reports with a scrubber (no bodies, headers,
  users, local variables; tokens, e-mails, vectors and long secrets redacted).
- Test mode (`FACE_API_TEST_MODE`, `include_unnamed=1`) to check matching against unnamed clusters.
- Alpine image running as uid 65532 with a read-only root filesystem; CI with tests, Trivy and GHCR images;
  release workflow with Cosign signatures and SBOM.

[Unreleased]: https://github.com/alee-ibrahim/immich-face-api/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/alee-ibrahim/immich-face-api/releases/tag/v0.1.0
