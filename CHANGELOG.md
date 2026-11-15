# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed (cold start and latency)
- `/readyz` waits for the face cache: no traffic until the first load is done, so rolling updates no longer send
  requests to cold pods. Until then identify/verify answer `503 busy` (`Retry-After: 10`) at once instead of running the
  exact SQL search (which hit the 15 s statement timeout on a busy database). Old behaviour:
  `FACE_API_FACE_CACHE_SQL_FALLBACK=true`.
- `FACE_API_FACE_CACHE_DTYPE=float32` (opt-in): no float16 conversion per search (measured in a pod: ~250 ms per
  request at 118k cached faces -> ~10 ms), at 2 KiB per cached face.
- `FACE_API_FACE_CACHE_FACES_REFRESH_SECONDS` (900): face-only changes rebuild at most every 15 min; new or renamed
  people still within `FACE_API_FACE_CACHE_REFRESH_SECONDS`. Less database load while Immich is recognising.
- Rebuild decodes a batch of binary embeddings in one numpy call (no Python work per vector).
- Images go to the ML server as PNG with compress_level 0 (still lossless; ~2.5x faster to encode).

### Added (labeler: look-alike guard for the named-neighbour vote)
- A vote is held when its faces are not clearly closer to the voted name than to another named person (median
  per-face gap < `vote_min_name_gap`, default 0.05), or < `vote_lookalike_min_gap` (0.10) when the voted name has a
  known look-alike (`vote_lookalike_distance`, 0.40, between name centroids). The hold names the runner-up. Found on
  a real archive: a vote spread a member's name to a different person's clusters.
- `LABELER_DB_STATEMENT_TIMEOUT` (seconds, default 1800; was fixed at 600): at ~71k clusters the centroid query
  outgrew 10 minutes.

### Added (main subject)
- **Main-subject scoring** in `POST /v1/identify`: per face `prominence` (0–1), `role` (`main`, `co_subject`,
  `background`) and the raw `subject_signals` (size, dominance, lead, sharpness and sharpness relative to the other faces
  and to the background, centrality, rule of thirds, edge cut, exposure, detection score, and head pose from landmarks);
  per photo `photo_type` (`portrait`, `small_group`, `crowd`), `main_face_index` and `subject_model_version`. Computed
  in-process from the decoded image (~15–30 ms per photo). `FACE_API_SUBJECT_SCORING` (`always`, `request`, `off`) and
  `?subject=0|1`.
- `POST /v1/verify` (and batch items) with `subject=1`: `is_main_subject`, `role` and `prominence` per claim.
- `FACE_API_SUBJECT_LANDMARKS`: head pose from the detector's 5 landmarks (one more, detection-only ML call per image).
- Versioned subject model (weights + thresholds) as JSON: `FACE_API_SUBJECT_MODEL_FILE` / `FACE_API_SUBJECT_MODEL`.
- Admin **Label subjects**: curators mark main / co-subject / background on random Immich photos with faces (previews
  fetched through Immich's API with a view-only key, never stored) or uploads; only boxes, signals and labels are kept
  (store migration `004_subject_labels.sql`).
- `python -m immich_face_api calibrate`: logistic regression + threshold search on the labels, cross-validated
  precision/recall for "main" and "subject", writes a new versioned model; `--export` / `--labels` JSONL.
- Admin **curator** role: `FACE_API_OIDC_CURATOR_VALUES` (values of `FACE_API_OIDC_ADMIN_CLAIM`); curators only see
  *Label subjects*. `FACE_API_IMMICH_API_URL`, `FACE_API_IMMICH_API_KEY[_FILE]`.
- Readiness checks `asset_face."assetId"` (photo sampling for labelling; no new grant).

### Fixed (face cache memory)
- The face cache rebuild no longer holds every fetched row plus float32 copies (~8 KiB per cached face, OOM kills at
  ~110k cached faces in 1 GiB). Faces are counted per person first, the float16 matrix is allocated once, and
  embeddings are streamed through a server-side cursor in batches of 2,048 straight into it; the old snapshot keeps
  serving until the swap. Peak ≈ 2 × 1 KiB per cached face plus one batch.
- Search converts the cache to float32 in chunks of 16,384 faces (was 65,536: 128 MiB per concurrent request).

### Added (face cache)
- `FACE_API_FACE_CACHE_MAX_FACES` (default 1,000,000): total cap; above it the per-person cap is lowered to fit.
- Lazy cache: `FACE_API_FACE_CACHE_IDLE_SECONDS` (API, default 0 = always loaded) and
  `FACE_API_ADMIN_FACE_CACHE_IDLE_SECONDS` (admin, default 1800): load on first use, drop when idle. The admin UI no
  longer holds a full cache all the time.
- Metrics: `face_api_face_cache_{faces,people,bytes,per_person,loaded_timestamp_seconds}`,
  `face_api_face_cache_refresh_seconds`, `face_api_face_cache_refresh_errors_total`.

### Added (labeler: roster source)
- Web source `"kind": "roster"`: people (name, portrait URL, other-language names, details such as position and
  organisation) from a JSON roster file, a local path or an http(s) URL, typically written by a separate scraper.
  Position and organisation are shown in the evidence (report and audit), never in the name. A broken roster is
  logged and skipped; the other sources still run. Example: `examples/labeler/roster.example.json`.

### Added (people changes and verify)
- **People change log**: `GET /v1/people/changes?since=<cursor|ISO time>` returns, in order, `created`, `renamed`,
  `faces_moved` (splits and reassignments, counted per window and person pair), `merged` and `deleted` (deleted /
  hidden / unnamed) events for the owner's named people. Derived from a snapshot in the store (people and which named
  person each face belongs to; ids only), diffed every `FACE_API_CHANGES_INTERVAL` (300 s) against faces with a newer
  `asset_face."updateId"`, plus a daily full reconcile; one process at a time (advisory lock). Immich's audit tables
  record only deletions, so they are not used; no new database grants. Store migration `003_people_changes.sql`.
- `POST /v1/people/resolve`: where stored person ids stand now (current, merged into ..., hidden, unnamed, deleted,
  unknown); `GET /v1/people` returns `changes_cursor` for a full re-sync.
- `POST /v1/verify`: re-check claimed people (stored tags) against a photo: distance, within the threshold, verdict
  (`match`, `match_not_best`, `no_match`, `no_face`, `no_reference`, `person_gone`) and the current best person;
  optional `box` from an earlier identify; merges followed. `POST /v1/verify/batch` (JSON, base64, up to 10 photos,
  one rate-limit token each). Logged as request lines only (no faces, no crops).
- Admin: *Changes* page (status, events, filters by type and person) and recent changes on the dashboard; Docs page
  sections for verify and changes.

### Added (labeler: titles)
- `names.titles` / `names.keep_titles` in the labeler config: honorifics and titles are removed from the start and end
  of every name (folders, web cards, Wikidata, and at the single write point), kept titles are written one way
  (e.g. `Dr` / `Doctor` -> `Dr.`), whitespace and casing are normalised. Whole words at the edges only.
- `tidy-names` command (dry run unless `--apply`): removes titles from names already in Immich, owner names through
  the API, curators' copies (`--copies`) through the name-copy role only where the copy follows the owner's name;
  doubtful names (a title word inside) and curators' own spellings are listed, never changed. Audited; `undo --run` /
  `undo --copies --run` revert.

### Fixed (same-name candidates)
- A face whose two best candidates are **one person split over two Immich people with the same name** is no longer
  `ambiguous`. Same normalised name and centroids within `FACE_API_SAME_NAME_MAX_DISTANCE` (0.6) = same person;
  namesakes (different faces) stay ambiguous. Needs the face cache; the candidates list is unchanged.

### Added (request log and missed faces)
- **Request log** in the admin store (on by default when a store is configured, 90 days): per identify call the app,
  status / error code, latency, image size, counts of matched, unknown, ambiguous and near-threshold faces, and per face
  (at most 50) the match and the nearest named people with distances, also beyond the threshold.
- **Face crops** (`FACE_API_STORE_CROPS=none|unknown|all`, default `none`, 30 days): a 160 px JPEG per face without
  metadata, plus its embedding for *find similar in Immich*. Biometric data: see README "Request log and face crops".
- Admin pages *Requests* (filters, detail with crops) and *Missed faces* (unknown, ambiguous, near-threshold faces,
  newest first, with candidates and distances); *find similar in Immich* lists the nearest named and unnamed people
  with links to Immich (`FACE_API_IMMICH_URL`).
- Dashboard: missed-face rate over time, top apps by unknown faces.
- Retention purge every few hours; delete-now in the admin UI (typed confirmation, audited) and
  `python -m immich_face_api purge --crops|--all|--retention`.
- Written asynchronously through a bounded queue: identify never waits for the store; entries are dropped and counted
  (`face_api_request_log_dropped_total`) when it is down.

### Changed
- The store is no longer "aggregates only": see the privacy notes in README, SECURITY.md and the admin Docs page.

### Added (labeler coverage)
- Named-neighbour vote (`named_people`): more clusters of people already named in Immich.
- `write_sources`: only approved sources are written; others are proposed and held.
- Web sources: `kind: "wikidata"` (SPARQL, Commons licence recorded), `name_template`, `image_template`,
  `alt_name_groups`, `immutable`, `licence`.
- `worklist` command: the biggest unnamed people the labeler will not name, as an HTML/CSV worklist for curators.

### Fixed
- Identify timed out once many people were named. The exact SQL search read every named face (about 4.4 s per
  detected face at 55,000 named faces), so group photos hit the statement timeout. Named faces are now cached in
  memory (float16, up to 100 per person, refreshed on change) and searched with numpy, in milliseconds.
- Database timeouts and a busy connection pool now return `503 busy` with `Retry-After` instead of `502`.

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
- People labeler (`python -m immich_face_api.labeler`): proposes names for unnamed people from person-named folders
  (dominant identity, Immich's stored embeddings) and web listing pages (portraits embedded by Immich ML), with
  confidence bands, merge candidates, an HTML/CSV review report, dry run by default, audited writes through the Immich
  API with never-overwrite checks, and undo; held proposals (name conflicts, ambiguous namesakes) are never written;
  `share-names` copies the owner's names into still-empty shared copies (audited, undoable). Adds the `numpy` dependency.
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
