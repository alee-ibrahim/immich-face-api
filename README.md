# immich-face-api

**"Who is in this photo?"** A small, read-only HTTP API next to an [Immich](https://immich.app) server: send it a
photo, get back the faces in it, each with the **named Immich person** it most likely shows (or `null`).

It is meant for a back end (a website's admin, a DAM, a newsroom tool) that wants *suggested* people tags for photos
that are **not** in Immich, using the people a curator has already named in Immich. It does not change Immich, never
stores an image, and only reads four tables of Immich's database.

```text
your back end ──POST /v1/identify (photo)──> immich-face-api ──POST /predict──> immich-machine-learning
                                                    │                          (the same face model Immich uses)
                                                    └──read-only SQL──> Immich PostgreSQL (named faces of ONE owner)
```

- Python 3.13, FastAPI, Pillow (+ pillow-heif), psycopg 3. MIT licence.
- **Pinned to Immich 3.4.1** (`FACE_API_EXPECTED_IMMICH_VERSION`): it reads Immich's internal tables, so on any other
  version, or if a column it needs changed, it reports **not ready** and refuses requests (`503`).
- Does not copy any Immich code. It talks to Immich's machine-learning server over HTTP with the same request the
  Immich server makes, and reads Immich's database with its own SQL.

## How it works

1. **Validate** (in memory only): API key, body size (default 15 MB), magic bytes (JPEG, PNG, WebP, HEIC/HEIF only),
   header-only open with that one decoder, pixel limit (default 50 MP, before decoding), then decode. EXIF orientation is
   applied; the image sent on is a **new** RGB JPEG without any metadata (EXIF, GPS, XMP, ICC dropped), longest edge at
   most `FACE_API_MAX_EDGE` (default 1440 = Immich's preview size, which is what Immich itself runs face detection on).
2. **Detect + embed**: `POST {ML_URL}/predict` with
   `{"facial-recognition": {"detection": {"modelName": <model>, "options": {"minScore": <minScore>}}, "recognition": {"modelName": <model>}}}`.
   Model and `minScore` **must equal** your Immich config (`machineLearning.facialRecognition`), or the embeddings are not
   comparable.
3. **Search**: for each face, the `k` nearest faces (exact cosine distance) among the faces of persons that belong to
   **one Immich user** (the library owner), have a **non-empty name** and are **not hidden**; faces on deleted, trashed
   or *locked* assets are ignored.
4. **Decide** — "nearest face wins": drop neighbours farther than `FACE_API_MAX_DISTANCE` (default 0.5, Immich's
   default `maxDistance`), group by person, rank persons by their **best** distance, ties broken by how many of the k
   neighbours they have. This mirrors Immich, which assigns a new face to the person of its nearest matching face.
   A majority vote was rejected because it favours people with many photos and misses people who were just named.
   `ambiguous: true` means the runner-up is within `FACE_API_AMBIGUITY_MARGIN` (0.03) of the winner: show the
   candidates instead of pre-selecting.

## API

All `/v1` endpoints need `Authorization: Bearer <api key>`. Responses are JSON with `Cache-Control: no-store`.
There is no CORS: this is a **server-to-server** API; never call it from a browser.

### `POST /v1/identify`

Body: `multipart/form-data` with the file in field **`image`**, or the raw image bytes (any `Content-Type`).

| Query | Default | |
| --- | --- | --- |
| `candidates` | `0` | also return up to N (max `FACE_API_MAX_CANDIDATES`) ranked candidates per face |
| `include_unnamed` | off | **test mode only** (`FACE_API_TEST_MODE=true`): also match unnamed person clusters (name `null`) to check the matching on an archive where few people are named. `400` otherwise |

```bash
curl -sS https://faces.example.org/v1/identify?candidates=3 \
  -H "Authorization: Bearer $FACE_API_KEY" \
  -F image=@event.jpg
```

```json
{
  "faces": [
    {
      "box": {"x": 0.412, "y": 0.18, "w": 0.061, "h": 0.103},
      "detection_score": 0.83,
      "person_id": "6f1c1f0e-8d0a-4c55-9d43-4f0b1d1f2a10",
      "name": "Jane Example",
      "confidence": 0.71,
      "distance": 0.29,
      "ambiguous": false,
      "candidates": [
        {"person_id": "6f1c1f0e-8d0a-4c55-9d43-4f0b1d1f2a10", "name": "Jane Example", "confidence": 0.71, "distance": 0.29, "votes": 7},
        {"person_id": "0b7e3b7c-1a52-4d0e-9a9e-2f7c9a1d5e33", "name": "John Example", "confidence": 0.53, "distance": 0.47, "votes": 1}
      ]
    },
    {
      "box": {"x": 0.702, "y": 0.22, "w": 0.055, "h": 0.094},
      "detection_score": 0.79,
      "person_id": null, "name": null, "confidence": null, "distance": null, "ambiguous": false,
      "candidates": []
    }
  ],
  "image": {"w": 6000, "h": 4000},
  "model": "buffalo_l",
  "max_distance": 0.5,
  "immich_version": "3.4.1"
}
```

- `box`: fractions (0–1) of the **upright** image (after EXIF orientation), `x`/`y` = top-left corner. Multiply by
  `image.w`/`image.h` for pixels. Faces are sorted left to right.
- `person_id`: Immich's person group id (stable across people sharing). `name`: as typed in Immich.
- `distance`: cosine distance of the nearest face of that person (0 = identical; the match limit is `max_distance`).
  `confidence` = 1 − distance, a **hint for a human**, not a probability.

Errors: `{"error": {"code": "...", "message": "..."}}` with `400` (bad request / undecodable image), `401` (key),
`413` (too large / too many pixels), `415` (not JPEG/PNG/WebP/HEIC), `429` (rate limit, `Retry-After`), `502` (ML or
database failed), `503` (not ready, or too many images in progress, `Retry-After`).

### `GET /v1/people`

Named, not hidden persons of the owner, with the number of their faces:
`{"people": [{"person_id": "...", "name": "Jane Example", "face_count": 42}], "count": 1}`.
Disable with `FACE_API_ENABLE_PEOPLE=false`.

### `GET /healthz`, `GET /readyz`

`/healthz` = the process is up. `/readyz` = `200` only when the Immich version and table shapes match the pin, the
role can read the four tables, the owner exists and the ML server answers; re-checked every
`FACE_API_READINESS_INTERVAL` seconds. Details go to the log, not the response.

No thumbnail endpoint: serving face crops would need access to Immich's files or API and would turn a "names only"
service into an image service.

## Configuration

Every setting is an environment variable. `*_FILE` variants read the value from a file (mounted secrets) and win.

| Variable | Default | |
| --- | --- | --- |
| `FACE_API_ML_URL` | **required** | Immich machine-learning server, e.g. `http://immich-machine-learning:3003` |
| `FACE_API_FACE_MODEL` | `buffalo_l` | = Immich `machineLearning.facialRecognition.modelName` |
| `FACE_API_MIN_SCORE` | `0.7` | = Immich `facialRecognition.minScore` |
| `FACE_API_MAX_DISTANCE` | `0.5` | match limit; Immich's `maxDistance`. Lower (e.g. 0.45) = fewer, safer suggestions |
| `FACE_API_MAX_EDGE` | `1440` | = Immich preview size |
| `FACE_API_DB_HOST` / `_PORT` / `_NAME` / `_USER` | `localhost` / `5432` / `immich` / `immich_face_api` | |
| `FACE_API_DB_PASSWORD[_FILE]` | | |
| `FACE_API_DB_SSLMODE` | `verify-full` | `disable`, `require`, `verify-ca`, `verify-full` |
| `FACE_API_DB_SSLROOTCERT` | | CA file for `verify-*` |
| `FACE_API_DB_POOL_SIZE` | `4` | |
| `FACE_API_OWNER_ID` | | Immich user id (uuid) whose named people count. Preferred (no access to the `user` table needed) |
| `FACE_API_OWNER_EMAIL` | | alternative to `OWNER_ID`; needs `SELECT (id, email) ON "user"` |
| `FACE_API_EXPECTED_IMMICH_VERSION` | `3.4.1` | the pin; change only after testing a new Immich version |
| `FACE_API_API_KEYS[_FILE]` | **required** | JSON `{"client-name": "<sha256 hex of its key>"}`; plain keys are refused |
| `FACE_API_RATE_LIMIT_PER_MINUTE` / `_BURST` | `30` / `10` | token bucket per client, per replica |
| `FACE_API_MAX_CONCURRENT_ML` | `2` | images in the ML server at once, per replica |
| `FACE_API_ML_QUEUE_TIMEOUT` / `FACE_API_ML_TIMEOUT` | `20` / `60` s | wait for a slot (then `503`) / ML request timeout |
| `FACE_API_MAX_UPLOAD_BYTES` / `FACE_API_MAX_PIXELS` | 15 MiB / 50 000 000 | |
| `FACE_API_KNN` / `_MAX_CANDIDATES` / `_AMBIGUITY_MARGIN` | `20` / `5` / `0.03` | |
| `FACE_API_ENABLE_PEOPLE` | `true` | `GET /v1/people` |
| `FACE_API_TEST_MODE` | `false` | allows `include_unnamed=1`. **Keep off in production** |
| `FACE_API_METRICS_PORT` | `9090` | Prometheus `/metrics` on its own port (`0` = off) |
| `FACE_API_SENTRY_DSN[_FILE]` | | Sentry / GlitchTip; no request bodies, headers or PII are sent |
| `FACE_API_ENVIRONMENT` | `production` | Sentry environment |

API port: **8080**. Image: `ghcr.io/alee-ibrahim/immich-face-api` (runs as uid 65532, works with a read-only root
filesystem; nothing is written to disk).

### API keys

```bash
docker run --rm --entrypoint python ghcr.io/alee-ibrahim/immich-face-api:main -m immich_face_api.keys new > website.key
docker run --rm -i --entrypoint python ghcr.io/alee-ibrahim/immich-face-api:main -m immich_face_api.keys hash < website.key
# -> put {"website": "<that hex>"} into FACE_API_API_KEYS; give website.key to the client over a secure channel
```

Only the SHA-256 is configured; keys are compared in constant time. A key is a long random string, so a plain hash is
enough. Rotate: add a second entry (`"website-2026b"`), switch the client, remove the old one.

### Database role

[`examples/readonly-role.sql`](examples/readonly-role.sql): `SELECT` on `person`, `asset_face`, `face_search`,
`version_history` only, transactions read-only, TLS-only `pg_hba` line. The service also sets
`default_transaction_read_only` and a statement timeout on its connections.

## Security model

- **Server-to-server only.** Keep the API key on your back end; no CORS headers are ever sent. Put it behind an
  authenticating proxy too if you expose it to the internet (e.g. an access service token and an IP allowlist).
- **Least privilege:** a read-only DB role on four tables; no Immich API key; no access to Immich's files.
  Network: it needs only the ML server and PostgreSQL.
- **Input:** size, magic bytes, one-decoder open, pixel limit before decode, metadata never passed on.
- **Abuse:** per-client token bucket, bounded ML concurrency, no OpenAPI/docs endpoints, metrics on another port.
- **Logs:** one JSON line per request (client name, status, duration, number of faces and matches, request id, the
  `CF-Connecting-IP` header if present). Never image data, embeddings or names.
- Report security issues privately (GitHub security advisory), not as public issues.

## Privacy

Face embeddings are **biometric data**. Deploying this service means processing biometric data of the people in your
photos, and naming people in Immich means building a face register. **Whether that is lawful, and under what
conditions (consent, legitimate interest, public role, data protection impact assessment, retention, information to
the people concerned), is the deployer's responsibility.** Design choices that help:

- It stores **no images and no embeddings** of uploads: everything stays in memory for the duration of the request.
- It only ever returns names a curator chose to type into Immich, for one owner's library; unnamed and hidden people
  are never returned (except in test mode, and then without a name).
- Treat the output as **suggestions for a human** to confirm, never as an automatic identification.

## Scaling

The search is exact (no ANN index) over the owner's **named** faces: Immich's `vchordrq` index is searched before the
"named, this owner" filter, so it can return too few or no rows for such a selective filter. Exact search costs roughly
20 ms per 1,000 named faces per detected face; it is fine up to some tens of thousands of named faces. Beyond that,
move to an index with pre-filtering or an in-memory cache of the named embeddings.

## Upgrading Immich

1. Test the new Immich version on a copy: run this service against it with `FACE_API_EXPECTED_IMMICH_VERSION` set to
   the new version; `/readyz` lists nothing in its body, the log line `readiness changed` says what differs.
2. Check the ML request in Immich's `server/src/repositories/machine-learning.repository.ts` (`detectFaces`) and
   face filters (`asset_face.deletedAt`, `isVisible`, `assetDeletedAt`, `assetVisibility`) are unchanged.
3. Release a version of this service with the new pin (and SQL changes if needed).

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
ruff check . && ruff format --check . && pytest
```

Tests use synthetic images (gradients, solid colours) and a mocked ML server and database: there are no photos of
real people, and no embeddings, in this repository. Please keep it that way in contributions.

## Licence

MIT, see [LICENSE](LICENSE). Immich is a trademark of its owners; this project is not affiliated with Immich.
