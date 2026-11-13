-- Request log and face crops (see README "Request log and face crops").
-- This changes what the store holds: per identify request, the app, image size, faces found and the names, person
-- ids and distances of the matches; per face, optionally a small crop (JPEG, no metadata) and its embedding.
-- Crops and embeddings are biometric data. Both are purged after FACE_API_CROP_RETENTION_DAYS, requests (with their
-- faces) after FACE_API_REQUEST_LOG_RETENTION_DAYS. The uploaded photo itself is never stored.

CREATE TABLE request_log (
    id            bigserial PRIMARY KEY,
    at            timestamptz NOT NULL DEFAULT now(),
    rid           text NOT NULL DEFAULT '',          -- request id (X-Request-Id), to find the service log line
    client        text NOT NULL,                      -- app name of the API key
    endpoint      text NOT NULL,
    status        smallint NOT NULL,
    error_code    text,                               -- "invalid_image", "busy", ... for failed requests
    latency_ms    integer NOT NULL,
    image_bytes   integer,
    image_w       integer,
    image_h       integer,
    image_format  text,
    faces         integer NOT NULL DEFAULT 0,         -- all faces detected (faces logged below: at most LOG_MAX_FACES)
    matched       integer NOT NULL DEFAULT 0,
    unknown       integer NOT NULL DEFAULT 0,
    ambiguous     integer NOT NULL DEFAULT 0,
    near          integer NOT NULL DEFAULT 0          -- matches within NEAR_MARGIN of max_distance
);
CREATE INDEX request_log_at ON request_log (at DESC);
CREATE INDEX request_log_client_at ON request_log (client, at DESC);

CREATE TABLE face_log (
    id               bigserial PRIMARY KEY,
    request_id       bigint NOT NULL REFERENCES request_log (id) ON DELETE CASCADE,
    at               timestamptz NOT NULL,
    client           text NOT NULL,
    idx              smallint NOT NULL,               -- position in the response (left to right)
    kind             text NOT NULL CHECK (kind IN ('matched', 'near', 'ambiguous', 'unknown')),
    box              jsonb NOT NULL,                  -- {x, y, w, h} as fractions of the image
    detection_score  real NOT NULL,
    person_id        text,                            -- the match (NULL when unknown)
    name             text,
    distance         real,
    candidates       jsonb NOT NULL DEFAULT '[]',     -- nearest named people, also beyond the threshold (near-misses)
    crop             bytea,                           -- small JPEG, only with FACE_API_STORE_CROPS; NULL after purge
    embedding        bytea                            -- float16 x 512, kept only together with the crop
);
CREATE INDEX face_log_request ON face_log (request_id);
CREATE INDEX face_log_missed ON face_log (at DESC) WHERE kind <> 'matched';
CREATE INDEX face_log_crop_at ON face_log (at) WHERE crop IS NOT NULL;
