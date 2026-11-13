-- immich-face-api store (its OWN database, never Immich's): API keys made in the admin UI and hourly usage
-- aggregates. No images, embeddings, person names or plain keys are ever stored here.

CREATE TABLE api_keys (
    id                     uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    app                    text NOT NULL,                -- the calling application ("website"); analytics group by it
    owner                  text NOT NULL DEFAULT '',     -- who is responsible for it (team, person, e-mail)
    label                  text NOT NULL DEFAULT '',     -- free note: environment, purpose
    prefix                 text NOT NULL UNIQUE,         -- public part of the key, shown in the UI: ifa_<prefix>_...
    secret_hash            text NOT NULL,                -- sha256 hex of the whole key
    rate_limit_per_minute  integer CHECK (rate_limit_per_minute IS NULL OR rate_limit_per_minute > 0),
    expires_at             timestamptz,
    created_at             timestamptz NOT NULL DEFAULT now(),
    created_by             text NOT NULL DEFAULT '',
    last_used_at           timestamptz,
    revoked_at             timestamptz,
    revoked_by             text
);
CREATE INDEX api_keys_app ON api_keys (app);

-- One row per hour, client (app name) and endpoint. Counts only.
CREATE TABLE usage_hourly (
    hour            timestamptz NOT NULL,
    client          text NOT NULL,
    endpoint        text NOT NULL,
    requests        bigint NOT NULL DEFAULT 0,
    client_errors   bigint NOT NULL DEFAULT 0,      -- 4xx (incl. 401 with a known client, 413, 415, 429)
    server_errors   bigint NOT NULL DEFAULT 0,      -- 5xx
    faces           bigint NOT NULL DEFAULT 0,
    matched         bigint NOT NULL DEFAULT 0,      -- faces matched to a named person
    ambiguous       bigint NOT NULL DEFAULT 0,
    latency_ms_sum  bigint NOT NULL DEFAULT 0,
    latency_hist    bigint[] NOT NULL,              -- counts per bucket, bounds in usage.LATENCY_BOUNDS_MS (+ overflow)
    PRIMARY KEY (hour, client, endpoint)
);
CREATE INDEX usage_hourly_client_hour ON usage_hourly (client, hour);

-- Who changed keys, and when. No secrets.
CREATE TABLE audit_log (
    id      bigserial PRIMARY KEY,
    at      timestamptz NOT NULL DEFAULT now(),
    actor   text NOT NULL,
    action  text NOT NULL,
    detail  jsonb NOT NULL DEFAULT '{}'
);
