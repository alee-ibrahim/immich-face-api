# Security policy

## Reporting a vulnerability

Please **do not open a public issue**. Report it privately through GitHub:
**Security → Advisories → Report a vulnerability** on this repository. Include the version (image tag or digest), what
an attacker can do, and steps to reproduce. You should get an answer within a week. Fixes are released as a new
version; the advisory is published after users had a chance to update.

## Supported versions

Only the newest release (and `main`) gets security fixes.

## Threat model in short

immich-face-api is meant to be called **server to server** by a trusted back end, behind a TLS reverse proxy
(with a body-size cap, a timeout and a per-IP rate limit; an IP allowlist or an access-service token is optional
extra hardening, the API key is the authentication). It assumes:

- the network between it and Immich (ML server, PostgreSQL) is trusted or encrypted (`FACE_API_DB_SSLMODE=verify-full`);
- its database role is the read-only role from `examples/readonly-role.sql`;
- only `/v1/identify`, `/v1/people` and `/healthz` are reachable from outside; `/readyz`, `/openapi.json`, the
  metrics port and the admin UI (with its request log and face crops) stay internal.

What it defends against: unauthenticated use (hashed per-client API keys, constant-time compare), abuse (per-client
rate limit, bounded concurrency, size and pixel limits), hostile files (magic-byte check, one decoder per format, pixel
limit before decoding, metadata never passed on), data leaks (no image, embedding or name in logs or error reports;
nothing written to disk; read-only database access to Immich; no CORS).

## Stored data (admin store)

With the optional store (`FACE_API_STORE_URL`) the service keeps hourly counts and, by default, a **request log**
(per request: app, status, image size, and per face the matched name, person id and distances; 90 days). With
`FACE_API_STORE_CROPS=unknown|all` it also keeps **face crops and embeddings** (30 days). These are **biometric
data**. They are readable only through the admin UI (OIDC or bootstrap token, internal network), crops are served with
`Cache-Control: no-store` and never by the API port, and both can be deleted at once (admin UI or
`python -m immich_face_api purge`), with an audit entry. The uploaded photo is never stored. Protect the store like
any database of personal data: TLS, its own credentials, backups with the same retention in mind. Consent and the
legal basis are the deployer's responsibility. Details: README, "Request log and face crops".

Out of scope: a compromised client back end (it can do whatever its key allows), and the accuracy of face recognition
itself (results are suggestions for a human).

## Supply chain

- Dependencies are pinned (`requirements.txt`) and updated by Dependabot.
- Images are built by GitHub Actions from this repository, scanned with Trivy (the build fails on Critical findings),
  and release images are signed with Cosign (keyless, GitHub OIDC) and carry an SBOM and provenance attestation.
  Verify a release image:

  ```bash
  cosign verify ghcr.io/alee-ibrahim/immich-face-api:<version> \
    --certificate-identity-regexp 'https://github.com/alee-ibrahim/immich-face-api/.github/workflows/release.yml@refs/tags/v.*' \
    --certificate-oidc-issuer https://token.actions.githubusercontent.com
  ```
