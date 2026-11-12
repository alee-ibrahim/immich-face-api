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
- only `/v1/identify`, `/v1/people` and `/healthz` are reachable from outside; `/readyz`, `/openapi.json` and the
  metrics port stay internal.

What it defends against: unauthenticated use (hashed per-client API keys, constant-time compare), abuse (per-client
rate limit, bounded concurrency, size and pixel limits), hostile files (magic-byte check, one decoder per format, pixel
limit before decoding, metadata never passed on), data leaks (no image, embedding or name in logs or error reports;
nothing written to disk; read-only database access; no CORS).

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
