# Contributing

Thanks for helping. Small, focused pull requests are easiest to review.

## Development

```bash
python -m venv .venv && . .venv/bin/activate        # Python 3.13
pip install -r requirements-dev.txt
ruff check . && ruff format --check . && pytest
docker build -t immich-face-api:dev .
```

The tests need no Immich: the ML server and the database are mocked.

## Rules

- **No real people in the repository.** No photos of real persons, no face crops, no embeddings, no names of real
  people, not even in tests or issues. Use synthetic images (gradients, shapes) and mocked ML responses, as the tests do.
- **No secrets, hostnames or internal details** of your own deployment in code, examples or issues.
- Keep the service **read-only** towards Immich and **stateless**: nothing written to disk or a database.
- Never log or report image bytes, embeddings, person names or API keys. If you add a log line or an error path, check
  `immich_face_api/scrub.py` still covers it. The request log in the admin store (`requestlog.py`) is the one
  documented place where names, crops and embeddings are kept; keep it behind its settings and retention.
- Changing anything that reads Immich's tables or calls its ML server: say which Immich version you tested against,
  and update `EXPECTED_COLUMNS` in `db.py`, the README "Compatibility" table and the CHANGELOG.
- API changes: update `immich_face_api/openapi.json` (a test checks that every route is documented) and the README.
- Add a line under "Unreleased" in `CHANGELOG.md`.

## Commits and releases

Conventional, descriptive commit messages are welcome but not required. Maintainers release by pushing a tag
`vX.Y.Z` (semantic versioning); the release workflow builds, signs and publishes the image and the GitHub release.
