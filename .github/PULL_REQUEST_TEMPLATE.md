## What and why

<!-- Short description of the change and the problem it solves. -->

## Checklist

- [ ] `ruff check . && ruff format --check . && pytest` pass
- [ ] No photos of real people, embeddings, names, secrets or internal hostnames added
- [ ] Nothing new is logged or reported that could contain image data, embeddings, names or keys
- [ ] API change: `immich_face_api/openapi.json` and README updated
- [ ] Immich schema/ML change: tested Immich version stated, `EXPECTED_COLUMNS` and README "Compatibility" updated
- [ ] `CHANGELOG.md` "Unreleased" updated
