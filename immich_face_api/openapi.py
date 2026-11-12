"""The OpenAPI 3.1 description of the API (hand-written, ``openapi.json`` next to this file, also linked from the README)."""

import json
from importlib.resources import files

OPENAPI_SPEC = json.loads(files(__package__).joinpath("openapi.json").read_text(encoding="utf-8"))
