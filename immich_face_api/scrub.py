"""Error-report scrubbing for Sentry / GlitchTip (``before_send`` and ``before_breadcrumb``).

The service never puts images, embeddings or person names into logs, but an exception message or a local
variable could still carry something. Events therefore leave the process with:

- no request body, headers, cookies, query string or environment (only method and URL path are kept);
- no user information and no server name;
- no stack-frame local variables (``include_local_variables=False`` at init, and stripped here again);
- every string in messages, exception values, breadcrumbs, tags and extra passed through ``redact``:
  bearer tokens, long key-like strings, e-mail addresses, data URIs / base64 blobs and embedding-like number lists
  are replaced.
"""

from __future__ import annotations

import re
from typing import Any

_PATTERNS = [
    (
        re.compile(r"(?i)\b(bearer|token|api[_-]?key|authorization|cf-access-client-secret)\b\s*[:=]?\s*\S+"),
        r"\1 [redacted]",
    ),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[email]"),
    (re.compile(r"data:[\w/+.-]+;base64,[A-Za-z0-9+/=]+"), "[data-uri]"),
    (re.compile(r"\[\s*-?\d+\.\d+(?:\s*,\s*-?\d+\.\d+){7,}[^\]]*\]?"), "[vector]"),
    (re.compile(r"\b[A-Za-z0-9_\-+/=]{32,}\b"), "[secret]"),
]
_MAX_STRING = 2000


def redact(value: str) -> str:
    for pattern, repl in _PATTERNS:
        value = pattern.sub(repl, value)
    return value[:_MAX_STRING]


def _walk(obj: Any, depth: int = 0) -> Any:
    if depth > 8:
        return "[truncated]"
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: _walk(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_walk(v, depth + 1) for v in obj]
    return obj


def scrub_event(event: dict, _hint: Any = None) -> dict:
    req = event.get("request")
    if isinstance(req, dict):
        event["request"] = {k: req[k] for k in ("method", "url") if k in req}
        if "url" in event["request"]:
            event["request"]["url"] = str(event["request"]["url"]).split("?", 1)[0]
    event.pop("user", None)
    event.pop("server_name", None)
    for exc in (event.get("exception") or {}).get("values", []) or []:
        if isinstance(exc.get("value"), str):
            exc["value"] = redact(exc["value"])
        for frame in (exc.get("stacktrace") or {}).get("frames", []) or []:
            frame.pop("vars", None)
    for key in ("message", "logentry", "breadcrumbs", "extra", "tags", "contexts"):
        if key in event:
            event[key] = _walk(event[key])
    return event


def scrub_breadcrumb(crumb: dict, _hint: Any = None) -> dict | None:
    if crumb.get("category") in ("httplib", "http"):  # outgoing HTTP: keep method/status, drop URLs with params
        data = crumb.get("data") or {}
        crumb["data"] = {k: data[k] for k in ("method", "status_code") if k in data}
    return _walk(crumb)
