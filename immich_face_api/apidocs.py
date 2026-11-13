"""Turn the bundled OpenAPI document into plain data for the admin's server-rendered API reference."""

from __future__ import annotations

import json
from typing import Any

from .openapi import OPENAPI_SPEC

METHOD_ORDER = ("get", "post", "put", "patch", "delete")


def _resolve(spec: dict, node: Any) -> Any:
    seen = 0
    while isinstance(node, dict) and "$ref" in node and seen < 10:
        ref = node["$ref"]
        if not ref.startswith("#/"):
            break
        target: Any = spec
        for part in ref[2:].split("/"):
            target = target[part]
        node, seen = target, seen + 1
    return node


def _type(schema: dict) -> str:
    t = schema.get("type")
    if isinstance(t, list):
        t = " | ".join(t)
    if t == "array":
        items = schema.get("items", {})
        name = items.get("$ref", "").rsplit("/", 1)[-1] if "$ref" in items else _type(items)
        return f"array of {name or 'any'}"
    if "$ref" in schema:
        return schema["$ref"].rsplit("/", 1)[-1]
    if "enum" in schema:
        return f"{t or 'string'}: " + ", ".join(str(v) for v in schema["enum"])
    if "const" in schema:
        return f"{t or 'value'} = {schema['const']!r}"
    fmt = schema.get("format")
    return f"{t or 'any'}{f' ({fmt})' if fmt else ''}"


def fields(spec: dict, schema: dict, prefix: str = "", depth: int = 0) -> list[dict]:
    """Flatten an object schema into rows (name, type, required, description); nested objects get dotted names."""
    schema = _resolve(spec, schema)
    if schema.get("type") == "array":
        return fields(spec, schema.get("items", {}), prefix + "[]", depth)
    rows = []
    required = set(schema.get("required", []))
    for name, sub in (schema.get("properties") or {}).items():
        raw = sub
        sub = _resolve(spec, sub)
        rows.append(
            {
                "name": prefix + name,
                "type": _type(raw),
                "required": name in required,
                "description": sub.get("description") or raw.get("description") or "",
                "depth": depth,
            }
        )
        if depth < 3:
            if sub.get("type") == "object" or "properties" in sub:
                rows += fields(spec, sub, prefix + name + ".", depth + 1)
            elif sub.get("type") == "array":
                items = _resolve(spec, sub.get("items", {}))
                if "properties" in items:
                    rows += fields(spec, items, prefix + name + "[].", depth + 1)
    return rows


def reference(spec: dict | None = None) -> list[dict]:
    spec = spec or OPENAPI_SPEC
    ops = []
    for path, item in spec["paths"].items():
        for method in METHOD_ORDER:
            op = item.get(method)
            if not op:
                continue
            body = op.get("requestBody")
            responses = []
            for status, resp in op.get("responses", {}).items():
                resp = _resolve(spec, resp)
                content = resp.get("content", {})
                media, mobj = next(iter(content.items()), (None, {}))
                example = mobj.get("example")
                responses.append(
                    {
                        "status": status,
                        "description": resp.get("description", ""),
                        "media": media,
                        "fields": fields(spec, mobj["schema"]) if mobj.get("schema") else [],
                        "example": json.dumps(example, indent=2) if example is not None else None,
                    }
                )
            ops.append(
                {
                    "id": op.get("operationId", f"{method}-{path}"),
                    "method": method.upper(),
                    "path": path,
                    "summary": op.get("summary", ""),
                    "auth": op.get("security", spec.get("security")) != [],
                    "parameters": [
                        {
                            "name": p["name"],
                            "in": p["in"],
                            "type": _type(p.get("schema", {})),
                            "required": p.get("required", False),
                            "description": p.get("description", ""),
                        }
                        for p in op.get("parameters", [])
                    ],
                    "body": [
                        {
                            "media": media,
                            "fields": fields(spec, m["schema"]) if m.get("schema", {}).get("properties") else [],
                        }
                        for media, m in (body or {}).get("content", {}).items()
                    ],
                    "responses": responses,
                }
            )
    return ops
