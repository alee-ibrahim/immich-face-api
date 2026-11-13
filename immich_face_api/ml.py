"""Client for Immich's machine-learning server (``POST /predict``), asking exactly what Immich asks for faces."""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx


class MLError(RuntimeError):
    pass


@dataclass(frozen=True)
class DetectedFace:
    box: dict  # x1, y1, x2, y2 in pixels of the image sent
    score: float
    embedding: str  # JSON text "[f1, f2, ...]" (512 floats), as Immich stores it


def face_entries(model: str, min_score: float) -> str:
    """The ``entries`` form field Immich sends for face detection + recognition."""
    return json.dumps(
        {
            "facial-recognition": {
                "detection": {"modelName": model, "options": {"minScore": min_score}},
                "recognition": {"modelName": model},
            }
        }
    )


def parse_faces(payload: dict) -> list[DetectedFace]:
    faces = payload.get("facial-recognition")
    if not isinstance(faces, list):
        raise MLError("unexpected ML response")
    out = []
    for f in faces:
        try:
            box = {k: float(f["boundingBox"][k]) for k in ("x1", "y1", "x2", "y2")}
            emb = f["embedding"]
            if not isinstance(emb, str):  # some versions return a list
                emb = json.dumps(emb)
            out.append(DetectedFace(box=box, score=float(f.get("score", 0.0)), embedding=emb))
        except (KeyError, TypeError, ValueError) as exc:
            raise MLError("unexpected ML face entry") from exc
    return out


class MLClient:
    def __init__(self, base_url: str, model: str, min_score: float, timeout: float):
        self.entries = face_entries(model, min_score)
        self.client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def close(self) -> None:
        await self.client.aclose()

    async def ping(self) -> bool:
        try:
            r = await self.client.get("/ping", timeout=5)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def detect(self, image: bytes) -> list[DetectedFace]:
        try:
            r = await self.client.post(
                "/predict",
                data={"entries": self.entries},
                files={"image": ("image.png", image, "image/png")},
            )
        except httpx.HTTPError as exc:
            raise MLError(f"ML server unreachable: {type(exc).__name__}") from exc
        if r.status_code != 200:
            raise MLError(f"ML server answered {r.status_code}")
        return parse_faces(r.json())
