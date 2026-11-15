"""Client for Immich's machine-learning server (``POST /predict``), asking exactly what Immich asks for faces."""

from __future__ import annotations

import asyncio
import dataclasses
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
    landmarks: list | None = None  # 5 x [x, y] (eyes, nose, mouth corners), only from a detection-only call


def detection_entries(model: str, min_score: float) -> str:
    """Detection only: Immich's ML then answers with boxes, scores and the 5 landmarks per face (the combined
    detection + recognition answer has no landmarks)."""
    return json.dumps({"facial-recognition": {"detection": {"modelName": model, "options": {"minScore": min_score}}}})


def parse_landmarks(payload: dict) -> list[tuple[dict, list]]:
    """[(box, landmarks)] from a detection-only answer."""
    det = payload.get("facial-recognition")
    if not isinstance(det, dict):
        raise MLError("unexpected ML detection response")
    try:
        boxes, marks = det.get("boxes") or [], det.get("landmarks") or []
        out = []
        for b, lm in zip(boxes, marks, strict=True):
            box = {"x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3])}
            out.append((box, [[float(p[0]), float(p[1])] for p in lm][:5]))
        return out
    except (TypeError, ValueError, IndexError) as exc:
        raise MLError("unexpected ML detection entry") from exc


def attach_landmarks(faces: list[DetectedFace], found: list[tuple[dict, list]]) -> list[DetectedFace]:
    """Give each face the landmarks of the detection with the same box (same image, same detector: the boxes agree
    to rounding). Faces without a counterpart keep None."""
    out = []
    for f in faces:
        best, err = None, 4.0  # pixels of total corner difference
        for box, lm in found:
            e = sum(abs(f.box[k] - box[k]) for k in ("x1", "y1", "x2", "y2"))
            if e <= err:
                best, err = lm, e
        out.append(dataclasses.replace(f, landmarks=best))
    return out


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
        self.detection_only = detection_entries(model, min_score)
        self.client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def close(self) -> None:
        await self.client.aclose()

    async def ping(self) -> bool:
        try:
            r = await self.client.get("/ping", timeout=5)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def _predict(self, image: bytes, entries: str) -> dict:
        try:
            r = await self.client.post(
                "/predict",
                data={"entries": entries},
                files={"image": ("image.png", image, "image/png")},
            )
        except httpx.HTTPError as exc:
            raise MLError(f"ML server unreachable: {type(exc).__name__}") from exc
        if r.status_code != 200:
            raise MLError(f"ML server answered {r.status_code}")
        return r.json()

    async def detect(self, image: bytes, landmarks: bool = False) -> list[DetectedFace]:
        """Faces with embeddings; with ``landmarks``, a second, detection-only call runs at the same time and its
        landmarks are attached to the faces (about one more face detection of ML work per image)."""
        if not landmarks:
            return parse_faces(await self._predict(image, self.entries))
        full, det = await asyncio.gather(self._predict(image, self.entries), self._predict(image, self.detection_only))
        return attach_landmarks(parse_faces(full), parse_landmarks(det))
