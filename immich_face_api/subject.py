"""Main-subject scoring: is a face the subject of the photo (portrait, speech, handshake) or just in it?

Pure computation on the decoded image and the detector's boxes (no I/O, no extra model). For every face it measures
a few cheap signals, combines them with a logistic model into ``prominence`` (0..1) and then applies a handful of
rules to give a ``role``: ``main``, ``co_subject`` or ``background``. The weights and thresholds live in a versioned
JSON model (``FACE_API_SUBJECT_MODEL_FILE``); ``python -m immich_face_api calibrate`` fits a new one from curator
labels. See README "Main subject".

Signals (``subject_signals`` in the API answer):

* ``area``: face box area / image area; ``size``: its square root (the face's side relative to the image).
* ``dominance``: this face's area / the largest face's area (1 for the largest face).
* ``lead``: this face's area / the largest other face's area (above 1 only for the largest face; ``null`` for a lone
  face): how clearly a face stands out.
* ``sharpness``: focus of the face, 0..1: the share of its pixel-to-pixel variation that a slight re-blur removes
  (a "blur effect" measure), on the face crop resampled to 48 px so faces of any size are judged at one scale. It
  hardly depends on contrast or on content (smooth skin versus glasses and beards). In-focus faces measure about
  0.5-0.7, visibly blurred ones below about 0.4.
* ``sharpness_rel``: ``sharpness`` / the sharpest face of the photo (1 for the sharpest).
* ``focus_bg``: log2(face sharpness / median sharpness of textured background tiles, both measured on the photo
  scaled to at most 640 px): positive when the face is
  sharper than the background (shallow depth of field around a subject); ``null`` when there is no textured
  background to compare with.
* ``centrality`` (1 at the centre, 0 in a corner), ``thirds`` (1 on a rule-of-thirds point), ``edge_cut`` (share of
  the detector's box outside the frame) and ``touches_edge``.
* ``yaw`` / ``pitch`` (degrees) and ``frontal`` (1 = facing the camera, 0 = profile) from the detector's 5 landmarks.
  Immich's ML answer for detection + recognition carries no landmarks, so they need a second, detection-only call
  (``FACE_API_SUBJECT_LANDMARKS``); without it the three are ``null`` and the model uses a neutral 0.5.
* ``detection_score``, ``luminance`` (mean, 0..1), ``clipped`` (share of near-black or near-white pixels) and
  ``exposure`` (1 = well exposed).

Per photo: ``face_count``, ``subject_count`` (faces at least ``type_min_dominance`` of the largest and not tiny) and
``photo_type`` from it: ``portrait`` (1), ``small_group`` (2..``small_group_max``) or ``crowd``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

ROLES = ("main", "co_subject", "background")
PHOTO_TYPES = ("none", "portrait", "small_group", "crowd")

# Feature names of the logistic model, in a fixed order. Each is a transform of the raw signals into about 0..1.
FEATURES = (
    "size",  # (log10(area) + 4) / 4, clipped: area 1e-4 -> 0, 1e-2 -> 0.5, 1 -> 1
    "dominance",
    "lead",  # log2(area / the largest other face's area) / 3, clipped to -1..1 (1 for a lone face)
    "sharpness",  # (sharpness - 0.2) / 0.5, clipped to 0..1
    "sharpness_rel",
    "focus_bg",  # focus_bg / 3, clipped to -1..1 (0 when unknown)
    "position",  # max(centrality, thirds)
    "edge_cut",
    "frontal",
    "detection_score",
    "exposure",
    "crowd",  # log2(subject_count) / 4, clipped to 0..1
)

DEFAULT_MODEL: dict = {
    "version": "default-1",
    "description": "hand-set defaults, to be recalibrated from curator labels",
    "intercept": -12.0,
    "weights": {
        "size": 8.0,
        "dominance": 2.5,
        "lead": 2.5,
        "sharpness": 2.0,
        "sharpness_rel": 3.0,
        "focus_bg": 1.5,
        "position": 2.0,
        "edge_cut": -3.0,
        "frontal": 0.8,
        "detection_score": 1.5,
        "exposure": 0.5,
        "crowd": -3.0,
    },
    "rules": {
        # never a subject: face side below this share of the image side, or below this many pixels (sent image)
        "min_size": 0.04,
        "min_face_px": 28,
        # never a subject: blurred (absolute) or much softer than the sharpest face of the photo
        "min_sharpness": 0.3,
        "min_sharpness_rel": 0.55,
        # main: prominence at least main_min; co_subject: at least co_min
        "main_min": 0.45,
        "co_min": 0.45,
        # rivals of the top face make them co-subjects, no main: faces at least tie_dominance of its area (a
        # handshake: same distance to the camera), or at least co_dominance and within co_margin of its prominence
        "tie_dominance": 0.9,
        "co_dominance": 0.6,
        "co_margin": 0.25,
        # crowd photos: a main face must be this many times larger than the next face
        "crowd_dominance": 2.5,
        # photo type: count the faces at least this share of the largest face's area (and not tiny)
        "type_min_dominance": 0.2,
        "small_group_max": 5,
    },
}


class ModelError(ValueError):
    pass


@dataclass(frozen=True)
class SubjectModel:
    version: str
    intercept: float
    weights: dict[str, float]
    rules: dict[str, float]
    description: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> SubjectModel:
        if not isinstance(data, dict):
            raise ModelError("subject model: expected a JSON object")
        version = data.get("version")
        if not isinstance(version, str) or not version.strip() or len(version) > 64:
            raise ModelError("subject model: 'version' (string, up to 64 characters) is required")
        weights = data.get("weights") or {}
        unknown = set(weights) - set(FEATURES)
        if unknown:
            raise ModelError(f"subject model: unknown feature(s) {sorted(unknown)}")
        rules = dict(DEFAULT_MODEL["rules"])
        extra = set(data.get("rules") or {}) - set(rules)
        if extra:
            raise ModelError(f"subject model: unknown rule(s) {sorted(extra)}")
        rules.update(data.get("rules") or {})
        try:
            w = {f: float(weights.get(f, 0.0)) for f in FEATURES}
            r = {k: float(v) for k, v in rules.items()}
            b = float(data.get("intercept", 0.0))
        except (TypeError, ValueError) as exc:
            raise ModelError("subject model: weights, intercept and rules must be numbers") from exc
        if not all(math.isfinite(v) for v in (*w.values(), *r.values(), b)):
            raise ModelError("subject model: numbers must be finite")
        if not 0 <= r["co_min"] <= r["main_min"] <= 1:
            raise ModelError("subject model: need 0 <= co_min <= main_min <= 1")
        return cls(
            version=version.strip(), intercept=b, weights=w, rules=r, description=str(data.get("description", ""))
        )

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "description": self.description,
            "intercept": round(self.intercept, 6),
            "weights": {f: round(self.weights[f], 6) for f in FEATURES},
            "rules": dict(self.rules),
        }


def load_model(path: str | None = None, text: str | None = None) -> SubjectModel:
    """The model from a JSON file or JSON text; the built-in defaults if neither is given."""
    if path:
        text = Path(path).read_text(encoding="utf-8")
    if text:
        try:
            return SubjectModel.from_dict(json.loads(text))
        except json.JSONDecodeError as exc:
            raise ModelError("subject model: not valid JSON") from exc
    return SubjectModel.from_dict(DEFAULT_MODEL)


# ---------------------------------------------------------------------------------------------------- measurements


REBLUR = 5  # length of the 1-D box filter of the re-blur sharpness measure
FACE_SCALE = 48  # faces are judged for sharpness at this size (pixels), whatever their size in the photo


def _box1d(a: np.ndarray, axis: int, k: int = REBLUR) -> np.ndarray:
    """1-D moving average of length k along an axis (edges repeated), same shape."""
    pad = [(0, 0), (0, 0)]
    pad[axis] = (k // 2 + 1, k // 2)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float32)
    n = c.shape[axis]
    hi = c[k:n] if axis == 0 else c[:, k:n]
    lo = c[: n - k] if axis == 0 else c[:, : n - k]
    return (hi - lo) * (1.0 / k)


def _reblur_maps(g: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per axis: (|difference| of the image, the part of it a re-blur removes); (H-1, W) and (H, W-1) maps."""
    out = []
    for axis in (0, 1):
        d = np.abs(np.diff(g, axis=axis))
        db = np.abs(np.diff(_box1d(g, axis), axis=axis))
        out.append((d, np.maximum(0.0, d - db)))
    return out


def sharpness_of(g: np.ndarray) -> float:
    """Re-blur sharpness of a grey patch, 0..1 (after Crete et al., "The blur effect", 2007).

    Blur the patch again with a short box filter and see how much of its pixel-to-pixel variation disappears: a sharp
    patch loses much of it, an already blurred one hardly any. The ratio does not depend on contrast and depends
    little on content (smooth skin versus glasses and beards), unlike a plain Laplacian variance.
    """
    if g.shape[0] < REBLUR + 2 or g.shape[1] < REBLUR + 2:
        return 0.0
    best = 0.0
    for d, v in _reblur_maps(g):
        total = float(d.sum())
        if total > 1e-6:
            best = max(best, float(v.sum()) / total)
    return best


def _fixed(lum: Image.Image, box: tuple[int, int, int, int], size: int = FACE_SCALE) -> np.ndarray:
    """A region of a grey image resampled to size x size (area-averaged down, bicubic up), as floats 0..1."""
    w, h = box[2] - box[0], box[3] - box[1]
    method = Image.Resampling.BOX if min(w, h) >= size else Image.Resampling.BICUBIC
    return np.asarray(lum.resize((size, size), method, box=box), dtype=np.float32) / 255.0


@dataclass
class _FaceGeom:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    landmarks: list | None = None


@dataclass
class FaceSubject:
    prominence: float
    role: str
    signals: dict = field(default_factory=dict)


@dataclass
class PhotoSubjects:
    faces: list[FaceSubject]
    photo_type: str
    main_face_index: int | None  # index into the faces given to analyse()
    face_count: int
    subject_count: int
    model_version: str


def _crop(gray: np.ndarray, x1: float, y1: float, x2: float, y2: float, inset: float = 0.0) -> np.ndarray:
    h, w = gray.shape
    dx, dy = (x2 - x1) * inset, (y2 - y1) * inset
    left, top = max(0, int(x1 + dx)), max(0, int(y1 + dy))
    right, bottom = min(w, int(math.ceil(x2 - dx))), min(h, int(math.ceil(y2 - dy)))
    if right - left < 2 or bottom - top < 2:
        return np.zeros((0, 0), dtype=np.float32)
    return gray[top:bottom, left:right]


def pose_from_landmarks(lm) -> tuple[float, float]:
    """(yaw, pitch) in degrees, roughly, from 5 points: left eye, right eye, nose, left and right mouth corner.

    The points are first rotated so the eyes are level (roll removed). Yaw: how far the nose tip is off the eyes'
    midpoint (0 = centred; about 45 degrees when it is above one eye). Pitch: where the nose sits between the eye line
    and the mouth line (0 near 55% of the way down).
    """
    p = np.asarray(lm, dtype=np.float64).reshape(5, 2)
    le, re_, nose, lm_, rm = p
    ang = math.atan2(re_[1] - le[1], re_[0] - le[0])
    c, s = math.cos(-ang), math.sin(-ang)
    rot = np.array([[c, -s], [s, c]])
    q = (p - le) @ rot.T
    le, re_, nose, lm_, rm = q
    eye_d = re_[0] - le[0]
    if eye_d <= 1e-6:
        return 90.0, 0.0
    mid = (le[0] + re_[0]) / 2
    # the nose tip sits about 0.9 half-eye-distances in front of the eye plane: its offset from the eyes' midpoint,
    # in projected half-eye-distances, is about 1.1 * tan(yaw)
    r = (nose[0] - mid) / (eye_d / 2)
    yaw = math.degrees(math.atan(r / 1.1))
    eye_y = (le[1] + re_[1]) / 2
    mouth_y = (lm_[1] + rm[1]) / 2
    span = mouth_y - eye_y
    if span <= 1e-6:
        return yaw, 0.0
    t = (nose[1] - eye_y) / span
    pitch = max(-60.0, min(60.0, (t - 0.55) * 150.0))
    return yaw, pitch


def _working(lum: Image.Image, edge: int = 640) -> tuple[np.ndarray, float]:
    """The grey image with its long edge at most ``edge`` (and the scale used): background focus and the face's
    focus relative to it are measured on this one image, at one scale."""
    w, h = lum.size
    scale = min(1.0, edge / max(w, h))
    if scale < 1.0:
        lum = lum.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BOX)
    return np.asarray(lum, dtype=np.float32) / 255.0, scale


class _Maps:
    """The re-blur maps of the working image, for the sharpness of face regions and background tiles."""

    def __init__(self, work: np.ndarray):
        self.maps = _reblur_maps(work)

    def region(self, x1: float, y1: float, x2: float, y2: float) -> float:
        best = 0.0
        for d, v in self.maps:
            h, w = d.shape
            a, b = max(0, min(h, int(y1))), max(0, min(h, math.ceil(y2)))
            c, e = max(0, min(w, int(x1))), max(0, min(w, math.ceil(x2)))
            if b - a < 3 or e - c < 3:
                continue
            total = float(d[a:b, c:e].sum())
            if total > 1e-6:
                best = max(best, float(v[a:b, c:e].sum()) / total)
        return best

    def tiles(self, ny: int, nx: int, tile: int) -> np.ndarray:
        """Sharpness of every tile x tile cell of an ny x nx grid (best axis)."""
        out = np.zeros((ny, nx), dtype=np.float32)
        for d, v in self.maps:
            sd = d[: ny * tile, : nx * tile].reshape(ny, tile, nx, tile).sum(axis=(1, 3))
            sv = v[: ny * tile, : nx * tile].reshape(ny, tile, nx, tile).sum(axis=(1, 3))
            out = np.maximum(out, np.where(sd > 1e-6, sv / np.maximum(sd, 1e-6), 0.0))
        return out


def _background_sharpness(
    work: np.ndarray, maps: _Maps, scale: float, boxes: list[_FaceGeom], tile: int = 24
) -> float | None:
    """Median sharpness of the textured tiles of the working image away from every face (head and shoulders)."""
    h, w = work.shape
    ny, nx = (h - 1) // tile, (w - 1) // tile  # the maps are one pixel shorter along their axis
    if ny < 1 or nx < 1:
        return None
    mask = np.zeros((ny, nx), dtype=bool)
    for b in boxes:
        bw, bh = b.x2 - b.x1, b.y2 - b.y1
        x1, x2 = (b.x1 - bw * 0.5) * scale, (b.x2 + bw * 0.5) * scale
        y1, y2 = (b.y1 - bh * 0.3) * scale, (b.y2 + bh * 1.5) * scale
        r0, r1 = max(0, int(y1 // tile)), max(0, math.ceil(y2 / tile))
        c0, c1 = max(0, int(x1 // tile)), max(0, math.ceil(x2 / tile))
        mask[r0:r1, c0:c1] = True
    std = work[: ny * tile, : nx * tile].reshape(ny, tile, nx, tile).std(axis=(1, 3))
    ok = (~mask) & (std > 0.03)  # textured tiles only: a plain wall says nothing about focus
    if ok.sum() < 4:
        return None
    return float(np.median(maps.tiles(ny, nx, tile)[ok]))


def _exposure(face: np.ndarray) -> tuple[float, float, float]:
    if face.size == 0:
        return 0.0, 0.0, 0.0
    lum = float(face.mean())
    clipped = float(((face <= 0.02) | (face >= 0.98)).mean())
    # well exposed: mean luminance 0.25..0.75; penalise beyond, and clipped pixels
    off = max(0.0, 0.25 - lum, lum - 0.75) / 0.25
    exposure = max(0.0, 1.0 - off) * max(0.0, 1.0 - 2.0 * clipped)
    return lum, clipped, exposure


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def feature_vector(sig: dict) -> list[float]:
    """The model inputs (``FEATURES`` order) from one face's raw signals and its photo's subject count."""
    area = max(1e-6, sig["area"])
    focus = sig.get("focus_bg")
    return [
        min(1.0, max(0.0, (math.log10(area) + 4.0) / 4.0)),
        sig["dominance"],
        1.0 if sig.get("lead") is None else max(-1.0, min(1.0, math.log2(max(1e-6, sig["lead"])) / 3.0)),
        min(1.0, max(0.0, (sig["sharpness"] - 0.2) / 0.5)),
        sig["sharpness_rel"],
        0.0 if focus is None else max(-1.0, min(1.0, focus / 3.0)),
        max(sig["centrality"], sig["thirds"]),
        sig["edge_cut"],
        0.5 if sig.get("frontal") is None else sig["frontal"],
        sig["detection_score"],
        sig["exposure"],
        min(1.0, max(0.0, math.log2(max(1, sig["subject_count"])) / 4.0)),
    ]


def prominence(model: SubjectModel, sig: dict) -> float:
    z = model.intercept + sum(model.weights[f] * x for f, x in zip(FEATURES, feature_vector(sig), strict=True))
    return _sigmoid(z)


def measure(image: Image.Image, faces: list[dict]) -> tuple[list[dict], dict]:
    """Raw signals per face (no model) and per photo. ``faces``: dicts with x1, y1, x2, y2 (pixels of ``image``),
    ``score`` and optionally ``landmarks`` (5 x [x, y])."""
    W, H = image.size
    geoms = [
        _FaceGeom(
            float(f["x1"]),
            float(f["y1"]),
            float(f["x2"]),
            float(f["y2"]),
            float(f.get("score", 0.0)),
            f.get("landmarks"),
        )
        for f in faces
    ]
    if not geoms:
        return [], {"face_count": 0, "background_focus": None}
    grey = image.convert("L")
    gray = np.asarray(grey, dtype=np.float32) / 255.0
    work, scale = _working(grey)
    maps = _Maps(work)
    areas, sharp, sigs, local = [], [], [], []
    for g in geoms:
        cx1, cy1, cx2, cy2 = max(0.0, g.x1), max(0.0, g.y1), min(float(W), g.x2), min(float(H), g.y2)
        raw_area = max(1e-9, (g.x2 - g.x1) * (g.y2 - g.y1))
        vis_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
        areas.append(vis_area / (W * H))
        # sharpness of the face resampled to FACE_SCALE px, so faces of any size are judged at the same scale
        ix, iy = (cx2 - cx1) * 0.1, (cy2 - cy1) * 0.1
        box = (int(cx1 + ix), int(cy1 + iy), math.ceil(cx2 - ix), math.ceil(cy2 - iy))
        s = sharpness_of(_fixed(grey, box)) if box[2] - box[0] >= 2 and box[3] - box[1] >= 2 else 0.0
        sharp.append(s)
        # the same measure on the working image, comparable with the background tiles
        local.append(maps.region((cx1 + ix) * scale, (cy1 + iy) * scale, (cx2 - ix) * scale, (cy2 - iy) * scale))
        lum, clipped, exposure = _exposure(_crop(gray, cx1, cy1, cx2, cy2, inset=0.15))
        if g.landmarks is not None:
            yaw, pitch = pose_from_landmarks(g.landmarks)
            frontal = max(0.0, math.cos(math.radians(yaw))) ** 2 * max(0.0, math.cos(math.radians(pitch)))
            source = "landmarks"
        else:  # no landmarks: pose unknown (the model then uses a neutral 0.5)
            yaw = pitch = frontal = None
            source = None
        cx, cy = (cx1 + cx2) / 2 / W, (cy1 + cy2) / 2 / H
        d_center = math.hypot(cx - 0.5, cy - 0.5) / math.hypot(0.5, 0.5)
        d_thirds = min(math.hypot(cx - tx, cy - ty) for tx in (1 / 3, 2 / 3) for ty in (1 / 3, 2 / 3))
        m = 0.01
        sigs.append(
            {
                "area": round(areas[-1], 6),
                "size": round(math.sqrt(areas[-1]), 4),
                "face_px": int(min(cx2 - cx1, cy2 - cy1)),
                "sharpness": round(s, 4),
                "centrality": round(1.0 - d_center, 4),
                "thirds": round(max(0.0, 1.0 - d_thirds / 0.25), 4),
                "edge_cut": round(max(0.0, 1.0 - vis_area / raw_area), 4),
                "touches_edge": bool(g.x1 <= m * W or g.y1 <= m * H or g.x2 >= (1 - m) * W or g.y2 >= (1 - m) * H),
                "yaw": None if yaw is None else round(yaw, 1),
                "pitch": None if pitch is None else round(pitch, 1),
                "frontal": None if frontal is None else round(frontal, 4),
                "pose_source": source,
                "detection_score": round(g.score, 4),
                "luminance": round(lum, 4),
                "clipped": round(clipped, 4),
                "exposure": round(exposure, 4),
            }
        )
    a_max = max(areas) or 1e-9
    s_max = max(sharp)
    bg = _background_sharpness(work, maps, scale, geoms)
    top = max(range(len(areas)), key=areas.__getitem__)
    second = max((a for j, a in enumerate(areas) if j != top), default=0.0)
    for i, (sig, a, s, loc) in enumerate(zip(sigs, areas, sharp, local, strict=True)):
        sig["dominance"] = round(a / a_max, 4)
        other = second if i == top else a_max
        sig["lead"] = round(min(1000.0, a / other), 4) if other > 0 else None
        sig["sharpness_rel"] = round(s / s_max, 4) if s_max > 0 else 0.0
        fb = None if bg is None or bg <= 0 or loc <= 0 else max(-3.0, min(3.0, math.log2(loc / bg)))
        sig["focus_bg"] = None if fb is None else round(fb, 3)
    return sigs, {"face_count": len(sigs), "background_focus": None if bg is None else round(bg, 4)}


def _tiny(sig: dict, rules: dict) -> bool:
    return sig["size"] < rules["min_size"] or sig["face_px"] < rules["min_face_px"]


def _blurred(sig: dict, rules: dict, n: int) -> bool:
    return sig["sharpness"] < rules["min_sharpness"] or (n > 1 and sig["sharpness_rel"] < rules["min_sharpness_rel"])


def is_gated(sig: dict, rules: dict, n: int) -> bool:
    """Never a subject: tiny, or blurred (absolutely, or next to the photo's sharpest face)."""
    return _tiny(sig, rules) or _blurred(sig, rules, n)


def photo_type_of(sigs: list[dict], rules: dict) -> tuple[str, int]:
    """(photo type, subject count). Faces that could be subjects count: at least type_min_dominance of the largest
    face, neither tiny nor blurred (an out-of-focus foreground or a sea of tiny faces does not make a speaker's photo
    a crowd shot). With none of them, the plain face count decides."""
    n = len(sigs)
    if not n:
        return "none", 0
    count = sum(
        1
        for s in sigs
        if s["dominance"] >= rules["type_min_dominance"] and not _tiny(s, rules) and not _blurred(s, rules, n)
    )
    k = count or n
    kind = "portrait" if k <= 1 else "small_group" if k <= rules["small_group_max"] else "crowd"
    return kind, count


def assign_roles(
    prom: list[float], sigs: list[dict], gated: list[bool], photo_type: str, rules: dict
) -> tuple[list[str], int | None]:
    """(role per face, index of the main face or None) from prominences and the rules. Gated faces (tiny or
    blurred) are always background."""
    n = len(prom)
    roles = ["co_subject" if p >= rules["co_min"] and not g else "background" for p, g in zip(prom, gated, strict=True)]
    if not n:
        return roles, None
    top = max(range(n), key=lambda i: (roles[i] != "background", prom[i]))
    if roles[top] == "background" or prom[top] < rules["main_min"]:
        return roles, None
    # rivals: other faces about as large (a handshake: same distance to the camera), or fairly large and nearly as
    # prominent (two speakers): then there is no single main face, they are co-subjects
    for i in range(n):
        if i == top or roles[i] != "co_subject":
            continue
        dom = sigs[i]["dominance"]
        if dom >= rules["tie_dominance"] or (
            dom >= rules["co_dominance"] and prom[top] - prom[i] <= rules["co_margin"]
        ):
            return roles, None
    if photo_type == "crowd":  # in a crowd only a clearly dominant face is the main one
        second = max((sigs[i]["area"] for i in range(n) if i != top), default=0.0)
        if sigs[top]["area"] < rules["crowd_dominance"] * second:
            return roles, None
    roles[top] = "main"
    return roles, top


def raw_prominences(sigs: list[dict], model: SubjectModel) -> tuple[list[float], list[bool], str, int]:
    """(prominence per face before gating, gated per face, photo type, subject count)."""
    r = model.rules
    photo_type, subject_count = photo_type_of(sigs, r)
    n = len(sigs)
    gated = [is_gated(sig, r, n) for sig in sigs]
    prom = [prominence(model, x) for x in model_inputs(sigs, gated, subject_count)]
    return prom, gated, photo_type, subject_count


def model_inputs(sigs: list[dict], gated: list[bool], subject_count: int) -> list[dict]:
    """The signals as the model sees them: dominance and lead measured against the faces that could be subjects
    (a big but blurred foreground head or a tiny face is no competition), plus the photo's subject count."""
    rivals = [sig["area"] for sig, g in zip(sigs, gated, strict=True) if not g] or [sig["area"] for sig in sigs]
    top = max(rivals)
    out = []
    for sig, g in zip(sigs, gated, strict=True):
        a = sig["area"]
        others = list(rivals)
        if not g and a in others:
            others.remove(a)
        other = max(others, default=0.0)
        out.append(
            {
                **sig,
                "dominance": min(1.0, a / top) if top > 0 else 1.0,
                "lead": (min(1000.0, a / other) if other > 0 else None),
                "subject_count": subject_count,
            }
        )
    return out


def score(sigs: list[dict], model: SubjectModel) -> PhotoSubjects:
    """Prominence and role for every face of one photo from its raw signals (``measure``), with the photo type and
    the main face. Works on stored signals too (calibration)."""
    r = model.rules
    if not sigs:
        return PhotoSubjects([], "none", None, 0, 0, model.version)
    prom, gated, photo_type, subject_count = raw_prominences(sigs, model)
    # a gated face (tiny or blurred) is never a subject: its score is kept below every subject threshold
    shown = [round(min(p, r["co_min"] * 0.5) if g else p, 4) for p, g in zip(prom, gated, strict=True)]
    roles, main = assign_roles(shown, sigs, gated, photo_type, r)
    faces = [FaceSubject(prominence=p, role=role, signals=sig) for p, role, sig in zip(shown, roles, sigs, strict=True)]
    return PhotoSubjects(faces, photo_type, main, len(sigs), subject_count, model.version)


def analyse(image: Image.Image, faces: list[dict], model: SubjectModel) -> PhotoSubjects:
    """Signals, prominence and role for every face of one photo, plus the photo type and the main face.

    ``faces``: dicts with x1, y1, x2, y2 (pixels of ``image``), ``score`` and optionally ``landmarks``."""
    sigs, _ = measure(image, faces)
    return score(sigs, model)
