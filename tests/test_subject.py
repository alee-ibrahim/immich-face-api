"""Main-subject scoring with synthetic images: textured patches stand in for faces (the signals need only pixels and
boxes, no real person). Sharp vs blurred, sizes, positions, pose, photo types, the model file and calibration."""

import json
import math
import time

import numpy as np
import pytest
from PIL import Image, ImageFilter

from immich_face_api import calibrate as cal
from immich_face_api import subject as sj

RNG = np.random.default_rng(42)


def texture(w: int, h: int, seed: int = 0) -> Image.Image:
    """A detailed grey/colour texture (noise at several scales): sharp, like a face in focus at any size."""
    rng = np.random.default_rng(seed)
    base = np.zeros((h, w), dtype=np.float32)
    for scale in (2, 4, 8):
        small = rng.random((max(1, h // scale), max(1, w // scale))).astype(np.float32)
        base += (
            np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR))
            / 255.0
        )
    base = (base / 3 - 0.5) * 1.6 + 0.5
    rgb = np.stack([base, base * 0.9, base * 0.8], axis=-1)
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB")


def scene(w=1440, h=960, faces=(), background="soft", seed=1):
    """A photo: a background (soft = out of focus, sharp = textured) with textured "faces" pasted at boxes.
    faces: (x1, y1, x2, y2, blur_radius)."""
    bg = texture(w, h, seed)
    if background == "soft":
        bg = bg.filter(ImageFilter.GaussianBlur(6))
    for i, (x1, y1, x2, y2, blur) in enumerate(faces):
        patch = texture(int(x2 - x1), int(y2 - y1), seed + 100 + i)
        if blur:
            patch = patch.filter(ImageFilter.GaussianBlur(blur))
        bg.paste(patch, (int(x1), int(y1)))
    boxes = [{"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": 0.85} for x1, y1, x2, y2, _ in faces]
    return bg, boxes


MODEL = sj.load_model()


def roles(result):
    return [f.role for f in result.faces]


# ---------------------------------------------------------------------------------------------------- sharpness


def test_sharpness_drops_with_blur():
    img = texture(200, 200, 3)
    g = lambda im: np.asarray(im.convert("L"), dtype=np.float32) / 255.0  # noqa: E731
    values = [
        sj.sharpness_of(sj._fixed(im.convert("L"), (0, 0, 200, 200)))
        for im in (img, img.filter(ImageFilter.GaussianBlur(3)), img.filter(ImageFilter.GaussianBlur(10)))
    ]
    assert values[0] > values[1] > values[2]
    assert values[0] > 0.45 and values[2] < 0.35
    assert sj.sharpness_of(g(Image.new("L", (40, 40), 128))) == 0.0  # flat: no detail at all
    assert sj.sharpness_of(np.zeros((4, 4), dtype=np.float32)) == 0.0  # too small


def test_sharpness_ignores_contrast():
    img = texture(120, 120, 5).convert("L")
    low = Image.eval(img, lambda v: 100 + v // 4)  # a quarter of the contrast
    a = sj.sharpness_of(sj._fixed(img, (0, 0, 120, 120)))
    b = sj.sharpness_of(sj._fixed(low, (0, 0, 120, 120)))
    assert abs(a - b) < 0.12


def test_blurred_face_is_background_and_sharp_one_main():
    img, boxes = scene(faces=[(200, 200, 420, 480, 0), (800, 220, 1010, 480, 10)])
    res = sj.analyse(img, boxes, MODEL)
    sharp, blurred = res.faces
    assert sharp.signals["sharpness"] > blurred.signals["sharpness"] + 0.15
    assert blurred.signals["sharpness_rel"] < MODEL.rules["min_sharpness_rel"]
    assert blurred.role == "background"
    assert sharp.role == "main" and res.main_face_index == 0


def test_focus_against_background():
    sharp_on_soft, boxes = scene(faces=[(600, 300, 840, 600, 0)], background="soft")
    soft_on_sharp, boxes2 = scene(faces=[(600, 300, 840, 600, 3)], background="sharp")
    a = sj.measure(sharp_on_soft, boxes)[0][0]["focus_bg"]
    b = sj.measure(soft_on_sharp, boxes2)[0][0]["focus_bg"]
    assert a is not None and b is not None
    assert a > 0.3 and b < -0.3


# ---------------------------------------------------------------------------------------------------- size, type


def test_portrait():
    img, boxes = scene(faces=[(520, 180, 920, 700, 0)])
    res = sj.analyse(img, boxes, MODEL)
    assert res.photo_type == "portrait" and res.face_count == 1
    f = res.faces[0]
    assert f.role == "main" and f.prominence > 0.9
    assert f.signals["dominance"] == 1 and f.signals["lead"] is None
    assert f.signals["area"] == pytest.approx(400 * 520 / (1440 * 960), rel=1e-3)
    assert f.signals["size"] == pytest.approx(math.sqrt(f.signals["area"]), rel=1e-3)


def test_tiny_faces_are_background_even_alone_with_a_speaker():
    faces = [(560, 160, 860, 520, 0)] + [(60 + 70 * i, 820, 60 + 70 * i + 30, 856, 0) for i in range(12)]
    img, boxes = scene(faces=faces)
    res = sj.analyse(img, boxes, MODEL)
    assert res.photo_type == "portrait"  # tiny faces do not make it a group photo
    assert res.faces[0].role == "main"
    assert all(f.role == "background" for f in res.faces[1:])
    assert res.faces[1].signals["size"] < MODEL.rules["min_size"]


def test_handshake_two_equal_faces_are_co_subjects():
    img, boxes = scene(faces=[(330, 250, 530, 490, 0), (900, 245, 1102, 487, 0)])
    res = sj.analyse(img, boxes, MODEL)
    assert res.photo_type == "small_group"
    assert roles(res) == ["co_subject", "co_subject"] and res.main_face_index is None


def test_crowd_of_equal_faces_has_no_main():
    faces = [
        (80 + 140 * c, 150 + 220 * r, 80 + 140 * c + 70, 150 + 220 * r + 86, 0) for r in range(3) for c in range(9)
    ]
    img, boxes = scene(faces=faces, background="sharp")
    res = sj.analyse(img, boxes, MODEL)
    assert res.photo_type == "crowd" and res.subject_count == 27
    assert res.main_face_index is None
    assert all(f.role == "background" for f in res.faces)


def test_crowd_with_one_clearly_dominant_face():
    faces = [(80 + 140 * c, 650, 80 + 140 * c + 60, 724, 0) for c in range(8)] + [(560, 80, 860, 440, 0)]
    img, boxes = scene(faces=faces)
    res = sj.analyse(img, boxes, MODEL)
    assert res.main_face_index == 8 and res.faces[8].role == "main"
    assert res.faces[8].signals["lead"] > 10


# ---------------------------------------------------------------------------------------------------- geometry, pose


def test_position_and_edges():
    img, boxes = scene(faces=[(670, 430, 770, 530, 0), (-60, 100, 100, 300, 0), (1300, 50, 1440, 200, 0)])
    sigs, _ = sj.measure(img, boxes)
    centre, cut, edge = sigs
    assert centre["centrality"] > 0.98 and centre["edge_cut"] == 0 and not centre["touches_edge"]
    assert cut["edge_cut"] == pytest.approx(60 / 160, rel=1e-3) and cut["touches_edge"]
    assert edge["touches_edge"] and edge["edge_cut"] == 0
    # a face on a rule-of-thirds point
    sigs, _ = sj.measure(img, [{"x1": 430, "y1": 270, "x2": 530, "y2": 370, "score": 0.9}])
    assert sigs[0]["thirds"] > 0.95


def test_pose_from_landmarks():
    frontal = [[40, 50], [80, 50], [60, 72], [45, 92], [75, 92]]
    yaw, pitch = sj.pose_from_landmarks(frontal)
    assert abs(yaw) < 1 and abs(pitch) < 10
    turned = [[40, 50], [80, 50], [76, 72], [50, 92], [80, 92]]  # nose near the right eye
    yaw, _ = sj.pose_from_landmarks(turned)
    assert yaw > 30
    rolled = [[40, 40], [80, 60], [55, 70], [36, 82], [70, 98]]  # the same face tilted: roll is removed
    yaw, _ = sj.pose_from_landmarks(rolled)
    assert abs(yaw) < 15


def test_landmarks_fill_pose_signals():
    img, boxes = scene(faces=[(520, 180, 920, 700, 0)])
    boxes[0]["landmarks"] = [[640, 380], [800, 380], [720, 470], [660, 560], [780, 560]]
    sig = sj.measure(img, boxes)[0][0]
    assert sig["pose_source"] == "landmarks" and sig["frontal"] > 0.9 and abs(sig["yaw"]) < 5
    img, boxes = scene(faces=[(520, 180, 920, 700, 0)])
    sig = sj.measure(img, boxes)[0][0]
    assert sig["pose_source"] is None and sig["yaw"] is None and sig["frontal"] is None


def test_exposure():
    dark = Image.new("RGB", (400, 400), (2, 2, 2))
    sig = sj.measure(dark, [{"x1": 100, "y1": 100, "x2": 300, "y2": 300, "score": 0.9}])[0][0]
    assert sig["luminance"] < 0.05 and sig["clipped"] > 0.9 and sig["exposure"] == 0
    img, boxes = scene(faces=[(520, 180, 920, 700, 0)])
    assert sj.measure(img, boxes)[0][0]["exposure"] > 0.8


def test_no_faces():
    res = sj.analyse(Image.new("RGB", (100, 100)), [], MODEL)
    assert res.photo_type == "none" and res.faces == [] and res.main_face_index is None


def test_speed_per_face():
    faces = [
        (60 + 45 * (i % 30), 100 + 120 * (i // 30), 60 + 45 * (i % 30) + 40, 100 + 120 * (i // 30) + 48, 0)
        for i in range(60)
    ]
    img, boxes = scene(faces=faces)
    sj.analyse(img, boxes, MODEL)  # warm up
    t = time.perf_counter()
    sj.analyse(img, boxes, MODEL)
    per_face_ms = (time.perf_counter() - t) * 1000 / len(boxes)
    assert per_face_ms < 20


# ---------------------------------------------------------------------------------------------------- model file


def test_model_defaults_and_file(tmp_path):
    m = sj.load_model()
    assert m.version == sj.DEFAULT_MODEL["version"] and set(m.weights) == set(sj.FEATURES)
    d = m.to_dict()
    d["version"] = "custom-1"
    d["weights"]["size"] = 1.0
    d["rules"] = {"main_min": 0.7}
    path = tmp_path / "m.json"
    path.write_text(json.dumps(d))
    m2 = sj.load_model(str(path))
    assert m2.version == "custom-1" and m2.weights["size"] == 1.0
    assert m2.rules["main_min"] == 0.7 and m2.rules["co_min"] == sj.DEFAULT_MODEL["rules"]["co_min"]


@pytest.mark.parametrize(
    "bad",
    [
        {"weights": {}},  # no version
        {"version": "x", "weights": {"nope": 1}},
        {"version": "x", "rules": {"nope": 1}},
        {"version": "x", "rules": {"co_min": 0.9, "main_min": 0.5}},
        {"version": "x", "intercept": "a"},
        [],
    ],
)
def test_model_validation(bad):
    with pytest.raises(sj.ModelError):
        sj.load_model(text=json.dumps(bad))
    with pytest.raises(sj.ModelError):
        sj.load_model(text="{not json")


def test_weights_change_roles():
    img, boxes = scene(faces=[(520, 180, 920, 700, 0)])
    sigs, _ = sj.measure(img, boxes)
    d = sj.DEFAULT_MODEL | {"version": "never", "intercept": -50.0}
    assert sj.score(sigs, sj.SubjectModel.from_dict(d)).faces[0].role == "background"


# ---------------------------------------------------------------------------------------------------- calibration


def synthetic_photo(rng, kind: str) -> dict:
    """Signals of a photo and the labels a curator would give (by construction)."""

    def sig(area, dom, sharp, rel, lead=None, pos=0.6):
        return {
            "area": area,
            "size": math.sqrt(area),
            "face_px": int(math.sqrt(area) * 1200),
            "dominance": dom,
            "lead": lead,
            "sharpness": sharp,
            "sharpness_rel": rel,
            "focus_bg": None,
            "centrality": pos,
            "thirds": pos,
            "edge_cut": 0.0,
            "frontal": None,
            "detection_score": 0.85,
            "exposure": 1.0,
        }

    j = lambda: float(rng.uniform(-0.05, 0.05))  # noqa: E731
    if kind == "portrait":
        return {"signals": [sig(0.08 + j() / 5, 1, 0.6 + j(), 1)], "labels": ["main"]}
    if kind == "speaker":
        s = [sig(0.03 + j() / 10, 1, 0.6 + j(), 1, lead=8)] + [sig(0.002, 0.07, 0.5, 0.8, lead=0.07) for _ in range(5)]
        return {"signals": s, "labels": ["main"] + ["background"] * 5}
    if kind == "pair":
        return {
            "signals": [sig(0.02, 1, 0.6, 1, lead=1.05), sig(0.019, 0.95, 0.58 + j(), 0.97, lead=0.95)],
            "labels": ["co_subject"] * 2,
        }
    s = [sig(0.003 + j() / 100, 0.9, 0.55, 0.9, lead=1.1) for _ in range(10)]
    return {"signals": s, "labels": ["background"] * 10}


def test_calibrate_fits_and_reports(tmp_path):
    rng = np.random.default_rng(1)
    photos = [synthetic_photo(rng, k) for k in ("portrait", "speaker", "pair", "crowd") * 12]
    assert cal.check_enough(photos) is None
    x, y = cal.design(photos, MODEL.rules)
    assert x.shape == (sum(len(p["labels"]) for p in photos), len(sj.FEATURES))
    w, b = cal.fit_logistic(x, y)
    assert w[sj.FEATURES.index("size")] > 0
    model, fitted = cal.fit(MODEL, photos, l2=1.0, version="cal-test")
    assert model.version == "cal-test"
    main_p, main_r, _ = fitted.main()
    assert main_p > 0.9 and main_r > 0.9
    cv = cal.cross_validate(MODEL, photos, folds=4, l2=1.0)
    assert cv.main()[2] > 0.85 and cv.faces == x.shape[0]
    # the command: from a JSONL file to a model file the API can load
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(json.dumps(p) for p in photos))
    out = tmp_path / "model.json"
    assert cal.main(["--labels", str(labels), "--out", str(out), "--version", "cal-cli", "--folds", "3"]) == 0
    loaded = sj.load_model(str(out))
    assert loaded.version == "cal-cli"
    meta = json.loads(out.read_text())["calibration"]
    assert meta["photos"] == len(photos) and "cross_validated" in meta


def test_calibrate_refuses_few_labels(tmp_path):
    rng = np.random.default_rng(2)
    labels = tmp_path / "few.jsonl"
    labels.write_text("\n".join(json.dumps(synthetic_photo(rng, "portrait")) for _ in range(3)))
    assert cal.main(["--labels", str(labels), "--out", str(tmp_path / "m.json")]) == 2
    assert cal.main(["--labels", str(labels)]) == 0  # report only
