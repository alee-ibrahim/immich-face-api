import io
import json

import pytest
from PIL import Image

from immich_face_api.config import ConfigError, Settings, parse_api_keys
from immich_face_api.imaging import ImageRejected, prepare, sniff
from immich_face_api.matching import Neighbour, decide, normalise_box
from immich_face_api.ml import MLError, face_entries, parse_faces
from immich_face_api.security import ApiKeys, RateLimiter, hash_key

from .conftest import KEY, synthetic, synthetic_jpeg

LIMITS = dict(max_bytes=15 * 1024 * 1024, max_pixels=50_000_000, max_edge=1440)


# --- config -------------------------------------------------------------------------------------------------
def test_api_keys_must_be_hashes():
    with pytest.raises(ConfigError):
        parse_api_keys(json.dumps({"website": "plain-text-key"}))
    with pytest.raises(ConfigError):
        parse_api_keys("{}")
    with pytest.raises(ConfigError):
        parse_api_keys(json.dumps({"bad name!": hash_key(KEY)}))
    assert parse_api_keys(json.dumps({"website": hash_key(KEY)}))["website"] == bytes.fromhex(hash_key(KEY))


def test_settings_from_env(monkeypatch, tmp_path):
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps({"website": hash_key(KEY)}))
    pw = tmp_path / "pw"
    pw.write_text("s3cret\n")
    monkeypatch.setenv("FACE_API_ML_URL", "http://ml:3003/")
    monkeypatch.setenv("FACE_API_OWNER_ID", "00000000-0000-4000-8000-000000000001")
    monkeypatch.setenv("FACE_API_API_KEYS_FILE", str(keys))
    monkeypatch.setenv("FACE_API_DB_PASSWORD_FILE", str(pw))
    monkeypatch.setenv("FACE_API_MAX_DISTANCE", "0.45")
    s = Settings.from_env()
    assert s.ml_url == "http://ml:3003"
    assert s.db_password == "s3cret"
    assert s.max_distance == 0.45
    assert "s3cret" not in repr(s)
    monkeypatch.setenv("FACE_API_MAX_DISTANCE", "7")
    with pytest.raises(ConfigError):
        Settings.from_env()


def test_settings_requires_owner(monkeypatch):
    monkeypatch.setenv("FACE_API_ML_URL", "http://ml")
    monkeypatch.delenv("FACE_API_OWNER_ID", raising=False)
    monkeypatch.delenv("FACE_API_OWNER_EMAIL", raising=False)
    with pytest.raises(ConfigError):
        Settings.from_env()


# --- security -----------------------------------------------------------------------------------------------
def test_api_key_lookup():
    keys = ApiKeys({"website": bytes.fromhex(hash_key(KEY))})
    assert keys.client_for(f"Bearer {KEY}") == "website"
    assert keys.client_for(f"bearer {KEY}") == "website"
    assert keys.client_for(f"Bearer {KEY}x") is None
    assert keys.client_for(KEY) is None
    assert keys.client_for("Basic abc") is None
    assert keys.client_for(None) is None
    assert keys.client_for("Bearer " + "a" * 1000) is None


def test_rate_limiter():
    now = [0.0]
    rl = RateLimiter(per_minute=60, burst=2, clock=lambda: now[0])
    assert rl.take("a")[0] and rl.take("a")[0]
    allowed, retry = rl.take("a")
    assert not allowed and retry == 1
    assert rl.take("b")[0]  # per client
    now[0] += 1.0
    assert rl.take("a")[0]


# --- imaging ------------------------------------------------------------------------------------------------
def test_sniff():
    assert sniff(synthetic("JPEG")) == "JPEG"
    assert sniff(synthetic("PNG")) == "PNG"
    assert sniff(synthetic("WEBP")) == "WEBP"
    assert sniff(synthetic("GIF")) is None
    assert sniff(b"%PDF-1.7") is None
    assert sniff(b"") is None


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_prepare_formats(fmt):
    p = prepare(synthetic(fmt), **LIMITS)
    assert (p.width, p.height) == (320, 240)
    assert Image.open(io.BytesIO(p.jpeg)).format == "JPEG"


def test_prepare_heic():
    pytest.importorskip("pillow_heif")
    p = prepare(synthetic("HEIF"), **LIMITS)
    assert (p.width, p.height, p.source_format) == (320, 240, "HEIF")


def test_prepare_mpo_camera_jpeg():
    # Many cameras write JPEGs with an embedded second picture (MPO); Pillow opens those as "MPO".
    first = Image.new("RGB", (640, 480), (200, 40, 40))
    second = Image.new("RGB", (160, 120), (40, 200, 40))
    out = io.BytesIO()
    first.save(out, format="MPO", save_all=True, append_images=[second])
    data = out.getvalue()
    assert sniff(data) == "JPEG" and Image.open(io.BytesIO(data)).format == "MPO"
    p = prepare(data, **LIMITS)
    assert (p.width, p.height, p.source_format) == (640, 480, "MPO")


def test_prepare_png_with_alpha():
    p = prepare(synthetic("PNG", mode="RGBA"), **LIMITS)
    assert Image.open(io.BytesIO(p.jpeg)).mode == "RGB"


def test_orientation_applied_and_metadata_stripped():
    data = synthetic_jpeg(800, 600, exif_orientation=6)  # 6 = rotate 90 degrees: upright is portrait
    assert Image.open(io.BytesIO(data)).getexif()
    p = prepare(data, **LIMITS)
    assert (p.width, p.height) == (600, 800)
    out = Image.open(io.BytesIO(p.jpeg))
    assert out.size == (600, 800)
    assert not out.getexif()
    assert "exif" not in out.info and "icc_profile" not in out.info


def test_downscale_to_max_edge():
    p = prepare(synthetic_jpeg(3000, 2000), **LIMITS)
    assert (p.width, p.height) == (3000, 2000)
    assert (p.sent_width, p.sent_height) == (1440, 960)


@pytest.mark.parametrize(
    "data,status",
    [
        (b"", 400),
        (b"hello world, not an image", 415),
        (b"GIF89a" + b"\0" * 100, 415),
        (b"\x89PNG\r\n\x1a\n" + b"garbage" * 20, 400),  # right magic, broken content
        (b"\xff\xd8\xff" + b"\0" * 50, 400),
    ],
)
def test_rejects(data, status):
    with pytest.raises(ImageRejected) as e:
        prepare(data, **LIMITS)
    assert e.value.status == status


def test_rejects_too_big_and_too_many_pixels():
    data = synthetic_jpeg(1000, 1000)
    with pytest.raises(ImageRejected) as e:
        prepare(data, max_bytes=100, max_pixels=50_000_000, max_edge=1440)
    assert e.value.status == 413
    with pytest.raises(ImageRejected) as e:
        prepare(data, max_bytes=10**8, max_pixels=999_999, max_edge=1440)
    assert e.value.status == 413


def test_html_polyglot_with_jpeg_magic_is_rejected():
    with pytest.raises(ImageRejected):
        prepare(b"\xff\xd8\xff<html><script>alert(1)</script></html>", **LIMITS)


# --- matching -----------------------------------------------------------------------------------------------
def n(pid, d, name=None):
    return Neighbour(person_id=pid, name=name if name is not None else f"Name {pid}", distance=d)


def test_nearest_face_wins_over_votes():
    rows = [n("b", 0.40), n("b", 0.41), n("b", 0.42), n("a", 0.30)]
    d = decide(rows, max_distance=0.5, max_candidates=5, ambiguity_margin=0.03)
    assert d.best.person_id == "a" and d.best.votes == 1
    assert [c.person_id for c in d.candidates] == ["a", "b"]
    assert d.candidates[1].votes == 3
    assert not d.ambiguous


def test_threshold_and_unknown():
    d = decide([n("a", 0.51), n("b", 0.9)], max_distance=0.5, max_candidates=5, ambiguity_margin=0.03)
    assert d.best is None and d.candidates == []


def test_ambiguous_and_tiebreak():
    d = decide([n("a", 0.30), n("b", 0.31), n("b", 0.33)], max_distance=0.5, max_candidates=1, ambiguity_margin=0.03)
    assert d.ambiguous and d.best.person_id == "a" and len(d.candidates) == 1
    d = decide([n("a", 0.30), n("b", 0.30), n("b", 0.35)], max_distance=0.5, max_candidates=5, ambiguity_margin=0.0)
    assert d.best.person_id == "b"  # same distance, more votes


def test_confidence():
    d = decide([n("a", 0.25)], max_distance=0.5, max_candidates=5, ambiguity_margin=0.03)
    assert d.best.confidence == 0.75


def test_normalise_box():
    assert normalise_box({"x1": 100, "y1": 50, "x2": 300, "y2": 250}, 1000, 500) == {
        "x": 0.1,
        "y": 0.1,
        "w": 0.2,
        "h": 0.4,
    }
    b = normalise_box({"x1": -10, "y1": -5, "x2": 1200, "y2": 600}, 1000, 500)
    assert b == {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}


# --- ML protocol --------------------------------------------------------------------------------------------
def test_entries_match_immich_request():
    assert json.loads(face_entries("buffalo_l", 0.7)) == {
        "facial-recognition": {
            "detection": {"modelName": "buffalo_l", "options": {"minScore": 0.7}},
            "recognition": {"modelName": "buffalo_l"},
        }
    }


def test_parse_faces():
    payload = {
        "facial-recognition": [
            {"boundingBox": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}, "embedding": "[0.1,0.2]", "score": 0.9},
            {"boundingBox": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}, "embedding": [0.1, 0.2], "score": 0.8},
        ],
        "imageWidth": 10,
        "imageHeight": 10,
    }
    faces = parse_faces(payload)
    assert faces[0].embedding == "[0.1,0.2]" and faces[1].embedding == "[0.1, 0.2]"
    with pytest.raises(MLError):
        parse_faces({"clip": []})


def test_api_reference_covers_every_operation():
    from immich_face_api.apidocs import reference
    from immich_face_api.openapi import OPENAPI_SPEC

    ops = reference()
    documented = {(p, m.upper()) for p, item in OPENAPI_SPEC["paths"].items() for m in item}
    assert {(o["path"], o["method"]) for o in ops} == documented
    identify = next(o for o in ops if o["path"] == "/v1/identify")
    names = {f["name"] for f in identify["responses"][0]["fields"]}
    assert {"faces", "faces[].box", "faces[].box.x", "faces[].candidates[].votes", "image"} <= names
    assert identify["auth"] and not next(o for o in ops if o["path"] == "/healthz")["auth"]
    assert identify["responses"][0]["example"]


def test_inline_code_filter_escapes():
    from immich_face_api.admin import inline_code

    assert str(inline_code("use `X-API-Key` <b>")) == "use <code>X-API-Key</code> &lt;b&gt;"
