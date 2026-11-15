"""Main-subject scoring through the API (fake ML and database): identify, verify, the modes and the model file."""

import dataclasses
import io
import json

import httpx
import pytest
from PIL import ImageFilter

from immich_face_api.app import create_app
from immich_face_api.config import ConfigError
from immich_face_api.ml import DetectedFace, attach_landmarks, detection_entries, parse_landmarks

from .conftest import KEY
from .test_app import FakeDB, FakeML
from .test_changes import VFACES, VerifyDB
from .test_subject import texture

pytestmark = pytest.mark.asyncio
AUTH = {"Authorization": f"Bearer {KEY}"}


def photo(sharp_left: bool = True) -> bytes:
    """800x600: textured "faces" at the VFACES boxes; the right one blurred (out of focus) when sharp_left."""
    img = texture(800, 600, 9).filter(ImageFilter.GaussianBlur(5))
    left = texture(100, 120, 10)
    right = texture(100, 120, 11)
    if sharp_left:
        right = right.filter(ImageFilter.GaussianBlur(12))
    else:
        left = left.filter(ImageFilter.GaussianBlur(12))
    img.paste(left, (100, 100))
    img.paste(right, (600, 100))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def app_with(settings, faces=VFACES, db=None, **kw):
    app = create_app(dataclasses.replace(settings, face_cache=False, **kw))
    st = app.state.face
    st.ml, st.db, st.ready = FakeML(faces), db or VerifyDB(), True
    return app, httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def test_identify_scores_subjects_by_default(settings):
    app, c = app_with(settings)
    async with c:
        r = await c.post("/v1/identify", headers=AUTH, content=photo())
        assert r.status_code == 200, r.text
        body = r.json()
        left, right = body["faces"]  # sorted left to right
        assert left["role"] == "main" and right["role"] == "background"
        assert body["main_face_index"] == 0 and body["photo_type"] == "portrait"  # the blurred face does not count
        assert body["subject_model_version"] == "default-1"
        assert 0 <= right["prominence"] < left["prominence"] <= 1
        sig = left["subject_signals"]
        for key in (
            "area",
            "size",
            "dominance",
            "lead",
            "sharpness",
            "sharpness_rel",
            "focus_bg",
            "centrality",
            "thirds",
            "edge_cut",
            "touches_edge",
            "yaw",
            "pitch",
            "frontal",
            "detection_score",
            "exposure",
        ):
            assert key in sig
        assert sig["sharpness_rel"] == 1 and right["subject_signals"]["sharpness_rel"] < 0.55

        # opt out per request
        r = await c.post("/v1/identify?subject=0", headers=AUTH, content=photo())
        body = r.json()
        assert "photo_type" not in body and "role" not in body["faces"][0]
        assert (await c.post("/v1/identify?subject=maybe", headers=AUTH, content=photo())).status_code == 400


async def test_main_face_index_follows_sorted_faces(settings):
    app, c = app_with(settings)  # VFACES lists the right face first; the answer sorts left to right
    async with c:
        body = (await c.post("/v1/identify", headers=AUTH, content=photo(sharp_left=False))).json()
        assert body["faces"][1]["role"] == "main" and body["main_face_index"] == 1


async def test_modes(settings):
    app, c = app_with(settings, subject_scoring="request")
    async with c:
        body = (await c.post("/v1/identify", headers=AUTH, content=photo())).json()
        assert "role" not in body["faces"][0]
        body = (await c.post("/v1/identify?subject=1", headers=AUTH, content=photo())).json()
        assert body["faces"][0]["role"] == "main"
    app, c = app_with(settings, subject_scoring="off")
    async with c:
        body = (await c.post("/v1/identify", headers=AUTH, content=photo())).json()
        assert "role" not in body["faces"][0]
        r = await c.post("/v1/identify?subject=1", headers=AUTH, content=photo())
        assert r.status_code == 400 and "off" in r.json()["error"]["message"]


async def test_verify_is_main_subject(settings):
    app, c = app_with(settings)
    db = app.state.face.db
    async with c:
        # without subject=1 verify answers as before
        r = await c.post(
            "/v1/verify", headers=AUTH, files={"image": ("x.png", photo(), "image/png")}, data={"person_id": [db.P1]}
        )
        assert "is_main_subject" not in r.json()["claims"][0]
        r = await c.post(
            "/v1/verify",
            headers=AUTH,
            files={"image": ("x.png", photo(), "image/png")},
            data={"person_id": [db.P1, db.P2], "subject": "1"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        by = {cl["person_id"]: cl for cl in body["claims"]}
        assert by[db.P1]["verdict"] == "match" and by[db.P1]["is_main_subject"] is True
        assert by[db.P1]["role"] == "main" and by[db.P1]["prominence"] > 0.5
        assert by[db.P2]["is_main_subject"] is False and by[db.P2]["role"] == "background"
        assert body["main_face_index"] == 0 and body["faces"][0]["role"] == "main"

        # batch: per item
        import base64

        img = base64.b64encode(photo()).decode()
        r = await c.post(
            "/v1/verify/batch",
            headers=AUTH,
            json={
                "items": [{"image": img, "person_ids": [db.P1], "subject": True}, {"image": img, "person_ids": [db.P1]}]
            },
        )
        a, b = r.json()["items"]
        assert a["claims"][0]["is_main_subject"] is True and "is_main_subject" not in b["claims"][0]


async def test_landmarks_mode_uses_detection_only_call(settings):
    class LandmarkML(FakeML):
        async def detect(self, image, landmarks=False):
            self.landmarks = landmarks
            faces = await super().detect(image)
            lm = [
                [[640, 140], [670, 140], [655, 160], [643, 185], [667, 185]],
                [[140, 140], [170, 140], [155, 160], [143, 185], [167, 185]],
            ]
            return attach_landmarks(faces, [(f.box, m) for f, m in zip(faces, lm, strict=True)]) if landmarks else faces

    app, c = app_with(settings, subject_landmarks=True)
    app.state.face.ml = LandmarkML(VFACES)
    async with c:
        body = (await c.post("/v1/identify", headers=AUTH, content=photo())).json()
        assert app.state.face.ml.landmarks is True
        sig = body["faces"][0]["subject_signals"]
        assert sig["pose_source"] == "landmarks" and abs(sig["yaw"]) < 5 and sig["frontal"] > 0.8


async def test_ml_landmark_parsing():
    payload = {
        "facial-recognition": {
            "boxes": [[10, 20, 50, 70]],
            "scores": [0.9],
            "landmarks": [[[20, 35], [40, 35], [30, 45], [22, 58], [38, 58]]],
        }
    }
    found = parse_landmarks(payload)
    assert found[0][0] == {"x1": 10.0, "y1": 20.0, "x2": 50.0, "y2": 70.0}
    faces = [
        DetectedFace(box={"x1": 10.4, "y1": 20, "x2": 50, "y2": 70}, score=0.9, embedding="[0]"),
        DetectedFace(box={"x1": 300, "y1": 20, "x2": 350, "y2": 70}, score=0.9, embedding="[0]"),
    ]
    a, b = attach_landmarks(faces, found)
    assert a.landmarks[2] == [30.0, 45.0] and b.landmarks is None
    assert json.loads(detection_entries("buffalo_l", 0.7)) == {
        "facial-recognition": {"detection": {"modelName": "buffalo_l", "options": {"minScore": 0.7}}}
    }


async def test_model_file_and_bad_model(settings, tmp_path):
    from immich_face_api.subject import DEFAULT_MODEL

    path = tmp_path / "m.json"
    path.write_text(json.dumps({**DEFAULT_MODEL, "version": "cal-test-7"}))
    app, c = app_with(settings, subject_model_file=str(path))
    async with c:
        body = (await c.post("/v1/identify", headers=AUTH, content=photo())).json()
        assert body["subject_model_version"] == "cal-test-7"
    with pytest.raises(ConfigError):
        create_app(dataclasses.replace(settings, subject_model_json='{"version": "x", "weights": {"bogus": 1}}'))
    with pytest.raises(ConfigError):
        create_app(dataclasses.replace(settings, subject_model_file=str(tmp_path / "missing.json")))


async def test_no_faces(settings):
    app, c = app_with(settings, faces=[], db=FakeDB([]))
    async with c:
        body = (await c.post("/v1/identify", headers=AUTH, content=photo())).json()
        assert body["faces"] == [] and body["photo_type"] == "none" and body["main_face_index"] is None
