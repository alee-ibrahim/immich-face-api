import io

import pytest
from PIL import Image

from immich_face_api.config import Settings
from immich_face_api.security import hash_key

KEY = "test-key-" + "x" * 40
OTHER_KEY = "other-key-" + "y" * 40


def synthetic_jpeg(w=800, h=600, exif_orientation=None, **save):
    """A plain gradient: no real person, no faces. Faces come from the mocked ML server."""
    img = Image.linear_gradient("L").resize((w, h)).convert("RGB")
    out = io.BytesIO()
    if exif_orientation:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        exif[0x010F] = "TestCam"  # Make
        save["exif"] = exif.tobytes()
    img.save(out, format="JPEG", **save)
    return out.getvalue()


def synthetic(fmt, w=320, h=240, mode="RGB"):
    color = (10, 120, 200) if mode == "RGB" else (10, 120, 200, 128)
    img = Image.new(mode, (w, h), color)
    out = io.BytesIO()
    img.save(out, format=fmt)
    return out.getvalue()


@pytest.fixture
def settings():
    return Settings(
        ml_url="http://ml.test",
        db_host="db.test",
        owner_id="00000000-0000-4000-8000-000000000001",
        api_keys={"website": bytes.fromhex(hash_key(KEY)), "other": bytes.fromhex(hash_key(OTHER_KEY))},
        rate_limit_per_minute=600,
        rate_limit_burst=50,
        metrics_port=0,
    )
