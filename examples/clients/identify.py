"""Minimal Python client (server side). Needs: pip install httpx

    FACE_API_URL=https://faces.example.org FACE_API_KEY_FILE=client.key python identify.py photo.jpg

Add your proxy's headers (e.g. an access-service token) to EXTRA_HEADERS if it needs them.
"""

import os
import sys
import time
from pathlib import Path

import httpx

URL = os.environ["FACE_API_URL"].rstrip("/")
KEY = Path(os.environ["FACE_API_KEY_FILE"]).read_text().strip()  # never put the key on the command line
EXTRA_HEADERS: dict[str, str] = {}


def identify(path: str, candidates: int = 3, attempts: int = 3) -> dict:
    with httpx.Client(timeout=120, headers={"Authorization": f"Bearer {KEY}", **EXTRA_HEADERS}) as client:
        for attempt in range(1, attempts + 1):
            with open(path, "rb") as fh:
                r = client.post(f"{URL}/v1/identify", params={"candidates": candidates}, files={"image": fh})
            if r.status_code in (429, 502, 503) and attempt < attempts:
                time.sleep(int(r.headers.get("Retry-After", "5")))
                continue
            r.raise_for_status()
            return r.json()
    raise RuntimeError("unreachable")


if __name__ == "__main__":
    result = identify(sys.argv[1])
    w, h = result["image"]["w"], result["image"]["h"]
    for face in result["faces"]:
        b = face["box"]
        px = (round(b["x"] * w), round(b["y"] * h), round(b["w"] * w), round(b["h"] * h))
        label = face["name"] or "unknown"
        if face["ambiguous"]:
            label += " (ambiguous: " + ", ".join(c["name"] or "?" for c in face.get("candidates", [])) + ")"
        print(f"{px}  {label}  distance={face['distance']}")
