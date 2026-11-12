"""Validate and normalise an uploaded image entirely in memory.

Order matters: magic bytes first (cheap, before any decoder runs), then the header-only open with an
explicit format allowlist, the pixel limit (before decoding pixels: decompression bombs), and only then
the full decode. The output is a fresh RGB JPEG without any metadata (EXIF, GPS, XMP, ICC are dropped),
rotated upright according to its EXIF orientation, with its longest edge at most ``max_edge``.
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

from PIL import Image, ImageOps

try:  # HEIC/HEIF support is optional at import time so unit tests run without it
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:  # pragma: no cover
    HEIF_AVAILABLE = False

ALLOWED_FORMATS = ("JPEG", "PNG", "WEBP", "HEIF")
_HEIF_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1"}


class ImageRejected(ValueError):
    """The upload is not an acceptable image. ``status`` is the HTTP status to answer with."""

    def __init__(self, message: str, status: int = 415):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class PreparedImage:
    jpeg: bytes  # what is sent to the ML server
    width: int  # upright size of the original
    height: int
    sent_width: int  # size of the image sent to ML (boxes are relative to it)
    sent_height: int
    source_format: str


def sniff(data: bytes) -> str | None:
    """Format from magic bytes, or None."""
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    if len(data) >= 12 and data[4:8] == b"ftyp" and data[8:12] in _HEIF_BRANDS:
        return "HEIF"
    return None


def prepare(data: bytes, *, max_bytes: int, max_pixels: int, max_edge: int) -> PreparedImage:
    if not data:
        raise ImageRejected("empty body", 400)
    if len(data) > max_bytes:
        raise ImageRejected("image too large", 413)
    kind = sniff(data)
    if kind is None:
        raise ImageRejected("unsupported image type (JPEG, PNG, WebP or HEIC only)")
    if kind == "HEIF" and not HEIF_AVAILABLE:  # pragma: no cover
        raise ImageRejected("HEIC is not supported by this build")

    formats = ["HEIF"] if kind == "HEIF" else [kind]
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            img = Image.open(io.BytesIO(data), formats=formats)
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise ImageRejected("image has too many pixels", 413) from exc
        except Exception as exc:  # noqa: BLE001 - any decoder error means "not a valid image"
            raise ImageRejected("image could not be decoded", 400) from exc

        if img.format not in ALLOWED_FORMATS:
            raise ImageRejected("unsupported image type (JPEG, PNG, WebP or HEIC only)")
        w, h = img.size
        if w < 1 or h < 1:
            raise ImageRejected("image has no pixels", 400)
        if w * h > max_pixels:
            raise ImageRejected("image has too many pixels", 413)

        try:
            if getattr(img, "n_frames", 1) > 1:
                img.seek(0)  # animated WebP/PNG: first frame only
            img.load()
            upright = ImageOps.exif_transpose(img)
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise ImageRejected("image has too many pixels", 413) from exc
        except Exception as exc:  # noqa: BLE001
            raise ImageRejected("image could not be decoded", 400) from exc

    if upright.mode != "RGB":
        if upright.mode in ("RGBA", "LA") or (upright.mode == "P" and "transparency" in upright.info):
            rgba = upright.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.split()[3])
            upright = background
        else:
            upright = upright.convert("RGB")
    width, height = upright.size

    sent = upright
    if max(width, height) > max_edge:
        sent = upright.copy()
        sent.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)

    out = io.BytesIO()
    # A new image object carries no info dict: nothing of the original metadata is written.
    clean = Image.new("RGB", sent.size)
    clean.paste(sent)
    clean.save(out, format="JPEG", quality=92, optimize=False)
    return PreparedImage(
        jpeg=out.getvalue(),
        width=width,
        height=height,
        sent_width=clean.width,
        sent_height=clean.height,
        source_format=img.format or kind,
    )
