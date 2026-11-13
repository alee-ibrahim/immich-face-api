"""Web references: listing pages with one card per person (name + portrait), fetched politely and cached; each
portrait is embedded by Immich's machine-learning server with Immich's own request (see ``..ml``)."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from urllib.parse import quote, unquote, urlencode, urljoin

import httpx
import numpy as np

from ..imaging import ImageRejected, prepare
from ..ml import face_entries, parse_faces
from .match import Reference, normalise, parse_vector
from .settings import LabelerSettings, WebSource

log = logging.getLogger(__name__)

MAX_PAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_BYTES = 20 * 1024 * 1024


@dataclass
class Card:
    source: str
    id: str
    name: str
    alt_names: dict[str, str] = field(default_factory=dict)
    image_url: str | None = None
    details: dict[str, str] = field(default_factory=dict)


class Fetcher:
    """GET with a persistent cache (``store``: http_get/http_put/http_touch). Pages are re-checked after
    ``refresh_hours`` with If-None-Match / If-Modified-Since; images never change their URL, so a cached image is
    used as is. One request at a time, ``delay`` seconds apart."""

    def __init__(self, store, user_agent: str, delay: float, refresh_hours: float, timeout: float):
        self.store = store
        self.delay = delay
        self.refresh = timedelta(hours=refresh_hours)
        self.client = httpx.Client(headers={"user-agent": user_agent}, timeout=timeout, follow_redirects=True)
        self.last = 0.0
        self.requests = 0

    def close(self) -> None:
        self.client.close()

    def _wait(self) -> None:
        pause = self.last + self.delay - time.monotonic()
        if pause > 0:
            time.sleep(pause)
        self.last = time.monotonic()

    def get(self, url: str, *, immutable: bool, limit: int) -> bytes | None:
        cached = self.store.http_get(url)
        if cached and cached["status"] == 200:
            fresh = immutable or datetime.now(UTC) - cached["fetched_at"] < self.refresh
            if fresh:
                return bytes(cached["body"])
        headers = {}
        if cached and cached["status"] == 200:
            if cached.get("etag"):
                headers["if-none-match"] = cached["etag"]
            if cached.get("last_modified"):
                headers["if-modified-since"] = cached["last_modified"]
        self._wait()
        self.requests += 1
        try:
            r = self.client.get(url, headers=headers)
        except httpx.HTTPError as exc:
            log.warning("fetch failed (%s): %s", type(exc).__name__, url)
            return bytes(cached["body"]) if cached and cached["status"] == 200 else None
        if r.status_code == 304 and cached:
            self.store.http_touch(url)
            return bytes(cached["body"])
        if r.status_code != 200 or len(r.content) > limit:
            log.warning("fetch %s: HTTP %s", url, r.status_code)
            return bytes(cached["body"]) if cached and cached["status"] == 200 else None
        self.store.http_put(
            url, 200, r.headers.get("etag"), r.headers.get("last-modified"), r.headers.get("content-type"), r.content
        )
        return r.content


def _words_capitalised(name: str) -> str:
    return " ".join(w[:1].upper() + w[1:] if w.islower() else w for w in name.replace("_", " ").split())


def parse_cards(src: WebSource, pages: dict[str, str]) -> list[Card]:
    """Cards of the primary-language page, with names from the other languages joined by card id (or taken from
    ``alt_name_groups`` of the same card)."""
    cards: dict[str, Card] = {}
    base = {p.lang: p.url for p in src.pages}
    for m in src.regex.finditer(pages.get(src.primary_lang, "")):
        g = {k: html.unescape(v).strip() for k, v in m.groupdict().items() if v is not None}
        if not g.get("id") or g["id"] in cards:
            continue
        name = _words_capitalised(src.name_template.format(**g)) if src.name_template else g.get("name", "")
        if not name:
            continue
        cid = g.pop("id")
        g.pop("name", None)
        image = src.image_template.format(**g) if src.image_template and g.get("image") else g.get("image")
        g.pop("image", None)
        alt = {lang: " ".join(g[grp].split()) for lang, grp in src.alt_name_groups.items() if g.get(grp)}
        for grp in src.alt_name_groups.values():
            g.pop(grp, None)
        if src.licence:
            g["licence"] = src.licence
        cards[cid] = Card(
            source=src.key,
            id=cid,
            name=" ".join(name.split()),
            alt_names=alt,
            image_url=urljoin(base[src.primary_lang], image) if image else None,
            details=g,
        )
    for lang, text in pages.items():
        if lang == src.primary_lang:
            continue
        for m in src.regex.finditer(text):
            cid, name = m.group("id"), m.groupdict().get("name")
            if cid in cards and name:
                cards[cid].alt_names.setdefault(lang, " ".join(html.unescape(name).split()))
    return list(cards.values())


WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
_TAGS = re.compile(r"<[^>]+>")


def wikidata_cards(src: WebSource, fetcher: Fetcher) -> list[Card]:
    """Cards from a Wikidata SPARQL query; each Commons image with its licence and author."""
    body = fetcher.get(
        WIKIDATA_SPARQL + "?" + urlencode({"query": src.query, "format": "json"}), immutable=False, limit=MAX_PAGE_BYTES
    )
    if body is None:
        return []
    cards: dict[str, Card] = {}
    for b in json.loads(body)["results"]["bindings"]:
        qid = b["item"]["value"].rsplit("/", 1)[-1]
        name = b.get("itemLabel", {}).get("value", "")
        img = b.get("image", {}).get("value", "")
        if not img or not name or name == qid or qid in cards:
            continue
        file = unquote(img.rsplit("/", 1)[-1])
        alt = {k[:-5]: v["value"] for k, v in b.items() if k.endswith("Label") and k != "itemLabel" and v.get("value")}
        details = {"wikidata": qid, "commons_file": file}
        if b.get("role", {}).get("value"):
            details["role"] = b["role"]["value"]
        meta = fetcher.get(
            COMMONS_API
            + "?"
            + urlencode(
                {
                    "action": "query",
                    "prop": "imageinfo",
                    "iiprop": "extmetadata",
                    "format": "json",
                    "titles": "File:" + file,
                }
            ),
            immutable=False,
            limit=MAX_PAGE_BYTES,
        )
        if meta:
            for page in json.loads(meta).get("query", {}).get("pages", {}).values():
                em = (page.get("imageinfo") or [{}])[0].get("extmetadata", {})
                details["licence"] = em.get("LicenseShortName", {}).get("value", "unknown")
                details["author"] = _TAGS.sub("", em.get("Artist", {}).get("value", "")).strip()[:200]
        cards[qid] = Card(
            source=src.key,
            id=qid,
            name=name,
            alt_names=alt,
            image_url="https://commons.wikimedia.org/wiki/Special:FilePath/" + quote(file) + "?width=800",
            details=details,
        )
    return list(cards.values())


class Embedder:
    """Portrait -> faces, exactly as the face API does it (same image preparation, same ML request)."""

    def __init__(self, s: LabelerSettings, store):
        self.s = s
        self.store = store
        self.entries = face_entries(s.face_model, s.min_score)
        self.key = f"{s.face_model}|{s.min_score}|{s.max_edge}|png"  # input format is part of the cache key
        self.client = httpx.Client(base_url=s.ml_url, timeout=s.http_timeout)
        self.calls = 0

    def close(self) -> None:
        self.client.close()

    def faces(self, image: bytes) -> list[dict]:
        sha = hashlib.sha256(image).hexdigest()
        cached = self.store.embedding_get(sha, self.key)
        if cached is not None:
            return cached
        try:
            prep = prepare(image, max_bytes=MAX_IMAGE_BYTES, max_pixels=100_000_000, max_edge=self.s.max_edge)
        except ImageRejected:
            faces = []
        else:
            self.calls += 1
            r = self.client.post(
                "/predict",
                data={"entries": self.entries},
                files={"image": ("image.png", prep.encoded, "image/png")},  # exactly what the API sends (lossless)
            )
            if r.status_code != 200:
                raise RuntimeError(f"ML server answered {r.status_code}")
            faces = [
                {
                    "area": max(0.0, (f.box["x2"] - f.box["x1"]) * (f.box["y2"] - f.box["y1"])),
                    "score": f.score,
                    "embedding": f.embedding,
                }
                for f in parse_faces(r.json())
            ]
        self.store.embedding_put(sha, self.key, faces)
        return faces


@dataclass
class WebResult:
    refs: list[Reference]
    cards: int
    no_face: list[str]  # card keys without a usable face
    images: dict[str, bytes]  # reference key -> portrait bytes (for the report)


def web_references(s: LabelerSettings, store, fetcher: Fetcher, embedder: Embedder) -> WebResult:
    refs, no_face, images, n = [], [], {}, 0
    for src in s.sources.web:
        pages = {}
        if src.kind == "wikidata":
            cards = wikidata_cards(src, fetcher)
        else:
            for p in src.pages:
                body = fetcher.get(p.url, immutable=src.immutable, limit=MAX_PAGE_BYTES)
                if body is not None:
                    pages[p.lang] = body.decode("utf-8", errors="replace")
            cards = parse_cards(src, pages)
        n += len(cards)
        if pages and not cards:
            log.warning("web source %s: no cards matched (page layout changed?)", src.key)
        for c in cards:
            key = f"web:{src.key}:{c.id}"
            img = fetcher.get(c.image_url, immutable=True, limit=MAX_IMAGE_BYTES) if c.image_url else None
            faces = embedder.faces(img) if img else []
            if not faces:
                no_face.append(key)
                continue
            faces.sort(key=lambda f: f["area"], reverse=True)
            ambiguous = len(faces) > 1 and faces[1]["area"] * 2 > faces[0]["area"]
            emb = normalise(parse_vector(faces[0]["embedding"]))
            refs.append(
                Reference(
                    key=key,
                    kind="web",
                    source=src.key,
                    name=c.name,
                    centroid=emb,
                    n_photos=1,
                    label=f"{src.label} {c.id}",
                    alt_names=c.alt_names,
                    details=c.details,
                    ambiguous=ambiguous,
                    url=c.image_url,
                    image_key=key,
                )
            )
            images[key] = img
    return WebResult(refs, n, no_face, images)


def stack(refs: list[Reference]) -> np.ndarray:
    return np.stack([r.centroid for r in refs]) if refs else np.zeros((0, 512), dtype=np.float32)
