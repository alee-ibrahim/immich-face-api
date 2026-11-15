"""Purity sweep: find people (clusters) that hold faces of SEVERAL persons and split them, report first.

Immich gives a new face the person of the nearest already-assigned face within ``maxDistance`` (single linkage), so
small or blurry faces can chain other people into a big cluster. The sweep looks at each cluster on its own:

1. sample up to ``sample`` faces and group them into sub-identities (average linkage, cosine ``link``);
2. the biggest sub-identity is the MAIN one (the person the cluster is about);
3. a sub-identity whose centroid is within ``keep`` of the main centroid stays; one further than ``foreign`` is
   FOREIGN; in between it is borderline (left for curators);
4. a foreign sub-identity goes to the nearest OTHER cluster when that cluster's centroid is within ``match``, else
   to a new unnamed person (only when it has at least ``min_new`` sampled faces; smaller ones are borderline).

``purity`` (read-only) writes a report (HTML contact sheet + CSV) and stores the plan (sub-identity centroids).
``purity-apply`` moves faces ONLY for the rows a curator approved in the review CSV (column ``approve`` = yes):
every current face of the person is matched to its nearest stored sub-identity centroid; faces nearest to an
approved foreign sub-identity are moved one by one through the Immich API (``PUT /api/faces/{person}``), each move
audited (labeler_face_move). ``purity-undo --run`` moves them back (only faces still where the run put them).
"""

from __future__ import annotations

import base64
import csv
import html
import io
import logging
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .match import normalise

log = logging.getLogger(__name__)

APPROVE_VALUES = {"yes", "y", "1", "true", "approve", "ok"}
ACTIONS = ("keep", "borderline", "move_existing", "move_new")


@dataclass(frozen=True)
class PurityThresholds:
    link: float = 0.5  # average-linkage cut (cosine distance) for sub-identities
    keep: float = 0.45  # sub-identity centroid within this of the main centroid: same person
    foreign: float = 0.5  # further than this from the main centroid: another person
    match: float = 0.35  # foreign sub-identity -> existing cluster whose centroid is within this
    min_new: int = 3  # sampled faces a foreign sub-identity needs to become a new person
    sample: int = 300  # faces sampled per cluster


@dataclass
class Sub:
    label: int
    idx: np.ndarray  # indices into the sample
    centroid: np.ndarray
    d_main: float = 0.0
    is_main: bool = False
    action: str = "keep"
    dest: str = ""  # existing person id (move_existing); "" for move_new
    dest_name: str = ""
    dest_d: float = 0.0


@dataclass
class ClusterPlan:
    person_id: str
    name: str
    faces: int  # all faces of the person
    sampled: int
    subs: list[Sub] = field(default_factory=list)

    @property
    def main(self) -> Sub:
        return next(s for s in self.subs if s.is_main)

    def share(self, *actions: str) -> float:
        n = sum(len(s.idx) for s in self.subs if s.action in actions)
        return n / self.sampled if self.sampled else 0.0

    def est_faces(self, sub: Sub) -> int:
        return round(len(sub.idx) / self.sampled * self.faces) if self.sampled else 0


def average_linkage(emb: np.ndarray, cut: float) -> np.ndarray:
    """Labels of agglomerative clustering with average linkage on cosine distance, merging while the closest pair of
    groups is at most ``cut`` apart (same result as scikit-learn's AgglomerativeClustering(linkage="average",
    distance_threshold=cut) up to label numbering). O(n^3) in the worst case: meant for a few hundred faces."""
    n = len(emb)
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    e = normalise(emb).astype(np.float64)
    d = 1.0 - e @ e.T
    np.fill_diagonal(d, np.inf)
    size = np.ones(n)
    alive = np.ones(n, dtype=bool)
    parent = np.arange(n)
    while alive.sum() > 1:
        sub = d[np.ix_(alive, alive)]
        k = int(np.argmin(sub))
        ids = np.flatnonzero(alive)
        i, j = ids[k // len(ids)], ids[k % len(ids)]
        if d[i, j] > cut:
            break
        # Lance-Williams update for average linkage: merge j into i
        new = (size[i] * d[i] + size[j] * d[j]) / (size[i] + size[j])
        d[i, :] = new
        d[:, i] = new
        d[i, i] = np.inf
        d[j, :] = np.inf
        d[:, j] = np.inf
        size[i] += size[j]
        alive[j] = False
        parent[parent == j] = i
    _, labels = np.unique(parent, return_inverse=True)
    return labels


def plan_cluster(
    person_id: str,
    name: str,
    faces: int,
    emb: np.ndarray,
    others: np.ndarray,
    other_ids: list[str],
    other_names: list[str],
    t: PurityThresholds,
    skip: int | None = None,
) -> ClusterPlan:
    """Sub-identities of one cluster's sampled faces and what to do with each. ``others`` = normalised centroids of
    the clusters that may receive a foreign sub-identity; ``skip`` = index of the person itself among them."""
    emb = normalise(emb)
    plan = ClusterPlan(person_id, name, faces, len(emb))
    if len(emb) == 0:
        return plan
    labels = average_linkage(emb, t.link)
    subs = []
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        subs.append(Sub(int(lab), idx, normalise(emb[idx].mean(0))))
    subs.sort(key=lambda s: (-len(s.idx), s.label))
    main = subs[0]
    main.is_main = True
    for s in subs:
        s.d_main = 0.0 if s.is_main else float(1.0 - s.centroid @ main.centroid)
        if s.is_main or s.d_main <= t.keep:
            s.action = "keep"
            continue
        if s.d_main <= t.foreign:
            s.action = "borderline"
            continue
        if len(others):
            dd = 1.0 - others @ s.centroid
            if skip is not None:
                dd[skip] = np.inf
            k = int(np.argmin(dd))
            s.dest_d = float(dd[k])
            if dd[k] <= t.match:
                s.action, s.dest, s.dest_name = "move_existing", other_ids[k], other_names[k]
                continue
            s.dest_name = other_names[k]  # nearest, for the report only
        s.action = "move_new" if len(s.idx) >= t.min_new else "borderline"
    plan.subs = subs
    return plan


def assign_to_subs(emb: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest stored sub-identity centroid of each face: (index, distance)."""
    d = 1.0 - normalise(emb) @ normalise(centroids).T
    k = d.argmin(1)
    return k, d[np.arange(len(k)), k]


def order_plans(plans: list[ClusterPlan], top_unnamed: int) -> list[ClusterPlan]:
    """Named clusters with something to move first (most estimated foreign faces first), then the worst unnamed ones
    (highest foreign share), at most ``top_unnamed`` of them. Clusters with nothing foreign are left out."""

    def foreign_faces(p: ClusterPlan) -> int:
        return sum(p.est_faces(s) for s in p.subs if s.action in ("move_existing", "move_new"))

    named = [p for p in plans if p.name and foreign_faces(p)]
    unnamed = [p for p in plans if not p.name and foreign_faces(p)]
    named.sort(key=lambda p: (-foreign_faces(p), p.person_id))
    unnamed.sort(key=lambda p: (-p.share("move_existing", "move_new"), -p.faces, p.person_id))
    return named + unnamed[:top_unnamed]


CSV_FIELDS = [
    "run_id",
    "person_id",
    "name",
    "faces",
    "sampled",
    "foreign_share",
    "sub",
    "sub_sampled",
    "sub_est_faces",
    "d_main",
    "action",
    "dest",
    "dest_name",
    "dest_d",
    "approve",
]


def plans_csv(run_id: str, plans: list[ClusterPlan]) -> str:
    """One row per sub-identity that is not kept (the review file: set ``approve`` to yes on rows to apply)."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS)
    w.writeheader()
    for p in plans:
        for s in p.subs:
            if s.action == "keep":
                continue
            w.writerow(
                {
                    "run_id": run_id,
                    "person_id": p.person_id,
                    "name": p.name,
                    "faces": p.faces,
                    "sampled": p.sampled,
                    "foreign_share": f"{p.share('move_existing', 'move_new'):.3f}",
                    "sub": s.label,
                    "sub_sampled": len(s.idx),
                    "sub_est_faces": p.est_faces(s),
                    "d_main": f"{s.d_main:.3f}",
                    "action": s.action,
                    "dest": s.dest,
                    "dest_name": s.dest_name,
                    "dest_d": f"{s.dest_d:.3f}" if s.dest_d else "",
                    "approve": "",
                }
            )
    return buf.getvalue()


def read_review(text: str) -> dict[tuple[str, str], set[int]]:
    """Approved rows of a review CSV: {(run_id, person_id): {sub labels}}. Only move_existing / move_new rows count."""
    out: dict[tuple[str, str], set[int]] = {}
    for row in csv.DictReader(io.StringIO(text)):
        if (row.get("approve") or "").strip().lower() not in APPROVE_VALUES:
            continue
        if row.get("action") not in ("move_existing", "move_new"):
            continue
        out.setdefault((row["run_id"], row["person_id"]), set()).add(int(row["sub"]))
    return out


def summary(plans_all: list[ClusterPlan], shown: list[ClusterPlan]) -> dict:
    c: Counter = Counter()
    for p in plans_all:
        c["clusters"] += 1
        c["faces"] += p.faces
        foreign = sum(p.est_faces(s) for s in p.subs if s.action in ("move_existing", "move_new"))
        border = sum(p.est_faces(s) for s in p.subs if s.action == "borderline")
        c["foreign_faces_est"] += foreign
        c["borderline_faces_est"] += border
        if foreign:
            c["clusters_with_foreign"] += 1
            c["named_with_foreign" if p.name else "unnamed_with_foreign"] += 1
        share = p.share("move_existing", "move_new")
        for lim in (0.1, 0.3):
            if share > lim:
                c[f"clusters_over_{int(lim * 100)}pct"] += 1
        for s in p.subs:
            if s.action in ("move_existing", "move_new"):
                c[s.action + "_subs"] += 1
    c["shown"] = len(shown)
    return dict(c)


def _img(data: bytes | None) -> str:
    if not data:
        return '<div class="noimg"></div>'
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(data).decode()}">'


def plans_html(run_id: str, plans: list[ClusterPlan], crops: dict[tuple[str, int], list[bytes]], stats: dict, t) -> str:
    """Contact sheet: per cluster the main sub-identity and every non-kept one, a few face crops each."""
    e = html.escape
    out = [
        "<!doctype html><meta charset='utf-8'><title>Purity sweep " + e(run_id) + "</title>",
        "<style>body{font:13px sans-serif;margin:16px}h2{font-size:15px;margin:18px 0 4px}"
        ".row{display:flex;align-items:center;gap:4px;margin:2px 0}.lab{width:330px;font-size:12px}"
        "img,.noimg{width:72px;height:72px;object-fit:cover;background:#eee}.move{color:#a00}.border{color:#a60}"
        ".keep{color:#070}</style>",
        f"<h1>Purity sweep {e(run_id)}</h1>",
        "<p>"
        + e(
            f"link {t.link}, keep <= {t.keep}, foreign > {t.foreign}, existing match <= {t.match}, "
            f"new person >= {t.min_new} sampled faces, {t.sample} faces sampled per cluster. "
            "Review: set approve=yes in the CSV on the rows to apply, then run purity-apply --review <csv>."
        )
        + "</p><pre>"
        + e("\n".join(f"{k}: {v}" for k, v in sorted(stats.items())))
        + "</pre>",
    ]
    for p in plans:
        title = p.name or "(unnamed)"
        out.append(
            f"<h2>{e(title)} &middot; {e(p.person_id)} &middot; {p.faces} faces &middot; "
            f"foreign ~{p.share('move_existing', 'move_new'):.0%}</h2>"
        )
        for s in p.subs:
            if s.action == "keep" and not s.is_main:
                continue
            if s.is_main:
                label, cls = f"MAIN: {len(s.idx)} of {p.sampled} sampled", "keep"
            elif s.action == "move_existing":
                label = f"sub {s.label}: ~{p.est_faces(s)} faces, {s.d_main:.2f} from main -> {s.dest_name or s.dest[:8]} ({s.dest_d:.2f})"
                cls = "move"
            elif s.action == "move_new":
                label, cls = f"sub {s.label}: ~{p.est_faces(s)} faces, {s.d_main:.2f} from main -> new person", "move"
            else:
                label, cls = f"sub {s.label}: ~{p.est_faces(s)} faces, {s.d_main:.2f} from main (borderline)", "border"
            imgs = "".join(_img(b) for b in crops.get((p.person_id, s.label), []))
            out.append(f"<div class='row'><div class='lab {cls}'>{e(label)}</div>{imgs}</div>")
    return "\n".join(out)


def crop_face(preview: bytes, box: tuple[int, int, int, int, int, int], size: int = 72) -> bytes | None:
    """A square JPEG crop around a face box (box in the coordinates of an image of width/height box[4], box[5])."""
    from PIL import Image

    try:
        im = Image.open(io.BytesIO(preview)).convert("RGB")
    except Exception:  # noqa: BLE001 - an unreadable preview only costs the picture
        return None
    x1, y1, x2, y2, w, h = box
    if not w or not h:
        return None
    sx, sy = im.width / w, im.height / h
    bw, bh = x2 - x1, y2 - y1
    m = 0.3
    rect = (
        max(0, int((x1 - m * bw) * sx)),
        max(0, int((y1 - m * bh) * sy)),
        min(im.width, int((x2 + m * bw) * sx)),
        min(im.height, int((y2 + m * bh) * sy)),
    )
    if rect[2] <= rect[0] or rect[3] <= rect[1]:
        return None
    out = io.BytesIO()
    im.crop(rect).resize((size, size)).save(out, "JPEG", quality=70)
    return out.getvalue()
