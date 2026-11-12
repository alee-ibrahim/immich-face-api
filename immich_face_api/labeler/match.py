"""Matching maths, without any I/O: dominant identity of a folder, identities from references, cluster proposals,
merge candidates. All embeddings are L2-normalised float32 rows, so cosine distance = 1 - dot product."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np

from .names import compatible, name_key, name_tokens
from .settings import Thresholds


def normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def parse_vector(text: str) -> np.ndarray:
    """pgvector text form "[1.0,2.0,...]" -> float32 array."""
    return np.array(text.strip()[1:-1].split(","), dtype=np.float32)


@dataclass
class Dominant:
    members: np.ndarray  # indices of the faces of the dominant identity
    centroid: np.ndarray
    coverage: float  # share of the folder's photos (with faces) that show the dominant identity
    second_coverage: float  # same for the next identity
    n_assets: int  # photos with faces


def dominant_identity(emb: np.ndarray, assets: list[str], areas: list[int], t: Thresholds) -> Dominant | None:
    """The person who appears in most photos of a folder. ``emb`` rows are normalised face embeddings, ``assets`` the
    photo of each face, ``areas`` the face box areas (to pick the subject of a single photo)."""
    n = len(assets)
    if n == 0:
        return None
    uniq = sorted(set(assets))
    if n > t.folder_max_faces:  # deterministic sample by photo
        keep_assets = set(sorted(uniq, key=lambda a: hashlib.sha256(a.encode()).hexdigest())[: max(1, len(uniq) // 2)])
        while sum(a in keep_assets for a in assets) > t.folder_max_faces and len(keep_assets) > 1:
            keep_assets = set(sorted(keep_assets)[: len(keep_assets) * 3 // 4])
        idx = np.array([i for i, a in enumerate(assets) if a in keep_assets])
        sub = dominant_identity(emb[idx], [assets[i] for i in idx], [areas[i] for i in idx], t)
        if sub is None:
            return None
        members = idx[sub.members]
        return Dominant(members, sub.centroid, sub.coverage, sub.second_coverage, sub.n_assets)

    m = len(uniq)
    a_index = {a: i for i, a in enumerate(uniq)}
    col = np.array([a_index[a] for a in assets])
    if m == 1:
        order = np.argsort(areas)[::-1]
        if n > 1 and areas[order[1]] * 3 > areas[order[0]]:
            return None  # a group photo on its own: no clear subject
        k = int(order[0])
        return Dominant(np.array([k]), emb[k].copy(), 1.0, 0.0, 1)

    onehot = np.zeros((n, m), dtype=np.float32)
    onehot[np.arange(n), col] = 1.0
    near = (emb @ emb.T) >= (1.0 - t.face_distance)

    def best(mask: np.ndarray) -> tuple[np.ndarray, float]:
        sel = np.where(mask)[0]
        if len(sel) == 0:
            return np.array([], dtype=int), 0.0
        cover = (near[np.ix_(sel, sel)].astype(np.float32) @ onehot[sel]) > 0
        seed = sel[int(cover.sum(1).argmax())]
        c = normalise(emb[sel][near[seed, sel]].mean(0))
        members = sel[(emb[sel] @ c) >= (1.0 - t.face_distance)]
        if len(members) == 0:
            members = np.array([seed])
        return members, len(set(col[members])) / m

    members, cov1 = best(np.ones(n, dtype=bool))
    rest = np.ones(n, dtype=bool)
    rest[members] = False
    _, cov2 = best(rest)
    return Dominant(members, normalise(emb[members].mean(0)), cov1, cov2, m)


@dataclass
class Reference:
    key: str  # unique, e.g. "folder:/photos/people/Jane Doe/" or "web:members:12"
    kind: str  # "folder" | "web"
    name: str
    centroid: np.ndarray
    n_photos: int  # photos behind the reference (folder: photos showing the dominant identity; web: 1)
    label: str = ""
    alt_names: dict[str, str] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)
    coverage: float | None = None
    second_coverage: float | None = None
    ambiguous: bool = False
    face_ids: set[str] = field(default_factory=set)  # the faces of the dominant identity (folder references)
    url: str | None = None
    image_key: str | None = None  # cache key of a portrait (web references), for the report


@dataclass
class Identity:
    key: str
    name: str
    refs: list[Reference]
    centroid: np.ndarray
    alt_names: dict[str, str] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)  # names of joined references that do not fit the chosen name
    homonym: bool = False  # another identity (a different face) has the same name
    loose: bool = False  # the joined references are not all close to each other
    suggested_name: str | None = None  # for homonyms: a distinguishing name

    @property
    def proposed_name(self) -> str:
        return self.suggested_name or self.name

    @property
    def n_photos(self) -> int:
        return sum(r.n_photos for r in self.refs)

    @property
    def kinds(self) -> set[str]:
        return {r.kind for r in self.refs}

    @property
    def ambiguous(self) -> bool:
        return all(r.ambiguous for r in self.refs)


def _find(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def build_identities(refs: list[Reference], t: Thresholds, homonym_detail: str = "") -> list[Identity]:
    """Join references whose centroids are within ``identity_merge_distance`` (or ``same_name_merge_distance`` when
    the names are compatible). The name comes from a web reference when there is one (official spelling), otherwise
    from the fullest folder name. Identities that share a name but not a face are homonyms; with ``homonym_detail``
    they get a suggested name "Name (detail)" from a web card."""
    if not refs:
        return []
    c = np.stack([r.centroid for r in refs])
    d = 1.0 - c @ c.T
    parent = list(range(len(refs)))
    for i in range(len(refs)):
        for j in np.where(d[i, i + 1 :] <= max(t.identity_merge_distance, t.same_name_merge_distance))[0] + i + 1:
            j = int(j)
            if d[i, j] <= t.identity_merge_distance or compatible(refs[i].name, refs[j].name):
                parent[_find(parent, j)] = _find(parent, i)
    groups: dict[int, list[int]] = {}
    for i in range(len(refs)):
        groups.setdefault(_find(parent, i), []).append(i)

    out = []
    for members in groups.values():
        rs = [refs[i] for i in members]
        # official spelling first (web), then the fullest name (most tokens), then the folder with most photos
        rs.sort(key=lambda r: (r.kind != "web", -len(name_tokens(r.name)), -r.n_photos, r.key))
        chosen = rs[0]
        weights = np.array([3.0 if r.kind == "web" else float(min(r.n_photos, 20)) for r in rs], dtype=np.float32)
        centroid = normalise((np.stack([r.centroid for r in rs]) * weights[:, None]).sum(0))
        conflicts = sorted({r.name for r in rs if not compatible(r.name, chosen.name)})
        alt: dict[str, str] = {}
        for r in rs:
            for lang, v in r.alt_names.items():
                alt.setdefault(lang, v)
        sub = d[np.ix_(members, members)]
        out.append(
            Identity(
                key=chosen.key,
                name=chosen.name,
                refs=rs,
                centroid=centroid,
                alt_names=alt,
                conflicts=conflicts,
                loose=bool(sub.max() > max(t.face_distance, t.same_name_merge_distance)),
            )
        )
    by_name: dict[str, list[Identity]] = {}
    for ident in out:
        by_name.setdefault(name_key(ident.name), []).append(ident)
    for same in by_name.values():
        if len(same) > 1:
            for ident in same:
                ident.homonym = True
                detail = next(
                    (r.details.get(homonym_detail) for r in ident.refs if r.details.get(homonym_detail)), None
                )
                if homonym_detail and detail:
                    ident.suggested_name = f"{ident.name} ({detail})"
    out.sort(key=lambda i: (i.name.casefold(), i.key))
    return out


@dataclass
class Cluster:
    person_id: str
    faces: int
    centroid: np.ndarray
    name: str = ""
    hidden: bool = False


@dataclass
class Proposal:
    person_id: str
    faces: int
    identity: Identity
    distance: float
    margin: float  # distance to the next other identity minus distance to the best (1.0 when there is none)
    runner_up: str | None
    consistency: float | None  # share of the cluster's sampled faces within face_distance of the identity
    folder_faces: int  # faces of the cluster inside the identity's reference folders
    score: float
    band: str  # high | medium | low
    why_not_high: list[str] = field(default_factory=list)

    def evidence(self) -> list[str]:
        ev = []
        for r in self.identity.refs:
            if r.kind == "folder":
                cov = f", in {r.coverage:.0%} of its photos" if r.coverage is not None else ""
                amb = " (ambiguous folder)" if r.ambiguous else ""
                ev.append(f"folder {r.label or r.key}: {r.n_photos} photo(s){cov}{amb}")
            else:
                ev.append(f"portrait {r.label or r.key}")
        ev.append(f"centroid distance {self.distance:.3f}, margin {self.margin:.3f}")
        if self.consistency is not None:
            ev.append(f"{self.consistency:.0%} of the cluster's faces match")
        if self.folder_faces:
            ev.append(f"{self.folder_faces} of the cluster's faces are in the reference folder(s)")
        if self.identity.conflicts:
            ev.append("name conflict: " + "; ".join(self.identity.conflicts))
        if self.identity.homonym:
            hint = f"; suggested {self.identity.suggested_name!r}" if self.identity.suggested_name else ""
            ev.append("another person has the same name" + hint)
        return ev


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def propose(
    clusters: list[Cluster],
    identities: list[Identity],
    t: Thresholds,
    samples: dict[str, np.ndarray] | None = None,
    cluster_faces_in_refs: dict[tuple[str, str], int] | None = None,
) -> list[Proposal]:
    """One proposal per cluster (at most), for clusters without a name. ``samples``: person_id -> normalised face
    embeddings of that cluster; ``cluster_faces_in_refs``: (person_id, identity key) -> faces of the cluster that
    are inside the identity's reference folders."""
    todo = [c for c in clusters if not c.name and not c.hidden and c.faces >= t.min_cluster_faces]
    if not todo or not identities:
        return []
    cm = np.stack([c.centroid for c in todo])
    im = np.stack([i.centroid for i in identities])
    dist = 1.0 - cm @ im.T
    samples = samples or {}
    overlap = cluster_faces_in_refs or {}
    out = []
    for ci, c in enumerate(todo):
        order = np.argsort(dist[ci])
        best = int(order[0])
        d1 = float(dist[ci, best])
        if d1 > t.match_max_distance:
            continue
        ident = identities[best]
        d2, runner = 1.0 + d1, None
        if len(order) > 1:  # every other identity is another person (same-name people included)
            j = int(order[1])
            d2, runner = float(dist[ci, j]), identities[j].proposed_name
        margin = min(1.0, d2 - d1)
        cons = None
        if c.person_id in samples and len(samples[c.person_id]):
            cons = float(((samples[c.person_id] @ ident.centroid) >= (1.0 - t.face_distance)).mean())
        in_refs = sum(overlap.get((c.person_id, r.key), 0) for r in ident.refs)

        score = 0.5 * _clamp((t.match_max_distance - d1) / max(1e-6, t.match_max_distance - 0.2))
        score += 0.25 * _clamp(margin / 0.2)
        score += 0.25 * (cons if cons is not None else 0.5)
        if len(ident.kinds) > 1:
            score += 0.05
        if in_refs >= 2:
            score += 0.05
        if ident.conflicts or ident.homonym or ident.ambiguous or ident.loose:
            score -= 0.2

        why = []
        if d1 > t.high_max_distance:
            why.append(f"distance {d1:.3f} > {t.high_max_distance}")
        if margin < t.high_min_margin:
            why.append(f"margin {margin:.3f} < {t.high_min_margin}")
        if cons is None or cons < t.high_min_consistency:
            why.append(f"consistency {'n/a' if cons is None else f'{cons:.2f}'} < {t.high_min_consistency}")
        if c.faces < t.high_min_cluster_faces:
            why.append(f"cluster has {c.faces} face(s) < {t.high_min_cluster_faces}")
        if len([w for w in re.split(r"[\s.]+", ident.name) if len(w) >= 2]) < t.high_min_name_tokens:
            why.append("single-word name")
        if ident.n_photos < t.high_min_reference_faces:
            why.append(f"reference has {ident.n_photos} photo(s) < {t.high_min_reference_faces}")
        if ident.conflicts:
            why.append("name conflict between sources")
        if ident.homonym:
            why.append("same name as another person (needs a distinguishing name)")
        if ident.ambiguous:
            why.append("ambiguous reference folder")
        if ident.loose:
            why.append("joined references are not all close")
        if not why:
            band = "high"
        elif d1 <= t.medium_max_distance and margin >= t.medium_min_margin:
            band = "medium"
        else:
            band = "low"
        out.append(
            Proposal(
                person_id=c.person_id,
                faces=c.faces,
                identity=ident,
                distance=round(d1, 4),
                margin=round(margin, 4),
                runner_up=runner,
                consistency=None if cons is None else round(cons, 3),
                folder_faces=in_refs,
                score=round(_clamp(score), 3),
                band=band,
                why_not_high=why,
            )
        )
    rank = {"high": 0, "medium": 1, "low": 2}
    out.sort(key=lambda p: (rank[p.band], -p.score, p.person_id))
    return out


@dataclass
class MergeCandidate:
    a: str
    b: str
    distance: float
    reason: str
    names: tuple[str, str] = ("", "")
    faces: tuple[int, int] = (0, 0)


def merge_candidates(
    clusters: list[Cluster], proposals: list[Proposal], t: Thresholds, block: int = 2048
) -> list[MergeCandidate]:
    """Pairs of clusters that are probably the same person: centroids within ``merge_distance``, or both proposed
    for the same identity (medium or high). Pairs where both already have (different) names are skipped."""
    out: dict[tuple[str, str], MergeCandidate] = {}
    live = [c for c in clusters if not c.hidden]
    if len(live) > 1:
        cm = np.stack([c.centroid for c in live])
        for s in range(0, len(live), block):
            sims = cm[s : s + block] @ cm.T
            rows, cols = np.where(sims >= 1.0 - t.merge_distance)
            for i, j in zip(rows + s, cols, strict=True):
                if j <= i:
                    continue
                a, b = live[int(i)], live[int(j)]
                if a.name and b.name:
                    continue
                key = (a.person_id, b.person_id)
                out[key] = MergeCandidate(
                    a.person_id,
                    b.person_id,
                    round(float(1.0 - sims[i - s, j]), 4),
                    "close centroids",
                    (a.name, b.name),
                    (a.faces, b.faces),
                )
    by_ident: dict[str, list[Proposal]] = {}
    for p in proposals:
        if p.band in ("high", "medium"):
            by_ident.setdefault(p.identity.key, []).append(p)
    faces = {c.person_id: c for c in clusters}
    for ps in by_ident.values():
        for x in range(len(ps)):
            for y in range(x + 1, len(ps)):
                a, b = sorted((ps[x].person_id, ps[y].person_id))
                if (a, b) in out:
                    out[(a, b)].reason += "; same proposed person"
                    continue
                ca, cb = faces[a], faces[b]
                out[(a, b)] = MergeCandidate(
                    a,
                    b,
                    round(float(1.0 - ca.centroid @ cb.centroid), 4),
                    "same proposed person",
                    (ca.name, cb.name),
                    (ca.faces, cb.faces),
                )
    return sorted(out.values(), key=lambda m: (m.distance, m.a, m.b))
