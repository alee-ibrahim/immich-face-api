"""Folder references: photos grouped by the person-named folder they are in; the dominant identity of each folder
becomes a reference, built from the embeddings Immich already stored for those photos."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .immich import FolderFace
from .match import Reference, dominant_identity
from .names import clean_name
from .settings import SourcesConfig


@dataclass
class FolderResult:
    refs: list[Reference]
    folders: int  # name folders that had at least one face
    rejected: dict[str, str]  # folder -> reason (no dominant person, ...)
    face_person: dict[str, str]  # face id -> person id (clustered folder faces only), for overlap evidence


def group_faces(cfg: SourcesConfig, faces_by_root: dict[str, list[FolderFace]]):
    """-> {folder key: (name, label, [faces])}. A face can count for more than one rule; each rule's match is its
    own folder."""
    groups: dict[str, tuple[str, str, list[FolderFace]]] = {}
    for rule in cfg.folders:
        for f in faces_by_root.get(rule.root, []):
            if not f.path.startswith(rule.root):
                continue
            m = rule.regex.match(f.path[len(rule.root) :])
            if not m or "/" not in m.group(0):
                continue  # the name must be a folder, not a file
            name = clean_name(m.group("name"), cfg.names)
            if not name:
                continue
            key = rule.root + m.group(0)
            if key not in groups:
                groups[key] = (name, rule.label, [])
            groups[key][2].append(f)
    return groups


def folder_references(cfg: SourcesConfig, faces_by_root: dict[str, list[FolderFace]]) -> FolderResult:
    t = cfg.thresholds
    refs, rejected, face_person = [], {}, {}
    groups = group_faces(cfg, faces_by_root)
    for key, (name, label, faces) in sorted(groups.items()):
        seen = set()
        uniq = [f for f in faces if not (f.face_id in seen or seen.add(f.face_id))]
        for f in uniq:
            if f.person_id:
                face_person[f.face_id] = f.person_id
        dom = dominant_identity(
            np.stack([f.embedding for f in uniq]), [f.asset_id for f in uniq], [f.area for f in uniq], t
        )
        if dom is None:
            rejected[key] = "no clear subject"
            continue
        if dom.coverage < t.folder_min_coverage:
            rejected[key] = f"no dominant person (best in {dom.coverage:.0%} of photos)"
            continue
        members = [uniq[i] for i in dom.members]
        ambiguous = dom.n_assets > 1 and dom.second_coverage >= t.folder_ambiguity_ratio * dom.coverage
        refs.append(
            Reference(
                key="folder:" + key,
                kind="folder",
                name=name,
                centroid=dom.centroid,
                n_photos=len({f.asset_id for f in members}),
                label=f"{label}: {key}" if label else key,
                coverage=round(dom.coverage, 3),
                second_coverage=round(dom.second_coverage, 3),
                ambiguous=ambiguous,
                face_ids={f.face_id for f in members},
            )
        )
    return FolderResult(refs, len(groups), rejected, face_person)


def overlap(refs: list[Reference], face_person: dict[str, str]) -> dict[tuple[str, str], int]:
    """(person id, reference key) -> faces of that person (cluster) that are in the reference's dominant identity."""
    out: Counter = Counter()
    for r in refs:
        for fid in r.face_ids:
            pid = face_person.get(fid)
            if pid:
                out[(pid, r.key)] += 1
    return dict(out)
