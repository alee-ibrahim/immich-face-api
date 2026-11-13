"""People labeler: pure logic with synthetic embeddings, and the write path with fake Immich/store objects."""

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from immich_face_api.labeler import cli
from immich_face_api.labeler.folders import folder_references
from immich_face_api.labeler.immich import FolderFace
from immich_face_api.labeler.match import (
    Cluster,
    Reference,
    build_identities,
    dominant_identity,
    merge_candidates,
    normalise,
    parse_vector,
    propose,
)
from immich_face_api.labeler.names import clean_name, compatible, name_key
from immich_face_api.labeler.settings import ConfigError, NameRules, SourcesConfig, Thresholds, WebSource
from immich_face_api.labeler.web import parse_cards

RNG = np.random.default_rng(7)
DIM = 512


def person():
    return normalise(RNG.normal(size=DIM))


def face_of(p, noise=0.025):
    """A face of person p. With 512 dims, noise 0.025 per dim keeps faces at cosine distance ~0.25 of p."""
    return normalise(p + RNG.normal(scale=noise, size=DIM))


T = Thresholds()


# --- names


RULES = NameRules(
    strip_patterns=[r"^\s*\d+\s*[-.):]*\s*", r"\s*\([^)]*\)?\s*$", r"^MP\s+", r"\s+(?:retake|edited)\s*$"],
    exclude=["new folder", "staff"],
    exclude_patterns=[r"^\d"],
)


@pytest.mark.parametrize(
    "raw,want",
    [
        ("12 - Jane Doe", "Jane Doe"),
        ("07.john smith", "John Smith"),
        ("MP JOHN SMITH", "John Smith"),
        ("Jane Doe ( North District)", "Jane Doe"),
        ("Jane Doe Retake", "Jane Doe"),
        ("New folder", None),
        ("109", None),
        ("Staff", None),
        ("Al", None),
    ],
)
def test_clean_name(raw, want):
    assert clean_name(raw, RULES) == want


def test_compatible_and_homonym_keys():
    assert compatible("Jabir", "Abdulla Jabir")
    assert compatible("Ibrahim Suja", "Ibrahim Shujau")
    assert compatible("Rasheedh Ali", "Rasheed Ali")
    assert not compatible("Ali Rifau", "Abdulla Rifau")
    assert compatible("Adbul Sattar", "Abdul Sattar")  # one typo per word
    assert compatible("Jane AbdulRahmaan", "Jane Abdul Rahman")  # joined words
    assert not compatible("Ali Riza", "Ali Rifau")
    assert clean_name("jane doe", RULES) == "Jane Doe" and clean_name("Jane doe", RULES) == "Jane Doe"
    assert clean_name("Jane McDoe", RULES) == "Jane McDoe"
    assert name_key("Mohamed Rasheed") == name_key("Mohamed Rashid")
    assert name_key("Hassan Waheed") != name_key("Hussain Waheed")
    assert name_key("Abdulla Ahmed") != name_key("Ahmed Abdulla")


# --- config


def test_config_rejects_unknown_keys_and_bad_patterns():
    with pytest.raises(ConfigError):
        SourcesConfig.from_dict({"folderz": []})
    with pytest.raises(ConfigError):
        SourcesConfig.from_dict({"folders": [{"root": "/p/", "pattern": "(?P<nom>[^/]+)/"}]})
    with pytest.raises(ConfigError):
        SourcesConfig.from_dict({"thresholds": {"nope": 1}})
    cfg = SourcesConfig.from_dict({"folders": [{"root": "/p/", "pattern": "(?P<name>[^/]+)/"}], "thresholds": {}})
    assert cfg.folders[0].regex.match("Jane/x.jpg")["name"] == "Jane"


def test_vector_parsing():
    assert parse_vector("[1,2.5,-3]").tolist() == [1.0, 2.5, -3.0]


# --- dominant identity


def test_dominant_identity_portrait_folder_and_ambiguity():
    a, b, c = person(), person(), person()
    # 10 photos: a in all of them, b in 3, c in 1
    emb, assets, areas = [], [], []
    for i in range(10):
        emb.append(face_of(a))
        assets.append(f"p{i}")
        areas.append(900)
        if i < 3:
            emb.append(face_of(b))
            assets.append(f"p{i}")
            areas.append(400)
    emb.append(face_of(c))
    assets.append("p9")
    areas.append(100)
    d = dominant_identity(np.stack(emb), assets, areas, T)
    assert d.coverage == 1.0 and d.second_coverage == pytest.approx(0.3)
    assert 1.0 - float(d.centroid @ a) < 0.1

    # two people in every photo: ambiguous
    emb2 = [face_of(x) for _ in range(5) for x in (a, b)]
    assets2 = [f"q{i}" for i in range(5) for _ in range(2)]
    d2 = dominant_identity(np.stack(emb2), assets2, [500] * 10, T)
    assert d2.coverage == 1.0 and d2.second_coverage == 1.0


def test_dominant_identity_single_group_photo_has_no_subject():
    a, b = person(), person()
    assert dominant_identity(np.stack([face_of(a), face_of(b)]), ["x", "x"], [500, 450], T) is None
    d = dominant_identity(np.stack([face_of(a), face_of(b)]), ["x", "x"], [5000, 450], T)
    assert d is not None and d.coverage == 1.0


def test_folder_references_from_rules():
    a = person()
    cfg = SourcesConfig.from_dict(
        {"folders": [{"root": "/photos/", "pattern": r"people/(?P<name>[^/]+)/", "label": "people"}], "names": {}}
    )
    faces = [
        FolderFace(f"f{i}", f"a{i}", f"/photos/people/Jane Doe/sub/{i}.jpg", "P1" if i < 3 else None, 900, face_of(a))
        for i in range(4)
    ]
    faces.append(FolderFace("x", "ax", "/photos/people/loose.jpg", None, 900, face_of(a)))  # a file, not a folder
    res = folder_references(cfg, {"/photos/": faces})
    assert len(res.refs) == 1
    r = res.refs[0]
    assert r.name == "Jane Doe" and r.n_photos == 4 and r.key == "folder:/photos/people/Jane Doe/"
    assert res.face_person == {"f0": "P1", "f1": "P1", "f2": "P1"}


# --- identities, proposals, merges


def ref(key, kind, name, p, n=5, **kw):
    return Reference(
        key=key,
        kind=kind,
        name=name,
        centroid=normalise(np.mean([face_of(p) for _ in range(n)], 0)),
        n_photos=n if kind == "folder" else 1,
        **kw,
    )


def test_identities_join_by_face_and_prefer_web_spelling():
    a, b = person(), person()
    refs = [
        ref("folder:1", "folder", "Jabir", a, 8),
        ref("web:s:1", "web", "Abdulla Jabir", a, 1, alt_names={"dv": "x"}),
        ref("folder:2", "folder", "Jane Roe", b),
    ]
    ids = build_identities(refs, T)
    assert len(ids) == 2
    joined = next(i for i in ids if len(i.refs) == 2)
    assert joined.name == "Abdulla Jabir" and joined.alt_names == {"dv": "x"} and not joined.conflicts


def test_identities_conflict_and_homonym():
    a, b = person(), person()
    ids = build_identities([ref("folder:1", "folder", "Ali Rifau", a), ref("web:s:1", "web", "Abdulla Rifau", a, 1)], T)
    assert len(ids) == 1 and ids[0].conflicts == ["Ali Rifau"]
    ids = build_identities(
        [
            ref("web:s:1", "web", "Mohamed Rasheed", a, 1, details={"area": "North"}),
            ref("web:s:2", "web", "Mohamed Rasheedh", b, 1, details={"area": "South"}),
        ],
        T,
        homonym_detail="area",
    )
    assert len(ids) == 2 and all(i.homonym for i in ids)
    assert sorted(i.proposed_name for i in ids) == ["Mohamed Rasheed (North)", "Mohamed Rasheedh (South)"]


def test_propose_bands_and_skips():
    a, b, c = person(), person(), person()
    ids = build_identities(
        [
            ref("folder:a", "folder", "Jane Doe", a, 10),
            ref("web:s:1", "web", "John Roe", b, 1),
            ref("folder:c", "folder", "Single", c, 10),
        ],
        T,
    )
    fa = np.stack([face_of(a) for _ in range(6)])
    fc = np.stack([face_of(c) for _ in range(6)])
    clusters = [
        Cluster("A", 6, normalise(fa.mean(0))),
        Cluster("A2", 6, normalise(fa.mean(0)), name="Already Named"),
        Cluster("H", 6, normalise(fa.mean(0)), hidden=True),
        Cluster("C", 6, normalise(fc.mean(0))),
        Cluster("Z", 6, person()),  # nobody we know
    ]
    props = propose(clusters, ids, T, {"A": fa, "C": fc}, {("A", "folder:a"): 3})
    by = {p.person_id: p for p in props}
    assert set(by) == {"A", "C"}
    assert by["A"].band == "high", by["A"].why_not_high
    assert by["A"].identity.name == "Jane Doe" and by["A"].folder_faces == 3
    assert by["C"].band != "high" and "single-word name" in by["C"].why_not_high
    assert any("folder" in e for e in by["A"].evidence())


def test_no_consistency_sample_is_never_high():
    a = person()
    ids = build_identities([ref("folder:a", "folder", "Jane Doe", a, 10)], T)
    fa = np.stack([face_of(a) for _ in range(6)])
    (p,) = propose([Cluster("A", 6, normalise(fa.mean(0)))], ids, T, {}, {})
    assert p.band != "high"


def test_merge_candidates():
    a = person()
    fa = [normalise(np.mean([face_of(a) for _ in range(5)], 0)) for _ in range(3)]
    clusters = [
        Cluster("A", 5, fa[0]),
        Cluster("B", 5, fa[1]),
        Cluster("C", 5, fa[2], name="Jane"),
        Cluster("D", 5, person(), name="John"),
    ]
    pairs = {(m.a, m.b) for m in merge_candidates(clusters, [], T)}
    assert pairs == {("A", "B"), ("A", "C"), ("B", "C")}


# --- web cards


def test_parse_cards_joins_languages_by_id():
    src = WebSource(
        key="s",
        label="members",
        pages=[{"url": "https://example.org/en/m", "lang": "en"}, {"url": "https://example.org/dv/m", "lang": "dv"}],
        card_pattern=r'<a href="/m/(?P<id>\d+)"><img src="(?P<image>[^"]+)"><h5>(?P<name>[^<]+)</h5><i>(?P<area>[^<]*)</i>',
    )
    en = '<a href="/m/1"><img src="/img/1.jpg"><h5>Jane &amp; Doe</h5><i>North</i></a>'
    dv = '<a href="/m/1"><img src="/img/1.jpg"><h5>ޖޭން</h5><i>x</i></a>'
    (card,) = parse_cards(src, {"en": en, "dv": dv})
    assert card.name == "Jane & Doe" and card.alt_names == {"dv": "ޖޭން"}
    assert card.image_url == "https://example.org/img/1.jpg" and card.details == {"area": "North"}


# --- the write path


class FakeStore:
    def __init__(self, before=()):
        self.audit, self.before = [], set(before)

    def ever_applied(self, pid):
        return pid in self.before

    def audit_skip(self, run, src, p, owner, old, note):
        self.audit.append(("skipped", p["person_id"], note))

    def audit_pending(self, run, src, p, owner, old):
        self.audit.append(["pending", p["person_id"], None])
        return len(self.audit) - 1

    def audit_set(self, i, status, note=None):
        self.audit[i][0] = status


class FakeAPI:
    def __init__(self, people):
        self.people, self.writes = people, []

    def person(self, pid):
        return dict(self.people[pid])

    def set_name(self, pid, name):
        self.writes.append((pid, name))
        self.people[pid]["name"] = name
        return {"name": name}


class FakeDB:
    def __init__(self, names):
        self.names = names

    def group_names(self, pid):
        return [("owner", self.names.get(pid, ""))]


def test_write_names_never_overwrites_and_respects_cap():
    people = {k: {"name": "", "isHidden": False} for k in "ABCDE"}
    people["D"]["isHidden"] = True
    ctx = SimpleNamespace(
        s=SimpleNamespace(max_apply=1),
        store=FakeStore(before={"E"}),
        api=FakeAPI(people),
        db=FakeDB({"B": "Curator Set"}),
    )
    props = [{"person_id": k, "name": f"Name {k}"} for k in "BDEAC"]
    res = cli._write_names(ctx, "r1", "r0", "owner", props)
    assert ctx.api.writes == [("A", "Name A")]  # B named by a user, D hidden, E labelled before, C over the cap
    assert res == Counter(applied=1, skipped_named=2, skipped_before=1, over_cap=1)


def test_undo_only_reverts_unchanged_names(monkeypatch):
    people = {"A": {"name": "Name A"}, "B": {"name": "Changed By Curator"}}
    api = FakeAPI(people)
    undone = []
    store = SimpleNamespace(
        start_run=lambda *a: None,
        applied=lambda run, persons: [
            {"id": 1, "person_id": "A", "old_name": "", "new_name": "Name A"},
            {"id": 2, "person_id": "B", "old_name": "", "new_name": "Name B"},
        ],
        audit_undone=lambda i, run: undone.append(i),
        finish_run=lambda *a: None,
    )

    class Ctx:
        def __init__(self, s, need_db=True):
            self.s, self.store, self.api, self.db = s, store, api, None

        def owner(self):
            return "owner"

        def close(self):
            pass

    monkeypatch.setattr(cli, "Context", Ctx)
    assert cli.cmd_undo(SimpleNamespace(), SimpleNamespace(run="r1", person=None)) == 0
    assert api.writes == [("A", "")] and undone == [1] and people["B"]["name"] == "Changed By Curator"
