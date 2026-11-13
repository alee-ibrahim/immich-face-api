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
    assert not compatible("Abdul Salaam Ibrahim", "Abdulla Shazeem")  # short skeletons must match exactly
    assert compatible("Mumthazim Ibrahim", "Munthazim Ibrahim")
    assert not compatible("Liraaru Hameed", "Ahmed Didi")  # hmd vs ahmd: no tolerance for short skeletons
    assert compatible("Jane AbdulRahmaan", "Jane Abdul Rahman")  # joined words
    assert not compatible("Ali Riza", "Ali Rifau")
    assert compatible("Abdul Salaam Ibrahim", "Abdulsalaam Ibrahim")
    assert compatible("Abdul Ghani Abdul Hakeem", "Abdullganee Abdullhakeem")
    assert not compatible("Saadha Ali", "Saudhullah Hilmy")  # a joined short name must not absorb a word
    assert not compatible("Ali Rifau", "Abdulla Rifau")
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
    assert cli.cmd_undo(SimpleNamespace(), SimpleNamespace(run=["r1"], person=None, copies=False)) == 0
    assert api.writes == [("A", "")] and undone == [1] and people["B"]["name"] == "Changed By Curator"


def test_write_names_skips_held_and_vanished_people():
    from immich_face_api.labeler.immich import ImmichError

    class API(FakeAPI):
        def person(self, pid):
            if pid == "G":
                raise ImmichError("GET /api/people/G: HTTP 404")
            return super().person(pid)

    people = {k: {"name": "", "isHidden": False} for k in "AH"}
    ctx = SimpleNamespace(s=SimpleNamespace(max_apply=10), store=FakeStore(), api=API(people), db=FakeDB({}))
    props = [
        {"person_id": "H", "name": "Name H", "evidence": ["x", "HOLD: the sources disagree on the name"]},
        {"person_id": "G", "name": "Name G", "evidence": []},
        {"person_id": "A", "name": "Name A", "evidence": ["x"]},
    ]
    res = cli._write_names(ctx, "r1", "r0", "owner", props)
    assert ctx.api.writes == [("A", "Name A")]
    assert res == Counter(applied=1, skipped_held=1, skipped_gone=1)


def test_namesake_about_as_close_is_held_and_conflicts_are_held():
    from immich_face_api.labeler.match import held

    a, b = person(), person()
    both = normalise(a + b)  # a cluster halfway between two namesakes
    ids = build_identities(
        [ref("web:s:1", "web", "Mohamed Rasheed", a, 1), ref("web:s:2", "web", "Mohamed Rasheed", b, 1)], T
    )
    assert all(i.homonym for i in ids) and all(i.proposed_name == "Mohamed Rasheed" for i in ids)  # plain names
    t = Thresholds(match_max_distance=0.9)
    (p,) = propose([Cluster("X", 5, both)], ids, t, {}, {})
    assert held(p.evidence()) and any("namesake" in h for h in p.hold)
    # a clear match to one of two namesakes is not held
    fa = np.stack([face_of(a) for _ in range(6)])
    (q,) = propose([Cluster("Y", 6, normalise(fa.mean(0)))], ids, T, {"Y": fa}, {})
    assert not q.hold and q.band == "medium"  # single portrait: reference has 1 photo < 2
    ids2 = build_identities(
        [ref("folder:1", "folder", "Ali Rifau", a), ref("web:s:1", "web", "Abdulla Rifau", a, 1)], T
    )
    (r,) = propose([Cluster("Z", 6, normalise(fa.mean(0)))], ids2, T, {"Z": fa}, {})
    assert held(r.evidence())


class FakeWriter:
    def __init__(self, rows, names):
        self.rows, self.names = rows, names  # names: (user, group) -> current name of the copy

    def candidates(self, owner):
        return [r for r in self.rows if self.names[(r[1], r[0])] == ""]

    def copy(self, owner, user, group, name):
        if self.names[(user, group)] != "":
            return False
        self.names[(user, group)] = name
        return True

    def revert(self, user, group, new, old):
        if self.names[(user, group)] != new:
            return False
        self.names[(user, group)] = old
        return True


class CopyStore:
    def __init__(self):
        self.rows, self.undone = [], []

    def copy_pending(self, run, group, user, name):
        self.rows.append(
            {
                "id": len(self.rows),
                "person_id": group,
                "owner_id": user,
                "old_name": "",
                "new_name": name,
                "status": "pending",
            }
        )
        return len(self.rows) - 1

    def audit_set(self, i, status, note=None):
        self.rows[i]["status"] = status

    def audit_undone(self, i, run):
        self.undone.append(i)


def test_share_names_fills_only_empty_copies_and_undo_reverts_only_unchanged():
    from immich_face_api.labeler.share_names import share_names, undo_copies

    names = {("u1", "g1"): "", ("u2", "g1"): "Curator Name", ("u1", "g2"): ""}
    w = FakeWriter([("g1", "u1", "Jane Doe"), ("g1", "u2", "Jane Doe"), ("g2", "u1", "John Roe")], names)
    st = CopyStore()
    assert share_names(w, st, "r", "owner", True, 100) == Counter(candidates=2, would_copy=2)
    assert names[("u1", "g1")] == ""  # dry run wrote nothing
    res = share_names(w, st, "r", "owner", False, 100)
    assert res == Counter(candidates=2, copied=2)
    assert names == {("u1", "g1"): "Jane Doe", ("u2", "g1"): "Curator Name", ("u1", "g2"): "John Roe"}
    names[("u1", "g2")] = "Renamed By Curator"
    applied = [r for r in st.rows if r["status"] == "applied"]
    assert undo_copies(w, st, "u", applied) == Counter(undone=1, changed_since=1)
    assert names[("u1", "g1")] == "" and names[("u1", "g2")] == "Renamed By Curator"


def test_run_is_a_dry_run_unless_bands_are_given():
    import inspect

    p = cli.build_parser()
    assert cli.apply_bands_of(p.parse_args(["run"])) == []
    assert cli.apply_bands_of(p.parse_args(["run", "--apply-band", "none"])) == []
    assert cli.apply_bands_of(p.parse_args(["run", "--apply-band", "high", "--apply-band", "medium"])) == [
        "high",
        "medium",
    ]
    # regression: the band list must not be shadowed later in cmd_run (a Counter of proposals once was)
    src = inspect.getsource(cli.cmd_run)
    assert src.count("apply_bands =") == 1 and "bands =" not in src.replace("apply_bands =", "")


# --- coverage: named-neighbour vote, source gate, more web source kinds, worklist


def test_named_vote_finds_more_clusters_of_a_named_person():
    from immich_face_api.labeler.match import named_vote

    a, b = person(), person()
    named = {"NA": np.stack([face_of(a) for _ in range(10)]), "NB": np.stack([face_of(b) for _ in range(10)])}
    names = {"NA": "Jane Doe", "NB": "John Roe"}
    unnamed = {"U1": np.stack([face_of(a) for _ in range(5)]), "U2": np.stack([face_of(person()) for _ in range(5)])}
    (v,) = named_vote(unnamed, named, names, {"U1": 5, "U2": 5}, T)
    assert v.person_id == "U1" and v.identity.name == "Jane Doe" and v.band == "high" and v.vote[:2] == (5, 5)
    assert v.identity.refs[0].source == "named" and "named-neighbour vote" in v.evidence()[0]


def test_unapproved_sources_are_held():
    from immich_face_api.labeler.cli import hold_unapproved_sources

    a, b = person(), person()
    refs = [
        ref("folder:a", "folder", "Jane Doe", a, 10, source="folder"),
        ref("web:wd:Q1", "web", "John Roe", b, 1, source="wikidata"),
    ]
    ids = build_identities(refs, T)
    fa = np.stack([face_of(a) for _ in range(6)])
    fb = np.stack([face_of(b) for _ in range(6)])
    clusters = [Cluster("A", 6, normalise(fa.mean(0))), Cluster("B", 6, normalise(fb.mean(0)))]
    props = propose(clusters, ids, T, {"A": fa, "B": fb}, {})
    n = hold_unapproved_sources(props, refs, ["folder"], clusters, T, {"A": fa, "B": fb}, {}, "")
    by = {p.person_id: p for p in props}
    assert n == 1 and not by["A"].hold and any("wikidata" in h for h in by["B"].hold)
    assert hold_unapproved_sources(props, refs, None, clusters, T, {}, {}, "") == 0


def test_listing_with_templates_and_alt_name_groups():
    src = WebSource(
        key="old",
        label="old members",
        pages=[{"url": "https://archive.example.org/old/members", "lang": "en"}],
        card_pattern=r'<a href="/m/(?P<id>[a-z_]+)" title="(?P<dv>[^"]+) - (?P<area>[^"]+)"><img src="t\.php\?src=(?P<image>[^&"]+)',
        name_template="{id}",
        image_template="https://archive.example.org/im_/{image}",
        alt_name_groups={"dv": "dv"},
        licence="official site, archived",
    )
    page = '<a href="/m/jane_doe" title="ޖޭން - North"><img src="t.php?src=http://x.org/p/jane.jpg&w=24">'
    (c,) = parse_cards(src, {"en": page})
    assert c.name == "Jane Doe" and c.alt_names == {"dv": "ޖޭން"}
    assert c.image_url == "https://archive.example.org/im_/http://x.org/p/jane.jpg"
    assert c.details == {"area": "North", "licence": "official site, archived"}


def test_wikidata_cards_record_licence():
    import json as _json

    from immich_face_api.labeler.web import wikidata_cards

    class F:
        def get(self, url, immutable, limit):
            if "sparql" in url:
                return _json.dumps(
                    {
                        "results": {
                            "bindings": [
                                {
                                    "item": {"value": "http://www.wikidata.org/entity/Q1"},
                                    "itemLabel": {"value": "Jane Doe"},
                                    "image": {
                                        "value": "http://commons.wikimedia.org/wiki/Special:FilePath/Jane%20Doe.jpg"
                                    },
                                    "dvLabel": {"value": "ޖޭން"},
                                },
                                {
                                    "item": {"value": "http://www.wikidata.org/entity/Q2"},
                                    "itemLabel": {"value": "Q2"},
                                    "image": {"value": "http://commons.wikimedia.org/wiki/Special:FilePath/x.jpg"},
                                },
                            ]
                        }
                    }
                ).encode()
            return _json.dumps(
                {
                    "query": {
                        "pages": {
                            "1": {
                                "imageinfo": [
                                    {
                                        "extmetadata": {
                                            "LicenseShortName": {"value": "CC BY-SA 4.0"},
                                            "Artist": {"value": "<a href='x'>Someone</a>"},
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
            ).encode()

    src = WebSource(key="wd", label="wikidata", kind="wikidata", query="SELECT ?item ?image WHERE {}")
    (c,) = wikidata_cards(src, F())
    assert c.name == "Jane Doe" and c.alt_names == {"dv": "ޖޭން"}
    assert c.details["licence"] == "CC BY-SA 4.0" and c.details["author"] == "Someone"
    assert c.image_url.startswith("https://commons.wikimedia.org/wiki/Special:FilePath/Jane%20Doe.jpg")


def test_worklist_csv():
    from immich_face_api.labeler.report import worklist_csv

    rows = [Cluster("A", 40, person()), Cluster("B", 7, person())]
    props = {"B": {"name": "Jane Doe", "band": "low", "evidence": ["HOLD: x"]}}
    out = worklist_csv(rows, props, "https://photos.example.org").splitlines()
    assert out[1].startswith("1,A,40,,,,https://photos.example.org/people/A")
    assert out[2].startswith("2,B,7,Jane Doe,low,HOLD: x,")


def test_overrides_only_for_pure_name_conflicts_and_only_names_from_the_evidence():
    from immich_face_api.labeler.cli import override_proposals

    conflict = "HOLD: the sources disagree on the name (Aishtah Sheema)"
    props = [
        {"person_id": "A", "name": "Aminath Sheema", "evidence": ["name conflict: Aishtah Sheema", conflict]},
        {
            "person_id": "B",
            "name": "Jane Doe",
            "evidence": ["name conflict: J Doe", conflict, "HOLD: a namesake is about as close"],
        },
        {"person_id": "C", "name": "John Roe", "evidence": ["name conflict: Johnny", conflict]},
        {"person_id": "D", "name": "Ann Lee", "evidence": ["x"]},
    ]
    ov = {
        k: {"name": v, "source": "directory", "note": "unique match"}
        for k, v in {"A": "Aishath Sheema", "B": "Jane Doe", "C": "Someone Else", "D": "Ann Lee"}.items()
    }
    out, c = override_proposals(props, ov)
    assert [o["person_id"] for o in out] == ["A"] and out[0]["name"] == "Aishath Sheema"
    assert not any(e.startswith("HOLD") for e in out[0]["evidence"])
    assert out[0]["evidence"][-1] == "name decided by directory: unique match"
    assert c == Counter(usable=1, not_only_a_name_conflict=2, name_not_among_the_evidence=1)
