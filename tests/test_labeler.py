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
        s=SimpleNamespace(max_apply=1, sources=SimpleNamespace(names=NameRules())),
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
    ctx = SimpleNamespace(
        s=SimpleNamespace(max_apply=10, sources=SimpleNamespace(names=NameRules())),
        store=FakeStore(),
        api=API(people),
        db=FakeDB({}),
    )
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


def test_nearest_named_is_the_brute_force_nearest_face_across_batches_and_tiles():
    from immich_face_api.labeler.match import nearest_named

    nmat = normalise(RNG.normal(size=(37, DIM))).astype(np.float32)
    nmat[20] = nmat[5]  # a tie: the earlier named face wins, as one argmax over the whole matrix would
    unnamed = [normalise(RNG.normal(size=(k, DIM))).astype(np.float32) for k in (1, 3, 7, 2, 9, 4)]
    unnamed[2][0] = nmat[5]
    for rows, cols in ((4, 5), (8, 64), (1, 1), (100, 100)):  # a cluster bigger than a batch still goes in one piece
        dists, owners = nearest_named(unnamed, nmat, rows=rows, cols=cols)
        for u, d, o in zip(unnamed, dists, owners, strict=True):
            sim = u @ nmat.T
            assert (o == sim.argmax(1)).all()
            assert np.allclose(d, 1.0 - sim.max(1), atol=1e-6)
        assert owners[2][0] == 5


def test_named_vote_same_result_for_any_batch_size(monkeypatch):
    from immich_face_api.labeler import match

    people = [person() for _ in range(6)]
    named = {f"N{i}": np.stack([face_of(p) for _ in range(4 + i)]) for i, p in enumerate(people[:4])}
    names = {"N0": "Jane Doe", "N1": "John Roe", "N2": "Jane Doe", "N3": "Ali Hassan"}
    unnamed = {f"U{i}": np.stack([face_of(people[i % 6]) for _ in range(1 + i % 5)]) for i in range(12)}
    faces = {k: len(v) for k, v in unnamed.items()}

    def key(votes):
        return [(v.person_id, v.identity.name, v.band, v.distance, v.margin, v.vote) for v in votes]

    want = key(match.named_vote(unnamed, named, names, faces, T))
    assert want and {w[1] for w in want} >= {"Jane Doe", "John Roe", "Ali Hassan"}
    real = match.nearest_named
    monkeypatch.setattr(match, "nearest_named", lambda u, n: real(u, n, rows=3, cols=7))
    assert key(match.named_vote(unnamed, named, names, faces, T)) == want


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


# --- titles ---------------------------------------------------------------------------------------------------------
TITLES = NameRules(
    titles=["Hon", "Honourable", "MP", "Mr", "Mrs", "His Excellency", "Al-Ustaz", "Deputy Speaker", "Speaker"],
    keep_titles={"Dr": "Dr.", "Doctor": "Dr."},
)


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("Hon. Jane Example", "Jane Example"),
        ("Jane Example, MP", "Jane Example"),
        ("Jane Example (MP)", "Jane Example"),
        ("Jane Example M.P.", "Jane Example"),
        ("His Excellency Jane Example", "Jane Example"),
        ("Deputy Speaker Hon. Jane Example", "Jane Example"),
        ("Dr Jane Example", "Dr. Jane Example"),
        ("Doctor Jane Example", "Dr. Jane Example"),
        ("Hon. Dr. Jane Example", "Dr. Jane Example"),
        ("Dr. Jane Example", "Dr. Jane Example"),
        ("JANE  EXAMPLE ", "Jane Example"),
        ("Honey Example", "Honey Example"),  # a name, not a title
        ("Jane Dhonexample", "Jane Dhonexample"),
        ("Mr", "Mr"),  # nothing left: unchanged
        ("Jane Mr Example", "Jane Mr Example"),  # only the edges are touched
    ],
)
def test_tidy_name(raw, want):
    from immich_face_api.labeler.names import tidy_name

    assert tidy_name(raw, TITLES) == want


def test_tidy_name_without_rules_only_spacing_and_casing():
    from immich_face_api.labeler.names import tidy_name

    assert tidy_name("  hon.  jane   example ", NameRules()) == "Hon. Jane Example"


def test_clean_name_strips_titles():
    assert clean_name("Hon. Jane Example", TITLES) == "Jane Example"
    assert clean_name("Dr Jane Example", TITLES) == "Dr. Jane Example"


def test_write_names_never_writes_a_title():
    people = {"A": {"name": "", "isHidden": False}}
    ctx = SimpleNamespace(
        s=SimpleNamespace(max_apply=5, sources=SimpleNamespace(names=TITLES)),
        store=FakeStore(before=set()),
        api=FakeAPI(people),
        db=FakeDB({}),
    )
    cli._write_names(ctx, "r1", "r0", "owner", [{"person_id": "A", "name": "Hon. Jane Example"}])
    assert ctx.api.writes == [("A", "Jane Example")]


class RenameStore:
    def __init__(self):
        self.rows = []

    def audit_rename(self, run, pid, owner, old, new, target, note):
        self.rows.append(
            {
                "person_id": pid,
                "owner_id": owner,
                "old_name": old,
                "new_name": new,
                "target": target,
                "status": "pending",
                "id": len(self.rows),
            }
        )
        return len(self.rows) - 1

    def audit_set(self, i, status, note=None):
        self.rows[i]["status"] = status

    def audit_undone(self, i, run):
        self.rows[i]["status"] = "undone"


def test_tidy_owner_plan_apply_and_doubtful():
    from immich_face_api.labeler.tidy import apply_owner, plan_owner, summary

    rows = [
        ("a", "Hon. Jane Example"),
        ("b", "Dr. Sam Example"),
        ("c", "Jane Mr Example"),
        ("d", "Dr Ali Example"),
        ("e", "Changed Later Hon."),
    ]
    changes, doubtful = plan_owner(rows, TITLES)
    assert [(c["person_id"], c["new"]) for c in changes] == [
        ("a", "Jane Example"),
        ("d", "Dr. Ali Example"),
        ("e", "Changed Later"),
    ]
    assert [d["person_id"] for d in doubtful] == ["c"]
    assert summary(changes, doubtful)["by_title"] == {"Hon.": 2, "Dr": 1}
    people = {"a": {"name": "Hon. Jane Example"}, "d": {"name": "Dr Ali Example"}, "e": {"name": "Renamed By Hand"}}
    api, st = FakeAPI(people), RenameStore()
    res = apply_owner(api, st, "r", "owner", changes, cap=10)
    assert res == Counter(applied=2, skipped_changed=1)
    assert people["a"]["name"] == "Jane Example" and people["e"]["name"] == "Renamed By Hand"
    assert [r["old_name"] for r in st.rows] == ["Hon. Jane Example", "Dr Ali Example"]


def test_tidy_copies_follow_owner_and_keep_own_spelling():
    from immich_face_api.labeler.share_names import undo_copies
    from immich_face_api.labeler.tidy import apply_copies, plan_copies

    names = {("u1", "g1"): "Hon. Jane Example", ("u2", "g1"): "Hon. Jane Exampel", ("u1", "g2"): "Hon. Sam Example"}

    class W(FakeWriter):
        def rename(self, user, group, old, new):
            return self.revert(user, group, old, new)

    w = W([], names)
    rows = [
        ("g1", "u1", names[("u1", "g1")], "Jane Example"),
        ("g1", "u2", names[("u2", "g1")], "Jane Example"),
        ("g2", "u1", names[("u1", "g2")], "Hon. Sam Example"),
    ]
    changes, own = plan_copies(rows, TITLES)
    assert [c["user_id"] + c["person_id"] for c in changes] == ["u1g1", "u1g2"]
    assert [o["user_id"] for o in own] == ["u2"]  # the curator's own spelling stays
    st = RenameStore()
    res = apply_copies(w, st, "r", changes, cap=10)
    assert res == Counter(applied=1, skipped_owner_not_tidy=1)  # g2: owner not tidied yet
    assert names[("u1", "g1")] == "Jane Example" and names[("u2", "g1")] == "Hon. Jane Exampel"
    applied = [r for r in st.rows if r["status"] == "applied"]
    assert undo_copies(w, st, "u", applied) == Counter(undone=1)
    assert names[("u1", "g1")] == "Hon. Jane Example"


def test_tidy_names_parser():
    p = cli.build_parser()
    a = p.parse_args(["tidy-names"])
    assert not a.apply and not a.copies
    a = p.parse_args(["tidy-names", "--copies", "--apply"])
    assert a.apply and a.copies


def test_roster_cards_from_file(tmp_path):
    import json as _json

    from immich_face_api.labeler.web import roster_cards

    roster = tmp_path / "roster.json"
    roster.write_text(
        _json.dumps(
            {
                "people": [
                    {
                        "id": "a1",
                        "name": "  Jane   Doe ",
                        "image_url": "https://example.org/p/jane.jpg",
                        "alt_names": {"dv": "ޖޭން"},
                        "details": {"position": "Minister of Things", "organisation": "Ministry", "page_url": "u"},
                    },
                    {"id": "a1", "name": "Repeated Id", "image_url": "https://example.org/p/x.jpg"},
                    {"id": "b2", "name": "No Image"},
                    {"id": "c3", "name": "Bad Scheme", "image_url": "file:///etc/passwd"},
                    "not an object",
                    {
                        "id": "d4",
                        "name": "John Roe",
                        "image_url": "https://example.org/p/john.jpg",
                        "details": {"label": "Judge, Court"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    src = WebSource(key="officials", label="Officials", kind="roster", roster=str(roster), licence="public pages")
    a, d = roster_cards(src, fetcher=None)
    assert (a.id, a.name, a.alt_names, a.image_url) == (
        "a1",
        "Jane Doe",
        {"dv": "ޖޭން"},
        "https://example.org/p/jane.jpg",
    )
    assert a.label == "Minister of Things, Ministry"
    assert a.details["licence"] == "public pages" and a.details["page_url"] == "u"
    assert (d.name, d.label) == ("John Roe", "Judge, Court")


def test_roster_cards_from_url_and_validation():
    import json as _json

    from immich_face_api.labeler.web import roster_cards

    class F:
        def __init__(self, body):
            self.body, self.urls = body, []

        def get(self, url, immutable, limit):
            self.urls.append(url)
            return self.body

    body = _json.dumps({"people": [{"id": 7, "name": "Jane Doe", "image_url": "http://x.org/j.png"}]}).encode()
    f = F(body)
    src = WebSource(key="r", label="R", kind="roster", roster="https://x.org/roster.json")
    (c,) = roster_cards(src, f)
    assert f.urls == ["https://x.org/roster.json"] and c.id == "7" and c.label == ""
    with pytest.raises(ValueError):
        roster_cards(src, F(b'[{"id": 1}]'))
    assert roster_cards(src, F(None)) == []
    with pytest.raises(ConfigError):
        WebSource(key="r", label="R", kind="roster")


def test_look_alike_guard_holds_in_the_batched_vote(monkeypatch):
    """The guard runs on the batched nearest faces: same holds with tiny batches/tiles as with the default ones, and a
    look-alike cluster is held while the clear one is not, also when they share a batch with other clusters."""
    from immich_face_api.labeler import match

    a, c = person(), person()
    twin = normalise(a + RNG.normal(scale=0.02, size=DIM))
    named = {
        "NA": np.stack([face_of(a) for _ in range(10)]),
        "NT": np.stack([face_of(twin) for _ in range(10)]),
        "NC": np.stack([face_of(c) for _ in range(10)]),
    }
    names = {"NA": "Ahmed Didi", "NT": "Ahmed Hisham Wajeeh", "NC": "Mariyam Shifa"}
    unnamed = {f"A{i}": np.stack([face_of(a) for _ in range(6)]) for i in range(3)}
    unnamed |= {f"C{i}": np.stack([face_of(c) for _ in range(6)]) for i in range(3)}
    unnamed["X"] = np.stack([face_of(person()) for _ in range(4)])
    faces = {k: len(v) for k, v in unnamed.items()}

    def key(votes):
        return sorted((v.person_id, v.identity.name, v.band, v.distance, tuple(v.hold)) for v in votes)

    want = key(match.named_vote(unnamed, named, names, faces, T))
    by = {w[0]: w for w in want}
    assert all(by[f"A{i}"][4] and "look-alike" in by[f"A{i}"][4][0] for i in range(3))
    assert all(not by[f"C{i}"][4] and by[f"C{i}"][2] == "high" for i in range(3)) and "X" not in by
    real = match.nearest_named
    monkeypatch.setattr(match, "nearest_named", lambda u, n: real(u, n, rows=5, cols=4))
    assert key(match.named_vote(unnamed, named, names, faces, T)) == want


def test_named_vote_holds_look_alikes():
    from immich_face_api.labeler.match import lookalike_names, named_vote

    a = person()
    twin = normalise(a + RNG.normal(scale=0.02, size=DIM))  # a different person who looks almost the same
    named = {"NA": np.stack([face_of(a) for _ in range(10)]), "NT": np.stack([face_of(twin) for _ in range(10)])}
    names = {"NA": "Jane Doe", "NT": "Joan Dee"}
    assert set(lookalike_names(named, names, T.vote_lookalike_distance)) == {"Jane Doe", "Joan Dee"}
    unnamed = {"U1": np.stack([face_of(a) for _ in range(6)])}
    (v,) = named_vote(unnamed, named, names, {"U1": 6}, T)
    assert v.hold and "look-alike" in v.hold[0] and any(e.startswith("HOLD: look-alike") for e in v.evidence())
    off = Thresholds(vote_min_name_gap=0, vote_lookalike_distance=0)
    (w,) = named_vote(unnamed, named, names, {"U1": 6}, off)
    assert not w.hold


def test_named_vote_no_hold_for_clearly_different_people():
    from immich_face_api.labeler.match import named_vote

    a, b = person(), person()
    named = {"NA": np.stack([face_of(a) for _ in range(10)]), "NB": np.stack([face_of(b) for _ in range(10)])}
    names = {"NA": "Jane Doe", "NB": "Joan Dee"}
    (v,) = named_vote({"U": np.stack([face_of(a) for _ in range(5)])}, named, names, {"U": 5}, T)
    assert not v.hold and v.band == "high"


def test_same_person_name_is_stricter_than_compatible():
    from immich_face_api.labeler.match import same_person_name

    assert compatible("Jane", "Jane Mary Roe") and not same_person_name("Jane", "Jane Mary Roe")
    assert same_person_name("Jonathan Smith", "Jonathon Smith") and same_person_name(
        "Catherine Brown", "Dr. Catherine Brown"
    )


# --- anchored vote: no vote chains, review holds, cleared people, read-only re-check


def _chain_world():
    """Anchor NA = Jane Doe (person a). Hub H was vote-named Jane Doe but mixes a and b. U = unnamed, person b."""
    a, b = person(), person()
    hub = np.stack([face_of(a) for _ in range(5)] + [face_of(b) for _ in range(15)])
    named = {"NA": np.stack([face_of(a) for _ in range(10)]), "H": hub}
    names = {"NA": "Jane Doe", "H": "Jane Doe"}
    u = np.stack([face_of(b) for _ in range(6)])
    clusters = [
        Cluster("NA", 10, normalise(named["NA"].mean(0)), "Jane Doe"),
        Cluster("H", 20, normalise(hub.mean(0)), "Jane Doe"),
        Cluster("U", 6, normalise(u.mean(0)), ""),
    ]
    return a, b, named, names, clusters, {"U": u}


def test_vote_chain_through_a_vote_named_cluster_is_stopped():
    from immich_face_api.labeler.match import anchored_voters, named_vote

    _, _, named, names, clusters, unnamed = _chain_world()
    # the old rule: every named person votes -> U gets the anchor's name through the hub
    (v,) = named_vote(unnamed, named, names, {"U": 6}, T)
    assert v.identity.name == "Jane Doe"
    voters, c = anchored_voters({c.person_id: (c.name, c.centroid) for c in clusters if c.name}, {"H"}, T)
    assert voters == {"NA"} and c == {"anchors": 1, "vote_named_near_anchor": 0, "vote_named_dropped": 1}
    assert named_vote(unnamed, {p: named[p] for p in voters}, names, {"U": 6}, T) == []


def test_vote_named_person_near_an_anchor_still_votes_unless_disabled():
    from immich_face_api.labeler.match import anchored_voters

    a = person()
    cen = {k: normalise(np.stack([face_of(a) for _ in range(8)]).mean(0)) for k in ("NA", "V")}
    named = {"NA": ("Jane Doe", cen["NA"]), "V": ("Jane Doe", cen["V"]), "W": ("John Roe", cen["V"])}
    voters, _ = anchored_voters(named, {"V", "W"}, T)
    assert voters == {"NA", "V"}  # W: vote-named, and no anchor has the name John Roe
    assert anchored_voters(named, {"V", "W"}, Thresholds(vote_anchor_distance=0))[0] == {"NA"}


def test_date_eras_and_folder_contradiction():
    from immich_face_api.labeler.match import date_eras, folder_contradiction

    assert date_eras([2001, 2002, 2003, 2019, 2020, 2020], 8, 0.2) == ("2001-2003", "2019-2020")
    assert date_eras([2001, 2002, 2003, 2004, 2005, 2006], 8, 0.2) is None
    assert date_eras([2001] + [2020] * 9, 8, 0.2) is None  # one stray old photo is not an era
    assert date_eras([2001, 2020], 8, 0.2) is None  # too few photos
    assert folder_contradiction("Jane Doe", {"John Roe": 3}, 2) == ("John Roe", 3)
    assert folder_contradiction("Jane Doe", {"Jane Doe": 5, "John Roe": 3}, 2) is None
    assert folder_contradiction("Jane Doe", {"Jane Doe": 1, "John Roe": 1}, 2) is None
    assert folder_contradiction("Jane Doe", None, 2) is None


def test_review_holds_for_a_mixed_cluster():
    from immich_face_api.labeler.match import spread, vote_review_holds

    a = person()
    one = np.stack([face_of(a) for _ in range(20)])
    mixed = np.stack([face_of(p) for p in (a, person(), person(), person()) for _ in range(5)])  # a hub of four
    assert spread(one) < T.vote_max_spread <= spread(mixed)
    assert vote_review_holds("Jane Doe", one, T) == []
    (h,) = vote_review_holds("Jane Doe", mixed, T)
    assert h.startswith("mixed cluster: its faces are spread")
    hs = vote_review_holds("Jane Doe", one, T, [2001] * 6 + [2020] * 6, {"John Roe": 4})
    assert [x.split(":")[0] for x in hs] == ["mixed cluster", "folder evidence"] and "2001-2001" in hs[0]
    off = Thresholds(vote_max_spread=0, vote_era_gap_years=0, vote_folder_min_faces=0)
    assert vote_review_holds("Jane Doe", mixed, off, [2001] * 6 + [2020] * 6, {"John Roe": 4}) == []


def test_write_origins_tracks_vote_tidy_correction_and_clear():
    from immich_face_api.labeler.store import write_origins

    vote_ev = ["named-neighbour vote: 5 of 5 sampled faces"]
    rows = [
        ("P1", "Hon Jane Doe", vote_ev, None, "r1"),
        ("P1", "Jane Doe", [], "tidy-names", "r2"),  # title removed: still from the vote
        ("P2", "Jane Doe", vote_ev, None, "r1"),
        ("P2", "John Roe", [], "owner-approved correction", "r3"),  # corrected by the owner: an anchor now
        ("P3", "Jane Doe", vote_ev, None, "r1"),
        ("P3", "", [], "cleared-for-review: wrong name", "r4"),
        ("P4", "Jane Doe", ["folder /people/Jane Doe/: 3 photo(s)"], None, "r1"),
    ]
    w = write_origins(rows)
    assert (w["P1"]["vote"], w["P1"]["name"]) == (True, "Jane Doe")
    assert (w["P2"]["vote"], w["P2"]["cleared"]) == (False, False)
    assert (w["P3"]["vote"], w["P3"]["cleared"], w["P3"]["name"]) == (False, True, "")
    assert not w["P4"]["vote"]


class _VoteDB:
    def __init__(self, samples, years=None):
        self.s, self.y = samples, years or {}

    def samples(self, owner, ids, k):
        return {i: self.s[i] for i in ids if i in self.s}

    def face_years(self, owner, ids, k):
        return {i: self.y[i] for i in ids if i in self.y}


class _VoteStore:
    def __init__(self, writes):
        self.w = writes

    def latest_owner_writes(self):
        return self.w


def test_add_named_votes_anchored_and_cleared():
    _, b, named, _, clusters, unnamed = _chain_world()
    c_faces = np.stack([face_of(b) for _ in range(6)])
    clusters.append(Cluster("C", 6, normalise(c_faces.mean(0)), ""))
    writes = {
        "H": {"vote": True, "cleared": False, "name": "Jane Doe", "run_id": "r1"},
        "C": {"vote": False, "cleared": True, "name": "", "run_id": "r2"},
    }
    ctx = SimpleNamespace(store=_VoteStore(writes), db=_VoteDB({**named, **unnamed, "C": c_faces}))
    props, st = cli.add_named_votes(ctx, "owner", clusters, [], T, None)
    assert props == [] and st["vote_voters_vote_named_dropped"] == 1 and st["votes"] == 0
    # with the hub as an anchor (say a person named it), U and C are voted; C is held as cleared for review
    ctx.store = _VoteStore({"C": writes["C"]})
    props, st = cli.add_named_votes(ctx, "owner", clusters, [], T, None)
    by = {p.person_id: p for p in props}
    assert set(by) == {"U", "C"} and st["vote_held_cleared"] == 1
    assert any("cleared for curator review" in h for h in by["C"].hold)
    assert not any("cleared" in h for h in by["U"].hold)


def test_add_named_votes_holds_on_folder_evidence_and_eras():
    a = person()
    named = {"NA": np.stack([face_of(a) for _ in range(10)])}
    u = np.stack([face_of(a) for _ in range(12)])
    clusters = [
        Cluster("NA", 10, normalise(named["NA"].mean(0)), "Jane Doe"),
        Cluster("U", 12, normalise(u.mean(0)), ""),
    ]
    ctx = SimpleNamespace(store=_VoteStore({}), db=_VoteDB({**named, "U": u}, {"U": [1999] * 6 + [2015] * 6}))
    props, st = cli.add_named_votes(ctx, "owner", clusters, [], T, None, folder_names={"U": {"John Roe": 5}})
    (p,) = props
    assert p.identity.name == "Jane Doe" and st["vote_held_folder"] == 1 and st["vote_held_eras"] == 1
    assert any(h.startswith("folder evidence") for h in p.hold) and any("two eras" in h for h in p.hold)


def test_recheck_flags_a_vote_named_cluster_the_anchors_do_not_support():
    from immich_face_api.labeler.report import recheck_csv, recheck_html

    _, _, named, _, clusters, _ = _chain_world()
    a2 = person()
    good = np.stack([face_of(a2) for _ in range(8)])
    anchor2 = np.stack([face_of(a2) for _ in range(10)])
    clusters += [
        Cluster("NB", 10, normalise(anchor2.mean(0)), "John Roe"),
        Cluster("G", 8, normalise(good.mean(0)), "John Roe"),  # vote-named and right
    ]
    usamp = {"H": named["H"], "G": good}
    nsamp = {"NA": named["NA"], "NB": anchor2}
    rows = cli.recheck(clusters, {"H", "G"}, usamp, nsamp, T, {"H": [2003] * 5 + [2021] * 15})
    by = {r["person_id"]: r for r in rows}
    assert by["H"]["suspect"] and not by["G"]["suspect"] and rows[0]["person_id"] == "H"
    assert by["H"]["eras"] == "2003-2003 / 2021-2021" and by["G"]["anchor_distance"] < T.vote_anchor_distance
    assert "Jane Doe" in recheck_html("r", {"suspects": 1}, rows[:1], {}, "")
    assert recheck_csv(rows, "").count("\n") == 3
