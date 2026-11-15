"""Purity sweep: sub-identities of a mixed cluster (synthetic embeddings), the review file, apply and undo with fakes."""

import io
from collections import Counter
from types import SimpleNamespace

import numpy as np
from PIL import Image

from immich_face_api.labeler import cli
from immich_face_api.labeler.match import Cluster, normalise
from immich_face_api.labeler.purity import (
    PurityThresholds,
    assign_to_subs,
    average_linkage,
    crop_face,
    order_plans,
    plan_cluster,
    plans_csv,
    plans_html,
    read_review,
    summary,
)

RNG = np.random.default_rng(11)
DIM = 512


def identity():
    return normalise(RNG.normal(size=DIM))


def faces(center, n, spread=0.025):
    return normalise(center + RNG.normal(scale=spread, size=(n, DIM)))


def naive_average_linkage(emb, cut):
    """Reference implementation: recompute the mean pairwise distance of every pair of groups each step."""
    e = normalise(emb)
    d = 1 - e @ e.T
    groups = [[i] for i in range(len(e))]
    while len(groups) > 1:
        best = None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                v = d[np.ix_(groups[a], groups[b])].mean()
                if best is None or v < best[0]:
                    best = (v, a, b)
        if best[0] > cut:
            break
        _, a, b = best
        groups[a] += groups.pop(b)
    lab = np.zeros(len(e), int)
    for k, g in enumerate(groups):
        lab[g] = k
    return lab


def same_partition(a, b):
    return {frozenset(np.flatnonzero(a == x)) for x in set(a)} == {frozenset(np.flatnonzero(b == x)) for x in set(b)}


def test_average_linkage_matches_the_naive_definition():
    a, b, c = identity(), identity(), identity()
    emb = np.vstack([faces(a, 12, 0.03), faces(b, 7, 0.03), faces(c, 3, 0.03), faces(identity(), 1)])
    for cut in (0.3, 0.5, 0.9):
        assert same_partition(average_linkage(emb, cut), naive_average_linkage(emb, cut))
    assert len(set(average_linkage(emb, 0.5))) == 4
    assert average_linkage(emb[:1], 0.5).tolist() == [0]
    assert average_linkage(np.zeros((0, DIM)), 0.5).size == 0


def _world():
    main, near_main, other, stranger = identity(), None, identity(), identity()
    near_main = normalise(main + 0.4 * identity())  # same person, other light: within keep of main
    emb = np.vstack([faces(main, 40), faces(near_main, 6), faces(other, 10), faces(stranger, 5), faces(identity(), 2)])
    others = normalise(np.vstack([main, other, identity()]))  # index 0 = the cluster itself, 1 = an existing cluster
    return emb, others


def test_plan_cluster_keeps_main_moves_foreign_and_leaves_borderline():
    emb, others = _world()
    t = PurityThresholds()
    p = plan_cluster("p-self", "Person A", 630, emb, others, ["p-self", "p-other", "p-x"], ["", "", ""], t, skip=0)
    by_size = sorted(p.subs, key=lambda s: -len(s.idx))
    assert by_size[0].is_main and len(by_size[0].idx) >= 40
    acts = Counter(s.action for s in p.subs)
    moved = [s for s in p.subs if s.action == "move_existing"]
    assert len(moved) == 1 and moved[0].dest == "p-other" and len(moved[0].idx) == 10
    new = [s for s in p.subs if s.action == "move_new"]
    assert len(new) == 1 and len(new[0].idx) == 5
    assert acts["borderline"] == 1  # the 2-face stranger: too small for a new person
    assert all(s.dest != "p-self" for s in p.subs)  # never the cluster itself
    assert p.est_faces(moved[0]) == round(10 / len(emb) * 630)


def test_review_round_trip_and_ordering():
    emb, others = _world()
    t = PurityThresholds()
    named = plan_cluster("p-named", "Person A", 63, emb, others, ["p-named", "p-o", "p-x"], ["", "", ""], t, 0)
    clean = plan_cluster("p-clean", "", 500, faces(identity(), 30), others, ["a", "b", "c"], ["", "", ""], t)
    worse = plan_cluster("p-worse", "", 400, emb, others, ["p-worse", "p-o", "p-x"], ["", "", ""], t, 0)
    shown = order_plans([clean, worse, named], top_unnamed=10)
    assert [p.person_id for p in shown] == ["p-named", "p-worse"]  # named first, clean clusters left out
    text = plans_csv("run1", shown)
    assert "keep" not in {r.split(",")[10] for r in text.splitlines()[1:]}
    assert read_review(text) == {}
    rows = text.splitlines()
    approved = [rows[0]] + [r + "yes" if ",move_" in r and "p-named" in r else r for r in rows[1:]]
    got = read_review("\n".join(approved))
    assert set(got) == {("run1", "p-named")} and len(got[("run1", "p-named")]) == 2
    stats = summary([clean, worse, named], shown)
    assert stats["clusters_with_foreign"] == 2 and stats["named_with_foreign"] == 1
    page = plans_html("run1", shown, {}, stats, t)
    assert "Person A" in page and "new person" in page


def test_assign_and_crop():
    a, b = identity(), identity()
    k, d = assign_to_subs(np.vstack([faces(a, 2), faces(b, 1)]), np.vstack([a, b]))
    assert k.tolist() == [0, 0, 1] and (d < 0.2).all()
    img = io.BytesIO()
    Image.new("RGB", (400, 300), (90, 90, 90)).save(img, "JPEG")
    out = crop_face(img.getvalue(), (100, 50, 140, 100, 800, 600))
    assert out and Image.open(io.BytesIO(out)).size == (72, 72)
    assert crop_face(b"not an image", (0, 0, 1, 1, 1, 1)) is None


class FakeDB:
    def __init__(self, faces_of):
        self.faces_of = faces_of  # person -> (ids, emb)
        self.where = {f: p for p, (ids, _) in faces_of.items() for f in ids}

    def check(self, need_paths):
        return "3.4.1"

    def person_faces(self, owner, pid):
        return self.faces_of[pid]

    def face_people(self, ids):
        return {f: self.where.get(f) for f in ids}


class FakeAPI:
    def __init__(self, db):
        self.db, self.created, self.moves = db, 0, []

    def create_person(self):
        self.created += 1
        return f"new-{self.created}"

    def person(self, pid):
        return {"name": ""}

    def move_face(self, pid, face):
        self.moves.append((pid, face))
        self.db.where[face] = pid


class FakeStore:
    def __init__(self, plan):
        self.plan, self.audit, self.runs = plan, [], {}

    def start_run(self, run_id, command):
        self.runs[run_id] = command

    def finish_run(self, run_id, status, stats, *a):
        self.runs[run_id] = (status, stats)

    def purity_plan(self, run, pid):
        return self.plan.get((run, pid), [])

    def purity_moved(self, run, pid):
        return sum(1 for m in self.audit if m["src_run"] == run and m["src"] == pid and not m["undone"])

    def audit_face_move(self, run_id, src_run, face, src, dst, sub, new):
        self.audit.append(
            dict(id=len(self.audit) + 1, run=run_id, src_run=src_run, face=face, src=src, dst=dst, undone=False)
        )

    def face_moves(self, run):
        return [(m["id"], m["face"], m["src"], m["dst"]) for m in self.audit if m["run"] == run and not m["undone"]]

    def face_move_undone(self, move_id, undo_run):
        self.audit[move_id - 1]["undone"] = True


def test_apply_moves_only_approved_subs_and_undo_reverts_unchanged(monkeypatch, tmp_path):
    main, other, third = identity(), identity(), identity()
    ids = [f"f{i}" for i in range(9)]
    emb = np.vstack([faces(main, 5), faces(other, 2), faces(third, 2)])
    plan = {
        ("r1", "p1"): [
            (0, True, "keep", "", list(main)),
            (1, False, "move_existing", "p-dest", list(other)),
            (2, False, "move_new", "", list(third)),
        ]
    }
    db = FakeDB({"p1": (ids, emb)})
    api, store = FakeAPI(db), FakeStore(plan)

    class Ctx:
        def __init__(self, s, need_db=True):
            self.db, self.api, self.store = db, api, store

        def owner(self):
            return "owner"

        def close(self):
            pass

    monkeypatch.setattr(cli, "Context", Ctx)
    review = tmp_path / "review.csv"
    review.write_text(
        "run_id,person_id,sub,action,approve\nr1,p1,1,move_existing,yes\nr1,p1,2,move_new,\n", encoding="utf-8"
    )
    # dry run: nothing written
    assert cli.cmd_purity_apply(None, SimpleNamespace(review=review, apply=False, max_faces=100)) == 0
    assert api.moves == [] and api.created == 0
    assert cli.cmd_purity_apply(None, SimpleNamespace(review=review, apply=True, max_faces=100)) == 0
    assert sorted(api.moves) == [("p-dest", "f5"), ("p-dest", "f6")] and api.created == 0  # sub 2 not approved
    run = next(r for r, v in store.runs.items() if isinstance(v, tuple) and v[1].get("faces_moved") == 2)
    # applying the same review again does nothing
    assert cli.cmd_purity_apply(None, SimpleNamespace(review=review, apply=True, max_faces=100)) == 0
    assert len(api.moves) == 2
    # a curator moved f6 elsewhere since: undo leaves it alone
    db.where["f6"] = "p-curator"
    api.moves.clear()
    assert cli.cmd_purity_undo(None, SimpleNamespace(run=run)) == 0
    assert api.moves == [("p1", "f5")]


def test_cluster_dataclass_is_what_cmd_purity_reads():
    c = Cluster("p", 3, identity(), "", False)
    assert (c.person_id, c.faces, c.name, c.hidden) == ("p", 3, "", False)
