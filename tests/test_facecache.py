import asyncio
import dataclasses
import json
import struct

import numpy as np
import psycopg
import pytest

from immich_face_api.facecache import DIM, FaceCache, build, same_person_rule, search, vector_from_pg

from .conftest import synthetic_jpeg
from .test_app import AUTH, FACES, FakeDB, FakeML, make


def pg_binary(v):
    return struct.pack(">HH", len(v), 0) + np.asarray(v, dtype=">f4").tobytes()


def test_vector_from_pg_formats():
    v = np.linspace(-1, 1, DIM).astype(np.float32)
    assert np.allclose(vector_from_pg(pg_binary(v)), v)
    assert np.allclose(vector_from_pg("[" + ",".join(f"{x:.6f}" for x in v) + "]"), v, atol=1e-6)
    assert np.allclose(vector_from_pg(list(v)), v)


def test_search_matches_exact_cosine():
    rng = np.random.default_rng(7)
    people = [f"p{i}" for i in range(40)]
    rows, mats = [], []
    for i in range(3000):
        pid = people[i % 40]
        e = rng.normal(size=DIM).astype(np.float32)
        rows.append((pid, pid.upper(), pg_binary(e)))
        mats.append(e)
    m = np.stack(mats)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    snap = build(rows)
    assert snap.faces == 3000 and len(snap.person_ids) == 40 and snap.vectors.dtype == np.float16
    queries = [m[5] + rng.normal(scale=0.05, size=DIM), m[2999] + rng.normal(scale=0.05, size=DIM)]
    qtext = [json.dumps([float(x) for x in q]) for q in queries]
    got = search(snap, qtext, k=10, chunk=700)  # several chunks
    for q, res in zip(queries, got, strict=True):
        qn = q / np.linalg.norm(q)
        exact = 1 - m @ qn
        order = np.argsort(exact)[:10]
        assert [n.person_id for n in res][0] == rows[order[0]][0]
        assert abs(res[0].distance - exact[order[0]]) < 2e-3  # float16 storage
        assert len(res) == 10 and all(a.distance <= b.distance for a, b in zip(res, res[1:], strict=False))


def test_search_empty_and_no_queries():
    snap = build([])
    assert search(snap, ["[0.1]"], k=5) == [[]]
    assert search(build([("a", "A", pg_binary(np.ones(DIM)))]), [], k=5) == []


def test_same_person_rule_uses_centroids():
    rng = np.random.default_rng(3)
    base, other = rng.normal(size=DIM), rng.normal(size=DIM)
    rows = [("s1", "Jane Example", pg_binary(base + rng.normal(scale=0.3, size=DIM))) for _ in range(5)]
    rows += [("s2", "Jane Example", pg_binary(base + rng.normal(scale=0.3, size=DIM))) for _ in range(3)]
    rows += [("h", "Jane Example", pg_binary(other + rng.normal(scale=0.3, size=DIM))) for _ in range(4)]
    snap = build(rows)
    same = same_person_rule(snap, 0.6)
    assert same("s1", "s2")  # one face split over two clusters
    assert not same("s1", "h")  # a namesake: a different face
    assert not same("s1", "missing")
    assert same_person_rule(snap, 0.0) is None and same_person_rule(None, 0.6) is None
    assert build([]).centroids.shape == (0, DIM)


class SigDB:
    def __init__(self):
        self.sig = (1,)
        self.loads = 0

    async def named_signature(self):
        return self.sig

    async def named_faces(self, per_person):
        self.loads += 1
        return [("p1", "Alice Example", pg_binary(np.ones(DIM)))]


def test_refresh_only_on_change():
    async def run():
        db = SigDB()
        cache = FaceCache(db, per_person=10, refresh_seconds=60, knn=5)
        assert not cache.ready
        assert await cache.refresh() and cache.ready and db.loads == 1
        assert not await cache.refresh() and db.loads == 1
        db.sig = (2,)
        assert await cache.refresh() and db.loads == 2
        res = await cache.nearest([json.dumps([1.0] * DIM)])
        assert res[0][0].name == "Alice Example" and res[0][0].distance < 1e-3

    asyncio.run(run())


@pytest.mark.asyncio
async def test_identify_uses_cache_when_loaded(settings):
    app, c = make(settings)
    st = app.state.face

    class NoDB(FakeDB):
        async def nearest(self, embeddings, include_unnamed=False):
            raise AssertionError("database search used although the cache is loaded")

    st.db = NoDB([])
    st.faces = FaceCache(SigDB(), per_person=10, refresh_seconds=60, knn=5)
    await st.faces.refresh()
    st.ml = FakeML([dataclasses.replace(f, embedding=json.dumps([1.0] * DIM)) for f in FACES])
    r = await c.post("/v1/identify", headers=AUTH, content=synthetic_jpeg())
    assert r.status_code == 200 and len(r.json()["faces"]) == 2


@pytest.mark.asyncio
async def test_db_timeout_is_503_with_retry_after(settings):
    app, c = make(settings)

    class SlowDB(FakeDB):
        async def nearest(self, embeddings, include_unnamed=False):
            raise psycopg.errors.QueryCanceled("canceling statement due to statement timeout")

        async def people(self):
            raise psycopg.errors.QueryCanceled("canceling statement due to statement timeout")

    app.state.face.db = SlowDB([])
    r = await c.post("/v1/identify", headers=AUTH, content=synthetic_jpeg())
    assert r.status_code == 503 and r.json()["error"]["code"] == "busy" and r.headers["retry-after"] == "10"
    r = await c.get("/v1/people", headers=AUTH)
    assert r.status_code == 503


@pytest.mark.asyncio
async def test_identify_split_person_not_ambiguous_but_namesakes_are(settings):
    """Two clusters of one person with the same name are not an ambiguity; two namesakes with different faces are."""
    q = np.zeros(DIM, dtype=np.float32)
    q[0] = 1.0
    near_a = q.copy()
    near_a[1] = 0.55  # distance ~0.13 from q
    near_b = q.copy()
    near_b[1] = 0.58  # ~0.14: within the 0.03 margin
    far = np.zeros(DIM, dtype=np.float32)
    far[2] = 1.0

    class TwoDB(SigDB):
        def __init__(self, rows):
            super().__init__()
            self.rows = rows

        async def named_faces(self, per_person):
            return self.rows

    async def identify(rows):
        app, c = make(settings)
        st = app.state.face
        st.faces = FaceCache(TwoDB(rows), per_person=10, refresh_seconds=60, knn=5)
        await st.faces.refresh()
        st.ml = FakeML([dataclasses.replace(FACES[0], embedding=json.dumps([float(x) for x in q]))])
        r = await c.post("/v1/identify", headers=AUTH, content=synthetic_jpeg())
        assert r.status_code == 200
        return r.json()["faces"][0]

    # split person: both clusters are the same face (centroids close)
    split = [("s1", "Jane Example", pg_binary(near_a)), ("s2", "Jane Example", pg_binary(near_b))]
    f = await identify(split)
    assert f["name"] == "Jane Example" and f["ambiguous"] is False
    # namesakes: s2's other faces put its centroid far away -> a different person, still ambiguous
    namesakes = split + [("s2", "Jane Example", pg_binary(far))] * 3
    f = await identify(namesakes)
    assert f["ambiguous"] is True
