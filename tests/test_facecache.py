import asyncio
import dataclasses
import json
import struct

import numpy as np
import psycopg
import pytest

from immich_face_api import metrics as m
from immich_face_api.facecache import (
    DIM,
    Builder,
    FaceCache,
    build,
    per_person_cap,
    same_person_rule,
    search,
    vector_from_pg,
)

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

    async def named_face_counts(self):
        return [1]

    async def iter_named_faces(self, per_person, batch=2048):
        rows = await self.named_faces(per_person)
        for i in range(0, len(rows), batch):
            yield rows[i : i + batch]


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

        async def named_face_counts(self):
            return [len(self.rows)]

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


def test_builder_batches_equal_one_build_and_grow():
    rng = np.random.default_rng(11)
    rows = [(f"p{i % 7}", f"P{i % 7}", pg_binary(rng.normal(size=DIM))) for i in range(1000)]
    whole = build(rows)
    for capacity in (1000, 10, 0):  # exact, too small (grows), unknown
        b = Builder(capacity, 3)
        for i in range(0, len(rows), 97):
            b.add(rows[i : i + 97])
        snap = b.finish(("sig",))
        assert snap.faces == 1000 and snap.vectors.dtype == np.float16 and snap.signature == ("sig",)
        assert np.array_equal(snap.vectors, whole.vectors) and np.array_equal(snap.person_idx, whole.person_idx)
        assert snap.person_ids == whole.person_ids and np.allclose(snap.centroids, whole.centroids, atol=1e-6)
    # a too-high estimate is trimmed (a copy only when many rows are spare)
    b = Builder(100_000)
    b.add(rows[:10])
    snap = b.finish()
    assert snap.faces == 10 and (snap.vectors.base is None or snap.vectors.base.shape[0] < 100_000)


def test_per_person_cap_fits_total():
    counts = [500, 200, 40, 3, 1]
    assert per_person_cap(counts, 100, 0) == 100  # no total cap
    assert per_person_cap(counts, 100, 10_000) == 100  # fits
    cap = per_person_cap(counts, 100, 150)
    assert sum(min(c, cap) for c in counts) <= 150 < sum(min(c, cap + 1) for c in counts)
    assert per_person_cap(counts, 100, 2) == 1  # never 0


class StreamDB(SigDB):
    def __init__(self, counts):
        super().__init__()
        self.counts = counts
        self.caps = []
        rng = np.random.default_rng(5)
        self.rows = {f"p{i}": [pg_binary(rng.normal(size=DIM)) for _ in range(c)] for i, c in enumerate(counts)}

    async def named_face_counts(self):
        return list(self.counts)

    async def iter_named_faces(self, per_person, batch=2048):
        self.loads += 1
        self.caps.append(per_person)
        flat = [(p, p.upper(), e) for p, embs in self.rows.items() for e in embs[:per_person]]
        for i in range(0, len(flat), batch):
            yield flat[i : i + batch]


def test_refresh_streams_with_total_cap_and_metrics():
    async def run():
        db = StreamDB([50, 30, 5])
        cache = FaceCache(db, per_person=40, refresh_seconds=60, knn=5, max_faces=40, batch=7)
        assert await cache.refresh()
        assert db.caps == [17] and cache.effective_per_person == 17  # 17 + 17 + 5 = 39 <= 40 < 18 + 18 + 5
        assert cache.snap.faces == 39 and len(cache.snap.person_ids) == 3
        assert m.FACE_CACHE_FACES._value.get() == cache.snap.faces
        assert m.FACE_CACHE_PEOPLE._value.get() == 3
        assert m.FACE_CACHE_BYTES._value.get() == cache.snap.nbytes > 0

    asyncio.run(run())


def test_lazy_cache_loads_on_use_and_unloads_when_idle():
    async def run():
        db = StreamDB([3, 2])
        cache = FaceCache(db, per_person=10, refresh_seconds=60, knn=5, idle_seconds=1)
        task = asyncio.create_task(cache.loop())
        await asyncio.sleep(0.05)
        assert not cache.ready and db.loads == 0  # lazy: nothing loaded at start
        assert await cache.use() and cache.ready and db.loads == 1
        cache.last_used -= 5  # idle for longer than idle_seconds
        await asyncio.sleep(1.2)
        assert not cache.ready and m.FACE_CACHE_FACES._value.get() == 0
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # an eager cache does not load in use()
        eager = FaceCache(StreamDB([1]), per_person=10, refresh_seconds=60, knn=5)
        assert not await eager.use()

    asyncio.run(run())


@pytest.mark.asyncio
async def test_cold_cache_fails_fast_and_is_not_ready(settings):
    """Until the face cache is loaded: /readyz 503 (no traffic) and identify 503 busy at once, no slow SQL search."""
    app, c = make(dataclasses.replace(settings, face_cache_sql_fallback=False))
    st = app.state.face

    class NoDB(FakeDB):
        async def nearest(self, embeddings, include_unnamed=False):
            raise AssertionError("exact SQL search used while the cache loads")

    st.db = NoDB([])
    st.faces = FaceCache(SigDB(), per_person=10, refresh_seconds=60, knn=5)
    st.ml = FakeML([dataclasses.replace(f, embedding=json.dumps([1.0] * DIM)) for f in FACES])
    st.ready = True
    r = await c.get("/readyz")
    assert r.status_code == 503 and r.json()["reason"] == "face cache loading"
    r = await c.post("/v1/identify", headers=AUTH, content=synthetic_jpeg())
    assert r.status_code == 503 and r.json()["error"]["code"] == "busy" and r.headers["retry-after"] == "10"
    await st.faces.refresh()
    assert (await c.get("/readyz")).status_code == 200
    r = await c.post("/v1/identify", headers=AUTH, content=synthetic_jpeg())
    assert r.status_code == 200


def test_vectorised_decode_matches_per_row():
    rng = np.random.default_rng(9)
    vs = [rng.normal(size=DIM).astype(np.float32) for _ in range(50)]
    rows = [(f"p{i % 3}", "N", pg_binary(v)) for i, v in enumerate(vs)]
    text_rows = [(p, n, "[" + ",".join(repr(float(x)) for x in v) + "]") for (p, n, _), v in zip(rows, vs, strict=True)]
    a, b = build(rows), build(text_rows)  # binary: one numpy call; text: row by row
    assert np.array_equal(a.vectors, b.vectors) and np.array_equal(a.person_idx, b.person_idx)
    f32 = build(rows, dtype=np.float32)
    assert f32.vectors.dtype == np.float32 and np.allclose(f32.vectors, a.vectors, atol=2e-3)
    q = [json.dumps([float(x) for x in vs[7]])]
    assert search(f32, q, k=3)[0][0].distance < 1e-5 and search(a, q, k=3)[0][0].person_id == "p1"


def test_face_only_changes_rebuild_less_often():
    async def run():
        db = SigDB()
        db.sig = (1, "people-t1", "faces-t1")
        cache = FaceCache(db, per_person=10, refresh_seconds=60, knn=5, faces_refresh_seconds=900)
        assert await cache.refresh() and db.loads == 1
        db.sig = (2, "people-t1", "faces-t2")  # only faces changed: wait
        assert not await cache.refresh() and db.loads == 1
        db.sig = (3, "people-t2", "faces-t2")  # a person named/renamed: now
        assert await cache.refresh() and db.loads == 2
        db.sig = (4, "people-t2", "faces-t3")
        cache.snap.loaded_at -= 901  # older than faces_refresh_seconds: rebuild
        assert await cache.refresh() and db.loads == 3

    asyncio.run(run())
