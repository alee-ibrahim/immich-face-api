"""Main-subject labels in the store ("Label subjects" in the admin UI, ``python -m immich_face_api calibrate``).

A *sample* is one photo shown to a curator: its face boxes, the signals the API measured for every face and the
model's prediction. When the curator saves, the sample gets one label per face (``main``, ``co_subject`` or
``background``). Calibration needs only the stored signals and labels, never the photo again. See
migrations/004_subject_labels.sql for what is stored (no images, no embeddings).
"""

from __future__ import annotations

import json

from psycopg.types.json import Jsonb

from .subject import ROLES

PENDING_HOURS = 24  # unlabelled samples older than this are deleted (the curator moved on)


async def create_sample(
    store,
    *,
    source: str,
    asset_id: str | None,
    created_by: str,
    image_w: int,
    image_h: int,
    model_version: str,
    photo_type: str,
    faces: list[dict],
) -> int:
    async with store.pool.connection() as conn:
        await conn.execute(
            "DELETE FROM subject_sample WHERE labeled_at IS NULL AND NOT skipped "
            "AND created_at < now() - make_interval(hours => %s)",
            (PENDING_HOURS,),
        )
        row = await (
            await conn.execute(
                "INSERT INTO subject_sample (source, asset_id, created_by, image_w, image_h, model_version, "
                "photo_type, faces) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (source, asset_id, created_by, image_w, image_h, model_version, photo_type, Jsonb(faces)),
            )
        ).fetchone()
    return int(row["id"])


async def get_sample(store, sample_id: int) -> dict | None:
    async with store.pool.connection() as conn:
        return await (
            await conn.execute(
                "SELECT id, source, asset_id::text, created_by, created_at, image_w, image_h, model_version, "
                "photo_type, faces, labels, skipped, labeled_by, labeled_at FROM subject_sample WHERE id = %s",
                (sample_id,),
            )
        ).fetchone()


async def save_labels(store, sample_id: int, labels: list[str] | None, by: str, skipped: bool = False) -> bool:
    if labels is not None and any(x not in ROLES for x in labels):
        raise ValueError("unknown label")
    async with store.pool.connection() as conn:
        cur = await conn.execute(
            "UPDATE subject_sample SET labels = %s, skipped = %s, labeled_by = %s, labeled_at = now() "
            "WHERE id = %s AND (%s::int IS NULL OR jsonb_array_length(faces) = %s)",
            (
                Jsonb(labels) if labels is not None else None,
                skipped,
                by,
                sample_id,
                *([len(labels)] * 2 if labels is not None else [None, None]),
            ),
        )
        return cur.rowcount == 1


async def labelled_assets(store, asset_ids: list[str]) -> set[str]:
    """Asset ids that already have a sample (labelled, skipped or being labelled)."""
    if not asset_ids:
        return set()
    async with store.pool.connection() as conn:
        rows = await (
            await conn.execute(
                "SELECT DISTINCT asset_id::text AS a FROM subject_sample WHERE asset_id = ANY(%s::uuid[])",
                (asset_ids,),
            )
        ).fetchall()
    return {r["a"] for r in rows}


async def stats(store) -> dict:
    async with store.pool.connection() as conn:
        totals = await (
            await conn.execute(
                "SELECT count(*) FILTER (WHERE labels IS NOT NULL AND NOT skipped) AS photos, "
                "count(*) FILTER (WHERE skipped) AS skipped, "
                "count(*) FILTER (WHERE labeled_at IS NULL AND NOT skipped) AS pending "
                "FROM subject_sample"
            )
        ).fetchone()
        faces = await (
            await conn.execute(
                "SELECT l AS label, count(*) AS n FROM subject_sample, jsonb_array_elements_text(labels) l "
                "WHERE labels IS NOT NULL AND NOT skipped GROUP BY l"
            )
        ).fetchall()
        people = await (
            await conn.execute(
                "SELECT labeled_by AS who, count(*) AS photos, max(labeled_at) AS last FROM subject_sample "
                "WHERE labeled_at IS NOT NULL GROUP BY labeled_by ORDER BY count(*) DESC LIMIT 20"
            )
        ).fetchall()
        agree = await (
            await conn.execute(
                "SELECT count(*) FILTER (WHERE f->>'role' = l.value) AS same, count(*) AS total "
                "FROM subject_sample s, jsonb_array_elements(s.faces) WITH ORDINALITY f(f, i), "
                "jsonb_array_elements_text(s.labels) WITH ORDINALITY l(value, j) "
                "WHERE s.labels IS NOT NULL AND NOT s.skipped AND i = j"
            )
        ).fetchone()
    by_label = {r["label"]: r["n"] for r in faces}
    return {
        "photos": totals["photos"],
        "skipped": totals["skipped"],
        "pending": totals["pending"],
        "faces": {k: by_label.get(k, 0) for k in ROLES},
        "people": people,
        "agreement": (agree["same"] / agree["total"]) if agree and agree["total"] else None,
    }


async def recent(store, by: str | None = None, limit: int = 20) -> list[dict]:
    async with store.pool.connection() as conn:
        return await (
            await conn.execute(
                "SELECT id, source, labeled_by, labeled_at, photo_type, skipped, jsonb_array_length(faces) AS faces, "
                "(SELECT count(*) FROM jsonb_array_elements_text(labels) x WHERE x = 'main') AS mains "
                "FROM subject_sample WHERE labeled_at IS NOT NULL AND (%s::text IS NULL OR labeled_by = %s) "
                "ORDER BY labeled_at DESC LIMIT %s",
                (by, by, limit),
            )
        ).fetchall()


async def training_photos(store) -> list[dict]:
    """Every labelled, not skipped photo as {"signals": [...], "labels": [...], "id": n} (for calibrate)."""
    async with store.pool.connection() as conn:
        rows = await (
            await conn.execute(
                "SELECT id, faces, labels FROM subject_sample WHERE labels IS NOT NULL AND NOT skipped ORDER BY id"
            )
        ).fetchall()
    out = []
    for r in rows:
        faces = r["faces"] if not isinstance(r["faces"], str) else json.loads(r["faces"])
        labels = r["labels"] if not isinstance(r["labels"], str) else json.loads(r["labels"])
        if len(faces) == len(labels) and faces:
            out.append({"id": r["id"], "signals": [f["signals"] for f in faces], "labels": labels})
    return out


async def delete_all(store) -> int:
    async with store.pool.connection() as conn:
        cur = await conn.execute("DELETE FROM subject_sample")
        return cur.rowcount
