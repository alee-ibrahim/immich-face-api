"""Command line: ``python -m immich_face_api.labeler <command>``.

  run      build references, match the unnamed people, store proposals + report (dry run unless --apply-band high)
  apply    write the names of a stored run's proposals (chosen bands, minus --exclude / only --only)
  undo     revert names written by the labeler (a run, or single people), only where nobody changed them since
  export   write a stored run's report.html and proposals.csv to a directory
  status   recent runs

Configuration comes from LABELER_* environment variables and the JSON file in LABELER_CONFIG_FILE (see README,
"People labeler"). Secrets are read from files or the environment, never from the command line.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .folders import folder_references, overlap
from .immich import ImmichAPI, ImmichDB, ImmichError
from .match import build_identities, merge_candidates, propose
from .report import proposals_csv, proposals_html
from .settings import ConfigError, LabelerSettings
from .store import Store, new_run_id
from .web import Embedder, Fetcher, web_references

log = logging.getLogger("labeler")
BANDS = ("high", "medium", "low")


class Context:
    def __init__(self, s: LabelerSettings, need_db: bool = True):
        self.s = s
        self.store = Store(s)
        self.api = ImmichAPI(s)
        self.db = ImmichDB(s) if need_db else None

    def close(self):
        for x in (self.db, self.api, self.store):
            if x is not None:
                x.close()

    def owner(self) -> str:
        if self.s.expected_immich_version:
            v = self.api.version()
            if v != self.s.expected_immich_version:
                raise ImmichError(f"Immich {v} != pinned {self.s.expected_immich_version}")
        me = self.api.me()
        if self.s.owner_email and me["email"].lower() != self.s.owner_email.lower():
            raise ImmichError("the API key does not belong to the configured owner")
        return me["id"]


def _write_names(ctx: Context, run_id: str, source_run: str | None, owner: str, props: list[dict]) -> Counter:
    """The only place that writes to Immich. Every candidate is re-checked right before its write."""
    c: Counter = Counter()
    for p in props:
        if c["applied"] >= ctx.s.max_apply:
            c["over_cap"] += 1
            continue
        pid = p["person_id"]
        if ctx.store.ever_applied(pid):
            ctx.store.audit_skip(run_id, source_run, p, owner, "", "labeler named this person before")
            c["skipped_before"] += 1
            continue
        named = [n for _, n in ctx.db.group_names(pid) if n]
        if named:
            ctx.store.audit_skip(run_id, source_run, p, owner, named[0], "already named (by a user)")
            c["skipped_named"] += 1
            continue
        cur = ctx.api.person(pid)
        if cur.get("name") or cur.get("isHidden"):
            ctx.store.audit_skip(run_id, source_run, p, owner, cur.get("name") or "", "named or hidden in Immich")
            c["skipped_named"] += 1
            continue
        aid = ctx.store.audit_pending(run_id, source_run, p, owner, "")
        try:
            res = ctx.api.set_name(pid, p["name"])
        except ImmichError as exc:
            ctx.store.audit_set(aid, "failed", str(exc))
            c["failed"] += 1
            continue
        if res.get("name") != p["name"]:
            ctx.store.audit_set(aid, "failed", "Immich answered with a different name")
            c["failed"] += 1
            continue
        ctx.store.audit_set(aid, "applied")
        c["applied"] += 1
    return c


def cmd_run(s: LabelerSettings, args) -> int:
    ctx = Context(s)
    run_id = new_run_id()
    ctx.store.start_run(run_id, "run")
    t0 = time.monotonic()
    stats: dict = {"run_id": run_id, "mode": "apply high" if args.apply_band == "high" else "dry run"}
    notes: list[str] = []
    try:
        owner = ctx.owner()
        stats["immich_version"] = ctx.db.check(need_paths=bool(s.sources.folders))
        t = s.sources.thresholds

        roots = sorted({r.root for r in s.sources.folders})
        faces_by_root = {root: ctx.db.folder_faces(owner, root) for root in roots}
        fres = folder_references(s.sources, faces_by_root)
        stats["folder_faces"] = sum(len(v) for v in faces_by_root.values())
        stats["name_folders_with_faces"] = fres.folders
        stats["folder_references"] = len(fres.refs)
        stats["folder_references_ambiguous"] = sum(r.ambiguous for r in fres.refs)
        ref_faces = {f for r in fres.refs for f in r.face_ids}
        if ref_faces:
            unclustered = len(ref_faces - set(fres.face_person))
            stats["reference_faces_not_in_a_person_yet"] = unclustered
            if unclustered / len(ref_faces) > 0.1:
                notes.append(
                    f"Clustering is partial: {unclustered} of {len(ref_faces)} faces of the reference folders are not in "
                    "any person yet (Immich's facial recognition is still running). Those people get proposals in a "
                    "later run, once their faces are clustered."
                )

        portraits: dict[str, bytes] = {}
        wrefs = []
        if s.sources.web:
            fetcher = Fetcher(
                ctx.store,
                s.sources.user_agent,
                s.sources.request_delay_seconds,
                s.sources.page_refresh_hours,
                s.http_timeout,
            )
            emb = Embedder(s, ctx.store)
            try:
                w = web_references(s, ctx.store, fetcher, emb)
                wrefs, portraits = w.refs, w.images
                stats["web_cards"] = w.cards
                stats["web_references"] = len(w.refs)
                stats["web_cards_without_face"] = len(w.no_face)
                stats["web_requests"] = fetcher.requests
                stats["ml_calls"] = emb.calls
            except Exception as exc:  # noqa: BLE001 - the web is optional evidence
                notes.append(f"web references skipped: {type(exc).__name__}: {exc}")
                log.warning("web references skipped: %s", type(exc).__name__)
            finally:
                fetcher.close()
                emb.close()

        refs = fres.refs + wrefs
        identities = build_identities(refs, t, s.sources.names.homonym_detail)
        stats["identities"] = len(identities)
        stats["identities_with_name_conflict"] = sum(bool(i.conflicts) for i in identities)
        stats["identities_homonym"] = sum(i.homonym for i in identities)

        clusters = ctx.db.clusters(owner)
        unnamed = [c for c in clusters if not c.name and not c.hidden]
        stats["clusters"] = len(clusters)
        stats["clusters_unnamed"] = len(unnamed)
        stats["clusters_named"] = sum(bool(c.name) for c in clusters)

        cand: list[str] = []
        if unnamed and identities:
            dmin = (1.0 - np.stack([c.centroid for c in unnamed]) @ np.stack([i.centroid for i in identities]).T).min(1)
            cand = [c.person_id for c, d in zip(unnamed, dmin, strict=True) if d <= t.match_max_distance]
        samples = ctx.db.samples(owner, cand, t.cluster_sample_faces) if cand else {}
        props = propose(clusters, identities, t, samples, overlap(fres.refs, fres.face_person))
        merges = merge_candidates(clusters, props, t)
        bands = Counter(p.band for p in props)
        for b in BANDS:
            stats[f"proposals_{b}"] = bands.get(b, 0)
        stats["merge_candidates"] = len(merges)

        ids = {p.person_id for p in props} | {m.a for m in merges} | {m.b for m in merges}
        thumbs = {pid: ctx.api.thumbnail(pid) for pid in sorted(ids)[:5000]}

        ctx.store.save_proposals(
            run_id,
            [
                {
                    "person_id": p.person_id,
                    "band": p.band,
                    "score": p.score,
                    "name": p.identity.proposed_name,
                    "alt_names": p.identity.alt_names,
                    "identity_key": p.identity.key,
                    "distance": p.distance,
                    "margin": p.margin,
                    "faces": p.faces,
                    "evidence": p.evidence(),
                }
                for p in props
            ],
        )
        if args.apply_band == "high":
            todo = ctx.store.proposals(run_id, ["high"])
            res = _write_names(ctx, run_id, run_id, owner, todo)
            stats.update({f"apply_{k}": v for k, v in res.items()})
        stats["seconds"] = round(time.monotonic() - t0, 1)
        html = proposals_html(run_id, stats, props, merges, thumbs, portraits, s.sources.immich_web_url, notes)
        csv = proposals_csv(run_id, props, s.sources.immich_web_url)
        ctx.store.finish_run(run_id, "ok", stats, html, csv)
        if s.output_dir:
            out = Path(s.output_dir) / run_id
            out.mkdir(parents=True, exist_ok=True)
            (out / "report.html").write_text(html, encoding="utf-8")
            (out / "proposals.csv").write_text(csv, encoding="utf-8")
        log.info("run %s: %s", run_id, {k: v for k, v in stats.items() if k != "run_id"})
        return 0
    except Exception as exc:
        ctx.store.finish_run(run_id, "failed", {**stats, "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        ctx.close()


def cmd_apply(s: LabelerSettings, args) -> int:
    ctx = Context(s)
    try:
        source = ctx.store.resolve_run(args.run)
        run_id = new_run_id()
        ctx.store.start_run(run_id, "apply")
        owner = ctx.owner()
        ctx.db.check(need_paths=False)
        props = ctx.store.proposals(source, args.band)
        if args.only:
            props = [p for p in props if p["person_id"] in set(args.only)]
        if args.exclude:
            props = [p for p in props if p["person_id"] not in set(args.exclude)]
        res = _write_names(ctx, run_id, source, owner, props)
        stats = {"source_run": source, "bands": args.band, "candidates": len(props), **res}
        ctx.store.finish_run(run_id, "ok", stats)
        log.info("apply %s from %s: %s", run_id, source, dict(res))
        return 0
    finally:
        ctx.close()


def cmd_undo(s: LabelerSettings, args) -> int:
    if not args.run and not args.person:
        raise ConfigError("undo needs --run or --person")
    ctx = Context(s, need_db=False)
    try:
        run_id = new_run_id()
        ctx.store.start_run(run_id, "undo")
        ctx.owner()
        rows = ctx.store.applied(args.run, args.person)
        c: Counter = Counter()
        for row in rows:
            cur = ctx.api.person(row["person_id"])
            if cur.get("name") != row["new_name"]:
                c["changed_since"] += 1  # someone renamed it: leave it alone
                continue
            ctx.api.set_name(row["person_id"], row["old_name"])
            ctx.store.audit_undone(row["id"], run_id)
            c["undone"] += 1
        ctx.store.finish_run(run_id, "ok", {"run": args.run, "people": len(args.person or []), **c})
        log.info("undo %s: %s", run_id, dict(c))
        return 0
    finally:
        ctx.close()


def cmd_export(s: LabelerSettings, args) -> int:
    store = Store(s)
    try:
        run = store.resolve_run(args.run)
        html, csv = store.report(run)
        if html is None:
            raise LookupError(f"run {run} has no stored report")
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"labeler-{run}.html").write_text(html, encoding="utf-8")
        if csv:
            (out / f"labeler-{run}.csv").write_text(csv, encoding="utf-8")
        log.info("exported %s to %s", run, out)
        return 0
    finally:
        store.close()


def cmd_status(s: LabelerSettings, args) -> int:
    store = Store(s)
    try:
        for run_id, command, status, started, _finished, stats in store.recent_runs(args.limit):
            print(f"{run_id}  {command:6s} {status:8s} {started:%Y-%m-%d %H:%M}  {stats}")
        return 0
    finally:
        store.close()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m immich_face_api.labeler", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="propose names (dry run by default)")
    r.add_argument("--apply-band", choices=["none", "high"], default="none", help="also write the high band")
    a = sub.add_parser("apply", help="write the names of a stored run")
    a.add_argument("--run", required=True, help="run id, or 'latest'")
    a.add_argument("--band", action="append", choices=BANDS, required=True)
    a.add_argument("--exclude", nargs="*", default=[], help="person ids to leave out")
    a.add_argument("--only", nargs="*", default=[], help="only these person ids")
    u = sub.add_parser("undo", help="revert names the labeler wrote")
    u.add_argument("--run", help="an apply/run id whose writes to revert")
    u.add_argument("--person", nargs="*", help="person ids to revert")
    e = sub.add_parser("export", help="write a stored report to a directory")
    e.add_argument("--run", default="latest")
    e.add_argument("--out", required=True)
    st = sub.add_parser("status", help="recent runs")
    st.add_argument("--limit", type=int, default=10)
    return ap


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request (person ids) is noise
    args = build_parser().parse_args(argv)
    try:
        s = LabelerSettings.from_env()
        return {"run": cmd_run, "apply": cmd_apply, "undo": cmd_undo, "export": cmd_export, "status": cmd_status}[
            args.command
        ](s, args)
    except (ConfigError, ImmichError, LookupError) as exc:
        log.error("%s", exc)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
