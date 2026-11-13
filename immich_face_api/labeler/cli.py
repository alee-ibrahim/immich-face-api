"""Command line: ``python -m immich_face_api.labeler <command>``.

  run          build references, match the unnamed people, store proposals + report (dry run unless --apply-band)
  apply        write the names of a stored run's proposals (chosen bands, minus --exclude / only --only)
  share-names  copy the owner's names into the still-empty copies of users the owner shares people with
  undo         revert names written by the labeler (a run, or single people), only where nobody changed them since
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
from .match import build_identities, held, merge_candidates, named_vote, propose
from .report import proposals_csv, proposals_html, worklist_csv, worklist_html
from .settings import ConfigError, LabelerSettings
from .share_names import NameWriter, share_names, undo_copies
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
        if held(p.get("evidence") or []):
            ctx.store.audit_skip(
                run_id, source_run, p, owner, "", "held: " + "; ".join(e for e in p["evidence"] if e.startswith("HOLD"))
            )
            c["skipped_held"] += 1
            continue
        if ctx.store.ever_applied(pid):
            ctx.store.audit_skip(run_id, source_run, p, owner, "", "labeler named this person before")
            c["skipped_before"] += 1
            continue
        named = [n for _, n in ctx.db.group_names(pid) if n]
        if named:
            ctx.store.audit_skip(run_id, source_run, p, owner, named[0], "already named (by a user)")
            c["skipped_named"] += 1
            continue
        try:
            cur = ctx.api.person(pid)
        except ImmichError as exc:  # merged or deleted since the run
            ctx.store.audit_skip(run_id, source_run, p, owner, "", f"person not readable now: {exc}")
            c["skipped_gone"] += 1
            continue
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


def add_named_votes(ctx, owner, clusters, props, t, write_sources):
    """Named-neighbour vote for the unnamed clusters; merged with the reference proposals: same name = more
    evidence, different name = both held, no reference proposal = a new proposal (source "named")."""
    named = [c for c in clusters if c.name and not c.hidden and not c.name.startswith("(")]
    unnamed = [c for c in clusters if not c.name and not c.hidden and c.faces >= t.min_cluster_faces]
    names = {c.person_id: c.name for c in named}
    nsamp = ctx.db.samples(owner, [c.person_id for c in named], t.vote_named_sample) if named else {}
    usamp = ctx.db.samples(owner, [c.person_id for c in unnamed], t.vote_unnamed_sample) if unnamed else {}
    votes = named_vote(usamp, nsamp, names, {c.person_id: c.faces for c in clusters}, t)
    by = {p.person_id: p for p in props}
    st = Counter(votes=len(votes))
    allowed = None if write_sources is None else set(write_sources)
    for v in votes:
        p = by.get(v.person_id)
        if p is None:
            if allowed is not None and "named" not in allowed:
                v.hold.append("based on sources not yet approved for writing (named)")
                st["vote_held_unapproved"] += 1
            props.append(v)
            st["vote_new"] += 1
        elif p.identity.proposed_name == v.identity.name:
            p.why_not_high = [w for w in p.why_not_high]  # unchanged band; the vote is extra evidence
            p.vote_note = v.evidence()[0]
            st["vote_agrees"] += 1
        else:
            p.hold.append(f"named-neighbour vote says {v.identity.name!r}")
            st["vote_disagrees"] += 1
    st.setdefault("vote_held_unapproved", 0)
    rank = {"high": 0, "medium": 1, "low": 2}
    props.sort(key=lambda p: (rank[p.band], -p.score, p.person_id))
    return props, {f"named_{k}" if not k.startswith("vote") else k: n for k, n in st.items()}


def hold_unapproved_sources(props, refs, write_sources, clusters, t, samples, ov, homonym_detail) -> int:
    """With ``write_sources`` set, a proposal may be written only if the approved sources ALONE lead to the same
    name for that person; anything else is held ("needs owner OK"). Returns how many were held for this reason."""
    if write_sources is None:
        return 0
    allowed = set(write_sources)
    approved = build_identities([r for r in refs if r.source in allowed], t, homonym_detail)
    base = {p.person_id: p.identity.proposed_name for p in propose(clusters, approved, t, samples, ov)}
    n = 0
    for p in props:
        if base.get(p.person_id) == p.identity.proposed_name:
            continue
        new = sorted({r.source for r in p.identity.refs if r.source not in allowed})
        p.hold.append("based on sources not yet approved for writing (" + ", ".join(new or ["combination"]) + ")")
        n += 1
    return n


def apply_bands_of(args) -> list[str]:
    """Bands `run` writes: only those given with --apply-band (none = dry run)."""
    return [b for b in (args.apply_band or []) if b in BANDS]


def cmd_run(s: LabelerSettings, args) -> int:
    ctx = Context(s)
    run_id = new_run_id()
    ctx.store.start_run(run_id, "run")
    t0 = time.monotonic()
    apply_bands = apply_bands_of(args)
    stats: dict = {"run_id": run_id, "mode": ("apply " + "+".join(apply_bands)) if apply_bands else "dry run"}
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

        clusters = ctx.db.clusters(owner)
        unnamed = [c for c in clusters if not c.name and not c.hidden]
        stats["clusters"] = len(clusters)
        stats["clusters_unnamed"] = len(unnamed)
        stats["clusters_named"] = sum(bool(c.name) for c in clusters)
        refs = fres.refs + wrefs
        identities = build_identities(refs, t, s.sources.names.homonym_detail)
        stats["identities"] = len(identities)
        stats["identities_with_name_conflict"] = sum(bool(i.conflicts) for i in identities)
        stats["identities_homonym"] = sum(i.homonym for i in identities)

        cand: list[str] = []
        if unnamed and identities:
            dmin = (1.0 - np.stack([c.centroid for c in unnamed]) @ np.stack([i.centroid for i in identities]).T).min(1)
            cand = [c.person_id for c, d in zip(unnamed, dmin, strict=True) if d <= t.match_max_distance]
        samples = ctx.db.samples(owner, cand, t.cluster_sample_faces) if cand else {}
        ov = overlap(fres.refs, fres.face_person)
        props = propose(clusters, identities, t, samples, ov)
        held_new = hold_unapproved_sources(
            props, refs, s.sources.write_sources, clusters, t, samples, ov, s.sources.names.homonym_detail
        )
        if s.sources.named_people:
            props, vstats = add_named_votes(ctx, owner, clusters, props, t, s.sources.write_sources)
            stats.update(vstats)
            held_new += vstats["vote_held_unapproved"]
        stats["proposals_needing_source_approval"] = held_new
        merges = merge_candidates(clusters, props, t)
        band_counts = Counter(p.band for p in props)
        for b in BANDS:
            stats[f"proposals_{b}"] = band_counts.get(b, 0)
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
        if apply_bands:
            todo = ctx.store.proposals(run_id, apply_bands)
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


def cmd_worklist(s: LabelerSettings, args) -> int:
    """The biggest unnamed people that the latest run will NOT name (no unheld high/medium proposal), for curators."""
    ctx = Context(s)
    run_id = new_run_id()
    ctx.store.start_run(run_id, "worklist")
    try:
        owner = ctx.owner()
        ctx.db.check(need_paths=False)
        source = ctx.store.resolve_run("latest")
        props = {p["person_id"]: p for p in ctx.store.proposals(source, list(BANDS))}
        clusters = [c for c in ctx.db.clusters(owner) if not c.name and not c.hidden]
        unnamed_faces = sum(c.faces for c in clusters)

        def todo(c) -> bool:
            p = props.get(c.person_id)
            return not p or p["band"] == "low" or held(p["evidence"] or [])

        rows = sorted((c for c in clusters if todo(c)), key=lambda c: (-c.faces, c.person_id))[: args.top]
        thumbs = {c.person_id: ctx.api.thumbnail(c.person_id) for c in rows}
        covered = sum(c.faces for c in rows)
        stats = {
            "source_run": source,
            "unnamed_clusters": len(clusters),
            "unnamed_faces": unnamed_faces,
            "worklist_clusters": len(rows),
            "worklist_faces": covered,
            "worklist_share_of_unnamed_faces": round(covered / unnamed_faces, 3) if unnamed_faces else 0,
        }
        html = worklist_html(run_id, stats, rows, props, thumbs, s.sources.immich_web_url)
        csv = worklist_csv(rows, props, s.sources.immich_web_url)
        ctx.store.finish_run(run_id, "ok", stats, html, csv)
        log.info("worklist %s: %s", run_id, stats)
        return 0
    except Exception as exc:
        ctx.store.finish_run(run_id, "failed", {"error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        ctx.close()


def cmd_share_names(s: LabelerSettings, args) -> int:
    ctx = Context(s, need_db=False)
    writer = NameWriter(s)
    run_id = new_run_id()
    ctx.store.start_run(run_id, "share-names")
    try:
        owner = ctx.owner()
        writer.check()
        res = share_names(writer, ctx.store, run_id, owner, args.dry_run, s.max_copies)
        stats = {"mode": "dry run" if args.dry_run else "write", **res}
        ctx.store.finish_run(run_id, "ok", stats)
        log.info("share-names %s: %s", run_id, stats)
        return 0
    except Exception as exc:
        ctx.store.finish_run(run_id, "failed", {"error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        writer.close()
        ctx.close()


def cmd_undo(s: LabelerSettings, args) -> int:
    if not args.run and not args.person:
        raise ConfigError("undo needs --run or --person")
    if args.copies:
        return _undo_copies(s, args)
    ctx = Context(s, need_db=False)
    try:
        run_id = new_run_id()
        ctx.store.start_run(run_id, "undo")
        ctx.owner()
        rows = [r for run in (args.run or [None]) for r in ctx.store.applied(run, args.person)]
        c: Counter = Counter()
        for row in rows:
            try:
                cur = ctx.api.person(row["person_id"])
            except ImmichError:
                c["gone"] += 1
                continue
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


def _undo_copies(s: LabelerSettings, args) -> int:
    """Revert copies made by share-names: of the given share-names runs, or of the given people. To roll back an
    apply run completely, undo it first, then undo the copies of its people (``--copies --person ...``) or of the
    share-names runs that followed it."""
    store = Store(s)
    writer = NameWriter(s)
    run_id = new_run_id()
    store.start_run(run_id, "undo-copies")
    try:
        writer.check()
        rows = [r for run in (args.run or [None]) for r in store.applied(run, args.person, target="copy")]
        c = undo_copies(writer, store, run_id, rows)
        store.finish_run(run_id, "ok", {"run": args.run, "people": len(args.person or []), **c})
        log.info("undo copies %s: %s", run_id, dict(c))
        return 0
    finally:
        writer.close()
        store.close()


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
    r.add_argument("--apply-band", action="append", choices=["none", *BANDS], help="also write this band (repeatable)")
    a = sub.add_parser("apply", help="write the names of a stored run")
    a.add_argument("--run", required=True, help="run id, or 'latest'")
    a.add_argument("--band", action="append", choices=BANDS, required=True)
    a.add_argument("--exclude", nargs="*", default=[], help="person ids to leave out")
    a.add_argument("--only", nargs="*", default=[], help="only these person ids")
    u = sub.add_parser("undo", help="revert names the labeler wrote")
    u.add_argument("--run", nargs="*", help="apply/run (or share-names) run ids whose writes to revert")
    u.add_argument("--person", nargs="*", help="person ids to revert")
    u.add_argument("--copies", action="store_true", help="revert share-names copies instead of the owner's names")
    wl = sub.add_parser("worklist", help="biggest unnamed people the latest run will not name (for curators)")
    wl.add_argument("--top", type=int, default=200)
    sn = sub.add_parser("share-names", help="copy owner names into empty shared copies")
    sn.add_argument("--dry-run", action="store_true", help="count only, write nothing")
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
        commands = {
            "run": cmd_run,
            "apply": cmd_apply,
            "share-names": cmd_share_names,
            "worklist": cmd_worklist,
            "undo": cmd_undo,
            "export": cmd_export,
            "status": cmd_status,
        }
        return commands[args.command](s, args)
    except (ConfigError, ImmichError, LookupError) as exc:
        log.error("%s", exc)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
