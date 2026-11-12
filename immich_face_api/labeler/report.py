"""Review report: one self-contained HTML file (images inlined, no external requests) and a CSV for bulk review."""

from __future__ import annotations

import base64
import csv
import html
import io
import json

from PIL import Image

from .match import MergeCandidate, Proposal

CSV_COLUMNS = [
    "run_id",
    "band",
    "score",
    "person_id",
    "faces",
    "proposed_name",
    "alt_names",
    "distance",
    "margin",
    "consistency",
    "runner_up",
    "evidence",
    "why_not_high",
    "immich_url",
    "decision",
    "corrected_name",
]


def small_image(data: bytes | None, edge: int = 96) -> str | None:
    """Any image -> small JPEG data URI (no metadata)."""
    if not data:
        return None
    try:
        img = Image.open(io.BytesIO(data))
        img.thumbnail((edge, edge))
        clean = Image.new("RGB", img.size)
        clean.paste(img.convert("RGB"))
        out = io.BytesIO()
        clean.save(out, format="JPEG", quality=80)
    except Exception:  # noqa: BLE001 - a broken thumbnail must not break the report
        return None
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()


def person_url(base: str, person_id: str) -> str:
    return f"{base}/people/{person_id}" if base else ""


def proposals_csv(run_id: str, props: list[Proposal], web_base: str) -> str:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(CSV_COLUMNS)
    for p in props:
        w.writerow(
            [
                run_id,
                p.band,
                p.score,
                p.person_id,
                p.faces,
                p.identity.proposed_name,
                json.dumps(p.identity.alt_names, ensure_ascii=False) if p.identity.alt_names else "",
                p.distance,
                p.margin,
                "" if p.consistency is None else p.consistency,
                p.runner_up or "",
                " | ".join(p.evidence()),
                " | ".join(p.why_not_high),
                person_url(web_base, p.person_id),
                "",
                "",
            ]
        )
    return out.getvalue()


_CSS = """
body{font:14px system-ui,sans-serif;margin:24px;color:#1b1b1b}
h1{font-size:20px}h2{font-size:16px;margin-top:28px}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:6px;vertical-align:top;text-align:left}
th{background:#f4f4f4;position:sticky;top:0}
.b-high{background:#e3f6e5}.b-medium{background:#fff6d6}.b-low{background:#fbe9e9}
.tag{display:inline-block;padding:1px 6px;border-radius:8px;font-size:12px;border:1px solid #bbb}
img{border-radius:6px}.ev{font-size:12px;color:#444}.dv{font-size:15px}
.stats td{border:0;padding:2px 12px 2px 0}
"""


def proposals_html(
    run_id: str,
    stats: dict,
    props: list[Proposal],
    merges: list[MergeCandidate],
    thumbs: dict[str, bytes | None],
    portraits: dict[str, bytes],
    web_base: str,
    notes: list[str],
) -> str:
    e = html.escape
    rows = []
    for p in props:
        t = small_image(thumbs.get(p.person_id))
        portrait = next((portraits[r.image_key] for r in p.identity.refs if r.image_key in portraits), None)
        pt = small_image(portrait)
        url = person_url(web_base, p.person_id)
        alt = " / ".join(
            f"<span class='dv' dir='rtl' lang='{e(k)}'>{e(v)}</span>" for k, v in p.identity.alt_names.items()
        )
        ev = "<br>".join(e(x) for x in p.evidence())
        why = ("<br><b>Not high:</b> " + e("; ".join(p.why_not_high))) if p.why_not_high else ""
        rows.append(
            f"<tr class='b-{p.band}'><td>{f'<img src={t!r} width=80>' if t else '-'}</td>"
            f"<td>{f'<img src={pt!r} width=80>' if pt else ''}</td>"
            f"<td><b>{e(p.identity.proposed_name)}</b><br>{alt}</td>"
            f"<td><span class='tag'>{e(p.band)}</span><br>score {p.score}</td>"
            f"<td>{p.faces}</td>"
            f"<td class='ev'>{ev}{why}</td>"
            f"<td class='ev'>{f'<a href={url!r}>open</a><br>' if url else ''}{e(p.person_id)}</td></tr>"
        )
    mrows = []
    for m in merges:
        ta, tb = small_image(thumbs.get(m.a)), small_image(thumbs.get(m.b))
        cell = lambda pid, t, n, f: (  # noqa: E731
            f"<td>{f'<img src={t!r} width=64>' if t else ''}<br>{e(n or '(no name)')}<br>"
            f"<span class='ev'>{f} faces · {e(pid)}</span></td>"
        )
        mrows.append(
            "<tr>"
            + cell(m.a, ta, m.names[0], m.faces[0])
            + cell(m.b, tb, m.names[1], m.faces[1])
            + f"<td>{m.distance}</td><td class='ev'>{e(m.reason)}</td></tr>"
        )
    stat_rows = "".join(f"<tr><td>{e(str(k))}</td><td><b>{e(str(v))}</b></td></tr>" for k, v in stats.items())
    note_html = "".join(f"<li>{e(n)}</li>" for n in notes)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>People labeler {e(run_id)}</title>
<meta name="robots" content="noindex"><style>{_CSS}</style></head><body>
<h1>People labeler: proposals (run {e(run_id)})</h1>
<p>Dry-run review report. Nothing was written to Immich by this run unless the stats say so. Internal: contains
photos of people; do not forward outside the organisation.</p>
<table class="stats">{stat_rows}</table>
<ul>{note_html}</ul>
<h2>Proposals ({len(props)})</h2>
<table><tr><th>Cluster</th><th>Portrait</th><th>Proposed name</th><th>Band</th><th>Faces</th><th>Evidence</th><th>Person</th></tr>
{"".join(rows)}</table>
<h2>Merge candidates ({len(merges)}) - proposals only, never merged automatically</h2>
<table><tr><th>Person A</th><th>Person B</th><th>Distance</th><th>Why</th></tr>{"".join(mrows)}</table>
</body></html>"""
